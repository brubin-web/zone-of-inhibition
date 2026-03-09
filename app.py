"""
Zone of Inhibition Measurement Tool — Streamlit Web App
Upload agar plate images, detect spots, label them, and get measurements + graphs.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
import io
import math


# ─── Image Analysis Functions ───────────────────────────────────────────────


def load_and_convert(uploaded_file):
    """Load uploaded image file and return RGB array + grayscale."""
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def detect_plates(gray, rgb):
    """Detect circular petri dishes using threshold + erosion + contours."""
    blurred = cv2.GaussianBlur(gray, (11, 11), 5)

    for thresh_val in [215, 210, 205, 200, 220, 195]:
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        eroded = cv2.erode(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        plates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100000:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            circularity = area / (math.pi * r * r)
            if circularity > 0.7 and r > 80:
                plates.append((int(x), int(y), int(r)))

        if plates:
            break

    plates.sort(key=lambda p: (p[1] // 300, p[0]))
    return plates


def remove_pen_marks(gray, rgb, mask):
    """Create a pen-mark mask using color channels and inpaint to remove them."""
    pen_mask = np.zeros(gray.shape, dtype=np.uint8)
    r_ch, g_ch, b_ch = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)

    # Blue ink: B channel notably higher than R
    pen_mask[(b_ch - r_ch > 15) & (mask > 0)] = 255
    # Red ink: R channel notably higher than G and B
    pen_mask[(r_ch - g_ch > 40) & (mask > 0)] = 255

    # Dilate to cover edges of pen marks
    pen_mask = cv2.dilate(pen_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

    # Inpaint pen marks
    clean = cv2.inpaint(gray, pen_mask, 25, cv2.INPAINT_TELEA)
    return clean


def find_spots_in_plate(gray, rgb, pcx, pcy, pr, sensitivity=5):
    """
    Find spots within a plate using median filter + background subtraction.
    Detects both lighter regions (clear zones) and darker regions (dense drops).
    sensitivity: lower = more spots detected (threshold in gray levels).
    """
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (pcx, pcy), pr - 70, 255, -1)

    # Remove pen marks using color information
    clean = remove_pen_marks(gray, rgb, mask)

    # Strong median filter to suppress linear streaking while preserving circular features
    median = cv2.medianBlur(clean, 21)

    # Large Gaussian blur for background model
    bg = cv2.GaussianBlur(median, (201, 201), 60)
    diff = median.astype(np.float32) - bg.astype(np.float32)
    diff[mask == 0] = 0

    all_contours = []
    for sign in [1, -1]:  # lighter, then darker
        directed = sign * diff
        thresh = np.zeros(gray.shape, dtype=np.uint8)
        thresh[directed > sensitivity] = 255
        thresh[mask == 0] = 0

        # Morphological cleanup
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k5)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 300 or area > 80000:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            circ = area / (math.pi * radius ** 2) if radius > 0 else 0
            if circ < 0.35 or radius < 10:
                continue

            # Measure mean absolute difference in this region
            contour_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [c], -1, 255, -1)
            mean_diff = abs(diff[contour_mask > 0].mean())
            raw_diff = diff[int(y), int(x)]
            direction = "lighter" if raw_diff > 0 else "darker"

            if mean_diff > sensitivity * 0.6:
                all_contours.append((int(x), int(y), int(radius), area, circ,
                                     mean_diff, direction))

    # Merge nearby detections (lighter+darker overlapping the same spot)
    merged = []
    used = set()
    all_contours.sort(key=lambda s: -(s[4] * s[5]))  # circularity * strength
    for i, item in enumerate(all_contours):
        if i in used:
            continue
        group = [item]
        for j, other in enumerate(all_contours):
            if j != i and j not in used:
                if abs(item[0] - other[0]) < 60 and abs(item[1] - other[1]) < 60:
                    group.append(other)
                    used.add(j)
        used.add(i)
        # Keep the detection with best circularity
        best = max(group, key=lambda g: g[4])
        merged.append(best)

    return merged, diff


def measure_spot(gray_clean, diff, cx, cy, radius, pcx, pcy, pr):
    """
    Measure a spot: find the zone boundary using the background-subtracted
    difference image and radial profile analysis.
    """
    max_r = min(150, int(pr * 0.8))
    dist_to_edge = math.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
    max_r = min(max_r, int(pr - dist_to_edge - 20))
    max_r = max(max_r, 50)

    # Radial profile on the difference image (more meaningful than raw gray)
    prof_diff = np.zeros(max_r)
    prof_gray = np.zeros(max_r)
    for r in range(1, max_r + 1):
        na = max(24, int(2 * math.pi * r))
        angles = np.linspace(0, 2 * math.pi, na, endpoint=False)
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, diff.shape[1] - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, diff.shape[0] - 1)
        prof_diff[r - 1] = diff[ys, xs].mean()
        prof_gray[r - 1] = gray_clean[ys, xs].mean()

    # Smooth profiles
    kernel = np.ones(5) / 5
    prof_d = np.convolve(prof_diff, kernel, mode='same')
    prof_g = np.convolve(prof_gray, kernel, mode='same')

    # Drop center intensity
    center_intensity = float(gray_clean[cy, cx])

    # The detected radius from contour analysis is a good estimate of the spot/zone edge
    # Let's refine it: find where the absolute difference drops below a threshold
    abs_prof = np.abs(prof_d)

    # Zone radius: where the difference signal falls to ~30% of its peak
    peak_diff = abs_prof[:min(100, len(abs_prof))].max()
    threshold = peak_diff * 0.3

    zone_radius = radius  # default to contour radius
    for r in range(max(5, radius - 10), min(len(abs_prof), radius + 30)):
        if abs_prof[r] < threshold:
            zone_radius = r + 1
            break

    # Drop radius: look for a dark center within the spot
    # If the center is darker than surroundings, find the drop edge
    drop_radius = 3  # default small
    if center_intensity < prof_g[min(radius, len(prof_g) - 1)] - 10:
        # There's a dark drop at center
        half_val = (center_intensity + prof_g[min(radius, len(prof_g) - 1)]) / 2
        for r in range(len(prof_g)):
            if prof_g[r] > half_val:
                drop_radius = r + 1
                break

    # Compute areas
    drop_area = math.pi * drop_radius ** 2
    zone_area = math.pi * zone_radius ** 2
    zone_diameter = zone_radius * 2

    # Mean intensity within the drop
    drop_pixels = []
    for r in range(1, max(2, drop_radius + 1)):
        na = max(12, int(2 * math.pi * r))
        angles = np.linspace(0, 2 * math.pi, na, endpoint=False)
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, gray_clean.shape[1] - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, gray_clean.shape[0] - 1)
        drop_pixels.extend(gray_clean[ys, xs].tolist())
    drop_mean_intensity = float(np.mean(drop_pixels)) if drop_pixels else center_intensity

    has_zone = zone_radius > drop_radius + 5

    return {
        'drop_radius': drop_radius,
        'drop_area': drop_area,
        'drop_mean_intensity': drop_mean_intensity,
        'zone_radius': zone_radius,
        'zone_diameter': zone_diameter,
        'zone_area': zone_area,
        'has_zone': has_zone,
    }


def analyze_image(rgb, gray, sensitivity=3.5):
    """Full analysis pipeline. Returns plates, spot data, and diff images."""
    plates = detect_plates(gray, rgb)

    all_spots = []
    diff_images = {}

    for plate_idx, (pcx, pcy, pr) in enumerate(plates):
        # Remove pen marks for this plate
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (pcx, pcy), pr - 70, 255, -1)
        clean = remove_pen_marks(gray, rgb, mask)
        median = cv2.medianBlur(clean, 21)

        spots, diff = find_spots_in_plate(gray, rgb, pcx, pcy, pr, sensitivity)
        diff_images[plate_idx] = diff

        for cx, cy, radius, area, circ, mean_diff, direction in spots:
            m = measure_spot(median, diff, cx, cy, radius, pcx, pcy, pr)
            m.update({
                'plate': plate_idx + 1,
                'center_x': cx,
                'center_y': cy,
                'detect_radius': radius,
                'detect_circ': circ,
                'detect_diff': mean_diff,
                'detect_direction': direction,
            })
            all_spots.append(m)

    # Sort by plate then position
    all_spots.sort(key=lambda m: (m['plate'], m['center_y'] // 50, m['center_x']))
    for i, m in enumerate(all_spots):
        m['spot_id'] = i

    return plates, all_spots


def draw_annotations(rgb, plates, spots, enabled_ids=None):
    """Create annotated image showing detected plates and spots."""
    annotated = rgb.copy()

    for cx, cy, r in plates:
        cv2.circle(annotated, (cx, cy), r, (0, 200, 0), 3)

    for i, m in enumerate(spots):
        if enabled_ids is not None and i not in enabled_ids:
            continue

        cx, cy = m['center_x'], m['center_y']
        zr = m['zone_radius']
        dr = m['drop_radius']

        # Zone boundary (red)
        if m['has_zone']:
            cv2.circle(annotated, (cx, cy), zr, (255, 60, 60), 2)
        # Spot boundary from detection (green)
        cv2.circle(annotated, (cx, cy), m['detect_radius'], (100, 255, 100), 2)
        # Drop center (blue dot — small marker)
        cv2.circle(annotated, (cx, cy), 4, (50, 100, 255), -1)
        # Number label
        label = f"#{i + 1}"
        cv2.putText(annotated, label, (cx + max(zr, m['detect_radius']) + 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return annotated


# ─── Streamlit App ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Zone of Inhibition Analyzer", layout="wide")
st.title("Zone of Inhibition Analyzer")
st.markdown("Upload an agar plate image to detect and measure zones of inhibition.")

uploaded_file = st.file_uploader(
    "Drag and drop a plate image (TIF, PNG, JPG)",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # ─── Detection Settings ─────────────────────────────────────────────
    sensitivity = st.slider(
        "Detection sensitivity (lower = finds more spots, may include false positives)",
        min_value=1.5, max_value=8.0, value=5.0, step=0.5,
        key="sensitivity"
    )

    # Analyze (cached per file + sensitivity)
    cache_key = f"{uploaded_file.name}_{sensitivity}"
    if st.session_state.get("_cache_key") != cache_key:
        with st.spinner("Analyzing image..."):
            rgb, gray = load_and_convert(uploaded_file)
            plates, spots = analyze_image(rgb, gray, sensitivity)
            st.session_state._cache_key = cache_key
            st.session_state._rgb = rgb
            st.session_state._plates = plates
            st.session_state._spots = spots
            # Reset enabled flags when re-analyzing
            st.session_state._enabled = {i: True for i in range(len(spots))}
            # Reset labels
            st.session_state._labels = {}

    rgb = st.session_state._rgb
    plates = st.session_state._plates
    spots = st.session_state._spots

    st.success(f"Detected **{len(plates)} plate(s)** and **{len(spots)} spot(s)**. "
               "Use the sensitivity slider and checkboxes below to refine.")

    # ─── Spot Selection ─────────────────────────────────────────────────
    st.subheader("Select Spots")
    st.markdown("Uncheck any false positives. Only checked spots will be measured and graphed.")

    # Build enabled set from checkboxes
    enabled = st.session_state.get("_enabled", {i: True for i in range(len(spots))})

    cols_per_row = 4
    for row_start in range(0, len(spots), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, spot_idx in enumerate(range(row_start, min(row_start + cols_per_row, len(spots)))):
            m = spots[spot_idx]
            with cols[col_idx]:
                checked = st.checkbox(
                    f"#{spot_idx + 1} (P{m['plate']}, {m['detect_direction']})",
                    value=enabled.get(spot_idx, True),
                    key=f"enable_{spot_idx}"
                )
                enabled[spot_idx] = checked
    st.session_state._enabled = enabled

    enabled_ids = {i for i, v in enabled.items() if v}
    active_spots = [s for i, s in enumerate(spots) if i in enabled_ids]

    # ─── Annotated Image ────────────────────────────────────────────────
    st.subheader("Detected Spots")
    annotated = draw_annotations(rgb, plates, spots, enabled_ids)
    st.image(annotated, use_container_width=True,
             caption="Green = spot boundary, Red = zone boundary, Blue = drop center, Yellow = spot #")

    if not active_spots:
        st.warning("No spots selected. Check some spots above or lower the sensitivity.")
        st.stop()

    # ─── Label Spots ────────────────────────────────────────────────────
    st.subheader("Label Your Spots")
    st.markdown(
        "Enter a **sample type** (e.g. 'Ampicillin', 'Control') and "
        "**concentration** (e.g. '100') for each spot. "
        "Matching sample types will be grouped in the graphs."
    )

    labels = st.session_state.get("_labels", {})

    for row_start in range(0, len(active_spots), 3):
        cols = st.columns(3)
        for col_idx, idx in enumerate(range(row_start, min(row_start + 3, len(active_spots)))):
            m = active_spots[idx]
            sid = m['spot_id']
            with cols[col_idx]:
                st.markdown(f"**Spot #{spots.index(m) + 1}** — Plate {m['plate']}")
                sample = st.text_input(
                    "Sample", value=labels.get(sid, {}).get("sample", ""),
                    key=f"sample_{sid}", label_visibility="collapsed",
                    placeholder="Sample type"
                )
                conc = st.text_input(
                    "Conc", value=labels.get(sid, {}).get("conc", ""),
                    key=f"conc_{sid}", label_visibility="collapsed",
                    placeholder="Concentration"
                )
                labels[sid] = {"sample": sample, "conc": conc}
    st.session_state._labels = labels

    # ─── Measurements Table ─────────────────────────────────────────────
    st.subheader("Measurements")

    rows = []
    for m in active_spots:
        sid = m['spot_id']
        label = labels.get(sid, {})
        sample_type = label.get("sample", "").strip() or "Unlabeled"
        conc_str = label.get("conc", "").strip()
        try:
            conc_val = float(conc_str) if conc_str else None
        except ValueError:
            conc_val = None

        drop_circ = 2 * math.pi * m['drop_radius']
        zoi_per_area = m['zone_diameter'] / m['drop_area'] if m['drop_area'] > 0 else 0
        zoi_per_circ = m['zone_diameter'] / drop_circ if drop_circ > 0 else 0

        rows.append({
            "Spot #": spots.index(m) + 1,
            "Plate": m['plate'],
            "Sample Type": sample_type,
            "Concentration": conc_val,
            "Conc Label": conc_str if conc_str else "N/A",
            "Center X": m['center_x'],
            "Center Y": m['center_y'],
            "Spot Radius (px)": round(m['detect_radius'], 1),
            "Drop Radius (px)": round(m['drop_radius'], 1),
            "Drop Area (px\u00b2)": round(m['drop_area'], 1),
            "Drop Intensity": round(m['drop_mean_intensity'], 1),
            "Zone Diameter (px)": round(m['zone_diameter'], 1),
            "Zone Radius (px)": round(m['zone_radius'], 1),
            "Zone Area (px\u00b2)": round(m['zone_area'], 1),
            "Has Zone": m['has_zone'],
            "ZOI / Drop Area": round(zoi_per_area, 4),
            "ZOI / Drop Circumference": round(zoi_per_circ, 4),
        })

    df = pd.DataFrame(rows)

    display_cols = [
        "Spot #", "Plate", "Sample Type", "Conc Label",
        "Spot Radius (px)", "Drop Intensity",
        "Zone Diameter (px)", "Has Zone",
        "ZOI / Drop Area", "ZOI / Drop Circumference"
    ]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download CSV", csv_buffer.getvalue(),
        file_name=f"measurements_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
        mime="text/csv"
    )

    # ─── Summary Graphs ─────────────────────────────────────────────────
    st.subheader("Summary Graphs")

    has_labels = any(r["Sample Type"] != "Unlabeled" for r in rows)
    has_conc = any(r["Concentration"] is not None for r in rows)

    if not has_labels:
        st.info("Label your spots above to generate grouped summary graphs.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Zone Diameter by Sample",
            "ZOI vs Concentration",
            "Spot Intensity",
            "Normalized ZOI"
        ])

        with tab1:
            fig = px.box(
                df[df["Sample Type"] != "Unlabeled"],
                x="Sample Type", y="Zone Diameter (px)",
                color="Sample Type", points="all",
                title="Zone of Inhibition Diameter by Sample Type",
                hover_data=["Spot #", "Plate", "Conc Label"]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if has_conc:
                df_conc = df[df["Concentration"].notna() & (df["Sample Type"] != "Unlabeled")].copy()
                if len(df_conc) > 0:
                    fig = px.scatter(
                        df_conc, x="Concentration", y="Zone Diameter (px)",
                        color="Sample Type", symbol="Sample Type",
                        title="Zone Diameter vs Concentration",
                        hover_data=["Spot #", "Plate"],
                        trendline="ols" if len(df_conc) > 2 else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No labeled spots with concentration values yet.")
            else:
                st.info("Enter concentration values to see this graph.")

        with tab3:
            fig = px.box(
                df[df["Sample Type"] != "Unlabeled"],
                x="Sample Type", y="Drop Intensity",
                color="Sample Type", points="all",
                title="Spot Darkness by Sample Type (lower = darker)",
                hover_data=["Spot #", "Plate", "Conc Label"]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            df_labeled = df[df["Sample Type"] != "Unlabeled"]
            fig1 = px.box(
                df_labeled, x="Sample Type", y="ZOI / Drop Area",
                color="Sample Type", points="all",
                title="Zone Diameter Normalized by Drop Area",
                hover_data=["Spot #", "Plate", "Conc Label"]
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.box(
                df_labeled, x="Sample Type", y="ZOI / Drop Circumference",
                color="Sample Type", points="all",
                title="Zone Diameter Normalized by Drop Circumference",
                hover_data=["Spot #", "Plate", "Conc Label"]
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ─── Downloads ──────────────────────────────────────────────────────
    st.subheader("Download Annotated Image")
    img_buffer = io.BytesIO()
    Image.fromarray(annotated).save(img_buffer, format="PNG")
    st.download_button(
        "Download Annotated Image", img_buffer.getvalue(),
        file_name=f"annotated_{uploaded_file.name.rsplit('.', 1)[0]}.png",
        mime="image/png"
    )
