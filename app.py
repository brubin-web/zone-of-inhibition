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
import plotly.graph_objects as go
import io
import math


# ─── Image Analysis Functions ───────────────────────────────────────────────


def load_and_convert(uploaded_file):
    """Load uploaded image file and return RGB array + grayscale."""
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def detect_plates(gray):
    """Detect circular petri dishes using threshold + erosion + contours."""
    blurred = cv2.GaussianBlur(gray, (11, 11), 5)

    # Try multiple thresholds to handle different image brightnesses
    for thresh_val in [215, 210, 205, 200, 220]:
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
            if circularity > 0.7 and r > 100:
                plates.append((int(x), int(y), int(r)))

        if plates:
            break

    plates.sort(key=lambda p: (p[1] // 300, p[0]))
    return plates


def radial_profile(gray, cx, cy, max_r):
    """Compute average intensity at each radius from center."""
    profile = np.zeros(max_r)
    for r in range(1, max_r + 1):
        na = max(24, int(2 * math.pi * r))
        angles = np.linspace(0, 2 * math.pi, na, endpoint=False)
        xs = (cx + r * np.cos(angles)).astype(int)
        ys = (cy + r * np.sin(angles)).astype(int)
        xs = np.clip(xs, 0, gray.shape[1] - 1)
        ys = np.clip(ys, 0, gray.shape[0] - 1)
        profile[r - 1] = gray[ys, xs].mean()
    return profile


def smooth(arr, window=5):
    """Moving average smoothing."""
    return np.convolve(arr, np.ones(window) / window, mode='same')


def find_drops_in_plate(gray, pcx, pcy, pr):
    """Find drop/spot locations within a single plate."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (pcx, pcy), pr - 40, 255, -1)

    bg = cv2.GaussianBlur(gray, (151, 151), 50)
    diff = gray.astype(np.float32) - bg.astype(np.float32)

    # Dark blobs (drops are darker than surrounding lawn)
    dark_mask = np.zeros(gray.shape, dtype=np.uint8)
    dark_mask[(diff < -15) & (mask > 0)] = 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_clean = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k5)

    dark_contours, _ = cv2.findContours(dark_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in dark_contours:
        area = cv2.contourArea(c)
        if area < 80 or area > 5000:
            continue
        (x, y), radius = cv2.minEnclosingCircle(c)
        circ = area / (math.pi * radius * radius) if radius > 0 else 0
        if circ < 0.3:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(x), int(y)
        half = max(3, int(radius * 0.5))
        y1, y2 = max(0, cy - half), min(gray.shape[0], cy + half)
        x1, x2 = max(0, cx - half), min(gray.shape[1], cx + half)
        mean_int = gray[y1:y2, x1:x2].mean()
        if mean_int < 150:
            candidates.append((cx, cy, mean_int, area))

    # Also find bright circular zones with dark centers
    bright_mask = np.zeros(gray.shape, dtype=np.uint8)
    bright_mask[(gray > 188) & (mask > 0)] = 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright_filled = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, k_close)
    bright_filled = cv2.morphologyEx(bright_filled, cv2.MORPH_OPEN, k5)
    bright_contours, _ = cv2.findContours(bright_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in bright_contours:
        area = cv2.contourArea(c)
        if area < 1000:
            continue
        (x, y), radius = cv2.minEnclosingCircle(c)
        circ = area / (math.pi * radius * radius) if radius > 0 else 0
        if circ < 0.5 or radius < 15 or radius > 120:
            continue
        cx, cy = int(x), int(y)
        center_val = gray[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5].mean()
        if center_val < 170:
            already = any(abs(cx - c2[0]) < 25 and abs(cy - c2[1]) < 25 for c2 in candidates)
            if not already:
                candidates.append((cx, cy, center_val, -1))

    # Merge nearby candidates
    merged = []
    used = set()
    for i, (cx, cy, inten, area) in enumerate(candidates):
        if i in used:
            continue
        group = [(cx, cy, inten, area)]
        for j, (cx2, cy2, inten2, area2) in enumerate(candidates):
            if j <= i or j in used:
                continue
            if abs(cx - cx2) < 25 and abs(cy - cy2) < 25:
                group.append((cx2, cy2, inten2, area2))
                used.add(j)
        used.add(i)
        best = min(group, key=lambda g: g[2])
        merged.append(best)

    return merged


def measure_drop_and_zone(gray, cx, cy, pcx, pcy, pr):
    """Measure drop radius, zone diameter using radial profile analysis."""
    max_r = min(150, int(pr * 0.8))
    dist_to_edge = math.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
    max_r = min(max_r, int(pr - dist_to_edge - 20))
    max_r = max(max_r, 40)

    prof = radial_profile(gray, cx, cy, max_r)
    prof_s = smooth(prof, window=5)
    deriv = np.diff(prof_s)

    search_end = min(30, len(deriv))
    if search_end < 3:
        return None

    # Find drop edge
    center_val = prof_s[0]
    drop_edge_r = np.argmax(deriv[:search_end]) + 1
    if drop_edge_r + 10 < len(prof_s):
        near_lawn = prof_s[drop_edge_r + 5:drop_edge_r + 15].mean()
    else:
        near_lawn = prof_s[-10:].mean()
    half_val = (center_val + near_lawn) / 2
    for r in range(len(prof_s)):
        if prof_s[r] > half_val:
            drop_edge_r = r + 1
            break
    drop_radius = drop_edge_r

    # Drop properties
    drop_area = math.pi * drop_radius ** 2
    drop_pixels = []
    for r in range(1, max(2, drop_radius)):
        angles = np.linspace(0, 2 * math.pi, max(12, int(2 * math.pi * r)), endpoint=False)
        xs = (cx + r * np.cos(angles)).astype(int)
        ys = (cy + r * np.sin(angles)).astype(int)
        xs = np.clip(xs, 0, gray.shape[1] - 1)
        ys = np.clip(ys, 0, gray.shape[0] - 1)
        drop_pixels.extend(gray[ys, xs].tolist())
    drop_mean_intensity = float(np.mean(drop_pixels)) if drop_pixels else float(gray[cy, cx])

    # Find zone boundary
    search_start = drop_edge_r + 3
    if search_start >= len(prof_s) - 5:
        return {
            'drop_radius': drop_radius, 'drop_area': drop_area,
            'drop_mean_intensity': drop_mean_intensity,
            'zone_radius': drop_radius, 'zone_diameter': drop_radius * 2,
            'zone_area': math.pi * drop_radius ** 2, 'has_zone': False
        }

    far_start = min(len(prof_s) - 10, max(60, search_start + 30))
    lawn_level = prof_s[far_start:].mean() if far_start < len(prof_s) else prof_s[-10:].mean()

    search_profile = prof_s[search_start:]
    if len(search_profile) < 10:
        return {
            'drop_radius': drop_radius, 'drop_area': drop_area,
            'drop_mean_intensity': drop_mean_intensity,
            'zone_radius': drop_radius, 'zone_diameter': drop_radius * 2,
            'zone_area': math.pi * drop_radius ** 2, 'has_zone': False
        }

    search_deriv = np.diff(search_profile)
    peaks = []
    for i in range(1, len(search_deriv)):
        if search_deriv[i - 1] > 0 and search_deriv[i] <= 0:
            peak_r = search_start + i
            if peak_r < len(prof_s) and prof_s[peak_r] > lawn_level + 3:
                peaks.append((peak_r, prof_s[peak_r]))

    zone_region = prof_s[search_start:min(search_start + 60, len(prof_s))]
    zone_dip = lawn_level - zone_region.min()

    has_zone = False
    zone_radius = drop_radius

    if peaks:
        zone_radius = peaks[0][0] + 1
        has_zone = True
    elif zone_dip > 5:
        for r in range(search_start, len(prof_s)):
            if prof_s[r] >= lawn_level - 2:
                zone_radius = r + 1
                has_zone = True
                break

    return {
        'drop_radius': drop_radius,
        'drop_area': drop_area,
        'drop_mean_intensity': drop_mean_intensity,
        'zone_radius': zone_radius,
        'zone_diameter': zone_radius * 2,
        'zone_area': math.pi * zone_radius ** 2,
        'has_zone': has_zone
    }


def analyze_image(rgb, gray):
    """Full analysis pipeline. Returns plates list and measurements list."""
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 1)
    plates = detect_plates(gray_blur)

    all_measurements = []
    for plate_idx, (pcx, pcy, pr) in enumerate(plates):
        drops = find_drops_in_plate(gray_blur, pcx, pcy, pr)
        plate_measurements = []
        for cx, cy, intensity, area in drops:
            result = measure_drop_and_zone(gray_blur, cx, cy, pcx, pcy, pr)
            if result is None:
                continue
            m = {'plate': plate_idx + 1, 'center_x': cx, 'center_y': cy, **result}
            plate_measurements.append(m)
        plate_measurements.sort(key=lambda m: (m['center_y'] // 50, m['center_x']))
        for i, m in enumerate(plate_measurements):
            m['drop_num'] = i + 1
        all_measurements.extend(plate_measurements)

    return plates, all_measurements


def draw_annotations(rgb, plates, measurements):
    """Create annotated image with detected features marked."""
    annotated = rgb.copy()

    for cx, cy, r in plates:
        cv2.circle(annotated, (cx, cy), r, (0, 200, 0), 3)

    for i, m in enumerate(measurements):
        cx, cy = m['center_x'], m['center_y']
        # Blue dot at center
        cv2.circle(annotated, (cx, cy), 5, (0, 80, 255), -1)
        # Cyan circle at drop edge
        cv2.circle(annotated, (cx, cy), m['drop_radius'], (0, 220, 220), 2)
        # Red circle at zone boundary
        if m['has_zone']:
            cv2.circle(annotated, (cx, cy), m['zone_radius'], (255, 50, 50), 2)
        # Number label
        label = f"#{i + 1}"
        cv2.putText(annotated, label, (cx + m['zone_radius'] + 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return annotated


# ─── Streamlit App ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Zone of Inhibition Analyzer", layout="wide")
st.title("Zone of Inhibition Analyzer")
st.markdown("Upload an agar plate image to detect and measure zones of inhibition around spots.")

# File upload
uploaded_file = st.file_uploader(
    "Drag and drop a plate image (TIF, PNG, JPG)",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # ─── Step 1: Analyze image ──────────────────────────────────────────
    with st.spinner("Analyzing image..."):
        rgb, gray = load_and_convert(uploaded_file)
        plates, measurements = analyze_image(rgb, gray)
        annotated = draw_annotations(rgb, plates, measurements)

    st.success(f"Detected **{len(plates)} plate(s)** and **{len(measurements)} spot(s)**.")

    # Show annotated image
    st.subheader("Detected Spots")
    st.image(annotated, use_container_width=True,
             caption="Green = plate boundary, Cyan = drop edge, Red = zone boundary, Yellow = spot number")

    if len(measurements) == 0:
        st.warning("No spots detected. Try a different image or adjust the image quality.")
        st.stop()

    # ─── Step 2: Label spots ────────────────────────────────────────────
    st.subheader("Label Your Spots")
    st.markdown(
        "For each detected spot, enter a **sample type** (e.g. 'Ampicillin', 'Control') "
        "and a **concentration** (numeric, e.g. '100' for 100 µg/mL). "
        "Spots with matching sample types will be grouped in the graphs."
    )

    # Initialize labels in session state
    if "labels" not in st.session_state or st.session_state.get("_file_id") != uploaded_file.name:
        st.session_state.labels = {}
        st.session_state._file_id = uploaded_file.name

    # Create a compact labeling form
    cols_per_row = 3
    spot_rows = [measurements[i:i + cols_per_row] for i in range(0, len(measurements), cols_per_row)]

    for row_spots in spot_rows:
        cols = st.columns(cols_per_row)
        for col, m in zip(cols, row_spots):
            idx = measurements.index(m)
            with col:
                st.markdown(f"**Spot #{idx + 1}** (Plate {m['plate']})")
                sample = st.text_input(
                    "Sample type", value=st.session_state.labels.get(idx, {}).get("sample", ""),
                    key=f"sample_{idx}", label_visibility="collapsed",
                    placeholder="Sample type (e.g. Ampicillin)"
                )
                conc = st.text_input(
                    "Concentration", value=st.session_state.labels.get(idx, {}).get("conc", ""),
                    key=f"conc_{idx}", label_visibility="collapsed",
                    placeholder="Concentration (e.g. 100)"
                )
                st.session_state.labels[idx] = {"sample": sample, "conc": conc}

    # ─── Step 3: Results ────────────────────────────────────────────────
    st.subheader("Measurements")

    # Build results dataframe
    rows = []
    for idx, m in enumerate(measurements):
        label = st.session_state.labels.get(idx, {})
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
            "Spot #": idx + 1,
            "Plate": m['plate'],
            "Sample Type": sample_type,
            "Concentration": conc_val,
            "Conc Label": conc_str if conc_str else "N/A",
            "Center X": m['center_x'],
            "Center Y": m['center_y'],
            "Drop Radius (px)": round(m['drop_radius'], 1),
            "Drop Area (px²)": round(m['drop_area'], 1),
            "Drop Intensity (0-255)": round(m['drop_mean_intensity'], 1),
            "Zone Diameter (px)": round(m['zone_diameter'], 1),
            "Zone Radius (px)": round(m['zone_radius'], 1),
            "Zone Area (px²)": round(m['zone_area'], 1),
            "Has Zone": m['has_zone'],
            "ZOI Diam / Drop Area": round(zoi_per_area, 4),
            "ZOI Diam / Drop Circumference": round(zoi_per_circ, 4),
        })

    df = pd.DataFrame(rows)

    # Display table
    display_cols = [
        "Spot #", "Plate", "Sample Type", "Conc Label",
        "Drop Radius (px)", "Drop Intensity (0-255)",
        "Zone Diameter (px)", "Has Zone",
        "ZOI Diam / Drop Area", "ZOI Diam / Drop Circumference"
    ]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    # CSV download
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download CSV",
        csv_buffer.getvalue(),
        file_name=f"measurements_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
        mime="text/csv"
    )

    # ─── Step 4: Graphs ─────────────────────────────────────────────────
    st.subheader("Summary Graphs")

    has_labels = any(r["Sample Type"] != "Unlabeled" for r in rows)
    has_conc = any(r["Concentration"] is not None for r in rows)

    if not has_labels:
        st.info("Label your spots above to generate grouped summary graphs.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Zone Diameter by Sample",
            "Zone Diameter vs Concentration",
            "Spot Intensity by Sample",
            "Normalized ZOI"
        ])

        color_col = "Sample Type"

        with tab1:
            fig = px.strip(
                df, x="Sample Type", y="Zone Diameter (px)",
                color="Sample Type",
                title="Zone of Inhibition Diameter by Sample Type",
                hover_data=["Spot #", "Plate", "Conc Label"]
            )
            fig.update_layout(showlegend=False)
            # Add box overlay
            fig2 = px.box(df, x="Sample Type", y="Zone Diameter (px)", color="Sample Type")
            for trace in fig2.data:
                trace.showlegend = False
                fig.add_trace(trace)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if has_conc:
                df_conc = df[df["Concentration"].notna()].copy()
                fig = px.scatter(
                    df_conc, x="Concentration", y="Zone Diameter (px)",
                    color="Sample Type", symbol="Sample Type",
                    title="Zone Diameter vs Concentration",
                    hover_data=["Spot #", "Plate"],
                    trendline="ols" if len(df_conc) > 2 else None
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Enter concentration values for spots to see this graph.")

        with tab3:
            fig = px.strip(
                df, x="Sample Type", y="Drop Intensity (0-255)",
                color="Sample Type",
                title="Spot Darkness by Sample Type (lower = darker)",
                hover_data=["Spot #", "Plate", "Conc Label"]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            fig = px.strip(
                df, x="Sample Type", y="ZOI Diam / Drop Area",
                color="Sample Type",
                title="Zone Diameter Normalized by Drop Area",
                hover_data=["Spot #", "Plate", "Conc Label"]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.strip(
                df, x="Sample Type", y="ZOI Diam / Drop Circumference",
                color="Sample Type",
                title="Zone Diameter Normalized by Drop Circumference",
                hover_data=["Spot #", "Plate", "Conc Label"]
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ─── Annotated image download ───────────────────────────────────────
    st.subheader("Download Annotated Image")
    img_buffer = io.BytesIO()
    Image.fromarray(annotated).save(img_buffer, format="PNG")
    st.download_button(
        "Download Annotated Image",
        img_buffer.getvalue(),
        file_name=f"annotated_{uploaded_file.name.rsplit('.', 1)[0]}.png",
        mime="image/png"
    )
