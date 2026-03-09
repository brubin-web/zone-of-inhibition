"""
Zone of Inhibition Measurement Tool — Streamlit Web App
Upload agar plate images, interactively edit spots, label them on the plate,
and get measurements + summary graphs.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import plotly.express as px
import io
import math
from streamlit_image_coordinates import streamlit_image_coordinates


# ─── Image Analysis ─────────────────────────────────────────────────────────


def load_and_convert(uploaded_file):
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def detect_plates(gray, rgb):
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
            if area / (math.pi * r * r) > 0.7 and r > 80:
                plates.append({"cx": int(x), "cy": int(y), "r": int(r)})
        if plates:
            break
    plates.sort(key=lambda p: (p["cy"] // 300, p["cx"]))
    return plates


def remove_pen_marks(gray, rgb, mask):
    pen_mask = np.zeros(gray.shape, dtype=np.uint8)
    r_ch, g_ch, b_ch = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
    pen_mask[(b_ch - r_ch > 15) & (mask > 0)] = 255
    pen_mask[(r_ch - g_ch > 40) & (mask > 0)] = 255
    pen_mask = cv2.dilate(pen_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    return cv2.inpaint(gray, pen_mask, 25, cv2.INPAINT_TELEA)


def find_spots_in_plate(gray, rgb, plate, sensitivity=5.0):
    pcx, pcy, pr = plate["cx"], plate["cy"], plate["r"]
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (pcx, pcy), pr - 70, 255, -1)
    clean = remove_pen_marks(gray, rgb, mask)
    median = cv2.medianBlur(clean, 21)
    bg = cv2.GaussianBlur(median, (201, 201), 60)
    diff = median.astype(np.float32) - bg.astype(np.float32)
    diff[mask == 0] = 0

    all_contours = []
    for sign in [1, -1]:
        directed = sign * diff
        thresh = np.zeros(gray.shape, dtype=np.uint8)
        thresh[directed > sensitivity] = 255
        thresh[mask == 0] = 0
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
            contour_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [c], -1, 255, -1)
            mean_diff = abs(diff[contour_mask > 0].mean())
            if mean_diff > sensitivity * 0.6:
                all_contours.append((int(x), int(y), int(radius), circ, mean_diff))

    # Merge nearby
    merged = []
    used = set()
    all_contours.sort(key=lambda s: -(s[3] * s[4]))
    for i, item in enumerate(all_contours):
        if i in used:
            continue
        for j, other in enumerate(all_contours):
            if j != i and j not in used:
                if abs(item[0] - other[0]) < 60 and abs(item[1] - other[1]) < 60:
                    used.add(j)
        used.add(i)
        merged.append(item)

    return merged, diff, median


def filter_spots_by_size(spots):
    """Keep only spots near the typical size (real spots are similar size).
    Uses upper-quartile radius as the reference, keeps spots > 50% of that."""
    if len(spots) < 4:
        return spots
    radii = sorted([s["radius"] for s in spots])
    q75 = radii[int(len(radii) * 0.75)]
    min_r = q75 * 0.5
    filtered = [s for s in spots if s["radius"] >= min_r]
    # Don't filter if it removes more than half — detection may just be noisy
    if len(filtered) < len(spots) * 0.4:
        return spots
    return filtered


def measure_spot(gray_clean, diff, spot, plate):
    """Measure zone of inhibition using radial profile from spot center."""
    cx, cy, radius = spot["x"], spot["y"], spot["radius"]
    pcx, pcy, pr = plate["cx"], plate["cy"], plate["r"]

    max_r = min(150, int(pr * 0.8))
    dist = math.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
    max_r = min(max_r, int(pr - dist - 20))
    max_r = max(max_r, 50)

    prof_diff = np.zeros(max_r)
    prof_gray = np.zeros(max_r)
    for r in range(1, max_r + 1):
        na = max(24, int(2 * math.pi * r))
        angles = np.linspace(0, 2 * math.pi, na, endpoint=False)
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, diff.shape[1] - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, diff.shape[0] - 1)
        prof_diff[r - 1] = diff[ys, xs].mean()
        prof_gray[r - 1] = gray_clean[ys, xs].mean()

    kernel = np.ones(5) / 5
    prof_d = np.convolve(prof_diff, kernel, mode='same')
    prof_g = np.convolve(prof_gray, kernel, mode='same')

    center_intensity = float(gray_clean[cy, cx])
    abs_prof = np.abs(prof_d)
    peak_diff = abs_prof[:min(100, len(abs_prof))].max()
    threshold = peak_diff * 0.3

    zone_radius = radius
    for r in range(max(5, radius - 10), min(len(abs_prof), radius + 30)):
        if abs_prof[r] < threshold:
            zone_radius = r + 1
            break

    # Drop radius: look for dark center
    drop_radius = max(3, radius // 5)
    if center_intensity < prof_g[min(radius, len(prof_g) - 1)] - 10:
        half_val = (center_intensity + prof_g[min(radius, len(prof_g) - 1)]) / 2
        for r in range(len(prof_g)):
            if prof_g[r] > half_val:
                drop_radius = r + 1
                break

    # Drop intensity
    drop_pixels = []
    for r in range(1, max(2, drop_radius + 1)):
        na = max(12, int(2 * math.pi * r))
        angles = np.linspace(0, 2 * math.pi, na, endpoint=False)
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, gray_clean.shape[1] - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, gray_clean.shape[0] - 1)
        drop_pixels.extend(gray_clean[ys, xs].tolist())
    drop_intensity = float(np.mean(drop_pixels)) if drop_pixels else center_intensity

    return {
        "drop_radius": drop_radius,
        "drop_area": math.pi * drop_radius ** 2,
        "drop_intensity": drop_intensity,
        "zone_radius": zone_radius,
        "zone_diameter": zone_radius * 2,
        "zone_area": math.pi * zone_radius ** 2,
        "has_zone": zone_radius > drop_radius + 5,
    }


def run_detection(rgb, gray, sensitivity=5.0):
    """Detect plates and spots. Returns plates list and spots list."""
    plates = detect_plates(gray, rgb)
    spots = []
    diff_cache = {}
    median_cache = {}

    for pi, plate in enumerate(plates):
        raw_spots, diff, median = find_spots_in_plate(gray, rgb, plate, sensitivity)
        diff_cache[pi] = diff
        median_cache[pi] = median
        for cx, cy, radius, circ, strength in raw_spots:
            spots.append({
                "x": cx, "y": cy, "radius": radius,
                "plate_idx": pi, "label": "", "circ": circ,
            })

    # Size filter: reject outlier-sized detections
    spots = filter_spots_by_size(spots)
    # Sort by plate then position
    spots.sort(key=lambda s: (s["plate_idx"], s["y"] // 50, s["x"]))

    return plates, spots, diff_cache, median_cache


# ─── Drawing ────────────────────────────────────────────────────────────────


def draw_annotated(rgb, plates, spots, selected_idx=None):
    """Draw the annotated plate image with spots, labels, and zone circles."""
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
            font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_sm = font

    # Plate outlines
    for plate in plates:
        cx, cy, r = plate["cx"], plate["cy"], plate["r"]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 200, 0), width=3)

    # Spots
    for i, s in enumerate(spots):
        x, y, r = s["x"], s["y"], s["radius"]
        is_selected = (i == selected_idx)
        label = s.get("label", "")

        # Zone boundary (red)
        zone_r = s.get("zone_radius", r)
        if zone_r > r:
            color = (255, 100, 100) if not is_selected else (255, 0, 255)
            draw.ellipse([x - zone_r, y - zone_r, x + zone_r, y + zone_r],
                         outline=color, width=2)

        # Spot boundary (green, or magenta if selected)
        outline_color = (255, 0, 255) if is_selected else (100, 255, 100)
        width = 3 if is_selected else 2
        draw.ellipse([x - r, y - r, x + r, y + r], outline=outline_color, width=width)

        # Center dot
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(50, 100, 255))

        # Spot number
        num_text = f"#{i + 1}"
        tx, ty = x + max(r, zone_r) + 6, y - 12
        # Background for readability
        draw.rectangle([tx - 1, ty - 1, tx + 40, ty + 20], fill=(0, 0, 0, 160))
        draw.text((tx, ty), num_text, fill=(255, 255, 0), font=font_sm)

        # Label on the plate
        if label:
            lx, ly = x + max(r, zone_r) + 6, y + 12
            text_w = len(label) * 12 + 8
            draw.rectangle([lx - 2, ly - 2, lx + text_w, ly + 22], fill=(0, 0, 0, 180))
            draw.text((lx, ly), label, fill=(255, 255, 255), font=font_sm)

    return np.array(img)


# ─── Helpers ────────────────────────────────────────────────────────────────


def find_nearest_spot(spots, x, y, max_dist=80):
    """Find the spot nearest to (x, y), within max_dist pixels."""
    best_idx = None
    best_dist = max_dist
    for i, s in enumerate(spots):
        d = math.sqrt((s["x"] - x) ** 2 + (s["y"] - y) ** 2)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def assign_plate(plates, x, y):
    """Find which plate a point belongs to."""
    for i, p in enumerate(plates):
        d = math.sqrt((p["cx"] - x) ** 2 + (p["cy"] - y) ** 2)
        if d < p["r"] - 20:
            return i
    return 0  # default to first plate


# ─── Streamlit App ──────────────────────────────────────────────────────────


st.set_page_config(page_title="Zone of Inhibition Analyzer", layout="wide")
st.title("Zone of Inhibition Analyzer")

uploaded_file = st.file_uploader(
    "Drag and drop a plate image (TIF, PNG, JPG)",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.markdown(
        "Upload an image of agar plates. The tool will detect plates and spots, "
        "then let you edit detections, label spots, and generate measurements."
    )
    st.stop()

# ─── Initialize / Re-detect ────────────────────────────────────────────────

file_id = uploaded_file.name + str(uploaded_file.size)

# Sidebar controls
with st.sidebar:
    st.header("Detection Settings")
    sensitivity = st.slider(
        "Sensitivity (lower = more spots)",
        min_value=2.0, max_value=8.0, value=5.0, step=0.5,
    )
    if st.button("Re-detect spots", type="primary"):
        st.session_state.pop("_spots", None)

    st.divider()
    st.header("Spot Editing")
    st.markdown(
        "**Click on the image** to:\n"
        "- Select a spot (click near it)\n"
        "- Add a new spot (click on empty area)\n\n"
        "Use controls below to label or delete."
    )

# Load image (once per file)
if st.session_state.get("_file_id") != file_id:
    rgb, gray = load_and_convert(uploaded_file)
    st.session_state._file_id = file_id
    st.session_state._rgb = rgb
    st.session_state._gray = gray
    st.session_state.pop("_spots", None)
    st.session_state.pop("_selected", None)

rgb = st.session_state._rgb
gray = st.session_state._gray

# Run detection if needed
if "_spots" not in st.session_state:
    with st.spinner("Detecting plates and spots..."):
        plates, spots, diff_cache, median_cache = run_detection(rgb, gray, sensitivity)
        # Measure each spot
        for s in spots:
            pi = s["plate_idx"]
            m = measure_spot(median_cache[pi], diff_cache[pi], s, plates[pi])
            s.update(m)
        st.session_state._plates = plates
        st.session_state._spots = spots
        st.session_state._diff = diff_cache
        st.session_state._median = median_cache
        st.session_state._selected = None

plates = st.session_state._plates
spots = st.session_state._spots
selected = st.session_state.get("_selected")

# ─── Display annotated image with click interaction ─────────────────────────

st.subheader(f"Detected {len(plates)} plate(s), {len(spots)} spot(s)")

annotated = draw_annotated(rgb, plates, spots, selected)

# Scale image for display (streamlit_image_coordinates returns coords in display space)
display_width = min(1200, rgb.shape[1])
scale = rgb.shape[1] / display_width

# Convert to PIL for the click widget
pil_annotated = Image.fromarray(annotated)

coords = streamlit_image_coordinates(
    pil_annotated,
    key=f"img_{len(spots)}_{selected}",
    width=display_width,
)

# Handle click
if coords is not None:
    click_x = int(coords["x"] * scale)
    click_y = int(coords["y"] * scale)

    nearest = find_nearest_spot(spots, click_x, click_y, max_dist=80)

    if nearest is not None:
        # Select existing spot
        st.session_state._selected = nearest
    else:
        # Add new spot at click location
        pi = assign_plate(plates, click_x, click_y)
        median_r = sorted([s["radius"] for s in spots])[len(spots) // 2] if spots else 35
        new_spot = {
            "x": click_x, "y": click_y, "radius": median_r,
            "plate_idx": pi, "label": "",
        }
        # Measure the new spot
        diff_cache = st.session_state._diff
        median_cache = st.session_state._median
        if pi in diff_cache:
            m = measure_spot(median_cache[pi], diff_cache[pi], new_spot, plates[pi])
            new_spot.update(m)
        else:
            new_spot.update({"zone_radius": median_r, "zone_diameter": median_r * 2,
                             "zone_area": math.pi * median_r ** 2, "has_zone": False,
                             "drop_radius": 3, "drop_area": 28.3, "drop_intensity": 128.0})
        spots.append(new_spot)
        spots.sort(key=lambda s: (s["plate_idx"], s["y"] // 50, s["x"]))
        st.session_state._spots = spots
        st.session_state._selected = spots.index(new_spot)

    st.rerun()

# ─── Spot editing panel ────────────────────────────────────────────────────

if selected is not None and 0 <= selected < len(spots):
    spot = spots[selected]
    st.divider()
    st.subheader(f"Editing Spot #{selected + 1} (Plate {spot['plate_idx'] + 1})")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        new_label = st.text_input(
            "Label (shown on plate)", value=spot.get("label", ""),
            key="edit_label", placeholder="e.g. Amp 100µg/mL"
        )
        if new_label != spot.get("label", ""):
            spot["label"] = new_label
            st.session_state._spots = spots
            st.rerun()

    with col2:
        if st.button("Delete spot", type="secondary"):
            spots.pop(selected)
            st.session_state._spots = spots
            st.session_state._selected = None
            st.rerun()

    with col3:
        if st.button("Deselect"):
            st.session_state._selected = None
            st.rerun()

    # Fine-tune position
    st.caption("Fine-tune position:")
    pc1, pc2 = st.columns(2)
    with pc1:
        new_x = st.number_input("X", value=spot["x"], step=5, key="edit_x")
    with pc2:
        new_y = st.number_input("Y", value=spot["y"], step=5, key="edit_y")
    if new_x != spot["x"] or new_y != spot["y"]:
        spot["x"] = new_x
        spot["y"] = new_y
        # Re-measure
        pi = spot["plate_idx"]
        diff_cache = st.session_state._diff
        median_cache = st.session_state._median
        if pi in diff_cache:
            m = measure_spot(median_cache[pi], diff_cache[pi], spot, plates[pi])
            spot.update(m)
        st.session_state._spots = spots
        st.rerun()

else:
    st.info("Click on a spot to select it for labeling, or click on empty space to add a new spot.")

# ─── Quick-label all spots ──────────────────────────────────────────────────

with st.expander("Quick-label all spots"):
    st.markdown("Enter labels for all spots at once. Format: **Sample Type, Concentration** (e.g. `Amp, 100`)")
    for i, s in enumerate(spots):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**#{i+1}** (P{s['plate_idx']+1})")
        with col2:
            val = st.text_input(
                f"Label {i+1}", value=s.get("label", ""),
                key=f"qlabel_{i}", label_visibility="collapsed",
                placeholder="Sample, Concentration"
            )
            if val != s.get("label", ""):
                s["label"] = val
                st.session_state._spots = spots

# ─── Measurements ───────────────────────────────────────────────────────────

st.divider()
st.subheader("Measurements")

rows = []
for i, s in enumerate(spots):
    raw_label = s.get("label", "").strip()
    # Parse "Sample, Concentration" format
    if "," in raw_label:
        parts = raw_label.split(",", 1)
        sample_type = parts[0].strip()
        conc_str = parts[1].strip()
    else:
        sample_type = raw_label if raw_label else "Unlabeled"
        conc_str = ""

    try:
        conc_val = float(conc_str) if conc_str else None
    except ValueError:
        conc_val = None

    dr = s.get("drop_radius", 3)
    drop_circ = 2 * math.pi * dr
    da = s.get("drop_area", math.pi * dr ** 2)
    zd = s.get("zone_diameter", s["radius"] * 2)
    zoi_area = zd / da if da > 0 else 0
    zoi_circ = zd / drop_circ if drop_circ > 0 else 0

    rows.append({
        "Spot": i + 1,
        "Plate": s["plate_idx"] + 1,
        "Label": raw_label or "Unlabeled",
        "Sample Type": sample_type,
        "Concentration": conc_val,
        "Conc Label": conc_str or "N/A",
        "X": s["x"], "Y": s["y"],
        "Spot Radius (px)": s["radius"],
        "Drop Radius (px)": dr,
        "Drop Area (px\u00b2)": round(da, 1),
        "Drop Intensity": round(s.get("drop_intensity", 128), 1),
        "Zone Diameter (px)": round(zd, 1),
        "Zone Area (px\u00b2)": round(s.get("zone_area", 0), 1),
        "Has Zone": s.get("has_zone", False),
        "ZOI / Drop Area": round(zoi_area, 4),
        "ZOI / Drop Circ": round(zoi_circ, 4),
    })

df = pd.DataFrame(rows)

display_cols = [
    "Spot", "Plate", "Label",
    "Spot Radius (px)", "Drop Intensity",
    "Zone Diameter (px)", "Has Zone",
    "ZOI / Drop Area", "ZOI / Drop Circ"
]
st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download CSV", csv_buf.getvalue(),
                       file_name=f"measurements_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                       mime="text/csv")
with col_dl2:
    img_buf = io.BytesIO()
    Image.fromarray(annotated).save(img_buf, format="PNG")
    st.download_button("Download Annotated Image", img_buf.getvalue(),
                       file_name=f"annotated_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                       mime="image/png")

# ─── Graphs ─────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Summary Graphs")

has_labels = any(r["Sample Type"] != "Unlabeled" for r in rows)
has_conc = any(r["Concentration"] is not None for r in rows)

if not has_labels:
    st.info("Label your spots to generate summary graphs. Use format: **Sample, Concentration** (e.g. `Amp, 100`)")
else:
    df_labeled = df[df["Sample Type"] != "Unlabeled"].copy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "ZOI by Sample", "ZOI vs Concentration", "Spot Intensity", "Normalized ZOI"
    ])

    with tab1:
        fig = px.box(df_labeled, x="Sample Type", y="Zone Diameter (px)",
                     color="Sample Type", points="all",
                     title="Zone of Inhibition by Sample Type",
                     hover_data=["Spot", "Plate", "Conc Label"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if has_conc:
            df_conc = df_labeled[df_labeled["Concentration"].notna()]
            if len(df_conc) > 0:
                fig = px.scatter(df_conc, x="Concentration", y="Zone Diameter (px)",
                                 color="Sample Type", symbol="Sample Type",
                                 title="Zone Diameter vs Concentration",
                                 hover_data=["Spot", "Plate"],
                                 trendline="ols" if len(df_conc) > 2 else None)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add concentrations to spot labels to see this graph.")

    with tab3:
        fig = px.box(df_labeled, x="Sample Type", y="Drop Intensity",
                     color="Sample Type", points="all",
                     title="Spot Darkness by Sample (lower = darker)",
                     hover_data=["Spot", "Plate", "Conc Label"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = px.box(df_labeled, x="Sample Type", y="ZOI / Drop Area",
                     color="Sample Type", points="all",
                     title="ZOI Normalized by Drop Area",
                     hover_data=["Spot", "Plate"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(df_labeled, x="Sample Type", y="ZOI / Drop Circ",
                      color="Sample Type", points="all",
                      title="ZOI Normalized by Drop Circumference",
                      hover_data=["Spot", "Plate"])
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
