"""
Zone of Inhibition Measurement Tool — Streamlit Web App
Drag-and-drop plate images, drag spots around on the plate, label them,
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
import json
from streamlit_drawable_canvas import st_canvas


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
        thresh_img = np.zeros(gray.shape, dtype=np.uint8)
        thresh_img[directed > sensitivity] = 255
        thresh_img[mask == 0] = 0
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, k5)
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
    if len(spots) < 4:
        return spots
    radii = sorted([s["radius"] for s in spots])
    q75 = radii[int(len(radii) * 0.75)]
    min_r = q75 * 0.5
    filtered = [s for s in spots if s["radius"] >= min_r]
    if len(filtered) < len(spots) * 0.4:
        return spots
    return filtered


def measure_spot(gray_clean, diff, spot, plate):
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

    drop_radius = max(3, radius // 5)
    if center_intensity < prof_g[min(radius, len(prof_g) - 1)] - 10:
        half_val = (center_intensity + prof_g[min(radius, len(prof_g) - 1)]) / 2
        for r in range(len(prof_g)):
            if prof_g[r] > half_val:
                drop_radius = r + 1
                break

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
    plates = detect_plates(gray, rgb)
    spots = []
    diff_cache = {}
    median_cache = {}
    for pi, plate in enumerate(plates):
        raw_spots, diff, median = find_spots_in_plate(gray, rgb, plate, sensitivity)
        diff_cache[pi] = diff
        median_cache[pi] = median
        for cx, cy, radius, circ, strength in raw_spots:
            spots.append({"x": cx, "y": cy, "radius": radius,
                          "plate_idx": pi, "label": ""})
    spots = filter_spots_by_size(spots)
    spots.sort(key=lambda s: (s["plate_idx"], s["y"] // 50, s["x"]))
    return plates, spots, diff_cache, median_cache


# ─── Drawing ────────────────────────────────────────────────────────────────


def get_font():
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, 20)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_background(rgb, plates, spots, scale):
    """Draw plate outlines, zone rings, labels, and spot numbers on the background.
    The draggable circles are handled by the canvas — this is everything else."""
    h, w = rgb.shape[:2]
    dw, dh = int(w / scale), int(h / scale)
    img = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = get_font()

    # Plate outlines
    for plate in plates:
        cx, cy, r = plate["cx"] / scale, plate["cy"] / scale, plate["r"] / scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#00C800", width=2)

    # Zone boundaries + labels + numbers (everything except the draggable circle)
    for i, s in enumerate(spots):
        x, y, r = s["x"] / scale, s["y"] / scale, s["radius"] / scale
        zone_r = s.get("zone_radius", s["radius"]) / scale

        # Zone boundary (red ring)
        if zone_r > r + 2:
            draw.ellipse([x - zone_r, y - zone_r, x + zone_r, y + zone_r],
                         outline="#FF4444", width=2)

        # Spot number
        num = f"#{i + 1}"
        tx = x + max(r, zone_r) + 4
        ty = y - 10
        draw.rectangle([tx - 1, ty - 1, tx + 30, ty + 18], fill="#000000")
        draw.text((tx + 1, ty), num, fill="#FFFF00", font=font)

        # Label
        label = s.get("label", "")
        if label:
            lx = x + max(r, zone_r) + 4
            ly = y + 10
            tw = len(label) * 10 + 6
            draw.rectangle([lx - 1, ly - 1, lx + tw, ly + 18], fill="#000000BB")
            draw.text((lx + 1, ly), label, fill="#FFFFFF", font=font)

    return img


def spots_to_canvas_objects(spots, scale):
    """Convert our spot list to Fabric.js circle objects for the canvas."""
    objects = []
    for i, s in enumerate(spots):
        r = s["radius"] / scale
        objects.append({
            "type": "circle",
            "left": s["x"] / scale - r,
            "top": s["y"] / scale - r,
            "radius": r,
            "fill": "rgba(100, 255, 100, 0.15)",
            "stroke": "#66FF66",
            "strokeWidth": 2,
            "name": f"spot_{i}",
        })
    return {"version": "4.4.0", "objects": objects}


def canvas_objects_to_spots(json_data, spots, scale):
    """Read back canvas circle positions and update spot coordinates.
    Returns True if any positions changed."""
    if json_data is None or "objects" not in json_data:
        return False
    changed = False
    canvas_objs = json_data["objects"]
    for obj in canvas_objs:
        name = obj.get("name", "")
        if not name.startswith("spot_"):
            continue
        try:
            idx = int(name.split("_")[1])
        except (ValueError, IndexError):
            continue
        if idx >= len(spots):
            continue

        r = obj["radius"] * obj.get("scaleX", 1)
        new_x = int((obj["left"] + r) * scale)
        new_y = int((obj["top"] + r) * scale)

        if abs(new_x - spots[idx]["x"]) > 2 or abs(new_y - spots[idx]["y"]) > 2:
            spots[idx]["x"] = new_x
            spots[idx]["y"] = new_y
            changed = True
    return changed


def assign_plate(plates, x, y):
    for i, p in enumerate(plates):
        d = math.sqrt((p["cx"] - x) ** 2 + (p["cy"] - y) ** 2)
        if d < p["r"] - 20:
            return i
    return 0


def cluster_1d(values, min_gap_frac=0.3):
    """Cluster sorted 1D values into groups separated by large gaps.
    Returns list of (cluster_center, [indices])."""
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    clusters = [[indexed[0]]]
    for i in range(1, len(indexed)):
        gap = indexed[i][1] - indexed[i - 1][1]
        # A gap > min_gap_frac of the plate radius separates clusters
        if gap > min_gap_frac:
            clusters.append([])
        clusters[-1].append(indexed[i])
    result = []
    for cluster in clusters:
        center = np.mean([v for _, v in cluster])
        indices = [idx for idx, _ in cluster]
        result.append((center, indices))
    result.sort(key=lambda c: c[0])
    return result


def detect_grid(spots, plates):
    """Detect row/column structure within each plate.
    Returns spots with 'row' and 'col' assignments."""
    for plate_idx in range(len(plates)):
        plate = plates[plate_idx]
        plate_spots = [(i, s) for i, s in enumerate(spots) if s["plate_idx"] == plate_idx]
        if not plate_spots:
            continue

        # Use position relative to plate center
        pr = plate["r"]
        gap_threshold = pr * 0.08  # ~8% of plate radius separates rows/cols

        # Cluster by Y for rows
        ys = [s["y"] for _, s in plate_spots]
        y_clusters = cluster_1d(ys, gap_threshold)
        for row_idx, (_, indices) in enumerate(y_clusters):
            for idx in indices:
                spots[plate_spots[idx][0]]["row"] = row_idx

        # Cluster by X for columns
        xs = [s["x"] for _, s in plate_spots]
        x_clusters = cluster_1d(xs, gap_threshold)
        for col_idx, (_, indices) in enumerate(x_clusters):
            for idx in indices:
                spots[plate_spots[idx][0]]["col"] = col_idx

    return spots


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
        "then you can **drag spots** to adjust positions, label them, and generate measurements."
    )
    st.stop()

# ─── Sidebar ────────────────────────────────────────────────────────────────

file_id = uploaded_file.name + str(uploaded_file.size)

with st.sidebar:
    st.header("Detection")
    sensitivity = st.slider("Sensitivity (lower = more spots)",
                            min_value=2.0, max_value=8.0, value=5.0, step=0.5)
    if st.button("Re-detect spots", type="primary"):
        st.session_state.pop("_spots", None)
        st.session_state._canvas_version = st.session_state.get("_canvas_version", 0) + 1

    st.divider()
    st.header("Editing")
    mode = st.radio("Mode", ["Move spots", "Add spot"],
                    help="**Move**: drag existing spots. **Add**: click to place a new spot.")

    st.divider()
    st.header("Instructions")
    st.markdown(
        "1. Upload an image\n"
        "2. Adjust sensitivity if needed\n"
        "3. **Drag spots** to correct positions\n"
        "4. Switch to *Add spot* mode to add missing spots\n"
        "5. Label spots below the image\n"
        "6. Download CSV and graphs"
    )

# ─── Load image ─────────────────────────────────────────────────────────────

if st.session_state.get("_file_id") != file_id:
    rgb, gray = load_and_convert(uploaded_file)
    st.session_state._file_id = file_id
    st.session_state._rgb = rgb
    st.session_state._gray = gray
    st.session_state.pop("_spots", None)
    st.session_state._canvas_version = 0

rgb = st.session_state._rgb
gray = st.session_state._gray

# ─── Detect ─────────────────────────────────────────────────────────────────

if "_spots" not in st.session_state:
    with st.spinner("Detecting plates and spots..."):
        plates, spots, diff_cache, median_cache = run_detection(rgb, gray, sensitivity)
        for s in spots:
            pi = s["plate_idx"]
            m = measure_spot(median_cache[pi], diff_cache[pi], s, plates[pi])
            s.update(m)
        spots = detect_grid(spots, plates)
        st.session_state._plates = plates
        st.session_state._spots = spots
        st.session_state._diff = diff_cache
        st.session_state._median = median_cache

plates = st.session_state._plates
spots = st.session_state._spots

# ─── Canvas ─────────────────────────────────────────────────────────────────

st.subheader(f"{len(plates)} plate(s), {len(spots)} spot(s) — drag to adjust")

# Scale for display
DISPLAY_WIDTH = min(1100, rgb.shape[1])
scale = rgb.shape[1] / DISPLAY_WIDTH
display_h = int(rgb.shape[0] / scale)

# Background: plate image with outlines, zone rings, labels (no draggable circles)
bg_img = draw_background(rgb, plates, spots, scale)

# Canvas drawing mode
if mode == "Move spots":
    drawing_mode = "transform"
else:
    drawing_mode = "circle"

# Build initial drawing from spots
canvas_version = st.session_state.get("_canvas_version", 0)
initial = spots_to_canvas_objects(spots, scale)

canvas_result = st_canvas(
    fill_color="rgba(100, 255, 100, 0.15)",
    stroke_width=2,
    stroke_color="#66FF66",
    background_image=bg_img,
    initial_drawing=initial,
    update_streamlit=True,
    height=display_h,
    width=DISPLAY_WIDTH,
    drawing_mode=drawing_mode,
    key=f"canvas_{canvas_version}",
)

# ─── Sync canvas → spots ───────────────────────────────────────────────────

if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])

    # Check for newly drawn circles (from "Add spot" mode)
    existing_names = {f"spot_{i}" for i in range(len(spots))}
    new_circles = [o for o in objects
                   if o["type"] == "circle" and o.get("name", "") not in existing_names]
    for obj in new_circles:
        r_canvas = obj["radius"] * obj.get("scaleX", 1)
        cx = int((obj["left"] + r_canvas) * scale)
        cy = int((obj["top"] + r_canvas) * scale)
        r_orig = int(r_canvas * scale)
        if r_orig < 5:
            r_orig = spots[0]["radius"] if spots else 35

        pi = assign_plate(plates, cx, cy)
        new_spot = {"x": cx, "y": cy, "radius": r_orig, "plate_idx": pi, "label": ""}
        diff_cache = st.session_state._diff
        median_cache = st.session_state._median
        if pi in diff_cache:
            m = measure_spot(median_cache[pi], diff_cache[pi], new_spot, plates[pi])
            new_spot.update(m)
        else:
            new_spot.update({"zone_radius": r_orig, "zone_diameter": r_orig * 2,
                             "zone_area": math.pi * r_orig ** 2, "has_zone": False,
                             "drop_radius": 3, "drop_area": 28.3, "drop_intensity": 128.0})
        spots.append(new_spot)
        spots.sort(key=lambda s: (s["plate_idx"], s["y"] // 50, s["x"]))
        st.session_state._spots = spots
        st.session_state._canvas_version = canvas_version + 1
        st.rerun()

    # Sync moved positions
    if canvas_objects_to_spots(canvas_result.json_data, spots, scale):
        # Re-measure moved spots
        diff_cache = st.session_state._diff
        median_cache = st.session_state._median
        for s in spots:
            pi = s["plate_idx"]
            if pi in diff_cache:
                m = measure_spot(median_cache[pi], diff_cache[pi], s, plates[pi])
                s.update(m)
        st.session_state._spots = spots

# ─── Spot labeling (grid-based) ─────────────────────────────────────────────

st.divider()
st.subheader("Label Spots")
st.markdown(
    "Spots in the **same row** share a sample type. "
    "Spots in the **same column** share a concentration. "
    "Label rows and columns below — each spot auto-labels from its row + column."
)

# Collect unique rows and columns across all plates
all_rows = sorted(set(s.get("row", 0) for s in spots))
all_cols = sorted(set(s.get("col", 0) for s in spots))

# Initialize row/col labels in session state
if "_row_labels" not in st.session_state:
    st.session_state._row_labels = {}
if "_col_labels" not in st.session_state:
    st.session_state._col_labels = {}

row_labels = st.session_state._row_labels
col_labels = st.session_state._col_labels

# Per-plate labeling
for plate_idx in range(len(plates)):
    plate_spots = [s for s in spots if s["plate_idx"] == plate_idx]
    if not plate_spots:
        continue

    p_rows = sorted(set(s.get("row", 0) for s in plate_spots))
    p_cols = sorted(set(s.get("col", 0) for s in plate_spots))

    st.markdown(f"**Plate {plate_idx + 1}** — {len(plate_spots)} spots "
                f"({len(p_rows)} rows x {len(p_cols)} columns)")

    # Row labels (sample types)
    r_cols = st.columns(max(1, len(p_rows)))
    for ri, row_id in enumerate(p_rows):
        key = f"p{plate_idx}_r{row_id}"
        with r_cols[ri % len(r_cols)]:
            val = st.text_input(
                f"Row {ri + 1} sample",
                value=row_labels.get(key, ""),
                key=f"rlbl_{key}",
                placeholder=f"Sample for row {ri + 1}"
            )
            row_labels[key] = val

    # Column labels (concentrations)
    c_cols = st.columns(max(1, len(p_cols)))
    for ci, col_id in enumerate(p_cols):
        key = f"p{plate_idx}_c{col_id}"
        with c_cols[ci % len(c_cols)]:
            val = st.text_input(
                f"Col {ci + 1} conc",
                value=col_labels.get(key, ""),
                key=f"clbl_{key}",
                placeholder=f"Conc for col {ci + 1}"
            )
            col_labels[key] = val

st.session_state._row_labels = row_labels
st.session_state._col_labels = col_labels

# Apply row/col labels to each spot
for s in spots:
    pi = s["plate_idx"]
    r_key = f"p{pi}_r{s.get('row', 0)}"
    c_key = f"p{pi}_c{s.get('col', 0)}"
    sample = row_labels.get(r_key, "").strip()
    conc = col_labels.get(c_key, "").strip()
    if sample and conc:
        s["label"] = f"{sample}, {conc}"
    elif sample:
        s["label"] = sample
    elif conc:
        s["label"] = conc
    else:
        s["label"] = ""
st.session_state._spots = spots

# Delete individual spots
with st.expander("Delete individual spots"):
    for i, s in enumerate(spots):
        c1, c2 = st.columns([6, 1])
        with c1:
            st.text(f"#{i+1} P{s['plate_idx']+1} R{s.get('row',0)+1}C{s.get('col',0)+1} — {s.get('label','')}")
        with c2:
            if st.button("X", key=f"del_{i}"):
                spots.pop(i)
                st.session_state._spots = spots
                st.session_state._canvas_version = canvas_version + 1
                st.rerun()

# ─── Measurements ───────────────────────────────────────────────────────────

st.divider()
st.subheader("Measurements")

rows = []
for i, s in enumerate(spots):
    raw_label = s.get("label", "").strip()
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
    da = s.get("drop_area", math.pi * dr ** 2)
    zd = s.get("zone_diameter", s["radius"] * 2)
    drop_circ = 2 * math.pi * dr
    rows.append({
        "Spot": i + 1, "Plate": s["plate_idx"] + 1,
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
        "ZOI / Drop Area": round(zd / da if da > 0 else 0, 4),
        "ZOI / Drop Circ": round(zd / drop_circ if drop_circ > 0 else 0, 4),
    })

df = pd.DataFrame(rows)
st.dataframe(df[["Spot", "Plate", "Label", "Spot Radius (px)", "Drop Intensity",
                  "Zone Diameter (px)", "Has Zone", "ZOI / Drop Area", "ZOI / Drop Circ"]],
             use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download CSV", buf.getvalue(),
                       file_name=f"measurements_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                       mime="text/csv")
with c2:
    # Build final annotated image at full resolution for download
    final_img = draw_background(rgb, plates, spots, 1.0)  # scale=1 → full res
    # Add the spot circles at full res
    draw = ImageDraw.Draw(final_img)
    for s in spots:
        r = s["radius"]
        draw.ellipse([s["x"] - r, s["y"] - r, s["x"] + r, s["y"] + r],
                     outline="#66FF66", width=3)
    img_buf = io.BytesIO()
    final_img.save(img_buf, format="PNG")
    st.download_button("Download Annotated Image", img_buf.getvalue(),
                       file_name=f"annotated_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                       mime="image/png")

# ─── Graphs ─────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Summary Graphs")

has_labels = any(r["Sample Type"] != "Unlabeled" for r in rows)
has_conc = any(r["Concentration"] is not None for r in rows)

if not has_labels:
    st.info("Label your spots to see graphs. Format: **Sample, Concentration** (e.g. `Amp, 100`)")
else:
    df_lab = df[df["Sample Type"] != "Unlabeled"]
    tab1, tab2, tab3, tab4 = st.tabs(["ZOI by Sample", "ZOI vs Conc", "Intensity", "Normalized"])

    with tab1:
        fig = px.box(df_lab, x="Sample Type", y="Zone Diameter (px)",
                     color="Sample Type", points="all",
                     title="Zone of Inhibition by Sample Type",
                     hover_data=["Spot", "Plate", "Conc Label"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if has_conc:
            df_c = df_lab[df_lab["Concentration"].notna()]
            if len(df_c) > 0:
                fig = px.scatter(df_c, x="Concentration", y="Zone Diameter (px)",
                                 color="Sample Type", symbol="Sample Type",
                                 title="Zone Diameter vs Concentration",
                                 trendline="ols" if len(df_c) > 2 else None)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add concentrations to see this graph.")

    with tab3:
        fig = px.box(df_lab, x="Sample Type", y="Drop Intensity",
                     color="Sample Type", points="all",
                     title="Spot Darkness (lower = darker)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = px.box(df_lab, x="Sample Type", y="ZOI / Drop Area",
                     color="Sample Type", points="all", title="ZOI / Drop Area")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.box(df_lab, x="Sample Type", y="ZOI / Drop Circ",
                      color="Sample Type", points="all", title="ZOI / Drop Circumference")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
