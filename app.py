"""
Zone of Inhibition Measurement Tool — Streamlit Web App
Upload plate images, auto-detect spots (2 rows x 3 columns per plate),
label them, get measurements + graphs.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import plotly.express as px
import io
import math


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
    pen_mask[(b_ch - r_ch > 12) & (mask > 0)] = 255
    pen_mask[(r_ch - g_ch > 30) & (r_ch > 150) & (mask > 0)] = 255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pen_mask[(hsv[:, :, 1] > 130) & (mask > 0)] = 255
    pen_mask = cv2.dilate(pen_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    return cv2.inpaint(gray, pen_mask, 25, cv2.INPAINT_TELEA)


def build_spot_score_map(gray, rgb, plate):
    """Build a map where each pixel's value indicates how spot-like the area is.
    Uses background subtraction + matched circular filter."""
    pcx, pcy, pr = plate["cx"], plate["cy"], plate["r"]
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (pcx, pcy), int(pr * 0.82), 255, -1)

    clean = remove_pen_marks(gray, rgb, mask)
    median = cv2.medianBlur(clean, 21)
    bg = cv2.GaussianBlur(median, (201, 201), 60)
    diff = median.astype(np.float32) - bg.astype(np.float32)
    diff[mask == 0] = 0

    # Absolute difference — spots show up as either lighter or darker than background
    abs_diff = np.abs(diff)

    # Matched filter: smooth with a Gaussian the size of a typical spot (~25px)
    # This boosts real spots (coherent circular signal) and suppresses noise/streaks
    score = cv2.GaussianBlur(abs_diff, (51, 51), 20)
    score[mask == 0] = 0

    return score, diff, median


def score_grid_position(score, x, y, radius=25):
    """Get the peak score within radius of a grid position."""
    h, w = score.shape
    ix, iy = int(x), int(y)
    y1 = max(0, iy - radius)
    y2 = min(h, iy + radius + 1)
    x1 = max(0, ix - radius)
    x2 = min(w, ix + radius + 1)
    patch = score[y1:y2, x1:x2]
    return float(patch.max()) if patch.size > 0 else 0.0


def snap_to_peak(score, x, y, radius=30):
    """Snap position to the highest score point within radius."""
    h, w = score.shape
    x0, y0 = int(x), int(y)
    best_val = -1
    best_x, best_y = x0, y0
    for dy in range(-radius, radius + 1, 2):
        for dx in range(-radius, radius + 1, 2):
            if dx * dx + dy * dy > radius * radius:
                continue
            px, py = x0 + dx, y0 + dy
            if 0 <= px < w and 0 <= py < h and score[py, px] > best_val:
                best_val = score[py, px]
                best_x, best_y = px, py
    return best_x, best_y


def find_clear_spots(score, plate, min_distance=80, max_count=6):
    """Find clearly visible spots on a plate using local maxima detection.
    Returns only spots with strong signal — doesn't force a specific count."""
    pcx, pcy, pr = plate["cx"], plate["cy"], plate["r"]
    rim_limit = pr * 0.78
    h, w = score.shape

    # Use non-maximum suppression to find peaks
    ks = min_distance if min_distance % 2 == 1 else min_distance + 1
    dilated = cv2.dilate(score, np.ones((ks, ks)))
    local_max = (score == dilated) & (score > 0.5)

    ys, xs = np.where(local_max)
    raw = []
    for x, y in zip(xs, ys):
        if math.sqrt((x - pcx) ** 2 + (y - pcy) ** 2) < rim_limit:
            raw.append((int(x), int(y), float(score[y, x])))
    raw.sort(key=lambda c: -c[2])

    if not raw:
        return []

    # Adaptive threshold: take spots that are at least 30% of the strongest signal
    top_score = raw[0][2]
    threshold = top_score * 0.30

    selected = []
    for c in raw:
        if c[2] < threshold:
            break
        too_close = any(math.sqrt((c[0] - s[0]) ** 2 + (c[1] - s[1]) ** 2) < min_distance
                        for s in selected)
        if not too_close:
            selected.append(c)
        if len(selected) >= max_count:
            break

    # Snap each to its peak
    final = []
    for x, y, sc in selected:
        sx, sy = snap_to_peak(score, x, y, 30)
        final.append((sx, sy, sc))

    return final


def estimate_spot_radius(diff, x, y, default=30):
    """Estimate the radius of a spot at (x, y) from the diff image."""
    h, w = diff.shape
    profile = []
    for r in range(1, 60):
        na = max(12, int(2 * math.pi * r))
        angles = np.linspace(0, 2 * math.pi, na, endpoint=False)
        pxs = np.clip((x + r * np.cos(angles)).astype(int), 0, w - 1)
        pys = np.clip((y + r * np.sin(angles)).astype(int), 0, h - 1)
        profile.append(abs(diff[pys, pxs].mean()))
    if not profile:
        return default
    peak = max(profile[:30])
    if peak < 1.0:
        return default
    threshold = peak * 0.3
    for r_idx, val in enumerate(profile):
        if r_idx > 5 and val < threshold:
            return max(15, r_idx + 1)
    return default


def find_spots_on_plate(gray, rgb, plate, sensitivity=5.0):
    """Find clearly visible spots on a plate. Returns only spots with strong signal."""
    pr = plate["r"]
    score, diff, median = build_spot_score_map(gray, rgb, plate)

    clear = find_clear_spots(score, plate,
                             min_distance=int(pr * 0.22),
                             max_count=6)

    if not clear:
        return [], diff, median

    # Estimate radius for each spot
    radii = []
    for x, y, sc in clear:
        radii.append(estimate_spot_radius(diff, x, y))

    # Enforce size similarity: all drops are the same volume
    if radii:
        median_r = int(np.median(radii))
        spots = []
        for (x, y, sc), r in zip(clear, radii):
            if 0.5 * median_r <= r <= 1.5 * median_r:
                spots.append((x, y, r))
            else:
                spots.append((x, y, median_r))
    else:
        spots = [(x, y, 30) for x, y, sc in clear]

    return spots, diff, median


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
        plate_spots, diff, median = find_spots_on_plate(gray, rgb, plate, sensitivity)
        diff_cache[pi] = diff
        median_cache[pi] = median
        for x, y, radius in plate_spots:
            spots.append({"x": x, "y": y, "radius": radius,
                          "plate_idx": pi, "label": ""})

    # Assign row/col by clustering Y positions (row) and X positions (col)
    for pi in range(len(plates)):
        plate_spots = [(i, s) for i, s in enumerate(spots) if s["plate_idx"] == pi]
        if not plate_spots:
            continue
        # Sort by Y to assign rows
        by_y = sorted(plate_spots, key=lambda t: t[1]["y"])
        # Simple row clustering: split into groups where Y gap > 30% of plate radius
        row_gap = plates[pi]["r"] * 0.3
        row_id = 0
        by_y[0][1]["row"] = 0
        for j in range(1, len(by_y)):
            if by_y[j][1]["y"] - by_y[j-1][1]["y"] > row_gap:
                row_id += 1
            by_y[j][1]["row"] = row_id
        # Within each row, sort by X to assign columns
        for row in range(row_id + 1):
            row_spots = sorted([(i, s) for i, s in plate_spots if s["row"] == row],
                               key=lambda t: t[1]["x"])
            for col, (i, s) in enumerate(row_spots):
                s["col"] = col

    spots.sort(key=lambda s: (s["plate_idx"], s["row"], s["col"]))
    return plates, spots, diff_cache, median_cache


# ─── Drawing ────────────────────────────────────────────────────────────────


def get_font(size=20):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    # Pillow 10+ supports load_default with size parameter
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_outlined_text(draw, pos, text, fill, font, outline_color="#000000", width=4):
    """Draw text with an outline for readability."""
    x, y = pos
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, fill=outline_color, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def draw_annotated_image(rgb, plates, spots, selected_idx=None):
    """Draw plates, spots, zones, and labels on the image."""
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    # Scale font to image size — 4% of image height for visibility
    font_size = max(48, int(rgb.shape[0] * 0.04))
    small_size = max(36, int(font_size * 0.75))
    font = get_font(font_size)
    small_font = get_font(small_size)

    for plate in plates:
        cx, cy, r = plate["cx"], plate["cy"], plate["r"]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#00C800", width=4)

    for i, s in enumerate(spots):
        x, y, r = s["x"], s["y"], s["radius"]
        zone_r = s.get("zone_radius", r)

        spot_color = "#FFD700" if i == selected_idx else "#66FF66"
        spot_width = 5 if i == selected_idx else 3
        draw.ellipse([x - r, y - r, x + r, y + r], outline=spot_color, width=spot_width)

        if zone_r > r + 2:
            zone_color = "#FF8800" if i == selected_idx else "#FF4444"
            draw.ellipse([x - zone_r, y - zone_r, x + zone_r, y + zone_r],
                         outline=zone_color, width=3)

        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(50, 100, 255))

        # Labels with outline
        tx = x + max(r, zone_r) + 10
        num_text = f"#{i+1}"
        label = s.get("label", "")
        grid_label = f"R{s.get('row', 0)+1}C{s.get('col', 0)+1}"
        text_color = "#FFFF00" if i != selected_idx else "#FFD700"
        draw_outlined_text(draw, (tx, y - font_size - 2), num_text,
                           fill=text_color, font=font)
        draw_outlined_text(draw, (tx, y + 2), grid_label,
                           fill="#00FF88", font=small_font)
        if label:
            draw_outlined_text(draw, (tx, y + small_size + 4), label,
                               fill="#FFFFFF", font=small_font)

    return img


def assign_plate(plates, x, y):
    for i, p in enumerate(plates):
        d = math.sqrt((p["cx"] - x) ** 2 + (p["cy"] - y) ** 2)
        if d < p["r"] - 20:
            return i
    return 0


def remeasure_spot(spots, idx, plates):
    s = spots[idx]
    pi = s["plate_idx"]
    diff_cache = st.session_state.get("_diff", {})
    median_cache = st.session_state.get("_median", {})
    if pi in diff_cache:
        m = measure_spot(median_cache[pi], diff_cache[pi], s, plates[pi])
        s.update(m)


# ─── Streamlit App ──────────────────────────────────────────────────────────


st.set_page_config(page_title="Zone of Inhibition Analyzer", layout="wide")
st.title("Zone of Inhibition Analyzer")

uploaded_file = st.file_uploader(
    "Drag and drop a plate image (TIF, PNG, JPG)",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.markdown(
        "Upload an image of agar plates. The tool will detect plates and find "
        "spots in a **2-row x 3-column grid** on each plate, then you can "
        "adjust positions, label them, and generate measurements."
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

    st.divider()
    st.header("Instructions")
    st.markdown(
        "1. Upload an image\n"
        "2. Adjust sensitivity if needed\n"
        "3. Select spots to adjust or delete\n"
        "4. Label rows (sample) and columns (concentration)\n"
        "5. Download CSV and graphs"
    )

# ─── Load image ─────────────────────────────────────────────────────────────

if st.session_state.get("_file_id") != file_id:
    rgb, gray = load_and_convert(uploaded_file)
    st.session_state._file_id = file_id
    st.session_state._rgb = rgb
    st.session_state._gray = gray
    st.session_state.pop("_spots", None)

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
        st.session_state._plates = plates
        st.session_state._spots = spots
        st.session_state._diff = diff_cache
        st.session_state._median = median_cache

plates = st.session_state._plates
spots = st.session_state._spots

# ─── Annotated Image ───────────────────────────────────────────────────────

selected = st.session_state.get("_selected_spot", None)
ann_img = draw_annotated_image(rgb, plates, spots, selected_idx=selected)

st.subheader(f"{len(plates)} plate(s), {len(spots)} spot(s)")
st.image(ann_img, use_container_width=True)

# ─── Spot Editing ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Edit Spots")

col_select, col_actions = st.columns([3, 2])

with col_select:
    spot_options = [f"#{i+1} — Plate {s['plate_idx']+1}, "
                    f"R{s.get('row',0)+1}C{s.get('col',0)+1}"
                    + (f" ({s.get('label','')})" if s.get('label') else "")
                    for i, s in enumerate(spots)]
    if spots:
        sel_idx = st.selectbox(
            "Select a spot to edit",
            range(len(spots)),
            format_func=lambda i: spot_options[i],
            index=selected if selected is not None and selected < len(spots) else 0,
            key="_spot_selector"
        )
        if sel_idx != st.session_state.get("_selected_spot"):
            st.session_state._selected_spot = sel_idx
            st.rerun()

with col_actions:
    if spots:
        ac1, ac2 = st.columns(2)
        with ac1:
            if st.button("Delete this spot", type="secondary"):
                spots.pop(sel_idx)
                st.session_state._spots = spots
                st.session_state._selected_spot = max(0, sel_idx - 1) if spots else None
                st.rerun()
        with ac2:
            if st.button("Add spot at plate center"):
                if plates:
                    p = plates[0]
                    new_spot = {"x": p["cx"], "y": p["cy"], "radius": 35,
                                "plate_idx": 0, "label": "", "row": 0, "col": 0}
                    remeasure_spot([new_spot], 0, plates)
                    spots.append(new_spot)
                    spots.sort(key=lambda s: (s["plate_idx"], s["row"], s["col"]))
                    st.session_state._spots = spots
                    st.session_state._selected_spot = spots.index(new_spot)
                    st.rerun()

# Position adjustment for selected spot
if spots and sel_idx is not None and sel_idx < len(spots):
    s = spots[sel_idx]
    st.markdown(f"**Spot #{sel_idx+1}** — Plate {s['plate_idx']+1}, "
                f"Row {s.get('row',0)+1}, Col {s.get('col',0)+1}")

    p = plates[s["plate_idx"]]
    x_min, x_max = p["cx"] - p["r"], p["cx"] + p["r"]
    y_min, y_max = p["cy"] - p["r"], p["cy"] + p["r"]

    c1, c2 = st.columns(2)
    with c1:
        new_x = st.slider("X position", min_value=x_min, max_value=x_max,
                           value=s["x"], key=f"x_{sel_idx}")
    with c2:
        new_y = st.slider("Y position", min_value=y_min, max_value=y_max,
                           value=s["y"], key=f"y_{sel_idx}")

    if new_x != s["x"] or new_y != s["y"]:
        s["x"] = new_x
        s["y"] = new_y
        remeasure_spot(spots, sel_idx, plates)
        st.session_state._spots = spots
        st.rerun()

# ─── Spot Labeling (grid-based) ────────────────────────────────────────────

st.divider()
st.subheader("Label Spots")
st.markdown(
    "Spots in the **same row** share a sample type. "
    "Spots in the **same column** share a concentration. "
    "Label rows and columns below — each spot auto-labels from its row + column."
)

if "_row_labels" not in st.session_state:
    st.session_state._row_labels = {}
if "_col_labels" not in st.session_state:
    st.session_state._col_labels = {}

row_labels = st.session_state._row_labels
col_labels = st.session_state._col_labels

for plate_idx in range(len(plates)):
    plate_spots = [s for s in spots if s["plate_idx"] == plate_idx]
    if not plate_spots:
        continue

    p_rows = sorted(set(s.get("row", 0) for s in plate_spots))
    p_cols = sorted(set(s.get("col", 0) for s in plate_spots))

    st.markdown(f"**Plate {plate_idx + 1}** — {len(plate_spots)} spots "
                f"({len(p_rows)} rows x {len(p_cols)} columns)")

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
        "Grid": f"R{s.get('row',0)+1}C{s.get('col',0)+1}",
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
st.dataframe(df[["Spot", "Plate", "Grid", "Label", "Spot Radius (px)", "Drop Intensity",
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
    img_buf = io.BytesIO()
    ann_img.save(img_buf, format="PNG")
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
