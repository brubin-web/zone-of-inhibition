"""
Zone of Inhibition Measurement Tool — Streamlit Web App
Upload plate images, auto-detect spots, label them, get measurements + graphs.
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
    # Blue ink (pen labels)
    pen_mask[(b_ch - r_ch > 12) & (mask > 0)] = 255
    # Red ink (pen labels)
    pen_mask[(r_ch - g_ch > 30) & (r_ch > 150) & (mask > 0)] = 255
    # Highly saturated colors (marker/pen) — use HSV saturation
    # Only flag very saturated pixels (>130/255) to avoid removing real plate features
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pen_mask[(hsv[:, :, 1] > 130) & (mask > 0)] = 255
    pen_mask = cv2.dilate(pen_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
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

    # Also compute HSV hue-based difference for better spot detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    hue_blur = cv2.GaussianBlur(hue, (21, 21), 5)
    hue_bg = cv2.GaussianBlur(hue_blur, (201, 201), 60)
    hue_diff = hue_blur - hue_bg
    hue_diff[mask == 0] = 0
    sat_blur = cv2.GaussianBlur(sat, (21, 21), 5)
    sat_bg = cv2.GaussianBlur(sat_blur, (201, 201), 60)
    sat_diff = sat_blur - sat_bg
    sat_diff[mask == 0] = 0

    # Rim exclusion: reject spots within 15% of plate edge
    rim_limit = pr * 0.85

    all_contours = []

    def _find_contours_from(diff_img, threshold_val, min_strength):
        """Extract circular contours from a difference image."""
        found = []
        for sign in [1, -1]:
            directed = sign * diff_img
            thresh_img = np.zeros(gray.shape, dtype=np.uint8)
            thresh_img[directed > threshold_val] = 255
            thresh_img[mask == 0] = 0
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, k5)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 300 or area > 100000:
                    continue
                (x, y), radius = cv2.minEnclosingCircle(c)
                circ = area / (math.pi * radius ** 2) if radius > 0 else 0
                if circ < 0.4 or radius < 10:
                    continue
                # Reject spots too close to plate rim
                dist_from_center = math.sqrt((x - pcx) ** 2 + (y - pcy) ** 2)
                if dist_from_center > rim_limit:
                    continue
                contour_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [c], -1, 255, -1)
                mean_val = abs(diff_img[contour_mask > 0].mean())
                if mean_val > min_strength:
                    found.append((int(x), int(y), int(radius), circ, mean_val))
        return found

    # Grayscale detection at two sensitivity levels
    for s_mult in [1.0, 0.7]:
        thresh = sensitivity * s_mult
        all_contours.extend(_find_contours_from(diff, thresh, thresh * 0.6))

    # HSV hue detection — spots often have a distinct hue shift
    all_contours.extend(_find_contours_from(hue_diff, 3.0, 1.5))

    # HSV saturation detection — spots may differ in saturation
    all_contours.extend(_find_contours_from(sat_diff, 8.0, 4.0))

    # Merge duplicates (same spot found by multiple methods)
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
    if len(spots) < 3:
        return spots
    radii = sorted([s["radius"] for s in spots])

    # Find the largest relative gap — if it's really clear (>80% jump),
    # use it to separate artifacts from real spots
    best_gap_rel = 0
    best_cutoff = 0
    for i in range(1, len(radii)):
        gap = radii[i] - radii[i - 1]
        rel_gap = gap / max(radii[i - 1], 1)
        remaining = len(radii) - i
        if rel_gap > best_gap_rel and gap >= 5 and remaining >= 3:
            best_gap_rel = rel_gap
            best_cutoff = (radii[i - 1] + radii[i]) / 2

    # Only use gap filter if the gap is very obvious (>80% size jump)
    if best_cutoff > 0 and best_gap_rel > 0.8:
        filtered = [s for s in spots if s["radius"] >= best_cutoff]
        if len(filtered) >= 3:
            return filtered

    # Fallback: keep spots with radius >= 35% of the 80th-percentile spot
    # (using percentile instead of max to be robust against outliers)
    ref_r = radii[min(int(len(radii) * 0.8), len(radii) - 1)]
    min_r = ref_r * 0.35
    filtered = [s for s in spots if s["radius"] >= min_r]
    if len(filtered) >= 3:
        return filtered
    return spots


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


def infer_grid(plate_spots, plate):
    """Given detected spots on one plate, infer the grid structure.
    Returns (row_positions, col_positions) — only rows/columns with
    at least 2 spots in them (to avoid false grid lines)."""
    if len(plate_spots) < 4:
        return [], []

    pr = plate["r"]
    gap_threshold = pr * 0.12

    ys = [s["y"] for s in plate_spots]
    xs = [s["x"] for s in plate_spots]

    def _cluster(vals):
        sorted_vals = sorted(vals)
        groups = [[sorted_vals[0]]]
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] - sorted_vals[i - 1] > gap_threshold:
                groups.append([])
            groups[-1].append(sorted_vals[i])
        # Only keep clusters with >= 2 members (real grid lines have multiple spots)
        return [np.mean(g) for g in groups if len(g) >= 2]

    return _cluster(ys), _cluster(xs)


def find_spot_at_position(diff, median, ex, ey, search_radius, plate, min_signal=3.0):
    """Search for a spot at an expected grid position.
    Returns (x, y, radius) or None. Requires meaningful signal."""
    pcx, pcy, pr = plate["cx"], plate["cy"], plate["r"]
    h, w = diff.shape

    if math.sqrt((ex - pcx) ** 2 + (ey - pcy) ** 2) > pr * 0.82:
        return None

    sr = search_radius
    x1, x2 = max(0, ex - sr), min(w, ex + sr)
    y1, y2 = max(0, ey - sr), min(h, ey + sr)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    # Find the peak signal in the search area
    abs_diff_region = np.abs(diff[y1:y2, x1:x2])
    max_loc = np.unravel_index(abs_diff_region.argmax(), abs_diff_region.shape)
    best_y = y1 + max_loc[0]
    best_x = x1 + max_loc[1]

    # Measure radial profile from the peak to check if it looks like a real spot
    profile = []
    for r in range(1, min(sr, 50)):
        na = max(12, int(2 * math.pi * r))
        angles = np.linspace(0, 2 * math.pi, na, endpoint=False)
        pxs = np.clip((best_x + r * np.cos(angles)).astype(int), 0, w - 1)
        pys = np.clip((best_y + r * np.sin(angles)).astype(int), 0, h - 1)
        profile.append(abs(diff[pys, pxs].mean()))

    if not profile:
        return None

    peak = max(profile[:min(25, len(profile))])
    if peak < min_signal:
        return None

    # Find radius where signal drops to 30% of peak
    threshold = peak * 0.3
    radius = 20
    for r_idx, val in enumerate(profile):
        if r_idx > 8 and val < threshold:
            radius = r_idx + 1
            break

    radius = max(15, min(radius, 55))
    return (best_x, best_y, radius)


def deduplicate_spots(spots, min_dist=40):
    """Remove duplicate spots that are too close together, keeping the larger one."""
    if len(spots) < 2:
        return spots
    spots_sorted = sorted(spots, key=lambda s: -s["radius"])
    kept = []
    for s in spots_sorted:
        is_dup = False
        for k in kept:
            d = math.sqrt((s["x"] - k["x"]) ** 2 + (s["y"] - k["y"]) ** 2)
            if d < min_dist:
                is_dup = True
                break
        if not is_dup:
            kept.append(s)
    return kept


def grid_refine_spots(plate_spots, diff, median, plate):
    """Use grid structure to find missing spots and remove non-grid detections."""
    if len(plate_spots) < 4:
        return plate_spots

    row_positions, col_positions = infer_grid(plate_spots, plate)
    if len(row_positions) < 2 or len(col_positions) < 2:
        return plate_spots

    pr = plate["r"]
    pcx, pcy = plate["cx"], plate["cy"]
    match_tolerance = pr * 0.10
    typical_radius = int(np.median([s["radius"] for s in plate_spots]))

    # Build expected grid positions (only inside plate)
    expected = []
    for ry in row_positions:
        for cx in col_positions:
            if math.sqrt((cx - pcx) ** 2 + (ry - pcy) ** 2) < pr * 0.80:
                expected.append((int(cx), int(ry)))

    # Match existing spots to grid positions
    matched_spots = []
    used_expected = set()
    for s in plate_spots:
        best_dist = float('inf')
        best_ei = -1
        for ei, (ex, ey) in enumerate(expected):
            d = math.sqrt((s["x"] - ex) ** 2 + (s["y"] - ey) ** 2)
            if d < best_dist:
                best_dist = d
                best_ei = ei
        # Keep spots that match a grid position
        if best_dist < match_tolerance:
            matched_spots.append(s)
            used_expected.add(best_ei)

    # If grid matching removed too many spots, grid inference was wrong — abort
    if len(matched_spots) < len(plate_spots) * 0.5:
        return plate_spots

    # Search for missing spots at unmatched grid positions
    search_r = max(40, int(pr * 0.07))
    pi = plate_spots[0]["plate_idx"]
    for ei, (ex, ey) in enumerate(expected):
        if ei in used_expected:
            continue
        result = find_spot_at_position(diff, median, ex, ey, search_r, plate)
        if result:
            bx, by, br = result
            matched_spots.append({
                "x": bx, "y": by,
                "radius": max(br, int(typical_radius * 0.6)),
                "plate_idx": pi, "label": ""
            })

    return deduplicate_spots(matched_spots)


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

    # Grid-based refinement: per plate, infer grid and fill in missing spots
    refined = []
    for pi, plate in enumerate(plates):
        plate_spots = [s for s in spots if s["plate_idx"] == pi]
        if len(plate_spots) >= 3:
            plate_spots = grid_refine_spots(
                plate_spots, diff_cache[pi], median_cache[pi], plate)
        refined.extend(plate_spots)

    refined.sort(key=lambda s: (s["plate_idx"], s["y"] // 50, s["x"]))
    return plates, refined, diff_cache, median_cache


# ─── Drawing ────────────────────────────────────────────────────────────────


def get_font(size=20):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_annotated_image(rgb, plates, spots, selected_idx=None):
    """Draw plates, spots, zones, and labels on the image."""
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    # Scale font to image size — aim for ~2.5% of image height
    font_size = max(32, int(rgb.shape[0] * 0.025))
    small_size = max(26, int(font_size * 0.8))
    font = get_font(font_size)
    small_font = get_font(small_size)

    for plate in plates:
        cx, cy, r = plate["cx"], plate["cy"], plate["r"]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#00C800", width=4)

    for i, s in enumerate(spots):
        x, y, r = s["x"], s["y"], s["radius"]
        zone_r = s.get("zone_radius", r)

        # Spot circle
        spot_color = "#FFD700" if i == selected_idx else "#66FF66"
        spot_width = 5 if i == selected_idx else 3
        draw.ellipse([x - r, y - r, x + r, y + r], outline=spot_color, width=spot_width)

        # Zone boundary
        if zone_r > r + 2:
            zone_color = "#FF8800" if i == selected_idx else "#FF4444"
            draw.ellipse([x - zone_r, y - zone_r, x + zone_r, y + zone_r],
                         outline=zone_color, width=3)

        # Center dot
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(50, 100, 255))

        # Label — use text with dark outline for readability (no giant boxes)
        tx = x + max(r, zone_r) + 10
        num_text = f"#{i+1}"
        label = s.get("label", "")
        text_color = "#FFFF00" if i != selected_idx else "#FFD700"
        # Draw text with outline for contrast
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0:
                    continue
                draw.text((tx + dx, y - font_size // 2 + dy), num_text,
                          fill="#000000", font=font)
                if label:
                    draw.text((tx + dx, y + font_size // 2 + 2 + dy), label,
                              fill="#000000", font=small_font)
        draw.text((tx, y - font_size // 2), num_text, fill=text_color, font=font)
        if label:
            draw.text((tx, y + font_size // 2 + 2), label, fill="#FFFFFF", font=small_font)

    return img


def assign_plate(plates, x, y):
    for i, p in enumerate(plates):
        d = math.sqrt((p["cx"] - x) ** 2 + (p["cy"] - y) ** 2)
        if d < p["r"] - 20:
            return i
    return 0


def cluster_1d(values, min_gap_frac=0.3):
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    clusters = [[indexed[0]]]
    for i in range(1, len(indexed)):
        gap = indexed[i][1] - indexed[i - 1][1]
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
    for plate_idx in range(len(plates)):
        plate = plates[plate_idx]
        plate_spots = [(i, s) for i, s in enumerate(spots) if s["plate_idx"] == plate_idx]
        if not plate_spots:
            continue
        pr = plate["r"]
        gap_threshold = pr * 0.08

        ys = [s["y"] for _, s in plate_spots]
        y_clusters = cluster_1d(ys, gap_threshold)
        for row_idx, (_, indices) in enumerate(y_clusters):
            for idx in indices:
                spots[plate_spots[idx][0]]["row"] = row_idx

        xs = [s["x"] for _, s in plate_spots]
        x_clusters = cluster_1d(xs, gap_threshold)
        for col_idx, (_, indices) in enumerate(x_clusters):
            for idx in indices:
                spots[plate_spots[idx][0]]["col"] = col_idx

    return spots


def remeasure_spot(spots, idx, plates):
    """Re-measure a single spot after it's been moved."""
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
        "Upload an image of agar plates. The tool will detect plates and spots, "
        "then you can adjust positions, label them, and generate measurements."
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
        spots = detect_grid(spots, plates)
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
                # Re-detect grid after deletion
                spots = detect_grid(spots, plates)
                st.session_state._spots = spots
                st.rerun()
        with ac2:
            if st.button("Add spot at plate center"):
                if plates:
                    p = plates[0]
                    new_spot = {"x": p["cx"], "y": p["cy"], "radius": 35,
                                "plate_idx": 0, "label": ""}
                    remeasure_spot([new_spot], 0, plates)
                    spots.append(new_spot)
                    spots.sort(key=lambda s: (s["plate_idx"], s["y"] // 50, s["x"]))
                    spots = detect_grid(spots, plates)
                    st.session_state._spots = spots
                    st.session_state._selected_spot = spots.index(new_spot)
                    st.rerun()

# Position adjustment for selected spot
if spots and sel_idx is not None and sel_idx < len(spots):
    s = spots[sel_idx]
    st.markdown(f"**Spot #{sel_idx+1}** — Plate {s['plate_idx']+1}")

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
        spots = detect_grid(spots, plates)
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
