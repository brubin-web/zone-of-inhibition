#!/usr/bin/env python3
"""
Zone of Inhibition Measurement Tool — CLI version
Detects petri dishes, spots, and zones of inhibition from agar plate images.
Outputs annotated image and CSV measurements.
"""

import numpy as np
import cv2
from PIL import Image
import csv
import math
import sys
from pathlib import Path


def detect_plates(gray, rgb):
    """Detect circular petri dishes."""
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
    """Remove colored pen marks via inpainting."""
    pen_mask = np.zeros(gray.shape, dtype=np.uint8)
    r_ch = rgb[:, :, 0].astype(int)
    g_ch = rgb[:, :, 1].astype(int)
    b_ch = rgb[:, :, 2].astype(int)
    pen_mask[(b_ch - r_ch > 15) & (mask > 0)] = 255
    pen_mask[(r_ch - g_ch > 40) & (mask > 0)] = 255
    pen_mask = cv2.dilate(pen_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    clean = cv2.inpaint(gray, pen_mask, 25, cv2.INPAINT_TELEA)
    return clean


def find_spots_in_plate(gray, rgb, pcx, pcy, pr, sensitivity=5.0):
    """Find spots using median filter + background subtraction."""
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
            raw_diff = diff[int(y), int(x)]
            direction = "lighter" if raw_diff > 0 else "darker"
            if mean_diff > sensitivity * 0.6:
                all_contours.append((int(x), int(y), int(radius), area, circ,
                                     mean_diff, direction))

    merged = []
    used = set()
    all_contours.sort(key=lambda s: -(s[4] * s[5]))
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


def measure_spot(gray_clean, diff, cx, cy, radius, pcx, pcy, pr):
    """Measure spot and zone using radial profile analysis."""
    max_r = min(150, int(pr * 0.8))
    dist_to_edge = math.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
    max_r = min(max_r, int(pr - dist_to_edge - 20))
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

    drop_radius = 3
    if center_intensity < prof_g[min(radius, len(prof_g) - 1)] - 10:
        half_val = (center_intensity + prof_g[min(radius, len(prof_g) - 1)]) / 2
        for r in range(len(prof_g)):
            if prof_g[r] > half_val:
                drop_radius = r + 1
                break

    drop_area = math.pi * drop_radius ** 2
    zone_area = math.pi * zone_radius ** 2
    zone_diameter = zone_radius * 2

    drop_pixels = []
    for r in range(1, max(2, drop_radius + 1)):
        na = max(12, int(2 * math.pi * r))
        angles = np.linspace(0, 2 * math.pi, na, endpoint=False)
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, gray_clean.shape[1] - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, gray_clean.shape[0] - 1)
        drop_pixels.extend(gray_clean[ys, xs].tolist())
    drop_mean_intensity = float(np.mean(drop_pixels)) if drop_pixels else center_intensity

    return {
        'drop_radius': drop_radius,
        'drop_area': drop_area,
        'drop_mean_intensity': drop_mean_intensity,
        'zone_radius': zone_radius,
        'zone_diameter': zone_diameter,
        'zone_area': zone_area,
        'has_zone': zone_radius > drop_radius + 5,
    }


def main():
    script_dir = Path(__file__).parent
    input_file = script_dir / "ds_18hr006.tif"

    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    print(f"Processing: {input_file}")

    pil_img = Image.open(str(input_file)).convert("RGB")
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(f"Image: {gray.shape[1]} x {gray.shape[0]}")

    plates = detect_plates(gray, rgb)
    print(f"Detected {len(plates)} plates")

    annotated = rgb.copy()
    all_rows = []
    spot_num = 0

    for pi, (pcx, pcy, pr) in enumerate(plates):
        cv2.circle(annotated, (pcx, pcy), pr, (0, 200, 0), 3)
        print(f"\nPlate {pi + 1}: center=({pcx},{pcy}), r={pr}")

        spots, diff, median = find_spots_in_plate(gray, rgb, pcx, pcy, pr)
        print(f"  Found {len(spots)} spots")

        for cx, cy, radius, area, circ, mean_diff, direction in spots:
            spot_num += 1
            m = measure_spot(median, diff, cx, cy, radius, pcx, pcy, pr)
            drop_circ = 2 * math.pi * m['drop_radius']
            zoi_area = m['zone_diameter'] / m['drop_area'] if m['drop_area'] > 0 else 0
            zoi_circ = m['zone_diameter'] / drop_circ if drop_circ > 0 else 0

            print(f"    #{spot_num} ({cx},{cy}) r={radius} zone_d={m['zone_diameter']} [{direction}]")

            # Draw annotations
            cv2.circle(annotated, (cx, cy), radius, (100, 255, 100), 2)
            if m['has_zone']:
                cv2.circle(annotated, (cx, cy), m['zone_radius'], (255, 60, 60), 2)
            cv2.circle(annotated, (cx, cy), 4, (50, 100, 255), -1)
            cv2.putText(annotated, f"#{spot_num}", (cx + max(radius, m['zone_radius']) + 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            all_rows.append({
                'Plate': pi + 1, 'Spot': spot_num,
                'Center_X': cx, 'Center_Y': cy,
                'Spot_Radius_px': radius, 'Direction': direction,
                'Drop_Radius_px': m['drop_radius'],
                'Drop_Area_px2': round(m['drop_area'], 1),
                'Drop_Intensity': round(m['drop_mean_intensity'], 1),
                'Zone_Diameter_px': m['zone_diameter'],
                'Zone_Radius_px': m['zone_radius'],
                'Zone_Area_px2': round(m['zone_area'], 1),
                'Has_Zone': m['has_zone'],
                'ZOI_per_Drop_Area': round(zoi_area, 4),
                'ZOI_per_Drop_Circumference': round(zoi_circ, 4),
            })

    # Save outputs
    stem = input_file.stem
    ann_path = str(script_dir / f"annotated_{stem}.png")
    csv_path = str(script_dir / f"measurements_{stem}.csv")

    cv2.imwrite(ann_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"\nAnnotated image: {ann_path}")

    if all_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"CSV: {csv_path}")

    print(f"\nTotal: {spot_num} spots across {len(plates)} plates")


if __name__ == "__main__":
    main()
