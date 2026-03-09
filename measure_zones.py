#!/usr/bin/env python3
"""
Zone of Inhibition Measurement Tool
Detects petri dishes, drops, and zones of inhibition from agar plate images.
Outputs annotated image and CSV measurements.
"""

import numpy as np
import cv2
from PIL import Image
import csv
import os
import sys
from pathlib import Path


def load_image(path):
    """Load TIF image and convert to grayscale."""
    pil_img = Image.open(path)
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def detect_plates(gray):
    """Detect petri dishes using threshold + erosion + contour approach."""
    blurred = cv2.GaussianBlur(gray, (11, 11), 5)
    _, thresh = cv2.threshold(blurred, 215, 255, cv2.THRESH_BINARY_INV)

    # Erode to separate touching plates
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    eroded = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200000:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        circularity = area / (np.pi * r * r)
        if circularity > 0.7:
            plates.append((int(x), int(y), int(r)))

    # Sort plates: top-left first, then top-right, then bottom
    plates.sort(key=lambda p: (p[1] // 300, p[0]))
    return plates


def radial_profile(gray, cx, cy, max_r, n_angles=None):
    """Compute average intensity at each radius from (cx, cy)."""
    profile = np.zeros(max_r)
    for r in range(1, max_r + 1):
        na = n_angles if n_angles else max(24, int(2 * np.pi * r))
        angles = np.linspace(0, 2 * np.pi, na, endpoint=False)
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        xs = np.clip(xs.astype(int), 0, gray.shape[1] - 1)
        ys = np.clip(ys.astype(int), 0, gray.shape[0] - 1)
        profile[r - 1] = gray[ys, xs].mean()
    return profile


def smooth(arr, window=5):
    """Simple moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def find_drops_in_plate(gray, plate_cx, plate_cy, plate_r):
    """Find drop/spot locations within a plate using multiple detection methods."""
    # Create plate mask (avoid rim)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (plate_cx, plate_cy), plate_r - 40, 255, -1)

    # Local background
    bg = cv2.GaussianBlur(gray, (151, 151), 50)
    diff = gray.astype(np.float32) - bg.astype(np.float32)

    # Method 1: Find dark blobs (drops are darker than lawn)
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
        circ = area / (np.pi * radius * radius) if radius > 0 else 0
        if circ < 0.3:
            continue

        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(x), int(y)

        # Check it's actually dark
        half = max(3, int(radius * 0.5))
        y1, y2 = max(0, cy - half), min(gray.shape[0], cy + half)
        x1, x2 = max(0, cx - half), min(gray.shape[1], cx + half)
        mean_intensity = gray[y1:y2, x1:x2].mean()

        if mean_intensity < 150:
            candidates.append((cx, cy, mean_intensity, area))

    # Method 2: Find bright circular zones and look for dark centers
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
        circ = area / (np.pi * radius * radius) if radius > 0 else 0
        if circ < 0.5 or radius < 15 or radius > 120:
            continue

        cx, cy = int(x), int(y)
        center_val = gray[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5].mean()

        # If center is dark, it's a drop inside a zone
        if center_val < 170:
            # Check not already found
            already = any(abs(cx - c2[0]) < 20 and abs(cy - c2[1]) < 20 for c2 in candidates)
            if not already:
                candidates.append((cx, cy, center_val, -1))

    # Merge nearby candidates (within 25px)
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
        # Pick the darkest center in the group
        best = min(group, key=lambda g: g[2])
        merged.append(best)

    return merged


def measure_drop_and_zone(gray, cx, cy, plate_cx, plate_cy, plate_r):
    """Measure drop radius, zone diameter using radial profile analysis."""
    # Compute radial profile
    max_r = min(150, int(plate_r * 0.8))
    dist_to_edge = np.sqrt((cx - plate_cx)**2 + (cy - plate_cy)**2)
    max_r = min(max_r, int(plate_r - dist_to_edge - 20))
    max_r = max(max_r, 40)

    prof = radial_profile(gray, cx, cy, max_r)
    prof_smooth = smooth(prof, window=5)
    deriv = np.diff(prof_smooth)

    # --- Find drop edge ---
    # The drop edge is where intensity rises steeply from the dark drop
    # Look for the steepest positive derivative in the first 30 pixels
    search_end = min(30, len(deriv))
    if search_end < 3:
        return None

    drop_deriv = deriv[:search_end]
    drop_edge_r = np.argmax(drop_deriv) + 1

    # Refine: find where intensity reaches halfway between drop center and local lawn
    center_val = prof_smooth[0]
    # Estimate lawn: look at profile values beyond a reasonable drop radius
    if drop_edge_r + 10 < len(prof_smooth):
        near_lawn = prof_smooth[drop_edge_r + 5:drop_edge_r + 15].mean()
    else:
        near_lawn = prof_smooth[-10:].mean()
    half_val = (center_val + near_lawn) / 2

    # Find where profile crosses half_val
    for r in range(len(prof_smooth)):
        if prof_smooth[r] > half_val:
            drop_edge_r = r + 1
            break

    drop_radius = drop_edge_r

    # --- Measure drop properties ---
    drop_area = np.pi * drop_radius**2
    # Mean intensity within the drop
    drop_pixels = []
    for r in range(1, max(2, drop_radius)):
        angles = np.linspace(0, 2 * np.pi, max(12, int(2 * np.pi * r)), endpoint=False)
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        xs = np.clip(xs.astype(int), 0, gray.shape[1] - 1)
        ys = np.clip(ys.astype(int), 0, gray.shape[0] - 1)
        drop_pixels.extend(gray[ys, xs].tolist())
    drop_mean_intensity = np.mean(drop_pixels) if drop_pixels else gray[cy, cx]

    # --- Find zone boundary ---
    # Look for the halo ring: a local maximum in the radial profile beyond the drop
    # The halo is where bacteria pile up at the inhibition boundary
    search_start = drop_edge_r + 3
    if search_start >= len(prof_smooth) - 5:
        # No room for zone detection
        return {
            'drop_radius': drop_radius,
            'drop_area': drop_area,
            'drop_mean_intensity': float(drop_mean_intensity),
            'zone_radius': drop_radius,
            'zone_diameter': drop_radius * 2,
            'has_zone': False
        }

    # Find the lawn level (far from drop)
    far_start = min(len(prof_smooth) - 10, max(60, search_start + 30))
    lawn_level = prof_smooth[far_start:].mean() if far_start < len(prof_smooth) else prof_smooth[-10:].mean()

    # Look for halo peak: local maximum above lawn level
    search_profile = prof_smooth[search_start:]
    if len(search_profile) < 10:
        return {
            'drop_radius': drop_radius,
            'drop_area': drop_area,
            'drop_mean_intensity': float(drop_mean_intensity),
            'zone_radius': drop_radius,
            'zone_diameter': drop_radius * 2,
            'has_zone': False
        }

    # Find peaks in the profile (above lawn level)
    search_deriv = np.diff(search_profile)
    peaks = []
    for i in range(1, len(search_deriv)):
        if search_deriv[i - 1] > 0 and search_deriv[i] <= 0:
            peak_r = search_start + i
            peak_val = prof_smooth[peak_r]
            if peak_val > lawn_level + 3:  # Must be notably above lawn
                peaks.append((peak_r, peak_val))

    # Also check if the profile has a sustained dip below lawn before a rise
    # (the zone interior is slightly lighter than lawn)
    zone_region = prof_smooth[search_start:search_start + 60] if search_start + 60 < len(prof_smooth) else prof_smooth[search_start:]
    min_in_zone = zone_region.min()
    zone_dip = lawn_level - min_in_zone

    has_zone = False
    zone_radius = drop_radius

    if peaks:
        # Use the first significant peak as the zone boundary (halo ring)
        best_peak = peaks[0]
        zone_radius = best_peak[0] + 1
        has_zone = True
    elif zone_dip > 5:
        # No clear halo, but there's a dip - find where intensity returns to lawn level
        for r in range(search_start, len(prof_smooth)):
            if prof_smooth[r] >= lawn_level - 2:
                zone_radius = r + 1
                has_zone = True
                break

    zone_diameter = zone_radius * 2
    zone_area = np.pi * zone_radius**2

    return {
        'drop_radius': drop_radius,
        'drop_area': drop_area,
        'drop_mean_intensity': float(drop_mean_intensity),
        'zone_radius': zone_radius,
        'zone_diameter': zone_diameter,
        'zone_area': zone_area,
        'has_zone': has_zone
    }


def create_annotated_image(rgb, plates, all_measurements, output_path):
    """Draw annotations on the image and save."""
    annotated = rgb.copy()

    # Draw plates (green circles)
    for cx, cy, r in plates:
        cv2.circle(annotated, (cx, cy), r, (0, 200, 0), 3)

    # Draw drops and zones
    for plate_idx, plate_measurements in enumerate(all_measurements):
        for m in plate_measurements:
            cx, cy = m['center_x'], m['center_y']

            # Blue dot at drop center
            cv2.circle(annotated, (cx, cy), 4, (0, 80, 255), -1)

            # Cyan circle at drop edge
            cv2.circle(annotated, (cx, cy), m['drop_radius'], (0, 220, 220), 2)

            if m['has_zone']:
                # Red circle at zone boundary
                cv2.circle(annotated, (cx, cy), m['zone_radius'], (255, 50, 50), 2)

                # Label with zone diameter
                label = f"ZD={m['zone_diameter']:.0f}"
                label_pos = (cx + m['zone_radius'] + 5, cy - 5)
                cv2.putText(annotated, label, label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)

            # Label drop intensity
            drop_label = f"I={m['drop_mean_intensity']:.0f}"
            cv2.putText(annotated, drop_label, (cx - 25, cy + m['drop_radius'] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)

    # Convert RGB to BGR for OpenCV saving
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, annotated_bgr)
    print(f"Annotated image saved to: {output_path}")


def write_csv(all_measurements, output_path):
    """Write measurements to CSV."""
    headers = [
        'Plate', 'Drop', 'Center_X', 'Center_Y',
        'Drop_Radius_px', 'Drop_Area_px2', 'Drop_Mean_Intensity',
        'Zone_Diameter_px', 'Zone_Radius_px', 'Zone_Area_px2',
        'Has_Zone',
        'ZOI_Diameter_per_Drop_Area', 'ZOI_Diameter_per_Drop_Circumference'
    ]

    rows = []
    for plate_idx, plate_measurements in enumerate(all_measurements):
        for drop_idx, m in enumerate(plate_measurements):
            drop_circumference = 2 * np.pi * m['drop_radius']
            zoi_per_area = m['zone_diameter'] / m['drop_area'] if m['drop_area'] > 0 else 0
            zoi_per_circ = m['zone_diameter'] / drop_circumference if drop_circumference > 0 else 0

            rows.append([
                plate_idx + 1,
                drop_idx + 1,
                m['center_x'],
                m['center_y'],
                f"{m['drop_radius']:.1f}",
                f"{m['drop_area']:.1f}",
                f"{m['drop_mean_intensity']:.1f}",
                f"{m['zone_diameter']:.1f}",
                f"{m['zone_radius']:.1f}",
                f"{m.get('zone_area', 0):.1f}",
                'Yes' if m['has_zone'] else 'No',
                f"{zoi_per_area:.4f}",
                f"{zoi_per_circ:.4f}"
            ])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"CSV saved to: {output_path}")


def print_summary(all_measurements):
    """Print a summary table to terminal."""
    print("\n" + "=" * 100)
    print("ZONE OF INHIBITION MEASUREMENTS")
    print("=" * 100)

    header = f"{'Plate':>5} {'Drop':>5} {'Center':>12} {'Drop r':>8} {'Drop Area':>10} " \
             f"{'Intensity':>10} {'Zone Diam':>10} {'Zone?':>6} {'ZOI/Area':>10} {'ZOI/Circ':>10}"
    print(header)
    print("-" * 100)

    for plate_idx, plate_measurements in enumerate(all_measurements):
        for drop_idx, m in enumerate(plate_measurements):
            drop_circ = 2 * np.pi * m['drop_radius']
            zoi_per_area = m['zone_diameter'] / m['drop_area'] if m['drop_area'] > 0 else 0
            zoi_per_circ = m['zone_diameter'] / drop_circ if drop_circ > 0 else 0

            row = f"{plate_idx + 1:>5} {drop_idx + 1:>5} " \
                  f"({m['center_x']:>4},{m['center_y']:>4}) " \
                  f"{m['drop_radius']:>7.1f} {m['drop_area']:>9.0f} " \
                  f"{m['drop_mean_intensity']:>9.1f} {m['zone_diameter']:>9.0f} " \
                  f"{'Yes' if m['has_zone'] else 'No':>6} " \
                  f"{zoi_per_area:>9.4f} {zoi_per_circ:>9.4f}"
            print(row)
        if plate_measurements:
            print()

    print("=" * 100)


def main():
    # Determine input file
    script_dir = Path(__file__).parent
    input_file = script_dir / "ds_18hr006.tif"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Processing: {input_file}")

    # Load image
    rgb, gray = load_image(str(input_file))
    print(f"Image size: {gray.shape[1]} x {gray.shape[0]}")

    # Apply mild blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 1)

    # Detect plates
    plates = detect_plates(gray_blur)
    print(f"Detected {len(plates)} plates")
    for i, (cx, cy, r) in enumerate(plates):
        print(f"  Plate {i + 1}: center=({cx}, {cy}), radius={r}")

    if len(plates) == 0:
        print("Error: No plates detected!")
        sys.exit(1)

    # Process each plate
    all_measurements = []

    for plate_idx, (pcx, pcy, pr) in enumerate(plates):
        print(f"\nAnalyzing Plate {plate_idx + 1}...")

        # Find drops
        drops = find_drops_in_plate(gray_blur, pcx, pcy, pr)
        print(f"  Found {len(drops)} candidate drops")

        plate_measurements = []
        for cx, cy, intensity, area in drops:
            result = measure_drop_and_zone(gray_blur, cx, cy, pcx, pcy, pr)
            if result is None:
                continue

            measurement = {
                'center_x': cx,
                'center_y': cy,
                **result
            }
            plate_measurements.append(measurement)
            status = "ZONE" if result['has_zone'] else "no zone"
            print(f"    Drop at ({cx},{cy}): intensity={result['drop_mean_intensity']:.0f}, "
                  f"drop_r={result['drop_radius']}, zone_d={result['zone_diameter']:.0f} [{status}]")

        # Sort by position (top-to-bottom, left-to-right)
        plate_measurements.sort(key=lambda m: (m['center_y'] // 50, m['center_x']))
        all_measurements.append(plate_measurements)

    # Print summary
    print_summary(all_measurements)

    # Save outputs
    stem = input_file.stem
    annotated_path = str(script_dir / f"annotated_{stem}.png")
    csv_path = str(script_dir / f"measurements_{stem}.csv")

    create_annotated_image(rgb, plates, all_measurements, annotated_path)
    write_csv(all_measurements, csv_path)

    total_drops = sum(len(pm) for pm in all_measurements)
    total_zones = sum(1 for pm in all_measurements for m in pm if m['has_zone'])
    print(f"\nTotal: {total_drops} drops detected, {total_zones} with zones of inhibition")


if __name__ == "__main__":
    main()
