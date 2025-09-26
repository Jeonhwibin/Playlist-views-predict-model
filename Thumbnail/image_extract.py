import os
import cv2
import numpy as np
import cupy as cp
from cuml.cluster import KMeans as cuKMeans
from ultralytics import YOLO, SAM
import pandas as pd
import webcolors
from tqdm import tqdm
import torch
import gc
import time
import glob

def closest_css3_name(requested_hex):
    r, g, b = webcolors.hex_to_rgb(requested_hex)
    min_colors = {}
    for name in webcolors.names("css3"):
        cr, cg, cb = webcolors.name_to_rgb(name)
        distance = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
        min_colors[distance] = name
    return min_colors[min(min_colors.keys())]

def get_main_mask(masks):
    if not masks:
        return None
    areas = [mask.data.sum().item() for mask in masks]
    return masks[np.argmax(areas)]

def get_bbox_from_mask(mask):
    if mask is None:
        return None
    coords = np.where(mask.data[0].cpu().numpy())
    y1, y2 = min(coords[0]), max(coords[0])
    x1, x2 = min(coords[1]), max(coords[1])
    return (x1, y1, x2, y2)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-5)

def merge_boxes(boxes, iou_thresh=0.3):
    merged = []
    used = [False]*len(boxes)
    for i, box in enumerate(boxes):
        if used[i]: continue
        group = [box]
        used[i] = True
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            if iou(box, boxes[j]) > iou_thresh:
                group.append(boxes[j])
                used[j] = True
        x1 = min([b[0] for b in group])
        y1 = min([b[1] for b in group])
        x2 = max([b[2] for b in group])
        y2 = max([b[3] for b in group])
        merged.append([x1, y1, x2, y2])
    return merged

def extract_top3_colors(image_region, filename_inplace):
    roi_rgb = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
    roi_np = roi_rgb.reshape(-1, 3)
    if roi_np.shape[0] < 10: return None
    if roi_np.shape[0] > 10000:
        roi_np = roi_np[np.random.choice(roi_np.shape[0], 10000, replace=False)]
    roi_cp = cp.asarray(roi_np)
    if cp.isnan(roi_cp).any() or cp.isinf(roi_cp).any(): return None
    kmeans = cuKMeans(n_clusters=30, random_state=42)
    kmeans.fit(roi_cp)
    centers = cp.asnumpy(kmeans.cluster_centers_).astype(int)
    labels_k = cp.asnumpy(kmeans.labels_)
    counts = np.bincount(labels_k)
    hex_codes = ['#%02x%02x%02x' % tuple(c) for c in centers]
    color_info_all = []
    for hex_code, count in zip(hex_codes, counts):
        try: color_name = webcolors.hex_to_name(hex_code, spec='css3')
        except: color_name = closest_css3_name(hex_code)
        color_info_all.append((color_name, count))
    color_count = {}
    for cn, cnt in color_info_all:
        color_count[cn] = color_count.get(cn,0) + cnt
    total_pixels = sum(color_count.values())
    top3 = sorted(color_count.items(), key=lambda x:x[1], reverse=True)[:3]
    row = {'filename':filename_inplace}
    for i,(cn,cnt) in enumerate(top3,1):
        row[f'color{i}'] = cn
        row[f'ratio{i}'] = round(cnt/total_pixels*100,2)
    for i in range(len(top3)+1,4):
        row[f'color{i}'] = None
        row[f'ratio{i}'] = 0
    return row

# ========== 메인 프로세스 ==========
image_folder = './images'
save_folder = './images_out'
csv_folder = './images_val'
final_folder = './images_final'
os.makedirs(save_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(final_folder, exist_ok=True)

yolo12_model = YOLO('yolov12x.pt')
yolo8_model = YOLO('yolov8x6_animeface.pt')
sam_model = SAM('sam2_l.pt')

image_color_data = []
none_detected = []
last_break_time = time.time()
csv_count = 0

if os.path.exists(image_folder):
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png','jpg','jpeg','bmp','gif'))]
    files = sorted(files)  # 순서 일치 보장
    for idx, filename in enumerate(tqdm(files, desc="Processing")):
        # === 30분마다 5분 휴식 ===
        current_time = time.time()
        if current_time - last_break_time >= 1800:
            print("\n⚠️ 시스템 과열 방지를 위해 5분간 휴식합니다...")
            time.sleep(300)
            last_break_time = time.time()
        
        img_path = os.path.join(image_folder, filename)
        image = cv2.imread(img_path)
        if image is None: continue
        
        h, w = image.shape[:2]
        filename_inplace = filename[:11]
        all_boxes = []
        detectors = []

        # 1. YOLOv12x 탐지
        yolo12_res = yolo12_model(image, verbose=False)[0]
        if yolo12_res.boxes:
            box = yolo12_res.boxes[0].xyxy.cpu().numpy()[0].astype(int).tolist()
            all_boxes.append(box)
            detectors.append('yolov12x')

        # 2. YOLOv8x6 탐지
        yolo8_res = yolo8_model(image, verbose=False)[0]
        if yolo8_res.boxes:
            box = yolo8_res.boxes[0].xyxy.cpu().numpy()[0].astype(int).tolist()
            all_boxes.append(box)
            detectors.append('yolov8x6')

        # 3. SAM2 전체 탐지 (여러 객체 중 제일 큰 것 1개만)
        sam_res = sam_model(image, verbose=False)[0]
        if sam_res.masks:
            mask = get_main_mask(sam_res.masks)
            sam_box = get_bbox_from_mask(mask)
            if sam_box:
                all_boxes.append(sam_box)
                detectors.append('sam2')

        # 4. 박스 병합 및 SAM2 정밀 마스크 추출
        merged_boxes = merge_boxes(all_boxes)
        final_mask = np.zeros((h,w), dtype=bool)
        used_detectors = set()
        
        for box in merged_boxes:
            x1,y1,x2,y2 = map(int, box)
            sam2_res = sam_model.predict(image, bboxes=[[x1,y1,x2,y2]])
            if not sam2_res[0].masks: continue
            mask = sam2_res[0].masks.data[0].cpu().numpy().astype(bool)
            final_mask |= mask
            used_detectors.update(detectors)

        # 5. 색상 분석 및 이미지 처리
        if final_mask.sum() > 0:
            roi_pixels = image[final_mask]
            row = extract_top3_colors(roi_pixels.reshape(-1,1,3), filename_inplace)
            if row:
                row['detecter'] = '+'.join(sorted(used_detectors)) if used_detectors else 'sam2'
                image_color_data.append(row)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                gray_bgr[final_mask] = image[final_mask]
                cv2.imwrite(os.path.join(save_folder, filename), gray_bgr)
            else:
                none_detected.append(filename)
                cv2.imwrite(os.path.join(save_folder, filename), image)
        else:
            none_detected.append(filename)
            cv2.imwrite(os.path.join(save_folder, filename), image)

        # 100개마다 분할 저장 (각 파일마다 1행 = 1이미지)
        if (idx+1) % 100 == 0:
            csv_path = os.path.join(csv_folder, f'color_results_{csv_count}.csv')
            pd.DataFrame(image_color_data).to_csv(csv_path, index=False)
            image_color_data.clear()
            csv_count += 1
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_bytes()
            gc.collect()

    # 남은 데이터 저장 (예: 마지막 2개)
    if image_color_data:
        csv_path = os.path.join(csv_folder, f'color_results_{csv_count}.csv')
        pd.DataFrame(image_color_data).to_csv(csv_path, index=False)
        image_color_data.clear()

# 탐지 실패 이미지 저장
if none_detected:
    with open('none_detected.txt','w') as f:
        f.write('\n'.join(none_detected))

# ========== CSV 병합 ==========
csv_files = sorted(glob.glob(os.path.join(csv_folder, 'color_results_*.csv')))
dfs = [pd.read_csv(f) for f in csv_files]
if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    os.makedirs(final_folder, exist_ok=True)
    merged_path = os.path.join(final_folder, 'color_results_merged.csv')
    merged.to_csv(merged_path, index=False)
    print(f"\n모든 결과가 병합되어 {merged_path}에 저장되었습니다.")
else:
    print("\n병합할 CSV 파일이 없습니다.")
