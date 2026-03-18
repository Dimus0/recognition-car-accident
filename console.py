import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os
import shutil  
from modules.utils.utils import (
    load_models,
    load_motion_lstm,
    is_point_in_polygon,
    process_cnn_batch,
    calculate_box_distance,
    build_accident_description
    )
from modules.services import (
    TrafficAnalyzer, 
    AccidentStateTracker, 
    AccidentFrameCapture,
    AccidentVideoBuffer
)
from modules.metrics import MetricsTracker
from modules.config.config import Config
from modules.notification.bot import schedule_notification,wait_for_notifications
from collections import deque, defaultdict

# ROI
if os.path.basename(Config.VIDEO_PATH) == "accident_video.mp4":
    '''
        Область для відео із ДТП
    '''
    ROI_POLYGON = np.array([
        (9, 216),
        (19, 996),
        (1450, 791),
        (511, 230)
    ], dtype=np.int32)

elif os.path.basename(Config.VIDEO_PATH) == "highway_traffic.mp4":
    ROI_POLYGON = np.array([
        (931, 1074),
        (120, 956),
        (816, 591),
        (1025, 597)
    ], dtype=np.int32)

else:
    ROI_POLYGON = np.array([
        (335, 1058),
        (1520, 1077),
        (1127, 591),
        (820, 486)
    ], dtype=np.int32)



os.makedirs(Config.OUTPUT_DIR,exist_ok=True)
if os.path.exists(Config.OUTPUT_DIR_CLIP):
    shutil.rmtree(Config.OUTPUT_DIR_CLIP)
os.makedirs(Config.OUTPUT_DIR_CLIP,exist_ok=True)
os.makedirs(Config.ARTIFACTS_PATH,exist_ok=True)

if os.path.exists(Config.RESNET_CROPS_DIR):
    shutil.rmtree(Config.RESNET_CROPS_DIR)
os.makedirs(Config.RESNET_CROPS_DIR,exist_ok=True)

log_file = os.path.join(Config.LOG_DIR, "accident_detection.log")
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    filename=log_file,
    encoding='utf-8',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("accident_detector")


# ---------------------- INIT ----------------------

cnn_model,yolo_model,deepsort = load_models(
    Config.DEVICE, 
    Config.CLASSIFIER_WEIGHTS_PATH, 
    Config.YOLO_MODEL_PATH
)

logger.info(
    "Класифікатор завантажено | класи=%s | device=%s",
    cnn_model.class_names, Config.DEVICE
)

motion_lstm = load_motion_lstm(Config.LSTM_MODEL_PATH, Config.LSTM_SCALER_PATH, Config.DEVICE)
logger.info(f"Models loaded successfully. MotionLSTM: {'ON' if motion_lstm else 'OFF'}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


"""
    ============================= MAIN PIPELINE =============================
"""

def accident_detection(input_video):

    metrics_tracker = MetricsTracker(
        ground_truth_path=Config.GROUND_TRUTH_PATH,
        auto_generate_gt=True)

    cap = cv2.VideoCapture(input_video)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    metrics_tracker.set_video_info(width, height, fps, total_frames, input_video)
    metrics_tracker.start_processing()

    output_path = os.path.join(Config.OUTPUT_DIR, "result.mp4")
    out = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    video_buffer = AccidentVideoBuffer(Config.OUTPUT_DIR_CLIP, fps)

    analyzer = TrafficAnalyzer(
        collision_ttc_threshold=Config.COLLISION_TTC_THRESHOLD,
        sudden_stop_threshold=Config.SUDDEN_STOP_THRESHOLD,
        min_speed_for_stop=Config.MIN_SPEED_FOR_STOP
    )
    accident_capture = AccidentFrameCapture(Config.OUTPUT_DIR)
    accident_state = AccidentStateTracker()

    frame_count = 0
    cnn_results_cache = {}

    notified_accidents: set = set()
    notification_threads: list = []

    logger.info(f"Starting processing: {total_frames} frames | LSTM: {'ON' if motion_lstm else 'OFF'}")

    """
        START PROCESS
    """
    while cap.isOpened():

        frame_start = cv2.getTickCount()

        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        clean_frame = frame.copy()
        video_buffer.push(clean_frame)

        cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)

        '''
            ======== YOLO Inference ========
        '''
        results = yolo_model(frame, conf=Config.CONF_YOLO, iou=0.7, verbose=False)

        yolo_confidences = []
        detections = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf < 0.3:
                    continue
                
                center_x = int((x1 + x2) / 2)
                bottom_y = int(y2)

                if is_point_in_polygon((center_x, bottom_y), ROI_POLYGON):
                    detections.append((
                        [x1, y1, x2 - x1, y2 - y1], # XYWH format for DeepSort
                        conf,
                        cls
                    ))
                    yolo_confidences.append(conf)


        metrics_tracker.update_yolo_metrics(detections, yolo_confidences)
        tracks = deepsort.update_tracks(detections, frame=frame)

        boxes = []
        ids = []
        active_tracks_objects = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = map(int, track.to_ltrb())
            tid = track.track_id

            if not is_point_in_polygon((int((l+r)/2), int(b)), ROI_POLYGON): continue

            boxes.append((l, t, r, b))
            ids.append(tid)
            active_tracks_objects.append(track)
        
        # --- Оцінка трекінгу (MOTA/IDF1) має бути ТУТ, коли boxes/ids вже заповнені ---
        predicted_tracks = [
            (tid, box, 1.0)
            for tid, box in zip(ids, boxes)
        ]

        metrics_tracker.update_tracking_for_evaluation(
            frame_num=frame_count,
            predicted_tracks=predicted_tracks,
            iou_threshold=0.7
        )

        analyzer.update_tracks(boxes, ids)
        analyzer.clean_old_tracks(ids)
        metrics_tracker.update_tracking_metrics(ids, analyzer.track_history)
        metrics_tracker.record_tracks(frame_count, boxes, ids)

        # Ризик зіткнення за кінематичними ознаками (без LSTM)
        collision_risky_ids = analyzer.predict_collision(boxes=boxes, ids=ids)
        collision_risky_ids_kin = set(collision_risky_ids)
        accident_state.cleanup_old_accidents(frame_count)

        lstm_risk_scores = {}
        lstm_risk_ids = set()
        lstm_predicted_positions = {}
        lstm_high_risk_ids = set()

        if motion_lstm:

            for tid in ids:
                history = analyzer.track_history.get(tid, [])
                motion_lstm.update_from_history(tid, history)

            motion_lstm.run_batch(ids)

            lstm_risk_scores = motion_lstm.get_collision_risk_ttc(ids)
            lstm_risk_ids = set(lstm_risk_scores.keys())

            metrics_tracker.update_lstm_metrics(lstm_risk_scores)

            lstm_predicted_positions = motion_lstm.get_predicted_positions(ids, steps=10)
            if lstm_predicted_positions:
                analyzer.update_predicted_positions(lstm_predicted_positions)

            # треки з LSTM ризиком >= 0.70 входять до фінального risky set
            lstm_high_risk_ids = {
                tid for tid, risk in lstm_risk_scores.items() if risk >= 0.70
            }
            if lstm_high_risk_ids:
                logger.info(
                    f"[FRAME {frame_count}] LSTM high-risk: "
                    f"{ {tid: round(lstm_risk_scores[tid], 2) for tid in lstm_high_risk_ids} }"
                )

        # Підсумковий risky set: кінематика ∪ LSTM
        collision_risky_ids_total = collision_risky_ids_kin | lstm_high_risk_ids

        crops_to_process, ids_to_process, box_map_for_cnn = [], [], []
        sudden_stop_cache: dict = {}

        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]

            is_sudden_stop = analyzer.detect_sudden_stop(tid, boxes, ids)
            sudden_stop_cache[tid] = is_sudden_stop

            should_run = (
                accident_state.is_accident_active(tid, frame_count)
                or tid in collision_risky_ids_total
                # or tid in lstm_risk_ids              # ← LSTM виявив ризик
                or is_sudden_stop
                or frame_count % Config.HEARTBEAT_RATE == 0
            )

            if should_run:
                h_frame, w_frame = frame.shape[:2]
                pad = Config.CNN_CONTEXT_PADDING
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w_frame, x2 + pad)
                cy2 = min(h_frame, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crops_to_process.append(crop)
                    ids_to_process.append(tid)
                    box_map_for_cnn.append((x1, y1, x2, y2))

        accident_detected_this_frame = False
        accident_objects = []
        accident_ids_set= set()

        # 6. CNN Inference
        cnn_start = cv2.getTickCount()
        if crops_to_process:
            scores = process_cnn_batch(crops_to_process, cnn_model)
            cnn_time = ((cv2.getTickCount() - cnn_start) / cv2.getTickFrequency()) * 1000
            
            metrics_tracker.update_cnn_metrics(len(crops_to_process), cnn_time, scores)
            
            for (x1, y1, x2, y2), score, tid,crop_img in zip(box_map_for_cnn, scores, ids_to_process,crops_to_process):
                # lstm
                lstm_risk = lstm_risk_scores.get(tid,0.0)
                lstm_boost = (lstm_risk ** 2) * 0.25 * score
                eff_score = min(1.0, score + lstm_boost)

                is_kinematic = tid in collision_risky_ids_kin

                if is_kinematic:
                    lstm_boost = (lstm_risk ** 2) * 0.25 * score
                else:
                    lstm_boost = 0.0
                eff_score = min(1.0, score + lstm_boost)

                if score >= Config.CNN_SAVE_CROP_THRESH:
                    from datetime import datetime as _dt
                    _ts = _dt.now().strftime("%H%M%S_%f")
                    _crop_name = f"f{frame_count:06d}_id{tid}_s{score:.3f}_{_ts}.jpg"
                    _crop_path = os.path.join(Config.RESNET_CROPS_DIR, _crop_name)
                    try:
                        cv2.imwrite(_crop_path, crop_img)
                    except Exception as _e:
                        logger.warning(f"[CROP SAVE] Помилка: {_e}")

                accident_state.update_score(tid, eff_score, frame_count)
                is_already_confirmed = accident_state.is_confirmed_accident(tid)

                should_confirm = accident_state.should_confirm_accident(
                    tid, eff_score,
                    lstm_risk=lstm_risk,
                    is_kinematic=is_kinematic,   # LSTM reduction тільки якщо кінематика згодна
                )
                should_confirm = accident_state.should_confirm_accident(tid, eff_score,lstm_risk=lstm_risk)
                is_kinematic = tid in collision_risky_ids_kin
                is_lstm_flag = tid in lstm_high_risk_ids or tid in lstm_risk_ids
                is_sudden    = sudden_stop_cache.get(tid, False)

                if is_already_confirmed:
                    label = f"ACCIDENT {eff_score:.2f}"
                    color = (0, 0, 255)
                    accident_detected_this_frame = True
                    accident_objects.append({
                        'track_id': tid,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': eff_score,
                        'type': 'primary'
                    })
                    accident_ids_set.add(tid)

                    metrics_tracker.record_accident(
                        frame_num=frame_count,
                        accident_type='confirmed',
                        confidence=eff_score,
                        involved_tracks=[tid],
                        severity='high'
                    )
                    
                    logger.info(f"[FRAME {frame_count}] CONFIRMED ACCIDENT | ID={tid} | SCORE={score:.2f}")
                    logger.info(
                        "[FRAME %d] ✅ CONFIRMED ACCIDENT | "
                        "ID=%s | cnn_score=%.4f | eff_score=%.4f | "
                        "lstm_risk=%.4f | lstm_boost=%.4f | "
                        "kinematic=%s | lstm_flag=%s | sudden_stop=%s",
                        frame_count, tid,
                        score, eff_score,
                        lstm_risk, lstm_boost,
                        is_kinematic, is_lstm_flag, is_sudden,
                    )
                elif should_confirm and eff_score > Config.CONF_ACCIDENT_HIGH:
                    accident_state.confirm_accident(tid, frame_count)
                    
                    label = f"ACCIDENT {eff_score:.2f}"
                    color = (0, 0, 255)
                    
                    accident_detected_this_frame = True
                    accident_objects.append({
                        'track_id': tid,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': eff_score,
                        'type': 'primary'
                    })

                    accident_ids_set.add(tid)

                    metrics_tracker.record_accident(
                        frame_num=frame_count,
                        accident_type='collision',
                        confidence=eff_score,
                        involved_tracks=[tid],
                        severity='high'
                    )
                    
                    logger.warning(f"[FRAME {frame_count}] NEW ACCIDENT | ID={tid} | SCORE={score:.2f}")
                    logger.warning(
                        "[FRAME %d] 🚨 NEW ACCIDENT DETECTED | "
                        "ID=%s | cnn_score=%.4f | eff_score=%.4f | "
                        "lstm_risk=%.4f | lstm_boost=%.4f | "
                        "kinematic=%s | lstm_flag=%s | sudden_stop=%s | "
                        "bbox=(%d,%d,%d,%d) | threshold=%.2f",
                        frame_count, tid,
                        score, eff_score,
                        lstm_risk, lstm_boost,
                        is_kinematic, is_lstm_flag, is_sudden,
                        x1, y1, x2, y2,
                        Config.CONF_ACCIDENT_HIGH,
                    )

                elif tid in collision_risky_ids_total and eff_score > Config.CONF_ACCIDENT_LOW:
                    label = f"WARNING {eff_score:.2f}"
                    color = (0, 255, 255)
                    
                    if accident_detected_this_frame:
                        accident_objects.append({
                            'track_id': tid,
                            'bbox': (x1, y1, x2, y2),
                            'confidence': eff_score,
                            'type': 'secondary'
                        })

                        accident_ids_set.add(tid)

                        metrics_tracker.record_accident(
                            frame_num=frame_count,
                            accident_type='warning',
                            confidence=eff_score,
                            involved_tracks=[tid],
                            severity='medium'
                        )

                    logger.info(
                        "[FRAME %d] ⚠️  WARNING | "
                        "ID=%s | cnn_score=%.4f | eff_score=%.4f | "
                        "lstm_risk=%.4f | kinematic=%s | lstm_flag=%s | sudden_stop=%s",
                        frame_count, tid,
                        score, eff_score,
                        lstm_risk,
                        is_kinematic, is_lstm_flag, is_sudden,
                    )

                    # Детальний лог причини попередження
                
                else:
                    if tid in lstm_risk_ids:
                        risk = lstm_risk_scores[tid]
                        label = f"LSTM\u26A0 {risk:.2f}"
                        color = (0,165,255)
                    else:
                        label = f"NORMAL {eff_score:.2f}"
                        color = (0, 255, 0)
                
                cnn_results_cache[tid] = (color, label, eff_score)

        # ── Secondary авто (поблизу підтвердженої аварії) ─────────
        if accident_detected_this_frame and accident_objects:
            primary_bboxes = [o['bbox'] for o in accident_objects if o['type'] == 'primary']
            for i, tid in enumerate(ids):
                if tid in accident_ids_set:
                    continue
                x1, y1, x2, y2 = boxes[i]
                for pb in primary_bboxes:
                    if calculate_box_distance((x1, y1, x2, y2), pb) < 250:
                        accident_objects.append({
                            'track_id': tid, 'bbox': (x1, y1, x2, y2),
                            'confidence': cnn_results_cache.get(tid, (None, None, 0.0))[2],
                            'type': 'secondary'
                        })
                        accident_ids_set.add(tid)
                        prev = cnn_results_cache.get(tid, (None, f"ID {tid}", 0.0))
                        cnn_results_cache[tid] = ((0, 0, 255), f"NEAR {prev[2]:.2f}", prev[2])
                        break
            
            should_save = any(
                accident_capture.should_save_accident(obj['track_id'], frame_count)
                for obj in accident_objects if obj['type'] == 'primary'
            )

            if should_save:
                accident_photo_path = accident_capture.save_accident_frame(
                    frame=frame,
                    frame_number=frame_count,
                    accident_objects=accident_objects,
                    video_path=input_video
                )
            
                accident_video_path = video_buffer.trigger(frame_count)

                accident_description = build_accident_description(
                    accident_objects=accident_objects,
                    frame_count=frame_count,
                    fps=fps,
                    camera_id="CAM_001",       #  ідентифікатор камери
                    camera_location="ТЕСТУВАННЯ", #  адреса / назва перехрестя
                )
                accident_description["photo_path"] = accident_photo_path
                accident_description["video_path"] = accident_video_path

                logger.critical(
                    f"[ACCIDENT SAVED] Frame {frame_count} | "
                    f"Vehicles: {len(accident_objects)} | "
                    f"Severity: {accident_description['severity']} | "
                    f"Photo: {accident_photo_path} | Video: {accident_video_path}"
                )
                logger.critical(
                    "[ACCIDENT SAVED] Frame=%d | "
                    "Vehicles=%d (primary=%d secondary=%d) | "
                    "Severity=%s | max_conf=%.4f | avg_conf=%.4f | "
                    "timestamp=%s | Photo=%s | Video=%s",
                    frame_count,
                    accident_description['vehicles_total'],
                    accident_description['vehicles_primary'],
                    accident_description['vehicles_secondary'],
                    accident_description['severity'],
                    accident_description['max_confidence'],
                    accident_description['avg_confidence'],
                    accident_description['timestamp_formatted'],
                    accident_photo_path,
                    accident_video_path,
                )

                new_ids = {
                    obj['track_id']
                    for obj in accident_objects
                    if obj['type'] == 'primary' and obj['track_id'] not in notified_accidents
                }

                if new_ids:
                    notified_accidents.update(new_ids)
                    # video_path вже відомий (trigger повертає майбутній шлях),
                    # затримка потрібна щоб файл встиг записатись
                    post_delay = 4.0 if accident_video_path is None else 4.0
                    t = schedule_notification(
                        photo_path=accident_photo_path,
                        video_path=accident_video_path,
                        description=accident_description,
                        delay_sec=post_delay,
                        poll_timeout=240.0,
                    )
                    notification_threads.append(t)
                    logger.info(f"[NOTIFICATION SCHEDULED] IDs={new_ids} delay={post_delay}s")
        
        collision_warnings = len(collision_risky_ids_total)
        sudden_stops = sum(1 for tid in ids if sudden_stop_cache.get(tid, False))
        
        metrics_tracker.update_accident_metrics(
            accident_detected_this_frame,
            len(accident_objects) if accident_objects else 0,
            collision_warnings,
            sudden_stops
        )

        if motion_lstm:
            motion_lstm.cleanup(ids)

        # 7. Візуалізація (Draw Loop)
        if motion_lstm:
            frame = motion_lstm.draw_predictions(
                frame, ids, accident_ids_set | lstm_risk_ids, steps=15
            )

        # 2) Боксинг, підписи, спостережні траєкторії
        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]

            history = analyzer.track_history.get(tid, [])
            if len(history) > 1:
                pts       = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                traj_clr  = (0, 0, 200) if tid in accident_ids_set else (160, 70, 0)
                cv2.polylines(frame, [pts], False, traj_clr, 1)

            color, label, score = cnn_results_cache.get(
                tid, ((200, 200, 200), f"ID {tid}", 0.0)
            )
            if tid in accident_ids_set:
                color = (0, 0, 255)

            thickness = 3 if tid in accident_ids_set else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # LSTM risk overlay знизу від бокса
            if tid in lstm_risk_ids:
                risk_val = lstm_risk_scores.get(tid, 0.0)
                cv2.putText(frame, f"L:{risk_val:.2f}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
        out.write(frame)

        frame_time = ((cv2.getTickCount() - frame_start) / cv2.getTickFrequency()) * 1000
        metrics_tracker.update_frame_metrics(frame_time)

    cap.release()
    out.release()

    video_buffer.flush()

    wait_for_notifications(notification_threads)

    # ===================== METRICS PART SUMMARY =========================
    final_metrics = metrics_tracker.finalize()
    analyzer_metrics = analyzer.get_metrics_summary()
    final_metrics['analyzer'] = analyzer_metrics
    metrics_tracker.save_to_file(Config.METRICS_FILE)

    summary = accident_capture.get_accident_summary()
    metrics_tracker.print_summary()
    metrics_tracker.save_all_plots(Config.ARTIFACTS_PATH)

    print(f"\n{'*'*70}")
    print(" "*20 + "ЗВЕДЕННЯ ПО АВАРІЯХ") # ТРЕБА ПЕРЕВІРИТИ НЕ КОРЕКТНО
    print(f"{'*'*70}")
    print(f"Всього аварій:          {summary['total_accidents']}") # Не правильно рахує
    print(f"Всього авто задіяно:    {summary.get('total_vehicles_involved', 0)}")
    print(f"Унікальних авто:        {summary.get('unique_vehicles', 0)}")
    if motion_lstm:
        print(f"LSTM попереджень: {len(lstm_risk_ids)} (останній кадр)")
    print(f"Кліпів збережено: {video_buffer.clip_index}")
    print(f"Збережено до:           {accident_capture.accidents_dir}")
    print(f"Лог метаданих:          {accident_capture.metadata_file}")
    print(f"{'*'*70}\n")
    
    print(f"Обробку завершено!")
    print(f"Відео збережено: {output_path}")
    print(f"Метрики збережено: {Config.METRICS_FILE}")
    
    logger.info(f"Processing finished. Output: {output_path}")
    return output_path


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    try:
        output_path = accident_detection(Config.VIDEO_PATH)
        print(f"\nУспішно завершено!")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        print(f"\nПомилка: {e}")
        raise