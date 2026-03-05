import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os
from modules.utils.utils import load_models,load_motion_lstm,is_point_in_polygon,process_cnn_batch,calculate_box_distance
from modules.services import (
    TrafficAnalyzer, 
    AccidentStateTracker, 
    AccidentFrameCapture,
    AccidentVideoBuffer
)
from modules.metrics import MetricsTracker


# ---------------------- CONFIG ----------------------

DEVICE                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR                 = r'E:\personalproject\bachelor\logs'
OUTPUT_DIR              = os.path.join(LOG_DIR, "fragments")
METRICS_FILE            = os.path.join(LOG_DIR, "metrics_summary.json")
DETAILED_METRICS_FILE   = os.path.join(LOG_DIR, "detailed_metrics.json")

# ------------------------- WEIGHTS OF MODEL -------------------------------
YOLO_MODEL_PATH         = r"E:\personalproject\bachelor\model\weights\yolov8-fine-tuning.pt"
CNN_WEIGHTS_PATH        = r"E:\personalproject\bachelor\model\weights\accident_cnn_model.pth"
LSTM_MODEL_PATH         = r"E:\personalproject\bachelor\model\weights\accident_lstm_model.pt"
LSTM_SCALER_PATH  = r"E:\personalproject\bachelor\model\weights\motion_lstm_scaler.pkl"
# ------------------------- WEIGHTS OF MODEL -------------------------------

GROUND_TRUTH_PATH       = r"E:\personalproject\bachelor\logs\ground_truth.json" 

# VIDEO_PATH = r"D:\project\highway.mp4"
# VIDEO_PATH              = r"E:\personalproject\bachelor\data\video\videoplayback.mp4"
VIDEO_PATH              = r"E:\personalproject\bachelor\data\video\highway_traffic.mp4"

CONF_YOLO               = 0.3
CONF_ACCIDENT_HIGH      = 0.92     # Поріг точного ДТП
CONF_ACCIDENT_LOW       = 0.86       # Поріг попередження
HEARTBEAT_RATE          = 10             # Як часто перевіряти авто без підозр
LSTM_ACCIDENT_THRESH    = 0.70     # Поріг LSTM для передбачення аварії
VIDEO_BUFFER_SECONDS    = 2.0        # Буфер до/після аварії (секунди)

# ── MotionLSTM параметри (мають збігатись з тренуванням) ─────────
LSTM_OBS_LEN         = 20      # кадрів спостереження
LSTM_PRED_LEN        = 30      # кадрів прогнозу
LSTM_HIDDEN          = 128
LSTM_LAYERS          = 2
LSTM_DROPOUT         = 0.3
LSTM_COORD_SCALE     = 10.0     # ділення координат при тренуванні
LSTM_COLLISION_PX    = 80       # Поріг зближення траєкторій (в пікселях після зворотного масштабування)
LSTM_COLLISION_FRAMES= 10       # Скільки майбутніх кроків прогнозу перевіряти на зближення

os.makedirs(OUTPUT_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "accident_detection.log")
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

cnn_model, yolo_model, deepsort = load_models(DEVICE, CNN_WEIGHTS_PATH, YOLO_MODEL_PATH)
motion_lstm = load_motion_lstm(LSTM_MODEL_PATH, LSTM_SCALER_PATH, DEVICE)
logger.info(f"Models loaded successfully. MotionLSTM: {'ON' if motion_lstm else 'OFF'}")

cnn_transforms = transforms.Compose([
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
        ground_truth_path=GROUND_TRUTH_PATH,
        auto_generate_gt=True)

    cap = cv2.VideoCapture(input_video)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    metrics_tracker.set_video_info(width, height, fps, total_frames, input_video)
    metrics_tracker.start_processing()

    output_path = os.path.join(OUTPUT_DIR, "result.mp4")
    out = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    video_buffer = AccidentVideoBuffer(OUTPUT_DIR,fps)
    # -------- Лінії після яких відбувається трекінг --------

    '''
        Область для відео із шосе
    '''
    ROI_POLYGON = np.array([
        (931, 1074),
        (120, 956),
        (816, 591),
        (1025, 597)
    ], dtype=np.int32)

    '''
        Область для відео із ДТП
    '''
    # ROI_POLYGON = np.array([
    #     (9, 216),
    #     (19, 996),
    #     (1450, 791),
    #     (511, 230)
    # ], dtype=np.int32)

    # -------- END --------

    analyzer = TrafficAnalyzer(
        collision_ttc_threshold=1.5,
        sudden_stop_threshold=0.19,
        min_speed_for_stop=5.0
    )
    accident_capture = AccidentFrameCapture(OUTPUT_DIR)
    accident_state = AccidentStateTracker()

    frame_count = 0
    cnn_results_cache = {}

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
        logger.info()

        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)

        '''
            ======== YOLO Inference ========
        '''
        results = yolo_model(frame, conf=CONF_YOLO, verbose=False)

        detections =  []
        yolo_confidences = []

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
        if metrics_tracker.ground_truth is not None:
            predicted_tracks = [
                (tid, box, 1.0)
                for tid, box in zip(ids, boxes)
            ]
            metrics_tracker.update_tracking_for_evaluation(
                frame_num=frame_count,
                predicted_tracks=predicted_tracks,
                iou_threshold=0.5
            )

        analyzer.update_tracks(boxes, ids)
        analyzer.clean_old_tracks(ids)
        metrics_tracker.update_tracking_metrics(ids, analyzer.track_history)
        metrics_tracker.record_tracks(frame_count,boxes,ids)

        collision_risky_ids = analyzer.predict_collision(boxes=boxes, ids=ids)
        accident_state.cleanup_old_accidents(frame_count)

        lstm_risk_scores = {}
        lstm_risk_ids = set()

        if motion_lstm:

            for tid in ids:
                history = analyzer.track_history.get(tid, [])
                motion_lstm.update_from_history(tid,history)
            
            motion_lstm.run_batch(ids)

            lstm_risk_scores = motion_lstm.get_collision_risk_pairs(ids)
            lstm_risk_ids = set(lstm_risk_scores.keys())
            motion_lstm.cleanup(ids)

            metrics_tracker.update_lstm_metrics(lstm_risk_scores)

        crops_to_process, ids_to_process, box_map_for_cnn = [], [], []
        cnn_skipped = 0

        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            
            should_run = (
                accident_state.is_accident_active(tid, frame_count)
                or tid in collision_risky_ids
                or tid in lstm_risk_ids              # ← LSTM виявив ризик
                or analyzer.detect_sudden_stop(tid, boxes, ids)
                or frame_count % HEARTBEAT_RATE == 0
            )
            
            if should_run:
                crop = frame[y1:y2, x1:x2]
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
            scores = process_cnn_batch(crops_to_process, cnn_transforms, cnn_model)
            cnn_time = ((cv2.getTickCount() - cnn_start) / cv2.getTickFrequency()) * 1000
            
            metrics_tracker.update_cnn_metrics(len(crops_to_process), cnn_time, scores)
            
            for (x1, y1, x2, y2), score, tid in zip(box_map_for_cnn, scores, ids_to_process):
                # LSTM підсилення score: якщо прогноз показує зіткнення
                # і CNN на межі — злегка підштовхуємо через ризик
                lstm_boost = lstm_risk_scores.get(tid, 0.0) * 0.05
                eff_score = min(1.0, score +lstm_boost)

                accident_state.update_score(tid, eff_score, frame_count)
                is_already_confirmed = accident_state.is_confirmed_accident(tid)
                should_confirm = accident_state.should_confirm_accident(tid, eff_score)
                
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
                
                elif should_confirm and eff_score > CONF_ACCIDENT_HIGH:
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
                
                elif tid in collision_risky_ids and eff_score > CONF_ACCIDENT_LOW:
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
                    
                    logger.info(f"[FRAME {frame_count}] WARNING | ID={tid} | SCORE={eff_score:.2f}")
                
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

        # if accident_detected_this_frame and accident_objects:
        #     primary_positions = [obj['bbox'] for obj in accident_objects 
        #                          if obj['type'] == 'primary']

        #     for i, tid in enumerate(ids):
        #         if tid not in [obj['track_id'] for obj in accident_objects]:
        #             x1, y1, x2, y2 = boxes[i]

        #             for px1, py1, px2, py2 in primary_positions:
        #                 distance = calculate_box_distance(
        #                     (x1, y1, x2, y2),
        #                     (px1, py1, px2, py2)
        #                 )

        #                 if distance < 250:
        #                     accident_objects.append({
        #                         'track_id': tid,
        #                         'bbox': (x1, y1, x2, y2),
        #                         'confidence': cnn_results_cache.get(tid, (None, None, 0.0))[2],
        #                         'type': 'secondary'
        #                     })
        #                     break
            
            should_save = any(
                accident_capture.should_save_accident(obj['track_id'], frame_count)
                for obj in accident_objects if obj['type'] == 'primary'
            )

            if should_save:
                video_buffer.trigger(frame_count)
                saved_path = accident_capture.save_accident_frame(
                    frame=frame,
                    frame_number=frame_count,
                    accident_objects=accident_objects,
                    video_path=input_video
                )

                """
                    FUNCTION OF NOTIFICATIONS
                """

                logger.critical(
                    f"[ACCIDENT SAVED] Frame {frame_count} | "
                    f"Vehicles: {len(accident_objects)} | "
                    f"Saved to: {saved_path}"
                )

        collision_warnings = len(collision_risky_ids)
        sudden_stops = sum(1 for tid in ids if analyzer.detect_sudden_stop(tid, boxes, ids))
        
        metrics_tracker.update_accident_metrics(
            accident_detected_this_frame,
            len(accident_objects) if accident_objects else 0,
            collision_warnings,
            sudden_stops
        )

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
        # for i, tid in enumerate(ids):
        #     x1, y1, x2, y2 = boxes[i]
            
        #     # Малюємо траєкторію
        #     points = analyzer.track_history.get(tid, [])
        #     if len(points) > 1:
        #         pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        #         cv2.polylines(frame, [pts], isClosed=False, 
        #                       color=(255, 0, 0), thickness=2)

        #     # Беремо дані з кешу
        #     if tid in cnn_results_cache:
        #         color, label, score = cnn_results_cache[tid]
        #     else:
        #         color, label = (255, 255, 255), f"ID {tid}" # White defaults

        #     # Малюємо бокс і текст
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #     cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

        frame_time = ((cv2.getTickCount() - frame_start) / cv2.getTickFrequency()) * 1000
        metrics_tracker.update_frame_metrics(frame_time)

    cap.release()
    out.release()

    final_metrics = metrics_tracker.finalize()

    analyzer_metrics = analyzer.get_metrics_summary()
    final_metrics['analyzer'] = analyzer_metrics
    metrics_tracker.save_to_file(METRICS_FILE)

    summary = accident_capture.get_accident_summary()
    metrics_tracker.print_summary()
    metrics_tracker.save_all_plots(OUTPUT_DIR)

    print(f"\n{'*'*70}")
    print(" "*20 + "ЗВЕДЕННЯ ПО АВАРІЯХ")
    print(f"{'*'*70}")
    print(f"Всього аварій:          {summary['total_accidents']}")
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
    print(f"Метрики збережено: {METRICS_FILE}")
    
    logger.info(f"Processing finished. Output: {output_path}")
    return output_path


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    try:
        output_path = accident_detection(VIDEO_PATH)
        print(f"\nУспішно завершено!")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        print(f"\nПомилка: {e}")
        raise