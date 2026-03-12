import requests 
import os
import json
from dotenv import load_dotenv
import threading
import time
import logging

load_dotenv()
logger = logging.getLogger("accident_detector")

BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')


def notification_telegram_bot(photo_path, video_path, description):
    """
    Надсилає ТІЛЬКИ фото ДТП у Telegram.
    video_path — ігнорується (відео зберігається локально у clip-папці).
    """
    if not photo_path or not os.path.exists(photo_path):
        logger.error(f"[TELEGRAM BOT] Фото не знайдено: {photo_path!r}")
        return None

    if isinstance(description, dict):
        caption = description.get("dispatcher_summary", str(description))
    else:
        caption = str(description) if description else "⚠️ ДТП зафіксовано"

    caption = caption[:1024]

    api_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo_file:
        data  = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
        files = {'photo': photo_file}
        try:
            response = requests.post(api_url, data=data, files=files)
            response.raise_for_status()
            logger.info("[TELEGRAM BOT] Фото ДТП відправлено успішно.")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"[TELEGRAM BOT] Помилка відправки фото: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"[TELEGRAM BOT] Telegram деталі: {e.response.text}")
            return None

def schedule_notification(
    photo_path:   str,
    video_path:   str,       # зберігається, але НЕ надсилається в Telegram
    description,
    delay_sec:    float = 0.0,
    poll_timeout: float = 60.0,   # залишено для сумісності, не використовується
    poll_interval: float = 0.5,
):
    """
    Надсилає фото ДТП у фоновому потоці (daemon=False → чекаємо завершення).
    Відео НЕ надсилається: зберігається локально в OUTPUT_DIR_CLIP.
    """
    def _send():
        if delay_sec > 0:
            time.sleep(delay_sec)
        # video_path передається але ігнорується всередині notification_telegram_bot
        notification_telegram_bot(photo_path, video_path, description)

    t = threading.Thread(target=_send, daemon=False, name="tg-notify")
    t.start()
    logger.info(f"[TELEGRAM BOT] Фото-сповіщення заплановано | photo: {photo_path!r}")
    return t

def wait_for_notifications(threads: list, timeout: float = 180.0):
    if not threads:
        return

    logger.info(f"[TELEGRAM BOT] Очікуємо відправку {len(threads)} сповіщень (max {timeout:.0f}с)...")

    for t in threads:
        if not isinstance(t, threading.Thread):
            logger.info(f"[TG] Пропущено об'єкт {t} (не Thread)")
            continue

        t.join(timeout=timeout)

        if t.is_alive():
            logger.info(f"[TELEGRAM BOT] Потік {t.name} не завершився за {timeout}с — пропускаємо")

    logger.info("[TELEGRAM BOT] Всі сповіщення відправлено.")