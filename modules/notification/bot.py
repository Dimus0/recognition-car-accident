import requests 
import os
import json
from dotenv import load_dotenv
import threading
import time

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')


def notification_telegram_bot(photo_path, video_path, description):
    """
    Надсилає фото + відео ДТП у Telegram.
    Якщо video_path=None або файл ще не готовий — надсилає тільки фото.
    """

    if not photo_path or not os.path.exists(photo_path):
        print(f"ERROR: Фото не знайдено: {photo_path!r}")
        return None

    if isinstance(description, dict):
        caption = description.get("dispatcher_summary", str(description))
    else:
        caption = str(description) if description else "⚠️ ДТП зафіксовано"

    caption = caption[:1024]

    has_video = video_path and os.path.exists(video_path)

    if has_video:
        api_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMediaGroup"

        media_group = [
            {
                "type": "photo",
                "media": "attach://accident_photo",
                "caption": caption,
                "parse_mode": "HTML"
            },
            {
                "type": "video",
                "media": "attach://accident_video"
            }
        ]

        with open(photo_path, 'rb') as photo_file, open(video_path, 'rb') as video_file:
            files = {
                'accident_photo': photo_file,
                'accident_video': video_file,
            }
            data = {
                'chat_id': CHAT_ID,
                'media': json.dumps(media_group)
            }

            try:
                response = requests.post(api_url, data=data, files=files)
                response.raise_for_status()
                print("Фото + відео ДТП відправлені успішно!")
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Помилка відправки медіагрупи: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"   Telegram деталі: {e.response.text}")
                return None

    # ── Fallback: тільки фото якщо відео недоступне ─────────────────────────
    else:
        print(f"Відео не готове ({video_path!r}), надсилаємо тільки фото.")
        api_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

        with open(photo_path, 'rb') as photo_file:
            data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
            files = {'photo': photo_file}

            try:
                response = requests.post(api_url, data=data, files=files)
                response.raise_for_status()
                print("Фото ДТП відправлено (без відео).")
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Помилка відправки фото: {e}")
                return None


def schedule_notification(
    photo_path:   str,
    video_path:   str,
    description,
    delay_sec:    float = 0.0,
    poll_timeout: float = 60.0,
    poll_interval: float = 0.5,
):
    """
    Надсилає сповіщення у фоновому потоці.

    """
    def _send():
        if delay_sec > 0:
            time.sleep(delay_sec)

        if video_path:
            waited = 0.0
            while waited < poll_timeout:
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    print(f"Відео готове після {waited + delay_sec:.1f}с: {video_path}")
                    break
                time.sleep(poll_interval)
                waited += poll_interval
            else:
                print(f"Відео не з'явилось за {poll_timeout}с, надсилаємо без відео.")

        notification_telegram_bot(photo_path, video_path, description)

    t = threading.Thread(target=_send, daemon=True, name="tg-notify")
    t.start()
    total_max = delay_sec + poll_timeout
    print(f"Сповіщення заплановано (polling, max {total_max:.0f}с)...")
    return t