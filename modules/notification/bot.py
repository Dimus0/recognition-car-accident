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
                print("✅ Фото + відео ДТП відправлені успішно!")
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"❌ Помилка відправки медіагрупи: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"   Telegram деталі: {e.response.text}")
                return None

    # ── Fallback: тільки фото якщо відео недоступне ─────────────────────────
    else:
        print(f"⚠️  Відео не готове ({video_path!r}), надсилаємо тільки фото.")
        api_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

        with open(photo_path, 'rb') as photo_file:
            data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
            files = {'photo': photo_file}

            try:
                response = requests.post(api_url, data=data, files=files)
                response.raise_for_status()
                print("✅ Фото ДТП відправлено (без відео).")
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"❌ Помилка відправки фото: {e}")
                return None


def schedule_notification(photo_path, video_path, description, delay_sec):
    """
    Відправляє сповіщення через delay_sec секунд (у окремому потоці).
    Використовується коли відео ще не дозаписане (post_seconds буфер).
    """
    def _send():
        time.sleep(delay_sec)
        notification_telegram_bot(photo_path, video_path, description)

    t = threading.Thread(target=_send, daemon=True)
    t.start()
    print(f"Сповіщення заплановано через {delay_sec}с (очікуємо дозапис відео)...")
    return t