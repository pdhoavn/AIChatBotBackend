import re
import requests
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

GRAPH_API_URL = "https://graph.facebook.com/v21.0"


class FacebookService:
    def __init__(self):
        self.page_token = settings.FACEBOOK_PAGE_ACCESS_TOKEN
        self.api_url = GRAPH_API_URL

    @staticmethod
    def strip_markdown(text: str) -> str:
        """Convert markdown to plain text suitable for Messenger."""
        # Bold/italic/strikethrough: **text** or __text__ or ~~text~~ -> text
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'~~(.+?)~~', r'\1', text)
        # Italic: *text* or _text_ -> text
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
        text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'\1', text)
        # Inline code: `code` -> code
        text = re.sub(r'`(.+?)`', r'\1', text)
        # Headers: ### Title -> Title
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # List markers: - item or * item or 1. item -> • item
        text = re.sub(r'^[\*\-]\s+', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '• ', text, flags=re.MULTILINE)
        # Links: [text](url) -> text
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        # Horizontal rule: --- or *** -> (empty line)
        text = re.sub(r'^[\*\-]{3,}$', '', text, flags=re.MULTILINE)
        # Extra newlines -> single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _send_request(self, payload: dict) -> dict | None:
        """Gửi request lên Messenger Graph API."""
        url = f"{self.api_url}/me/messages"
        payload["access_token"] = self.page_token
        try:
            response = requests.post(url, json=payload, timeout=30)
            result = response.json()
            if response.status_code != 200:
                logger.error(f"Facebook API error: {result}")
                return None
            return result
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None

    def send_text_message(self, sender_psid: str, text: str) -> bool:
        """Gửi text message tới người dùng Messenger."""
        payload = {
            "messaging_type": "RESPONSE",
            "recipient": {"id": sender_psid},
            "message": {"text": text},
        }
        result = self._send_request(payload)
        return result is not None

    def send_typing_on(self, sender_psid: str) -> bool:
        """Bật typing indicator."""
        payload = {
            "recipient": {"id": sender_psid},
            "sender_action": "typing_on",
        }
        return self._send_request(payload) is not None

    def send_typing_off(self, sender_psid: str) -> bool:
        """Tắt typing indicator."""
        payload = {
            "recipient": {"id": sender_psid},
            "sender_action": "typing_off",
        }
        return self._send_request(payload) is not None

    def send_quick_replies(
        self, sender_psid: str, text: str, quick_replies: list[dict]
    ) -> bool:
        """
        Gửi message kèm quick reply buttons.
        quick_replies format: [{"content_type": "text", "title": "...", "payload": "..."}]
        """
        payload = {
            "messaging_type": "RESPONSE",
            "recipient": {"id": sender_psid},
            "message": {
                "text": text,
                "quick_replies": quick_replies,
            },
        }
        result = self._send_request(payload)
        return result is not None

    def verify_webhook(self, mode: str, token: str, challenge: str) -> str | None:
        """Verify webhook URL - Facebook gọi GET /webhook khi setup."""
        if mode == "subscribe" and token == settings.FACEBOOK_VERIFY_TOKEN:
            return challenge
        return None


# Singleton instance
facebook_service = FacebookService()
