# app/custom_worker.py
from uvicorn.workers import UvicornWorker

class CustomUvicornWorker(UvicornWorker):
    # Đưa các cấu hình ngầm của Uvicorn vào đây
    CONFIG_KWARGS = {
        "ws_per_message_deflate": False,
        "timeout_keep_alive": 120,
    }