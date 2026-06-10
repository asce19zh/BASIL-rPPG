import logging
import sys
import os

_IS_CONFIGURED = False
_info_handler = None
_detail_handler = None
_console_handler = None


# --- 輔助類別 ---
class LevelFilter(logging.Filter):
    """一個只允許特定級別通過的篩選器"""

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


class _StreamToLogger:
    """將 stdout 輸出重新導向到 logger 的輔助類別"""

    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


# --- 主要功能函式 ---
def init_logger(info_log_file, detail_log_file):
    """
    設定全域 Root Logger，並重新導向 stdout 和 excepthook。
    這個函式在應用程式的生命週期中只應被呼叫一次，用於首次初始化。
    """
    global _IS_CONFIGURED
    if _IS_CONFIGURED:
        logging.warning("Logger 變更中 ...")
        reset_log_files(info_log_file, detail_log_file)
        return

    # 定義自訂的 DETAIL 級別 (數字介於 INFO 和 WARNING 之間)
    DETAIL_LEVEL = 25
    logging.addLevelName(DETAIL_LEVEL, "DETAIL")

    # 為 Logger 類別動態新增 .detail() 方法，方便使用
    if not hasattr(logging.Logger, 'detail'):
        def detail_method(self, message, *args, **kwargs):
            if self.isEnabledFor(DETAIL_LEVEL):
                self._log(DETAIL_LEVEL, message, args, **kwargs)

        # 將此方法綁定到 Logger 類別
        logging.Logger.detail = detail_method

    # 實際的 handler 設定工作交給 reset_log_files
    reset_log_files(info_log_file, detail_log_file)

    # --- 全域重新導向 (這部分只需要在首次設定時執行一次) ---
    root_logger = logging.getLogger()
    sys.stdout = _StreamToLogger(root_logger, DETAIL_LEVEL)

    def _handle_exception(exc_type, exc_value, exc_traceback):
        # 如果是使用者手動中斷 (Ctrl+C)，則恢復預設行為
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        # 將所有未捕獲的異常記錄為 ERROR 等級
        root_logger.error("Uncaught Exception:", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _handle_exception

    _IS_CONFIGURED = True
    root_logger.info("Root Logger Initialized.")


def reset_log_files(new_info_file, new_detail_file):
    """
    重設日誌檔案的路徑。可以被重複呼叫以動態切換日誌檔案。
    """
    global _info_handler, _detail_handler, _console_handler

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # --- 移除舊的 handlers ---
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # --- 建立並設定新的 handlers ---
    formatter = logging.Formatter('%(asctime)s - %(name)-30s - %(levelname)-8s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # 1. 檔案日誌 (僅記錄 INFO 級別)
    os.makedirs(os.path.dirname(new_info_file) or '.', exist_ok=True)
    _info_handler = logging.FileHandler(new_info_file, mode='a', encoding='utf-8')
    _info_handler.setLevel(logging.INFO)
    _info_handler.setFormatter(formatter)
    _info_handler.addFilter(LevelFilter(logging.INFO))
    root_logger.addHandler(_info_handler)


    # 2. 檔案日誌 (記錄所有級別)
    os.makedirs(os.path.dirname(new_detail_file) or '.', exist_ok=True)
    _detail_handler = logging.FileHandler(new_detail_file, mode='a', encoding='utf-8')
    _detail_handler.setLevel(logging.INFO) # 接收 INFO 及以上所有級別
    _detail_handler.setFormatter(formatter)
    root_logger.addHandler(_detail_handler)


    # 3. 終端機日誌 (只記錄到終端機)
    _console_handler = logging.StreamHandler(sys.__stdout__)
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(formatter)
    root_logger.addHandler(_console_handler)

    if _IS_CONFIGURED:
        logging.info(f"日誌檔案已重設。 Info -> '{new_info_file}', Detail -> '{new_detail_file}'")