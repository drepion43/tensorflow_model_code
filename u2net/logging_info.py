import logging
import os

class LoggingConfig:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggingConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # 로그 파일 경로
        log_dir = './logs'
        log_file = 'train_and_eval.log'
        log_path = os.path.join(log_dir, log_file)
        
        # 로그 디렉토리 생성
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 로깅 설정    
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='$(asctime)s - %(name)s - $(levelname)s - $(message)s'
        )
        
        # 전역 로거 설정
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
    
    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)

logging_config = LoggingConfig()