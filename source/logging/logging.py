import logging
import os

LOG_OUTPUT = None

def set_log_output(filename='./etc/logs/app.log', log_level='DEBUG'):
    LOG_OUTPUT = filename
    FORMAT = '[%(levelname)s]:[%(asctime)s]:[PID:%(process)d]: %(message)s'
    
    if log_level.upper() == 'INFO':
        level = logging.INFO
    elif log_level.upper() == 'ERROR':
        level = logging.ERROR
    elif log_level.upper() == 'WARNING':
        level = logging.WARNING
    else:
        level = logging.DEBUG
    
    dir_path = filename.replace(filename.split('/')[-1],'')
    if os.path.exists( dir_path ) == False:
        os.makedirs( dir_path )

    logging.basicConfig(format=FORMAT,filename=filename, level=level)

def log(log_message='', level='D'):
    
    if LOG_OUTPUT == None:
        set_log_output()      

    if level.upper() in ['D', 'DEBUG']:
        logging.debug(log_message)
    elif level.upper() in ['I', 'INFO']:
        logging.info(log_message)
    elif level.upper() in ['W', 'WARNING']:
        logging.warning(log_message)
    elif level.upper() in ['E', 'ERROR']:
        logging.error(log_message)
    elif level.upper() in ['C', 'CRITICAL']:
        logging.critical(log_message)
    else:
        logging.warning(f'Unsupported log level {level}')
        logging.warning(f'{level} {log_message}')
    