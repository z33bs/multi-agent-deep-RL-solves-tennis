import time
from pathlib import Path


class Logger():
    def __init__(self, path='log/ddpg_'):
        Path('log').mkdir(parents=True, exist_ok=True)
        self.path = path + '{}'.format(time.time()) + '.txt'

    def write_line(self, log_txt, copy_to_screen=True):
        with open(self.path, 'a+') as log_file:
            log_file.write(log_txt + '\n')
        print(log_txt)