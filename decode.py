#!/bin/env python3
# import sys
# sys.modules[__name__].__dict__.clear()

# import importlib
# importlib.reload(modulename)

# Início
import itertools
from utils import util

sl = util.check_system()['sl']


def main():
    decode()


def decode():
    # Configura os objetos
    config = util.Config('config.json')

    # Cria objeto "video" com suas principais pastas
    video = util.VideoParams(config=config, yuv=f'..{sl}yuv-10s')

    # Set basic configuration
    video.decoder = 'ffmpeg'
    video.project = f'results{sl}ffmpeg_scale_12videos_60s'
    video.factor = 'scale'
    video.threads = 'single'  # 'single' or 'multi'
    video.dectime_base = f'dectime_{video.decoder}'
    video.quality_list = getattr(config, f'{video.factor}_list')

    for video.name in config.videos_list:
        for video.tile_format in config.tile_list:
            for video.quality in video.quality_list:
                for video.rodada in range(3):
                    util.decode(video=video)


if __name__ == '__main__':
    main()
