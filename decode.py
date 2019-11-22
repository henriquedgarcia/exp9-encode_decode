#!/bin/env python3
# import sys
# sys.modules[__name__].__dict__.clear()

# import importlib
# importlib.reload(modulename)

# Início
import itertools
from utils import util

sl = util.check_system()['sl']
config = util.Config('config.json', factor='crf')
output_folder = f'results{sl}ffmpeg_crf_12videos_60s'
yuv_folder = f'..{sl}yuv-10s'


def main():
    # Cria objeto "video" com suas principais pastas
    video = util.VideoParams(config=config, yuv=yuv_folder)

    # Configura objeto VideoParams
    video.project = output_folder
    video.factor = config.factor
    video.decoder = 'ffmpeg'
    video.threads = 'single'
    video.quality_list = config.quality_list
    video.dectime_base = f'dectime_{video.decoder}'

    # para cada video, para cada fmt, para cada qualidadae... decodificar 3
    # vezes todos os chunks de todos os tiles.
    for video.name in config.videos_list:
        for video.tile_format in config.tile_list:
            for video.rodada in range(3):
                for video.quality in video.quality_list:
                    util.decode(video=video)


if __name__ == '__main__':
    main()
