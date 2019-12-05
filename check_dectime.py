import os
from itertools import product as pd

from utils import util

cfg = util.Config('config.json', factor='crf')
sl = cfg.sl
cfg.project = 'ffmpeg_crf_12videos_60s'
path = f'results{sl}{cfg.project}{sl}dectime_ffmpeg'


def main():
    lines = []

    for name, quality, fmt in pd(
            cfg.videos_list,
            cfg.quality_list,
            cfg.tile_list
    ):
        m, n = list(map(int, fmt.split('x')))
        for tile, chunk in pd(
                range(1, m * n + 1),
                range(1, 61)
        ):
            base = (f'{name}_'
                    f'{cfg.scale}_'
                    f'{cfg.fps}_'
                    f'{fmt}_'
                    f'{cfg.factor}{quality}')
            file = f'tile{tile}_{chunk:03}_single.txt'
            # file = f'tile{tile}_{chunk:03}.mp4'
            # file = f'tile{tile}.mp4'
            file_path = f'{path}{sl}{base}{sl}{file}'
            msg = None

            try:
                if os.path.getsize(f'{file_path}') == 0:
                    msg = f'size error:\t{file_path} == 0'
                    # try:
                    #     os.remove(f'{file_path}')
                    # except:
                    #     pass
            except FileNotFoundError:
                msg = f'file error:\t{file_path} not found'

            if msg:
                print(msg)
                lines.append(f'{msg}\n')

    with open('check_dectime.log', 'w', encoding='utf-8') as f:
        f.writelines(lines, )


if __name__ == '__main__':
    main()
