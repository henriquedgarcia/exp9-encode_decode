#!/bin/env python3
from utils import util

cfg = util.Config('config.json', factor='crf')
sl = cfg.sl
project = (f'ffmpeg_{cfg.factor}_{len(cfg.videos_list)}videos_'
           f'{cfg.duration}s')

yuv_input = f'..{sl}yuv-full'

server = False
if server:
    gpds = f'{sl}mnt{sl}ssd{sl}henrique{sl}'
else:
    gpds = ''

output = f'{gpds}results{sl}{project}'


def main():
    encode()


def encode():
    # Create video object and your main folders
    video = util.VideoParams(config=cfg,
                             yuv=yuv_input)

    # Set basic configuration
    video.encoder = 'ffmpeg'
    video.project = output

    # iterate over 3 factors: video (complexity), tiles format, quality
    for video.name in cfg.videos_list:
        for video.tile_format in cfg.tile_list:
            for video.quality in getattr(cfg, f'{video.factor}_list'):
                util.encode(video)
                # util.encapsule(video)
                # util.extract_tile(video)
                util.make_segments(video)


if __name__ == '__main__':
    main()
