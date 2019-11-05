#!/bin/env python3
from utils import util

config = util.Config('config.json', factor='scale')
sl = util.check_system()['sl']

yuv_input = f'..{sl}yuv-full'

server = False
if server:
    gpds = f'{sl}mnt{sl}ssd{sl}henrique{sl}'
else:
    gpds = ''

output = (f'{gpds}results{sl}ffmpeg_'
          f'{config.factor}_{len(config.videos_list)}videos_'
          f'{config.duration}s_scale')


def main():
    encode()


def encode():
    # Configure objetcts

    # Create video object and your main folders
    video = util.VideoParams(config=config,
                             yuv=yuv_input)

    # Set basic configuration
    video.encoder = 'ffmpeg'
    video.project = output

    # iterate over 3 factors: video (complexity), tiles format, quality
    for video.name in config.videos_list:
        for video.tile_format in config.tile_list:
            for video.quality in getattr(config, f'{video.factor}_list'):
                util.encode(video)
                # util.encapsule(video)
                # util.extract_tile(video)
                util.make_segments(video)


if __name__ == '__main__':
    main()
