#!/bin/python3
import itertools.product as it:

from utils import util


def main():
    # stats()
    graph()


def graph():
    config = util.Config('config.json')
    times = util.load_json('times.json')
    video_graph = util.VideoSegment(config=config, dectime=times)

    decoder = ['ffmpeg', 'mp4client']
    factor = ['rate', 'qp']
    multithread = ['multi', 'single']

    for (video_graph.decoder,
         video_graph.name,
         video_graph.fmt,
         video_graph.factor,
         video_graph.multithread) in it(decoder,
                                        config.videos_list,
                                        config.tile_list,
                                        factor,
                                        multithread):

        # Ignore
        if video_graph.name not in ('om_nom',
                                    'lions',
                                    'pac_man',
                                    'rollercoaster'): continue

        for video_graph.quality in getattr(config, f'{video_graph.factor}_list'):
            for tiles in range(1, video_graph.num_tiles + 1):
                video_graph.tile = tiles
                for chunks in range(1, video_graph.duration * video_graph.fps + 1):
                    video_graph.chunk = chunks

                    size = video_graph.size
                    ut = video_graph.times['ut']
                    st = video_graph.times['st']
                    rt = video_graph.times['rt']


def stats():
    # Configura os objetos
    config = util.Config('config.json')

    # Base object
    video_seg = util.VideoSegment(config=config)
    video_seg.project = 'ffmpeg'
    video_seg.segment_base = 'segment'

    # To iterate
    decoders = ['ffmpeg', 'mp4client']
    videos_list = config.videos_list
    tile_list = config.tile_list
    q_factors = ['rate', 'qp']
    multithreads = [False, True]

    for factors in it(decoders, videos_list, tile_list, q_factors, multithreads):
        video_seg.decoder = factors[0]
        video_seg.name = factors[1]
        video_seg.fmt = factors[2]
        video_seg.factor = factors[3]
        video_seg.multithread = factors[4]

        # Ignore
        if video_seg.name not in ('om_nom', 'lions', 'pac_man', 'rollercoaster'):
            continue

        for video_seg.quality in getattr(config, f'{video_seg.factor}_list'):
            times = util.collect_data(video_seg=video_seg)

    util.save_json(times, 'times.json')


if __name__ == "__main__":
    main()
