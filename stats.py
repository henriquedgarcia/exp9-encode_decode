#!/bin/python3
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import util

sl = util.check_system()['sl']


def main():
    # stats()
    graph1()
    # graph2()
    # graph3()
    # graph4()
    # hist()
    pass


def hist():
    """
    Fazer um histograma para cada fator que estamos avaliando
    qualidade: 2000 kbps, 24000 kbps
    fmt: 1x1, 3x2, 6x4
    video: om_nom e rollercoaster
    Total: 2 x 3 x 2 = 12 histogramas

    :return:
    """
    pass


def graph4():
    """
    Este plot compara tile a tile a taxa e o tempo de decodificação para diferentes qualidades.

    :return:
    """
    config = util.Config('Config.json')
    dectime = util.load_json('times.json')

    dirname = f'results{sl}graph4'
    os.makedirs(f'{dirname}', exist_ok=True)

    for fmt in config.tile_list:
        m, n = list(map(int, fmt.split('x')))

        for tile in range(1, m * n + 1):
            times = util.AutoDict()
            sizes = util.AutoDict()
            times_a_ld = []
            times_a_hd = []
            sizes_a_ld = []
            sizes_a_hd = []
            times_b_ld = []
            times_b_hd = []
            sizes_b_ld = []
            sizes_b_hd = []

            # for name in config.videos_list:
            #     for quality in config.rate_list:
            #         t = []
            #         s = []
            #         for chunk in range(1, config.duration + 1):
            #             t.append(dectime['ffmpeg'][name][fmt]['rate'][str(quality)][str(tile)][str(chunk)]['single']['times']['ut'])
            #             s.append(dectime['ffmpeg'][name][fmt]['rate'][str(quality)][str(tile)][str(chunk)]['single']['size'])
            #         times[name][str(quality)] = t
            #         times[name][str(quality)] = s

            for chunk in range(1, config.duration + 1):
                times_a_ld.append(dectime['ffmpeg']['om_nom'][fmt]['rate'][str(2000000)][str(tile)][str(chunk)]['single']['times'][0])
                # sizes_a_ld.append(dectime['ffmpeg']['om_nom'][fmt]['rate'][str(2000000)][str(tile)][str(chunk)]['single']['size'])
                times_a_hd.append(dectime['ffmpeg']['om_nom'][fmt]['rate'][str(24000000)][str(tile)][str(chunk)]['single']['times'][0])
                # sizes_a_hd.append(dectime['ffmpeg']['om_nom'][fmt]['rate'][str(24000000)][str(tile)][str(chunk)]['single']['size'])

                times_b_ld.append(dectime['ffmpeg']['rollercoaster'][fmt]['rate'][str(2000000)][str(tile)][str(chunk)]['single']['times'][0])
                # sizes_b_ld.append(dectime['ffmpeg']['rollercoaster'][fmt]['rate'][str(2000000)][str(tile)][str(chunk)]['single']['size'])
                times_b_hd.append(dectime['ffmpeg']['rollercoaster'][fmt]['rate'][str(24000000)][str(tile)][str(chunk)]['single']['times'][0])
                # sizes_b_hd.append(dectime['ffmpeg']['rollercoaster'][fmt]['rate'][str(24000000)][str(tile)][str(chunk)]['single']['size'])

            # a = plt.Axes()
            plt.close()
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=100)
            ax[0].hist(times_a_ld, bins=10, histtype='step', label=f'Om_non_{fmt}_rate2000000')
            ax[0].hist(times_a_hd, bins=10, histtype='step', label=f'Om_non_{fmt}_rate24000000')
            ax[0].hist(times_b_ld, bins=10, histtype='step', label=f'rollercoaster_{fmt}_rate2000000')
            ax[0].hist(times_b_hd, bins=10, histtype='step', label=f'rollercoaster_{fmt}_rate24000000')
            ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            ax[0].set_title(f'Tile {tile}')
            ax[0].set_xlabel('Times')
            ax[0].set_ylabel("Occurrence")

            ax[1].hist(times_a_ld, bins=10, density=True, cumulative=True, histtype='step', label=f'Om_non_{fmt}_rate2000000')
            ax[1].hist(times_a_hd, bins=10, density=True, cumulative=True, histtype='step', label=f'Om_non_{fmt}_rate24000000')
            ax[1].hist(times_b_ld, bins=10, density=True, cumulative=True, histtype='step', label=f'rollercoaster_{fmt}_rate2000000')
            ax[1].hist(times_b_hd, bins=10, density=True, cumulative=True, histtype='step', label=f'rollercoaster_{fmt}_rate24000000')
            ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            ax[1].set_xlabel('Times')
            ax[1].set_ylabel("CDF")
            plt.tight_layout()
            plt.savefig(f'{dirname}{sl}hist_{fmt}_tile{tile}')
            # plt.show()
            print(f'hist_{fmt}_tile{tile}')

            # plt.hist(times, bins=20)
            plt.close()
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=100)
            ax[0].bar(np.array(range(len(times_a_ld))) - 0.3, times_a_ld, width=0.2, label=f'om_nom-{fmt}-rate{2000000}')
            ax[0].bar(np.array(range(len(times_a_hd))) - 0.1, times_a_hd, width=0.2, label=f'om_nom-{fmt}-rate{24000000}')
            ax[0].bar(np.array(range(len(times_b_ld))) + 0.1, times_b_ld, width=0.2, label=f'rollercoaster-{fmt}-rate{2000000}')
            ax[0].bar(np.array(range(len(times_b_hd))) + 0.3, times_b_hd, width=0.2, label=f'rollercoaster-{fmt}-rate{24000000}')
            ax[0].set_title(f'Tile {tile} - Atrasos')
            ax[0].set_ylabel("Time")

            ax[1].plot(sizes_a_ld, label=f'om_nom-{fmt}-rate{2000000}')
            ax[1].plot(sizes_a_hd, label=f'om_nom-{fmt}-rate{24000000}')
            ax[1].plot(sizes_b_ld, label=f'rollercoaster-{fmt}-rate{2000000}')
            ax[1].plot(sizes_b_hd, label=f'rollercoaster-{fmt}-rate{24000000}')
            ax[1].set_title(f'Tile {tile} - Taxas')
            ax[1].set_xlabel("Chunk")
            ax[1].set_ylabel("Time")

            ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

            plt.tight_layout()
            plt.savefig(f'{dirname}{sl}graph_{fmt}_tile{tile}')

            # plt.show()
            print(f'graph_{fmt}_tile{tile}')


def graph3() -> None:
    """
    bar
    fmt X average_dec_time (seconds) and fmt X average_rate (Bytes)
    :return: None
    """
    dirname = 'graph3'

    config = util.Config('config.json')
    dectime = util.load_json('times.json')

    # decoders = ['ffmpeg', 'mp4client']
    factors = ['rate']
    threads = ['single']

    # for decoder in decoders:
    for name in config.videos_list:
        for factor in factors:

            for thread in threads:
                df = pd.DataFrame()
                plt.close()
                fig, ax = plt.subplots(2, 1, figsize=(8, 5))
                quality_list = getattr(config, f'{factor}_list')
                offset = 0
                for quality in quality_list:
                    average_size = []
                    std_size = []
                    average_time = []
                    std_time = []
                    width = 0.8 / len(quality_list)
                    start_position = (0.8 - width) / 2

                    for fmt in config.tile_list:
                        m, n = list(map(int, fmt.split('x')))
                        size = []
                        time = []

                        for tile in range(1, m * n + 1):
                            for chunk in range(1, config.duration + 1):
                                size.append(dectime['ffmpeg'][name][fmt][factor][str(quality)][str(tile)][str(chunk)][thread]['size'])
                                time.append(dectime['ffmpeg'][name][fmt][factor][str(quality)][str(tile)][str(chunk)][thread]['times']['ut'])

                        average_size.append(np.average(size))
                        std_size.append(np.std(size))
                        average_time.append(np.average(time))
                        std_time.append(np.std(time))

                    x = np.array(range(1, len(average_time) + 1)) - start_position + offset
                    offset += width
                    ax[0].bar(x, average_time, width=width, yerr=std_time, label=f'rate_total={quality}')
                    ax[1].bar(x, average_size, width=width, yerr=std_size, label=f'rate_total={quality}')

                    df[f'times_{name}_{quality}'] = average_time

                ax[0].set_xticklabels(config.tile_list)
                ax[0].set_xticks(np.array(range(1, len(config.tile_list) + 1)))
                ax[1].set_xticklabels(config.tile_list)
                ax[1].set_xticks(np.array(range(1, len(config.tile_list) + 1)))

                ax[0].set_xlabel('Tile')
                ax[1].set_xlabel('Tile')
                ax[0].set_ylabel('Average Time')
                ax[1].set_ylabel('Average Rate')
                ax[0].set_title(f'{name} - Times by tiles, {factor}')
                ax[1].set_title(f'{name} - Rates by tiles, {factor}')
                ax[0].set_ylim(bottom=0)
                ax[1].set_ylim(bottom=0)
                ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
                ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
                plt.tight_layout()
                os.makedirs(dirname, exist_ok=True)
                print(f'Salvando {dirname}{sl}{name}_{factor}.')
                fig.savefig(f'{dirname}{sl}{name}_{factor}')
                # plt.show()
                1


def graph2() -> None:
    """
    bar
    tile X average_dec_time (seconds) and tile X average_rate (Bytes)
    :return: None
    """
    dirname = f'results{sl}graph2'
    os.makedirs(dirname, exist_ok=True)

    config = util.Config('config.json')
    dectime = util.load_json('times.json')

    # decoders = ['ffmpeg', 'mp4client']
    factors = ['crf']
    threads = ['single']

    # for decoder in decoders:
    for name in config.videos_list:
        for factor in factors:

            for thread in threads:
                for fmt in config.tile_list:
                    m, n = list(map(int, fmt.split('x')))
                    plt.close()
                    fig, ax = plt.subplots(2, 1, figsize=(10, 5))

                    quality_list = getattr(config, f'{factor}_list')
                    offset = 0
                    for quality in quality_list:
                        # average_size = []
                        # std_size = []
                        average_time = []
                        std_time = []

                        width = 0.8 / len(quality_list)
                        start_position = (0.8 - width) / 2

                        for tile in range(1, m * n + 1):
                            # size = []
                            time = []

                            for chunk in range(1, config.duration + 1):
                                # size.append(dectime['ffmpeg'][name][fmt][factor][str(quality)][str(tile)][str(chunk)][thread]['size'])
                                time.append(dectime['ffmpeg'][name][fmt][factor][str(quality)][str(tile)][str(chunk)][thread]['times'][0])

                            # average_size.append(np.average(size))
                            # std_size.append(np.std(size))
                            average_time.append(np.average(time))
                            std_time.append(np.std(time))

                        x = np.array(range(1, len(average_time) + 1)) - start_position + offset
                        offset += width

                        if factor in 'rate':
                            quality = int(quality / (m * n))

                        # ax[0].bar(x, average_time, width=width, yerr=std_time, label=f'quality={quality}_corr={np.corrcoef(x=(average_time, average_size))[1][0]}')
                        ax[0].bar(x, average_time, width=width, yerr=std_time, label=f'quality={quality}_corr=0')
                        # ax[1].bar(x, average_size, width=width, yerr=std_size, label=f'quality={quality}_ffmpeg')

                    ax[0].set_xlabel('Tile')
                    ax[1].set_xlabel('Tile')
                    ax[0].set_ylabel('Average Time')
                    ax[1].set_ylabel('Average Rate')
                    ax[0].set_title(f'{name} - Times by tiles, tile={fmt}, {factor}')
                    ax[1].set_title(f'{name} - Rates by tiles, tile={fmt}, {factor}')
                    ax[0].set_ylim(bottom=0)
                    ax[1].set_ylim(bottom=0)
                    ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
                    ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
                    plt.tight_layout()
                    os.makedirs(dirname, exist_ok=True)
                    print(f'Salvando {dirname}{sl}{name}_{fmt}_{factor}.')
                    fig.savefig(f'{dirname}{sl}{name}_{fmt}_{factor}')
                    # plt.show()
                    # print('1')


def graph1() -> None:
    """
    chunks X dec_time (seconds) and chunks X file_size (Bytes)
    :return:
    """
    dirname = f'results{sl}graph1'
    os.makedirs(dirname, exist_ok=True)

    config = util.Config('config.json')
    dectime = util.load_json('times.json')

    # decoders = ['ffmpeg', 'mp4client']
    factors = ['crf']
    threads = ['single']

    # for decoder in decoders:
    for name in config.videos_list:
        for factor in factors:
            for quality in getattr(config, f'{factor}_list'):
                quality = np.array(quality)
                for thread in threads:
                    for fmt in config.tile_list:
                        m, n = list(map(int, fmt.split('x')))
                        plt.close()
                        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

                        for tile in range(1, m * n + 1):
                            size = []
                            time_ffmpeg = []
                            # time_mp4client = []

                            for chunk in range(1, config.duration + 1):
                                # size.append(dectime['ffmpeg'][name][fmt][factor][str(quality)][str(tile)][str(chunk)][thread]['size'])
                                time_ffmpeg.append(dectime['ffmpeg'][name][fmt][factor][str(quality)][str(tile)][str(chunk)][thread]['times'][0])
                                # time_mp4client.append(dectime['mp4client'][name][fmt][factor][str(quality)][str(tile)][str(chunk)][thread]['times'])

                            ax[0].plot(time_ffmpeg, label=f'ffmpeg_tile={tile}_ffmpeg')
                            # ax[0][1].plot(time_mp4client, label=f'tile={tile}')
                            ax[1].plot(size, label=f'tile={tile}')
                            # ax[1][1].plot(time_ffmpeg, label=f'ffmpeg_tile={tile}_ffmpeg')
                            # ax[1][1].plot(time_mp4client, label=f'mp4client_tile={tile}_mp4client')

                        quality_ind = quality
                        if factor in 'rate':
                            quality_ind = int(quality / (m * n))

                        ax[0].set_xlabel('Chunks')
                        # ax[0][1].set_xlabel('Chunks')
                        ax[1].set_xlabel('Chunks')
                        # ax[1][1].set_xlabel('Chunks')
                        ax[0].set_ylabel('Time')
                        # ax[0][1].set_ylabel('Time')
                        # ax[1][1].set_ylabel('Time')
                        ax[1].set_ylabel('Rate')
                        ax[0].set_title(f'ffmpeg - {name} - Times by chunks, tile={fmt}, {factor}={quality_ind}')
                        # ax[0][1].set_title(f'mp4client {name} - Times by chunks, tile={fmt}, {factor}={quality_ind}')
                        ax[1].set_title(f'{name} - Rates by chunks, tile={fmt}, {factor}={quality_ind}')
                        # ax[1][1].set_title(f'mp4client x ffmpeg - {name} - Times by chunks, tile={fmt}, {factor}={quality_ind}')
                        # ax[0].set_ylim(bottom=0)
                        # ax[1].set_ylim(bottom=0)
                        ax[1].set_ylim(bottom=0)
                        # ax[1][1].set_ylim(bottom=0)
                        # ax[0][1].legend(loc='upper left', ncol=2, bbox_to_anchor=(1.01, 1.0))
                        ax[1].legend(loc='upper left', ncol=2, bbox_to_anchor=(1.01, 1.0))
                        plt.tight_layout()
                        # plt.()
                        print(f'Salvando {dirname}{sl}{name}_{fmt}_{factor}={quality_ind}.')
                        # fig.savefig(f'{dirname}{sl}{name}_{fmt}_{factor}={quality_ind}')
                        fig.show()
                        print('')





    for factors in product(decoders, videos_list, tile_list, q_factors, multithreads):
        video_seg.decoder = factors[0]
        video_seg.name = factors[1]
        video_seg.fmt = factors[2]
        video_seg.factor = factors[3]
        video_seg.multithread = factors[4]
        video_seg.dectime_base = f'dectime_{video_seg.decoder}'

        video_seg.quality_list = getattr(config, f'{video_seg.factor}_list')

        for video_seg.quality in video_seg.quality_list:
            times = util.collect_data(video_seg=video_seg)

    util.save_json(times, 'times.json')

    # graph_chunk_X_time_X_tile_rate()
    # graph2()
    # graph3()
def stats():
    # Configura os objetos
    config = util.Config('config.json')

    # Base object
    video_seg = util.Video(config=config)
    video_seg.project = 'results/ffmpeg_crf_18videos_60s'
    video_seg.factor = 'crf'
    video_seg.segment_base = 'segment'
    video_seg.dectime_base = f'dectime_ffmpeg'
    video_seg.bench_stamp = 'bench: utime'
    video_seg.multithread = 'single'

    video_seg.quality_list = config.crf_list

    for (video_seg.name, video_seg.fmt, video_seg.quality) in product(config.videos_list, config.tile_list, video_seg.quality_list):
        print(f'Processing {video_seg.basename}.txt')

        for video_seg.tile in range(1, video_seg.num_tiles + 1):
            for video_seg.chunk in range(1, video_seg.duration * video_seg.fps + 1):

                # Processing segment size (bytes)
                if os.path.isfile(f'{video_seg.segment_path}.mp4'):
                    video_seg.size = os.path.getsize(f'{video_seg.segment_path}.mp4')

                # Processing decoding time
                if os.path.isfile(f'{video_seg.log_path}.txt'):
                    times = []
                    with open(f'{video_seg.log_path}.txt', 'r') as f:
                        for line in f:
                            if line.find(video_seg.bench_stamp) >= 0:
                                line = line.replace('bench: ', ' ')  # Troca "bench" por " "
                                line = line.replace('s ', ' ')  # Troca "s " por " "
                                line = line.strip()[:-1]  # Remove o último char
                                line = line.split(' ')  # Faz o split com " "
                                for i in range(0, len(line), 3):
                                    times.append(float(line[i][6:]))

                    video_seg.times = np.average(times)

    util.save_json(dict(video_seg.dectime), 'times2.json')


if __name__ == "__main__":
    main()
