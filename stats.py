#!/bin/python3
import itertools
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import util

sl = util.check_system()['sl']


def main():
    # stats()
    # graph1()
    # graph1_a()
    # graph2()
    # graph2a()
    # graph3()
    # hist1()
    hist1samefig()
    hist1sameplt()
    # hist2samefig()
    # hist2sameplt()


def hist2samefig():
    """
    Compara os histogramas. Para cada video-quality plota todas os fmts (agrega tiles e chunks)
    :return:
    """
    config = util.Config('Config.json')
    dectime = util.load_json('times2.json')

    dirname = f'results{sl}{"hist2samefig"}'
    os.makedirs(f'{dirname}', exist_ok=True)

    for quality in config.crf_list:
        for name in config.single_videos_list:
            times = {}  # Lim superior
            sizes = {}

            for fmt in config.tile_list:
                m, n = list(map(int, fmt.split('x')))
                times[fmt] = []
                sizes[fmt] = []
                for tile in range(1, m * n + 1):
                    for chunk in range(1, config.duration + 1):
                        if name in 'ninja_turtles' and chunk > 58:
                            continue
                        times[fmt].append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times'])
                        sizes[fmt].append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size'])
                        # plt.hist

            plt.close()
            fig = plt.figure(figsize=(12, 6), dpi=100)

            it = enumerate(times, 1)
            color = iter(['blue', 'orange', 'green', 'red'])

            idx, fmt = next(it)
            ax = fig.add_subplot(3, 2, idx)
            ax.hist(times[fmt], color=next(color), bins=50, histtype='step', label=(f'Avg={np.average(times[fmt]):.03f}\n'
                                                                                    f'std={np.std(times[fmt]):.03f}'))
            ax.set_title(f'{name}, crf {quality}, {fmt}')
            ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

            for idx, fmt in it:
                ax = fig.add_subplot(3, 2, idx, sharex=ax)
                ax.set_title(f'{name}, crf {quality}, {fmt}')
                ax.hist(times[fmt], color=next(color), bins=50, histtype='step', label=(f'Avg={np.average(times[fmt]):.03f}\n'
                                                                                        f'std={np.std(times[fmt]):.03f}'))
                if idx in [1, 2]:
                    ax.set_ylabel("PDF")
                if idx in [3, 4]:
                    ax.set_xlabel('Times')
                # if n in [2, 4]:
                ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

            color = iter(['blue', 'orange', 'green', 'red'])
            ax = fig.add_subplot(3, 1, 3)
            for fmt in times:
                ax.hist(times[fmt], color=next(color), bins=50, density=True, cumulative=True, histtype='step', label=f'{name}_{fmt}_crf{quality}')
                ax.set_ylabel("CDF")
                ax.set_xlabel("Decoder Times")
                ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

            ''' histtype:
            ‘bar’ is a traditional bar-type histogram. If multiple data are given the bars are aranged side by side.
            ‘barstacked’ is a bar-type histogram where multiple data are stacked on top of each other.
            ‘step’ generates a lineplot that is by default unfilled.
            ‘stepfilled’ generates a lineplot that is by default filled.'''

            plt.tight_layout()

            # plt.savefig(f'{dirname}{sl}hist_{name}_{quality}')
            plt.show()
            print(f'hist_{name}_{quality}')


def hist1sameplt():
    """
    Compara os histogramas. Para cada video-fmt plota todas as qualidades (agrega tiles e chunks)
    :return:
    """
    config = util.Config('Config.json')
    dectime = util.load_json('times2.json')
    color_list = ['blue', 'orange', 'green', 'red']
    dirname = f'results{sl}{"hist1sameplt"}'
    os.makedirs(f'{dirname}', exist_ok=True)

    for fmt in config.tile_list:
        m, n = list(map(int, fmt.split('x')))

        for name in config.single_videos_list:
            times = {}  # Lim superior
            sizes = {}

            for quality in config.crf_list:
                times[quality] = []
                sizes[quality] = []
                for tile in range(1, m * n + 1):
                    for chunk in range(1, config.duration + 1):
                        if name in 'ninja_turtles' and chunk > 58:
                            continue
                        times[quality].append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times'])
                        sizes[quality].append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size'])
                        # plt.hist

            plt.close()
            fig = plt.figure(figsize=(12, 8), dpi=100)

            # O plot da PDF
            color = iter(color_list)
            ax = fig.add_subplot(2, 1, 1)
            ax.set_title(f'{name}, {fmt}')
            for idx, quality in enumerate(times, 1):
                label = (f'crf{quality},\n'
                         f'Avg={np.average(times[quality]):.03f}\n'
                         f''f'std={np.std(times[quality]):.03f}')
                ax.hist(times[quality], color=next(color), bins=50, histtype='step', label=label)
                ax.set_ylabel("PDF")
                ax.set_xlabel('Times')
                ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

            # O plot da CDF
            color = iter(['blue', 'orange', 'green', 'red'])
            ax = fig.add_subplot(2, 1, 2)
            ax.set_ylabel("CDF")
            ax.set_xlabel("Decoder Times")
            for quality in times:
                ax.hist(times[quality], color=next(color), bins=50, density=True, cumulative=True, histtype='step', label=f'crf{quality}')
            ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

            ''' histtype:
            ‘bar’ is a traditional bar-type histogram. If multiple data are given the bars are aranged side by side.
            ‘barstacked’ is a bar-type histogram where multiple data are stacked on top of each other.
            ‘step’ generates a lineplot that is by default unfilled.
            ‘stepfilled’ generates a lineplot that is by default filled.'''

            plt.tight_layout()

            plt.savefig(f'{dirname}{sl}hist_{name}_{fmt}')
            # plt.show()
            print(f'hist_{name}_{fmt}')


def hist1samefig():
    """
    Compara os histogramas. Para cada video-fmt plota todas as qualidades (agrega tiles e chunks)
    :return:
    """
    config = util.Config('Config.json')
    dectime = util.load_json('times2.json')
    color_list = ['blue', 'orange', 'green', 'red']
    dirname = f'results{sl}{"hist1samefig"}'

    os.makedirs(f'{dirname}', exist_ok=True)

    for fmt in config.tile_list:
        m, n = list(map(int, fmt.split('x')))

        for name in config.single_videos_list:
            times = {}  # Lim superior
            sizes = {}

            for quality in config.crf_list:
                times[quality] = []
                sizes[quality] = []
                for tile in range(1, m * n + 1):
                    for chunk in range(1, config.duration + 1):
                        if name in 'ninja_turtles' and chunk > 58:
                            continue
                        times[quality].append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times'])
                        sizes[quality].append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size'])
                        # plt.hist

            plt.close()
            fig = plt.figure(figsize=(12, 6), dpi=100)
            color = iter(color_list)
            ax = None

            for idx, quality in enumerate(times, 1):
                label = (f'dectime_avg={np.average(times[quality]):.03f}\n'
                         f'dectime_std={np.std(times[quality]):.03f}\n'
                         f'rate_avg={np.average(sizes[quality]) * 8 / 1000:,.0f} Kbps\n'
                         f'rate_std={np.std(sizes[quality]) * 8 / 1000:,.0f} Kbps\n'
                         f'time/chunk={np.average(times[quality]) * (m * n):.03f}')
                if idx == 1:
                    ax = fig.add_subplot(3, 2, idx)
                else:
                    ax = fig.add_subplot(3, 2, idx, sharex=ax)

                ax.hist(times[quality],
                        color=next(color),
                        bins=50,
                        histtype='step',
                        label=label)

                ax.set_title(f'{name}, {fmt}, crf {quality}')
                ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

                if idx in [1, 3]:
                    ax.set_ylabel("PDF")
                if idx in [3, 4]:
                    ax.set_xlabel('Times')

            color = iter(color_list)
            ax = fig.add_subplot(3, 1, 3)
            for quality in times:
                ax.hist(times[quality], color=next(color), bins=50, density=True, cumulative=True, histtype='step', label=f'{name}_{fmt}_crf{quality}')
                ax.set_ylabel("CDF")
                ax.set_xlabel("Decoder Times")
                ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

            ''' histtype:
            ‘bar’ is a traditional bar-type histogram. If multiple data are given the bars are aranged side by side.
            ‘barstacked’ is a bar-type histogram where multiple data are stacked on top of each other.
            ‘step’ generates a lineplot that is by default unfilled.
            ‘stepfilled’ generates a lineplot that is by default filled.'''

            plt.tight_layout()

            plt.savefig(f'{dirname}{sl}hist_{name}_{fmt}')
            # plt.show()
            print(f'hist_{name}_{fmt}')


def hist1():
    """
    Faz os histogramas e agrega por qualidade
    :return:
    """

    config = util.Config('Config.json')
    dectime = util.load_json('times2.json')

    dirname = f'results{sl}hist1'
    os.makedirs(f'{dirname}', exist_ok=True)

    for fmt in config.tile_list:
        m, n = list(map(int, fmt.split('x')))

        for name in config.single_videos_list:
            times = []  # Lim superior
            sizes = []
            plt.close()
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=100)
            for quality in config.crf_list:
                for tile in range(1, m * n + 1):
                    for chunk in range(1, config.duration + 1):
                        if name in 'ninja_turtles' and chunk > 58:
                            continue
                        times.append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times'])
                        sizes.append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size'])

                ax[0].hist(times, bins=50, histtype='step', label=f'{name}_{fmt}_crf{quality}')
                ax[1].hist(sizes, bins=50, density=True, cumulative=True, histtype='step', label=f'{name}_{fmt}_crf{quality}')

                ax[0].set_title(f'Video {name}, quality {quality}')
                ax[0].set_xlabel('Times')
                ax[0].set_ylabel("Occurrence")
                ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

                ax[1].set_xlabel('Times')
                ax[1].set_ylabel("CDF")
                ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

                plt.tight_layout()
                plt.savefig(f'{dirname}{sl}hist_{name}_{fmt}_quality{quality}')
                # plt.show()
                print(f'{dirname}{sl}hist_{name}_quality{quality}')

            # plt.hist(times, bins=20)
            # plt.close()
            # fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=100)
            # ax[0].bar(np.array(range(len(times_a_ld))) - 0.3, times_a_ld, width=0.2, label=f'om_nom-{fmt}-rate{2000000}')
            # ax[0].bar(np.array(range(len(times_a_hd))) - 0.1, times_a_hd, width=0.2, label=f'om_nom-{fmt}-rate{24000000}')
            # ax[0].bar(np.array(range(len(times_b_ld))) + 0.1, times_b_ld, width=0.2, label=f'rollercoaster-{fmt}-rate{2000000}')
            # ax[0].bar(np.array(range(len(times_b_hd))) + 0.3, times_b_hd, width=0.2, label=f'rollercoaster-{fmt}-rate{24000000}')
            # ax[0].set_title(f'Tile {tile} - Atrasos')
            # ax[0].set_ylabel("Time")
            #
            # ax[1].plot(sizes_a_ld, label=f'om_nom-{fmt}-rate{2000000}')
            # ax[1].plot(sizes_a_hd, label=f'om_nom-{fmt}-rate{24000000}')
            # ax[1].plot(sizes_b_ld, label=f'rollercoaster-{fmt}-rate{2000000}')
            # ax[1].plot(sizes_b_hd, label=f'rollercoaster-{fmt}-rate{24000000}')
            # ax[1].set_title(f'Tile {tile} - Taxas')
            # ax[1].set_xlabel("Chunk")
            # ax[1].set_ylabel("Time")
            #
            # ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            # ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            #
            # plt.tight_layout()
            # # plt.savefig(f'{dirname}{sl}graph_{fmt}_tile{tile}')
            # plt.show()
            # print(f'graph_{fmt}_tile{tile}')


def graph3() -> None:
    """
    bar
    fmt X average_dec_time (seconds) and fmt X average_rate (Bytes)
    :return: None
    """
    dirname = f'results{sl}graph3'
    os.makedirs(dirname, exist_ok=True)

    config = util.Config('config.json')
    dectime = util.load_json('times2.json')
    factor = 'crf'

    for name in config.single_videos_list:
        plt.close()
        fig, ax = plt.subplots(2, 1, figsize=(8, 5))
        quality_list = getattr(config, f'{factor}_list')
        offset = 0
        for quality in config.crf_list:
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

                # for tile, m, n in enumerate(itertools.product(range(1, m_ + 1), range(1, n_ + 1))):
                for tile in range(1, m * n + 1):
                    for chunk in range(1, config.duration + 1):
                        if name in 'ninja_turtles' and chunk > 58:
                            continue
                        size.append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size'])
                        time.append(dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times'])

                average_size.append(np.average(size))
                std_size.append(np.std(size))
                average_time.append(np.average(time))
                std_time.append(np.std(time))

            x = np.array(range(1, len(average_time) + 1)) - start_position + offset
            offset += width
            ax[0].bar(x, average_time, width=width, yerr=std_time, label=f'rate_total={quality}')
            ax[1].bar(x, average_size, width=width, yerr=std_size, label=f'rate_total={quality}')

            # df[f'times_{name}_{quality}'] = average_time

        ax[0].set_xticklabels(config.tile_list)
        ax[0].set_xticks(np.array(range(1, len(config.tile_list) + 1)))
        ax[1].set_xticklabels(config.tile_list)
        ax[1].set_xticks(np.array(range(1, len(config.tile_list) + 1)))

        ax[0].set_xlabel('Tile')
        ax[1].set_xlabel('Tile')
        ax[0].set_ylabel('Average Time/Tile')
        ax[1].set_ylabel('Average Rate/Tile')
        ax[0].set_title(f'{name} - Times by tiles, {factor}')
        ax[1].set_title(f'{name} - Rates by tiles, {factor}')
        ax[0].set_ylim(bottom=0)
        ax[1].set_ylim(bottom=0)
        ax[1].ticklabel_format(style='sci', axis='Y', scilimits=(6, 6))
        ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
        ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
        plt.tight_layout()
        print(f'Salvando {dirname}{sl}{name}_{factor}.')
        fig.savefig(f'{dirname}{sl}{name}_{factor}')
        # plt.show()
        print('')


def graph2() -> None:
    """
    bar
    tile X average_dec_time (seconds) and tile X average_rate (Bytes)
    :return: None
    """
    dirname = f'results{sl}graph2-2'
    os.makedirs(dirname, exist_ok=True)

    config = util.Config('config.json')
    dectime = util.load_json('times2.json')

    for name in config.single_videos_list:
        for fmt in config.tile_list:
            m, n = list(map(int, fmt.split('x')))

            plt.close()
            fig, ax = plt.subplots(2, 1, figsize=(10, 5))
            offset = 0
            width = 0.8 / len(config.crf_list)
            start_position = (0.8 - width) / 2

            for quality in config.crf_list:
                average_time = []
                std_time = []
                average_size = []
                std_size = []

                for tile in range(1, m * n + 1):
                    size = []
                    time = []

                    for chunk in range(1, config.duration + 1):
                        if name in 'ninja_turtles' and chunk > 58:
                            continue
                        s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
                        t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
                        size.append(float(s) * 8)
                        time.append(float(t))

                    average_size.append(np.average(size))
                    average_time.append(np.average(time))
                    std_size.append(np.std(size))
                    std_time.append(np.std(time))

                x = np.array(range(1, len(average_time) + 1)) - start_position + offset
                ax[0].bar(x, average_time, width=width, yerr=std_time, label=f'quality={quality}_corr={np.corrcoef(x=(average_time, average_size))[1][0]:.3f}')
                ax[1].bar(x, average_size, width=width, yerr=std_size, label=f'quality={quality}')
                offset += width

            ax[0].set_xlabel('Tile')
            ax[1].set_xlabel('Tile')
            ax[0].set_ylabel('Average Time/Tile (s)')
            ax[1].set_ylabel('Average Rate/Tile (bps)')
            ax[0].set_title(f'{name} - Times by tiles, tile={fmt}')
            ax[1].set_title(f'{name} - Rates by tiles, tile={fmt}')
            ax[0].set_ylim(bottom=0)
            ax[1].set_ylim(bottom=0)
            ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            plt.tight_layout()

            print(f'Salvando {dirname}{sl}{name}_{fmt}.')
            fig.savefig(f'{dirname}{sl}{name}_{fmt}')
            # plt.show()
            print('')


def graph2a() -> None:
    """
    bar
    tile X average_dec_time (seconds) and tile X average_rate (Bytes)
    :return: None
    """
    dirname = f'results{sl}graph2-2a'
    os.makedirs(dirname, exist_ok=True)

    config = util.Config('config.json')
    dectime = util.load_json('times2.json')

    for name in config.single_videos_list:
        for fmt in config.tile_list:
            plt.close()
            fig = plt.figure(figsize=(10, 6))
            ax = [None] * 6
            ax[0] = fig.add_subplot(3, 1, 1)
            ax[1] = fig.add_subplot(3, 1, 2)
            ax[2] = fig.add_subplot(3, 4, 9)
            ax[3] = fig.add_subplot(3, 4, 10)
            ax[4] = fig.add_subplot(3, 4, 11)
            ax[5] = fig.add_subplot(3, 4, 12)

            # fig, ax = plt.subplots(2, 1, figsize=(10, 5))
            offset = 0
            width = 0.8 / len(config.crf_list)
            start_position = (0.8 - width) / 2
            for count, quality in enumerate(config.crf_list):

                average_time = []
                std_time = []
                average_size = []
                std_size = []

                m_, n_ = list(map(int, fmt.split('x')))
                frame1 = np.zeros((n_, m_))

                for m, n in itertools.product(range(1, m_ + 1), range(1, n_ + 1)):
                    tile = m * n
                    # Inicializa
                    size = []
                    time = []

                    # Opera
                    for chunk in range(1, config.duration + 1):
                        if name in 'ninja_turtles' and chunk > 58:
                            continue
                        s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
                        t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
                        size.append(float(s) * 8)
                        time.append(float(t))

                    # Anexa
                    average_size.append(np.average(size))
                    average_time.append(np.average(time))
                    std_size.append(np.std(size))
                    std_time.append(np.std(time))
                    frame1[n - 1, m - 1] = np.average(size)

                # Plota
                x = np.array(range(1, len(average_time) + 1)) - start_position + offset
                ax[0].bar(x, average_time, width=width, yerr=std_time, label=f'crf={quality}_corr={np.corrcoef(x=(average_time, average_size))[1][0]:.3f}')
                ax[1].bar(x, average_size, width=width, yerr=std_size, label=f'crf={quality}')
                ax[count + 2].pcolor(frame1, cmap='inferno')
                ax[count + 2].set_xlabel(f'crf={quality}')
                ax[count + 2].set_title('Heatmap')
                offset += width
                # a=count

            # plt.colorbar(ax=ax[a + 2])

            # Encerra Plot
            ax[0].set_xlabel('Tile')
            ax[1].set_xlabel('Tile')
            ax[0].set_ylabel('Average Time (s)')
            ax[1].set_ylabel('Average Rate (bps)')
            ax[0].set_title(f'{name} - Times by tiles, tile={fmt}')
            ax[1].set_title(f'{name} - Rates by tiles, tile={fmt}')
            ax[0].set_ylim(bottom=0)
            ax[1].set_ylim(bottom=0)
            # ax[0].ticklabel_format(style='sci', axis='Y', scilimits=(-1, -1))
            ax[1].ticklabel_format(style='sci', axis='Y', scilimits=(6, 6))
            ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            plt.tight_layout()

            print(f'Salvando {dirname}{sl}{name}_{fmt}.')
            fig.savefig(f'{dirname}{sl}{name}_{fmt}')
            # plt.show()
            print('')


def graph1() -> None:
    """
    chunks X dec_time (seconds) and chunks X file_size (Bytes)
    :return:
    """
    dirname = f'results{sl}graph1-2'
    os.makedirs(dirname, exist_ok=True)

    config = util.Config('config.json')
    dectime = util.load_json('times2.json')

    for name in config.single_videos_list:
        for fmt in config.tile_list:
            for quality in config.crf_list:
                plt.close()
                fig, ax = plt.subplots(1, 2, figsize=(19, 6))

                # Para cada quadro, plotar time de todos os tiles daquele video ao longo do tempo
                m, n = list(map(int, fmt.split('x')))
                for tile in range(1, m * n + 1):
                    print(f'Processing {name}-{fmt}-{quality}-tile{tile}')
                    size = []
                    time_ffmpeg = []

                    for chunk in range(1, config.duration + 1):
                        if name in 'ninja_turtles' and chunk > 58:
                            continue
                        s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
                        t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
                        size.append(float(s) * 8)
                        time_ffmpeg.append(float(t))

                    ax[0].plot(time_ffmpeg)
                    ax[1].plot(size, label=f'tile={tile}, corr={np.corrcoef((time_ffmpeg, size))[1][0]:.3f}')

                ax[0].set_xlabel('Chunks')
                ax[1].set_xlabel('Chunks')
                ax[0].set_ylabel('Time (s)')
                ax[1].set_ylabel('Rate (bps)')
                ax[0].set_title(f'{name}-{fmt}-crf{quality} - Times by chunks')
                ax[1].set_title(f'{name}-{fmt}-crf{quality} - Rates by chunks')
                ax[0].set_ylim(bottom=0)
                ax[1].set_ylim(bottom=0)
                ax[1].legend(loc='upper left', ncol=2, bbox_to_anchor=(1.01, 1.0))
                plt.tight_layout()

                savename = f'{dirname}/{name}_{fmt}_crf{quality}'
                print(f'Salvando {savename}.png')
                fig.savefig(f'{savename}')
                # fig.show()
                print('')


def graph1_a() -> None:
    """
    chunks X dec_time (seconds) and chunks X file_size (Bytes)
    :return:
    """
    dirname = f'results{sl}graph1-2a'
    os.makedirs(dirname, exist_ok=True)

    config = util.Config('config.json')
    dectime = util.load_json('times2.json')

    for name in config.single_videos_list:
        for fmt in config.tile_list:
            m, n = list(map(int, fmt.split('x')))
            for tile in range(1, m * n + 1):
                plt.close()
                fig, ax = plt.subplots(1, 2, figsize=(19, 6))

                # Para cada quadro, plotar time de todos os tiles daquele video ao longo do tempo
                for quality in config.crf_list:
                    print(f'Processing {name}-{fmt}-{quality}-tile{tile}')
                    size = []
                    time_ffmpeg = []

                    for chunk in range(1, config.duration + 1):
                        if name in 'ninja_turtles' and chunk > 58:
                            continue
                        s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
                        t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
                        size.append(float(s) * 8)
                        time_ffmpeg.append(float(t))

                    ax[0].plot(time_ffmpeg)
                    ax[1].plot(size, label=f'crf={quality}, corr={np.corrcoef((time_ffmpeg, size))[1][0]:.3f}')

                ax[0].set_xlabel('Chunks')
                ax[1].set_xlabel('Chunks')
                ax[0].set_ylabel('Time/Tile (s)')
                ax[1].set_ylabel('Rate/Tile (bps)')
                ax[0].set_title(f'{name}-{fmt}-tile{tile} - Times by chunks')
                ax[1].set_title(f'{name}-{fmt}-tile{tile} - Rates by chunks')
                ax[0].set_ylim(bottom=0)
                ax[1].set_ylim(bottom=0)
                ax[1].legend(loc='upper left', ncol=2, bbox_to_anchor=(1.01, 1.0))
                plt.tight_layout()

                savename = f'{dirname}/{name}_{fmt}_tile{tile}'
                print(f'Salvando {savename}.png')
                fig.savefig(f'{savename}')
                # fig.show()
                print('')


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
