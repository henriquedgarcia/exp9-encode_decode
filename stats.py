#!/bin/python3
# coding=utf-8
import os
import pickle
from itertools import product as it

import fitter
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from utils import util

sl = util.check_system()['sl']

project = 'ffmpeg_scale_12videos_60s_qp'
config = util.Config('config.json', factor='scale')
dectime_name = f'times_{project}'
color_list = ['blue', 'orange', 'green', 'red']
dists = ['burr12', 'expon', 'fatiguelife', 'gamma', 'genpareto', 'halfnorm',
         'invgauss', 'rayleigh', 'rice', 't']


def main():
    # psnr()  # Processar psnr também. à fazer
    stats()
    # graph1(graph_folder='1_graph1-tiles-chunks_x_dec_time')
    # graph2(graph_folder='2_graph2-quality-chunks_x_dec_time')
    # graph3(graph_folder='3_graph3_heatmap')
    # histogram_name_fmt('histogram_name-fmt')
    # histogram_group_fmt('histogram_group-fmt')
    # heatmap_fmt_quality('heatmap_fmt-quality')
    # hist1samefig(graph_folder="hist1samefig")
    # hist1sameplt(graph_folder="hist1sameplt")
    # hist2samefig(graph_folder="hist2samefig")
    # hist2sameplt(graph_folder="hist2sameplt")
    # graph4()   # Por grupo
    # graph5b()  # Geralzão por tile
    # graph5c()  # Geralzão por qualidade
    # graph5d()  # Geralzão por fmt
    # graph5e()  # Geralzão por name


def json2pandas():
    # 12 videos x (1+6+24+92) tiles x 4 qualidades = 5.904 listas de 60 chunks
    # x 2 11.808
    df = {}
    for name in config.videos_list:
        for fmt in config.tile_list:
            m, n = list(map(int, fmt.split('x')))
            for quality in config.quality_list:
                for tile in range(1, m * n + 1):
                    col_name = (f'{config.videos_list[name]["group"]}_{name}_'
                                f'{fmt}_{config.factor}{quality}_tile{tile}')
                    time, size, _ = get_data_chunks(name, fmt, quality, tile)
                    df[f'{col_name}_time'] = time
                    df[f'{col_name}_size'] = size
    util.save_json(df, f'dectime_{config.factor}_singlekey.json')


def stats():
    # Base object
    video_seg = util.VideoStats(config=config,
                                project=f'results{sl}{project}')

    df = {}

    for (video_seg.name,
         video_seg.fmt,
         video_seg.quality) in it(config.videos_list,
                                  config.tile_list,
                                  config.quality_list):

        for video_seg.tile in range(1, video_seg.num_tiles + 1):
            c_name = (f'{config.videos_list[video_seg.name]["group"]}_'
                      f'{video_seg.name}_'
                      f'{video_seg.fmt}_'
                      f'{config.factor}{video_seg.quality}_'
                      f'tile{video_seg.tile}')

            # Processando taxa Salvando em bps
            size_list = []
            for video_seg.chunk in range(1, video_seg.chunks + 1):
                file = f'{video_seg.segment_path}.mp4'
                if os.path.isfile(file):
                    # O tamanho do chunk só pode ser considerado taxa porque o
                    # chunk tem 1 segndo
                    size_list.append(os.path.getsize(file) * 8)
            df[f'{c_name}_rate'] = size_list

            # Processando tempos
            times_list = []
            for video_seg.chunk in range(1, video_seg.chunks + 1):
                file = f'{video_seg.log_path}.txt'
                if os.path.isfile(file):
                    print(f'Processando {file}')
                    f = open(file, 'r', encoding='utf-8')

                    times = []
                    for line in f:
                        if line.find(video_seg.bench_stamp) >= 0:
                            # Pharse da antiga decodificação
                            if video_seg.factor in 'crf':
                                line = line.replace('bench: ', ' ')
                                line = line.replace('s ', ' ')
                                line = line.strip()[:-1]
                                line = line.split(' ')
                                for i in range(0, len(line), 3):
                                    times.append(float(line[i][6:]))
                            elif video_seg.factor in ['scale', 'qp']:
                                line = line.split(' ')[1]
                                line = line.split('=')[1]
                                times.append(float(line[:-1]))
                    f.close()

                    times_list.append(np.average(times))
            df[f'{c_name}_time'] = times_list

    name = f'{dectime_name}_single.json'

    util.save_json(df, name)


def graph1(graph_folder):
    """
    tiles-chunksXdec_time (seconds) and chunks X file_size (Bytes)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname, exist_ok=True)

    for name in config.videos_list:
        group = config.videos_list[name]['group']

        for fmt in config.tile_list:
            m, n = list(map(int, fmt.split('x')))

            for quality in config.quality_list:
                title = f'{group}-{name}-{fmt}-{quality}'

                plt.close()
                fig, ax = plt.subplots(1, 2, figsize=(19, 6))

                # plota os chunks de todos os tiles de um video-fmt-quality
                for tile in range(1, m * n + 1):
                    print(f'Processing {title}-tile{tile}')
                    time, size, corr = get_data_chunks(name=name,
                                                       fmt=fmt,
                                                       quality=quality,
                                                       tile=tile)

                    ax[0].plot(time, label=(f'tile={tile}, '
                                            f'corr={corr: .3f}'))
                    ax[1].plot(size, label=(f'tile={tile}, '
                                            f'corr={corr: .3f}'))

                # Configura o plot
                for i, axes in enumerate(ax):
                    if i == 0:
                        axes.set_ylabel('Times (s)')
                        axes.set_title(f'{title}-{config.factor}{quality} '
                                       f'- Times by chunks')
                    elif i == 1:
                        axes.set_ylabel('Rates (bps)')
                        axes.set_title(f'{title}-{config.factor}{quality} '
                                       f'- Rates by chunks')
                    axes.set_xlabel('Chunks')
                    axes.set_ylim(bottom=0)
                    axes.legend(loc='upper left', ncol=2,
                                bbox_to_anchor=(1.01, 1.0))

                plt.tight_layout()

                savename = (f'{dirname}{sl}'
                            f'{group}-'
                            f'{name}_{fmt}_{config.factor}{quality}')

                print(f'Salvando {savename}.png')
                fig.savefig(f'{savename}')
                # fig.show()
                print('')


def graph2(graph_folder):
    """
    quality-chunksXdec_time (seconds) and chunks X file_size (Bytes)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname, exist_ok=True)

    for name in config.videos_list:
        group = config.videos_list[name]['group']

        for fmt in config.tile_list:
            m, n = list(map(int, fmt.split('x')))

            for tile in range(1, m * n + 1):
                title = f'{group}-{name}-{fmt}-{tile}'

                plt.close()
                fig, ax = plt.subplots(1, 2, figsize=(19, 6))

                # plota os chunks de todos as qualidades de um video-fmt-tile
                for quality in config.quality_list:
                    print(f'Processing {title}-{config.factor}{quality}')
                    time, size, corr = get_data_chunks(name=name,
                                                       fmt=fmt,
                                                       quality=quality,
                                                       tile=tile)

                    ax[0].plot(time, label=(f'{config.factor}={quality}, '
                                            f'corr={corr: .3f}'))
                    ax[1].plot(size, label=(f'{config.factor}={quality}, '
                                            f'corr={corr: .3f}'))

                # Configura o plot
                for i, axes in enumerate(ax):
                    if i == 0:
                        axes.set_ylabel('Time/Tile (s)')
                        axes.set_title(f'{title}-tile{tile} '
                                       f'- Times by chunks')
                    elif i == 1:
                        axes.set_ylabel('Rate/Tile (bps)')
                        axes.set_title(f'{title}-tile{tile} '
                                       f'- Rates by chunks')
                    axes.set_xlabel('Chunks')
                    axes.set_ylim(bottom=0)
                    axes.legend(loc='upper left', ncol=2,
                                bbox_to_anchor=(1.01, 1.0))

                plt.tight_layout()

                savename = (f'{dirname}{sl}'
                            f'{group}-'
                            f'{name}_{fmt}_{config.factor}_tile{tile}')

                print(f'Salvando {savename}.png')
                fig.savefig(f'{savename}')
                # fig.show()
                print('')


def graph3(graph_folder):
    """
    bar
    tile X average_dec_time (seconds) and tile X average_rate (Bytes)
    :return: None
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname, exist_ok=True)

    for name in config.videos_list:
        group = config.videos_list[name]['group']

        for fmt in config.tile_list:
            m, n = list(map(int, fmt.split('x')))

            # Figure
            plt.close()
            fig = plt.figure(figsize=(10, 8), dpi=150)

            # Axes
            axes = [fig.add_subplot(4, 1, 1),
                    fig.add_subplot(4, 1, 2)]

            ax_hmt, ax_hms = [], []
            for x in range(9, 13): ax_hmt.append(fig.add_subplot(4, 4, x))
            for x in range(13, 17): ax_hms.append(fig.add_subplot(4, 4, x))

            # Bar config
            offset = 0
            width = 0.8 / len(config.crf_list)
            start_position = (0.8 - width) / 2

            average_time, average_size = {}, {}
            std_time, std_size = {}, {}
            c1, c2 = [], []  # colections of heatmap

            # Para cada qualidade
            for count, quality in enumerate(config.crf_list):
                heatmap = make_heatmap(name, fmt, quality)

                average_time[quality] = heatmap.average_time
                average_size[quality] = heatmap.average_size
                std_time[quality] = heatmap.std_time
                std_size[quality] = heatmap.std_size
                frame_time = heatmap.frame_time
                frame_size = heatmap.frame_size

                # Calcula a média dos chunks para cada tile
                # for tile, m_, n_ in it(range(1, m * n + 1),
                #                        range(1, m + 1),
                #                        range(1, n + 1)):
                #     # Coleta dados de todos os chunks
                #     time, size, _ = get_data_chunks(name, fmt, quality, tile)
                #
                #     # Anexa as médias e desvios padrões
                #     average_time[quality].append(np.average(time))
                #     average_size[quality].append(np.average(size))
                #     std_time[quality].append(np.std(time))
                #     std_size[quality].append(np.std(size))
                #
                #     # Preenche a matriz do heatmap
                #     frame1[n_ - 1, m_ - 1] = average_time[quality][-1]
                #     frame2[n_ - 1, m_ - 1] = average_size[quality][-1]

                # Plota dois bar

                x = np.arange(1, m * n + 1)
                # O eixo X, o numero de chunks
                x = x - start_position + offset
                corr = np.corrcoef(x=(average_time[quality],
                                      average_size[quality]))[1][0]
                axes[0].bar(x, average_time[quality],
                            yerr=std_time[quality],
                            width=width,
                            label=f'crf={quality}_corr={corr: .3f}')
                axes[1].bar(x, average_size[quality],
                            yerr=std_size[quality],
                            width=width,
                            label=f'crf={quality}')
                offset += width

                # Configura Bar
                for ax in axes:
                    ax.set_xlabel('Tile')
                    ax.set_ylim(bottom=0)
                    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
                axes[0].set_title(f'{name} - Times by tiles, tile={fmt}')
                axes[1].set_title(f'{name} - Rates by tiles, tile={fmt}')
                axes[0].set_ylabel('Average Time (s)')
                axes[1].set_ylabel('Average Rate (bps)')
                axes[1].ticklabel_format(style='sci', axis='Y',
                                         scilimits=(6, 6))

                # Plota um Pcolor (Heatmap) e pega sua collection
                c1.append(ax_hmt[count].pcolor(frame_time, cmap='jet'))
                c2.append(ax_hms[count].pcolor(frame_size, cmap='jet'))

                # Configura os eixos do heatmap da qualidade atual
                ax_hms[count].set_title('Rate Heatmap')
                ax_hms[count].set_xticklabels([])
                ax_hms[count].set_yticklabels([])

                ax_hmt[count].set_title('time Heatmap')
                ax_hmt[count].set_xlabel(f'crf={quality}')
                ax_hmt[count].set_xticklabels([])
                ax_hmt[count].set_yticklabels([])

            # Normaliza heatmap
            vmin1 = min(collection.get_array().min() for collection in c1)
            vmax1 = max(collection.get_array().max() for collection in c1)
            vmin2 = min(collection.get_array().min() for collection in c2)
            vmax2 = max(collection.get_array().max() for collection in c2)

            norm1 = colors.Normalize(vmin=vmin1, vmax=vmax1)
            norm2 = colors.Normalize(vmin=vmin2, vmax=vmax2)
            for collection1, collection2 in zip(c1, c2):
                collection1.set_norm(norm1)
                collection2.set_norm(norm2)

            # Colorbar
            fig.colorbar(c1[-1], ax=ax_hmt,
                         orientation='vertical', fraction=.04)
            fig.colorbar(c2[-1], ax=ax_hms,
                         orientation='vertical', fraction=.04)
            plt.tight_layout()

            # Finaliza
            print(f'Salvando {dirname}{sl}{name}_{fmt}.')
            fig.savefig(f'{dirname}{sl}{group}-{name}_{fmt}')
            # plt.show()
            print('')


# hist0
def histogram_name_fmt(graph_folder):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)

    for bins in ['auto', 25, 50, 75, 100, 125, 150, 200]:
        for name in config.videos_list:
            for fmt in config.tile_list:
                # Coleta dados
                tridata = get_data_quality(name, fmt)

                # Faz o plot
                fig = make_hist(tridata, dirname, bins, group=None, name=name,
                                fmt=fmt, quality=None, tile=None, chunk=None)

                # Salva
                fig.savefig(f'{dirname}{sl}'
                            f'hist_groups_bins{bins}_{name}_{fmt}')
                # fig.show()
                print(f'hist {bins} bins, {name}_{fmt}')


# hist1
def histogram_group_fmt(graph_folder):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)

    for bins in ['auto', 25, 50, 75, 100, 125, 150, 200]:
        for group in ['0', '1', '2', '3']:
            for fmt in config.tile_list:
                # Coleta dados
                tridata = get_data_group_fmt(group, fmt)

                # Faz o plot
                fig = make_hist(tridata, dirname, bins=bins, group=group,
                                name=None, fmt=fmt, quality=None, tile=None,
                                chunk=None)

                # Salva
                fig.savefig(f'{dirname}{sl}'
                            f'hist_groups_bins{bins}_group{group}_{fmt}')
                # fig.show()
                print(f'hist bins {bins}, group{group}_{fmt}')


def heatmap_fmt_quality(graph_folder):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}{sl}data', exist_ok=True)
    bins = 'auto'

    for fmt in config.tile_list:
        m, n = list(map(int, fmt.split('x')))

        fig, ax = plt.subplots(2, 4, figsize=(12, 5), dpi=150)

        c1, c2 = [], []
        for count, quality in enumerate(config.quality_list):
            # Cria e preenche o heatmap
            heatmap = Heatmap(m, n)
            for tile, (x, y) in zip(range(1, m * n + 1),
                                    it(range(1, m + 1), range(1, n + 1))):
                time, size, corr = get_data_name_chunk(fmt, tile, quality)

                time = np.average(time)
                size = np.average(size)

                heatmap.average_time.append(time)
                heatmap.average_size.append(size)
                heatmap.std_time.append(np.std(time))
                heatmap.std_size.append(np.std(size))

                heatmap.frame_time[y - 1, x - 1] = time
                heatmap.frame_size[y - 1, x - 1] = size

            # Plota um Pcolor (Heatmap) e pega sua collection
            c1.append(ax[0][count].pcolor(heatmap.frame_time,
                                          cmap='jet'))
            c2.append(ax[1][count].pcolor(heatmap.frame_size,
                                          cmap='jet'))

            # Configura os eixos do heatmap da qualidade atual
            ax[0][count].set_title(f'Dectime Heatmap {fmt}')
            ax[1][count].set_xlabel(f'crf={quality}')
            ax[0][count].set_xticklabels([])
            ax[0][count].set_yticklabels([])

            ax[1][count].set_title(f'Bitrate Heatmap {fmt}')
            ax[1][count].set_xlabel(f'crf={quality}')
            ax[1][count].set_xticklabels([])
            ax[1][count].set_yticklabels([])

        # Normaliza heatmap
        vmin1 = min(collection.get_array().min() for collection in c1)
        vmax1 = max(collection.get_array().max() for collection in c1)
        vmin2 = min(collection.get_array().min() for collection in c2)
        vmax2 = max(collection.get_array().max() for collection in c2)

        norm1 = colors.Normalize(vmin=vmin1, vmax=vmax1)
        norm2 = colors.Normalize(vmin=vmin2, vmax=vmax2)
        for collection1, collection2 in zip(c1, c2):
            collection1.set_norm(norm1)
            collection2.set_norm(norm2)

        # Colorbar
        # fig.colorbar(c1[-1], ax=ax[0],
        #              orientation='horizontal')
        # fig.colorbar(c2[-1], ax=ax[1],
        #              orientation='horizontal')
        # plt.legend()
        # fig.suptitle(f'Heatmap dectime and bitrate {fmt}.', y=1.05, fontsize=16)
        plt.tight_layout()

        # Finaliza
        print(f'Salvando {dirname}{sl}{fmt}_{config.factor}.')
        fig.savefig(f'{dirname}{sl}{fmt}_{config.factor}')
        # plt.show()
        print('')


def hist1samefig(graph_folder):
    """ Compara os histogramas. Para cada video-fmt plota todas as qualidades
    (agrega tiles e chunks)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}', exist_ok=True)

    best_dist_df = util.AutoDict()
    times = util.AutoDict()
    sizes = util.AutoDict()

    # Coleta dados
    iterator = it(config.videos_list, config.tile_list, config.quality_list)
    for name, fmt, quality in list(iterator):
        m, n = list(map(int, fmt.split('x')))
        times[name][fmt][quality] = []
        sizes[name][fmt][quality] = []

        for tile, chunk in list(it(range(1, m * n + 1),
                                   range(1, config.duration + 1))):
            if name in 'ninja_turtles' and chunk > 58: continue
            t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
            s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
            times[name][fmt][quality].append(t)
            sizes[name][fmt][quality].append(s)

    # Calcular melhores fit e plota
    for name, fmt in list(it(config.videos_list, config.tile_list)):
        # Faz um fit por name-fmt-qualidade

        fitter_dict = {}
        for quality in config.quality_list:
            data = times[name][fmt][quality]
            col_label = f'{fmt}-{quality}'

            # Faz o fit
            f = fitter.fitter.Fitter(data, bins='auto', distributions=dists,
                                     verbose=False)
            fitter_dict[quality] = f
            fitter_dict[quality].fit()

            # Procura os menores SSE
            short_sse = fitter_dict[quality]
            short_sse = short_sse.df_errors
            short_sse = short_sse.sort_values(by="sumsquare_error")
            short_sse = short_sse.index[0]

            # Melhor fit
            sse = fitter_dict[quality].df_errors["sumsquare_error"][short_sse]
            best_dist_df[name][col_label] = f'{short_sse}, SSE={sse: .1f}'

        # Para cada video e fmt cria uma figura. cuidado. se mudar o formato do
        # gráfico vai ter que mudar tudo dentro da função.
        plt.close()
        fig = plt.figure(figsize=(13, 9), dpi=150)

        color_it = iter(color_list)
        m, n = list(map(int, fmt.split('x')))
        ax_rate = fig.add_subplot(3, 2, 5)
        ax_cdf = fig.add_subplot(3, 2, 6)

        for idx, quality in enumerate(config.quality_list, 1):
            color = next(color_it)

            ###############################################################
            # plota o histograma desta qualidade
            t_avg = np.average(times[name][fmt][quality])
            t_std = np.std(times[name][fmt][quality])
            t_ct = np.average(times[name][fmt][quality]) * (m * n)

            label = (f'dectime_avg={t_avg:.03f} s\n'
                     f'dectime_std={t_std:.03f} s\n'
                     f'time/chunk/tile={t_ct:.03f} s')

            ax = fig.add_subplot(3, 2, idx, sharex=ax_cdf)
            ax.hist(times[name][fmt][quality], color=color, bins='auto',
                    histtype='bar',
                    density=True, label=label)
            ax.set_title(f'{name}, {fmt}, crf {quality}')
            ax.set_ylabel("PDF")
            ax.set_xlabel('Decoder Times')

            ###############################################################
            # plota os 3 best fit para esta qualidade
            best_dists = fitter_dict[quality].df_errors
            best_dists = best_dists.sort_values(by="sumsquare_error")
            best_dists = best_dists.index[0:5]

            for dist_name in best_dists:
                sse = fitter_dict[quality]
                sse = sse.df_errors["sumsquare_error"][dist_name]
                label = f'{dist_name},\nSSE = {sse: .3f}'
                ax.plot(fitter_dict[quality].x,
                        fitter_dict[quality].fitted_pdf[dist_name],
                        label=label)
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota o bar da taxa para esta qualidade
            s_avg = np.average(sizes[name][fmt][quality]) * 8
            s_std = np.std(sizes[name][fmt][quality]) * 8
            s_ct = np.average(sizes[name][fmt][quality]) * 8
            label = (f'rate_avg={s_avg / 1000:,.0f} Kbps\n'
                     f'rate_std={s_std / 1000:,.0f} Kbps\n'
                     f'rate/tile={s_ct / (1000 * m * n):.03f} kbps')
            ax_rate.bar(idx - 1, s_avg, color=color, yerr=s_std, label=label)
            ax_rate.set_title('Rate')
            ax_rate.set_ylabel("Rate (10³ bps)")
            ax_rate.set_xlabel("Decoder Time (s)")
            ax_rate.ticklabel_format(axis='y', style='scientific',
                                     scilimits=(3, 3))
            ax_rate.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota a CDF para esta qualidade
            label = f'{name}_{fmt}_crf{quality}'
            ax_cdf.hist(times[name][fmt][quality], color=color, bins='auto',
                        density=True, cumulative=True, histtype='step',
                        label=label)
            ax_cdf.set_title('CDF')
            ax_cdf.set_ylabel("CDF")
            ax_cdf.set_xlabel("Decoder Time (s)")
            ax_cdf.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

        ax_rate.set_xticklabels(['0'] + config.quality_list)
        plt.tight_layout()
        group = config.videos_list[name]["group"]
        fig.savefig(f'{dirname}{sl}{group}-hist_{name}_{fmt}')
        # plt.show()
        print(f'hist_{name}_{fmt}')

    # Salva melhores fits em um csv
    best_dist_df = pd.DataFrame(best_dist_df)
    best_dist_df.to_csv(f'{dirname}{config.sl}best_dist.csv')


def hist1sameplt(graph_folder):
    """ Compara os histogramas. Para cada video-fmt plota todas as qualidades
    (agrega tiles e chunks)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    times = util.AutoDict()
    sizes = util.AutoDict()

    os.makedirs(f'{dirname}', exist_ok=True)

    # Coleta dados
    for name, fmt, quality in it(config.videos_list, config.tile_list,
                                 config.quality_list):
        m, n = list(map(int, fmt.split('x')))
        times[name][fmt][quality] = []
        sizes[name][fmt][quality] = []

        for tile, chunk in it(range(1, m * n + 1),
                              range(1, config.duration + 1)):
            if name in 'ninja_turtles' and chunk > 58: continue
            t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
            s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
            times[name][fmt][quality].append(t)
            sizes[name][fmt][quality].append(s)

    # Calcular melhores fit e plota
    for name, fmt in it(config.videos_list, config.tile_list):
        # Para cada video e fmt cria uma figura. cuidado. se mudar o formato do
        # gráfico vai ter que mudar tudo dentro da função.
        plt.close()
        fig = plt.figure(figsize=(13, 9), dpi=150)

        color_it = iter(color_list)
        m, n = list(map(int, fmt.split('x')))
        ax_cdf = fig.add_subplot(2, 1, 2)
        ax = fig.add_subplot(2, 1, 1, sharex=ax_cdf)

        for idx, quality in enumerate(config.quality_list, 1):
            color = next(color_it)

            ###############################################################
            # plota o histograma desta qualidade
            t_avg = np.average(times[name][fmt][quality])
            t_std = np.std(times[name][fmt][quality])
            t_ct = np.average(times[name][fmt][quality]) * (m * n)

            label = (f'dectime_avg={t_avg:.03f} s\n'
                     f'dectime_std={t_std:.03f} s\n'
                     f'time/chunk/tile={t_ct:.03f} s')

            ax.hist(times[name][fmt][quality], color=color, bins='auto',
                    histtype='bar', density=True, label=label)
            ax.set_title(f'{name}, {fmt}, crf {quality}')
            ax.set_ylabel("PDF")
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota a CDF para esta qualidade
            label = f'{name}_{fmt}_crf{quality}'
            ax_cdf.hist(times[name][fmt][quality], color=color, bins='auto',
                        density=True, cumulative=True, histtype='step',
                        label=label)
            ax_cdf.set_title('CDF')
            ax_cdf.set_ylabel("CDF")
            ax_cdf.set_xlabel("Decoder Time (s)")
            ax_cdf.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

        plt.tight_layout()
        grupo = config.videos_list[name]["group"]
        fig.savefig(f'{dirname}{sl}{grupo}-hist_{name}_{fmt}')
        # plt.show()
        print(f'hist_{name}_{fmt}')


def hist2samefig(graph_folder):
    """ Compara os histogramas. Para cada video-fmt plota todas as qualidades
    (agrega tiles e chunks)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}', exist_ok=True)

    best_dist_df = util.AutoDict()
    times = util.AutoDict()
    sizes = util.AutoDict()

    # Coleta dados
    for name, fmt, quality in it(config.videos_list, config.tile_list,
                                 config.quality_list):
        m, n = list(map(int, fmt.split('x')))
        times[name][fmt][quality] = []
        sizes[name][fmt][quality] = []

        for tile, chunk in it(range(1, m * n + 1),
                              range(1, config.duration + 1)):
            if name in 'ninja_turtles' and chunk > 58: continue
            t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
            s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
            times[name][fmt][quality].append(t)
            sizes[name][fmt][quality].append(s)

    # Calcular melhores fit e plota
    for name, quality in it(config.videos_list, config.quality_list):
        # Faz um fit por name-fmt-qualidade
        fitter_dict = {}
        for fmt in config.tile_list:
            data = times[name][fmt][quality]
            col_label = f'{fmt}-{quality}'

            # Faz o fit
            f = fitter.fitter.Fitter(data, bins='auto', distributions=dists,
                                     verbose=False)
            fitter_dict[fmt] = f
            fitter_dict[fmt].fit()

            # Procura os menores SSE
            short_sse = fitter_dict[fmt]
            short_sse = short_sse.df_errors
            short_sse = short_sse.sort_values(by="sumsquare_error")
            short_sse = short_sse.index[0]

            # Melhor fit
            sse = fitter_dict[fmt].df_errors["sumsquare_error"][short_sse]
            best_dist_df[name][col_label] = f'{short_sse}, SSE={sse: .1f}'

        # Para cada video e fmt cria uma figura. cuidado. se mudar o formato do
        # gráfico vai ter que mudar tudo dentro da função.
        plt.close()
        fig = plt.figure(figsize=(13, 9), dpi=150)

        color_it = iter(color_list)
        ax_rate = fig.add_subplot(3, 2, 5)
        ax_cdf = fig.add_subplot(3, 2, 6)

        for idx, fmt in enumerate(config.tile_list, 1):
            color = next(color_it)
            m, n = list(map(int, fmt.split('x')))

            ###############################################################
            # plota o histograma desta qualidade
            t_avg = np.average(times[name][fmt][quality])
            t_std = np.std(times[name][fmt][quality])
            t_ct = np.average(times[name][fmt][quality]) * (m * n)

            label = (f'dectime_avg={t_avg:.03f} s\n'
                     f'dectime_std={t_std:.03f} s\n'
                     f'time/chunk/tile={t_ct:.03f} s')

            ax = fig.add_subplot(3, 2, idx, sharex=ax_cdf)
            ax.hist(times[name][fmt][quality], color=color, bins='auto',
                    histtype='bar',
                    density=True, label=label)
            ax.set_title(f'{name}, crf {quality}, {fmt}')
            ax.set_ylabel("PDF")
            ax.set_xlabel('Decoder Times')

            ###############################################################
            # plota os 3 best fit para esta qualidade
            best_dists = fitter_dict[fmt].df_errors
            best_dists = best_dists.sort_values(by="sumsquare_error")
            best_dists = best_dists.index[0:5]

            for dist_name in best_dists:
                sse = fitter_dict[fmt]
                sse = sse.df_errors["sumsquare_error"][dist_name]
                label = f'{dist_name},\nSSE = {sse: .3f}'
                ax.plot(fitter_dict[fmt].x,
                        fitter_dict[fmt].fitted_pdf[dist_name],
                        label=label)
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota o bar da taxa para esta qualidade
            s_avg = np.average(sizes[name][fmt][quality]) * 8
            s_std = np.std(sizes[name][fmt][quality]) * 8
            s_ct = np.average(sizes[name][fmt][quality]) * 8
            label = (f'rate_avg={s_avg / 1000:,.0f} Kbps\n'
                     f'rate_std={s_std / 1000:,.0f} Kbps\n'
                     f'rate/tile={s_ct / (1000 * m * n):.03f} kbps')
            ax_rate.bar(idx, s_avg, color=color, yerr=s_std, label=label)
            ax_rate.set_title('Rate')
            ax_rate.set_ylabel("Rate (10³ bps)")
            ax_rate.set_xlabel("Decoder Time (s)")
            ax_rate.ticklabel_format(axis='y', style='scientific',
                                     scilimits=(3, 3))
            ax_rate.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota a CDF para esta qualidade
            label = f'{name}_crf{quality}_{fmt}'
            ax_cdf.hist(times[name][fmt][quality], color=color, bins='auto',
                        density=True, cumulative=True, histtype='step',
                        label=label)
            ax_cdf.set_title('CDF')
            ax_cdf.set_ylabel("CDF")
            ax_cdf.set_xlabel("Decoder Time (s)")
            ax_cdf.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

        ax_rate.set_xticklabels(['0'] + config.tile_list)
        plt.tight_layout()
        grupo = config.videos_list[name]["group"]
        fig.savefig(f'{dirname}{sl}{grupo}-hist_{name}_{quality}')
        # plt.show()
        print(f'hist_{name}_{quality}')

    # Salva melhores fits em um csv
    # best_dist_df = pd.DataFrame(best_dist_df)
    # best_dist_df.to_csv(f'{dirname}{config.sl}best_dist.csv')


def hist2sameplt(graph_folder):
    """ Compara os histogramas. Para cada video-fmt plota todas as qualidades
    (agrega tiles e chunks)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    times = util.AutoDict()
    sizes = util.AutoDict()

    os.makedirs(f'{dirname}', exist_ok=True)

    # Coleta dados
    for name, fmt, quality in it(config.videos_list, config.tile_list,
                                 config.quality_list):
        m, n = list(map(int, fmt.split('x')))
        times[name][fmt][quality] = []
        sizes[name][fmt][quality] = []

        for tile, chunk in it(range(1, m * n + 1),
                              range(1, config.duration + 1)):
            if name in 'ninja_turtles' and chunk > 58: continue
            t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
            s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
            times[name][fmt][quality].append(t)
            sizes[name][fmt][quality].append(s)

    # Calcular melhores fit e plota
    for name, quality in it(config.videos_list, config.quality_list):
        # Para cada video e fmt cria uma figura. cuidado. se mudar o formato do
        # gráfico vai ter que mudar tudo dentro da função.
        plt.close()
        fig = plt.figure(figsize=(13, 9), dpi=150)

        color_it = iter(color_list)
        ax_cdf = fig.add_subplot(2, 1, 2)
        ax = fig.add_subplot(2, 1, 1, sharex=ax_cdf)

        for idx, fmt in enumerate(config.tile_list, 1):
            color = next(color_it)
            m, n = list(map(int, fmt.split('x')))

            ###############################################################
            # plota o histograma desta qualidade
            t_avg = np.average(times[name][fmt][quality])
            t_std = np.std(times[name][fmt][quality])
            t_ct = np.average(times[name][fmt][quality]) * (m * n)

            label = (f'dectime_avg={t_avg:.03f} s\n'
                     f'dectime_std={t_std:.03f} s\n'
                     f'time/chunk/tile={t_ct:.03f} s')

            ax.hist(times[name][fmt][quality], color=color, bins='auto',
                    histtype='bar', density=True, label=label)
            ax.set_title(f'{name}, crf {quality}, {fmt}')
            ax.set_ylabel("PDF")
            ax.set_xlabel('Decoder Times')
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota a CDF para esta qualidade
            label = f'{name}_crf{quality}_{fmt}'
            ax_cdf.hist(times[name][fmt][quality], color=color, bins='auto',
                        density=True, cumulative=True, histtype='step',
                        label=label)
            ax_cdf.set_title('CDF')
            ax_cdf.set_ylabel("CDF")
            ax_cdf.set_xlabel("Decoder Time (s)")
            ax_cdf.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

        plt.tight_layout()
        grupo = config.videos_list[name]["group"]
        fig.savefig(f'{dirname}{sl}{grupo}-hist_{name}_{quality}')
        # plt.show()
        print(f'hist_{name}_{quality}')


def make_hist(tridata, dirname, bins, group=None, name=None, fmt=None,
              quality=None, tile=None, chunk=None):
    m_, n_ = list(map(int, fmt.split('x')))
    time, size, corr = tridata
    data_stats_t = [np.average(time), np.std(time), corr]
    data_stats_s = [np.average(size), np.std(size), corr]

    # Persistência
    print('Calculando o fit.')
    f_t_name = ''
    f_s_name = ''
    if group is not None \
            and fmt is not None:
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_group{group}_{fmt}.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_group{group}_{fmt}.pickle')
        tile = f'{bins}bins_group{group}_{fmt}'
    if name is not None \
            and fmt is not None:
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_{name}_{fmt}.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_{name}_{fmt}.pickle')
        tile = f'{bins}bins_{name}_{fmt}'
    if fmt is not None \
            and quality is not None \
            and tile is not None:
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_{fmt}_{config.factor}{quality}_'
                    f'tile{tile}.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_{fmt}_{config.factor}{quality}_'
                    f'tile{tile}.pickle')
        tile = f'{bins}bins_{name}_{fmt}'

    f_t = make_fit(data=time, bins=bins, out_file=f_t_name)
    f_s = make_fit(data=size, bins=bins, out_file=f_s_name)

    # Faz histograma
    plt.close()
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), dpi=150)
    for ax, k in zip(axes, ['pdf_dectime', 'pdf_rate',
                            'cdf_dectime', 'cdf_rate']):
        if k in 'pdf_dectime':
            ax = plota_hist(f_t, ax, bins, data_stats_t, 'time', 'pdf')
            ax.set_title(f'PDF Dectime - {tile}')
            ax.set_xlabel('Decoder Time')
        elif k in 'pdf_rate':
            ax = plota_hist(f_s, ax, bins, data_stats_s, 'rate', 'pdf')
            ax.set_title(f'PDF Bitrate - {tile}')
            ax.set_xlabel('Bitrate')
        elif k in 'cdf_dectime':
            ax = plota_hist(f_t, ax, bins, data_stats_t, 'time', 'cdf')
            ax.set_title(f'CDF Dectime - {tile}')
            ax.set_xlabel('Decoder Time')
        elif k in 'cdf_rate':
            ax = plota_hist(f_s, ax, bins, data_stats_s, 'rate', 'cdf')
            ax.set_title(f'CDF Bitrate - {tile}')
            ax.set_xlabel('Bitrate')

    plt.tight_layout()
    return fig


def make_fit(data, out_file, bins, overwrite=False):
    if os.path.exists(out_file) and not overwrite:
        print(f'Carregando {out_file}.')
        with open(out_file, 'rb') as f1:
            f = pickle.load(f1)
    else:
        print('Calculando o fit do tempo.')
        f = fitter.fitter.Fitter(data, bins=bins,
                                 distributions=dists,
                                 verbose=False)
        f.fit()

        print(f'Salvando {out_file}.')
        with open(out_file, 'wb') as f1:
            pickle.dump(f, f1, pickle.HIGHEST_PROTOCOL)

    return f


def plota_hist(f, ax: matplotlib.axes.Axes, bins, data_stats, metric, func,
               label='') -> plt.Axes:
    [avg, std, corr] = data_stats
    errors = f.df_errors
    errors_sorted = errors.sort_values(by="sumsquare_error")
    short_sse = errors_sorted.index[0:5]

    if func in 'pdf':
        ax.set_ylabel("Probability Density")
        if metric is 'time':
            label = (f'dectime_avg={avg:.03f} s\n'
                     f'dectime_std={std:.03f} s\n'
                     f'corr={corr:.03f}')
            ax.ticklabel_format(axis='y', style='scientific')
        elif metric is 'rate':
            label = (f'rate_avg={avg:.03f} bps\n'
                     f'rate_std={std:.03f} bps\n'
                     f'corr={corr:.03f}')
        ax.hist(f._data,
                bins=bins,
                histtype='bar',
                density=True,
                label=label)
        # plota os 3 melhores fits
        for dist_name in short_sse:
            sse = f.df_errors["sumsquare_error"][dist_name]
            label = f'{dist_name},\nSSE = {sse: .3E}'
            ax.plot(f.x, f.fitted_pdf[dist_name], label=label)

    if func in 'cdf':
        ax.set_ylabel("Cumulative Distribution")
        if metric is 'time':
            label = (f'dectime_avg={avg:.03f} s\n'
                     f'dectime_std={std:.03f} s\n'
                     f'corr={corr:.03f}')
        elif metric is 'rate':
            label = (f'rate_avg={avg:.03f} bps\n'
                     f'rate_std={std:.03f} bps\n'
                     f'corr={corr:.03f}')
        ax.hist(f._data,
                bins=bins,
                density=True,
                cumulative=True,
                histtype='step',
                label=label)

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    return ax


def get_data_group_fmt(group, fmt):
    size = []
    time = []
    for name in config.videos_list:
        if config.videos_list[name]['group'] in group:
            t, s, _ = get_data_quality(name, fmt)
            time.extend(t)
            size.extend(s)
    corr = np.corrcoef((time, size))[1][0]

    return time, size, corr


def get_data_name():
    size = []
    time = []

    for name in config.videos_list:
        t, s, _ = get_data_fmt(name)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_fmt(name):
    size = []
    time = []

    for fmt in config.tile_list:
        t, s, corr = get_data_quality(name, fmt)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_quality(name, fmt):
    size = []
    time = []

    for quality in config.quality_list:
        t, s, corr = get_data_tiles(name, fmt, quality)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_name_chunk(fmt, tile, quality):
    size = []
    time = []

    for name in config.videos_list:
        t, s, corr = get_data_chunks(name, fmt, quality, tile)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_tiles(name, fmt, quality):
    size = []
    time = []
    m_, n_ = list(map(int, fmt.split('x')))

    for tile in range(1, m_ * n_ + 1):
        t, s, corr = get_data_chunks(name, fmt, quality, tile)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_chunks(name, fmt, quality, tile):
    size = []
    time = []

    # Lista todos os chunks
    for chunk in range(1, config.duration + 1):
        if name in 'ninja_turtles' and chunk > 58:
            size.append(0)
            time.append(0)
        dec = dectime[name]
        dec = dec[fmt]
        dec = dec[str(quality)]
        dec = dec[str(tile)]
        dec = dec[str(chunk)]
        s = dec['size']
        t = dec['times']
        size.append(float(s) * 8)
        time.append(float(t))

    # Plota os dados
    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


class Heatmap:
    def __init__(self, m, n):
        self.average_time = []
        self.average_size = []
        self.std_time = []
        self.std_size = []
        self.frame_time = np.zeros((n, m))
        self.frame_size = np.zeros((n, m))


def make_heatmap(name, fmt, quality):
    m, n = list(map(int, fmt.split('x')))
    heatmap = Heatmap(m, n)

    for tile, (x, y) in zip(range(1, m * n + 1),
                            it(range(1, m + 1),
                               range(1, n + 1))):
        # Coleta dados de todos os chunks
        time, size, _ = get_data_chunks(name, fmt, quality, tile)

        time = np.average(time)
        size = np.average(size)

        heatmap.average_time.append(time)
        heatmap.average_size.append(size)
        heatmap.std_time.append(np.std(time))
        heatmap.std_size.append(np.std(size))

        # Preenche a matriz do heatmap
        heatmap.frame_time[y - 1, x - 1] = time
        heatmap.frame_size[y - 1, x - 1] = size

    return heatmap


if __name__ == "__main__":
    main()
