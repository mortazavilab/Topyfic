import pandas as pd
import numpy as np
import anndata
import sys
from scipy import stats
import warnings
import random

from scipy.cluster.hierarchy import ward, dendrogram, leaves_list

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from Topyfic.topModel import *

sns.set_theme(style="white")
warnings.filterwarnings('ignore')


class Analysis:
    """
    A class used to investigate the topics and gene weights compositions
    :param Top_model: top model that used for analysing topics, gene weights compositions and calculate cell participation
    :type Top_model: TopModel
    :param colors_topics: dataframe that mapped colored to topics
    :type colors_topics: pandas dataframe
    :param cell_participation: anndata that stores cell participation along with cell information in obs
    :type cell_participation: anndata
    """
    def __init__(self,
                 Top_model,
                 colors_topics=None,
                 cell_participation=None):
        self.top_model = Top_model

        if colors_topics is None:
            colors = sns.color_palette("turbo", self.top_model.N).as_hex()

            def myfunction():
                return 0.1

            random.shuffle(colors, myfunction)
            index = list(self.top_model.topics.keys())
            self.colors_topics = pd.DataFrame({'colors': colors}, index=index)
        else:
            self.colors_topics = colors_topics

        self.cell_participation = cell_participation

    def calculate_cell_participation(self, data):
        if self.cell_participation is not None:
            print("cell participation is not empty!")
            print("new cell participation will be replaced!")

        lda_output = self.top_model.rLDA.transform(data.X)
        cell_participation = pd.DataFrame(lda_output,
                                          columns=[f"Topic_{i + 1}" for i in range(self.top_model.N)],
                                          index=data.obs.index)
        self.cell_participation = anndata.AnnData(cell_participation, obs=data.obs)

    def pie_structure_Chart(self,
                            level,
                            category=None,
                            ascending=None,
                            n=5,
                            save=True,
                            show=True,
                            figsize=None,
                            format_file="pdf",
                            file_name="piechart_topicAvgCell"):
        """
        plot pie charts that shows contribution of each topics to each category (i.e cell type)
        :param level: name of the column from cell_participation.obs
        :type level: str
        :param category: list of items you want to plot pie charts which are subsets of cell_participation.obs[level](default: all the unique items in cell_participation.obs[level])
        :type category: list of str
        :param ascending: for each pie chart on which order you want to sort your data (default is descending for all pie charts)
        :type ascending: list of bool
        :param n: number of topics you want to annotate in pie charts (default: 5)
        :type n: int
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param figsize: indicate the size of plot (default: (10 * (len(category) + 1), 10))
        :type figsize: tuple of int
        :param format_file: indicate the format of plot (default: pdf)
        :type format_file: str
        :param file_name: name and path of the plot use for save (default: piechart_topicAvgCell)
        :type file_name: str
        """
        if self.cell_participation is None:
            sys.exit("self.cell_participation is empty, you need to fill it by using 'calculate_cell_participation'")

        if category is None:
            category = self.cell_participation.obs[level].unique().tolist()

        if figsize is None:
            figsize = (10 * (len(category) + 1), 10)

        fig, axs = plt.subplots(ncols=len(category) + 1,
                                figsize=figsize,
                                facecolor='white')

        colors = self.colors_topics

        if ascending is None:
            ascending = [False] * len(category)

        for i in range(len(category)):
            tissue = self.cell_participation.obs[self.cell_participation.obs[level] == category[i]]
            tmp = self.cell_participation.to_df().loc[tissue.index, :]
            order = tmp.mean().sort_values(ascending=ascending[i]).index.tolist()
            index = tmp[order].sort_values(by=order, ascending=False).index.tolist()
            tmp = tmp.reindex(columns=order)
            tmp = tmp.reindex(index)
            colors = colors.reindex(order)
            labels = tmp.columns.tolist()
            labels[n:] = ['' for i in range(len(labels) - 5)]

            def make_autopct(values):
                def my_autopct(pct):
                    if pct > values[n] * 100:
                        return '{p:.0f}%'.format(p=pct)
                    else:
                        return ''

                return my_autopct

            axs[i].pie(tmp.mean(),
                       labels=labels,
                       colors=colors.colors.tolist(),
                       autopct=make_autopct(tmp.mean()),
                       wedgeprops={'linewidth': 0},
                       # labeldistance=0.8,
                       textprops={"fontsize": 25})

            axs[i].set_title(category[i], fontsize=40)

        colors = self.colors_topics
        handles = []
        for n in range(colors.shape[0]):
            patch = mpatches.Patch(color=colors.colors[n], label=colors.index[n])
            handles.append(patch)
            axs[len(category)].legend(loc='center left',
                                      title='Topic',
                                      ncol=3,
                                      handles=handles)
            axs[len(category)].axis('off')

        if save:
            fig.savefig(f"{file_name}.{format_file}")
        if show:
            plt.show()
        else:
            plt.close()

    def structure_plot(self,
                       level,
                       category,
                       ascending=None,
                       metaData=None,
                       metaData_palette=None,
                       width=None,
                       n=2,
                       order_cells=['hierarchy'],
                       save=True,
                       show=True,
                       figsize=None,
                       format_file="pdf",
                       file_name="structure_topicAvgCell"):
        """
        plot structure which shows contribution of each topics for each cells in given categories
        :param level: name of the column from cell_participation.obs
        :type level: str
        :param category: list of items you want to plot which are subsets of cell_participation.obs[level](default: all the unique items in cell_participation.obs[level])
        :type category: list of str
        :param ascending: for each structure plot on which order you want to sort your data (default is descending for all structure plot)
        :type ascending: list of bool
        :param metaData: if you want to add annotation for each cell add column name of that information (make sure you have that inforamtion in your cell_participation.obs)
        :type metaData: list
        :param metaData_palette: color palette for each metaData you add
        :type metaData_palette: dict
        :param width: width ratios of each category (default is based on the number of the cells we have in each category)
        :type width: list of int
        :param n: number of topics you want to annotate in pie charts (default: 5)
        :type n: int
        :param order_cells: determine which kind of sorting options you want to use ('sum', 'hierarchy', sort by metaData); sum: sort cells by sum of top n topics; hierarchy: sort data by doing hierarchical clustring; metaData sort by metaData (default: ['hierarchy'])
        :type order_cells: list
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param figsize: indicate the size of plot (default: (10 * (len(category) + 1), 10))
        :type figsize: tuple of int
        :param format_file: indicate the format of plot (default: pdf)
        :type format_file: str
        :param file_name: name and path of the plot use for save (default: piechart_topicAvgCell)
        :type file_name: str
        """
        if figsize is None:
            figsize = (10 * (len(category) + 1), 10)

        check = ["hierarchy", "sum"] + metaData
        print(check)
        for order_cell in order_cells:
            if order_cell not in check:
                return "order cell was not valid"

        a = []
        for i in range(len(category)):
            a.append(self.cell_participation.obs[self.cell_participation.obs[level] == category[i]].shape[0])
        a.append(min(a) / 2)
        if width is None:
            width = a
        b = [8] + [1] * len(metaData)
        fig, axs = plt.subplots(nrows=len(metaData) + 1,
                                ncols=len(category) + 1,
                                figsize=figsize,
                                gridspec_kw={'width_ratios': width,
                                             'height_ratios': b},
                                facecolor='white')

        colors = self.colors_topics

        if ascending is None:
            ascending = [True] * len(category)

        for i in range(len(category)):
            tissue = self.cell_participation.obs[self.cell_participation.obs[level] == category[i]]
            tmp = self.cell_participation.to_df().loc[tissue.index, :]
            order = tmp.mean().sort_values(ascending=False).index.tolist()
            # tmp['non_major'] = 1 - tmp.sum(axis=1)
            # tmp.non_major[tmp['non_major'] < 0] = 0
            index = tmp[order].sort_values(by=order, ascending=False).index.tolist()
            tmp = tmp.reindex(index)
            if len(order_cells) == 1:
                order_cell = order_cells[0]
                if order_cell == 'hierarchy':
                    Z = ward(tmp)
                    index = leaves_list(Z).tolist()
                    tmp = tmp.iloc[index, :]
                elif order_cell == 'sum':
                    index = tmp[order[:n]].sum(axis=1).sort_values(ascending=ascending[i]).index.tolist()
                    tmp = tmp.reindex(index)
                elif order_cell in metaData:
                    index = tissue.sort_values(by=[order_cell], ascending=ascending[i]).index.tolist()
                    tmp = tmp.reindex(index)
            else:
                order_cell = order_cells[:-1]
                tissue.sort_values(by=order_cell, ascending=ascending[i], inplace=True)
                tmp = tmp.reindex(tissue.index)
                tissue['count'] = 1
                groups = tissue.groupby(order_cell).sum().reset_index()[order_cell + ['count']]
                groups.sort_values(by=order_cell, ascending=ascending[i], inplace=True)
                count = 0
                index = []
                for j in range(groups.shape[0]):
                    sub = tissue.iloc[count:count + groups.loc[j, 'count'], :]
                    sub = tmp.loc[sub.index, :]
                    if sub.shape[0] > 1:
                        Z = ward(sub)
                        sub = leaves_list(Z).tolist()
                        sub = [x + count for x in sub]
                        index = index + sub
                    else:
                        index = index + [count]
                    count = count + groups.loc[j, 'count']

                tmp = tmp.iloc[index, :]

            # order.append('non_major')
            tmp = tmp.reindex(columns=order)

            colors = colors.reindex(order)
            if metaData is None:
                axs[i].stackplot(tmp.index.tolist(),
                                 tmp.T.to_numpy(),
                                 labels=tmp.columns.tolist(),
                                 colors=colors.colors.tolist(),
                                 linewidths=0)

                axs[i].xaxis.set_ticks([])
                axs[i].set_title(category[i], fontsize=40)
                axs[i].set_ylim(0, 1)
                axs[i].set_xlim(0, a[i])
                axs[0].set_ylabel("Topic proportion", fontsize=25)
            else:
                axs[0, i].stackplot(tmp.index.tolist(),
                                    tmp.T.to_numpy(),
                                    labels=tmp.columns.tolist(),
                                    colors=colors.colors.tolist(),
                                    linewidths=0)

                axs[0, i].xaxis.set_ticks([])
                axs[0, i].set_title(category[i], fontsize=40)
                axs[0, i].set_ylim(0, 1)
                axs[0, i].set_xlim(0, a[i])
                axs[0, 0].set_ylabel("Topic proportion", fontsize=25)

            tissue = tissue[metaData]
            tissue = tissue.reindex(tmp.index.tolist())
            for j in range(len(metaData)):
                tissue.replace(metaData_palette[metaData[j]], inplace=True)

            x = [i for i in range(tmp.shape[0])]
            y = np.repeat(3000, len(x))
            for j in range(len(metaData)):
                color = tissue[metaData[j]].values
                axs[j + 1, i].scatter(x, y, label=metaData_palette[metaData[j]],
                                      c=color, s=1000, marker="|", alpha=1,
                                      linewidths=1)
                axs[j + 1, i].axis('off')
                axs[j + 1, i].set_xlim(0, a[i])

        colors = self.colors_topics

        handles = []
        for n in range(colors.shape[0]):
            patch = mpatches.Patch(color=colors.colors[n], label=colors.index[n])
            handles.append(patch)
            if metaData is None:
                axs[len(category)].legend(loc='center left',
                                          title='Topic',
                                          ncol=3,
                                          handles=handles)
                axs[len(category)].axis('off')
            else:
                axs[0, len(category)].legend(loc='center left',
                                             title='Topic',
                                             ncol=4,
                                             handles=handles)
                axs[0, len(category)].axis('off')

        for j in range(len(metaData)):
            handles = []
            for met in metaData_palette[metaData[j]].keys():
                patch = mpatches.Patch(color=metaData_palette[metaData[j]][met], label=met)
                handles.append(patch)
                axs[j + 1, len(category)].legend(loc='center left',
                                                 title=metaData[j].capitalize(),
                                                 ncol=4,
                                                 handles=handles)
                axs[j + 1, len(category)].axis('off')

        if save:
            fig.savefig(f"{file_name}.{format_file}")
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def convertDatTraits(data):
        """
        get data trait module base on samples information
        :return: a dataframe contains information in suitable format for plotting module trait relationship heatmap
        :rtype: pandas dataframe
        """
        datTraits = pd.DataFrame(index=data.index)
        for i in range(data.shape[1]):
            data.iloc[:, i] = data.iloc[:, i].astype(str)
            if len(np.unique(data.iloc[:, i])) == 2:
                datTraits[data.columns[i]] = data.iloc[:, i]
                org = np.unique(data.iloc[:, i]).tolist()
                rep = list(range(len(org)))
                datTraits.replace(to_replace=org, value=rep,
                                  inplace=True)
            elif len(np.unique(data.iloc[:, i])) > 2:
                for name in np.unique(data.iloc[:, i]):
                    datTraits[name] = data.iloc[:, i]
                    org = np.unique(data.iloc[:, i])
                    rep = np.repeat(0, len(org))
                    rep[np.where(org == name)] = 1
                    org = org.tolist()
                    rep = rep.tolist()
                    datTraits.replace(to_replace=org, value=rep, inplace=True)

        return datTraits

    def TopicTraitRelationshipHeatmap(self,
                                      metaData,
                                      save=True,
                                      show=True,
                                      format_file="pdf",
                                      file_name='topic-traitRelationships'):
        """
        plot topic-trait relationship heatmap
        :param metaData: traits you would like to see the relationship with topics (must be column name of cell_participation.obs)
        :type metaData: list
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param format_file: indicate the format of plot (default: pdf)
        :type format_file: str
        :param file_name: name and path of the plot use for save (default: topic-traitRelationships)
        :type file_name: str
        """
        datTraits = Analysis.convertDatTraits(self.cell_participation.obs[metaData])

        topicsTraitCor = pd.DataFrame(index=self.cell_participation.to_df().columns,
                                      columns=datTraits.columns,
                                      dtype="float")
        topicsTraitPvalue = pd.DataFrame(index=self.cell_participation.to_df().columns,
                                         columns=datTraits.columns,
                                         dtype="float")

        for i in self.cell_participation.to_df().columns:
            for j in datTraits.columns:
                tmp = stats.pearsonr(self.cell_participation.to_df()[i], datTraits[j])
                topicsTraitCor.loc[i, j] = tmp[0]
                topicsTraitPvalue.loc[i, j] = tmp[1]

        fig, ax = plt.subplots(figsize=(topicsTraitPvalue.shape[0] * 1.5,
                                        topicsTraitPvalue.shape[1] * 1.5), facecolor='white')

        xlabels = self.cell_participation.to_df().columns
        ylabels = datTraits.columns

        # Loop over data dimensions and create text annotations.
        tmp_cor = topicsTraitCor.T.round(decimals=2)
        tmp_pvalue = topicsTraitPvalue.T.round(decimals=2)
        labels = (np.asarray(["{0}\n({1})".format(cor, pvalue)
                              for cor, pvalue in zip(tmp_cor.values.flatten(),
                                                     tmp_pvalue.values.flatten())])) \
            .reshape(topicsTraitCor.T.shape)

        sns.set(font_scale=1.5)
        res = sns.heatmap(topicsTraitCor.T, annot=labels, fmt="", cmap='RdBu_r',
                          vmin=-1, vmax=1, ax=ax, annot_kws={'size': 20, "weight": "bold"},
                          xticklabels=xlabels, yticklabels=ylabels)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=20, fontweight="bold", rotation=90)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize=20, fontweight="bold")
        plt.yticks(rotation=0)
        ax.set_title(f"Topic-trait Relationships heatmap",
                     fontsize=30, fontweight="bold")
        ax.set_facecolor('white')

        if save:
            fig.savefig(f"{file_name}.{format_file}", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def save_analysis(self, name="analysis", save_path=""):
        """
        save Analysis class as a pickle file
        :param name: name of the pickle file (default: analysis)
        :type name: str
        :param save_path: directory you want to use to save pickle file (default is saving near script)
        :type save_path: str
        """
        print(f"Saving analysis class as {name}.p")

        picklefile = open(f"{save_path}{name}.p", "wb")
        pickle.dump(self, picklefile)
        picklefile.close()
