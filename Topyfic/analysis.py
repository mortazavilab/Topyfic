import pandas as pd
import numpy as np
import anndata
import sys
from scipy import stats
import warnings
import random
from statsmodels.stats.multitest import fdrcorrection
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from Topyfic.topModel import *

sns.set_context('paper')
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
        """
        Calculate cell participation for give data

        :param data: processed expression data along with cells and genes/region information
        :type data: anndata
        """
        if self.cell_participation is not None:
            print("cell participation is not empty!")
            print("new cell participation will be replaced!")

        lda_output = self.top_model.rLDA.transform(data.X)
        cell_participation = pd.DataFrame(lda_output,
                                          columns=[f"Topic_{i + 1}" for i in range(self.top_model.N)],#list(self.top_model.topics.keys())
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
                            file_format="pdf",
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
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
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
            order = tmp.mean().sort_values(ascending=False).index.tolist()
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
            fig.savefig(f"{file_name}.{file_format}")
        if show:
            plt.show()
        else:
            plt.close()

    def structure_plot(self,
                       level,
                       category,
                       topic_order=None,
                       ascending=None,
                       metaData=None,
                       metaData_palette=None,
                       width=None,
                       n=2,
                       order_cells=['hierarchy'],
                       save=True,
                       show=True,
                       figsize=None,
                       file_format="pdf",
                       file_name="structure_topicAvgCell"):
        """
        plot structure which shows contribution of each topics for each cells in given categories

        :param level: name of the column from cell_participation.obs
        :type level: str
        :param category: list of items you want to plot which are subsets of cell_participation.obs[level](default: all the unique items in cell_participation.obs[level])
        :type category: list of str
        :param topic_order: indicate if you want to have a specific order of topics which it should be name of topics. if None, it's gonna sort by cell participation
        :type topic_order: list of str
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
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: piechart_topicAvgCell)
        :type file_name: str
        """
        if figsize is None:
            figsize = (10 * (len(category) + 1), 10)

        check = ["hierarchy", "sum"] + metaData
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
            if topic_order is None:
                order = tmp.mean().sort_values(ascending=False).index.tolist()
            else:
                order = topic_order
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
                    if groups.loc[j, 'count'] != 0:
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
                if type(metaData_palette[metaData[j]]) == dict:
                    tissue.replace(metaData_palette[metaData[j]], inplace=True)

            x = [i for i in range(tmp.shape[0])]
            y = np.repeat(3000, len(x))
            for j in range(len(metaData)):
                color = tissue[metaData[j]].values
                if type(metaData_palette[metaData[j]]) == dict:
                    axs[j + 1, i].scatter(x, y, label=metaData_palette[metaData[j]],
                                          c=color, s=1000, marker="|", alpha=1,
                                          linewidths=1)
                else:
                    axs[j + 1, i].scatter(x, y, label=metaData_palette[metaData[j]],
                                          c=color, cmap=metaData_palette[metaData[j]].get_cmap(), s=1000, marker="|",
                                          alpha=1,
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
            if type(metaData_palette[metaData[j]]) == dict:
                for met in metaData_palette[metaData[j]].keys():
                    patch = mpatches.Patch(color=metaData_palette[metaData[j]][met], label=met)
                    handles.append(patch)
                    axs[j + 1, len(category)].legend(loc='center left',
                                                     title=metaData[j].capitalize(),
                                                     ncol=4,
                                                     handles=handles)
                    axs[j + 1, len(category)].axis('off')
            else:
                clb = fig.colorbar(mappable=metaData_palette[metaData[j]],
                                   ax=axs[j + 1, len(category)],
                                   orientation='horizontal',
                                   fraction=0.9)
                clb.ax.set_title(metaData[j].capitalize())
                axs[j + 1, len(category)].axis('off')

        if save:
            fig.savefig(f"{file_name}.{file_format}")
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
                                      file_format="pdf",
                                      file_name='topic-traitRelationships'):
        """
        plot topic-trait relationship heatmap

        :param metaData: traits you would like to see the relationship with topics (must be column name of cell_participation.obs)
        :type metaData: list
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
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
                tmp = stats.pearsonr(self.cell_participation.to_df()[i], datTraits[j], alternative='greater')
                topicsTraitCor.loc[i, j] = tmp[0]
                topicsTraitPvalue.loc[i, j] = tmp[1]

        fig, ax = plt.subplots(figsize=(topicsTraitPvalue.shape[0] * 1.5,
                                        topicsTraitPvalue.shape[1] * 1.5), facecolor='white')

        xlabels = self.cell_participation.to_df().columns
        ylabels = datTraits.columns

        # print(topicsTraitPvalue)
        # for i in range(topicsTraitPvalue.shape[0]):
        #     rejected, tmp = fdrcorrection(topicsTraitPvalue.iloc[i, :])
        #     print(rejected)
        #     if not rejected:
        #         topicsTraitPvalue.iloc[i, :] = tmp
        # print(topicsTraitPvalue)

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
            fig.savefig(f"{file_name}.{file_format}", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def extract_cells(self,
                      level,
                      category,
                      top_cells=0.05,
                      min_cell_participation=0.05,
                      min_cells=50,
                      file_name=None,
                      save=False):
        """
        extract subset of cells and cells participation with specific criteria

        :param level: name of the column from cell_participation.obs
        :type level: str
        :param category: list of items you want to plot which are subsets of cell_participation.obs[level](default: all the unique items in cell_participation.obs[level])
        :type category: list of str
        :param top_cells: fraction of the cells you want to be considers (default: 0.05)
        :type top_cells: float
        :param min_cell_participation: minimum cell participation each cells in each topics should have to be count (default: 0.05)
        :type min_cell_participation: float
        :param min_cells: minimum number of cells each topics should have to be reported (default: 50)
        :type min_cells: int
        :param file_name: name and path of the plot use for save (default: selectedCells_top{top_cells}_{min_cell_score}min_score_{min_cells}min_cells.csv and cellParticipation_selectedCells_top{top_cells}_{min_cell_score}min_score_{min_cells}min_cells.csv)
        :type file_name: str
        :param save: indicate if you want to save the data or not (default: False)
        :type save: bool

        :return: table contains cell ID that pass threshold for each topic, table contains cell particiaption for cells that pass threshold for each topic (same order as fist table)
        :rtype: pandas dataframe, pandas dataframe
        """
        cells = self.cell_participation.obs[self.cell_participation.obs[level].isin(category)]
        cell_topics = self.cell_participation.to_df().loc[cells.index, :]
        cell_topics = cell_topics[cell_topics > min_cell_participation]
        res = cell_topics.count() * top_cells
        res = res.round()

        selected_cell = pd.DataFrame(index=range(cells.shape[0]),
                                     columns=cell_topics.columns)
        selected_cell_participation = pd.DataFrame(index=range(cells.shape[0]),
                                                   columns=cell_topics.columns)

        for i in range(cell_topics.shape[1]):
            tmp = cell_topics.sort_values(cell_topics.columns[i], ascending=False)
            selected_cell[tmp.columns[i]][:int(res[tmp.columns[i]])] = tmp.index.tolist()[:int(res[tmp.columns[i]])]
            a = tmp[:int(res[tmp.columns[i]])]
            selected_cell_participation[tmp.columns[i]][:int(res[tmp.columns[i]])] = a[tmp.columns[i]].values.tolist()

        selected_cell.dropna(axis=0, how='all', inplace=True)
        selected_cell.dropna(axis=1, how='all', inplace=True)
        selected_cell = selected_cell[selected_cell.columns[selected_cell.count() > min_cells]]
        selected_cell.fillna("", inplace=True)

        selected_cell_participation.dropna(axis=0, how='all', inplace=True)
        selected_cell_participation.dropna(axis=1, how='all', inplace=True)
        selected_cell_participation = selected_cell_participation[
            selected_cell_participation.columns[selected_cell_participation.count() > min_cells]]
        selected_cell_participation.fillna("", inplace=True)

        if save:
            if file_name is None:
                selected_cell.to_csv(
                    f"selectedCells_top{top_cells}_{min_cell_participation}min_score_{min_cells}min_cells.csv")
                selected_cell_participation.to_csv(
                    f"cellParticipation_selectedCells_top{top_cells}_{min_cell_participation}min_score_{min_cells}min_cells.csv")
            else:
                selected_cell.to_csv(f"selectedCells_{file_name}.csv")
                selected_cell_participation.to_csv(f"cellParticipation_selectedCells_{file_name}.csv")

        else:
            return selected_cell, selected_cell_participation

    def average_cell_participation(self,
                                   label=None,
                                   color="blue",
                                   save=True,
                                   show=True,
                                   figsize=None,
                                   file_format="pdf",
                                   file_name="average_cell_participation"):
        """
        barplot showing average of cell participation in each topic

        :param label: fill with dictionary contain mapping new name for each topics to name you want to show if you want to change default topic name
        :type label: dict
        :param color: color of bar plot (default: blue)
        :type color: str
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param figsize: indicate the size of plot (default: (10 * (len(category) + 1), 10))
        :type figsize: tuple of int
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: piechart_topicAvgCell)
        :type file_name: str
        """
        df = pd.DataFrame(self.cell_participation.to_df().mean())
        df.reset_index(inplace=True)
        if label is not None:
            df.replace(label, inplace=True)

        if figsize is None:
            figsize = (df.shape[0] / 1.5, 5)

        fig = plt.figure(figsize=figsize, facecolor='white')

        plt.bar(df['index'], df[0], color=color, width=0.5)

        plt.xlabel(self.top_model.name)
        plt.ylabel("Average cell participation")

        if save:
            fig.savefig(f"{file_name}.{file_format}", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def average_cell_participation_line_plot(self,
                                             topic,
                                             color,
                                             category,
                                             color_pallet=None,
                                             save=True,
                                             show=True,
                                             figsize=None,
                                             file_format="pdf",
                                             file_name="line_average_cell_participation"):
        """
        line plot showing average of cell participation in topic divided by two features of cells (i.e. cell type and time point)

        :param topic: name of the topic
        :type topic: str
        :param color: name of the feature you want to have one line per group of that (it should be column name of cell_participation.obs)
        :type color:str
        :type color_pallet: color of each category of color (if it None color assign randomly)
        :param color_pallet: dict
        :param category: name of the feature you want to have on x axis (it should be column name of cell_participation.obs)
        :type category: str
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param figsize: indicate the size of plot (default: (10 * (len(category) + 1), 10))
        :type figsize: tuple of int
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: piechart_topicAvgCell)
        :type file_name: str
        """

        if topic not in self.top_model.topics.keys():
            sys.exit("topic is not valid!")

        if self.cell_participation is None:
            sys.exit("Cell participation is not calculated yet!")

        if category not in self.cell_participation.obs.columns:
            sys.exit("category is not valid!")

        if color not in self.cell_participation.obs.columns:
            sys.exit("color is not valid!")

        if color_pallet is None:
            color_pallet = dict()
            for name in self.cell_participation.obs[color].unique():
                color_pallet[name] = "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])

        res = pd.DataFrame(columns=self.cell_participation.obs[category].unique(),
                           index=self.cell_participation.obs[color].unique())

        for index in res.index:
            for col in res.columns:
                tmp = self.cell_participation[np.logical_and(self.cell_participation.obs[color] == index,
                                                             self.cell_participation.obs[category] == col)]
                res.loc[index, col] = tmp.to_df().mean()[topic]
        res.fillna(0, inplace=True)

        if figsize is None:
            figsize = (res.shape[0] / 1.5, 7)

        fig = plt.figure(figsize=figsize, facecolor='white')

        for index in res.index:
            plt.plot(res.columns.tolist(), res.loc[index, :].tolist(),
                     '-o',
                     label=index,
                     color=color_pallet[index])

        plt.legend(loc='center left', title=color, bbox_to_anchor=(1, 0.5))
        plt.xlabel(category)
        plt.ylabel("Average cell participation")
        plt.title(topic)

        if save:
            fig.savefig(f"{file_name}.{file_format}", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_topic_composition(self,
                               category,
                               level="topic",
                               biotype="biotype",
                               label=False,
                               save=True,
                               show=True,
                               file_format="pdf",
                               file_name="gene_composition"):
        """
        plot gene composition dividing by gene biotype or topics

        :param category: topic name or gene biotype name you want to see gene composition for
        :type category: str
        :param level: indicate weather if you want to show it within each topic or gene biotype (options: "topic" or "gene_biotype") (default: topic)
        :type level: str
        :param biotype: name of the column in gene_weight to look for gene_biotype (default: biotype)
        :type biotype: str
        :param label: show label of each line within plot or not (default: False)
        :type label: bool
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: gene_composition)
        :type file_name: str
        """
        if level not in ["topic", "gene_biotype"]:
            sys.exit("level is not correct! it should be topic or gene_biotype!")
        minWeight = np.min(self.top_model.get_gene_weights())
        gene_weights = self.top_model.get_gene_weights()
        gene_weights = gene_weights[gene_weights > minWeight]

        if level == "topic":
            gene_biotype = self.top_model.topics[category].gene_information[biotype]
            columns = gene_biotype.unique().tolist()
            df = gene_weights.sort_values([f"{self.top_model.name}_{category}"], axis=0, ascending=False)[
                f"{self.top_model.name}_{category}"]

            df = pd.DataFrame(df)
            df.dropna(axis=0, inplace=True)
            df.reset_index(inplace=True)
            df.columns = ['genes', 'weight']
            df.index = df['genes']
            df = pd.concat([df, gene_biotype], axis=1)
            df.dropna(axis=0, inplace=True)

            res = pd.DataFrame(columns=columns, dtype=int)
            for index, row in df.iterrows():
                # tmp.loc[0, 'weight'] = row['weight']
                if res.shape[0] == 0:
                    tmp = pd.DataFrame(0, index=[0], columns=columns, dtype=int)
                    tmp.loc[0, row[biotype]] = 1
                else:
                    tmp = pd.DataFrame(res.loc[res.shape[0] - 1]).T
                    tmp.reset_index(drop=True, inplace=True)
                    tmp.loc[0, row[biotype]] = res[row[biotype]][res.shape[0] - 1] + 1
                res = res.append(tmp, ignore_index=True)

        else:
            gene_weights.columns = [f'Topic_{i + 1}' for i in range(gene_weights.shape[1])]
            res = pd.DataFrame(columns=gene_weights.columns,
                               index=range(gene_weights.shape[0]),
                               dtype=int)
            gene_biotype = self.top_model.topics["Topic_1"].gene_information[biotype]
            columns = gene_biotype.unique().tolist()
            for col in gene_weights.columns:
                df = gene_weights.sort_values([col], axis=0, ascending=False)[col]

                df = pd.DataFrame(df)
                df.dropna(axis=0, inplace=True)
                df.reset_index(inplace=True)
                df.columns = ['genes', 'weight']
                df.index = df['genes']
                df = pd.concat([df, gene_biotype], axis=1)
                df.dropna(axis=0, inplace=True)

                sres = pd.DataFrame(columns=columns, dtype=int)
                for index, row in df.iterrows():
                    # tmp.loc[0, 'weight'] = row['weight']
                    if sres.shape[0] == 0:
                        tmp = pd.DataFrame(0, index=[0], columns=columns, dtype=int)
                        tmp.loc[0, row[biotype]] = 1
                    else:
                        tmp = pd.DataFrame(sres.loc[sres.shape[0] - 1]).T
                        tmp.reset_index(drop=True, inplace=True)
                        tmp.loc[0, row[biotype]] = sres[row[biotype]][sres.shape[0] - 1] + 1
                    sres = sres.append(tmp, ignore_index=True)

                res.loc[range(sres.shape[0]), col] = sres[category].values
            res.dropna(axis=0, how='all', inplace=True)

        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(max(res.shape[0] / 50, 10), max(res.shape[0] / 100, 10)),
                               facecolor='white')
        sns.set(font_scale=1)
        if level == "topic":
            sns.lineplot(data=res, dashes=False, ax=ax)
        else:
            sns.lineplot(data=res, dashes=False, ax=ax, palette=self.colors_topics.to_dict()['colors'])
        ax.set_yscale('log', basey=2)
        if label:
            for line, name in zip(ax.lines, res.columns):
                y = int(line.get_ydata()[-1])
                if y == 0:
                    continue
                x = int(line.get_xdata()[-1])
                ax.text(x, y, name, color=line.get_color())
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1,
                  prop={'size': 15})
        ax.set_title(f"{self.top_model.name}_{category}", fontsize=25)
        ax.set_ylabel("#genes (log2)", fontsize=20)
        ax.set_xlabel("gene rank", fontsize=20)
        fig.tight_layout()

        if save:
            fig.savefig(f"{file_name}.{file_format}")
        if show:
            plt.show()
        else:
            plt.close()

    def save_analysis(self, name=None, save_path=""):
        """
        save Analysis class as a pickle file

        :param name: name of the pickle file (default: analysis_Analysis.top_model.name)
        :type name: str
        :param save_path: directory you want to use to save pickle file (default is saving near script)
        :type save_path: str
        """
        if name is None:
            name = f"analysis_{self.top_model.name}"
        print(f"Saving analysis class as {name}.p")

        picklefile = open(f"{save_path}{name}.p", "wb")
        pickle.dump(self, picklefile)
        picklefile.close()
