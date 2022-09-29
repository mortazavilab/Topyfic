import pandas as pd
import numpy as np
import anndata
import sys
from scipy.stats import t
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

    def __init__(self,
                 Top_model,
                 colors_topics=None):
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

        self.cell_participation = None

    def calculate_cell_participation(self, data):

        lda_output = self.top_model.rLDA.transform(data.X)
        cell_participation = pd.DataFrame(lda_output,
                                          columns=[f"Topic_{i + 1}" for i in range(self.top_model.N)],
                                          index=data.obs.index)
        self.cell_participation = anndata.AnnData(cell_participation, obs=data.obs)

    def pie_structure_Chart(self,
                            level='subtypes',
                            category=[],
                            ascending=None,
                            n=5,
                            save=True,
                            show=True,
                            figsize=None,
                            format_file="pdf",
                            file_name="piechart_topicAvgCell"):

        if figsize is None:
            figsize = (10 * (len(category) + 1), 10)

        fig, axs = plt.subplots(ncols=len(category) + 1,
                                figsize=figsize,
                                facecolor='white')

        colors = self.colors_topics

        if ascending is None:
            ascending = [True] * len(category)

        for i in range(len(category)):
            tissue = self.cell_participation.obs[self.cell_participation.obs[level] == category[i]]
            tmp = self.cell_participation.to_df().loc[tissue.index, :]
            order = tmp.mean().sort_values(ascending=False).index.tolist()
            index = tmp[order].sort_values(by=order, ascending=False).index.tolist()
            tmp = tmp.reindex(index)
            index = tmp[order[:2]].sum(axis=1).sort_values(ascending=ascending[i]).index.tolist()
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
                       level='subtypes',
                       category=[],
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

        if figsize is None:
            figsize = (10 * (len(category) + 1), 10)

        check = ['hierarchy', 'sum'] + metaData
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

    @staticmethod
    def corPvalue(cor, nSamples):
        T = np.sqrt(nSamples - 2) * (cor / np.sqrt(1 - (cor ** 2)))
        pt = 1 - pd.DataFrame(t.cdf(np.abs(T), nSamples - 2), index=T.index, columns=T.columns)
        return 2 * pt

    def TopicTraitRelationshipHeatmap(self,
                                      metadata,
                                      save=True,
                                      show=True,
                                      format_file="pdf",
                                      file_name='topic-traitRelationships'):
        # Define numbers of genes and cells
        nCells = self.cell_participation.shape[0]

        datTraits = Analysis.convertDatTraits(self.cell_participation.obs[metadata])

        names = np.concatenate((self.cell_participation.to_df().columns, datTraits.columns))
        topicsTraitCor = pd.DataFrame(np.corrcoef(self.cell_participation.to_df().T, datTraits.T),
                                      index=names, columns=names)
        topicTraitCor = topicsTraitCor.iloc[0:self.cell_participation.shape[1], self.cell_participation.shape[1]:]
        topicTraitPvalue = Analysis.corPvalue(topicTraitCor, nCells)

        fig, ax = plt.subplots(figsize=(max(30, topicTraitPvalue.shape[0] * 1.5),
                                        topicTraitPvalue.shape[1] * 1.5), facecolor='white')

        xlabels = self.cell_participation.to_df().columns
        ylabels = datTraits.columns

        # Loop over data dimensions and create text annotations.
        tmp_cor = topicTraitCor.T.round(decimals=2)
        topicTraitPvalue[topicTraitPvalue == 0] = sys.float_info.epsilon
        tmp_pvalue = (topicTraitPvalue.apply(np.log10))
        tmp_pvalue = -1 * tmp_pvalue
        tmp_pvalue = tmp_pvalue.T.round(decimals=2)
        labels = (np.asarray(["{0}\n({1})".format(cor, pvalue)
                              for cor, pvalue in zip(tmp_cor.values.flatten(),
                                                     tmp_pvalue.values.flatten())])) \
            .reshape(topicTraitCor.T.shape)

        sns.set(font_scale=1.5)
        res = sns.heatmap(topicTraitCor.T, annot=labels, fmt="", cmap='RdBu_r',
                          vmin=-1, vmax=1, ax=ax, annot_kws={'size': 20, "weight": "bold"},
                          xticklabels=xlabels, yticklabels=ylabels)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=20, fontweight="bold", rotation=90)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize=20, fontweight="bold")
        plt.yticks(rotation=0)
        ax.set_title(f"Topic-trait Relationships heatmap",
                     fontsize=30, fontweight="bold")
        ax.set_facecolor('white')

        if save:
            fig.savefig(f"{file_name}.{format_file}")
        if show:
            plt.show()
        else:
            plt.close()

    def save_analysis(self, name="analysis", save_path=""):

        print(f"Saving analysis class as {name}.p")

        picklefile = open(f"{save_path}{name}.p", "wb")
        pickle.dump(self, picklefile)
        picklefile.close()
