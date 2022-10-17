import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")

warnings.filterwarnings('ignore')


class Topic:
    """
    A class saved topic along with other useful information

    :param topic_id: ID of topic which is unique
    :type topic_id: str
    :param topic_name: name of the topic (default: topic_id)
    :type topic_name: str
    :param topic_gene_weights: dataframe that contains weights of the topics for each genes
    :type topic_gene_weights: pandas dataframe
    :param gene_information: dataframe that contains information of genes i.e gene biotype
    :type gene_information: pandas dataframe
    :param topic_information: dataframe that contains information of genes i.e cell state / cell type
    :type topic_information: pandas dataframe
    """
    def __init__(self,
                 topic_id,
                 topic_name=None,
                 topic_gene_weights=None,
                 gene_information=None,
                 topic_information=None):

        self.id = topic_id

        if topic_name is not None:
            self.name = topic_name
        else:
            self.name = topic_id

        self.gene_weights = topic_gene_weights

        if gene_information is None:
            self.gene_information = pd.DataFrame(index=topic_gene_weights.index)
        else:
            self.gene_information = gene_information

        if topic_information is None:
            self.topic_information = pd.DataFrame(index=topic_gene_weights.columns)
        else:
            self.topic_information = topic_information

    def update_gene_information(self, gene_information):
        """
        update/add genes information for each topics

        :param gene_information: dataframe contains genes information we would like to add/update (the index should be same as an index of gene_information in class)
        :type gene_information: pandas dataframe
        """
        same_columns = self.gene_information.columns.intersection(gene_information.columns)
        self.gene_information.drop(same_columns, axis=1, inplace=True)
        self.gene_information = pd.concat([self.gene_information, gene_information], axis=1)

    def plot_topic_composition(self,
                               biotype="biotype",
                               save=True,
                               show=True,
                               file_format="pdf",
                               file_name="topic_composition"):
        """
        plot topic composition

        :param biotype: name of the column in gene_weight to look for gene_biotype (default: biotype)
        :type biotype: str
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: topic_composition)
        :type file_name: str
        """
        minWeight = np.min(self.gene_weights)
        gene_weights = self.gene_weights[self.gene_weights > minWeight]

        gene_biotype = self.gene_information[biotype]

        columns = gene_biotype.unique().tolist()
        df = gene_weights.sort_values([self.id], axis=0, ascending=False)[self.id]
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
                tmp.loc[0, row['biotype']] = 1
            else:
                tmp = pd.DataFrame(res.loc[res.shape[0] - 1]).T
                tmp.reset_index(drop=True, inplace=True)
                tmp.loc[0, row['biotype']] = res[row['biotype']][res.shape[0] - 1] + 1
            res = res.append(tmp, ignore_index=True)

        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(max(res.shape[0] / 40, 10), 7), facecolor='white')
        sns.set(font_scale=1)
        sns.lineplot(data=res, dashes=False, ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1,
                  prop={'size': 15})
        ax.set_title(f"{self.id}", fontsize=30)
        ax.set_ylabel("#genes (log2)", fontsize=25)
        ax.set_yscale('log', basey=2)
        fig.tight_layout()

        if save:
            fig.savefig(f"{file_name}.{file_format}")
        if show:
            plt.show()
        else:
            plt.close()


