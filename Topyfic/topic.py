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


