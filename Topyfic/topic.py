import warnings
import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings('ignore')


class Topic:

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
