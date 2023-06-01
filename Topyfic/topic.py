import sys
import warnings
import pandas as pd
import yaml
from yaml.loader import SafeLoader

import seaborn as sns

sns.set_context('paper')

warnings.filterwarnings('ignore')

from Topyfic.utilsAnalyseModel import GSEA, functional_enrichment_analysis


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

    def functional_enrichment_analysis(self,
                                       type,
                                       organism,
                                       sets=None,
                                       p_value=0.05,
                                       file_format="pdf",
                                       file_name="functional_enrichment_analysis"):
        """
        Doing functional enrichment analysis including GO, KEGG and REACTOME

        :param type: indicate the type of databases which it should be one of "GO", "REACTOME"
        :type type: str
        :param organism: name of the organ you want to do functional enrichment analysis
        :type organism: str
        :param sets: str, list, tuple of Enrichr Library name(s). (you can add any Enrichr Libraries from here: https://maayanlab.cloud/Enrichr/#stats) only need to fill if the type is GO
        :type sets: str, list, tuple
        :param p_value: Defines the pValue threshold. (default: 0.05)
        :type p_value: float
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: gene_composition)
        :type file_name: str
        """
        gene_weights = self.gene_weights.copy(deep=True)
        gene_weights = gene_weights[gene_weights > gene_weights.min()]
        genes = gene_weights.dropna().index.tolist()

        functional_enrichment_analysis(gene_list=genes,
                                       type=type,
                                       organism=organism,
                                       sets=sets,
                                       p_value=p_value,
                                       file_format=file_format,
                                       file_name=file_name)

    def GSEA(self,
             gene_sets='GO_Biological_Process_2021',
             p_value=0.05,
             table=True,
             plot=True,
             file_format="pdf",
             file_name="GSEA",
             **kwargs):
        """
        Doing Gene Set Enrichment Analysis on based on the topic weights using GSEAPY package.

        :param gene_sets: Enrichr Library name or .gmt gene sets file or dict of gene sets. (you can add any Enrichr Libraries from here: https://maayanlab.cloud/Enrichr/#stats)
        :type gene_sets: str, list, tuple
        :param p_value: Defines the pValue threshold for plotting. (default: 0.05)
        :type p_value: float
        :param table: indicate if you want to save all GO terms that passed the threshold as a table (default: True)
        :type table: bool
        :param plot: indicate if you want to plot all GO terms that passed the threshold (default: True)
        :type plot: bool
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: gene_composition)
        :type file_name: str
        :param kwargs: Argument to pass to gseapy.prerank(). more info: https://gseapy.readthedocs.io/en/latest/run.html?highlight=gp.prerank#gseapy.prerank

        :return: dataframe contains these columns: Term: gene set name, ES: enrichment score, NES: normalized enrichment score, NOM p-val:  Nominal p-value (from the null distribution of the gene set, FDR q-val: FDR qvalue (adjusted False Discory Rate), FWER p-val: Family wise error rate p-values, Tag %: Percent of gene set before running enrichment peak (ES), Gene %: Percent of gene list before running enrichment peak (ES), Lead_genes: leading edge genes (gene hits before running enrichment peak)
        :rtype: pandas dataframe
        """

        gene_weights = self.gene_weights.copy(deep=True)
        gene_weights.index = gene_weights.index.str.upper()
        gene_weights.sort_values([self.id], ascending=False, inplace=True)
        gene_weights = gene_weights[gene_weights > gene_weights.min()]
        gene_weights.dropna(inplace=True)

        if gene_weights.shape[0] == 1:
            return

        GSEA_df = GSEA(gene_list=gene_weights,
                       gene_sets=gene_sets,
                       p_value=p_value,
                       table=table,
                       plot=plot,
                       file_format=file_format,
                       file_name=file_name,
                       **kwargs)

        return GSEA_df

    def gene_weight_variance(self, save=True):
        """
        calculate the gene weight variance

        :param save: added as an information to the Topic (default: True)
        :type save: bool

        :return: Gene weight variance for given topic
        :rtype: float
        """
        variance = self.gene_weights.var().tolist()[0]

        if save:
            self.topic_information['variance'] = self.gene_weights.var().tolist()[0]

        return print(f"Gene weight variance for given topic is {variance}")

    def write_topic_yaml(self, topic_id=None, model_yaml_path="model.yaml", topic_yaml_path="topic.yaml", save=True):
        """
        write topic in YAML format

        :param topic_id: unique topic ID (default is topic ID)
        :type topic_id: str
        :param model_yaml_path: model yaml path that has information about the dataset you use
        :type model_yaml_path: str
        :param topic_yaml_path: path that you use to save topic
        :type topic_yaml_path: str
        :param save: indicate if you want to save yaml file (True) or just show them (Fasle) (default: True)
        :type save: bool
        """

        # check require columns
        cols = self.gene_information.reset_index().columns
        if not {'gene_name', 'gene_id'}.issubset(cols):
            sys.exit(f"Gene information doesn't contain gene_name and gene_id columns!")

        # Open the file and load the file
        with open(model_yaml_path) as f:
            model_yaml = yaml.load(f, Loader=SafeLoader)

        if topic_id not in model_yaml['Topic IDs']:
            sys.exit("Topic_id is not in model YAML file!")

        topic_yaml = {'Topic ID': topic_id,
                      'Gene weights': self.gene_weights.to_dict()[self.id],
                      'Gene information': self.gene_information.to_dict(),
                      'Topic information': self.topic_information.T.to_dict()[self.id]}

        if save:
            file = open(topic_yaml_path, "w")
            yaml.dump(topic_yaml, file, default_flow_style=False)
            file.close()
        else:
            yaml_string = yaml.dump(topic_yaml)
            print("The Topic YAML is:")
            print(yaml_string)
