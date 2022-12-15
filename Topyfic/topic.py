import sys
import warnings
import joblib
import pandas as pd
import numpy as np

import gseapy as gp
from gseapy.plot import dotplot
from reactome2py import analysis

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')

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
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: gene_composition)
        :type file_name: str
        """
        if type not in ["GO", "REACTOME"]:
            sys.exit("Type is not valid! it should be one of them GO, KEGG, REACTOME")

        if type == "GO" and sets is None:
            sys.exit("sets is empty! you should define it if you want to perform Gene Ontology")

        gene_weights = self.gene_weights
        gene_weights = gene_weights[gene_weights > gene_weights.min()]
        genes = gene_weights.dropna().index.tolist()
        if type in ["GO", "KEGG"]:
            enr = gp.enrichr(gene_list=genes,
                             gene_sets=sets,
                             organism=organism,
                             description='',
                             outdir=f"{file_name}",
                             cutoff=p_value)
            dotplot(enr.res2d,
                    title=f"Gene ontology in Topic {self.name}",
                    cmap='viridis_r',
                    cutoff=p_value,
                    ofname=f"{file_name}.{file_format}")
        else:
            numGeneModule = len(genes)
            genes = ",".join(genes)
            result = analysis.identifiers(ids=genes,
                                          species=organism,
                                          p_value=str(p_value))
            token = result['summary']['token']
            analysis.report(token,
                            path=f"{file_name}/",
                            file=f"{file_name}.{file_format}",
                            number='50',
                            species=organism)
            token_result = analysis.token(token,
                                          species=organism,
                                          p_value=str(p_value))

            print(
                f"{numGeneModule - token_result['identifiersNotFound']} out of {numGeneModule} identifiers in the sample were found in Reactome.")
            print(
                f"{token_result['resourceSummary'][0]['pathways']} pathways were hit by at least one of them, which {len(token_result['pathways'])} of them have p-value more than {p_value}.")
            print(f"Report was saved {file_name}!")
            print(f"For more information please visit https://reactome.org/PathwayBrowser/#/DTAB=AN&ANALYSIS={token}")

    def MA_plot(self, topic):
        df = self.gene_weights.copy(deep=True)
        df = pd.concat([df, topic.gene_weights.copy(deep=True)])


