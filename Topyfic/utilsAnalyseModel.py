import sys
import random
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import warnings
import joblib
import os
from multiprocessing import Pool
from itertools import repeat
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from scipy import stats as st
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import scanpy.external as sce
import networkx as nx
import math
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from gseapy.plot import dotplot
from gseapy import gseaplot
from reactome2py import analysis
from adjustText import adjust_text
import umap
import obonet
import plotly.express as px

warnings.filterwarnings("ignore")


def compare_topModels(topModels,
                      comparison_method="Jensen–Shannon divergence",
                      output_type='graph',
                      threshold=0.8,
                      topModels_color=None,
                      topModels_label=None,
                      ignore_genes=True,
                      save=False,
                      plot_show=True,
                      figsize=None,
                      plot_format="pdf",
                      file_name="compare_topics"):
    """
    compare topModels using topic gene weights

    :param topModels: list of topModel class you want to compare to each other
    :type topModels: list of TopModel class
    :param comparison_method: indicate the method you want to use for comparing topics. if you used Jensen–Shannon, we show -log2 (options: pearson correlation, spearman correlation, Jensen–Shannon divergence, cosine similarity)
    :type comparison_method: str
    :param output_type: indicate the type of output you want. graph: plot as a graph, heatmap: plot as a heatmap, table: table contains correlation. Note: if you want to plot Jensen–Shannon divergence as a graph, we convert the values to be at the -log2(), so you need to take that account for defining threshold
    :type output_type: str
    :param threshold: only apply when you choose circular which only show correlation above that
    :type threshold: float
    :param topModels_color: dictionary of colors mapping each topics to each color (default: blue)
    :type topModels_color: dict
    :param topModels_label: dictionary of label mapping each topics to each label
    :type topModels_label: dict
    :param ignore_genes: idicate how to behave to the genes that only represent in one the topics. "True" means it's gonna ignore those genes and "False" means it's gonna asumne the weights are zero for those genes that we don't have any weights in one of the mouse models
    :type ignore_genes: bool
    :param save: indicate if you want to save the plot or not (default: True)
    :type save: bool
    :param plot_show: indicate if you want to show the plot or not (default: True)
    :type plot_show: bool
    :param figsize: indicate the size of plot (default: (10 * (len(category) + 1), 10))
    :type figsize: tuple of int
    :param plot_format: indicate the format of plot (default: pdf)
    :type plot_format: str
    :param file_name: name and path of the plot use for save (default: compare_topics)
    :type file_name: str

    :return: table contains correlation between topics only when table is choose and save is False
    :rtype: pandas dataframe
    """
    if output_type not in ['graph', 'heatmap', 'table']:
        sys.exit("output_type is not valid! it should be one of 'graph', 'heatmap', or 'table'")

    if comparison_method not in ['spearman correlation', 'pearson correlation', 'Jensen–Shannon divergence', 'cosine similarity']:
        sys.exit("comparison_method is not valid! it should be one of 'spearman correlation', 'pearson correlation', or 'Jensen–Shannon divergence'")

    names = [topModel.name for topModel in topModels]
    if len(names) != len(set(names)):
        sys.exit("Name of the TopModels should be unique!")

    all_gene_weights = None

    for topModel in topModels:
        gene_weights = topModel.get_gene_weights()
        if all_gene_weights is None:
            all_gene_weights = gene_weights
        else:
            all_gene_weights = pd.concat([gene_weights, all_gene_weights], axis=1)

    corrs = pd.DataFrame(index=all_gene_weights.columns,
                         columns=all_gene_weights.columns,
                         dtype='float64')

    for d1 in all_gene_weights.columns.tolist():
        for d2 in all_gene_weights.columns.tolist():
            if d1 == d2:
                corrs.at[d1, d2] = 1
                continue
            a = all_gene_weights[[d1, d2]]
            if ignore_genes:
                a.dropna(axis=0, how='any', inplace=True)
            else:
                a.dropna(axis=0, how='all', inplace=True)
                a.fillna(0, inplace=True)

            a = a[np.logical_or(a[d1] > a[d1].min(), a[d2] > a[d2].min())]
            a = a / a.sum()
            if comparison_method == "Jensen–Shannon divergence":
                JSd = distance.jensenshannon(a[d1].tolist(), a[d2].tolist())
                corrs.loc[d1, d2] = JSd * JSd
            elif comparison_method == "pearson correlation":
                corr = st.pearsonr(a[d1].tolist(), a[d2].tolist())
                corrs.at[d1, d2] = corr[0]
            elif comparison_method == "spearman correlation":
                corr = st.spearmanr(a[d1].tolist(), a[d2].tolist())
                corrs.at[d1, d2] = corr[0]
            elif comparison_method == 'cosine similarity':
                corr = distance.cosine(a[d1].tolist(), a[d2].tolist())
                corrs.at[d1, d2] = 1 - corr
    if comparison_method == "Jensen–Shannon divergence":
        corrs = corrs.applymap(math.log2)
        corrs = corrs * -1

    if output_type == 'table':
        if save:
            corrs.to_csv(f"{file_name}.csv")
        else:
            return corrs

    if output_type == 'heatmap':
        if topModels_label is not None:
            corrs.rename(columns=topModels_label, index=topModels_label, inplace=True)
        sns.clustermap(corrs,
                       figsize=figsize)
        if save:
            plt.savefig(f"{file_name}.{plot_format}")
        if plot_show:
            plt.show()
        else:
            plt.close()

        return

    if output_type == 'graph':
        np.fill_diagonal(corrs.values, 0)
        corrs[corrs < threshold] = np.nan
        res = corrs.stack()
        res = pd.DataFrame(res)
        res.reset_index(inplace=True)
        res.columns = ['source', 'dest', 'weight']
        res['weight'] = res['weight'].astype(float).round(decimals=2)
        res['source_label'] = res['source']
        res['dest_label'] = res['dest']
        res['source_color'] = res['source']
        res['dest_color'] = res['dest']

        if topModels_label is not None:
            res['source_label'].replace(topModels_label, inplace=True)
            res['dest_label'].replace(topModels_label, inplace=True)
        if topModels_color is None:
            res['source_color'] = "blue"
            res['dest_color'] = "blue"
        else:
            res['source_color'].replace(topModels_color, inplace=True)
            res['dest_color'].replace(topModels_color, inplace=True)

        G = nx.Graph()
        for i in range(res.shape[0]):
            G.add_node(res.source_label[i], color=res.source_color[i])
            G.add_node(res.dest_label[i], color=res.dest_color[i])
            G.add_edge(res.source_label[i], res.dest_label[i], weight=res.weight[i])

        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

        if figsize is None:
            figsize = (len(connected_components) * 2, len(connected_components) * 2)

        nrows = math.ceil(math.sqrt(len(connected_components)))
        ncols = math.ceil(len(connected_components) / nrows)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor='white')

        i = 0
        for connected_component in connected_components:
            g_connected_component = G.subgraph(connected_component)
            nodePos = nx.spring_layout(g_connected_component)

            edge_labels = nx.get_edge_attributes(g_connected_component, "weight")

            node_color = nx.get_node_attributes(g_connected_component, "color").values()
            weights = nx.get_edge_attributes(g_connected_component, 'weight').values()

            if len(connected_components) == 1:
                ax = axs
            elif ncols == 1:
                ax = axs[int(i / ncols)]
            else:
                ax = axs[int(i / ncols), i % ncols]

            nx.draw_networkx(g_connected_component,
                             pos=nodePos,
                             width=list(weights),
                             with_labels=True,
                             node_color=list(node_color),
                             font_size=8,
                             node_size=500,
                             font_family='Arial',
                             ax=ax)

            nx.draw_networkx_edge_labels(g_connected_component,
                                         nodePos,
                                         edge_labels=edge_labels,
                                         font_size=7,
                                         ax=ax)

            i += 1

        if len(connected_components) == 1:
            ax.axis('off')
        else:
            [axi.axis('off') for axi in axs.ravel()]
        plt.tight_layout()
        if save:
            plt.savefig(f"{file_name}.{plot_format}")
        if plot_show:
            plt.show()
        else:
            plt.close()

        return axs


def MA_plot(topic1,
            topic2,
            size=None,
            pseudocount=1,
            threshold=1,
            cutoff=2.0,
            consistency_correction=1.4826,
            #topN=None,
            labels=None,
            save=True,
            show=True,
            file_format="pdf",
            file_name="MA_plot"):
    """
        plot MA based on the gene weights on given topics

        :param topic1: gene weight of first topic to be compared
        :type topic1: pandas.series
        :param topic2: gene weight of second topic to be compared
        :type topic2: pandas.series
        :param size: table contains size of dot for each genes (genes are index)
        :type size: pandas dataframe
        :param pseudocount: pseudocount that you want to add (default: 1)
        :type pseudocount: float
        :param threshold: threshold to filter genes based on A values (default: 1)
        :type threshold: float
        :param cutoff: cutoff for categorized genes by modified z-score (default: 2)
        :type cutoff: float
        :param consistency_correction: the factor converts the MAD to the standard deviation for a given distribution. The default value (1.4826) is the conversion factor if the underlying data is normally distributed
        :type consistency_correction: float
        :param topN: number of genes to be consider for calculating z-score based on the A value (if it's none is gonna be avarage of # genes in both topics with weights above threshold
        :type topN: int
        :param labels: list of gene names wish to show in MA-plot
        :type labels: list
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: MA_plot)
        :type file_name: str

        :return: return M and A values
        """
    topic1 += pseudocount
    topic2 += pseudocount

    A = (np.log2(topic1) + np.log2(topic2)) / 2
    M = np.log2(topic1) - np.log2(topic2)

    # if topN is None:
    #     len_topic1 = sum(topic1 > threshold)
    #     len_topic2 = sum(topic2 > threshold)
    #     topN = round((len_topic1 + len_topic2) / 2)

    gene_zscore = pd.concat([A, M], axis=1)
    gene_zscore.columns = ["A", "M"]
    gene_zscore = gene_zscore[gene_zscore.A > threshold]

    if size is not None:
        size = size.loc[gene_zscore.index, :]
        gene_zscore = pd.concat([gene_zscore, size], axis=1)
        gene_zscore.columns = ["A", "M", "size"]

    gene_zscore.sort_values('A', ascending=False, inplace=True)
    #gene_zscore = gene_zscore.iloc[:topN, :]

    gene_zscore['mod_zscore'], mad = modified_zscore(gene_zscore['M'],
                                                     consistency_correction=consistency_correction)

    if gene_zscore.shape[0] == 0:
        print("there is no genes that pass the threshold!")
        return gene_zscore
    plot_df = gene_zscore.copy(deep=True)
    plot_df.mod_zscore = plot_df.mod_zscore.abs()
    plot_df.mod_zscore[plot_df.mod_zscore > cutoff] = cutoff
    plot_df.mod_zscore[plot_df.mod_zscore < cutoff] = 0
    plot_df.mod_zscore.replace(float('-inf'), -1*cutoff)
    plot_df.mod_zscore.replace(float('inf'), cutoff)
    plot_df.mod_zscore.fillna(0, inplace=True)
    plot_df.mod_zscore = plot_df.mod_zscore.astype(str)
    plot_df.mod_zscore[plot_df.mod_zscore == str(cutoff)] = f'> {cutoff}'
    plot_df.mod_zscore[plot_df.mod_zscore == '0.0'] = f'< {cutoff}'

    if labels is None:
        plot_df['label'] = ""
    else:
        plot_df['label'] = plot_df.index.tolist()
        plot_df.label[~plot_df.label.isin(labels)] = ""

    y = plot_df.M.median()
    ymin = y - consistency_correction * mad * cutoff
    ymax = y + consistency_correction * mad * cutoff
    xmin = round(plot_df.A.min()) - 1
    xmax = round(plot_df.A.max())
    if size is None:
        plot_df.columns = ["A", "M", "abs(mod_Zscore)", "label"]
    else:
        plot_df.columns = ["A", "M", "#topics GW >= 1", "abs(mod_Zscore)", "label"]

    color_palette = {f'> {cutoff}': "orchid",
                     f'< {cutoff}': "royalblue"}
    markers = {f'> {cutoff}': "o",
               f'< {cutoff}': "s"}

    if size is None:
        sns.scatterplot(data=plot_df, x="A", y="M", style="abs(mod_Zscore)", hue="abs(mod_Zscore)",
                        markers=markers, palette=color_palette, linewidth=0.1)
    else:
        sns.scatterplot(data=plot_df, x="A", y="M", style="abs(mod_Zscore)", hue="#topics GW >= 1",
                        linewidth=0.1, markers=markers)

    if labels is not None:
        texts = []
        for label in plot_df.label.unique():
            if label != "":
                texts.append(plt.text(plot_df.A[label], plot_df.M[label], label, horizontalalignment='left'))

        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
    plt.hlines(y=y, xmin=xmin, xmax=xmax, colors="red")
    plt.hlines(y=ymin, xmin=xmin, xmax=xmax, colors="orange", linestyles='--')
    plt.hlines(y=ymax, xmin=xmin, xmax=xmax, colors="orange", linestyles='--')

    if save:
        plt.savefig(f"{file_name}.{file_format}")
    if show:
        plt.show()
    else:
        plt.close()

    return gene_zscore


def modified_zscore(data, consistency_correction=1.4826):
    """
    Returns the modified z score and Median Absolute Deviation (MAD) from the scores in data.
    The consistency_correction factor converts the MAD to the standard deviation for a given
    distribution. The default value (1.4826) is the conversion factor if the underlying data
    is normally distributed
    """

    median = np.median(data)

    deviation_from_med = np.array(data) - median

    mad = np.median(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med / (consistency_correction * mad)

    return mod_zscore, mad


def functional_enrichment_analysis(gene_list,
                                   type,
                                   organism,
                                   sets=None,
                                   p_value=0.05,
                                   file_format="pdf",
                                   file_name="functional_enrichment_analysis"):
    """
    Doing functional enrichment analysis including GO, KEGG and REACTOME

    :param gene_list: list of gene name
    :type gene_list: list
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

    if type not in ["GO", "REACTOME"]:
        sys.exit("Type is not valid! it should be one of them GO, KEGG, REACTOME")

    if type == "GO" and sets is None:
        sets = ["GO_Biological_Process_2021"]
    elif type == "KEGG" and sets is None:
        sets = ["KEGG_2016"]

    if type in ["GO", "KEGG"]:
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=sets,
                         organism=organism,
                         outdir=f"{file_name}",
                         cutoff=p_value)
        dotplot(enr.res2d,
                title=f"Gene ontology",
                cmap='viridis_r',
                cutoff=p_value,
                ofname=f"{file_name}.{file_format}")
    else:
        numGeneModule = len(gene_list)
        genes = ",".join(gene_list)
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


def GSEA(gene_list,
         gene_sets='GO_Biological_Process_2021',
         p_value=0.05,
         table=True,
         plot=True,
         file_format="pdf",
         file_name="GSEA",
         **kwargs):
    """
    Doing Gene Set Enrichment Analysis on based on the topic weights using GSEAPY package.

    :param gene_list: pandas series with index as a gene names and their ranks/weights as value
    :type gene_list: pandas series
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

    gene_list.index = gene_list.index.str.upper()

    pre_res = gp.prerank(rnk=gene_list,
                         gene_sets=gene_sets,
                         format=file_format,
                         no_plot=~plot,
                         **kwargs)

    pre_res.res2d.sort_values(["NOM p-val"], inplace=True)
    pre_res.res2d.drop(["Name"], axis=1, inplace=True)

    if table:
        pre_res.res2d.to_csv(f"{file_name}.csv")

    if plot:
        res = pre_res.res2d.copy(deep=True)
        res = res[res['NOM p-val'] < p_value]
        for term in res.Term:
            name = term.split("(GO:")[1][:-1]
            gseaplot(rank_metric=pre_res.ranking,
                     term=term,
                     **pre_res.results[term],
                     ofname=f"{file_name}_GO_{name}.{file_format}")

    return pre_res.res2d


def summarize_GO_Term(GO_terms,
                      p_value=0.05,
                      file_format="html",
                      file_name="GO_sum"):
    """
    Summarize long, unintelligible lists of GO terms by finding a representative subset of the terms showing more unique (child) Go terms
    We suggest save it as html since it's gonna be plot by plotly so you can take an advantage of using plotly

    :param GO_terms: Dataframe contains results of gene ontology analysis performs by GSEAPY (https://gseapy.readthedocs.io/en/latest/index.html)
    :type GO_terms: pandas dataframe
    :param p_value: Defines the pValue threshold for plotting. (default: 0.05)
    :type p_value: float
    :param file_format: indicate the format of plot (default: html)
    :type file_format: str
    :param file_name: name and path of the plot use for save (default: gene_composition)
    :type file_name: str

    :return: dataframe used to plot the results
    :rtype: pandas dataframe
    """

    tmp = GO_terms.Term.str.split(pat="(", expand=True)
    GO_terms.Term = tmp[0]
    GO_terms['GO_id'] = tmp[1].str.split(pat=")", expand=True)[0]

    genes = ";".join(GO_terms['Lead_genes'].tolist())
    genes = genes.split(";")
    genes = list(set(genes))

    go_genes = pd.DataFrame(0, index=GO_terms.GO_id, columns=genes)
    for i in range(go_genes.shape[0]):
        tmp = GO_terms.Lead_genes[GO_terms.GO_id == go_genes.index[i]].tolist()[0]
        tmp = tmp.split(";")
        go_genes.loc[go_genes.index[i], tmp] = 1

    go_go_cosine_similarity = pd.DataFrame(cosine_similarity(go_genes),
                                           index=GO_terms.GO_id,
                                           columns=GO_terms.GO_id,
                                           dtype='float64')

    fit = umap.UMAP()

    # principalComponents = pca.fit_transform(go_genes)
    u = fit.fit_transform(go_genes)

    df = pd.DataFrame(data=u,
                      index=go_genes.index.tolist(),
                      columns=['umap1', 'umap2'],
                      dtype=float)

    GO_terms.index = GO_terms.GO_id.values
    df = pd.concat([df, GO_terms], axis=1)
    df['Gene %'] = df['Gene %'].str.replace('%', 'e-2').astype(float)

    url = 'http://purl.obolibrary.org/obo/go.obo'
    graph = obonet.read_obo(url)

    remove_goid = []
    for source in go_go_cosine_similarity.index.tolist():
        for target in go_go_cosine_similarity.index.tolist():
            if graph.has_node(source) and graph.has_node(target) and nx.has_path(graph, source=source,
                                                                                 target=target) and source != target:
                remove_goid.append(target)
                # print(nx.shortest_path(graph, source=source, target=target))
    remove_goid = list(set(remove_goid))
    # df = go_go_cosine_similarity.drop(remove_goid, axis=0)
    df.drop(remove_goid, axis=0, inplace=True)

    df = df[df['NOM p-val'] < p_value]

    fig = px.scatter(df,
                     x="umap1",
                     y="umap2",
                     size="Gene %",
                     color="NOM p-val",
                     hover_data=['GO_id', 'Term'],
                     color_continuous_scale=px.colors.sequential.Viridis_r)
    fig.show()
    fig.write_html(f"{file_name}.{file_format}")

    return df

