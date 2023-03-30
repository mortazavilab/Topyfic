Tutorials
=========

In general, you need to make three objects (Train, TopModel and Analysis). 

The Train object can be initialized either from (a) single cell RNA-seq dataset or (b) single cell ATAC-seq or (c) bulk RNA-seq.

Training part can be time-consuming depending on how big your data is, however you can learn each train model per random state in different jobs and then combine all together. Look at `this tutorial <https://github.com/mortazavilab/Topyfic/blob/main/tutorials/make_train_object.ipynb>`_ for mor information.

For guidance on using Topyfic to analyze your data look at our more depth-in tutorials:

-  `Analysing single cell C2C12 data only using regulatory elements <https://github.com/mortazavilab/Topyfic/blob/main/tutorials/C2C12_TFs_mirhgs_chromreg/C2C12.ipynb>`_: Analysing single cell and single nucleus using C2C12 ENCODE datasets using regulatory elements instead of all genes.
-  `Analysing single cell microglia data <https://github.com/mortazavilab/Topyfic/blob/main/tutorials/microglia_all_genes/microglia.ipynb>`_: Analysing single cell microglia data from `Model-AD portal <https://www.model-ad.org/>`_.

