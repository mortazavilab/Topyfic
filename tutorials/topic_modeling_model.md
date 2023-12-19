# Topic modeling data model

To be able to present data and topics in the human-readable way, I develop this data model.

One of the main advantage of that is you can embed your results in this format and will be able to use all the downstream analysis even though you used different software to find your topics!

You have to save three type of objects
1. Model YAML file: Capture all the information about the dataset that you used and which method and how you run it along with Topic_ID which points to Topic YAML files and Cell-Topic participation ID for pointing out the Cell-Topic participation h5ad file.
2. Topics YAML files: Capture all the information about Topics including Topic_ID, gene_weight, gene_id and gene_name and any other information that suits your data
3. Cell-Topic participation h5ad file

![data model](../docs/dataModel.png)


## Model YAML
```yaml
Assay: single nucleus RNA-seq
Cell-Topic participation file_name: topic_cell_participation.h5ad
Data source: ali-mortazavi:topyfic_annotation
Method Name: Topyfic
Number of topics: 19
Technology:
- Parse
- 10x
Topic file_name(s):
- IGVF_000001_Topic_1
- IGVF_000001_Topic_2
- IGVF_000001_Topic_3
- IGVF_000001_Topic_4
- IGVF_000001_Topic_5
- IGVF_000001_Topic_6
- IGVF_000001_Topic_7
- IGVF_000001_Topic_8
- IGVF_000001_Topic_9
- IGVF_000001_Topic_10
- IGVF_000001_Topic_11
- IGVF_000001_Topic_12
- IGVF_000001_Topic_13
- IGVF_000001_Topic_14
- IGVF_000001_Topic_15
- IGVF_000001_Topic_16
- IGVF_000001_Topic_17
- IGVF_000001_Topic_18
- IGVF_000001_Topic_19
- IGVF_000001_Topic_20
- IGVF_000001_Topic_21
- IGVF_000001_Topic_22
Train file_name(s):
- train_parse_adrenal_13
- train_10x_adrenal_15
level: tissue
tissue: Adrenal gland
```

## Topic YAML

```yaml
Gene information:
  gene_biotype:
    Actl6a: Chromatin_binding
    Actl6b: Chromatin_binding
    Actn4: Chromatin_binding
    Actr6: Chromatin_binding
    Actrt1: Chromatin_binding
    Adnp: Chromatin_binding
    Aire: Chromatin_binding
    Ajuba: Chromatin_binding
    Akap8: Chromatin_binding
    Ankrd2: Chromatin_binding
  gene_id:
    Actl6a: ENSG00000136518
    Actl6b: ENSG00000077080
    Actn4: ENSG00000130402
    Actr6: ENSG00000075089
    Actrt1: ENSG00000123165
    Adnp: ENSG00000101126
    Aire: ENSG00000160224
    Ajuba: ENSG00000129474
    Akap8: ENSG00000105127
    Ankrd2: ENSG00000165887
Gene weights:
  Actl6a: 852.4478344747525
  Actl6b: 0.045454545454545456
  Actn4: 1423.5839163462322
  Actr6: 739.7730004645322
  Actrt1: 0.045454545454545456
  Adnp: 816.1664489507675
  Aire: 0.045454545454545456
  Ajuba: 0.045454545454545456
  Akap8: 1676.8289759159088
  Ankrd2: 0.045454545454545456
Topic ID: IGVF_000001_Topic_1
Topic information:
  variance: 512971.1430917481
```

## Write Topyfic results in this format

you can use this script to write model yaml file.

```python 
import Topyfic

# Read analysis object
analysis_top_model = Topyfic.read_analysis(f"../analysis_10x_adrenal_15_parse_adrenal_13.p")

analysis_top_model.cell_participation.write_h5ad('topic_cell_participation.h5ad')

# information about model and datasets
model_info = {
    'Data source': 'ali-mortazavi:topyfic_annotation',
    'Assay': 'single nucleus RNA-seq',
    'Technology': ['Parse', '10x'],
    'level': 'tissue',
    'tissue': 'Adrenal gland',
    'Method Name': 'Topyfic',
    'Number of topics': 19,
    
}
model_info['Topic file_name(s)'] = list(top_model.topics.keys())
model_info['Cell-Topic participation file_name'] = 'topic_cell_participation.h5ad'
model_info['Train file_name(s)'] = ['train_parse_adrenal_13', 'train_10x_adrenal_15']

file = open('Adrenal_model_yaml.yaml', "w")
yaml.dump(model_info, file, default_flow_style=False)
file.close()
```

you can use `write_topic_yaml()` functions to embedded your topics in this format.

```python 
import Topyfic

top_model = Topyfic.read_topModel(f"topModel_10x_adrenal_15_parse_adrenal_13.p")
for topic in top_model.topics:
    print(topic)
    top_model.topics[topic].write_topic_yaml(model_yaml_path="Adrenal_model_yaml.yaml", 
                     topic_yaml_path=f"{topic}.yaml", 
                     save=True)
```

## Read results and embed them in Topyfic data model

```python 
import Topyfic

topic_yaml_path = 'topic_files/'
model_yaml_path = 'topic_files/model.yaml'
cell_topic_participation_path = "topic_files/cell_topic_participation.h5ad"

topModel, analysis = Topyfic.read_model_yaml(model_yaml_path=model_yaml_path,
                    topic_yaml_path=topic_yaml_path,
                    cell_topic_participation_path=cell_topic_participation_path,
                    save=True)

```


