nextflow.enable.dsl = 2

import groovy.json.JsonOutput


def validateParams() {
    def missing = []

    if (!(params.names instanceof List) || params.names.isEmpty()) {
        missing << 'names'
    }
    if (!(params.count_adata instanceof Map) || params.count_adata.isEmpty()) {
        missing << 'count_adata'
    }
    if (!(params.n_topics instanceof List) || params.n_topics.isEmpty()) {
        missing << 'n_topics'
    }
    if (!params.workdir) {
        missing << 'workdir'
    }
    if (!(params.train?.random_states instanceof List) || params.train.random_states.isEmpty()) {
        missing << 'train.random_states'
    }

    if (!missing.isEmpty()) {
        error "Missing required params: ${missing.join(', ')}"
    }

    def undefinedInputs = (params.names as List).findAll { !params.count_adata.containsKey(it) }
    if (!undefinedInputs.isEmpty()) {
        error "Missing count_adata entries for: ${undefinedInputs.join(', ')}"
    }

    params.count_adata = (params.count_adata as Map).collectEntries { key, value ->
        [(key): file(value.toString()).toAbsolutePath().toString()]
    }
}


process SINGLE_TRAIN {
    tag "${name}/${topic}/seed=${random_state}"
    publishDir({ "${params.workdir}/${name}/${topic}/train" }, mode: 'copy', overwrite: true)

    input:
    tuple val(name), val(topic), val(random_state), val(adata_path)

    output:
    tuple val(name), val(topic), val(adata_path), path("train_${name}_${topic}_${random_state}.p")

    script:
    def pythonCommand = params.plotting?.interactive ? 'python' : 'MPLBACKEND=Agg python'
    """
    ${pythonCommand} ${projectDir}/bin/single_train.py \
        --name ${name} \
        --adata-path '${adata_path}' \
        --k ${topic} \
        --random-state ${random_state} \
        --backend ${params.train.backend ?: 'default'} \
        --device ${params.train.device ?: 'auto'} \
        --dtype ${params.train.dtype ?: 'float32'} \
        --batch-size ${params.train.batch_size ?: 128} \
        --max-iter ${params.train.max_iter ?: 5} \
        --n-jobs ${params.train.n_jobs ?: 1} \
        --output-dir .
    """

    stub:
    """
    touch train_${name}_${topic}_${random_state}.p
    """
}


process COMBINE_TRAIN {
    tag "${name}/${topic}"
    publishDir({ "${params.workdir}/${name}/${topic}/train" }, mode: 'copy', overwrite: true)

    input:
    tuple val(name), val(topic), val(adata_path), path(train_files)

    output:
    tuple val(name), val(topic), val(adata_path), path("train_${name}_${topic}.p")

    script:
    def pythonCommand = params.plotting?.interactive ? 'python' : 'MPLBACKEND=Agg python'
    def trainArgs = train_files.collect { "--train-file '${it}'" }.join(' ')
    """
    ${pythonCommand} ${projectDir}/bin/combine_train.py \
        --name ${name} \
        --adata-path '${adata_path}' \
        --k ${topic} \
        --output-dir . \
        ${trainArgs}
    """

    stub:
    """
    touch train_${name}_${topic}.p
    """
}


process BUILD_TOPMODEL {
    tag "${name}/${topic}"
    publishDir({ "${params.workdir}/${name}/${topic}/topmodel" }, mode: 'copy', overwrite: true)

    input:
    tuple val(name), val(topic), val(adata_path), path(train_file)

    output:
    tuple val(name), val(topic), val(adata_path), path("topModel_${name}_${topic}.p"), emit: model
    path('topic_weight_umap.h5ad'), emit: topic_weight_umap
    path('topic_cluster_mapping.csv'), emit: topic_cluster_mapping
    path('figures'), optional: true, emit: figures

    script:
    def pythonCommand = params.plotting?.interactive ? 'python' : 'MPLBACKEND=Agg python'
    def nTopGenes = params.top_model?.n_top_genes == null ? 'None' : params.top_model.n_top_genes
    def minCellParticipation = params.top_model?.min_cell_participation == null ? 'None' : params.top_model.min_cell_participation
    """
    ${pythonCommand} ${projectDir}/bin/build_topmodel.py \
        --name ${name} \
        --adata-path '${adata_path}' \
        --train-file '${train_file}' \
        --n-top-genes '${nTopGenes}' \
        --resolution ${params.top_model?.resolution ?: 1} \
        --max-iter-harmony ${params.top_model?.max_iter_harmony ?: 10} \
        --min-cell-participation '${minCellParticipation}' \
        --output-dir .
    """

    stub:
    """
    mkdir -p figures
    touch topModel_${name}_${topic}.p
    touch topic_weight_umap.h5ad
    touch topic_cluster_mapping.csv
    """
}


process BUILD_ANALYSIS {
    tag "${name}/${topic}"
    publishDir({ "${params.workdir}/${name}/${topic}/topmodel" }, mode: 'copy', overwrite: true)

    input:
    tuple val(name), val(topic), val(adata_path), path(topmodel_file)

    output:
    tuple val(name), val(topic), path("analysis_${name}_${topic}.p"), emit: analysis

    script:
    def pythonCommand = params.plotting?.interactive ? 'python' : 'MPLBACKEND=Agg python'
    """
    ${pythonCommand} ${projectDir}/bin/build_analysis.py \
        --adata-path '${adata_path}' \
        --topmodel-file '${topmodel_file}' \
        --output-dir .
    """

    stub:
    """
    touch analysis_${name}_${topic}.p
    """
}


process MERGE_TOPMODELS {
    tag 'merge'
    publishDir(params.workdir, mode: 'copy', overwrite: true)

    input:
    path(analysis_files)

    when:
    params.merge && (params.names as List).size() > 1

    output:
    path("topModel_${(params.names as List).join('_')}.p")
    path("analysis_${(params.names as List).join('_')}.p")
    path('k_N.csv')
    path('best_k.csv')
    path('topic_weight_umap.h5ad')
    path('topic_cluster_mapping.csv')
    path('figures'), optional: true

    script:
    def pythonCommand = params.plotting?.interactive ? 'python' : 'MPLBACKEND=Agg python'
    def namesJson = JsonOutput.toJson(params.names)
    def topicsJson = JsonOutput.toJson(params.n_topics)
    def adataJson = JsonOutput.toJson(params.count_adata)
    def nTopGenes = params.top_model?.n_top_genes == null ? 'None' : params.top_model.n_top_genes
    def minCellParticipation = params.top_model?.min_cell_participation == null ? 'None' : params.top_model.min_cell_participation
    """
    ${pythonCommand} ${projectDir}/bin/merge_models.py \
        --names-json '${namesJson}' \
        --n-topics-json '${topicsJson}' \
        --count-adata-json '${adataJson}' \
        --workdir '${params.workdir}' \
        --n-top-genes '${nTopGenes}' \
        --resolution ${params.top_model?.resolution ?: 1} \
        --max-iter-harmony ${params.top_model?.max_iter_harmony ?: 10} \
        --min-cell-participation '${minCellParticipation}' \
        --output-dir .
    """

    stub:
    """
    mkdir -p figures
    touch topModel_${(params.names as List).join('_')}.p
    touch analysis_${(params.names as List).join('_')}.p
    touch k_N.csv
    touch best_k.csv
    touch topic_weight_umap.h5ad
    touch topic_cluster_mapping.csv
    """
}


workflow {
    validateParams()

    def names = (params.names as List).collect { it.toString() }
    def topics = (params.n_topics as List).collect { it as Integer }
    def randomStates = (params.train.random_states as List).collect { it as Integer }
    def countAdata = names.collectEntries { name ->
        [(name): file(params.count_adata[name].toString()).toAbsolutePath().toString()]
    }

    singleTrainJobs = Channel
        .fromList(names)
        .flatMap { name ->
            topics.collectMany { topic ->
                randomStates.collect { randomState ->
                    tuple(name, topic, randomState, countAdata[name])
                }
            }
        }

    singleTrains = SINGLE_TRAIN(singleTrainJobs)

    combinedTrainJobs = singleTrains
        .map { name, topic, adataPath, trainFile ->
            tuple([name: name, topic: topic, adataPath: adataPath], trainFile)
        }
        .groupTuple()
        .map { meta, trainFiles ->
            tuple(meta.name.toString(), meta.topic as Integer, meta.adataPath.toString(), trainFiles.sort { left, right -> left.name <=> right.name })
        }

    combinedTrains = COMBINE_TRAIN(combinedTrainJobs)
    topModelResults = BUILD_TOPMODEL(combinedTrains)
    analysisResults = BUILD_ANALYSIS(topModelResults.model)

    if (params.merge && names.size() > 1) {
        MERGE_TOPMODELS(
            analysisResults.analysis
                .map { name, topic, analysisFile -> analysisFile }
                .collect()
        )
    }
}