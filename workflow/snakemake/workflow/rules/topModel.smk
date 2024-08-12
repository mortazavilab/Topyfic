import pandas as pd

configfile: 'config/config.yaml'

# Rule to run topModel for one input adata
rule run_top_model:
    input:
        expand(f"{config['train']['workdir']}/train_{{name}}_{{n_topic}}.p",
            name=config["names"],
            n_topic=config["n_topics"])
    output:
        f"{config['top_model']['workdir']}/topModel_{{name}}_{{n_topic}}.p",
    params:
        name=lambda wildcards: config['names'],
        n_topic=lambda wildcards: config['n_topics']
    run:
        # df, res = make_topmodel.find_best_n_topic(n_topics=config["n_topics"],
        #     names=params.name,
        #     topmodel_outputs=config['top_model']['workdir'])
        #
        # df.to_csv(f"{config['top_model']['workdir']}/k_N_{params.name}.csv",index=False)

        make_topmodel.make_top_model(trains={input},
            adata_paths=config['count_adata'][params.name],
            n_top_genes=config['top_model']['n_top_genes'],
            resolution=config['top_model']['resolution'],
            max_iter_harmony=config['top_model']['max_iter_harmony'],
            min_cell_participation=config['top_model']['min_cell_participation'],
            topmodel_output=config['top_model']['workdir'])

# Rule to run topModel for merging multiple input adatas
if 'merging' in config:
    rule run_multiple_top_model:
        input:
            expand(f"{config['train']['workdir']}/train_{{name}}_{{n_topic}}.p",
                name=config["names"],
                n_topic=config["n_topics"])
        output:
            f"{config['merging']['workdir']}/topModel_{'_'.join(config['names'])}.p",
            f"{config['merging']['workdir']}/analysis_{'_'.join(config['names'])}.p",
        params:
            name=lambda wildcards: config['names'],
            n_topic=lambda wildcards: config['n_topics']
        run:
            df, res = make_topmodel.find_best_n_topic(n_topics=config["n_topics"],
                names=config["names"],
                topmodel_outputs=config['merging']['workdir'])

            df.to_csv(f"{config['merging']['workdir']}/k_N.csv",index=False)

            pd.DataFrame.from_dict(res, orient='index').to_csv(f"{config['merging']['workdir']}/best_k.csv")

            adata_paths = []
            for name in config['names']:
                adata_paths.append(config['count_adata'][name])

            trains = []
            for name in res.keys():
                n_topic = res[name]
                trains.append(f"{config['train']['workdir']}/train_{name}_{n_topic}.p")

            make_topmodel.make_top_model(trains=trains,
                adata_paths=adata_paths,
                n_top_genes=config['top_model']['n_top_genes'],
                resolution=config['top_model']['resolution'],
                max_iter_harmony=config['top_model']['max_iter_harmony'],
                min_cell_participation=config['top_model']['min_cell_participation'],
                topmodel_output=None)