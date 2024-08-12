configfile: 'config/config.yaml'

# Rule to run make_single_train_model
rule run_single_train:
    output:
        f"{config['train']['workdir']}/train_{{name}}_{{n_topic}}_{{random_state}}.p",
    params:
        name=lambda wildcards: config['names'],
        n_topic=lambda wildcards: config['n_topics'],
        random_state=lambda wildcards: config["train"]["random_states"]
    run:
        make_train.make_single_train_model(name=params.name,
            adata_path=config['count_adata'][params.name],
            k=params.n_topic,
            random_state=params.random_state,
            train_output=config['train']['workdir'])

# Rule to run make_train_model
rule run_train_model:
    input:
        expand(f"{config['train']['workdir']}/train_{{name}}_{{n_topic}}_{{random_state}}.p",
            name=config["names"],
            n_topic=config["n_topics"],
            random_state=config["train"]["random_states"])
    output:
        f"{config['train']['workdir']}/train_{{name}}_{{n_topic}}.p",
    params:
        name=lambda wildcards: config['names'],
        n_topic=lambda wildcards: config['n_topics'],
    run:
        make_train.make_train_model(name=params.name,
            adata_path=config['count_adata'][params.name],
            k=params.n_topic,
            n_runs=config['train']['n_runs'],
            random_state=config['train']['random_states'],
            train_output=config['train']['workdir'])