from click.testing import CliRunner
import pandas as pd

import Topyfic.main as main_module


def test_train_model_cli_passes_backend_and_device_options(tmp_path, synthetic_adata, monkeypatch):
    data_path = tmp_path / "input.h5ad"
    synthetic_adata.write_h5ad(data_path)
    captured = {}

    def fake_train_model(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_module, "train_model", fake_train_model)

    result = CliRunner().invoke(
        main_module.cli,
        [
            "train_model",
            "--name",
            "demo",
            "--data",
            str(data_path),
            "-k",
            "2",
            "--n-runs",
            "2",
            "--random-state",
            "3",
            "--random-state",
            "7",
            "--backend",
            "torch",
            "--device",
            "cpu",
            "--dtype",
            "float64",
            "--learning-method",
            "batch",
            "--batch-size",
            "5",
            "--max-iter",
            "9",
            "--save-path",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_name"] == "torch"
    assert captured["backend_kwargs"] == {"device": "cpu", "dtype": "float64"}
    assert captured["random_state_range"] == [3, 7]
    assert captured["learning_method"] == "batch"
    assert captured["batch_size"] == 5
    assert captured["max_iter"] == 9
    assert captured["data"].shape == synthetic_adata.shape


def test_train_model_cli_defaults_to_resolved_backend(tmp_path, synthetic_adata, monkeypatch):
    data_path = tmp_path / "input.h5ad"
    synthetic_adata.write_h5ad(data_path)
    captured = {}

    def fake_train_model(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_module, "train_model", fake_train_model)
    monkeypatch.setattr(main_module, "resolve_lda_backend_name", lambda _: "torch")

    result = CliRunner().invoke(
        main_module.cli,
        [
            "train_model",
            "--name",
            "demo",
            "--data",
            str(data_path),
            "-k",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_name"] == "torch"
    assert captured["backend_kwargs"] == {"device": "auto", "dtype": "float32"}


def test_make_topmodel_cli_loads_train_files(tmp_path, synthetic_adata, monkeypatch):
    data_path = tmp_path / "input.h5ad"
    synthetic_adata.write_h5ad(data_path)
    train_a = tmp_path / "train_a.p"
    train_b = tmp_path / "train_b.p"
    train_a.write_bytes(b"a")
    train_b.write_bytes(b"b")
    captured = {}

    def fake_read_train(path):
        return f"train:{path}"

    def fake_make_topmodel(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_module, "read_train", fake_read_train)
    monkeypatch.setattr(main_module, "make_topModel", fake_make_topmodel)

    result = CliRunner().invoke(
        main_module.cli,
        [
            "make_topModel",
            "--train-file",
            str(train_a),
            "--train-file",
            str(train_b),
            "--data",
            str(data_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["trains"] == [f"train:{train_a}", f"train:{train_b}"]
    assert captured["data"].shape == synthetic_adata.shape


def test_make_analysis_cli_loads_topmodel_and_colors(tmp_path, synthetic_adata, monkeypatch):
    data_path = tmp_path / "input.h5ad"
    synthetic_adata.write_h5ad(data_path)
    top_model_path = tmp_path / "top_model.p"
    top_model_path.write_bytes(b"unused")
    colors_path = tmp_path / "colors.csv"
    pd.DataFrame({"color": ["#000000"]}, index=["Topic_1"]).to_csv(colors_path)
    captured = {}
    sentinel = object()

    monkeypatch.setattr(main_module, "read_topModel", lambda path: sentinel)

    def fake_make_analysis_class(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_module, "make_analysis_class", fake_make_analysis_class)

    result = CliRunner().invoke(
        main_module.cli,
        [
            "make_analysis_class",
            "--top-model",
            str(top_model_path),
            "--data",
            str(data_path),
            "--colors-topics",
            str(colors_path),
            "--save-path",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["top_model"] is sentinel
    assert captured["data"].shape == synthetic_adata.shape
    assert captured["colors_topics"].index.tolist() == ["Topic_1"]