#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: workflow/nextflow/bin/run_with_reports.sh <params-file> <report-dir> [nextflow args...]

Runs the Topyfic Nextflow workflow and writes timing artifacts into <report-dir>:
- trace.txt
- report.html
- timeline.html
- nextflow.log

Environment:
- NEXTFLOW_BIN: path to the nextflow executable (default: nextflow)

Notes:
- Omit -resume when collecting performance data. Resumed runs skip work and distort timings.
- The script prepends the repo .venv/bin directory to PATH when it exists.
EOF
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 1
fi

params_file=$1
report_dir=$2
shift 2

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workflow_dir=$(cd "${script_dir}/.." && pwd)
repo_dir=$(cd "${workflow_dir}/../.." && pwd)

if [[ ! -f "${params_file}" ]]; then
    params_file="${repo_dir}/${params_file}"
fi

if [[ ! -f "${params_file}" ]]; then
    echo "Params file not found: ${params_file}" >&2
    exit 1
fi

if [[ "${report_dir}" != /* ]]; then
    report_dir="${repo_dir}/${report_dir}"
fi

mkdir -p "${report_dir}"
report_dir=$(cd "${report_dir}" && pwd)
params_file=$(cd "$(dirname "${params_file}")" && pwd)/$(basename "${params_file}")

if [[ -x "${repo_dir}/.venv/bin/python" ]]; then
    export PATH="${repo_dir}/.venv/bin:${PATH}"
fi

for arg in "$@"; do
    if [[ "${arg}" == "-resume" ]]; then
        echo "Warning: -resume reuses cached tasks and is not suitable for performance timing." >&2
    fi
done

nextflow_bin=${NEXTFLOW_BIN:-nextflow}

cmd=(
    "${nextflow_bin}"
    run
    "${workflow_dir}/main.nf"
    -params-file
    "${params_file}"
    -ansi-log
    false
    -with-trace
    "${report_dir}/trace.txt"
    -with-report
    "${report_dir}/report.html"
    -with-timeline
    "${report_dir}/timeline.html"
    "$@"
)

echo "Writing Nextflow timing artifacts to ${report_dir}" >&2

(
    cd "${repo_dir}"
    "${cmd[@]}" 2>&1 | tee "${report_dir}/nextflow.log"
)

echo "Trace: ${report_dir}/trace.txt"
echo "Report: ${report_dir}/report.html"
echo "Timeline: ${report_dir}/timeline.html"
echo "Log: ${report_dir}/nextflow.log"