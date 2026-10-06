#!/usr/bin/env python3
"""Execute the actual self-contained Colab in a fresh kernel; optionally save a backup."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the executed notebook at this path")
    parser.add_argument("--extensions", action="store_true", help="Also run the optional topology experiment")
    parser.add_argument("--in-process", action="store_true", help="Execute through IPython in this process when kernel sockets are unavailable")
    args = parser.parse_args()
    for name in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    notebook = nbformat.read(ROOT / "notebooks/01_gennet_in_one_hour.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and "{FIG}" in cell.source:
            raise ValueError("Unresolved figure URL in notebook")
        if args.extensions and cell.cell_type == "code":
            cell.source = cell.source.replace("RUN_TOPOLOGY_EXPERIMENT = False", "RUN_TOPOLOGY_EXPERIMENT = True")
    notebook.cells.append(nbformat.v4.new_code_cell(
        "assert gennet_auc['test'] >= 0.70, gennet_auc\n"
        "assert 'APOE' in set(gene_imp.head(5)['gene']), gene_imp.head(5)\n"
        "assert pair_rank is not None and pair_rank <= 15, pair_rank\n"
        "assert np.allclose(mixed_difference(oracle_additive), 0)\n"
        "assert np.array_equal(bundle['sets'], additive_bundle['sets'])\n"
        "print('Notebook recovery and interaction-control checks passed')"
    ))
    if args.in_process:
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        from IPython.utils.capture import capture_output

        shell = TerminalInteractiveShell.instance()
        shell.run_line_magic("matplotlib", "inline")
        shell.display_formatter.active_types = ["text/plain", "text/html", "image/png"]
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if result.error_before_exec or result.error_in_exec:
                raise result.error_before_exec or result.error_in_exec
            cell.execution_count = result.execution_count
            cell.outputs = []
            for name, value in (("stdout", captured.stdout), ("stderr", captured.stderr)):
                if value:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
    else:
        # Keep plot capture independent of the host's MPLBACKEND setting.
        notebook.cells.insert(0, nbformat.v4.new_code_cell("%matplotlib inline"))
        NotebookClient(
            notebook, timeout=180, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}}
        ).execute()
        notebook.cells.pop(0)
    print(notebook.cells[-1].outputs[0].text.strip())
    notebook.cells.pop()
    image_count = sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells for output in cell.get("outputs", [])
    )
    if image_count < 6:
        raise ValueError(f"Expected computed plots in executed notebook; found {image_count}")
    print(f"Captured {image_count} computed plots")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
        print(f"Saved executed backup: {args.output}")


if __name__ == "__main__":
    main()
