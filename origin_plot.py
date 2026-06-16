import sys
from pathlib import Path

import pandas as pd
import originpro as op


def origin_shutdown_exception_hook(exctype, value, traceback):
    op.exit()
    sys.__excepthook__(exctype, value, traceback)


def make_test_data():
    potential = [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    current = [-0.02, -0.04, -0.08, -0.13, -0.20, -0.31, -0.45, -0.60, -0.78, -0.95, -1.10]

    return pd.DataFrame({
        "Potential_V": potential,
        "Current_mA": current
    })


def create_origin_plot(df, output_dir):
    if op and op.oext:
        sys.excepthook = origin_shutdown_exception_hook
        op.set_show(True)

    op.new()

    wks = op.new_sheet("w")
    wks.name = "EChem_Data"
    wks.from_df(df)

    graph = op.new_graph(template="line")
    graph.name = "EChem_Plot"

    layer = graph[0]
    plot = layer.add_plot(wks, colx=0, coly=1, type="line")
    plot.color = "#FF5F05"

    layer.axis("x").title = "Potential / V"
    layer.axis("y").title = "Current / mA"
    layer.rescale()

    output_png = output_dir / "origin_test_plot.png"
    output_opju = output_dir / "origin_test_project.opju"

    graph.save_fig(str(output_png), width=1200)
    op.save(str(output_opju))

    if op.oext:
        op.exit()

    return output_png, output_opju


def main():
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "results"

    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    df = make_test_data()
    csv_path = data_dir / "test_echem_data.csv"
    df.to_csv(csv_path, index=False)

    output_png, output_opju = create_origin_plot(df, output_dir)

    print(f"CSV saved: {csv_path}")
    print(f"Plot saved: {output_png}")
    print(f"Origin project saved: {output_opju}")


if __name__ == "__main__":
    main()