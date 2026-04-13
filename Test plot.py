#%%
"""
Compare measured temperatures with COMSOL temperatures.
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def spread_labels(y_values, min_gap=1.8):
    """
    Adjust y-positions so labels do not overlap.

    Parameters
    ----------
    y_values : array-like
        Original y-values for the labels.
    min_gap : float
        Minimum vertical spacing between labels.

    Returns
    -------
    adjusted : np.ndarray
        Adjusted y-positions.
    """
    y_values = np.array(y_values, dtype=float)
    adjusted = y_values.copy()

    order = np.argsort(adjusted)

    for i in range(1, len(order)):
        prev_idx = order[i - 1]
        curr_idx = order[i]

        if adjusted[curr_idx] - adjusted[prev_idx] < min_gap:
            adjusted[curr_idx] = adjusted[prev_idx] + min_gap

    return adjusted


def compare_measured_and_comsol(
    meas_file,
    comsol_file,
    measurement_uncertainty=2.5,
    min_label_gap=2.0,
    title="Temperature"
):
    """
    Compare measured temperatures with COMSOL temperatures.

    Parameters
    ----------
    meas_file : str
        Path to measured CSV file.
    comsol_file : str
        Path to COMSOL CSV file.
    measurement_uncertainty : float, optional
        Uncertainty in measured values [Â°C]. Default is Â±2.5 Â°C.
    min_label_gap : float, optional
        Minimum vertical spacing between plot labels.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned measurement dataframe.
    df_last60_avg : pandas.DataFrame
        Average of the last 60 measured values.
    df_meas_comsol : pandas.DataFrame
        COMSOL temperatures with mapped column names.
    df_diff : pandas.DataFrame
        Comparison dataframe.
    df_diff_T : pandas.DataFrame
        Transposed comparison dataframe.
    avg_ambient : float
        Average ambient temperature over last 240 samples.
    """

    column_mapping = {
        "Temperature (degC), Point: 37": "307 <Condcutor BL mid> (C)",
        "307 Temperature (degC) BL cond": "307 <Condcutor BL mid> (C)",
        "Temperature (degC), Point: 43": "308 <Jacket BL mid> (C)",
        "308 Temperature (degC) BL jacket": "308 <Jacket BL mid> (C)",
        "Temperature (degC), Point: 44": "309 <Air BL mid> (C)",
        "309 Temperature (degC) BL air": "309 <Air BL mid> (C)",
        "Temperature (degC), Point: 52": "310 <Condcutor TL mid> (C)",
        "310 Temperature (degC) TL cond": "310 <Condcutor TL mid> (C)",
        "Temperature (degC), Point: 58": "311 <Jacket TL mid> (C)",
        "311 Temperature (degC) TL jacket": "311 <Jacket TL mid> (C)",
        "Temperature (degC), Point: 59": "312 <Air TL mid> (C)",
        "312 Temperature (degC) TL air": "312 <Air TL mid> (C)",
        "Temperature (degC), Point: 106": "303 <Conductor BR mid> (C)",
        "303 Temperature (degC) BR cond": "303 <Conductor BR mid> (C)",
        "Temperature (degC), Point: 112": "304 <Jacket BR mid> (C)",
        "304 Temperature (degC) BR jacket": "304 <Jacket BR mid> (C)",
        "Temperature (degC), Point: 113": "305 <Air BR mid> (C)",
        "305 Temperature (degC) BR air": "305 <Air BR mid> (C)"
    }

    measurement_cols = [
        '303 <Conductor BR mid> (C)',
        '304 <Jacket BR mid> (C)',
        '305 <Air BR mid> (C)',
        '307 <Condcutor BL mid> (C)',
        '308 <Jacket BL mid> (C)',
        '309 <Air BL mid> (C)',
        '310 <Condcutor TL mid> (C)',
        '311 <Jacket TL mid> (C)',
        '312 <Air TL mid> (C)',
        '313 <Top of multiduct inside material> (C)',
        '314 <Air ambient> (C)'
    ]

    # -------------------------------------------------------------------------
    # Read measurement data
    # -------------------------------------------------------------------------
    df_raw = pd.read_csv(
        meas_file,
        encoding='utf-16',
        sep=',',
        skiprows=26
    )

    df = df_raw[[col for col in measurement_cols if col in df_raw.columns]]

    # -------------------------------------------------------------------------
    # Plot measurement data with non-overlapping labels
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    max_value = df.max().max()
    cols = list(df.columns)

    for col in cols:
        plt.plot(df.index, df[col], label=col)

    # Last values
    last_idx = df.index[-1]
    last_values = [df[col].iloc[-1] for col in cols]
    adjusted_last_y = spread_labels(last_values, min_gap=min_label_gap)

    for col, y_point, y_text in zip(cols, last_values, adjusted_last_y):
        plt.plot(last_idx, y_point, marker='o', markersize=8, color='blue')
        plt.annotate(
            f"{y_point:.1f} °C",
            xy=(last_idx, y_point),
            xytext=(last_idx + 12, y_text),
            textcoords='data',
            fontsize=10,
            ha='left',
            va='center',
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

    # Values 60 samples before end
    idx_60_before = df.index[-61]
    values_60_before = [df[col].iloc[-61] for col in cols]
    adjusted_60_y = spread_labels(values_60_before, min_gap=min_label_gap)

    for col, y_point, y_text in zip(cols, values_60_before, adjusted_60_y):
        plt.plot(idx_60_before, y_point, marker='s', markersize=8, color='red')
        plt.annotate(
            f"{y_point:.1f} °C",
            xy=(idx_60_before, y_point),
            xytext=(idx_60_before - 12, y_text),
            textcoords='data',
            fontsize=10,
            ha='right',
            va='center',
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

    plt.xlabel("Index")
    plt.ylabel("Temperature (°C)")
    plt.title(title)
    plt.ylim(15, max(max_value + 5, max(adjusted_last_y) + 2))
    plt.xticks(np.arange(df.index.min(), df.index.max() + 1, 60))
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Average ambient temperature over last 240 samples
    # -------------------------------------------------------------------------
    avg_ambient = None
    ambient_col = '314 <Air ambient> (C)'

    if ambient_col in df_raw.columns:
        avg_ambient = df_raw[ambient_col].iloc[-240:].mean()

    # -------------------------------------------------------------------------
    # Average of last 60 samples
    # -------------------------------------------------------------------------
    df_last60_avg = pd.DataFrame(
        [df.iloc[-60:].mean()],
        index=['Measured_avg_last_60']
    )

    # -------------------------------------------------------------------------
    # Read COMSOL data
    # -------------------------------------------------------------------------
    with open(comsol_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_line = lines[4].lstrip('%').strip()
    data_line = lines[5].strip()

    headers = next(csv.reader([header_line]))
    values = [float(v) for v in next(csv.reader([data_line]))]

    df_meas_comsol = pd.DataFrame([values], columns=headers)
    df_meas_comsol = df_meas_comsol.rename(columns=column_mapping)
    df_meas_comsol.index = ['COMSOL']

    # -------------------------------------------------------------------------
    # Keep only common columns
    # -------------------------------------------------------------------------
    common_cols = df_last60_avg.columns.intersection(df_meas_comsol.columns)

    df_diff = pd.concat(
        [
            df_last60_avg[common_cols],
            df_meas_comsol[common_cols]
        ],
        axis=0
    )

    df_diff.index = ['Measured_avg_last_60', 'COMSOL']

    df_diff.loc['Measured_lower'] = (
        df_diff.loc['Measured_avg_last_60'] - measurement_uncertainty
    )

    df_diff.loc['Measured_upper'] = (
        df_diff.loc['Measured_avg_last_60'] + measurement_uncertainty
    )

    df_diff.loc['Difference'] = (
        df_diff.loc['Measured_avg_last_60'] - df_diff.loc['COMSOL']
    )

    df_diff = df_diff.loc[
        [
            'Measured_avg_last_60',
            'Measured_lower',
            'Measured_upper',
            'COMSOL',
            'Difference'
        ]
    ]

    df_diff_T = df_diff.T

    return (
        df,
        df_last60_avg,
        df_meas_comsol,
        df_diff,
        df_diff_T,
        avg_ambient
    )


#%%

#-------------------------------------------------------------------------
#          520 A
#-------------------------------------------------------------------------
meas_file = "520 A black tape on jacket conductor good setup.csv"
comsol_file = "520A temp from comsol.csv"

(
    df,
    df_last60_avg,
    df_meas_comsol,
    df_diff,
    df_diff_T,
    avg_ambient
) = compare_measured_and_comsol(
    meas_file,
    comsol_file,
    measurement_uncertainty=2.5,
    min_label_gap=2.0,
    title="Temperature 520 A"
)

print("Average ambient temperature:")
print(avg_ambient)

print("\nComparison table:")
print(df_diff)

print("\nTransposed comparison table:")
print(df_diff_T)


#%%
# -------------------------------------------------------------------------
# 300 A
# -------------------------------------------------------------------------
meas_file = "300 A black tape on jacket conductor good setup.csv"
comsol_file = "300A temp from comsol.csv"

(
    df_300,
    df_last60_avg_300,
    df_meas_comsol_300,
    df_diff_300,
    df_diff_T_300,
    avg_ambient_300
) = compare_measured_and_comsol(
    meas_file,
    comsol_file,
    measurement_uncertainty=2.5,
    min_label_gap=2.0,
    title="Temperature 300 A"
)

print("Average ambient temperature for 300A:")
print(avg_ambient_300)

print("\nComparison table for 300A:")
print(df_diff_300)

print("\nTransposed comparison table for 300A:")
print(df_diff_T_300)



#%%
# -------------------------------------------------------------------------
# Example usage for 420A
# -------------------------------------------------------------------------
meas_file = "420 A black tape on jacket conductor at ends not properly drilled into.csv"
comsol_file = "420A temp from comsol.csv"

(
    df_420,
    df_last60_avg_420,
    df_meas_comsol_420,
    df_diff_420,
    df_diff_T_420,
    avg_ambient_420
) = compare_measured_and_comsol(
    meas_file,
    comsol_file,
    measurement_uncertainty=2.5,
    min_label_gap=2.0,
    title="Temperature 420 A"
)

print("Average ambient temperature for 420A:")
print(avg_ambient_420)

print("\nComparison table for 420A:")
print(df_diff_420)

print("\nTransposed comparison table for 420A:")
print(df_diff_T_420)





