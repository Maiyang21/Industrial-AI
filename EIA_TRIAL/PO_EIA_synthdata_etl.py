# libraries to be used
import pandas as pd
import numpy as np
import timesynth as ts
from timesynth import *
from functools import reduce

"""Synthetic Data Generation and Feature Engineering of the Refinery Process Optimization foundational Dataset"""

# Dataset importation
ds_kero = pd.read_csv("C:/Users/PC/Documents/DATABASE/PROC_OPTIM/Weekly_U.S._Refiner_Net_Production_of_Kerosene.csv")
ds_jet = pd.read_csv(
    "C:/Users/PC/Documents/DATABASE/PROC_OPTIM/Weekly_U.S._Refiner_Net_Production_of_Kerosene-Type_Jet_Fuel.csv")
ds_naphtha = pd.read_csv("C:/Users/PC/Documents/DATABASE/PROC_OPTIM/oil.csv")


# Target data preparation
def PO_P_data_prep(ds: pd.DataFrame) -> pd.DataFrame:
    ds['Week'] = pd.to_datetime(ds['Week'])
    ds = ds.set_index('Week')
    ds['Barrels per Day'] = ds['Barrels per Day'].values.astype(int)
    ds['Barrels per Day'] = ds['Barrels per Day'].fillna(ds['Barrels per Day'].mean())
    return ds


# Synthetic data generation of foundational data
def generate_synthetic_timeseries(mean_value,
                                  std_value,
                                  sampling_freq=24 * 7,
                                  years=15,
                                  seasonal_period=24,
                                  trend_coef=0.1,
                                  seasonal_amplitude=0.05,
                                  name: str = None,
                                  output_file=None):
    """
    Generate synthetic time series data with trend, seasonality, and noise.
    """
    # Calculate total hours
    total_hours = sampling_freq * 52 * years  # hours/week * weeks/year * years

    # Time base
    time_sampler = ts.TimeSampler(stop_time=total_hours)
    regular_time_samples = time_sampler.sample_regular_time(resolution=1)

    # Components of the time series
    noise = ts.noise.GaussianNoise(std=std_value)
    seasonality = ts.signals.Sinusoidal(frequency=1 / seasonal_period, amplitude=mean_value * seasonal_amplitude)

    # Generate time series
    timeseries = ts.TimeSeries(signal_generator=seasonality, noise_generator=noise)

    #Handling of Sample Output
    sample_output = timeseries.sample(regular_time_samples)

    # If sample_output returns multiple elements, take the first one (time series data)
    if isinstance(sample_output, tuple):
        synthetic_data = sample_output[0]  # Extract only the main time series data
    else:
        synthetic_data = sample_output  # Directly assign if it's a single value

    # Ensuring synthetic_data is 1D
    synthetic_data = np.array(synthetic_data).flatten()

    # Manually Add a Linear Trend
    trend = trend_coef * np.arange(len(regular_time_samples))
    synthetic_data += trend  # Add trend to the generated data

    # Ensuring regular_time_samples is also 1D
    df = pd.DataFrame({
        "Time (hours)": np.array(regular_time_samples).flatten(),
        "{} Value".format(name): synthetic_data
    })

    # clamping the generated signal to ensure adequate boundary for operational values
    offset_val= -1*np.min(synthetic_data)
    df["{} Value".format(name)] = df["{} Value".format(name)] + offset_val

    # Print statistics
    print(f"Generated Mean: {np.mean(synthetic_data):.2f}")
    print(f"Generated Std: {np.std(synthetic_data):.2f}")

    # Save to CSV if filename provided
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Data saved to '{output_file}'")

    return df


# Example usage with original Kero parameters
if __name__ == "__main__":
    """Generate kero operational synthetic data"""

    # Original Kero parameters
    kero_mean_fflow = 993.75  # sm³/h
    kero_std_fflow = 132.5  # sm³/h
    kero_mean_pflow = 790  # sm³/h
    kero_std_pflow = 90  # sm³/h
    kero_mean_ftmp = np.random.randint(250, 380)  # degree cel
    kero_std_ftmp = np.random.randint(10, 30)  # degree cel
    kero_mean_colpres = np.random.randint(1.5, 3)  # bar
    kero_std_colpres = float(np.random.randint(2, 5) / 10)  # bar
    kero_mean_yld = float(np.random.randint(8, 15) / 100)  # percentage
    kero_std_yld = float(np.random.randint(2, 4) / 100)  # percentage

    # flow synthetic data
    df_kero_fflow = generate_synthetic_timeseries(
        mean_value=kero_mean_fflow,
        std_value=kero_std_fflow,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_fflow.csv",
        name="Kero Feed Flow"
    )
    # clamp feed temperature synthetic data
    df_kero_fflow['Kero Feed Flow Value'] = np.clip(df_kero_fflow['Kero Feed Flow Value'], 0, None)
    
    # feed temperature synthetic data
    df_kero_ftemp = generate_synthetic_timeseries(
        mean_value=kero_mean_ftmp,
        std_value=kero_std_ftmp,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_ftemp.csv",
        name="Kero Feed Temperature"
    )
    #  column pressure synthetic data
    df_kero_cpres = generate_synthetic_timeseries(
        mean_value=kero_mean_colpres,
        std_value=kero_std_colpres,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_colpress.csv",
        name="Kero Column Pressure"
    )
    # Yield percentage synthetic data
    df_kero_yperc = generate_synthetic_timeseries(
        mean_value=kero_mean_yld,
        std_value=kero_std_yld,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_yield.csv",
        name="Kero Yield"
    )
    # product flow synthetic data
    df_kero_pflow = generate_synthetic_timeseries(
        mean_value=kero_mean_pflow,
        std_value=kero_std_pflow,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_pflow.csv",
        name="Kero Product Flow"
    )

    """Jet fuel Synthetic Data generation"""

    # Original Jet Fuel parameters
    jfuel_mean_fflow = 993.75  # sm³/h
    jfuel_std_fflow = 132.5  # sm³/h
    jfuel_mean_pflow = 795  # sm³/h
    jfuel_std_pflow = 92.75  # sm³/h
    jfuel_mean_ftmp = np.random.randint(250, 380)  # degree cel
    jfuel_std_ftmp = np.random.randint(10, 30)  # degree cel
    jfuel_mean_colpres = np.random.randint(1.5, 3)  # bar
    jfuel_std_colpres = float(np.random.randint(2, 5) / 10)  # bar
    jfuel_mean_yld = float(np.random.randint(8, 12) / 100)  # percentage
    jfuel_std_yld = float(np.random.randint(2, 4) / 100)  # percentage

    # flow synthetic data
    df_jfuel_fflow = generate_synthetic_timeseries(
        mean_value=jfuel_mean_fflow,
        std_value=jfuel_std_fflow,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_fflow.csv",
        name="Jet Fuel Feed Flow"
    )
    # feed temperature synthetic data
    df_jfuel_ftemp = generate_synthetic_timeseries(
        mean_value=jfuel_mean_ftmp,
        std_value=jfuel_std_ftmp,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_ftemp.csv",
        name="Jet Fuel Feed Temperature"
    )
    #  column pressure synthetic data
    df_jfuel_cpres = generate_synthetic_timeseries(
        mean_value=jfuel_mean_colpres,
        std_value=jfuel_std_colpres,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_colpress.csv",
        name="Jet Fuel Column Pressure"
    )
    # Yield percentage synthetic data
    df_jfuel_yperc = generate_synthetic_timeseries(
        mean_value=jfuel_mean_yld,
        std_value=jfuel_std_yld,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_yield.csv",
        name="Jet Fuel Yield"
    )
    # product flow synthetic data
    df_jfuel_pflow = generate_synthetic_timeseries(
        mean_value=jfuel_mean_pflow,
        std_value=jfuel_std_pflow,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_kero_pflow.csv",
        name="Jet Fuel Product Flow"
    )
"""
    "Naphtha Synthetic Data generation"

    # Original Jet Fuel parameters
    naph_mean_fflow = 993.75  # sm³/h
    naph_std_fflow = 132.5  # sm³/h
    naph_mean_pflow = 795  # sm³/h
    naph_std_pflow = 92.75  # sm³/h
    naph_mean_ftmp = np.random.randint(250, 380)  # degree cel
    naph_std_ftmp = np.random.randint(10, 30)  # degree cel
    naph_mean_colpres = np.random.randint(1.5, 3)  # bar
    naph_std_colpres = np.random.randint(0.2, 0.5)  # bar
    naph_mean_yld = np.random.randint(0.08, 0.12)  # percentage
    naph_std_yld = np.random.randint(0.02, 0.04)  # percentage

    # flow synthetic data
    df_naph_fflow = generate_synthetic_timeseries(
        mean_value=naph_mean_fflow,
        std_value=naph_std_fflow,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_naph_fflow.csv",
        name="Naphtha Feed Flow"
    )
    # feed temperature synthetic data
    df_naph_ftemp = generate_synthetic_timeseries(
        mean_value=naph_mean_ftmp,
        std_value=naph_std_ftmp,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_naph_ftemp.csv",
        name="Naphtha Feed Temperature"
    )
    #  column pressure synthetic data
    df_naph_cpres = generate_synthetic_timeseries(
        mean_value=naph_mean_colpres,
        std_value=naph_std_colpres,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_naph_colpress.csv",
        name="Naphtha Column Pressure"
    )
    # Yield percentage synthetic data
    df_naph_yperc = generate_synthetic_timeseries(
        mean_value=naph_mean_yld,
        std_value=naph_std_yld,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_naph_yield.csv",
        name="Naphtha Fuel Yield"
    )
    # product flow synthetic data
    df_naph_pflow = generate_synthetic_timeseries(
        mean_value=naph_mean_pflow,
        std_value=naph_std_pflow,
        sampling_freq=24 * 7,  # Hourly samples for a week
        years=15,
        seasonal_period=24,  # Daily cycle
        trend_coef=0.1,  # sm³/h per hour
        seasonal_amplitude=0.05,
        output_file="synthetic_naph_pflow.csv",
        name="Naphtha Product Flow"
    )
"""


# parameter file concatenation and dataset production
def concatenating_df(df1, df2, df3, df4, df5, output_file: str) -> pd.DataFrame:
    dfs = [df1, df2, df3, df4, df5]
    df = reduce(lambda left, right: pd.merge(left, right, on='Time (hours)', how='left'), dfs)
    # Save to CSV if filename provided
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Data saved to 'C:/Users/PC/Documents/DATABASE/PROC_OPTIM/Synthetic_data/EIA_data{output_file}'")
    return df


# product file merging
def prod_merge(ds1: pd.DataFrame = None, ds2: pd.DataFrame = None,
               output_file: str = None) -> pd.DataFrame:
    df = pd.merge(ds1, ds2, on='Time (hours)', how='left')
    # Save to CSV if filename provided
    if output_file:
        df.to_csv("C:/Users/PC/Documents/DATABASE/PROC_OPTIM/Synthetic_data/EIA_data/{}".format(output_file),
                  index=False)
        print(f"Data saved to '{output_file}'")
    return df


# merging all parameter files
kero_file = concatenating_df(df_kero_fflow, df_kero_ftemp, df_kero_cpres, df_kero_yperc, df_kero_pflow,
                             "kero_synthetic.csv")
jetfuel_file = concatenating_df(df_jfuel_fflow, df_jfuel_ftemp, df_jfuel_cpres, df_jfuel_yperc, df_jfuel_pflow,
                                "jetfuel_synthetic.csv")
# naphtha_file=concatenating_df(df_naph_fflow, df_naph_ftemp, df_naph_cpres, df_naph_yperc, df_naph_pflow, "naphtha_synthetic.csv")

# merging all product files
prod_merge(kero_file, jetfuel_file, "synthetic_product_data.csv")
