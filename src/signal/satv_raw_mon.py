"""Example code for computing alphas and predicted betas prior to running an MVO backtest."""

import polars as pl
import datetime as dt
import sf_quant.data as sfd

#collect data using package
start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "satv_monthly"
price_filter = 5
IC = 0.05

data = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "price",
        "return",
        "specific_risk",
        "predicted_beta",
        "market_cap",
        "daily_volume",
    ],
    in_universe=True,
).with_columns(pl.col("return", "specific_risk").truediv(100))

#calculate turnover
df = data.sort(["barrid", "date"])
df = df.with_columns(
    (pl.col("daily_volume") / pl.col("market_cap")).alias("turnover")
)

#then resample to monthly, taking last value of each month
monthly_signal = (
    df.sort(["barrid", "date"])
    .group_by_dynamic("date", every="1mo", group_by="barrid")
    .agg(
        pl.col("turnover").last().alias("turnover"),
    )
)

#sort
monthly_signal = monthly_signal.sort(["barrid", "date"])

#mean and std turnover
monthly_signal = monthly_signal.with_columns(
    pl.col("turnover").rolling_mean(12, min_samples=12).over("barrid").alias("turnover_mean"),
    pl.col("turnover").rolling_std(12, min_samples=12).over("barrid").alias("turnover_std"),
)

#signal on monthly data
monthly_signal = monthly_signal.with_columns(
    ((pl.col("turnover") - pl.col("turnover_mean")) / pl.col("turnover_std"))
    .shift(1)
    .over("barrid")
    .alias("satv_monthly")
)

#join back to daily df and forward fill
df = df.join(monthly_signal.select("date", "barrid", "satv_monthly"), on=["date", "barrid"], how="left")
df = df.sort(["barrid", "date"])
df = df.with_columns(
    pl.col("satv_monthly").forward_fill().over("barrid")
)

#filter universe
filtered = df.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
)

#compute scores
scores = filtered.select(
    "date",
    "barrid",
    "predicted_beta",
    "specific_risk",
    "return",
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean())
    .truediv(pl.col(signal_name).std())
    .over("date")
    .alias("score"),
)

#compute alphas
alphas = (
    scores.with_columns(pl.col("score").mul(IC).mul(pl.col("specific_risk")).alias("alpha"))
    .select("date", "barrid", "alpha", "predicted_beta", "return")
    .sort("date", "barrid")
)

#create alphas file to use in backtests and testing
alphas.write_parquet(f"{signal_name}_alphas.parquet")