import polars as pl
import datetime as dt
import sf_quant.data as sfd

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "mom*satv_monthly"
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

satv_z = (
    data
    .sort("date", "barrid")
    .with_columns(
        (pl.col("market_cap") / pl.col("price")).alias("shrout"),
        (pl.col("daily_volume") /
         (pl.col("market_cap") / pl.col("price"))).alias("turnover"),
    )
    .with_columns(
        pl.col("turnover")
        .rolling_mean(230)
        .shift(21)                 
        .over("barrid")
        .alias("turnover_mean"),

        pl.col("turnover")
        .rolling_std(230)
        .shift(21)                 
        .over("barrid")
        .alias("turnover_std"),
    )
    .with_columns(
        ((pl.col("turnover") - pl.col("turnover_mean"))
         / pl.col("turnover_std"))
        .alias("satv_raw")
    )
)

#adding in mom
satv_mom = (
    satv_z
    .with_columns(
        pl.col("return")
        .log1p()
        .rolling_sum(230)
        .shift(21)
        .over("barrid")
        .alias("mom_raw")
    )
)


# Filter universe
filtered = satv_mom.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col("satv_raw").is_not_null(),
    pl.col("mom_raw").is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
)

# Compute scores
scores = (
    filtered

    # Clip
    .with_columns(
        pl.col("satv_raw").clip(-10, 10).alias("satv_clip"),
        pl.col("mom_raw").clip(-10, 10).alias("mom_clip"),
    )

    # Cross-sectional mean/std
    .with_columns(
        pl.col("satv_clip").mean().over("date").alias("satv_mean"),
        pl.col("satv_clip").std().over("date").alias("satv_std"),
        pl.col("mom_clip").mean().over("date").alias("mom_mean"),
        pl.col("mom_clip").std().over("date").alias("mom_std"),
    )

    # Safe Z-score
    .with_columns(
        pl.when(pl.col("satv_std") > 1e-8)
        .then((pl.col("satv_clip") - pl.col("satv_mean")) / pl.col("satv_std"))
        .otherwise(0.0)
        .alias("satv_z_cs"),

        pl.when(pl.col("mom_std") > 1e-8)
        .then((pl.col("mom_clip") - pl.col("mom_mean")) / pl.col("mom_std"))
        .otherwise(0.0)
        .alias("mom_z_cs"),
    )

    # Interaction
    .with_columns(
        (pl.col("satv_z_cs") * pl.col("mom_z_cs")).alias("score")
    )

    .select(
        "date",
        "barrid",
        "predicted_beta",
        "specific_risk",
        "score",
    )
)

# Compute alphas
alphas = (
    scores.with_columns(pl.col("score").mul(IC).mul("specific_risk").alias("alpha"))
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

# Add month identifier
alphas = alphas.with_columns(
    pl.col("date").dt.truncate("1mo").alias("month")
)

# Take last available alpha in each month (month-end signal)
monthly_alpha = (
    alphas
    .sort("date")
    .group_by(["month", "barrid"])
    .agg([
        pl.col("alpha").last().alias("alpha"),
        pl.col("predicted_beta").last().alias("predicted_beta"),
    ])
)

# Join back to daily dates and forward fill within each month
alphas_monthly = (
    alphas
    .select("date", "barrid", "month")
    .join(monthly_alpha, on=["month", "barrid"], how="left")
    .sort("barrid", "date")
)

# Final output
alphas_monthly = alphas_monthly.select(
    "date",
    "barrid",
    "alpha",
    "predicted_beta"
)

alphas_monthly.write_parquet(f"{signal_name}_alphas.parquet")