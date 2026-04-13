import os
import polars as pl
import datetime as dt
from dotenv import load_dotenv
import sf_quant.data as sfd


def load_data() -> pl.DataFrame:
    """
    Load and prepare market data for signal creation.

    Returns:
        pl.DataFrame: Market data with required columns
    """
    # TODO: Load data from source (API, file, database)
    start = dt.date(1996, 1, 1)
    end = dt.date(2024, 12, 31)

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
            "daily_volume",
            "market_cap",
        ],
        in_universe=True,
        ).with_columns(pl.col("return", "specific_risk").truediv(100))

    # TODO: Filter data as needed (date range, symbols, quality checks)

    return data

def create_signal():
    """
    Loads data, creates a simple signal, and saves it to parquet.
    """
    # Load environment variables from .env file
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/signal.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # TODO: Load Data
    df = load_data()

    # TODO: Add your signal logic here (remember alpha logic)

    ####### Daily #######

    #init turnover
    df = df.sort(["barrid", "date"])
    df = df.with_columns(
        (pl.col("daily_volume") / pl.col("market_cap")).alias("turnover")
    )

    #init turnover mean and std
    df = df.with_columns(
        pl.col("turnover").rolling_mean(252).over("barrid").alias("turnover_mean"),
        pl.col("turnover").rolling_std(252).over("barrid").alias("turnover_std"),
    )

    #calc satv signal
    df = df.with_columns(
        ((pl.col("turnover") - pl.col("turnover_mean")) / pl.col("turnover_std"))
        .shift(1)
        .over("barrid")
        .alias("signal")
    )


    #some filters
    df = df.filter(
        (pl.col("signal").is_not_null()) &
        (pl.col("return").is_not_null()) &
        (pl.col("price") >= 5)
    )

    #ic
    IC = 0.05

    #compute z scores for alpha
    scores = df.select(
        "date",
        "barrid",
        "predicted_beta",
        "specific_risk",
        "signal",
        "return",
        # "forward_return",
        pl.col('signal')
        .sub(pl.col('signal').mean())
        .truediv(pl.col('signal').std())
        .over("date")
        .alias("score"),
    )

    #compute alphas
    alphas = (
        scores.with_columns(pl.col("score").mul(IC).mul(pl.col("specific_risk")).alias("alpha"))
        # .select("date", "barrid", "alpha", "signal", "return", "predicted_beta")
        .sort("date", "barrid")
    )

    alphas.write_parquet(output_path)

if __name__ == "__main__":
    create_signal()