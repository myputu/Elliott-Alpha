import pandas as pd
from database import OHLCDatabase

def load_data(filepath):
    df = pd.read_csv(filepath, sep=r'\t', header=0)
    df.columns = [col.replace("<", "").replace(">", "") for col in df.columns]
    df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
    df = df.set_index("timestamp")
    df = df[["OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "VOL", "SPREAD"]]
    return df

if __name__ == "__main__":
    db = OHLCDatabase()

    backtest_filepath = "/home/ubuntu/upload/XAUUSD_M1_Backtest.csv"
    forwardtest_filepath = "/home/ubuntu/upload/XAUUSD_M1_Forwardtest.csv"

    backtest_df = load_data(backtest_filepath)
    forwardtest_df = load_data(forwardtest_filepath)

    print("Backtest Data Head:")
    print(backtest_df.head())
    print("\nForwardtest Data Head:")
    print(forwardtest_df.head())

    # Insert data into the database
    db.insert_ohlc_data(backtest_df, "XAUUSD")
    db.insert_ohlc_data(forwardtest_df, "XAUUSD")

    print("Data loaded and inserted into database.")


    # Verify data in database
    retrieved_data = db.get_ohlc_data("XAUUSD")
    print(f"Total records in database: {len(retrieved_data)}")
    print("Sample data from database:")
    print(retrieved_data.head())

