import sqlite3
import pandas as pd
from datetime import datetime

class OHLCDatabase:
    def __init__(self, db_path="data_ohlc.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database and create the OHLC table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlc_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER,
                UNIQUE(timestamp, symbol)
            )
        """)
        
        conn.commit()
        conn.close()

    def insert_ohlc_data(self, df, symbol="XAUUSD"):
        """Insert OHLC data into the database."""
        conn = sqlite3.connect(self.db_path)
        
        # Prepare data for insertion
        data_to_insert = []
        for index, row in df.iterrows():
            data_to_insert.append((
                index.strftime("%Y-%m-%d %H:%M:%S"),  # Convert Timestamp to string
                symbol,
                row["OPEN"],
                row["HIGH"],
                row["LOW"],
                row["CLOSE"],
                row.get("TICKVOL", 0)  # Use TICKVOL as volume, default to 0 if not present
            ))
        
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO ohlc_data 
            (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data_to_insert)
        
        conn.commit()
        conn.close()
        print(f"Inserted {len(data_to_insert)} records for {symbol}")

    def get_ohlc_data(self, symbol="XAUUSD", start_date=None, end_date=None):
        """Retrieve OHLC data from the database."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlc_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp"], index_col="timestamp")
        conn.close()
        
        return df

    def get_latest_timestamp(self, symbol="XAUUSD"):
        """Get the latest timestamp for a symbol."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT MAX(timestamp) FROM ohlc_data WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()[0]
        
        conn.close()
        return result

    def update_with_latest_data(self, new_data_df, symbol="XAUUSD"):
        """Update the database with the latest data, avoiding duplicates."""
        latest_timestamp = self.get_latest_timestamp(symbol)
        
        if latest_timestamp:
            # Filter new data to only include records after the latest timestamp
            latest_timestamp = pd.to_datetime(latest_timestamp)
            new_data_df = new_data_df[new_data_df.index > latest_timestamp]
        
        if not new_data_df.empty:
            self.insert_ohlc_data(new_data_df, symbol)
            print(f"Added {len(new_data_df)} new records")
        else:
            print("No new data to add")


