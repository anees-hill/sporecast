import os
import psycopg2
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import luigi
from luigi import LocalTarget

class FetchSporeData(luigi.Task):

    def output(self):
        return LocalTarget(os.getenv('BASE_DIRECTORY') + "/data/raw/daily_spores.csv")

    def run(self):
        load_dotenv()
        base_dir = os.getenv('BASE_DIRECTORY')
        base_dir = Path(base_dir)

        # Import fungal spore count data from local postgresql db
        conn = psycopg2.connect(
            host="localhost",
            database="daily_spores",
            user="postgres",
            password=os.getenv('POSTGRES_PASSWORD')
        )

        cur = conn.cursor()

        cur.execute("SELECT * FROM daily_spores WHERE aero_type IN ('spores');")

        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

        df.to_csv(self.output().path)

        cur.close()
        conn.close()
