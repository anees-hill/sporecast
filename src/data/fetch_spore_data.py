import os
import psycopg2
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

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

df.to_csv(base_dir/"data"/"raw"/"daily_spores.csv")

cur.close()
conn.close()





