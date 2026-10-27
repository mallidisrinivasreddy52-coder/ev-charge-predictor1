"""Create data/ev_sessions.csv with 1,500 simulated charging sessions.

Run:  python generate_data.py
"""
from pathlib import Path

from simulator import generate_sessions

OUT = Path(__file__).parent / "data" / "ev_sessions.csv"

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    df = generate_sessions()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} sessions to {OUT}")
