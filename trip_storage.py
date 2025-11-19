import sqlite3
DB_PATH = "trips.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trips(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        start_date TEXT,
        end_date TEXT,
        title TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS items(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trip_id INTEGER,
        name TEXT,
        day TEXT,
        time TEXT,
        note TEXT,
        FOREIGN KEY(trip_id) REFERENCES trips(id)
    );
    """)
    conn.commit()
    conn.close()

def create_or_get_trip(city, start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM trips WHERE city=? AND start_date=? AND end_date=?",
                (city, start_date, end_date))
    row = cur.fetchone()
    if row:
        trip_id = row[0]
    else:
        title = f"{city} 行程（{start_date}）"
        cur.execute("INSERT INTO trips (city, start_date, end_date, title) VALUES (?, ?, ?, ?)",
                    (city, start_date, end_date, title))
        conn.commit()
        trip_id = cur.lastrowid
    conn.close()
    return trip_id

def add_item(trip_id, name, day, time):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO items (trip_id, name, day, time, note) VALUES (?, ?, ?, ?, ?)",
                (trip_id, name, day, time, ""))
    conn.commit()
    conn.close()

def get_all_trips():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, city, start_date, end_date, title FROM trips")
    trips = cur.fetchall()
    conn.close()
    return trips
def get_items(trip_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, day, time, note FROM items WHERE trip_id=?", (trip_id,))
    items = cur.fetchall()
    conn.close()
    return items

def delete_item(item_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM items WHERE id=?", (item_id,))
    conn.commit()
    conn.close()
def update_note(item_id, note):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE items SET note=? WHERE id=?", (note, item_id))
    conn.commit()
    conn.close()

init_db()