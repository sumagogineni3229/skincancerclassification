import sqlite3

# Connect to the database
def connect_db():
    conn = sqlite3.connect('database.db')
    print("Connected to database successfully")

    # Check if the students table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
    table_exists = cursor.fetchone()

    if not table_exists:
        # Create the students table if it doesn't exist
        conn.execute('CREATE TABLE students (name TEXT, addr TEXT, city TEXT, zip TEXT, loginid TEXT, email TEXT, password TEXT)')
        print("Created table successfully!")
    else:
        print("Table 'students' already exists.")

    # Close the connection
    conn.close()
connect_db()
