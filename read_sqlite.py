import sqlite3
import argparse
from collections import defaultdict

def fetch_data_from_table(conn, table_name):
    # Create a cursor object
    cur = conn.cursor()

    # Execute a SQL query to fetch all data from the specified table
    cur.execute(f"SELECT * FROM {table_name}")

    # Fetch all the rows as a list of tuples
    rows = cur.fetchall()

    return rows

def fetch_all_tables(conn):
    # Create a cursor object
    cur = conn.cursor()

    # Execute a SQL query to fetch all table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

    # Fetch all the rows as a list of tuples
    tables = cur.fetchall()

    return [table[0] for table in tables]

def group_by_trial(rows):
    # Create a defaultdict of lists
    trials = defaultdict(list)

    # Group rows by trial number (assuming it's the second element of each row)
    for row in rows:
        if len(row) > 1:  # Check that the row has at least two elements
            trials[row[1]].append(row)

    return trials


def write_to_file(trials):
    # Open the output file in append mode
    with open('all.txt', 'a') as f:
        # Write each trial to the file
        for trial in trials.values():
            for row in trial:
                f.write(str(row))
                f.write('\n')
            f.write('\n')  # Separate trials by a blank line

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Fetch data from an SQLite table and write it to a file.")
    
    # Add an optional argument for the table name
    parser.add_argument('--table', metavar='table', type=str, help='the name of the table to fetch data from')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Connect to the SQLite database
    conn = sqlite3.connect('example.db')

    # If the table argument was provided, fetch data from the table
    if args.table:
        # Fetch data from the table
        rows = fetch_data_from_table(conn, args.table)

        # Group rows by trial
        trials = group_by_trial(rows)

        # Write the data to a file
        write_to_file(trials)
    else:
        # Fetch all table names
        tables = fetch_all_tables(conn)

        # Fetch data from each table and write it to a file named all
        for table in tables:
            rows = fetch_data_from_table(conn, table)

            # Group rows by trial
            trials = group_by_trial(rows)

            # Write the data to a file
            write_to_file(trials)

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    main()
