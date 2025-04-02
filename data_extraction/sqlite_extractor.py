import sqlite3
import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Dict, Optional, Union

def value_from_xpath(element: ET.Element, xpath: str, to_int: bool = False, index: int = 0) -> Optional[Union[str, int]]:
    """
    Extracts text or integer value from an XML element using XPath.

    Args:
        element: The parent XML element.
        xpath: The XPath expression to find the desired child element(s).
        to_int: Whether to convert the extracted text to an integer.
        index: Which element to select if XPath returns multiple elements.

    Returns:
        The extracted value (string or integer) or None if not found/empty.
    """
    try:
        found_elements = element.findall(xpath)
        if found_elements:
            value = found_elements[index].text
            if value:
                return int(value) if to_int else value
    except (IndexError, ValueError):
        # Handle cases where index is out of bounds or conversion fails
        pass
    return None

def node_to_dict(n: ET.Element, key: str) -> Dict:
    """
    Converts an XML node representing an event into a dictionary.

    Args:
        n: The XML element representing a single event instance (<value>).
        key: The primary type of event (e.g., 'goal', 'card').

    Returns:
        A dictionary containing the event's attributes.
    """
    # Extract coordinates safely
    lon = value_from_xpath(n, './coordinates/value', to_int=True, index=0)
    lat = value_from_xpath(n, './coordinates/value', to_int=True, index=1)

    data = {
        'id': value_from_xpath(n, './id', to_int=True),
        'type': value_from_xpath(n, './type'),
        'subtype1': value_from_xpath(n, './subtype'),
        'subtype2': value_from_xpath(n, f'./{key}_type'),
        'player1': value_from_xpath(n, './player1', to_int=True), # Assuming player IDs are integers
        'player2': value_from_xpath(n, './player2', to_int=True), # Assuming player IDs are integers
        'team': value_from_xpath(n, './team', to_int=True),      # Assuming team IDs are integers
        'lon': lon,
        'lat': lat,
        'elapsed': value_from_xpath(n, './elapsed', to_int=True),
        'elapsed_plus': value_from_xpath(n, './elapsed_plus', to_int=True)
    }
    # Handle the subtype swap logic seen in the R code
    if data['subtype2'] is not None:
        data['subtype1'], data['subtype2'] = data['subtype2'], data['subtype1']

    # Clean up None values if desired, or handle them downstream
    # data = {k: v for k, v in data.items() if v is not None}
    return data


def extract_and_parse_events(db_path: str, country_id: int = 1729,
                             event_keys: List[str] = ['goal', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'possession']) -> pd.DataFrame:
    """
    Connects to the SQLite DB, extracts match data for a specific country,
    parses XML event data, and returns a consolidated DataFrame of all events.

    Args:
        db_path: Path to the SQLite database file.
        country_id: The ID of the country/league to filter matches (e.g., 1729 for EPL).
        event_keys: A list of XML column names in the Match table containing event data.

    Returns:
        A pandas DataFrame containing all parsed events from the selected matches.
    """
    all_incidents = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        # Construct the query dynamically for the required event keys + id
        columns_to_select = ", ".join([f'"{k}"' for k in event_keys]) # Quote keys in case they are SQL keywords
        query = f"SELECT id, {columns_to_select} FROM Match WHERE country_id = ?"

        # Use pandas read_sql for efficiency
        matches_df = pd.read_sql(query, conn, params=(country_id,))

        print(f"Processing {len(matches_df)} matches for country_id {country_id}...")

        for _, row in matches_df.iterrows():
            game_id = row['id']
            for key in event_keys:
                xml_string = row[key]
                if pd.isna(xml_string) or not isinstance(xml_string, str) or not xml_string.strip():
                    continue # Skip if XML is missing, not a string, or empty

                try:
                    # Wrap the XML fragment in a root element if it doesn't have one
                    if not xml_string.strip().startswith('<'):
                       # Potentially handle malformed strings if necessary
                       continue
                    if not xml_string.strip().startswith(f'<{key}>'):
                        xml_string = f"<{key}>{xml_string}</{key}>" # Add root if missing

                    root = ET.fromstring(xml_string)
                    event_nodes = root.findall('./value')

                    for node in event_nodes:
                        event_data = node_to_dict(node, key)
                        if event_data.get('id') is not None: # Basic check for valid event data
                            event_data['game_id'] = game_id
                            # Explicitly set the primary event type based on the column key
                            # This overrides the potentially generic 'type' within the XML node itself
                            event_data['type'] = key
                            all_incidents.append(event_data)

                except ET.ParseError as e:
                    print(f"XML Parse Error for game_id {game_id}, key '{key}': {e}. XML: {xml_string[:100]}...") # Log problematic XML
                except Exception as e:
                     print(f"Unexpected Error for game_id {game_id}, key '{key}': {e}")


    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

    print(f"Total incidents extracted: {len(all_incidents)}")
    if not all_incidents:
        return pd.DataFrame() # Return empty DataFrame if no incidents

    return pd.DataFrame(all_incidents)

def extract_other_tables(db_path: str) -> Dict[str, pd.DataFrame]:
    """
    Extracts other relevant tables into pandas DataFrames.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A dictionary where keys are table names and values are DataFrames.
    """
    tables_to_extract = {
        "Match": "SELECT * FROM Match WHERE country_id = 1729", # Filter EPL matches here too
        "Player": "SELECT * FROM Player",
        "Player_Attributes": "SELECT * FROM Player_Attributes",
        "Team": "SELECT * FROM Team",
        "League": "SELECT * FROM League WHERE id = 1729", # Filter EPL league info
        "Country": "SELECT * FROM Country WHERE id = 1729" # Filter EPL country info
    }
    dataframes = {}
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        for name, query in tables_to_extract.items():
             print(f"Extracting table: {name}")
             dataframes[name] = pd.read_sql(query, conn)
             print(f" -> Extracted {len(dataframes[name])} rows.")
             # Specific handling for Match details as in R script
             if name == "Match":
                  dataframes["matchdetails"] = dataframes["Match"] # Keep original name convention

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
    return dataframes


# --- Main Execution ---
if __name__ == "__main__":
    DATABASE_PATH = 'data_extraction/database.sqlite' # Replace with your actual path
    OUTPUT_DIR = 'EPL_data_py'       # Output directory

    # Create output directory if it doesn't exist
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Extract and Parse Event Data
    print("--- Starting Event Extraction ---")
    incidents_df = extract_and_parse_events(DATABASE_PATH, country_id=1729)

    if not incidents_df.empty:
        incidents_output_path = os.path.join(OUTPUT_DIR, 'all_incidents.csv')
        incidents_df.to_csv(incidents_output_path, index=False)
        print(f"Saved all incidents data to {incidents_output_path}")
    else:
        print("No incident data extracted.")

    # 2. Extract Other Tables
    print("\n--- Starting Other Table Extraction ---")
    other_data = extract_other_tables(DATABASE_PATH)

    # 3. Save Other Tables (similar to R script)
    output_map = {
        "matchdetails": "matchdetails.csv",
        "Player": "players.csv", # Renaming to match R script output
        "Player_Attributes": "player_details.csv", # Renaming to match R script output
        "Team": "teams.csv"
        # Add League and Country if needed
    }

    for df_key, filename in output_map.items():
        if df_key in other_data and not other_data[df_key].empty:
            output_path = os.path.join(OUTPUT_DIR, filename)
            other_data[df_key].to_csv(output_path, index=False)
            print(f"Saved {df_key} data to {output_path}")
        else:
             print(f"DataFrame for '{df_key}' not found or empty, skipping save.")

    print("\n--- Data Extraction Complete ---")