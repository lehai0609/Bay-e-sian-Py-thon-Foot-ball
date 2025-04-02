import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def _clean_incidents(incidents_df: pd.DataFrame, match_details_df: pd.DataFrame) -> pd.DataFrame:
    """Internal helper to perform initial cleaning on incidents."""
    # Ensure required columns exist
    required_incident_cols = ['game_id', 'id', 'type', 'subtype1', 'team', 'elapsed']
    if not all(col in incidents_df.columns for col in required_incident_cols):
        raise ValueError(f"Incidents DataFrame missing one of {required_incident_cols}")

    required_match_cols = ['id', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal']
    if not all(col in match_details_df.columns for col in required_match_cols):
        raise ValueError(f"Match details DataFrame missing one of {required_match_cols}")

    # Make copies to avoid modifying original dataframes
    incidents = incidents_df.copy()
    match_details = match_details_df.copy()

    # 1. Filter goal types (keep 'n', 'o', 'p') and card subtypes (must exist)
    incidents = incidents[~((incidents['type'] == 'goal') &
                            (~incidents['subtype1'].isin(['n', 'o', 'p', None, np.nan])))].copy()
    incidents = incidents[~((incidents['type'] == 'card') & (incidents['subtype1'].isna()))].copy()

    # 2. Separate yellow cards ('y') into 'card_y' type
    incidents['type'] = np.where(
        (incidents['type'] == 'card') & (incidents['subtype1'] == 'y'),
        'card_y',
        incidents['type']
    )
    # Keep original red cards as 'card' type (assuming subtype1 != 'y' are red)

    # 3. Handle own goals ('o') - switch team ID
    # Merge with match details to get home/away teams for the game
    incidents = pd.merge(
        incidents,
        match_details[['id', 'home_team_api_id', 'away_team_api_id']],
        left_on='game_id',
        right_on='id',
        how='left'
    )

    # Condition for home team scoring own goal -> assign to away team
    cond_home_own_goal = (incidents['type'] == 'goal') & \
                         (incidents['subtype1'] == 'o') & \
                         (incidents['team'] == incidents['home_team_api_id'])

    # Condition for away team scoring own goal -> assign to home team
    cond_away_own_goal = (incidents['type'] == 'goal') & \
                         (incidents['subtype1'] == 'o') & \
                         (incidents['team'] == incidents['away_team_api_id'])

    # Apply the switch
    incidents['team_corrected'] = incidents['team'] # Start with original team
    incidents.loc[cond_home_own_goal, 'team_corrected'] = incidents.loc[cond_home_own_goal, 'away_team_api_id']
    incidents.loc[cond_away_own_goal, 'team_corrected'] = incidents.loc[cond_away_own_goal, 'home_team_api_id']

    # Clean up merge columns and use corrected team
    incidents = incidents.drop(columns=['id_y', 'home_team_api_id', 'away_team_api_id', 'team'])
    incidents = incidents.rename(columns={'id_x': 'id', 'team_corrected': 'team'})


    # 4. Filter out 'special' events (if any exist - not seen in provided XML sample but present in R code)
    incidents = incidents[incidents['type'] != 'special']

    # 5. Cap elapsed time at 90 minutes (as R code implicitly works with 1-90 structure)
    # Add 1 because elapsed time is often 0-indexed in source data for first minute
    incidents['elapsed'] = incidents['elapsed'].clip(1, 90).astype(int)


    return incidents[['game_id', 'type', 'team', 'elapsed']]

def calculate_cumulative_event_counts(incidents_df: pd.DataFrame, match_details_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the minute-by-minute cumulative counts for specified event types
    for both home and away teams, producing a wide-format DataFrame.

    Args:
        incidents_df: DataFrame of raw event incidents (needs game_id, type, team, elapsed, subtype1).
        match_details_df: DataFrame with match details (needs id, home_team_api_id, away_team_api_id).

    Returns:
        DataFrame with one row per game_id and columns for cumulative counts
        (e.g., Home_goal_cum_1, Away_goal_cum_1, ... Away_card_y_cum_90).
    """
    incidents = _clean_incidents(incidents_df, match_details_df)

    # Add Home/Away indicator (1 for Home, 2 for Away)
    match_teams = match_details_df[['id', 'home_team_api_id', 'away_team_api_id']].copy()
    match_teams.rename(columns={'id': 'game_id'}, inplace=True)

    incidents = pd.merge(incidents, match_teams, on='game_id', how='left')

    incidents['Indicator_HA'] = np.where(incidents['team'] == incidents['home_team_api_id'], 1,
                                    np.where(incidents['team'] == incidents['away_team_api_id'], 2, 0)) # 0 if team not home/away? Should not happen after cleaning

    incidents = incidents[incidents['Indicator_HA'] != 0] # Remove events from teams not playing?
    incidents['Team_HA'] = np.where(incidents['Indicator_HA'] == 1, 'Home', 'Away')

    # Calculate counts per game, type, team (Home/Away), and minute
    event_counts = incidents.groupby(['game_id', 'type', 'Team_HA', 'elapsed']).size().reset_index(name='count')

    # Create a full multi-index for all games, types, teams, and minutes 1-90
    all_games = incidents['game_id'].unique()
    all_types = incidents['type'].unique()
    all_teams_ha = ['Home', 'Away']
    all_minutes = range(1, 91)

    full_index = pd.MultiIndex.from_product(
        [all_games, all_types, all_teams_ha, all_minutes],
        names=['game_id', 'type', 'Team_HA', 'elapsed']
    )

    # Pivot and reindex to ensure all combinations are present, fill missing with 0
    event_counts_pivot = event_counts.set_index(['game_id', 'type', 'Team_HA', 'elapsed'])
    event_counts_full = event_counts_pivot.reindex(full_index, fill_value=0).reset_index()

    # Calculate cumulative sum over minutes for each group
    event_counts_full = event_counts_full.sort_values(by=['game_id', 'type', 'Team_HA', 'elapsed'])
    event_counts_full['cumulative_count'] = event_counts_full.groupby(
        ['game_id', 'type', 'Team_HA']
    )['count'].cumsum()

    # Pivot to the final wide format
    final_wide = event_counts_full.pivot_table(
        index='game_id',
        columns=['Team_HA', 'type', 'elapsed'],
        values='cumulative_count'
    )

    # Flatten the multi-index columns (e.g., ('Home', 'goal', 1) -> 'Home_goal_cum_1')
    final_wide.columns = [f"{col[0]}_{col[1]}_cum_{col[2]}" for col in final_wide.columns]

    # Fill any remaining NaNs (if a game had NO events of a certain type at all) with 0
    final_wide = final_wide.fillna(0).reset_index()

    return final_wide


def calculate_average_team_strength(match_details_df: pd.DataFrame, player_attributes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average overall rating of the starting eleven for home and away teams.

    Args:
        match_details_df: DataFrame with match details including player IDs (home_player_1..away_player_11) and date.
        player_attributes_df: DataFrame with player attributes including player_api_id, date, and overall_rating.

    Returns:
        DataFrame with game_id, home_team_strength, away_team_strength.
    """
    if 'date' not in match_details_df.columns or 'date' not in player_attributes_df.columns:
         raise ValueError("'date' column missing in input DataFrames.")

    # Make copies
    matches = match_details_df.copy()
    player_attrs = player_attributes_df.copy()

    # Convert date columns to datetime objects, coercing errors
    matches['match_date'] = pd.to_datetime(matches['date'], errors='coerce')
    player_attrs['rating_date'] = pd.to_datetime(player_attrs['date'], errors='coerce')

    # Drop rows where date conversion failed
    matches.dropna(subset=['match_date'], inplace=True)
    player_attrs.dropna(subset=['rating_date', 'overall_rating', 'player_api_id'], inplace=True)

    # Player columns in match_details
    player_cols = [f'{side}_player_{i}' for side in ['home', 'away'] for i in range(1, 12)]
    if not all(col in matches.columns for col in player_cols):
         raise ValueError("Match details DataFrame missing some player columns.")


    # Melt match details to long format: game_id, match_date, player_slot, player_api_id
    id_vars = ['id', 'match_date']
    matches_long = pd.melt(matches, id_vars=id_vars, value_vars=player_cols,
                           var_name='player_slot', value_name='player_api_id')
    matches_long.rename(columns={'id': 'game_id'}, inplace=True)
    matches_long.dropna(subset=['player_api_id'], inplace=True) # Drop if player ID is missing
    matches_long['player_api_id'] = matches_long['player_api_id'].astype(int)


    # Merge player ratings with the long match data
    # Use a cross merge then filter, or merge_asof if performance is critical and data is sorted
    merged_ratings = pd.merge(matches_long, player_attrs[['player_api_id', 'rating_date', 'overall_rating']],
                              on='player_api_id', how='left')

    # Filter ratings: keep only ratings *before* the match date
    merged_ratings = merged_ratings[merged_ratings['rating_date'] < merged_ratings['match_date']]

    # Find the most recent rating for each player *before* the match
    # Sort by rating_date descending and take the first one per group
    merged_ratings = merged_ratings.sort_values(by='rating_date', ascending=False)
    relevant_ratings = merged_ratings.drop_duplicates(subset=['game_id', 'player_api_id', 'player_slot'], keep='first')

    # Add team indicator (Home/Away) based on player_slot
    relevant_ratings['Team_HA'] = np.where(relevant_ratings['player_slot'].str.startswith('home'), 'Home', 'Away')

    # Calculate average strength per game and team type
    team_strength = relevant_ratings.groupby(['game_id', 'Team_HA'])['overall_rating'].mean().reset_index()

    # Pivot to wide format: game_id, Home_strength, Away_strength
    team_strength_wide = team_strength.pivot(index='game_id', columns='Team_HA', values='overall_rating').reset_index()
    team_strength_wide.rename(columns={'Home': 'home_team_strength', 'Away': 'away_team_strength'}, inplace=True)

    # Handle games where one or both teams might have no ratings (fill with NaN or a default value like global mean?)
    # Let's fill potential NaNs with NaN for now, can be handled in cleaning step.
    # team_strength_wide = team_strength_wide.fillna(team_strength_wide.mean()) # Example: fill with mean

    return team_strength_wide[['game_id', 'home_team_strength', 'away_team_strength']]


def assemble_feature_matrix(cumulative_events_df: pd.DataFrame,
                            team_strength_df: pd.DataFrame,
                            match_details_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges cumulative events, team strength, and outcome variables.

    Args:
        cumulative_events_df: DataFrame from calculate_cumulative_event_counts.
        team_strength_df: DataFrame from calculate_average_team_strength.
        match_details_df: DataFrame with match details (id, home_team_goal, away_team_goal).

    Returns:
        The final assembled (but uncleaned/unscaled) feature matrix.
    """

    # Merge events and strength
    features = pd.merge(cumulative_events_df, team_strength_df, on='game_id', how='left')

    # Add outcome
    outcomes = match_details_df[['id', 'home_team_goal', 'away_team_goal']].copy()
    outcomes.rename(columns={'id': 'game_id'}, inplace=True)
    outcomes['Outcome'] = np.select(
        [
            outcomes['home_team_goal'] > outcomes['away_team_goal'],
            outcomes['home_team_goal'] == outcomes['away_team_goal']
        ],
        [1, 0], # Win = 1, Draw = 0
        default=-1 # Loss = -1
    )
    features = pd.merge(features, outcomes[['game_id', 'Outcome']], on='game_id', how='left')

    # Add intercept
    features['intercept'] = 1

    # Reorder columns (example: put outcome and intercept last)
    meta_cols = ['game_id', 'Outcome', 'intercept', 'home_team_strength', 'away_team_strength']
    event_cols = [col for col in features.columns if col not in meta_cols]
    # Sort event columns CUMULATIVE TIME -> EVENT TYPE -> HOME/AWAY is often useful
    # This basic sort groups by time first. More complex sort needed for exact R order
    event_cols = sorted(event_cols, key=lambda x: (int(x.split('_')[-1]), x))

    final_cols = ['game_id'] + event_cols + ['home_team_strength', 'away_team_strength', 'intercept', 'Outcome']
    # Ensure all expected columns are present before reordering
    final_cols = [col for col in final_cols if col in features.columns]
    features = features[final_cols]


    return features