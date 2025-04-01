from nba_api.stats.endpoints import (
    leaguegamefinder, 
    teamgamelogs,
    commonteamroster,
    playergamelog,
    leaguedashteamstats
)
from nba_api.stats.static import teams
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBADataCollector:
    def __init__(self):
        self.teams_dict = teams.get_teams()
        logger.info(f"Found {len(self.teams_dict)} NBA teams")

    def get_season_games(self, season="2023-24"):
        """Collect games for a specific season"""
        try:
            game_finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00"
            )
            games_df = game_finder.get_data_frames()[0]
            time.sleep(1)  # Respect API rate limits
            
            logger.info(f"Found {len(games_df)} games for season {season}")
            return games_df
        except Exception as e:
            logger.error(f"Error collecting games: {str(e)}")
            return None

    def get_team_id(self, team_name):
        """Get team ID from team name"""
        team = next((team for team in self.teams_dict if team['full_name'].lower() == team_name.lower()), None)
        return team['id'] if team else None

    def get_team_stats(self, team_id):
        """Get team statistics"""
        try:
            logs = teamgamelogs.TeamGameLogs(
                season_nullable="2023-24",
                team_id_nullable=team_id
            )
            time.sleep(1)  # Respect API rate limits
            
            logs_df = logs.get_data_frames()[0]
            if not logs_df.empty:
                # Calculate home vs away performance
                home_games = logs_df[logs_df['MATCHUP'].str.contains('vs')]['WL']
                away_games = logs_df[logs_df['MATCHUP'].str.contains('@')]['WL']
                
                home_win_pct = (home_games == 'W').mean() if not home_games.empty else 0.5
                away_win_pct = (away_games == 'W').mean() if not away_games.empty else 0.5

                # Calculate recent form (last 5 games)
                recent_games = logs_df['WL'].head(5).tolist()
                current_streak = 0
                for result in recent_games:
                    if result == 'W':
                        current_streak += 1
                    else:
                        break

                stats = {
                    # Basic stats from game logs
                    'pts_per_game': logs_df['PTS'].mean(),
                    'fg_pct': logs_df['FG_PCT'].mean(),
                    'fg3_pct': logs_df['FG3_PCT'].mean(),
                    'ft_pct': logs_df['FT_PCT'].mean(),
                    'oreb': logs_df['OREB'].mean(),
                    'dreb': logs_df['DREB'].mean(),
                    'ast': logs_df['AST'].mean(),
                    'stl': logs_df['STL'].mean(),
                    'blk': logs_df['BLK'].mean(),
                    'tov': logs_df['TOV'].mean(),
                    'plus_minus': logs_df['PLUS_MINUS'].mean(),
                    
                    # Calculated stats
                    'net_rating': logs_df['PLUS_MINUS'].mean(),  # Using plus/minus as proxy
                    'home_win_pct': home_win_pct,
                    'away_win_pct': away_win_pct,
                    'winning_streak': current_streak,
                    'recent_form': recent_games.count('W') / len(recent_games) if recent_games else 0.5
                }
                return stats
            return None
        except Exception as e:
            logger.error(f"Error getting team stats: {str(e)}")
            return None

    def get_head_to_head(self, team1_id, team2_id):
        """Get head-to-head matchup statistics"""
        try:
            game_finder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team1_id,
                vs_team_id_nullable=team2_id,
                season_nullable="2023-24"
            )
            time.sleep(1)  # API rate limit
            games_df = game_finder.get_data_frames()[0]
            
            if not games_df.empty:
                team1_wins = len(games_df[games_df['WL'] == 'W'])
                total_games = len(games_df)
                return {
                    'matchup_win_pct': team1_wins / total_games if total_games > 0 else 0.5,
                    'games_played': total_games
                }
            return {'matchup_win_pct': 0.5, 'games_played': 0}
        except Exception as e:
            logger.error(f"Error getting head-to-head stats: {str(e)}")
            return {'matchup_win_pct': 0.5, 'games_played': 0}

    def prepare_game_data(self):
        """Prepare game data with features for training"""
        try:
            games_df = self.get_season_games()
            if games_df is None:
                return None
                
            training_data = []
            logger.info("Processing games for training data...")
            
            for _, game in games_df.iterrows():
                is_home_game = 'vs.' in game['MATCHUP']
                team_id = game['TEAM_ID']
                
                matchup_parts = game['MATCHUP'].split(' ')
                opponent_name = matchup_parts[-1]
                opponent_team = next((team for team in self.teams_dict if team['abbreviation'] == opponent_name), None)
                
                if opponent_team:
                    opponent_id = opponent_team['id']
                    home_team_id = team_id if is_home_game else opponent_id
                    away_team_id = opponent_id if is_home_game else team_id
                    
                    home_stats = self.get_team_stats(home_team_id)
                    away_stats = self.get_team_stats(away_team_id)
                    
                    if home_stats is not None and away_stats is not None:
                        features = {
                            # Home team stats
                            'home_pts_per_game': home_stats['pts_per_game'],
                            'home_fg_pct': home_stats['fg_pct'],
                            'home_fg3_pct': home_stats['fg3_pct'],
                            'home_ft_pct': home_stats['ft_pct'],
                            'home_oreb': home_stats['oreb'],
                            'home_dreb': home_stats['dreb'],
                            'home_ast': home_stats['ast'],
                            'home_stl': home_stats['stl'],
                            'home_blk': home_stats['blk'],
                            'home_tov': home_stats['tov'],
                            'home_plus_minus': home_stats['plus_minus'],
                            
                            # Away team stats
                            'away_pts_per_game': away_stats['pts_per_game'],
                            'away_fg_pct': away_stats['fg_pct'],
                            'away_fg3_pct': away_stats['fg3_pct'],
                            'away_ft_pct': away_stats['ft_pct'],
                            'away_oreb': away_stats['oreb'],
                            'away_dreb': away_stats['dreb'],
                            'away_ast': away_stats['ast'],
                            'away_stl': away_stats['stl'],
                            'away_blk': away_stats['blk'],
                            'away_tov': away_stats['tov'],
                            'away_plus_minus': away_stats['plus_minus'],
                            
                            # Target variable
                            'home_team_won': 1 if (is_home_game and game['WL'] == 'W') or 
                                               (not is_home_game and game['WL'] == 'L') else 0
                        }
                        training_data.append(features)
                
                time.sleep(1)
            
            logger.info(f"Prepared {len(training_data)} games for training")
            return pd.DataFrame(training_data)
            
        except Exception as e:
            logger.error(f"Error preparing game data: {str(e)}")
            return None

    def get_player_stats(self, team_abbr, season="2023-24"):
        """Fetch player statistics for a specific team."""
        try:
            url = f"https://www.basketball-reference.com/teams/{team_abbr}/{season}.html"
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract player stats table
            stats_table = soup.find('table', {'id': 'per_game'})
            if not stats_table:
                raise ValueError("Could not find player stats table")
            
            # Convert table to DataFrame
            df = pd.read_html(str(stats_table))[0]
            return df
        except Exception as e:
            logger.error(f"Error fetching player stats: {str(e)}")
            return None

    def get_recent_games(self, team_abbr, n_games=10):
        """Fetch recent game results for a team."""
        try:
            url = f"https://www.basketball-reference.com/teams/{team_abbr}/2024_games.html"
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract games table
            games_table = soup.find('table', {'id': 'games'})
            if not games_table:
                raise ValueError("Could not find games table")
            
            # Convert table to DataFrame
            df = pd.read_html(str(games_table))[0]
            return df.head(n_games)
        except Exception as e:
            logger.error(f"Error fetching recent games: {str(e)}")
            return None

    def get_injury_data(self):
        """Fetch current injury data from ESPN."""
        try:
            url = "https://www.espn.com/nba/injuries"
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract injury data
            injury_data = []
            injury_tables = soup.find_all('table', {'class': 'Table'})
            
            for table in injury_tables:
                team_name = table.find_previous('h2').text
                rows = table.find_all('tr')[1:]  # Skip header row
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        player = cols[0].text
                        status = cols[1].text
                        comment = cols[2].text
                        injury_data.append({
                            'team': team_name,
                            'player': player,
                            'status': status,
                            'comment': comment
                        })
            
            return pd.DataFrame(injury_data)
        except Exception as e:
            logger.error(f"Error fetching injury data: {str(e)}")
            return None

    def create_feature_set(self, home_team, away_team, season="2023-24"):
        """Create a feature set for prediction using all available data."""
        features = {}
        
        # Get team stats
        team_stats = self.get_team_stats(home_team)
        if team_stats is not None:
            home_stats = team_stats
            away_stats = self.get_team_stats(away_team)
            
            # Add team-level features
            features.update({
                'home_ppg': home_stats['pts_per_game'],
                'away_ppg': away_stats['pts_per_game'],
                'home_fg_pct': home_stats['fg_pct'],
                'away_fg_pct': away_stats['fg_pct'],
                'home_3pt_pct': home_stats['fg3_pct'],
                'away_3pt_pct': away_stats['fg3_pct'],
                'home_ft_pct': home_stats['ft_pct'],
                'away_ft_pct': away_stats['ft_pct'],
                'home_reb': home_stats['oreb'],
                'away_reb': away_stats['oreb'],
                'home_ast': home_stats['ast'],
                'away_ast': away_stats['ast'],
                'home_tov': home_stats['tov'],
                'away_tov': away_stats['tov']
            })
        
        # Get recent form
        home_recent = self.get_recent_games(home_team)
        away_recent = self.get_recent_games(away_team)
        
        if home_recent is not None and away_recent is not None:
            # Calculate win percentages in last 10 games
            home_wins = home_recent['W/L'].str.startswith('W').sum()
            away_wins = away_recent['W/L'].str.startswith('W').sum()
            
            features.update({
                'home_recent_win_pct': home_wins / 10,
                'away_recent_win_pct': away_wins / 10
            })
        
        # Get injury data
        injuries = self.get_injury_data()
        if injuries is not None:
            # Count injured players for each team
            home_injuries = len(injuries[injuries['team'] == home_team])
            away_injuries = len(injuries[injuries['team'] == away_team])
            
            features.update({
                'home_injuries': home_injuries,
                'away_injuries': away_injuries
            })
        
        return features 