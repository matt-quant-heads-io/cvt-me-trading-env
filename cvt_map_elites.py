import numpy as np
import pandas as pd
import random
import bt
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import itertools
import pickle
import os

class CVTMAPElites:
    """
    Centroidal Voronoi Tessellation MAP-Elites implementation for trading strategy optimization.
    
    CVT-MAP-Elites creates a collection of high-performing solutions that are diverse
    with respect to behavioral characteristics.
    """
    
    def __init__(self, feature_universe, periods, num_centroids=50, num_initial_solutions=100, 
                 data_file='synthetic_market_data.csv', behavior_chars=None):
        """
        Initialize the CVT-MAP-Elites algorithm.
        
        Args:
            feature_universe: List of available features
            periods: List of available time periods
            num_centroids: Number of centroids for CVT
            num_initial_solutions: Number of random solutions to generate initially
            data_file: File path to the time series data
            behavior_chars: List of behavioral characteristics to track
        """
        self.feature_universe = [f for f in feature_universe if f != 'period']
        self.periods = periods
        self.num_centroids = num_centroids
        self.num_initial_solutions = num_initial_solutions
        
        # Load data
        self.data = pd.read_csv(data_file)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data.set_index('datetime', inplace=True)
        
        # Prepare SP500 data for correlation calculations
        self.sp500_data = self.data[['sp500']].copy()
        
        # Define behavioral characteristics
        self.behavior_chars = behavior_chars or ["sortino", "correlation_to_sp500", 
                                                 "trade_frequency", "win_rate", "profit_factor"]
        
        # Initialize the archive
        self.archive = {}  # Maps centroid indices to (solution, performance, behavior)
        self.centroids = None
        self.behavior_ranges = {bc: {'min': float('inf'), 'max': float('-inf')} 
                                for bc in self.behavior_chars}
        
        # Initialize the CVT
        self._init_cvt()
    
    def _init_cvt(self):
        """Initialize the Centroidal Voronoi Tessellation."""
        # Generate random behavior characteristic vectors in the unit hypercube
        points = np.random.random((self.num_centroids * 10, len(self.behavior_chars)))
        
        # Compute the centroids
        kmeans = KMeans(n_clusters=self.num_centroids, random_state=42)
        kmeans.fit(points)
        self.centroids = kmeans.cluster_centers_
    
    def _generate_random_solution(self):
        """
        Generate a random solution (policy).
        
        A solution is a tuple of (selected_features, period)
        """
        # Select a random subset of features (at least one)
        num_features = random.randint(1, len(self.feature_universe))
        selected_features = random.sample(self.feature_universe, num_features)
        
        # Select a random period
        period = random.choice(self.periods)
        
        return (selected_features, period)
    
    def _mutate_solution(self, solution):
        """
        Mutate a solution by:
        1. Adding a feature
        2. Removing a feature (if there's more than one)
        3. Changing the period
        
        Args:
            solution: (selected_features, period) tuple
        
        Returns:
            New mutated solution
        """
        features, period = solution
        features = list(features)  # Convert to list for easier manipulation
        
        mutation_type = random.choice(['add', 'remove', 'change_period'])
        
        if mutation_type == 'add' and len(features) < len(self.feature_universe):
            # Add a new feature
            available_features = [f for f in self.feature_universe if f not in features]
            if available_features:
                features.append(random.choice(available_features))
        
        elif mutation_type == 'remove' and len(features) > 1:
            # Remove a feature (keeping at least one)
            features.remove(random.choice(features))
        
        elif mutation_type == 'change_period':
            # Change the period
            available_periods = [p for p in self.periods if p != period]
            period = random.choice(available_periods)
        
        return (features, period)
    
    def _resample_data(self, period):
        """
        Resample the data based on the specified period.
        
        Args:
            period: Time period for resampling ('1m', '5m', '15m', '60m', '1D')
        
        Returns:
            Resampled DataFrame
        """
        # No resampling needed for 1m data (it's already in 1m increments)
        if period == '1m':
            return self.data
        
        # Map period strings to pandas resample rule
        period_map = {
            '1m': '1min',   # 1 minute
            '5m': '5min',   # 5 minutes
            '15m': '15min', # 15 minutes
            '60m': '60min', # 60 minutes
            '1D': 'D'     # Daily
        }
        
        rule = period_map[period]
        
        # Resample using the appropriate rule
        resampled = self.data.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'sp500': 'last'
        })
        
        # Forward fill NaN values to avoid issues with missing data
        resampled = resampled.ffill()
        
        return resampled
    
    def _create_strategy(self, solution):
        """
        Create a backtesting strategy from a solution.
        
        Args:
            solution: (selected_features, period) tuple
        
        Returns:
            bt.Strategy object
        """
        features, period = solution
        
        # Resample data based on the period
        resampled = self._resample_data(period)
        
        # Create a simple strategy based on selected features
        # Here we'll implement a simple moving average crossover strategy
        if 'close' in features:
            # Create price data for bt
            price_data = pd.DataFrame(resampled['close'])
            price_data.columns = ['price']
            
            # Calculate moving averages directly with pandas
            fast_ma = price_data.rolling(window=10).mean()
            slow_ma = price_data.rolling(window=30).mean()
            
            # Create signal: 1 when fast_ma > slow_ma, else 0
            # Only generate signals where we have valid data for both MAs
            signal = (fast_ma > slow_ma).astype(int)
            signal[fast_ma.isna() | slow_ma.isna()] = 0
            
            # Create strategy using bt
            strategy = bt.Strategy('MACrossover', 
                                  [bt.algos.SelectAll(),
                                   bt.algos.WeighTarget(signal),
                                   bt.algos.Rebalance()])
            
            # Create backtest
            backtest = bt.Backtest(strategy, price_data)
            return backtest
        else:
            # If 'close' is not in features, use a simple buy and hold strategy
            price_data = pd.DataFrame(resampled['close'])
            price_data.columns = ['price']
            
            # Make sure we have no NaN values
            price_data = price_data.dropna()
            
            strategy = bt.Strategy('BuyHold', 
                                  [bt.algos.RunOnce(),
                                   bt.algos.SelectAll(),
                                   bt.algos.WeighEqually(),
                                   bt.algos.Rebalance()])
            
            backtest = bt.Backtest(strategy, price_data)
            return backtest
    
    def _evaluate_solution(self, solution):
        """
        Evaluate a solution to get performance and behavioral characteristics.
        
        Args:
            solution: (selected_features, period) tuple
        
        Returns:
            (performance, behavior_characteristics) tuple
        """
        try:
            backtest = self._create_strategy(solution)
            features, period = solution
            
            # Run the backtest
            results = bt.run(backtest)
            
            # Calculate performance metrics manually if not available in stats
            # First, try to get returns
            strategy_returns = None
            
            try:
                # Try different ways to access returns
                if hasattr(results, 'prices') and isinstance(results.prices, pd.DataFrame):
                    strategy_returns = results.prices.pct_change().dropna()
                elif hasattr(results, 'equity_curve') and isinstance(results.equity_curve, pd.DataFrame):
                    strategy_returns = results.equity_curve.pct_change().dropna()
                
                # If we have returns, calculate Sharpe ratio manually
                if strategy_returns is not None and len(strategy_returns) > 0:
                    # Convert to numpy array and handle NaN values
                    returns_array = np.array(strategy_returns.values, dtype=float)
                    returns_array = returns_array[~np.isnan(returns_array)]
                    
                    if len(returns_array) > 0:
                        # Calculate annualized Sharpe ratio (assuming daily returns)
                        mean_return = np.mean(returns_array)
                        std_return = np.std(returns_array)
                        
                        # Avoid division by zero
                        if std_return > 0:
                            # Annualize (assuming 252 trading days per year)
                            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
                            performance = float(sharpe_ratio)
                        else:
                            performance = 0.0 if mean_return == 0 else (1.0 if mean_return > 0 else -1.0)
                    else:
                        performance = 0.0
                else:
                    # Try to get Sharpe from stats
                    if hasattr(results, 'stats') and hasattr(results.stats, 'sharpe'):
                        performance = float(results.stats.sharpe)
                    else:
                        # Default performance
                        performance = 0.0
                        
                        # Try to get total return as a fallback performance measure
                        if hasattr(results, 'stats') and isinstance(results.stats, pd.DataFrame):
                            if 'total_return' in results.stats.index:
                                total_return = float(results.stats.loc['total_return'].iloc[0])
                                # Use total return as performance if it's non-zero
                                if total_return != 0:
                                    performance = total_return
            except Exception as e:
                print(f"Error calculating performance: {e}")
                performance = 0.0
            
            # Get performance metrics
            stats = results.stats
            
            # Calculate behavioral characteristics
            behavior = {}
            
            # 1. Sortino ratio - handle case where it might not exist
            try:
                if hasattr(stats, 'sortino'):
                    behavior['sortino'] = float(stats.sortino)
                else:
                    # Use a default value or calculate it manually
                    behavior['sortino'] = 0.0
            except Exception as e:
                print(f"Error calculating sortino: {e}")
                behavior['sortino'] = 0.0
            
            # 2. Correlation to SP500
            try:
                resampled = self._resample_data(period)
                
                # Access returns from the stats DataFrame
                strategy_returns = pd.Series(0, index=resampled.index)  # Default empty series
                
                try:
                    # Get total return from stats if available
                    if hasattr(results, 'stats') and isinstance(results.stats, pd.DataFrame):
                        # Check if 'total_return' exists in the index
                        if 'total_return' in results.stats.index:
                            total_return = float(results.stats.loc['total_return'].iloc[0])
                            strategy_returns = pd.Series(total_return/len(resampled), index=resampled.index)
                except Exception as e:
                    print(f"Error getting total return: {e}")
                
                # Try to get returns from other sources if available
                try:
                    if hasattr(results, 'prices') and isinstance(results.prices, pd.DataFrame):
                        strategy_returns = results.prices.pct_change().dropna()
                    elif hasattr(results, 'equity_curve') and isinstance(results.equity_curve, pd.DataFrame):
                        strategy_returns = results.equity_curve.pct_change().dropna()
                except Exception as e:
                    print(f"Error getting returns from prices/equity: {e}")
                
                # Calculate correlation with SP500
                sp500_returns = resampled['sp500'].pct_change().dropna()
                
                # Ensure both series have the same index type
                strategy_returns.index = pd.to_datetime(strategy_returns.index)
                sp500_returns.index = pd.to_datetime(sp500_returns.index)
                
                # Align the indices
                common_index = strategy_returns.index.intersection(sp500_returns.index)
                if len(common_index) > 1:  # Need at least 2 points for correlation
                    # Get values as numpy arrays to avoid index issues
                    strat_values = strategy_returns.loc[common_index].values
                    sp500_values = sp500_returns.loc[common_index].values
                    
                    # Check if arrays have the same length
                    if len(strat_values) == len(sp500_values) and len(strat_values) > 1:
                        # Make sure arrays are 1D
                        strat_values = strat_values.flatten()
                        sp500_values = sp500_values.flatten()
                        
                        # Check for NaN values
                        valid_indices = ~(np.isnan(strat_values) | np.isnan(sp500_values))
                        if np.any(valid_indices) and np.sum(valid_indices) > 1:
                            strat_values = strat_values[valid_indices]
                            sp500_values = sp500_values[valid_indices]
                            
                            # Calculate correlation manually to avoid dimension issues
                            correlation = np.corrcoef(strat_values, sp500_values)
                            if correlation.size > 1:  # Make sure correlation matrix has at least 2x2 elements
                                behavior['correlation_to_sp500'] = float(correlation[0, 1])
                            else:
                                behavior['correlation_to_sp500'] = 0.0
                        else:
                            behavior['correlation_to_sp500'] = 0.0
                    else:
                        behavior['correlation_to_sp500'] = 0.0
                else:
                    behavior['correlation_to_sp500'] = 0.0
            except Exception as e:
                print(f"Error calculating correlation: {e}")
                behavior['correlation_to_sp500'] = 0.0
            
            # 3. Trade frequency (number of trades per day)
            try:
                num_trades = 0
                if hasattr(results, 'stats') and isinstance(results.stats, pd.DataFrame):
                    if 'trade_count' in results.stats.index:
                        print(f"found trade_count!!")
                        num_trades = int(float(results.stats.loc['trade_count'].iloc[0]))
                
                trading_days = len(resampled) / 390  # Approx 390 minutes per trading day
                behavior['trade_frequency'] = float(num_trades / max(1, trading_days))
            except Exception as e:
                print(f"Error calculating trade frequency: {e}")
                behavior['trade_frequency'] = 0.0
            
            # 4. Win rate
            try:
                win_rate = 0.0
                if hasattr(results, 'stats') and isinstance(results.stats, pd.DataFrame):
                    if 'win_rate' in results.stats.index:
                        # Convert to float to handle potential string values
                        win_rate = float(results.stats.loc['win_rate'].iloc[0])
                        print(f"found win_rate!!")
                    else:
                        # Calculate from returns if available
                        if len(strategy_returns) > 0:
                            # Convert to numpy array to avoid index issues
                            returns_array = np.array(strategy_returns.values, dtype=float)
                            
                            # Resample to daily by simply counting positive days
                            # This is a simplification but avoids index issues
                            positive_days = np.sum(returns_array > 0)
                            total_days = len(returns_array)
                            
                            win_rate = float(positive_days / max(1, total_days))
                
                behavior['win_rate'] = win_rate
            except Exception as e:
                print(f"Error calculating win rate: {e}")
                behavior['win_rate'] = 0.0
            
            # 5. Profit factor
            try:
                profit_factor = 1.0
                if hasattr(results, 'stats') and isinstance(results.stats, pd.DataFrame):
                    if 'profit_factor' in results.stats.index:
                        print(f"found profit_factor!!")
                        profit_factor = float(results.stats.loc['profit_factor'].iloc[0])
                    else:
                        # Calculate from returns if available
                        if len(strategy_returns) > 0:
                            # Convert to numpy array to avoid index issues
                            returns_array = np.array(strategy_returns.values, dtype=float)
                            
                            # Filter out NaN values
                            returns_array = returns_array[~np.isnan(returns_array)]
                            
                            if len(returns_array) > 0:
                                # Calculate gains and losses
                                gains = np.sum(returns_array[returns_array > 0])
                                losses = np.abs(np.sum(returns_array[returns_array < 0]))
                                
                                if losses > 0:
                                    profit_factor = float(gains / losses)
                                else:
                                    profit_factor = 1.0 if gains == 0 else float(gains)
                
                behavior['profit_factor'] = profit_factor
            except Exception as e:
                print(f"Error calculating profit factor: {e}")
                behavior['profit_factor'] = 1.0
            
            # Update behavior ranges for normalization
            for bc in self.behavior_chars:
                self.behavior_ranges[bc]['min'] = min(self.behavior_ranges[bc]['min'], behavior[bc])
                self.behavior_ranges[bc]['max'] = max(self.behavior_ranges[bc]['max'], behavior[bc])
            
            # Print debug info about the performance
            print(f"Solution {solution} performance: {performance}")
            
            return performance, behavior
            
        except Exception as e:
            print(f"Error evaluating solution {solution}: {e}")
            # Return poor performance and neutral behavior
            default_behavior = {bc: 0.0 for bc in self.behavior_chars}
            return -999.0, default_behavior
    
    def _normalize_behavior(self, behavior):
        """
        Normalize behavior characteristics to [0, 1] range.
        
        Args:
            behavior: Dictionary of behavior characteristics
        
        Returns:
            Normalized behavior vector
        """
        normalized = []
        for bc in self.behavior_chars:
            bc_min = self.behavior_ranges[bc]['min']
            bc_max = self.behavior_ranges[bc]['max']
            
            # Avoid division by zero
            if bc_max == bc_min:
                normalized.append(0.5)
            else:
                normalized.append((behavior[bc] - bc_min) / (bc_max - bc_min))
        
        return np.array(normalized)
    
    def _find_nearest_centroid(self, behavior):
        """
        Find the index of the nearest centroid to the given behavior.
        
        Args:
            behavior: Dictionary of behavior characteristics
        
        Returns:
            Index of the nearest centroid
        """
        behavior_vec = self._normalize_behavior(behavior)
        distances = np.linalg.norm(self.centroids - behavior_vec, axis=1)
        return np.argmin(distances)
    
    def initialize_archive(self):
        """Initialize the archive with random solutions."""
        for _ in range(self.num_initial_solutions):
            solution = self._generate_random_solution()
            performance, behavior = self._evaluate_solution(solution)
            
            # Add to archive if better than current solution at the nearest centroid
            centroid_idx = self._find_nearest_centroid(behavior)
            
            if (centroid_idx not in self.archive or 
                performance > self.archive[centroid_idx][1]):
                self.archive[centroid_idx] = (solution, performance, behavior)
    
    def evolve(self, iterations=100):
        """
        Evolve the archive for a number of iterations.
        
        Args:
            iterations: Number of iterations to evolve the archive
        """
        for i in range(iterations):
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations}")
                print(f"Archive size: {len(self.archive)}/{self.num_centroids}")
            
            # Select a random solution from the archive
            if not self.archive:
                # If archive is empty, generate a random solution
                parent = self._generate_random_solution()
            else:
                # Randomly select a solution from the archive
                parent = random.choice(list(self.archive.values()))[0]
            
            # Mutate the solution
            child = self._mutate_solution(parent)
            
            # Evaluate the child
            performance, behavior = self._evaluate_solution(child)
            
            # Find the nearest centroid
            centroid_idx = self._find_nearest_centroid(behavior)
            
            # Add to archive if better than current solution at this centroid
            if (centroid_idx not in self.archive or 
                performance > self.archive[centroid_idx][1]):
                self.archive[centroid_idx] = (child, performance, behavior)
        
        # After evolution is complete, visualize backtest results
        self.visualize_backtest_results()
    
    def get_best_solution(self):
        """Get the overall best solution in the archive."""
        if not self.archive:
            return None
        
        best_solution = max(self.archive.values(), key=lambda x: x[1])
        return best_solution
    
    def visualize_archive(self, save_path=None):
        """
        Visualize the archive using a scatter plot of the first two behavior characteristics.
        
        Args:
            save_path: Path to save the visualization, if None, displays interactively
        """
        if not self.archive:
            print("Archive is empty. Creating placeholder visualization.")
            # Create an empty figure with a message
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Archive is empty - No solutions found", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved empty visualization to {save_path}")
            else:
                plt.tight_layout()
                plt.show()
            return
        
        # Extract behavioral characteristics for all solutions in the archive
        try:
            behaviors = np.array([self._normalize_behavior(self.archive[idx][2]) 
                                 for idx in self.archive])
            performances = np.array([self.archive[idx][1] for idx in self.archive])
            
            # Create figure
            fig = plt.figure(figsize=(15, 10))
            
            # Add 2D visualization for the first two behavioral characteristics
            ax1 = fig.add_subplot(121)
            scatter = ax1.scatter(behaviors[:, 0], behaviors[:, 1], 
                                 c=performances, cmap='viridis', s=100, alpha=0.7)
            ax1.set_xlabel(self.behavior_chars[0])
            ax1.set_ylabel(self.behavior_chars[1])
            ax1.set_title(f'Archive Solutions (2D Projection) - {len(self.archive)} solutions')
            fig.colorbar(scatter, ax=ax1, label='Performance (Sharpe)')
            
            # Add 3D visualization if there are at least 3 behavioral characteristics
            if len(self.behavior_chars) >= 3:
                ax2 = fig.add_subplot(122, projection='3d')
                scatter = ax2.scatter(behaviors[:, 0], behaviors[:, 1], behaviors[:, 2], 
                                     c=performances, cmap='viridis', s=100, alpha=0.7)
                ax2.set_xlabel(self.behavior_chars[0])
                ax2.set_ylabel(self.behavior_chars[1])
                ax2.set_zlabel(self.behavior_chars[2])
                ax2.set_title('Archive Solutions (3D Projection)')
            
            # Add text with archive statistics
            plt.figtext(0.5, 0.01, 
                       f"Archive size: {len(self.archive)}/{self.num_centroids} | " +
                       f"Best performance: {max(performances):.2f}", 
                       ha="center", fontsize=12, 
                       bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            # Save or display
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            else:
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            print(f"Error during visualization: {e}")
            # Create a simple error figure
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Visualization error: {e}\nArchive size: {len(self.archive)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved error visualization to {save_path}")
            else:
                plt.tight_layout()
                plt.show()
    
    def save_archive(self, filename='cvt_map_elites_archive.pkl'):
        """Save the archive to a file."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'archive': self.archive,
                'centroids': self.centroids,
                'behavior_ranges': self.behavior_ranges
            }, f)
    
    def load_archive(self, filename='cvt_map_elites_archive.pkl'):
        """Load the archive from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.archive = data['archive']
            self.centroids = data['centroids']
            self.behavior_ranges = data['behavior_ranges']
    
    def visualize_backtest_results(self, save_dir='backtest_visualizations'):
        """
        Visualize backtest results for each policy in the archive using bt's built-in plotting.
        All equity curves will be normalized to start at 0 for better comparison.
        
        Args:
            save_dir: Directory to save the visualizations
        """
        if not self.archive:
            print("Archive is empty. No backtest results to visualize.")
            return
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Visualize each policy in the archive
        for idx, (solution, performance, behavior) in self.archive.items():
            try:
                features, period = solution
                
                # Create a title for the plot
                title = f"Policy {idx}: {features} (Period: {period})"
                filename = f"{save_dir}/policy_{idx}_{period}.png"
                
                # Run backtest for this solution
                backtest = self._create_strategy(solution)
                results = bt.run(backtest)
                
                # Create figure
                fig = plt.figure(figsize=(15, 12))
                
                # Try to normalize and plot the equity curve
                try:
                    # Get equity curve data
                    if hasattr(results, 'equity_curve'):
                        equity_data = results.equity_curve.copy()
                    elif hasattr(results, 'prices'):
                        equity_data = results.prices.copy()
                    else:
                        # Create a dummy equity curve if none exists
                        equity_data = pd.DataFrame(index=self._resample_data(period).index)
                        equity_data['dummy'] = 100.0
                    
                    # Normalize to start at 0 (percentage returns from start)
                    if not equity_data.empty:
                        start_value = equity_data.iloc[0].values[0]
                        normalized_equity = (equity_data / start_value - 1) * 100  # Convert to percentage
                        
                        # Plot normalized equity curve
                        ax1 = fig.add_subplot(211)
                        normalized_equity.plot(ax=ax1)
                        ax1.set_title(f"Equity Curve (% Return) - {title}")
                        ax1.set_ylabel('Return (%)')
                        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Add a horizontal line at 0
                        ax1.grid(True)
                        
                        # Try to use bt's plot method for additional plots
                        try:
                            ax2 = fig.add_subplot(212)
                            results.plot(ax=ax2, figsize=(15, 5))
                            ax2.set_title("Backtest Details")
                        except Exception as e:
                            # Fallback to plotting drawdowns
                            ax2 = fig.add_subplot(212)
                            if hasattr(results, 'drawdown'):
                                results.drawdown.plot(ax=ax2, color='red')
                            else:
                                # Calculate drawdowns manually
                                rolling_max = equity_data.cummax()
                                drawdown = (equity_data - rolling_max) / rolling_max * 100  # Convert to percentage
                                drawdown.plot(ax=ax2, color='red')
                            ax2.set_title('Drawdown (%)')
                            ax2.set_ylabel('Drawdown (%)')
                            ax2.grid(True)
                    else:
                        # If equity data is empty, create a message
                        ax = fig.add_subplot(111)
                        ax.text(0.5, 0.5, "No equity data available for this strategy", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax.transAxes, fontsize=16)
                        ax.set_xticks([])
                        ax.set_yticks([])
                
                except Exception as e:
                    print(f"Error plotting normalized equity: {e}, falling back to standard plotting")
                    
                    # Fallback to standard bt plotting
                    try:
                        ax = fig.add_subplot(111)
                        results.plot(ax=ax)
                        ax.set_title(f"Backtest Results - {title}")
                    except Exception as e2:
                        print(f"Error using bt plot method: {e2}")
                        ax = fig.add_subplot(111)
                        ax.text(0.5, 0.5, f"Error plotting results: {e2}", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax.transAxes, fontsize=16)
                        ax.set_xticks([])
                        ax.set_yticks([])
                
                # Add performance metrics as text
                metrics_text = (
                    f"Performance: {performance:.4f}\n"
                    f"Sortino: {behavior['sortino']:.4f}\n"
                    f"Correlation to SP500: {behavior['correlation_to_sp500']:.4f}\n"
                    f"Trade Frequency: {behavior['trade_frequency']:.4f}\n"
                    f"Win Rate: {behavior['win_rate']:.4f}\n"
                    f"Profit Factor: {behavior['profit_factor']:.4f}"
                )
                
                plt.figtext(0.01, 0.01, metrics_text, fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.8))
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Saved backtest visualization for policy {idx} to {filename}")
                
            except Exception as e:
                print(f"Error visualizing backtest for policy {idx}: {e}")
        
        print(f"Saved {len(self.archive)} backtest visualizations to {save_dir}/")