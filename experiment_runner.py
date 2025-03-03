import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from synthetic_data_generator import generate_synthetic_data
from cvt_map_elites import CVTMAPElites

def run_experiment():
    """Run the CVT-MAP-Elites experiment."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data if it doesn't exist
    data_file = 'synthetic_market_data.csv'
    if not os.path.exists(data_file):
        print("Generating synthetic market data...")
        data = generate_synthetic_data(days=60)
        data.to_csv(data_file, index=False)
        print(f"Generated {len(data)} minutes of synthetic market data")
    else:
        print(f"Using existing data file: {data_file}")
    
    # Define the feature universe and periods
    feature_universe = ['open', 'high', 'low', 'close', 'volume', 'period']
    periods = ['1m', '5m', '15m', '60m', '1D']
    
    # Define behavioral characteristics
    behavior_chars = ["sortino", "correlation_to_sp500", "trade_frequency", "win_rate", "profit_factor"]
    
    # Create CVT-MAP-Elites instance
    print("Initializing CVT-MAP-Elites...")
    map_elites = CVTMAPElites(
        feature_universe=feature_universe,
        periods=periods,
        num_centroids=25,
        num_initial_solutions=10,
        data_file=data_file,
        behavior_chars=behavior_chars
    )
    
    # Initialize archive
    print("Initializing archive with random solutions...")
    start_time = time.time()
    map_elites.initialize_archive()
    print(f"Initialization completed in {time.time() - start_time:.2f} seconds")
    print(f"Archive size after initialization: {len(map_elites.archive)}/{map_elites.num_centroids}")
    
    # Visualize initial archive
    print("Visualizing initial archive...")
    map_elites.visualize_archive(save_path='initial_archive.png')
    
    # Evolve archive
    print("Evolving archive...")
    start_time = time.time()
    map_elites.evolve(iterations=25)
    print(f"Evolution completed in {time.time() - start_time:.2f} seconds")
    print(f"Final archive size: {len(map_elites.archive)}/{map_elites.num_centroids}")
    
    # Visualize final archive
    print("Visualizing final archive...")
    map_elites.visualize_archive(save_path='final_archive.png')
    
    # Save archive
    print("Saving archive...")
    map_elites.save_archive()
    
    # Get best solution
    best_solution = map_elites.get_best_solution()
    if best_solution:
        solution, performance, behavior = best_solution
        features, period = solution
        
        print("\nBest solution:")
        print(f"Features: {features}")
        print(f"Period: {period}")
        print(f"Performance (Sharpe): {performance:.4f}")
        print("Behavioral characteristics:")
        for bc, value in behavior.items():
            print(f"  {bc}: {value:.4f}")
    
    # Analyze the archive
    print("\nAnalyzing archive diversity...")
    if map_elites.archive:
        solutions = [map_elites.archive[idx][0] for idx in map_elites.archive]
        performances = [map_elites.archive[idx][1] for idx in map_elites.archive]
        
        # Count occurrences of each feature
        feature_counts = {feature: 0 for feature in feature_universe if feature != 'period'}
        period_counts = {period: 0 for period in periods}
        
        for solution in solutions:
            features, period = solution
            for feature in features:
                feature_counts[feature] += 1
            period_counts[period] += 1
        
        # Print feature statistics
        print("\nFeature usage in archive:")
        for feature, count in feature_counts.items():
            print(f"  {feature}: {count} solutions ({count/len(solutions)*100:.1f}%)")
        
        print("\nPeriod usage in archive:")
        for period, count in period_counts.items():
            print(f"  {period}: {count} solutions ({count/len(solutions)*100:.1f}%)")
        
        # Performance statistics
        print("\nPerformance statistics:")
        print(f"  Mean Sharpe: {np.mean(performances):.4f}")
        print(f"  Median Sharpe: {np.median(performances):.4f}")
        print(f"  Min Sharpe: {np.min(performances):.4f}")
        print(f"  Max Sharpe: {np.max(performances):.4f}")
        print(f"  Std Dev Sharpe: {np.std(performances):.4f}")

if __name__ == "__main__":
    run_experiment()