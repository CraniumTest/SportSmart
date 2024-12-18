import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

class SportSmart:
    def __init__(self, player_data_csv, game_data_csv):
        self.player_data = pd.read_csv(player_data_csv)
        self.game_data = pd.read_csv(game_data_csv)

    def performance_analytics(self):
        # Example function to analyze player's performance
        # Performance metrics analysis would be more complex in a real implementation
        performance_metrics = self.player_data.groupby('player_id').agg({
            'speed': 'mean',
            'endurance': 'mean',
            'accuracy': 'mean'
        })
        print("Performance Analytics:\n", performance_metrics)

    def injury_prediction(self):
        # Example Injury Prediction Model
        # This would typically involve more complex features and possibly deep learning models
        features = self.player_data.drop(['injury'], axis=1)
        target = self.player_data['injury']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        print("Injury Prediction Report:\n", classification_report(y_test, predictions))

        return model

    def strategy_analysis(self):
        # Example of analyzing game strategy
        # This should be expanded with real strategic insights
        strategy_metrics = self.game_data.groupby('team').agg({
            'possession': 'mean',
            'shots_on_goal': 'sum'
        })
        print("Strategy Analysis:\n", strategy_metrics)

    def plot_performance(self, player_id):
        # Visualize player's performance over time
        player_performance = self.player_data[self.player_data['player_id'] == player_id]
        player_performance.set_index('date', inplace=True)
        
        player_performance[['speed', 'endurance', 'accuracy']].plot(title=f"Performance of Player {player_id}")
        plt.show()

if __name__ == "__main__":
    sport_smart = SportSmart('player_data.csv', 'game_data.csv')
    sport_smart.performance_analytics()
    injury_model = sport_smart.injury_prediction()
    sport_smart.strategy_analysis()
    sport_smart.plot_performance(player_id=1)
