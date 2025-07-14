import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GoldPredictor:
    def __init__(self, data_path="gold_data.csv"):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.models = {}
        self.features = []
        
    def check_dependencies(self):
        """Check if all required packages are installed"""
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            print(" All required packages are installed!")
            return True
        except ImportError as e:
            print(f" Missing package: {e}")
            print("Please install required packages using: pip install -r requirements.txt")
            return False
    
    def check_data_file(self):
        """Check if the data file exists"""
        if not os.path.exists(self.data_path):
            print(f" Data file '{self.data_path}' not found!")
            print("Creating sample data...")
            try:
                self.create_sample_data()
                print(" Sample data created successfully!")
                return True
            except Exception as e:
                print(f" Error creating sample data: {e}")
                return False
        return True
    
    def create_sample_data(self):
        """Create sample gold data if it doesn't exist"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate dates for the last 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic gold price data
        np.random.seed(42)
        
        # Base price around $2000
        base_price = 2000
        
        # Create price series with trend and volatility
        n_days = len(dates)
        
        # Add trend component (slight upward trend)
        trend = np.linspace(0, 200, n_days)
        
        # Add seasonal component
        seasonal = 50 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        
        # Add random walk component
        random_walk = np.cumsum(np.random.normal(0, 5, n_days))
        
        # Add noise
        noise = np.random.normal(0, 10, n_days)
        
        # Combine all components
        prices = base_price + trend + seasonal + random_walk + noise
        
        # Ensure prices are positive
        prices = np.maximum(prices, 1500)
        
        # Create the dataframe
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices.round(2)
        })
        
        # Add some missing values randomly (about 5%)
        missing_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
        df.loc[missing_indices, 'Price'] = np.nan
        
        # Save to CSV
        df.to_csv(self.data_path, index=False)
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(" Dataset loaded successfully!")
            print(f" Dataset shape: {self.df.shape}")
            print(f" Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            print(f" Price range: ${self.df['Price'].min():.2f} to ${self.df['Price'].max():.2f}")
            print(f" Average price: ${self.df['Price'].mean():.2f}")
            print("\n First 5 rows:")
            print(self.df.head())
            return True
        except FileNotFoundError:
            print(" Error: gold_data.csv not found!")
            print("Please make sure the file exists in the current directory.")
            return False
        except Exception as e:
            print(f" Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n🧹 Preprocessing data...")
        
        # Check if data is loaded
        if self.df is None:
            print(" Error: No data loaded. Please load data first.")
            return False
        
        # Convert Date column to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Remove missing values
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        print(f" Removed {initial_rows - len(self.df)} rows with missing values")
        
        # Sort by date
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Create technical indicators
        self.create_technical_indicators()
        
        # Create target variable (next day's price)
        self.df['Target'] = self.df['Price'].shift(-1)
        
        # Remove the last row (no target for it)
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(" Data preprocessing completed!")
        return True
    
    def create_technical_indicators(self):
        """Create technical indicators for better prediction"""
        print(" Creating technical indicators...")
        
        # Check if data is loaded
        if self.df is None:
            print(" Error: No data loaded. Please load data first.")
            return
        
        # Moving averages
        self.df['MA_5'] = self.df['Price'].rolling(window=5).mean()
        self.df['MA_20'] = self.df['Price'].rolling(window=20).mean()
        self.df['MA_50'] = self.df['Price'].rolling(window=50).mean()
        
        # Price changes
        self.df['Price_Change'] = self.df['Price'].pct_change()
        self.df['Price_Change_5'] = self.df['Price'].pct_change(periods=5)
        
        # Volatility
        self.df['Volatility'] = self.df['Price'].rolling(window=20).std()
        
        # RSI (Relative Strength Index)
        delta = self.df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.df['BB_Upper'] = self.df['MA_20'] + (self.df['Price'].rolling(window=20).std() * 2)
        self.df['BB_Lower'] = self.df['MA_20'] - (self.df['Price'].rolling(window=20).std() * 2)
        self.df['BB_Position'] = (self.df['Price'] - self.df['BB_Lower']) / (self.df['BB_Upper'] - self.df['BB_Lower'])
        
        # MACD
        exp1 = self.df['Price'].ewm(span=12).mean()
        exp2 = self.df['Price'].ewm(span=26).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9).mean()
        
        print(" Technical indicators created!")
    
    def visualize_data(self):
        """Create comprehensive visualizations"""
        print("\n Creating visualizations...")
        
        # Check if data is loaded
        if self.df is None:
            print(" Error: No data loaded. Please load data first.")
            return
        
        try:
            # Set up the plotting area
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle('Gold Price Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Price over time
            axes[0, 0].plot(self.df['Date'], self.df['Price'], linewidth=2, color='gold')
            axes[0, 0].set_title('Gold Price Over Time', fontweight='bold')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Price ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Moving averages
            axes[0, 1].plot(self.df['Date'], self.df['Price'], label='Price', alpha=0.7)
            axes[0, 1].plot(self.df['Date'], self.df['MA_5'], label='MA 5', alpha=0.8)
            axes[0, 1].plot(self.df['Date'], self.df['MA_20'], label='MA 20', alpha=0.8)
            axes[0, 1].plot(self.df['Date'], self.df['MA_50'], label='MA 50', alpha=0.8)
            axes[0, 1].set_title('Price with Moving Averages', fontweight='bold')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Price ($)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. RSI
            axes[1, 0].plot(self.df['Date'], self.df['RSI'], color='purple', linewidth=2)
            axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            axes[1, 0].set_title('RSI (Relative Strength Index)', fontweight='bold')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('RSI')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. MACD
            axes[1, 1].plot(self.df['Date'], self.df['MACD'], label='MACD', color='blue')
            axes[1, 1].plot(self.df['Date'], self.df['MACD_Signal'], label='Signal', color='red')
            axes[1, 1].set_title('MACD', fontweight='bold')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('MACD')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 5. Price distribution
            axes[2, 0].hist(self.df['Price'], bins=30, color='gold', alpha=0.7, edgecolor='black')
            axes[2, 0].set_title('Price Distribution', fontweight='bold')
            axes[2, 0].set_xlabel('Price ($)')
            axes[2, 0].set_ylabel('Frequency')
            axes[2, 0].grid(True, alpha=0.3)
            
            # 6. Correlation heatmap
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            correlation_matrix = self.df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[2, 1], fmt='.2f', square=True, cbar_kws={"shrink": .8})
            axes[2, 1].set_title('Correlation Heatmap', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            print(" Visualizations completed!")
        except Exception as e:
            print(f" Error creating visualizations: {e}")
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("\n🔧 Preparing features for ML...")
        
        # Check if data is loaded
        if self.df is None:
            print(" Error: No data loaded. Please load data first.")
            return None, None, None, None
        
        # Select features (exclude Date and Target)
        feature_columns = [col for col in self.df.columns if col not in ['Date', 'Target']]
        self.features = feature_columns
        
        X = self.df[feature_columns]
        y = self.df['Target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f" Features used: {len(feature_columns)}")
        print(f" Training samples: {len(X_train)}")
        print(f" Testing samples: {len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple machine learning models"""
        print("\n Training machine learning models...")
        
        # Check if data is provided
        if X_train is None or X_test is None or y_train is None or y_test is None:
            print(" Error: No training data provided.")
            return {}
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f" Training {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                print(f" {name} - RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, R²: {r2:.3f}")
            except Exception as e:
                print(f" Error training {name}: {e}")
        
        self.models = results
        return results
    
    def evaluate_models(self):
        """Evaluate and compare model performance"""
        print("\n Model Evaluation Results:")
        print("=" * 60)
        
        if not self.models:
            print(" No models to evaluate. Please train models first.")
            return
        
        for name, results in self.models.items():
            print(f"\n🏆 {name}:")
            print(f"   RMSE: ${results['rmse']:.2f}")
            print(f"   MAE:  ${results['mae']:.2f}")
            print(f"   R²:   {results['r2']:.3f}")
        
        # Find best model
        best_model = min(self.models.items(), key=lambda x: x[1]['rmse'])
        print(f"\n Best Model: {best_model[0]} (RMSE: ${best_model[1]['rmse']:.2f})")
    
    def plot_predictions(self, X_test, y_test):
        """Plot actual vs predicted values"""
        print("\n Creating prediction plots...")
        
        # Check if data is provided
        if X_test is None or y_test is None:
            print(" Error: No test data provided.")
            return
        
        if not self.models:
            print(" No models to plot. Please train models first.")
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for i, (name, results) in enumerate(self.models.items()):
                ax = axes[i]
                
                # Plot actual vs predicted
                ax.scatter(y_test, results['predictions'], alpha=0.6, color='gold')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                
                ax.set_xlabel('Actual Price ($)')
                ax.set_ylabel('Predicted Price ($)')
                ax.set_title(f'{name} - Actual vs Predicted\nR² = {results["r2"]:.3f}')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(" Prediction plots completed!")
        except Exception as e:
            print(f" Error creating prediction plots: {e}")
    
    def make_future_prediction(self, days_ahead=5):
        """Make predictions for future days"""
        print(f"\n🔮 Making predictions for next {days_ahead} days...")
        
        # Check if data and models are available
        if self.df is None:
            print(" Error: No data loaded. Please load data first.")
            return None
        
        if not self.models:
            print(" Error: No models trained. Please train models first.")
            return None
        
        try:
            # Get the best model
            best_model_name = min(self.models.items(), key=lambda x: x[1]['rmse'])[0]
            best_model = self.models[best_model_name]['model']
            
            # Get the latest data
            latest_data = self.df[self.features].iloc[-1:].values
            latest_data_scaled = self.scaler.transform(latest_data)
            
            # Make prediction
            prediction = best_model.predict(latest_data_scaled)[0]
            current_price = self.df['Price'].iloc[-1]
            
            print(f" Current Gold Price: ${current_price:.2f}")
            print(f" Predicted Price ({days_ahead} days ahead): ${prediction:.2f}")
            print(f" Expected Change: ${prediction - current_price:.2f} ({((prediction/current_price)-1)*100:.2f}%)")
            
            # Provide investment advice
            change_percent = ((prediction/current_price)-1)*100
            if change_percent > 2:
                print(" Investment Advice: Strong upward trend predicted - Consider buying")
            elif change_percent > 0:
                print(" Investment Advice: Slight upward trend predicted - Monitor closely")
            elif change_percent > -2:
                print(" Investment Advice: Slight downward trend predicted - Hold position")
            else:
                print(" Investment Advice: Strong downward trend predicted - Consider selling")
            
            return prediction
        except Exception as e:
            print(f" Error making prediction: {e}")
            return None
    
    def answer_question(self, question):
        """
        Answer user questions about gold prices and predictions
        
        Args:
            question (str): User's question about gold prices
            
        Returns:
            str: Answer to the user's question
        """
        # Convert question to lowercase for easier matching
        question_lower = question.lower().strip()
        
        # Check if data is loaded
        if self.df is None:
            return " Error: No data loaded. Please load data first by running the analysis."
        
        try:
            # Current price questions
            if any(keyword in question_lower for keyword in ['current', 'today', 'now', 'latest']):
                if 'price' in question_lower:
                    current_price = self.df['Price'].iloc[-1]
                    current_date = self.df['Date'].iloc[-1]
                    return f" Current gold price: ${current_price:.2f} (as of {current_date.strftime('%Y-%m-%d')})"
            
            # Prediction questions
            elif any(keyword in question_lower for keyword in ['predict', 'forecast', 'tomorrow', 'future']):
                if not self.models:
                    return " Error: No models trained. Please train models first by running the analysis."
                
                # Extract number of days from question
                days_ahead = 1  # default
                if 'tomorrow' in question_lower:
                    days_ahead = 1
                elif 'week' in question_lower:
                    days_ahead = 7
                elif 'month' in question_lower:
                    days_ahead = 30
                else:
                    # Try to extract number from question
                    import re
                    numbers = re.findall(r'\d+', question_lower)
                    if numbers:
                        days_ahead = int(numbers[0])
                
                # Make prediction
                prediction = self.make_future_prediction(days_ahead)
                if prediction is not None:
                    current_price = self.df['Price'].iloc[-1]
                    change_percent = ((prediction/current_price)-1)*100
                    return f" Predicted gold price in {days_ahead} day(s): ${prediction:.2f} ({change_percent:+.2f}% change)"
                else:
                    return " Unable to make prediction at this time."
            
            # Trend questions
            elif any(keyword in question_lower for keyword in ['trend', 'increasing', 'decreasing', 'going up', 'going down', 'direction']):
                # Calculate recent trend (last 30 days)
                recent_data = self.df.tail(30)
                if len(recent_data) >= 2:
                    start_price = recent_data['Price'].iloc[0]
                    end_price = recent_data['Price'].iloc[-1]
                    trend_percent = ((end_price/start_price)-1)*100
                    
                    if trend_percent > 1:
                        trend_direction = "increasing"
                        emoji = ""
                    elif trend_percent < -1:
                        trend_direction = "decreasing"
                        emoji = ""
                    else:
                        trend_direction = "stable"
                        emoji = ""
                    
                    return f"{emoji} Gold price trend (last 30 days): {trend_direction} ({trend_percent:+.2f}%)"
                else:
                    return " Insufficient data to determine trend."
            
            # Price range questions
            elif any(keyword in question_lower for keyword in ['range', 'high', 'low', 'minimum', 'maximum']):
                min_price = self.df['Price'].min()
                max_price = self.df['Price'].max()
                avg_price = self.df['Price'].mean()
                
                return f" Gold price range:\n   • Lowest: ${min_price:.2f}\n   • Highest: ${max_price:.2f}\n   • Average: ${avg_price:.2f}"
            
            # Volatility questions
            elif any(keyword in question_lower for keyword in ['volatility', 'volatile', 'stable']):
                recent_volatility = self.df['Volatility'].iloc[-1] if 'Volatility' in self.df.columns else self.df['Price'].tail(20).std()
                avg_volatility = self.df['Volatility'].mean() if 'Volatility' in self.df.columns else self.df['Price'].rolling(20).std().mean()
                
                if recent_volatility > avg_volatility * 1.2:
                    volatility_status = "high"
                    emoji = ""
                elif recent_volatility < avg_volatility * 0.8:
                    volatility_status = "low"
                    emoji = ""
                else:
                    volatility_status = "normal"
                    emoji = ""
                
                return f"{emoji} Current volatility is {volatility_status} (${recent_volatility:.2f} vs average ${avg_volatility:.2f})"
            
            # Technical indicator questions
            elif any(keyword in question_lower for keyword in ['rsi', 'relative strength']):
                if 'RSI' in self.df.columns:
                    current_rsi = self.df['RSI'].iloc[-1]
                    if current_rsi > 70:
                        rsi_status = "overbought"
                        emoji = "🔴"
                    elif current_rsi < 30:
                        rsi_status = "oversold"
                        emoji = "🟢"
                    else:
                        rsi_status = "neutral"
                        emoji = "🟡"
                    
                    return f"{emoji} Current RSI: {current_rsi:.1f} ({rsi_status})"
                else:
                    return " RSI data not available. Please run the analysis first."
            
            elif any(keyword in question_lower for keyword in ['macd']):
                if 'MACD' in self.df.columns:
                    current_macd = self.df['MACD'].iloc[-1]
                    current_signal = self.df['MACD_Signal'].iloc[-1]
                    
                    if current_macd > current_signal:
                        macd_signal = "bullish"
                        emoji = "🟢"
                    else:
                        macd_signal = "bearish"
                        emoji = "🔴"
                    
                    return f"{emoji} MACD: {current_macd:.2f}, Signal: {current_signal:.2f} ({macd_signal})"
                else:
                    return " MACD data not available. Please run the analysis first."
            
            # Investment advice questions
            elif any(keyword in question_lower for keyword in ['buy', 'sell', 'hold', 'invest', 'advice']):
                if not self.models:
                    return " No prediction models available. Please run the analysis first."
                
                # Make a 5-day prediction for advice
                prediction = self.make_future_prediction(5)
                if prediction is not None:
                    current_price = self.df['Price'].iloc[-1]
                    change_percent = ((prediction/current_price)-1)*100
                    
                    if change_percent > 2:
                        return "💡 Investment Advice: Strong upward trend predicted - Consider buying"
                    elif change_percent > 0:
                        return "💡 Investment Advice: Slight upward trend predicted - Monitor closely"
                    elif change_percent > -2:
                        return "💡 Investment Advice: Slight downward trend predicted - Hold position"
                    else:
                        return "💡 Investment Advice: Strong downward trend predicted - Consider selling"
                else:
                    return " Unable to provide investment advice at this time."
            
            # General help
            elif any(keyword in question_lower for keyword in ['help', 'what can you do', 'capabilities']):
                return """ I can help you with:
   • Current gold price information
   • Price predictions (tomorrow, next week, etc.)
   • Price trends and direction
   • Price ranges and statistics
   • Volatility analysis
   • Technical indicators (RSI, MACD)
   • Investment advice
   
   Try asking: "What is the gold price today?", "Predict gold price for tomorrow", or "Is the gold price increasing?" """
            
            # Default response
            else:
                return " I'm not sure how to answer that. Try asking about current prices, predictions, trends, or type 'help' for more options."
                
        except Exception as e:
            return f" Error processing your question: {str(e)}"
    
    def run_complete_analysis(self):
        """Run the complete gold prediction analysis"""
        print(" Starting Gold Price Prediction Analysis")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            return
        
        # Check data file
        if not self.check_data_file():
            return
        
        # Load data
        if not self.load_data():
            return
        
        # Preprocess data
        if not self.preprocess_data():
            return
        
        # Create visualizations
        self.visualize_data()
        
        # Prepare features
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # Train models
        self.train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        self.evaluate_models()
        
        # Plot predictions
        self.plot_predictions(X_test, y_test)
        
        # Make future prediction
        self.make_future_prediction()
        
        print("\n Analysis completed successfully!")
        print("\n Tips:")
        print("   - Use this analysis as one of many tools for investment decisions")
        print("   - Always consult with financial advisors before making investments")
        print("   - Past performance doesn't guarantee future results")

def main():
    """Main function with user-friendly interface"""
    print(" Welcome to Gold Price Predictor!")
    print("=" * 40)
    
    try:
        predictor = GoldPredictor()
        predictor.run_complete_analysis()
        
        # Interactive question-answering mode
        print("\n" + "=" * 50)
        print(" Interactive Question Mode")
        print("=" * 50)
        print("You can now ask questions about gold prices!")
        print("Type 'help' to see what you can ask, or 'quit' to exit.")
        
        while True:
            print("\n" + "-" * 30)
            question = input(" Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print(" Thank you for using Gold Price Predictor!")
                break
            elif question.lower() == '':
                continue
            else:
                # Get answer from the predictor
                answer = predictor.answer_question(question)
                print(f"\n {answer}")
                
                # Ask if user wants to make a prediction
                if any(keyword in question.lower() for keyword in ['predict', 'forecast', 'tomorrow', 'future']):
                    print("\n" + "-" * 30)
                    choice = input("Would you like to make another prediction? (y/n): ").lower().strip()
                    
                    if choice in ['y', 'yes']:
                        try:
                            days = int(input("How many days ahead to predict? (default: 5): ") or "5")
                            predictor.make_future_prediction(days)
                        except ValueError:
                            print(" Invalid input. Using default 5 days.")
                            predictor.make_future_prediction()
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using Gold Price Predictor!")
    except Exception as e:
        print(f"\n An unexpected error occurred: {e}")
        print("Please check your data and try again.")

# Run the analysis
if __name__ == "__main__":
    main()

