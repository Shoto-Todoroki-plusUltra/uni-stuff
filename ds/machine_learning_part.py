import csv
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200  
PC_CSV_FILENAME = "data.csv"
V0 = 5.0  

def collect_data_from_pico(port, baud, filename, collection_duration_seconds=60):
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        return False

    header_written = os.path.exists(filename) and os.path.getsize(filename) > 0

    with open(filename, 'a', newline='') as f_pc:
        csv_writer = csv.writer(f_pc)
        if not header_written:
            csv_writer.writerow(['VOLTAGE_RECORDED', 'WIND_SPEED_RECORDED', 'T_V'])
            print("CSV header written.")

        start_time = time.time()
        print(f"Collecting data for {collection_duration_seconds} seconds...")

        try:
            while (time.time() - start_time) < collection_duration_seconds:
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        if line:
                            print(f"Received: {line}")
                            parts = line.split(',')
                            if len(parts) == 2:
                                v_recorded = float(parts[0])
                                wind_speed_recorded = float(parts[1])
                                t_v = (v_recorded - V0) / V0 if V0 != 0 else 0
                                csv_writer.writerow([v_recorded, wind_speed_recorded, t_v])
                            else:
                                print(f"Warning: Malformed data received: {line}")
                    except ValueError as ve:
                        print(f"Warning: Could not parse data '{line}'. Error: {ve}")
                    except UnicodeDecodeError:
                        print(f"Warning: Could not decode data (likely partial line): {line}")
                    except Exception as e:
                        print(f"Error processing line: {e}")
                time.sleep(0.01) 

        except KeyboardInterrupt:
            print("Data collection stopped by user.")
        except Exception as e:
            print(f"An error occurred during data collection: {e}")
        finally:
            ser.close()
            print("Serial port closed. Data collection finished.")
            return True

def train_and_evaluate_model(csv_filename):
    try:
        data = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: Data file '{csv_filename}' not found. Collect data first.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Data file '{csv_filename}' is empty. Collect more data.")
        return
    V0 = 5.0
    data['T_V'] = (data['VOLTAGE_RECORDED'] - V0) / V0


    if data.shape[0] < 10: 
        print(f"Not enough data ({data.shape[0]} rows) to train a model. Collect more data.")
        return

    print(f"Loaded {data.shape[0]} records from {csv_filename}.")

    X = data[['T_V']]
    y = data['WIND_SPEED_RECORDED']

    if X.isnull().any().any() or y.isnull().any().any():
        print("Warning: Data contains NaN values. Dropping rows with NaN.")
        data.dropna(subset=['T_V', 'WIND_SPEED_RECORDED'], inplace=True)
        X = data[['T_V']]
        y = data['WIND_SPEED_RECORDED']
        if data.shape[0] < 10:
             print(f"Not enough data ({data.shape[0]} rows) after dropping NaN values. Collect more data.")
             return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data size: {X_train.shape[0]}")
    print(f"Testing data size: {X_test.shape[0]}")

    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)

    model.fit(X_train, y_train)
    print(f"\nModel trained: {model.__class__.__name__}")

    if hasattr(model, 'oob_score_') and model.oob_score_:
         print(f"Model OOB Score: {model.oob_score_:.4f}")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  R-squared (R2 Score): {r2:.4f}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
    plt.xlabel("Actual Wind Speed")
    plt.ylabel("Predicted Wind Speed")
    plt.title("Actual vs. Predicted Wind Speed")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plot_X = pd.DataFrame({'T_V': sorted(X_test['T_V'].unique())})
    if not plot_X.empty:
        plot_y = model.predict(plot_X)

    plt.scatter(X['T_V'], y, alpha=0.5, label='All Data Points')
    if not plot_X.empty:
        plt.plot(plot_X['T_V'], plot_y, color='red', linewidth=2, label='Model Prediction Line')
    plt.xlabel("T(V) = (V - V0) / V0")
    plt.ylabel("Wind Speed Recorded")
    plt.title("T(V) vs. Wind Speed and Model Fit")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model


if __name__ == "__main__":
    # data_collection_time_seconds = 30 # 30 seconds for a quick test

    # success = collect_data_from_pico(SERIAL_PORT, BAUD_RATE, PC_CSV_FILENAME, data_collection_time_seconds)
    # if not success:
    #     print("Data collection failed. Exiting.")
    #     exit()

    trained_model = train_and_evaluate_model(PC_CSV_FILENAME)

    if trained_model:
        print("\n--- Prediction Example ---")
        example_t_values = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
        for t_v in example_t_values:
            predict_df = pd.DataFrame([[t_v]], columns=['T_V'])
            predicted_speed = trained_model.predict(predict_df)
            print(f"For T(V) = {t_v:.3f}, predicted wind speed: {predicted_speed[0]:.2f}")
