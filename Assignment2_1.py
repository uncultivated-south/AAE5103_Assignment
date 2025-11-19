import os
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = r'C:\Users\cooki\OneDrive\PolyU\AAE5103\HKIA_flight_data\flight_data\depart'
START_DATE = datetime(2023, 6, 1)
END_DATE = datetime(2023, 6, 18)
TARGET_AIRLINES = {
    'CX': 'Cathey Pacific',
    'UO': 'HK Express',
    'HX': 'Hong Kong Airlines'
}
DELAY_THRESHOLD_MINUTES = 30 


def parse_flight_status(status, planned_time):
    is_cancelled = False
    actual_dep_time = None
    delay_minutes = None

    if 'Dep' in status:
        try:
            # Extract actual departure time in HH:MM format
            actual_time_str = status.split(' ')[1]
            actual_hour = int(actual_time_str.split(':')[0])
            actual_minute = int(actual_time_str.split(':')[1])

            # Construct the actual departure datetime object
            actual_dep_time = planned_time.replace(hour=actual_hour, minute=actual_minute, second=0)

            # Overnight handling: If the actual departure time is earlier than the planned time (e.g., planned 23:55, actual 00:05),
            # assume the actual departure time is on the next day.
            if actual_dep_time < planned_time - timedelta(hours=12): # Avoid extreme cases, only consider significant overnight differences
                 actual_dep_time += timedelta(days=1)
            
            # Calculate delay in minutes
            delay_duration = actual_dep_time - planned_time
            delay_minutes = delay_duration.total_seconds() / 60
            
        except Exception as e:
            # print(f"Error parsing status {status} for planned time {planned_time}: {e}")
            pass # Ignore rows that cannot be parsed

    elif 'Cancelled' in status:
        is_cancelled = True

    return actual_dep_time, delay_minutes, is_cancelled

def load_and_process_data(data_dir, start_date, end_date):
    """
    Loads and preprocesses all departure CSV files within the specified date range.
    """
    all_flights = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        file_name = f"{date_str}-depart.csv"
        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            try:
                # Note: Based on your example, the CSV file might not have a standard header. Handle accordingly.
                # If the file includes a header, remove header=None and skiprows=1.
                df = pd.read_csv(file_path, header=None, names=['Date', 'Time', 'Status', 'Gate', 'NO.'], skiprows=1)
                
                # Data Cleaning: Remove rows containing NaN or corrupted format
                df = df.dropna(subset=['Date', 'Time', 'Status', 'NO.'])
                
                # Preprocessing: Remove potential whitespace from Time/Status/NO. fields
                for col in ['Time', 'Status', 'NO.']:
                    df[col] = df[col].astype(str).str.strip()


                # Extract IATA code
                df['IATA'] = df['NO.'].str[:2].str.upper()

                # Combine Date and Planned Time to create a datetime object
                df['Planned_Dep_Time'] = pd.to_datetime(df['Date'].str.strip() + ' ' + df['Time'], errors='coerce')
                
                # Filter out rows where time parsing failed
                df = df.dropna(subset=['Planned_Dep_Time'])

                # Apply parsing function
                results = df.apply(lambda row: parse_flight_status(row['Status'], row['Planned_Dep_Time']), axis=1, result_type='expand')
                results.columns = ['Actual_Dep_Time', 'Delay_Minutes', 'Is_Cancelled']

                df = pd.concat([df, results], axis=1)
                all_flights.append(df)

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
        else:
            print(f"Warning: File not found for {date_str}")
            
        current_date += timedelta(days=1)
        
    if not all_flights:
        return pd.DataFrame()
        
    return pd.concat(all_flights, ignore_index=True)

def calculate_delay_metrics(df, airlines):
    """
    Calculates the delay rate and average delay time.
    """
    results = {}
    
    # Filter data for target airlines
    target_df = df[df['IATA'].isin(airlines.keys())].copy()
    
    # Filter for actually departed flights (not cancelled)
    departed_df = target_df[target_df['Is_Cancelled'] == False]
    
    # Mark delayed flights
    departed_df['Is_Delayed'] = departed_df['Delay_Minutes'] >= DELAY_THRESHOLD_MINUTES

    for iata, name in airlines.items():
        airline_df = departed_df[departed_df['IATA'] == iata]
        
        total_flights = len(airline_df)
        delayed_flights = airline_df['Is_Delayed'].sum()
        
        if total_flights == 0:
            delay_rate = 0.0
            avg_delay = 0.0
        else:
            # Delay Rate = Delayed Flights / Total Departed Flights
            delay_rate = (delayed_flights / total_flights) * 100
            
            # Average delay time is calculated only for "delayed" flights
            delayed_only_df = airline_df[airline_df['Is_Delayed'] == True]
            if len(delayed_only_df) > 0:
                 avg_delay = delayed_only_df['Delay_Minutes'].mean()
            else:
                 avg_delay = 0.0
        
        results[name] = {
            'Total Departed Flights': total_flights,
            'Delayed Flights': delayed_flights,
            'Delay Rate (%)': f"{delay_rate:.2f}%",
            'Average Delay (minutes)': f"{avg_delay:.2f}"
        }
        
    return results, departed_df

def visualize_hourly_delays(df):
    """
    Visualizes the distribution of delayed flights per hour of the day.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Filter for delayed flights
    delayed_flights_df = df[df['Is_Delayed'] == True].copy()
    
    # Extract the hour from the planned departure time
    delayed_flights_df['Planned_Hour'] = delayed_flights_df['Planned_Dep_Time'].dt.hour
    
    # Count the number of delayed flights per hour
    hourly_delays = delayed_flights_df.groupby('Planned_Hour').size().reset_index(name='Delayed_Count')
    
    # Ensure all 0-23 hours are recorded, fill 0 for hours with no delays
    full_hours = pd.DataFrame({'Planned_Hour': range(24)})
    hourly_delays = pd.merge(full_hours, hourly_delays, on='Planned_Hour', how='left').fillna(0)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Planned_Hour', y='Delayed_Count', data=hourly_delays)
    plt.title('Number of Delayed Flights by Planned Departure Hour (CX, UO, HX)')
    plt.xlabel('Planned Departure Hour of Day (Hour)')
    plt.ylabel('Number of Delayed Flights')
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()


# --- Main Program ---
if __name__ == '__main__':
    print("--- Step 1: Load and Process Data ---")
    # Note: Please make sure to replace DATA_DIR with your actual path
    # Example: DATA_DIR = '/Users/yourname/Documents/AAE5103/data/depart'
    
    # Simulate data loading path check
    if DATA_DIR == 'path/to/your/depart/folder':
        print("\n!!! WARNING: Please change the DATA_DIR variable in the script to your actual data path.!!!")
    
    
    # Assuming you have set the correct DATA_DIR
    df_combined = load_and_process_data(DATA_DIR, START_DATE, END_DATE)
    
    if not df_combined.empty:
        print(f"\nSuccessfully loaded and processed {len(df_combined)} flight records.")
        
        # --- Step 2: Calculate Delay Metrics (Part 1. (1)) ---
        print("\n--- Step 2: Calculate Delay Rate and Average Delay Time ---")
        metrics, processed_df = calculate_delay_metrics(df_combined, TARGET_AIRLINES)

        print("\n✅ Results (Part 1. (1)):")
        results_table = pd.DataFrame(metrics).T
        print(results_table[['Total Departed Flights', 'Delayed Flights', 'Delay Rate (%)', 'Average Delay (minutes)']].to_markdown(numalign="left", stralign="left"))


        # --- Step 3: Visualize Hourly Delays (Part 1. (2)) ---
        print("\n--- Step 3: Visualize the number of delayed flights per hour ---")
        if not processed_df.empty:
            visualize_hourly_delays(processed_df)
            print("\n✅ Results (Part 1. (2)): Bar chart generated.")
        else:
            print("Not enough departed flight data to generate a visualization chart.")

    else:
        print("!!! ERROR: Failed to load any data. Please check the DATA_DIR path and CSV file format.")