import machine
import utime
import uos

VOLTAGE_PIN_NUM = 26  # ADC0
adc_voltage = machine.ADC(VOLTAGE_PIN_NUM)

ANEMOMETER_PIN_NUM = 27  # ADC1
adc_anemometer = machine.ADC(ANEMOMETER_PIN_NUM)

V0 = 5.0  # Volts

COLLECTION_INTERVAL_S = 0.1

SAVE_TO_PICO_CSV = True # Set to False if you only want to send to PC
PICO_CSV_FILENAME = "data.csv"
MAX_PICO_FILE_SIZE_KB = 1024*1024 

def read_voltage_wire():
    adc_value = adc_voltage.read_u16()
    voltage = (adc_value / 65535.0) * V0 
    return voltage

def read_wind_speed_anemometer():
    adc_value = adc_anemometer.read_u16()
    simulated_wind_speed = (adc_value / 65535.0) * 30.0 
    return simulated_wind_speed

def get_pico_file_size(filename):
    try:
        return uos.stat(filename)[6] / 1024 
    except OSError:
        return 0

if SAVE_TO_PICO_CSV:
    try:
        uos.stat(PICO_CSV_FILENAME)
    except OSError:
        with open(PICO_CSV_FILENAME, "w") as f_pico:
            f_pico.write("VOLTAGE_RECORDED,WIND_SPEED_RECORDED\n")
    print(f"Logging data to {PICO_CSV_FILENAME} on Pico.")


while True:
    try:
        v_recorded = read_voltage_wire()
        wind_speed_recorded = read_wind_speed_anemometer()

        data_string = f"{v_recorded:.4f},{wind_speed_recorded:.2f}"

        if SAVE_TO_PICO_CSV:
            current_size_kb = get_pico_file_size(PICO_CSV_FILENAME)
            if current_size_kb < MAX_PICO_FILE_SIZE_KB:
                try:
                    with open(PICO_CSV_FILENAME, "a") as f_pico:
                        f_pico.write(data_string + "\n")
                except Exception as e:
                    print(f"Error writing to Pico CSV: {e}")
            else:
                if SAVE_TO_PICO_CSV: 
                    print(f"Pico CSV file '{PICO_CSV_FILENAME}' reached max size of {MAX_PICO_FILE_SIZE_KB} KB. Stopping on-Pico logging.")
                    SAVE_TO_PICO_CSV = False 

        utime.sleep(COLLECTION_INTERVAL_S)

    except KeyboardInterrupt:
        print("Data acquisition stopped by user.")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        utime.sleep(1)
