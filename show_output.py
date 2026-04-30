
import time
import sys
from website.backend.sensor_handler import mqtt_handler, sensor_store

def main():
    print("Connecting to MQTT broker and listening for sensor data...")
    mqtt_handler.start()
    
    start_time = time.time()
    duration = 10  # Listen for 10 seconds
    
    try:
        while time.time() - start_time < duration:
            data = sensor_store.get_latest()
            if data and data['waist']:
                waist = data['waist']
                print(f"\r[WAIST SENSOR] Tilt: {waist.get('tilt', 0):.2f}° | Stability: {waist.get('stability', 0):.4f} | Accel: {waist.get('accel', {})}", end="")
            else:
                print("\rWaiting for data...", end="")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        mqtt_handler.stop()
        print("\n\nFinished listening.")

if __name__ == "__main__":
    main()
