
import paho.mqtt.client as mqtt
import json
import threading
import time
from collections import deque

class SensorDataStore:
    def __init__(self, maxlen=100):
        self.waist_data = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        
    def add_waist(self, data):
        with self.lock:
            data['server_time'] = time.time()
            self.waist_data.append(data)
            
    def get_latest(self):
        with self.lock:
            return {
                "waist": self.waist_data[-1] if self.waist_data else None
            }
            
    def get_history(self):
        with self.lock:
            return {
                "waist": list(self.waist_data)
            }

sensor_store = SensorDataStore()

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe("fitness/sensors/#")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic
        if "waist" in topic:
            sensor_store.add_waist(payload)
    except Exception as e:
        print(f"[MQTT] Error parsing message: {e}")

class MQTTSensorClient:
    def __init__(self, broker="10.239.220.14", port=1883):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.broker = broker
        self.port = port
        self.thread = None

    def start(self):
        try:
            self.client.connect(self.broker, self.port, 60)
            self.thread = threading.Thread(target=self.client.loop_forever, daemon=True)
            self.thread.start()
            print(f"[MQTT] Listener started on {self.broker}:{self.port}")
        except Exception as e:
            print(f"[MQTT] Failed to start: {e}")

    def stop(self):
        self.client.disconnect()

mqtt_handler = MQTTSensorClient()
