
#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <ArduinoJson.h>

// WiFi Configuration
const char* ssid = "RUPA";
const char* password = "vivek@3044";

// MQTT Configuration
const char* mqtt_server = "10.239.220.14";
const int mqtt_port = 1883;

WiFiClient espClient;
PubSubClient client(espClient);

// MPU6050 Registers
const int MPU_ADDR = 0x68;

// Calibration variables
float tilt_offset = 0;
bool calibrated = false;

void setup_wifi() {
  delay(10);
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32_Waist_MPU6050")) {
      Serial.println("connected");
    } else {
      delay(5000);
    }
  }
}

void calibrate() {
  Serial.println("Calibrating waist origin... Please stand still and upright.");
  float sum_tilt = 0;
  int samples = 50;
  
  for(int i = 0; i < samples; i++) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6, true);
    
    int16_t ax = Wire.read() << 8 | Wire.read();
    int16_t ay = Wire.read() << 8 | Wire.read();
    int16_t az = Wire.read() << 8 | Wire.read();
    
    float acc_y = ay / 16384.0;
    float acc_z = az / 16384.0;
    sum_tilt += atan2(acc_y, acc_z) * 180 / PI;
    delay(20);
  }
  
  tilt_offset = sum_tilt / samples;
  calibrated = true;
  Serial.print("Origin set at: ");
  Serial.println(tilt_offset);
}

void setup() {
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);

  Wire.begin(6, 7); // SDA=6, SCL=7 (Xiao ESP32 C3)
  
  // Power up MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  delay(500);
  calibrate();
  Serial.println("MPU6050 Activated via Raw I2C");
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Read data
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  int16_t ax = Wire.read() << 8 | Wire.read();
  int16_t ay = Wire.read() << 8 | Wire.read();
  int16_t az = Wire.read() << 8 | Wire.read();
  int16_t temp = Wire.read() << 8 | Wire.read();
  int16_t gx = Wire.read() << 8 | Wire.read();
  int16_t gy = Wire.read() << 8 | Wire.read();
  int16_t gz = Wire.read() << 8 | Wire.read();

  float acc_x = ax / 16384.0;
  float acc_y = ay / 16384.0;
  float acc_z = az / 16384.0;
  float gyr_x = gx / 131.0;
  float gyr_y = gy / 131.0;
  float gyr_z = gz / 131.0;

  // Calculate tilt relative to calibrated origin
  float raw_tilt = atan2(acc_y, acc_z) * 180 / PI;
  float relative_tilt = raw_tilt - tilt_offset;

  StaticJsonDocument<256> doc;
  doc["sensor"] = "waist_mpu";
  doc["timestamp"] = millis();
  
  JsonObject accel = doc.createNestedObject("accel");
  accel["x"] = acc_x * 9.81;
  accel["y"] = acc_y * 9.81;
  accel["z"] = acc_z * 9.81;

  doc["tilt"] = relative_tilt;
  doc["stability"] = (abs(gyr_x) + abs(gyr_y) + abs(gyr_z)) / 100.0;

  char buffer[256];
  serializeJson(doc, buffer);
  client.publish("fitness/sensors/waist", buffer);

  delay(100);
}
