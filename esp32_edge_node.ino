/*
  Agri-Tech DSS — ESP32-CAM Edge Node
  Publishes multi-modal sensor + image data to MQTT broker every 5 seconds.
  Board: AI-Thinker ESP32-CAM
  Libraries: PubSubClient, ArduinoJson, Base64 (built-in esp32 camera libs)
*/

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include "Base64.h"

// ── WiFi / MQTT config ────────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_BROKER   = "broker.hivemq.com";
const int   MQTT_PORT     = 1883;
const char* PUB_TOPIC     = "agri/sensors";
const char* SUB_TOPIC     = "agri/decision";
const char* CLIENT_ID     = "esp32cam-agri-node";

// ── Sensor pins (adjust for your wiring) ─────────────────────────────────
// Soil moisture: analog pin (0-4095 → 0-100%)
const int MOISTURE_PIN = 34;

// AI-Thinker ESP32-CAM pin mapping
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

WiFiClient   wifiClient;
PubSubClient mqttClient(wifiClient);

// ── Camera init ───────────────────────────────────────────────────────────
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_240X240; // closest to 224x224
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] Init failed: 0x%x\n", err);
  } else {
    Serial.println("[CAM] Initialized");
  }
}

// ── WiFi ──────────────────────────────────────────────────────────────────
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WiFi] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.printf("\n[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());
}

// ── MQTT ──────────────────────────────────────────────────────────────────
void onDecision(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
  Serial.printf("[MQTT] Decision received: %s\n", msg.c_str());
  // TODO: trigger LED, buzzer, or relay based on decision
}

void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("[MQTT] Connecting...");
    if (mqttClient.connect(CLIENT_ID)) {
      Serial.println(" connected");
      mqttClient.subscribe(SUB_TOPIC);
    } else {
      Serial.printf(" failed (rc=%d), retry in 5s\n", mqttClient.state());
      delay(5000);
    }
  }
}

// ── Sensor readings (stubs — replace with real sensor libraries) ──────────
int   readNitrogen()   { return random(40, 120); }   // mg/kg
int   readPhosphorus() { return random(20, 100); }
int   readPotassium()  { return random(30, 150); }
float readPH()         { return 5.5 + random(0, 30) / 10.0; }
int   readMoisture()   {
  int raw = analogRead(MOISTURE_PIN);
  return map(raw, 0, 4095, 100, 0); // invert: dry=high ADC
}
float readTemp()       { return 22.0 + random(0, 15); }   // °C
int   readHumidity()   { return 50 + random(0, 40); }     // %

// ── Main publish ──────────────────────────────────────────────────────────
void publishPayload() {
  // 1. Capture image
  camera_fb_t* fb = esp_camera_fb_get();
  String imgB64   = "";
  if (fb) {
    imgB64 = base64::encode(fb->buf, fb->len);
    esp_camera_fb_return(fb);
  } else {
    Serial.println("[CAM] Capture failed");
  }

  // 2. Build JSON
  DynamicJsonDocument doc(imgB64.length() + 256);
  doc["N"]            = readNitrogen();
  doc["P"]            = readPhosphorus();
  doc["K"]            = readPotassium();
  doc["pH"]           = readPH();
  doc["moist"]        = readMoisture();
  doc["temp"]         = readTemp();
  doc["hum"]          = readHumidity();
  doc["image_base64"] = imgB64;

  // 3. Serialize & publish
  String payload;
  serializeJson(doc, payload);

  if (mqttClient.publish(PUB_TOPIC, payload.c_str(), false)) {
    Serial.printf("[MQTT] Published %d bytes\n", payload.length());
  } else {
    Serial.println("[MQTT] Publish failed (payload may be too large)");
  }
}

// ── Arduino lifecycle ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  initCamera();
  connectWiFi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(onDecision);
  mqttClient.setBufferSize(1024 * 64); // 64 KB for Base64 image
}

void loop() {
  if (!mqttClient.connected()) reconnectMQTT();
  mqttClient.loop();
  publishPayload();
  delay(5000); // publish every 5 seconds
}
