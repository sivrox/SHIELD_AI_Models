#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h"
#include "heartRate.h" 
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ============================================
// CONFIGURATION
// ============================================
#define SCREEN_WIDTH 128 
#define SCREEN_HEIGHT 64 
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define DEVICE_NAME         "S.H.I.E.L.D. ESP32"

// ============================================
// OBJECTS
// ============================================
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
MAX30105 particleSensor;
BLEServer* pServer = NULL;
BLECharacteristic* pSensorCharacteristic = NULL;

// ============================================
// BLE STATUS
// ============================================
bool deviceConnected = false;

// ============================================
// REAL-TIME SENSOR DATA
// ============================================
// Heart Rate variables
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute;
int beatAvg;
float instantBPM = 0;

// SpO2 variables
int instantSpo2 = 98;
uint32_t irMax = 0; 
uint32_t irMin = 0xFFFFFFFF;
uint32_t redMax = 0; 
uint32_t redMin = 0xFFFFFFFF;
int sampleCounter = 0;
bool isFingerPresent = false;

// ============================================
// BLE CALLBACKS
// ============================================
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        Serial.println("📱 App Connected!");
    };

    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        Serial.println("📱 App Disconnected - Restarting advertising");
        pServer->getAdvertising()->start();
    }
};

// ============================================
// SEND REAL-TIME DATA VIA BLE (NO HRV)
// ============================================
void sendRealtimeData() {
    if (!deviceConnected) return;
    
    // Create JSON with BPM and SpO2 only
    String jsonData = "{";
    jsonData += "\"bpm\":" + String(instantBPM, 1) + ",";
    jsonData += "\"spo2\":" + String(instantSpo2);
    jsonData += "}";
    
    pSensorCharacteristic->setValue(jsonData.c_str());
    pSensorCharacteristic->notify();
}

// ============================================
// UPDATE OLED DISPLAY (NO HRV)
// ============================================
void updateDisplay(long irValue) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(0,0);
    
    if (irValue > 50000) {  // Finger present
        display.print("S.H.I.E.L.D. LIVE");
        display.drawLine(0, 9, 128, 9, WHITE);
        
        display.setCursor(30, 15); display.print("BPM");
        display.setCursor(80, 15); display.print("SpO2");
        
        display.setTextSize(2);
        display.setCursor(25, 28);
        if (instantBPM > 0) display.print((int)instantBPM); else display.print("--");
        display.setCursor(75, 28);
        display.print(instantSpo2); display.print("%");
        
        display.setTextSize(1);
        display.setCursor(0, 55);
        if (deviceConnected) display.print("BLE: Connected");
        else display.print("BLE: Waiting");
        
    } else {  // No finger
        display.print("S.H.I.E.L.D. BLE");
        display.drawLine(0, 9, 128, 9, WHITE);
        display.setCursor(0, 20);
        display.println("Place finger on");
        display.println("sensor");
        display.setCursor(0, 50);
        display.print("Name: " + String(DEVICE_NAME));
    }
    display.display();
}

// ============================================
// SETUP
// ============================================
void setup() {
    Serial.begin(115200);
    Serial.println("\n\n=== S.H.I.E.L.D. BLE STARTING ===");
    
    // ===== 1. Initialize BLE =====
    Serial.println("Initializing BLE...");
    BLEDevice::init(DEVICE_NAME);
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pSensorCharacteristic = pService->createCharacteristic(
                                CHARACTERISTIC_UUID,
                                BLECharacteristic::PROPERTY_READ |
                                BLECharacteristic::PROPERTY_NOTIFY
                            );
    pSensorCharacteristic->addDescriptor(new BLE2902());
    pService->start();
    pServer->getAdvertising()->start();
    Serial.println("✅ BLE Server started as: " + String(DEVICE_NAME));
    
    // ===== 2. Initialize I2C and OLED =====
    Serial.println("Initializing I2C...");
    Wire.begin(21, 22, 100000); 

    if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println("❌ OLED not found!");
        while(1);
    }
    Serial.println("✅ OLED initialized");
    
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(0, 0);
    display.println("S.H.I.E.L.D. V3.0");
    display.println("BLE: " + String(DEVICE_NAME));
    display.println("Starting...");
    display.display();
    
    // ===== 3. Initialize MAX30102 =====
    Serial.println("Initializing MAX30102...");
    if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
        Serial.println("❌ MAX30102 not found!");
        display.println("Sensor Error!");
        display.display();
        while (1);
    }
    Serial.println("✅ MAX30102 initialized");
    
    // Configure sensor
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
    
    display.println("Ready!");
    display.display();
    delay(1500);
}

// ============================================
// MAIN LOOP - REAL-TIME PROCESSING (NO HRV)
// ============================================
void loop() {
    // ===== 1. READ SENSOR =====
    long irValue = particleSensor.getIR();
    long redValue = particleSensor.getRed();

    if (irValue > 50000) {  // Finger detected
        if (!isFingerPresent) {
            isFingerPresent = true;
            lastBeat = 0;
            rateSpot = 0;
            beatAvg = 0;
        }

        // Beat detection
        if (checkForBeat(irValue) == true) {
            long delta = millis() - lastBeat;
            lastBeat = millis();
            
            beatsPerMinute = 60 / (delta / 1000.0);
            instantBPM = beatsPerMinute;

            // Store for averaging
            if (beatsPerMinute >= 40 && beatsPerMinute <= 220) {
                rates[rateSpot++] = (byte)beatsPerMinute;
                rateSpot %= RATE_SIZE;
                
                beatAvg = 0;
                for (byte x = 0; x < RATE_SIZE; x++)
                    beatAvg += rates[x];
                beatAvg /= RATE_SIZE;
            }
            
            // NO HRV CALCULATION
            
            // Send immediately on beat detection
            sendRealtimeData();
        }

        // SpO2 calculation
        if (irValue > irMax) irMax = irValue;
        if (irValue < irMin) irMin = irValue;
        if (redValue > redMax) redMax = redValue;
        if (redValue < redMin) redMin = redValue;
        sampleCounter++;

        if (sampleCounter >= 100) {
            float irAC = irMax - irMin;
            float redAC = redMax - redMin;
            float irDC = irMax;
            float redDC = redMax;
            
            float R = (redAC / redDC) / (irAC / irDC);
            instantSpo2 = 104 - (17 * R);
            if (instantSpo2 > 100) instantSpo2 = 100;
            if (instantSpo2 < 70) instantSpo2 = 98;
            
            irMax = 0; irMin = 0xFFFFFFFF;
            redMax = 0; redMin = 0xFFFFFFFF;
            sampleCounter = 0;
            
            sendRealtimeData();
        }
    } else {
        if (isFingerPresent) {
            isFingerPresent = false;
            instantBPM = 0;
            sendRealtimeData();
        }
    }

    // Continuous stream
    static unsigned long lastNotifyTime = 0;
    if (deviceConnected && millis() - lastNotifyTime > 200) {
        lastNotifyTime = millis();
        sendRealtimeData();
    }

    // Update OLED
    static unsigned long lastDisplayUpdate = 0;
    if (millis() - lastDisplayUpdate > 200) {
        updateDisplay(irValue);
        lastDisplayUpdate = millis();
    }
    
    delay(20);
}