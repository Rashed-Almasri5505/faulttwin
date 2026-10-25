// FaultTwin edge node, check 1: the UNO R4 WiFi's 12x8 LED matrix shows an icon.
// The serial line lets the laptop confirm the sketch is running.
#include "Arduino_LED_Matrix.h"

ArduinoLEDMatrix matrix;

// 8 rows x 12 columns: a check mark (the "healthy" state).
uint8_t check_mark[8][12] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
  {0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0},
  {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
};

void setup() {
  Serial.begin(115200);
  matrix.begin();
  matrix.renderBitmap(check_mark, 8, 12);
}

void loop() {
  static unsigned long last_ms = 0;
  if (millis() - last_ms >= 1000) {
    last_ms = millis();
    Serial.println("FAULTTWIN MATRIX OK");
  }
}
