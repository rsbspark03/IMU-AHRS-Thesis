#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <TFT_eSPI.h>

// Fixed I2C pins for Arduino Mega 2560
// #define I2C_SDA 20
// #define I2C_SCL 21

Adafruit_BNO055 bno = Adafruit_BNO055(55); // Use the default Wire instance
TFT_eSPI tft = TFT_eSPI();

void setup() {
  Serial.begin(115200);
  displayInit();

  // Initialize Wire (not strictly necessary since it happens automatically)
  Wire.begin();

  // Initialize the BNO055 sensor
  if (!bno.begin()) {
    printString("No BNO055 detected", TFT_BLACK, TFT_YELLOW);
    while (1) {
      delay(10);
    }
  }

  delay(100);

  uint8_t system, gyro, accel, mag = 0;
  while (system != 3) {
    bno.getCalibration(&system, &gyro, &accel, &mag);
    printString("Calibration!!!", TFT_WHITE, TFT_RED);
    delay(100);
  }

  printString("READING...", TFT_BLACK, TFT_DARKGREEN);
  delay(100);
}

void loop() {
  delay(8); // To make sampling rate around 100hz

  imu::Vector<3> acc = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  imu::Vector<3> gyr = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

  Serial.print(gyr.x(), 4);
  Serial.print(", ");
  Serial.print(gyr.y(), 4);
  Serial.print(", ");
  Serial.print(gyr.z(), 4);
  Serial.print(", ");
  Serial.print(acc.x(), 4);
  Serial.print(", ");
  Serial.print(acc.y(), 4);
  Serial.print(", ");
  Serial.print(acc.z(), 4);
  Serial.print(", ");
  Serial.println(millis());
}

void displayInit() {
  tft.begin();
  tft.setRotation(1);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.fillScreen(TFT_BLACK);
  tft.setSwapBytes(true);
  tft.setTextFont(4);
  tft.setTextDatum(MC_DATUM);
}

void printString(String text, uint16_t textColor, uint16_t bgColor) {
  tft.fillScreen(bgColor);
  tft.setTextColor(textColor, bgColor);
  tft.drawCentreString(text, tft.width() / 2, tft.height() / 2, 4);
}
