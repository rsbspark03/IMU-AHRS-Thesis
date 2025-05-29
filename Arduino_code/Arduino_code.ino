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
  while ((accel != 3) || (mag != 3) || (gyro != 3)) {
    bno.getCalibration(&system, &gyro, &accel, &mag);
    Serial.print("Calibration Levels - ");
    Serial.print("System: "); Serial.print(system);
    Serial.print(", Gyro: "); Serial.print(gyro);
    Serial.print(", Accel: "); Serial.print(accel);
    Serial.print(", Mag: "); Serial.println(mag);
    printString("Calibration!!!", TFT_WHITE, TFT_RED);
    delay(100);
  }
  Serial.println("READING-----------------");
  printString("READING...", TFT_BLACK, TFT_DARKGREEN);
  delay(5000);
}

void loop() {
  // delay(8) To make sampling rate around 100hz
  delay(8);
  sensors_event_t  orientationData;
  bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  imu::Vector<3> acc = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  imu::Vector<3> mag = bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
  imu::Vector<3> gyr = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  imu::Quaternion quat = bno.getQuat();


  char buffer[12];  // Enough to hold formatted float
  
 
  dtostrf(-acc.y(), 8, 4, buffer);
  Serial.print(buffer);
  Serial.print(", ");
  dtostrf(acc.x(), 8, 4, buffer);
  Serial.print(buffer);
  Serial.print(", ");
  dtostrf(acc.z(), 8, 4, buffer);
  Serial.print(buffer);
  
  Serial.print(", ");
  dtostrf(-mag.y(), 8, 4, buffer);
  Serial.print(buffer);
  Serial.print(", ");
  dtostrf(mag.x(), 8, 4, buffer);
  Serial.print(buffer);
  Serial.print(", ");
  dtostrf(mag.z(), 8, 4, buffer);
  Serial.print(buffer);
  
  Serial.print(", ");
  dtostrf(-gyr.y(), 8, 4, buffer);
  Serial.print(buffer);
  Serial.print(", ");
  dtostrf(gyr.x(), 8, 4, buffer);
  Serial.print(buffer);
  Serial.print(", ");
  dtostrf(gyr.z(), 8, 4, buffer);
  Serial.print(buffer);

  Serial.print(", ");
  Serial.print(millis());
  Serial.print(", ");
  Serial.print(orientationData.orientation.y, 4);
  Serial.print(", ");
  Serial.print(-orientationData.orientation.z, 4);
  Serial.print(", ");
  Serial.print(orientationData.orientation.x, 4);
  
  Serial.print(", ");
  Serial.print(quat.w(), 4);
  Serial.print(", ");
  Serial.print(quat.x(), 4);
  Serial.print(", ");
  Serial.print(quat.y(), 4);
  Serial.print(", ");
  Serial.print(quat.z(), 4);
  //
  Serial.println(' ');
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
