int vibrateInput;
const int motor_pin = 8;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(motor_pin, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  while (!Serial.available()); {
    vibrateInput = Serial.readString().toInt();
    if (vibrateInput == 1) {
      digitalWrite(motor_pin,1);
      delay(500);
      //Serial.print("ON");
  } else {
      digitalWrite(motor_pin,0);
      delay(500);
      //Serial.print("OFF");
  }
  }
}
