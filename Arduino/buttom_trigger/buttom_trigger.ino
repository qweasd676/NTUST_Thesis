//# include "Timer.h" 
//
//Timer t1;  //建立Timer物件
//
//const int TRIGGER_PIN = 11;
//const int BUTTON_PIN = 6;
//
//bool execute = false;
//
//void setup() {
//  Serial.begin(9600);
//  pinMode(TRIGGER_PIN, OUTPUT);
//  pinMode(BUTTON_PIN, INPUT_PULLUP);
//  t1.oscillate(TRIGGER_PIN, 1.16, LOW); //每1秒切換一次狀態，初始狀態LOW 2.17 1.16
//}
//
//void loop() {
//  if(execute)
//    t1.update(); //更新t1
//    
//  if(!execute && digitalRead(BUTTON_PIN) == LOW){          //如果按鍵按了
//    delay(100);
//    Serial.println("up!!");
//    execute = !execute;
//    digitalWrite(TRIGGER_PIN, LOW);
//    delay(100);
//  }
//  if(execute && digitalRead(BUTTON_PIN) != LOW){          //如果按鍵按了
//    delay(100);
//    Serial.println("down!!");
//    execute = !execute;
//    digitalWrite(TRIGGER_PIN, LOW);
//    delay(100);
//  }
//}

//# include "Timer.h" 
//
//Timer t1;  //建立Timer物件
//
//const int TRIGGER_PIN = 11;
//const int BUTTON_PIN = 6;
//
//bool execute = false;
//
//void setup() {
//  Serial.begin(9600);
//  pinMode(TRIGGER_PIN, OUTPUT);
//  pinMode(BUTTON_PIN, INPUT_PULLUP);
//}
//
//void loop() {
//  if(execute){
//    digitalWrite(TRIGGER_PIN, LOW);
//    delayMicroseconds(4000);
//    digitalWrite(TRIGGER_PIN, HIGH);
//    delayMicroseconds(4000);
//  }
//    
//  if(!execute && digitalRead(BUTTON_PIN) == LOW){          //如果按鍵按了
//    delay(100);
//    Serial.println("up!!");
//    execute = !execute;
//    digitalWrite(TRIGGER_PIN, LOW);
//    delay(100);
//  }
//  if(execute && digitalRead(BUTTON_PIN) != LOW){          //如果按鍵按了
//    delay(100);
//    Serial.println("down!!");
//    execute = !execute;
//    digitalWrite(TRIGGER_PIN, LOW);
//    delay(100);
//  }
//}


# include "Timer.h" 

Timer t1;  //建立Timer物件
const int TRIGGER_PIN_0 = 9;
const int TRIGGER_PIN_1 = 10;
const int TRIGGER_PIN_2 = 11;
const int BUTTON_PIN = 6;

bool execute = false;

void setup() {
  Serial.begin(9600);
  pinMode(TRIGGER_PIN_0, OUTPUT);
  pinMode(TRIGGER_PIN_1, OUTPUT);
  pinMode(TRIGGER_PIN_2, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
}

void loop() {
  if(execute){
    PORTB = B00000000;
    delayMicroseconds(8333);   // 8333   2100
    PORTB = B00001110;
    delayMicroseconds(8333);
  }
    
  if(!execute && digitalRead(BUTTON_PIN) == LOW){          //如果按鍵按了
    delay(100);
    Serial.println("up!!");
    execute = !execute;
    PORTB = B00000000;
    delay(100);
  }
  if(execute && digitalRead(BUTTON_PIN) != LOW){          //如果按鍵按了
    delay(100);
    Serial.println("down!!");
    execute = !execute;
    PORTB = B00000000;
    delay(100);
  }
}
