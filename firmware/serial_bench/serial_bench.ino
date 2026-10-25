// FaultTwin edge node, check 2: sustained serial throughput from the laptop, with checksums.
//
// Frame (little-endian): 0xA5 0x5A | seq u16 | n u8 | n x 3 x int16 samples (a, b, c) | Fletcher-16 u16
// The checksum covers seq, n and the samples. n = 0 is a control frame that resets the counters.
// Every second the board prints: STATS <ms> <good frames> <bad checksums> <good bytes> <seq gaps> <junk bytes>
// BENCH_BAUD must equal the laptop's baud rate (the USB bridge passes the rate through to this chip).
#include <Arduino.h>

#ifndef BENCH_BAUD
#define BENCH_BAUD 460800
#endif

const uint8_t MAX_N = 100;

enum State : uint8_t { SYNC0, SYNC1, BODY };
State state = SYNC0;
uint8_t buf[3 + MAX_N * 6 + 2];  // seq (2), n (1), samples, checksum (2)
uint16_t pos = 0, need = 0;
uint32_t good = 0, bad = 0, good_bytes = 0, gaps = 0, junk = 0;
uint16_t last_seq = 0;
bool have_seq = false;

static uint16_t fletcher16(const uint8_t *data, uint16_t len) {
  uint16_t s1 = 0, s2 = 0;
  for (uint16_t i = 0; i < len; i++) {
    s1 = (s1 + data[i]) % 255;
    s2 = (s2 + s1) % 255;
  }
  return (uint16_t)((s2 << 8) | s1);
}

static void frame_done() {
  uint16_t body = need - 2;
  uint16_t sum = (uint16_t)(buf[body] | (buf[body + 1] << 8));
  if (fletcher16(buf, body) != sum) {
    bad++;
    return;
  }
  uint16_t seq = (uint16_t)(buf[0] | (buf[1] << 8));
  if (buf[2] == 0) {  // control frame: start a new measurement
    good = bad = good_bytes = gaps = junk = 0;
    have_seq = false;
    return;
  }
  if (have_seq && seq != (uint16_t)(last_seq + 1)) gaps++;
  last_seq = seq;
  have_seq = true;
  good++;
  good_bytes += need + 2;  // the two sync bytes count too
}

void setup() {
  Serial.begin(BENCH_BAUD);
}

void loop() {
  while (Serial.available() > 0) {
    uint8_t b = (uint8_t)Serial.read();
    switch (state) {
      case SYNC0:
        if (b == 0xA5) state = SYNC1; else junk++;
        break;
      case SYNC1:
        if (b == 0x5A) { state = BODY; pos = 0; need = 3; }
        else { junk++; state = (b == 0xA5) ? SYNC1 : SYNC0; }
        break;
      case BODY:
        buf[pos++] = b;
        if (pos == 3) {
          if (buf[2] > MAX_N) { junk += 5; state = SYNC0; break; }
          need = 3 + buf[2] * 6 + 2;
        }
        if (pos == need) { frame_done(); state = SYNC0; }
        break;
    }
  }
  static unsigned long last_ms = 0;
  if (millis() - last_ms >= 1000) {
    last_ms = millis();
    Serial.print("STATS "); Serial.print(last_ms); Serial.print(' ');
    Serial.print(good); Serial.print(' '); Serial.print(bad); Serial.print(' ');
    Serial.print(good_bytes); Serial.print(' '); Serial.print(gaps); Serial.print(' ');
    Serial.println(junk);
  }
}
