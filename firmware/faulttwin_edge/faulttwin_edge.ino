// FaultTwin edge node for the Arduino UNO R4 WiFi.
//
// The laptop streams the simulated inverter's measured phase currents, one fundamental cycle per frame
// (100 block-averaged samples per phase). This board runs the edge monitor on them: Park's vector modulus,
// normalized mean currents, KCL residuals, w1 amplitude ratios and RMS over a 2-cycle window, the same
// threshold checks with latching and hysteresis as faulttwin/monitor.py, and reports features and events back.
// The 12x8 LED matrix shows the state; a button or a jumper wire from D2 to GND triggers a Mystery fault.
// The protocol is described in faulttwin/edge.py; faulttwin.edge.BoardModel is this algorithm in Python.
#include <Arduino.h>
#include <math.h>
#include "Arduino_LED_Matrix.h"

#define EDGE_BAUD 460800
#define FW_VERSION 2                       // 2: D2 needs a steady level, and one press per 2 s
#define MYSTERY_PIN 2

// ---------------------------------------------------------------- protocol
const uint16_t MAX_PAYLOAD = 8 + 100 * 6;   // a cycle frame with 100 samples per phase
const float SAMPLE_LSB_A = 0.002f;
const uint8_t FLAG_RESET = 1, FLAG_OP_CHANGE = 2;

enum RxState : uint8_t { SYNC0, SYNC1, HEADER, BODY };
RxState rx_state = SYNC0;
uint8_t rx_buf[3 + MAX_PAYLOAD + 2];
uint16_t rx_pos = 0, rx_need = 0;

static uint16_t fletcher16(const uint8_t *data, uint16_t len) {
  uint16_t s1 = 0, s2 = 0;
  for (uint16_t i = 0; i < len; i++) {
    s1 = (s1 + data[i]) % 255;
    s2 = (s2 + s1) % 255;
  }
  return (uint16_t)((s2 << 8) | s1);
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static float rdf(const uint8_t *p) { float f; memcpy(&f, p, 4); return f; }

// ---------------------------------------------------------------- monitor (as faulttwin/monitor.py)
struct Thresholds {
  float mu_abs_max = 0.10f;
  uint8_t mu_consecutive_cycles = 2;
  float r_kcl_dc_max = 0.03f, r_kcl_w1_max = 0.05f;
  float band_lo = 0.90f, band_hi = 1.10f;
  float low_rms_frac = 0.05f, i_s_step_frac = 0.20f;
  uint8_t warmup_cycles = 5;
  float min_i_s_A = 1.0f, clear_ratio = 0.8f;
  uint8_t clear_cycles = 3, window_cycles = 2;
  float phase_offset = 0.45f;
} th;

enum Check : uint8_t { MEAN_CURRENT, KCL_DC, KCL_W1, AMPLITUDE_RATIO, PHASE_LOSS, NO_CURRENT, TRANSIENT, N_CHECKS };
const char *CHECK_NAME[N_CHECKS] = {"mean_current", "kcl_dc", "kcl_w1", "amplitude_ratio", "phase_loss", "no_current", "transient"};
const char *CHECK_SEV[N_CHECKS] = {"high", "medium", "medium", "medium", "high", "high", "low"};

struct CycleSums {
  uint16_t n;
  float sum[3], sumsq[3], re[3], im[3];
  float total_sum, total_re, total_im, hyp;
};

const uint8_t MAX_WINDOW = 4;
CycleSums window_buf[MAX_WINDOW];
uint8_t window_len = 0;                      // cycles held, newest last
uint32_t seen = 0;
uint8_t mu_streak = 0, hold = 0;
bool have_prev = false;
float prev_i_s = 0.0f;
bool active[N_CHECKS];
uint8_t clear_streak[N_CHECKS];

float cos_tab[100], sin_tab[100];
uint16_t tab_n = 0;
float tab_offset = -1.0f;

struct Features {
  bool ready;
  float i_s, mu[3], r_dc, r_w1, ar[3], rms[3];
} feat;

static void monitor_reset() {
  window_len = 0; seen = 0; mu_streak = 0; hold = 0; have_prev = false; prev_i_s = 0.0f;
  for (uint8_t c = 0; c < N_CHECKS; c++) { active[c] = false; clear_streak[c] = 0; }
}

static void make_tables(uint16_t n) {
  if (n == tab_n && th.phase_offset == tab_offset) return;
  for (uint16_t j = 0; j < n; j++) {
    float theta = 2.0f * (float)M_PI * ((float)j + th.phase_offset) / (float)n;
    cos_tab[j] = cosf(theta);
    sin_tab[j] = sinf(theta);
  }
  tab_n = n;
  tab_offset = th.phase_offset;
}

static bool phase_loss(const float *rms, float frac, float min_ref) {
  for (uint8_t k = 0; k < 3; k++) {
    float ref = 0.0f;
    for (uint8_t j = 0; j < 3; j++) if (j != k) ref += rms[j];
    ref *= 0.5f;
    if (ref > min_ref && rms[k] < frac * ref) return true;
  }
  return false;
}

static void window_features() {
  float n = 0, hyp = 0, tsum = 0, tre = 0, tim = 0, sum[3] = {0}, sumsq[3] = {0}, re[3] = {0}, im[3] = {0};
  for (uint8_t w = 0; w < window_len; w++) {
    CycleSums &c = window_buf[w];
    n += c.n; hyp += c.hyp; tsum += c.total_sum; tre += c.total_re; tim += c.total_im;
    for (uint8_t k = 0; k < 3; k++) { sum[k] += c.sum[k]; sumsq[k] += c.sumsq[k]; re[k] += c.re[k]; im[k] += c.im[k]; }
  }
  feat.i_s = hyp / n;
  float scale = feat.i_s > 1e-9f ? feat.i_s : INFINITY;
  float amp[3], amp_mean = 0.0f;
  for (uint8_t k = 0; k < 3; k++) {
    feat.mu[k] = (sum[k] / n) / scale;
    amp[k] = 2.0f * sqrtf(re[k] * re[k] + im[k] * im[k]) / n;
    amp_mean += amp[k] / 3.0f;
    feat.rms[k] = sqrtf(sumsq[k] / n);
  }
  for (uint8_t k = 0; k < 3; k++) feat.ar[k] = amp_mean > 1e-9f ? amp[k] / amp_mean : 1.0f;
  feat.r_dc = fabsf(tsum / n) / scale;
  feat.r_w1 = 2.0f * sqrtf(tre * tre + tim * tim) / n / scale;
}

static void show_fault(uint8_t phase, bool down);

static void process_cycle(uint32_t cycle, float m, uint8_t flags, const uint8_t *samples, uint16_t n) {
  if (flags & FLAG_RESET) monitor_reset();
  if (flags & FLAG_OP_CHANGE) hold = max(hold, (uint8_t)(th.window_cycles + 1));
  make_tables(n);
  CycleSums c = {};
  c.n = n;
  for (uint16_t j = 0; j < n; j++) {
    float x[3];
    for (uint8_t k = 0; k < 3; k++) x[k] = (int16_t)rd16(samples + 6 * j + 2 * k) * SAMPLE_LSB_A;
    float total = x[0] + x[1] + x[2];
    for (uint8_t k = 0; k < 3; k++) {
      c.sum[k] += x[k]; c.sumsq[k] += x[k] * x[k];
      c.re[k] += x[k] * cos_tab[j]; c.im[k] += x[k] * sin_tab[j];
    }
    c.total_sum += total; c.total_re += total * cos_tab[j]; c.total_im += total * sin_tab[j];
    float ia = (2.0f / 3.0f) * (x[0] - 0.5f * x[1] - 0.5f * x[2]);
    float ib = (x[1] - x[2]) / sqrtf(3.0f);
    c.hyp += sqrtf(ia * ia + ib * ib);
  }
  uint8_t wmax = th.window_cycles < MAX_WINDOW ? th.window_cycles : MAX_WINDOW;
  if (window_len == wmax) { for (uint8_t w = 1; w < window_len; w++) window_buf[w - 1] = window_buf[w]; window_len--; }
  window_buf[window_len++] = c;
  seen++;
  float cycle_i_s = c.hyp / c.n;
  bool low_current = cycle_i_s < th.min_i_s_A;
  bool stopped = m == 0.0f;
  feat.ready = seen > th.warmup_cycles && window_len == wmax;
  if (feat.ready) window_features();
  bool flagged[N_CHECKS] = {false}, clr[N_CHECKS];
  for (uint8_t k = 0; k < N_CHECKS; k++) clr[k] = true;
  bool holding = hold > 0;
  if (stopped || !feat.ready) {
    mu_streak = 0;
  } else if (low_current) {
    flagged[NO_CURRENT] = true; clr[NO_CURRENT] = false; mu_streak = 0;
  } else {
    if (!holding) {
      float cmu = 0.0f;
      for (uint8_t k = 0; k < 3; k++) cmu = max(cmu, fabsf(c.sum[k] / c.n) / cycle_i_s);
      mu_streak = cmu > th.mu_abs_max ? (uint8_t)min(255, mu_streak + 1) : 0;
    }
    float mu_max = 0.0f;
    for (uint8_t k = 0; k < 3; k++) mu_max = max(mu_max, fabsf(feat.mu[k]));
    flagged[MEAN_CURRENT] = mu_streak >= th.mu_consecutive_cycles;
    clr[MEAN_CURRENT] = mu_max < th.clear_ratio * th.mu_abs_max;
    flagged[KCL_DC] = feat.r_dc > th.r_kcl_dc_max;
    clr[KCL_DC] = feat.r_dc < th.clear_ratio * th.r_kcl_dc_max;
    flagged[KCL_W1] = feat.r_w1 > th.r_kcl_w1_max;
    clr[KCL_W1] = feat.r_w1 < th.clear_ratio * th.r_kcl_w1_max;
    float lo_c = 1.0f - th.clear_ratio * (1.0f - th.band_lo), hi_c = 1.0f + th.clear_ratio * (th.band_hi - 1.0f);
    bool out = false, inside = true;
    for (uint8_t k = 0; k < 3; k++) {
      if (feat.ar[k] < th.band_lo || feat.ar[k] > th.band_hi) out = true;
      if (feat.ar[k] < lo_c || feat.ar[k] > hi_c) inside = false;
    }
    flagged[AMPLITUDE_RATIO] = out; clr[AMPLITUDE_RATIO] = inside;
    flagged[PHASE_LOSS] = phase_loss(feat.rms, th.low_rms_frac, th.min_i_s_A);
    clr[PHASE_LOSS] = !phase_loss(feat.rms, th.low_rms_frac / th.clear_ratio, th.min_i_s_A);
    if (have_prev && prev_i_s > 0.0f && !holding) {
      float change = fabsf(cycle_i_s - prev_i_s) / prev_i_s;
      flagged[TRANSIENT] = change > th.i_s_step_frac;
      clr[TRANSIENT] = change < th.clear_ratio * th.i_s_step_frac;
    }
  }
  if (stopped || low_current) { have_prev = false; } else { have_prev = true; prev_i_s = cycle_i_s; }

  // Features of this cycle.
  Serial.print("F "); Serial.print(cycle); Serial.print(feat.ready ? " 1 " : " 0 ");
  Serial.print(feat.ready ? feat.i_s : 0.0f, 5);
  for (uint8_t k = 0; k < 3; k++) { Serial.print(' '); Serial.print(feat.ready ? feat.mu[k] : 0.0f, 5); }
  Serial.print(' '); Serial.print(feat.ready ? feat.r_dc : 0.0f, 5);
  Serial.print(' '); Serial.print(feat.ready ? feat.r_w1 : 0.0f, 5);
  for (uint8_t k = 0; k < 3; k++) { Serial.print(' '); Serial.print(feat.ready ? feat.ar[k] : 1.0f, 5); }
  for (uint8_t k = 0; k < 3; k++) { Serial.print(' '); Serial.print(feat.ready ? feat.rms[k] : 0.0f, 5); }
  Serial.println();

  if (feat.ready && !holding && !stopped) {
    for (uint8_t k = 0; k < N_CHECKS; k++) {
      if (flagged[k] && !active[k]) {
        Serial.print("E "); Serial.print(cycle); Serial.print(' '); Serial.print(CHECK_NAME[k]);
        Serial.print(' '); Serial.println(CHECK_SEV[k]);
        active[k] = true; clear_streak[k] = 0;
        if (k != TRANSIENT) {                // medium or high: the board shows the phase and its direction
          uint8_t p = 0;
          float best = 0.0f;
          for (uint8_t j = 0; j < 3; j++) if (fabsf(feat.mu[j]) > best) { best = fabsf(feat.mu[j]); p = j; }
          bool down = feat.mu[p] < 0.0f;
          if (best < 0.05f) {                // no mean current (an open leg, a gain error): the odd amplitude
            float dev = 0.0f;
            for (uint8_t j = 0; j < 3; j++) if (fabsf(feat.ar[j] - 1.0f) > dev) { dev = fabsf(feat.ar[j] - 1.0f); p = j; }
            down = feat.ar[p] < 1.0f;
          }
          show_fault(p, down);
        }
      } else if (active[k] && !flagged[k] && clr[k]) {
        if (++clear_streak[k] >= th.clear_cycles) active[k] = false;
      } else if (active[k]) {
        clear_streak[k] = 0;
      }
    }
  }
  if (hold > 0) hold--;
}

// ---------------------------------------------------------------- 12x8 LED matrix
ArduinoLEDMatrix matrix;
uint8_t frame_px[8][12];

// 5x7 glyphs, one byte per column, bit 0 = top row (the classic 5x7 font).
struct Glyph { char c; uint8_t col[5]; };
const Glyph FONT[] = {
  {' ', {0x00, 0x00, 0x00, 0x00, 0x00}}, {'-', {0x08, 0x08, 0x08, 0x08, 0x08}}, {'.', {0x00, 0x60, 0x60, 0x00, 0x00}},
  {'=', {0x14, 0x14, 0x14, 0x14, 0x14}}, {'?', {0x02, 0x01, 0x51, 0x09, 0x06}},
  {'0', {0x3E, 0x51, 0x49, 0x45, 0x3E}}, {'1', {0x00, 0x42, 0x7F, 0x40, 0x00}}, {'2', {0x42, 0x61, 0x51, 0x49, 0x46}},
  {'3', {0x21, 0x41, 0x45, 0x4B, 0x31}}, {'4', {0x18, 0x14, 0x12, 0x7F, 0x10}}, {'5', {0x27, 0x45, 0x45, 0x45, 0x39}},
  {'6', {0x3C, 0x4A, 0x49, 0x49, 0x30}}, {'7', {0x01, 0x71, 0x09, 0x05, 0x03}}, {'8', {0x36, 0x49, 0x49, 0x49, 0x36}},
  {'9', {0x06, 0x49, 0x49, 0x29, 0x1E}},
  {'A', {0x7E, 0x11, 0x11, 0x11, 0x7E}}, {'B', {0x7F, 0x49, 0x49, 0x49, 0x36}}, {'C', {0x3E, 0x41, 0x41, 0x41, 0x22}},
  {'D', {0x7F, 0x41, 0x41, 0x22, 0x1C}}, {'E', {0x7F, 0x49, 0x49, 0x49, 0x41}}, {'F', {0x7F, 0x09, 0x09, 0x09, 0x01}},
  {'G', {0x3E, 0x41, 0x49, 0x49, 0x7A}}, {'H', {0x7F, 0x08, 0x08, 0x08, 0x7F}}, {'I', {0x00, 0x41, 0x7F, 0x41, 0x00}},
  {'J', {0x20, 0x40, 0x41, 0x3F, 0x01}}, {'K', {0x7F, 0x08, 0x14, 0x22, 0x41}}, {'L', {0x7F, 0x40, 0x40, 0x40, 0x40}},
  {'M', {0x7F, 0x02, 0x0C, 0x02, 0x7F}}, {'N', {0x7F, 0x04, 0x08, 0x10, 0x7F}}, {'O', {0x3E, 0x41, 0x41, 0x41, 0x3E}},
  {'P', {0x7F, 0x09, 0x09, 0x09, 0x06}}, {'Q', {0x3E, 0x41, 0x51, 0x21, 0x5E}}, {'R', {0x7F, 0x09, 0x19, 0x29, 0x46}},
  {'S', {0x46, 0x49, 0x49, 0x49, 0x31}}, {'T', {0x01, 0x01, 0x7F, 0x01, 0x01}}, {'U', {0x3F, 0x40, 0x40, 0x40, 0x3F}},
  {'V', {0x1F, 0x20, 0x40, 0x20, 0x1F}}, {'W', {0x3F, 0x40, 0x38, 0x40, 0x3F}}, {'X', {0x63, 0x14, 0x08, 0x14, 0x63}},
  {'Y', {0x07, 0x08, 0x70, 0x08, 0x07}}, {'Z', {0x61, 0x51, 0x49, 0x45, 0x43}},
};
const uint8_t ARROW_DOWN[5] = {0x08, 0x10, 0x3F, 0x10, 0x08};
const uint8_t ARROW_UP[5] = {0x04, 0x02, 0x3F, 0x02, 0x04};
uint8_t CHECK_MARK[8][12] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
  {0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
};

static const uint8_t *glyph(char c) {
  for (const Glyph &g : FONT) if (g.c == c) return g.col;
  return FONT[0].col;
}

static void clear_px() { memset(frame_px, 0, sizeof(frame_px)); }

static void draw_cols(const uint8_t *cols, int x0) {    // 5 columns at column x0, rows 0..6
  for (uint8_t i = 0; i < 5; i++) {
    int x = x0 + i;
    if (x < 0 || x > 11) continue;
    for (uint8_t r = 0; r < 7; r++) frame_px[r][x] = (cols[i] >> r) & 1;
  }
}

static void render() { matrix.renderBitmap(frame_px, 8, 12); }

// Scrolling text, advanced from loop() one column at a time (never blocking the serial input).
char scroll_text[64];
uint16_t scroll_len = 0, scroll_pos = 0;
bool scrolling = false;
unsigned long scroll_t = 0;
const unsigned long SCROLL_MS = 70;

// While the agent investigates, the matrix alternates '?' with the detected phase and its arrow.
bool investigating = false, have_fault = false, fault_down = false, blink_on = false;
uint8_t fault_phase = 0;
unsigned long blink_t = 0;
const unsigned long BLINK_MS = 700;

static void draw_question() { clear_px(); draw_cols(glyph('?'), 3); render(); }
static void draw_fault() {
  clear_px();
  draw_cols(glyph('A' + fault_phase), 0);
  draw_cols(fault_down ? ARROW_DOWN : ARROW_UP, 7);
  render();
}
static void show_check() {
  scrolling = investigating = have_fault = false;
  memcpy(frame_px, CHECK_MARK, sizeof(frame_px)); render();
}
static void show_question() {
  scrolling = false; investigating = true; blink_on = false; blink_t = millis();
  draw_question();
}
static void show_fault(uint8_t phase, bool down) {
  fault_phase = phase; fault_down = down; have_fault = true;
  if (investigating) return;                  // blink_step() shows it next to the '?'
  scrolling = false;
  draw_fault();
}
static void blink_step() {
  if (!investigating || !have_fault || millis() - blink_t < BLINK_MS) return;
  blink_t = millis();
  blink_on = !blink_on;
  if (blink_on) draw_fault(); else draw_question();
}
static void show_scroll(const char *text, uint16_t len) {
  investigating = false;
  if (len > sizeof(scroll_text) - 1) len = sizeof(scroll_text) - 1;
  memcpy(scroll_text, text, len);
  scroll_len = len; scroll_pos = 0; scrolling = true; scroll_t = 0;
}
static void scroll_step() {
  if (!scrolling || millis() - scroll_t < SCROLL_MS) return;
  scroll_t = millis();
  uint16_t width = 12 + scroll_len * 6;       // enter from the right, leave on the left, then repeat
  clear_px();
  int x0 = 12 - (int)scroll_pos;
  for (uint16_t i = 0; i < scroll_len; i++) draw_cols(glyph(scroll_text[i]), x0 + 6 * (int)i);
  render();
  scroll_pos = (scroll_pos + 1) % width;
}

// ---------------------------------------------------------------- frames
static void handle_frame(uint8_t type, const uint8_t *p, uint16_t len) {
  if (type == 'C' && len >= 8) {
    uint16_t n = p[7];
    if (n == 0 || n > 100 || len != 8 + 6 * n) { Serial.println("ERR bad cycle frame"); return; }
    process_cycle(rd32(p), rd16(p + 4) / 1000.0f, p[6], p + 8, n);
  } else if (type == 'K' && len == 44) {      // struct "<fBffffffBffBBf" in faulttwin/edge.py
    th.mu_abs_max = rdf(p); th.mu_consecutive_cycles = p[4]; th.r_kcl_dc_max = rdf(p + 5); th.r_kcl_w1_max = rdf(p + 9);
    th.band_lo = rdf(p + 13); th.band_hi = rdf(p + 17); th.low_rms_frac = rdf(p + 21); th.i_s_step_frac = rdf(p + 25);
    th.warmup_cycles = p[29]; th.min_i_s_A = rdf(p + 30); th.clear_ratio = rdf(p + 34); th.clear_cycles = p[38];
    th.window_cycles = p[39] == 0 ? 1 : (p[39] > MAX_WINDOW ? MAX_WINDOW : p[39]); th.phase_offset = rdf(p + 40);
    monitor_reset();
  } else if (type == 'D' && len >= 1) {
    if (p[0] == 'H') show_check();
    else if (p[0] == 'I') show_question();
    else if (p[0] == 'T') show_scroll((const char *)p + 1, len - 1);
  } else {
    Serial.println("ERR unknown frame");
  }
}

static void read_serial() {
  while (Serial.available() > 0) {
    uint8_t b = (uint8_t)Serial.read();
    switch (rx_state) {
      case SYNC0: if (b == 0xA5) rx_state = SYNC1; break;
      case SYNC1: rx_state = (b == 0x5A) ? HEADER : (b == 0xA5 ? SYNC1 : SYNC0); rx_pos = 0; break;
      case HEADER:
        rx_buf[rx_pos++] = b;
        if (rx_pos == 3) {
          uint16_t len = rd16(rx_buf + 1);
          if (len > MAX_PAYLOAD) { Serial.println("ERR frame too long"); rx_state = SYNC0; break; }
          rx_need = 3 + len + 2;
          rx_state = BODY;
        }
        break;
      case BODY:
        rx_buf[rx_pos++] = b;
        if (rx_pos == rx_need) {
          uint16_t body = rx_need - 2;
          if (fletcher16(rx_buf, body) == rd16(rx_buf + body)) handle_frame(rx_buf[0], rx_buf + 3, body - 3);
          else Serial.println("ERR checksum");
          rx_state = SYNC0;
        }
        break;
    }
  }
}

// ---------------------------------------------------------------- setup / loop
// D2 counts as pressed once its level has held LOW for DEBOUNCE_MS (a hand-held jumper wire chatters), and a
// second press within PRESS_GAP_MS is ignored.
const unsigned long DEBOUNCE_MS = 40, PRESS_GAP_MS = 2000;
bool button_down = false, button_raw = false;
unsigned long raw_change = 0, last_press = 0;

void setup() {
  Serial.begin(EDGE_BAUD);
  pinMode(MYSTERY_PIN, INPUT_PULLUP);
  matrix.begin();
  monitor_reset();
  show_check();
}

void loop() {
  read_serial();
  scroll_step();
  blink_step();
  bool raw = digitalRead(MYSTERY_PIN) == LOW;       // a button or a jumper wire to GND
  if (raw != button_raw) { button_raw = raw; raw_change = millis(); }
  if (button_raw != button_down && millis() - raw_change >= DEBOUNCE_MS) {
    button_down = button_raw;
    if (button_down && millis() - last_press >= PRESS_GAP_MS) { last_press = millis(); Serial.println("M"); }
  }
  static unsigned long hb = 0;
  if (millis() - hb >= 1000) {
    hb = millis();
    Serial.print("HB FAULTTWIN-EDGE "); Serial.print(FW_VERSION); Serial.print(" UNO-R4-WIFI "); Serial.println(hb);
  }
}
