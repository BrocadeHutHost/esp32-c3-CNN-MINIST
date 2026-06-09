#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
#include <cmath>
#include "esp_task_wdt.h"
const char* ssid = "*****";
const char* password = "*****";

const int WEB_PORT = 80;
const int IMG_H = 28, IMG_W = 28;
const int IMG_SIZE = IMG_H * IMG_W;
const int PACKED_SIZE = (IMG_SIZE + 7) / 8;  
const int MAX_SAMPLES = 300;                 

int g_train_epochs = 100;
float g_learning_rate = 0.01f;
const float LR_DECAY = 0.95f;       
const int LR_DECAY_STEP = 10;       

WebServer server(WEB_PORT);
uint8_t train_images[MAX_SAMPLES][PACKED_SIZE];
int train_labels[MAX_SAMPLES];
int sample_count = 0;

void pack_image(float img[IMG_H][IMG_W], uint8_t* packed) {
  memset(packed, 0, PACKED_SIZE);
  for (int y = 0; y < IMG_H; y++)
    for (int x = 0; x < IMG_W; x++) {
      int idx = y * IMG_W + x;
      if (img[y][x] >= 0.5f)
        packed[idx / 8] |= (1 << (idx % 8));
    }
}

void unpack_image(const uint8_t* packed, float img[IMG_H][IMG_W]) {
  for (int y = 0; y < IMG_H; y++)
    for (int x = 0; x < IMG_W; x++) {
      int idx = y * IMG_W + x;
      img[y][x] = (packed[idx / 8] >> (idx % 8)) & 1 ? 1.0f : 0.0f;
    }
}

#define CONV1_FILTERS 6
#define CONV2_FILTERS 12
#define CONV_KERNEL 3
#define FC1_UNITS 16
#define FC1_IN (CONV2_FILTERS * (IMG_H/4) * (IMG_W/4))  

float conv1_w[CONV1_FILTERS][1][CONV_KERNEL][CONV_KERNEL];
float conv1_b[CONV1_FILTERS];
float conv2_w[CONV2_FILTERS][CONV1_FILTERS][CONV_KERNEL][CONV_KERNEL];
float conv2_b[CONV2_FILTERS];
float fc1_w[FC1_UNITS][FC1_IN];
float fc1_b[FC1_UNITS];
float fc2_w[10][FC1_UNITS];
float fc2_b[10];

struct ForwardCache {
  float conv1_out[CONV1_FILTERS][IMG_H][IMG_W];
  float pool1_out[CONV1_FILTERS][IMG_H/2][IMG_W/2];
  float conv2_out[CONV2_FILTERS][IMG_H/2][IMG_W/2];
  float pool2_out[CONV2_FILTERS][IMG_H/4][IMG_W/4];
  float fc_in[FC1_IN];
  float fc1_out[FC1_UNITS];
  float logits[10];
} fcache;

struct GradCache {
  float grad_logits[10];
  float grad_fc1_out[FC1_UNITS];
  float grad_fc_in[FC1_IN];
  float grad_pool2[CONV2_FILTERS][IMG_H/4][IMG_W/4];
  float grad_conv2[CONV2_FILTERS][IMG_H/2][IMG_W/2];
  float grad_pool1[CONV1_FILTERS][IMG_H/2][IMG_W/2];
  float grad_conv1[CONV1_FILTERS][IMG_H][IMG_W];
} gcache;

enum AsyncState { ASYNC_IDLE, ASYNC_RUNNING, ASYNC_DONE };
AsyncState g_async_state = ASYNC_IDLE;
int g_async_epoch = 0;            
int g_async_total_epochs = 0;
int g_async_sample_idx = 0;
float g_async_loss_sum = 0;
float g_async_recent_loss[10];    
int g_async_loss_count = 0;
String g_async_message = "";      

float relu(float x) { return x > 0 ? x : 0; }
float relu_deriv(float x) { return x > 0 ? 1.0f : 0.0f; }

void softmax(float* x, int len) {
  float maxv = x[0];
  for (int i = 1; i < len; i++) if (x[i] > maxv) maxv = x[i];
  for (int i = 0; i < len; i++) {
    float t = x[i] - maxv;
    if (t > 20.0f) t = 20.0f; else if (t < -20.0f) t = -20.0f;
    x[i] = exp(t);
  }
  float sum = 0;
  for (int i = 0; i < len; i++) sum += x[i];
  for (int i = 0; i < len; i++) x[i] /= sum;
}

float cross_entropy(float* probs, int label) {
  float p = probs[label];
  if (p < 1e-7f) p = 1e-7f;
  return -log(p);
}

bool all_weights_valid() {
  float* ptrs[] = { (float*)conv1_w, (float*)conv1_b, (float*)conv2_w,
                    (float*)conv2_b, (float*)fc1_w, (float*)fc1_b,
                    (float*)fc2_w, (float*)fc2_b };
  size_t sizes[] = { sizeof(conv1_w), sizeof(conv1_b), sizeof(conv2_w),
                     sizeof(conv2_b), sizeof(fc1_w), sizeof(fc1_b),
                     sizeof(fc2_w), sizeof(fc2_b) };
  for (int k = 0; k < 8; k++) {
    float* p = ptrs[k];
    for (size_t i = 0; i < sizes[k] / 4; i++) {
      if (isnan(p[i]) || isinf(p[i])) return false;
    }
  }
  return true;
}

void init_weights() {
  float std1 = sqrt(2.0f / (1 * CONV_KERNEL * CONV_KERNEL));      
  float std2 = sqrt(2.0f / (CONV1_FILTERS * CONV_KERNEL * CONV_KERNEL));
  float std_fc1 = sqrt(2.0f / (float)FC1_IN);
  float std_fc2 = sqrt(2.0f / (float)FC1_UNITS);

  for (int f = 0; f < CONV1_FILTERS; f++) {
    for (int y = 0; y < CONV_KERNEL; y++)
      for (int x = 0; x < CONV_KERNEL; x++)
        conv1_w[f][0][y][x] = (random(200) - 100) / 100.0f * std1;
    conv1_b[f] = 0.0f;
  }
  for (int f = 0; f < CONV2_FILTERS; f++) {
    for (int c = 0; c < CONV1_FILTERS; c++)
      for (int y = 0; y < CONV_KERNEL; y++)
        for (int x = 0; x < CONV_KERNEL; x++)
          conv2_w[f][c][y][x] = (random(200) - 100) / 100.0f * std2;
    conv2_b[f] = 0.0f;
  }
  for (int i = 0; i < FC1_UNITS; i++) {
    for (int j = 0; j < FC1_IN; j++)
      fc1_w[i][j] = (random(200) - 100) / 100.0f * std_fc1;
    fc1_b[i] = 0.0f;
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < FC1_UNITS; j++)
      fc2_w[i][j] = (random(200) - 100) / 100.0f * std_fc2;
    fc2_b[i] = 0.0f;
  }
}

template<int H, int W>
void convolution(float input[H][W], float kernel[3][3], float output[H][W], int pad) {
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float sum = 0;
      for (int ky = 0; ky < 3; ky++) {
        int iy = y + ky - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < 3; kx++) {
          int ix = x + kx - pad;
          if (ix < 0 || ix >= W) continue;
          sum += input[iy][ix] * kernel[ky][kx];
        }
      }
      output[y][x] = sum;
    }
  }
}

template<int H, int W>
void maxpool2x2(float input[H][W], float output[H/2][W/2]) {
  for (int y = 0; y < H / 2; y++) {
    for (int x = 0; x < W / 2; x++) {
      float m = input[y*2][x*2];
      if (input[y*2][x*2+1] > m) m = input[y*2][x*2+1];
      if (input[y*2+1][x*2] > m) m = input[y*2+1][x*2];
      if (input[y*2+1][x*2+1] > m) m = input[y*2+1][x*2+1];
      output[y][x] = m;
    }
  }
}

void forward(float img[IMG_H][IMG_W]) {
  for (int f = 0; f < CONV1_FILTERS; f++) {
    convolution<IMG_H, IMG_W>(img, conv1_w[f][0], fcache.conv1_out[f], 1);
    for (int i = 0; i < IMG_SIZE; i++)
      ((float*)fcache.conv1_out[f])[i] = relu(((float*)fcache.conv1_out[f])[i] + conv1_b[f]);
    maxpool2x2<IMG_H, IMG_W>(fcache.conv1_out[f], fcache.pool1_out[f]);
  }

  for (int f = 0; f < CONV2_FILTERS; f++) {
    for (int y = 0; y < IMG_H/2; y++)
      for (int x = 0; x < IMG_W/2; x++)
        fcache.conv2_out[f][y][x] = conv2_b[f];
    for (int c = 0; c < CONV1_FILTERS; c++) {
      float temp[IMG_H/2][IMG_W/2];
      convolution<IMG_H/2, IMG_W/2>(fcache.pool1_out[c], conv2_w[f][c], temp, 1);
      for (int y = 0; y < IMG_H/2; y++)
        for (int x = 0; x < IMG_W/2; x++)
          fcache.conv2_out[f][y][x] += temp[y][x];
    }
    for (int i = 0; i < (IMG_H/2)*(IMG_W/2); i++)
      ((float*)fcache.conv2_out[f])[i] = relu(((float*)fcache.conv2_out[f])[i]);
    maxpool2x2<IMG_H/2, IMG_W/2>(fcache.conv2_out[f], fcache.pool2_out[f]);
  }

  for (int f = 0; f < CONV2_FILTERS; f++)
    memcpy(&fcache.fc_in[f * (IMG_H/4) * (IMG_W/4)], fcache.pool2_out[f],
           sizeof(float) * (IMG_H/4) * (IMG_W/4));

  for (int i = 0; i < FC1_UNITS; i++) {
    float sum = fc1_b[i];
    for (int j = 0; j < FC1_IN; j++) sum += fcache.fc_in[j] * fc1_w[i][j];
    fcache.fc1_out[i] = relu(sum);
  }

  for (int i = 0; i < 10; i++) {
    float sum = fc2_b[i];
    for (int j = 0; j < FC1_UNITS; j++) sum += fcache.fc1_out[j] * fc2_w[i][j];
    fcache.logits[i] = sum;
  }
}

int predict(float img[IMG_H][IMG_W]) {
  forward(img);
  float probs[10];
  memcpy(probs, fcache.logits, sizeof(probs));
  softmax(probs, 10);
  int maxi = 0;
  for (int i = 1; i < 10; i++) if (probs[i] > probs[maxi]) maxi = i;
  return maxi;
}

void train_step(float img[IMG_H][IMG_W], int label, float lr) {
  forward(img);

  float probs[10];
  memcpy(probs, fcache.logits, sizeof(probs));
  softmax(probs, 10);
  for (int i = 0; i < 10; i++) {
    gcache.grad_logits[i] = probs[i] - (i == label ? 1.0f : 0.0f);
    if (gcache.grad_logits[i] > 1.0f) gcache.grad_logits[i] = 1.0f;
    else if (gcache.grad_logits[i] < -1.0f) gcache.grad_logits[i] = -1.0f;
  }

  memset(gcache.grad_fc1_out, 0, sizeof(gcache.grad_fc1_out));
  for (int i = 0; i < 10; i++)
    for (int j = 0; j < FC1_UNITS; j++)
      gcache.grad_fc1_out[j] += gcache.grad_logits[i] * fc2_w[i][j];

  for (int j = 0; j < FC1_UNITS; j++)
    gcache.grad_fc1_out[j] *= relu_deriv(fcache.fc1_out[j]);

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < FC1_UNITS; j++)
      fc2_w[i][j] -= lr * gcache.grad_logits[i] * fcache.fc1_out[j];
    fc2_b[i] -= lr * gcache.grad_logits[i];
  }

  memset(gcache.grad_fc_in, 0, sizeof(gcache.grad_fc_in));
  for (int i = 0; i < FC1_UNITS; i++)
    for (int j = 0; j < FC1_IN; j++)
      gcache.grad_fc_in[j] += gcache.grad_fc1_out[i] * fc1_w[i][j];

  for (int i = 0; i < FC1_UNITS; i++) {
    for (int j = 0; j < FC1_IN; j++)
      fc1_w[i][j] -= lr * gcache.grad_fc1_out[i] * fcache.fc_in[j];
    fc1_b[i] -= lr * gcache.grad_fc1_out[i];
  }

  for (int f = 0; f < CONV2_FILTERS; f++)
    memcpy(gcache.grad_pool2[f], &gcache.grad_fc_in[f * (IMG_H/4) * (IMG_W/4)],
           sizeof(float) * (IMG_H/4) * (IMG_W/4));

  memset(gcache.grad_conv2, 0, sizeof(gcache.grad_conv2));
  for (int f = 0; f < CONV2_FILTERS; f++) {
    for (int y = 0; y < IMG_H/4; y++) {
      for (int x = 0; x < IMG_W/4; x++) {
        float m = fcache.conv2_out[f][y*2][x*2];
        int my = 0, mx = 0;
        if (fcache.conv2_out[f][y*2][x*2+1] > m)
          { m = fcache.conv2_out[f][y*2][x*2+1]; my = 0; mx = 1; }
        if (fcache.conv2_out[f][y*2+1][x*2] > m)
          { m = fcache.conv2_out[f][y*2+1][x*2]; my = 1; mx = 0; }
        if (fcache.conv2_out[f][y*2+1][x*2+1] > m)
          { my = 1; mx = 1; }
        gcache.grad_conv2[f][y*2+my][x*2+mx] =
          gcache.grad_pool2[f][y][x] *
          relu_deriv(fcache.conv2_out[f][y*2+my][x*2+mx]);
      }
    }
  }

  for (int f = 0; f < CONV2_FILTERS; f++) {
    for (int c = 0; c < CONV1_FILTERS; c++) {
      for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
          float dw = 0;
          for (int y = 0; y < IMG_H/2; y++) {
            int iy = y + ky - 1;
            if (iy < 0 || iy >= IMG_H/2) continue;
            for (int x = 0; x < IMG_W/2; x++) {
              int ix = x + kx - 1;
              if (ix < 0 || ix >= IMG_W/2) continue;
              dw += gcache.grad_conv2[f][y][x] * fcache.pool1_out[c][iy][ix];
            }
          }
          conv2_w[f][c][ky][kx] -= lr * dw;
        }
      }
    }
    float db = 0;
    for (int i = 0; i < (IMG_H/2)*(IMG_W/2); i++)
      db += ((float*)gcache.grad_conv2[f])[i];
    conv2_b[f] -= lr * db;
  }

  memset(gcache.grad_pool1, 0, sizeof(gcache.grad_pool1));
  for (int c = 0; c < CONV1_FILTERS; c++) {
    for (int f = 0; f < CONV2_FILTERS; f++) {
      for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
          for (int y = 0; y < IMG_H/2; y++) {
            int iy = y - ky + 1;
            if (iy < 0 || iy >= IMG_H/2) continue;
            for (int x = 0; x < IMG_W/2; x++) {
              int ix = x - kx + 1;
              if (ix < 0 || ix >= IMG_W/2) continue;
              gcache.grad_pool1[c][iy][ix] +=
                gcache.grad_conv2[f][y][x] * conv2_w[f][c][ky][kx];
            }
          }
        }
      }
    }
  }

  memset(gcache.grad_conv1, 0, sizeof(gcache.grad_conv1));
  for (int f = 0; f < CONV1_FILTERS; f++) {
    for (int y = 0; y < IMG_H/2; y++) {
      for (int x = 0; x < IMG_W/2; x++) {
        float m = fcache.conv1_out[f][y*2][x*2];
        int my = 0, mx = 0;
        if (fcache.conv1_out[f][y*2][x*2+1] > m)
          { m = fcache.conv1_out[f][y*2][x*2+1]; my = 0; mx = 1; }
        if (fcache.conv1_out[f][y*2+1][x*2] > m)
          { m = fcache.conv1_out[f][y*2+1][x*2]; my = 1; mx = 0; }
        if (fcache.conv1_out[f][y*2+1][x*2+1] > m)
          { my = 1; mx = 1; }
        gcache.grad_conv1[f][y*2+my][x*2+mx] =
          gcache.grad_pool1[f][y][x] *
          relu_deriv(fcache.conv1_out[f][y*2+my][x*2+mx]);
      }
    }
  }

  for (int f = 0; f < CONV1_FILTERS; f++) {
    for (int ky = 0; ky < 3; ky++) {
      for (int kx = 0; kx < 3; kx++) {
        float dw = 0;
        for (int y = 0; y < IMG_H; y++) {
          int iy = y + ky - 1;
          if (iy < 0 || iy >= IMG_H) continue;
          for (int x = 0; x < IMG_W; x++) {
            int ix = x + kx - 1;
            if (ix < 0 || ix >= IMG_W) continue;
            dw += gcache.grad_conv1[f][y][x] * img[iy][ix];
          }
        }
        conv1_w[f][0][ky][kx] -= lr * dw;
      }
    }
    float db = 0;
    for (int i = 0; i < IMG_SIZE; i++)
      db += ((float*)gcache.grad_conv1[f])[i];
    conv1_b[f] -= lr * db;
  }

  auto clip = [](float& v) {
    if (v > 2.0f) v = 2.0f;
    else if (v < -2.0f) v = -2.0f;
  };
  for (int f = 0; f < CONV1_FILTERS; f++) {
    clip(conv1_b[f]);
    for (int y = 0; y < 3; y++)
      for (int x = 0; x < 3; x++)
        clip(conv1_w[f][0][y][x]);
  }
  for (int f = 0; f < CONV2_FILTERS; f++) {
    clip(conv2_b[f]);
    for (int c = 0; c < CONV1_FILTERS; c++)
      for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++)
          clip(conv2_w[f][c][y][x]);
  }
  for (int i = 0; i < FC1_UNITS; i++) {
    clip(fc1_b[i]);
    for (int j = 0; j < FC1_IN; j++)
      clip(fc1_w[i][j]);
  }
  for (int i = 0; i < 10; i++) {
    clip(fc2_b[i]);
    for (int j = 0; j < FC1_UNITS; j++)
      clip(fc2_w[i][j]);
  }
}

void train_epochs(int epochs, float lr) {
  for (int e = 0; e < epochs; e++) {
    float total_loss = 0;
    float current_lr = lr;
    if (e > 0 && e % LR_DECAY_STEP == 0)
      current_lr = lr * powf(LR_DECAY, (float)e / LR_DECAY_STEP);

    for (int s = 0; s < sample_count; s++) {
      float img[IMG_H][IMG_W];
      unpack_image(train_images[s], img);
      forward(img);
      float probs[10];
      memcpy(probs, fcache.logits, sizeof(probs));
      softmax(probs, 10);
      total_loss += cross_entropy(probs, train_labels[s]);
      train_step(img, train_labels[s], current_lr);
      yield();
    }
    float avg = total_loss / sample_count;
    if (isnan(avg)) {
      Serial.println("训练出现 NaN，停止");
      break;
    }
    if ((e + 1) % 10 == 0 || e == 0 || e == epochs - 1)
      Serial.printf("  Epoch %d/%d | Loss: %.4f | LR: %.5f\n",
                    e + 1, epochs, avg, current_lr);
  }
}

void init_spiffs();
void load_weights();
void save_weights();
void load_samples();
void save_samples();

void start_async_train(int epochs) {
  if (sample_count == 0 || g_async_state == ASYNC_RUNNING) return;
  g_async_state = ASYNC_RUNNING;
  g_async_epoch = 0;
  g_async_total_epochs = epochs;
  g_async_sample_idx = 0;
  g_async_loss_sum = 0;
  g_async_loss_count = 0;
  memset(g_async_recent_loss, 0, sizeof(g_async_recent_loss));
  g_async_message = "";
  Serial.printf("异步训练启动: %d 轮\n", epochs);
}

void stop_async_train() {
  if (g_async_state == ASYNC_RUNNING) {
    g_async_state = ASYNC_IDLE;
    g_async_message = "训练已被用户取消";
    Serial.println("异步训练已取消");
  }
}

void async_train_tick() {
  if (g_async_state != ASYNC_RUNNING) return;
  for (int step = 0; step < 3; step++) {
    if (g_async_sample_idx >= sample_count) break;

    float img[IMG_H][IMG_W];
    unpack_image(train_images[g_async_sample_idx], img);
    train_step(img, train_labels[g_async_sample_idx], g_learning_rate);

    float probs[10];
    memcpy(probs, fcache.logits, sizeof(probs));
    softmax(probs, 10);
    g_async_loss_sum += cross_entropy(probs, train_labels[g_async_sample_idx]);

    g_async_sample_idx++;
    yield();
  }

  if (g_async_sample_idx >= sample_count) {
    float avg_loss = g_async_loss_sum / sample_count;
    Serial.printf("  [Async] Epoch %d/%d | Loss: %.4f\n",
                  g_async_epoch + 1, g_async_total_epochs, avg_loss);

    if (g_async_loss_count < 10) {
      g_async_recent_loss[g_async_loss_count++] = avg_loss;
    } else {
      for (int i = 0; i < 9; i++)
        g_async_recent_loss[i] = g_async_recent_loss[i + 1];
      g_async_recent_loss[9] = avg_loss;
    }

    g_async_epoch++;
    g_async_sample_idx = 0;
    g_async_loss_sum = 0;

    if (g_async_epoch % LR_DECAY_STEP == 0 && g_async_epoch > 0) {
      g_learning_rate *= LR_DECAY;
      Serial.printf("  学习率衰减至 %.5f\n", g_learning_rate);
    }

    if (g_async_epoch >= g_async_total_epochs) {
      g_async_state = ASYNC_DONE;
      if (all_weights_valid()) {
        save_weights();
        g_async_message = "训练完成！模型已保存 ("
                        + String(g_async_total_epochs) + "轮)";
      } else {
        load_weights();
        g_async_message = "训练异常，权重已回滚";
      }
      Serial.println("[Async] " + g_async_message);
    }
  }
}

void init_spiffs() {
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS 初始化失败，重启...");
    ESP.restart();
  }
  Serial.println("SPIFFS 就绪");
}

void load_weights() {
  File f = SPIFFS.open("/cnn_w.bin", "r");
  if (f) {
    size_t expected = sizeof(conv1_w) + sizeof(conv1_b) + sizeof(conv2_w)
                    + sizeof(conv2_b) + sizeof(fc1_w) + sizeof(fc1_b)
                    + sizeof(fc2_w) + sizeof(fc2_b);
    if (f.size() == expected) {
      f.readBytes((char*)conv1_w, sizeof(conv1_w));
      f.readBytes((char*)conv1_b, sizeof(conv1_b));
      f.readBytes((char*)conv2_w, sizeof(conv2_w));
      f.readBytes((char*)conv2_b, sizeof(conv2_b));
      f.readBytes((char*)fc1_w, sizeof(fc1_w));
      f.readBytes((char*)fc1_b, sizeof(fc1_b));
      f.readBytes((char*)fc2_w, sizeof(fc2_w));
      f.readBytes((char*)fc2_b, sizeof(fc2_b));
      if (all_weights_valid())
        Serial.println("权重加载成功");
      else {
        init_weights();
        Serial.println("权重损坏，已重新初始化");
      }
    } else {
      init_weights();
      Serial.println("权重文件大小不匹配，已重新初始化");
    }
    f.close();
  } else {
    init_weights();
    Serial.println("随机初始化权重（He初始化）");
  }
}

void save_weights() {
  File f = SPIFFS.open("/cnn_w.bin", "w");
  if (f) {
    f.write((uint8_t*)conv1_w, sizeof(conv1_w));
    f.write((uint8_t*)conv1_b, sizeof(conv1_b));
    f.write((uint8_t*)conv2_w, sizeof(conv2_w));
    f.write((uint8_t*)conv2_b, sizeof(conv2_b));
    f.write((uint8_t*)fc1_w, sizeof(fc1_w));
    f.write((uint8_t*)fc1_b, sizeof(fc1_b));
    f.write((uint8_t*)fc2_w, sizeof(fc2_w));
    f.write((uint8_t*)fc2_b, sizeof(fc2_b));
    f.close();
    Serial.println("权重已保存");
  }
}

void load_samples() {
  File f = SPIFFS.open("/samples.bin", "r");
  if (!f || f.size() < 2) { sample_count = 0; if(f) f.close(); return; }
  uint8_t hdr[2];
  f.readBytes((char*)hdr, 2);
  sample_count = hdr[0] | (hdr[1] << 8);
  if (sample_count < 0 || sample_count > MAX_SAMPLES) {
    sample_count = 0; f.close(); return;
  }
  for (int i = 0; i < sample_count; i++) {
    f.readBytes((char*)&train_labels[i], 1);
    f.readBytes((char*)train_images[i], PACKED_SIZE);
  }
  f.close();
  Serial.printf("加载 %d 个样本 (位打包, %d字节/张)\n", sample_count, PACKED_SIZE);
}

void save_samples() {
  File f = SPIFFS.open("/samples.bin", "w");
  if (f) {
    uint8_t hdr[2] = { (uint8_t)(sample_count & 0xFF), (uint8_t)(sample_count >> 8) };
    f.write(hdr, 2);
    for (int i = 0; i < sample_count; i++) {
      uint8_t lbl = (uint8_t)train_labels[i];
      f.write(&lbl, 1);
      f.write(train_images[i], PACKED_SIZE);
    }
    f.close();
    Serial.printf("样本已保存 (%d张)\n", sample_count);
  }
}

String get_sample_stats() {
  int counts[10] = {0};
  for (int i = 0; i < sample_count; i++)
    if (train_labels[i] >= 0 && train_labels[i] < 10)
      counts[train_labels[i]]++;

  String s = "[";
  for (int i = 0; i < 10; i++) {
    if (i > 0) s += ",";
    s += String(counts[i]);
  }
  s += "]";
  return s;
}

static const char PROGMEM webpage_content[] = R"raw(<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1.0,user-scalable=no'>
<title>CNN 28x28 手写数字识别</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',Arial,sans-serif;background:#f5f5f5;color:#222;
       max-width:500px;margin:0 auto;padding:15px;text-align:center}
  h2{font-size:1.3em;color:#111;margin-bottom:8px}
  .card{background:#fff;border:1px solid #ccc;border-radius:8px;padding:15px;margin:10px 0}
  canvas{border:2px solid #888;border-radius:4px;background:#fff;
         touch-action:none;cursor:crosshair;max-width:100%;height:auto}
  input[type=number]{padding:8px 12px;font-size:16px;width:70px;border:1px solid #888;
                     border-radius:4px;text-align:center;margin:0 5px}
  button{padding:9px 16px;font-size:14px;border:1px solid #555;border-radius:4px;cursor:pointer;
         font-weight:600;margin:3px;transition:all 0.15s;background:#555;color:#fff}
  button:active{transform:scale(0.96)}
  .btn-clr{background:#777;color:#fff;border-color:#666}
  .btn-clrS{background:#555;color:#fff;border-color:#444}
  .btn-train{background:#444;color:#fff;border-color:#333}
  .btn-trainAll{background:#222;color:#fff;border-color:#111}
  .btn-detect{background:#888;color:#fff;border-color:#777}
  .btn-stop{background:#555;color:#fff;border-color:#444}
  .btn-small{padding:5px 10px;font-size:12px}
  #res{font-size:18px;color:#111;min-height:28px;margin:8px 0;font-weight:bold}
  #status-line{font-size:13px;color:#555;margin:5px 0}
  #temp-line{font-size:12px;color:#777;margin:3px 0}
  .progress-bg{background:#ddd;border-radius:8px;height:18px;margin:8px 0;overflow:hidden}
  .progress-fg{background:#333;height:100%;border-radius:8px;
               transition:width 0.3s;width:0%}
  .sample-grid{display:flex;flex-wrap:wrap;gap:4px;justify-content:center;margin:5px 0}
  .sample-chip{background:#e0e0e0;color:#222;border-radius:8px;padding:3px 10px;font-size:12px;
               font-weight:600;min-width:36px;text-align:center}
  .sample-chip.zero{background:#f0f0f0;color:#888}
  label{font-weight:600;font-size:14px}
  .hint{font-size:11px;color:#888;margin-top:3px}
</style>
</head>
<body>

<div class='card'>
  <h2>CNN 手写数字识别</h2>
  <p style='font-size:13px;color:#777'>在画板上绘制数字 (0-9)</p>
</div>

<div class='card'>
  <label>数字标签: <input type='number' id='label' min='0' max='9' value='0'></label>
  <canvas id='c' width='280' height='280'></canvas>
  <div style='margin-top:8px'>
    <button class='btn-clr' id='clr'>清空画板</button>
    <button class='btn-clrS' id='clrS'>清除所有样本</button>
  </div>
</div>

<div class='card'>
  <button class='btn-train' id='trainBtn'>添加样本 + 训练5轮</button>
  <button class='btn-trainAll' id='trainAll'>全量训练</button>
  <button class='btn-detect' id='detect'>仅检测</button>
  <button class='btn-train' id='addSampleBtn'>仅添加样本(不训练)</button>
  <div class='hint'>快速训练约需10-30秒 | 全量训练在后台运行</div>
</div>

<div class='card' id='progress-card' style='display:none'>
  <div id='res'></div>
  <div id='probBars' style='margin:4px 0'></div>
  <div class='progress-bg'><div class='progress-fg' id='pbar'></div></div>
  <div id='status-line'></div>
  <div id='temp-line'></div>
  <button class='btn-stop btn-small' id='stopBtn' style='display:none'>停止训练</button>
</div>

<div class='card'>
  <div style='font-size:13px;font-weight:600;color:#555'>样本分布</div>
  <div class='sample-grid' id='sampleDist'>加载中...</div>
  <div id='res-static'></div>
</div>

<div class='card'>
  <div style='font-size:13px;font-weight:600;color:#555;margin-bottom:8px'>样本预览 (点击可删除)</div>
  <div id='sampleList' style='display:flex;flex-wrap:wrap;gap:8px;justify-content:center;min-height:40px;align-items:center;color:#80868b;font-size:13px'>暂无样本</div>
  <div class='hint'>提示：每个样本下方显示标签，悬停可预览大图</div>
</div>

<script>
const c=document.getElementById('c'),ctx=c.getContext('2d');
ctx.lineWidth=14; ctx.lineCap='round'; ctx.strokeStyle='black';
ctx.fillStyle='white'; ctx.fillRect(0,0,c.width,c.height);
let drawing=false;

function pos(e){
  let r=c.getBoundingClientRect(), sx=c.width/r.width, sy=c.height/r.height;
  let cx,cy;
  if(e.touches){ cx=e.touches[0].clientX; cy=e.touches[0].clientY; }
  else { cx=e.clientX; cy=e.clientY; }
  let x=(cx-r.left)*sx, y=(cy-r.top)*sy;
  return {x:Math.min(Math.max(0,x),c.width), y:Math.min(Math.max(0,y),c.height)};
}
function start(e){ drawing=true; let p=pos(e); ctx.beginPath(); ctx.moveTo(p.x,p.y); e.preventDefault(); }
function move(e){ if(!drawing)return; let p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x,p.y); e.preventDefault(); }
function end(e){ drawing=false; ctx.beginPath(); e.preventDefault(); }
c.addEventListener('mousedown',start); c.addEventListener('mousemove',move); c.addEventListener('mouseup',end);
c.addEventListener('touchstart',start); c.addEventListener('touchmove',move); c.addEventListener('touchend',end);

document.getElementById('clr').onclick=()=>{ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle='black';};

document.getElementById('addSampleBtn').onclick=async()=>{
  let img=getData(); let flat=[];
  for(let y=0;y<28;y++) for(let x=0;x<28;x++) flat.push(img[y][x]);
  let p=new URLSearchParams(); p.append('img',flat.join(','));
  p.append('label',document.getElementById('label').value);
  let r=await fetch('/addSample?t='+Date.now(),{method:'POST',body:p,cache:'no-store'});
  let txt=await r.text();
  document.getElementById('res').innerHTML=txt;
  document.getElementById('progress-card').style.display='block';
  refreshStatus();
};

async function delSample(idx){
  if(!confirm('确定删除样本 #'+idx+' 吗？')) return;
  let p=new URLSearchParams(); p.append('idx',idx);
  let r=await fetch('/delSample?t='+Date.now(),{method:'POST',body:p,cache:'no-store'});
  let txt=await r.text();
  document.getElementById('res').innerHTML=txt;
  document.getElementById('progress-card').style.display='block';
  refreshStatus();
}

async function renderSampleList(){
  let r=await fetch('/status?t='+Date.now(),{cache:'no-store'});
  let j=await r.json();
  let total=j.samples||0;
  let container=document.getElementById('sampleList');
  if(total===0){ container.innerHTML='暂无样本'; return; }

  let html='';
  for(let i=0;i<total;i++){
    try{
      let sr=await fetch('/sampleData?idx='+i+'&t='+Date.now(),{cache:'no-store'});
      let sj=await sr.json();
      if(sj.error){ continue; }
      let cid='sp_'+i;
      html+='<div style="position:relative;display:inline-block;cursor:pointer;border:2px solid #ccc;border-radius:4px;padding:2px;background:#fff;transition:all 0.15s" '+
            'onmouseenter="this.style.borderColor=\'#222\';this.style.boxShadow=\'0 2px 6px rgba(0,0,0,0.2)\'" '+
            'onmouseleave="this.style.borderColor=\'#ccc\';this.style.boxShadow=\'none\'" '+
            'onclick="delSample('+i+')" title="点击删除样本 #'+i+'">'+
            '<canvas id="'+cid+'" width="56" height="56" style="display:block;border-radius:2px"></canvas>'+
            '<span style="display:block;text-align:center;font-size:11px;font-weight:700;color:#222;margin-top:2px">#'+i+' 数字'+sj.label+'</span>'+
            '<span style="position:absolute;top:-6px;right:-6px;width:18px;height:18px;background:#555;color:#fff;border-radius:50%;'+
            'font-size:12px;line-height:18px;text-align:center;font-weight:bold;display:none" class="del-x">x</span>'+
            '</div>';
    }catch(e){}
  }
  container.innerHTML=html||'暂无样本';

  setTimeout(async()=>{
    for(let i=0;i<total;i++){
      try{
        let sr=await fetch('/sampleData?idx='+i+'&t='+Date.now(),{cache:'no-store'});
        let sj=await sr.json();
        if(sj.error) continue;
        let c=document.getElementById('sp_'+i);
        if(!c) continue;
        let ctx2=c.getContext('2d');
        for(let y=0;y<28;y++){
          for(let x=0;x<28;x++){
            ctx2.fillStyle=sj.pixels[y][x]>=0.5?'#222':'#fff';
            ctx2.fillRect(x*2,y*2,2,2);
          }
        }
      }catch(e){}
    }
    let xs=document.querySelectorAll('.del-x');
    xs.forEach(el=>el.style.display='block');
  },200);
}

function getData(){
  let img=new Array(28).fill().map(()=>new Array(28).fill(0));
  let px=ctx.getImageData(0,0,280,280).data;
  const threshold = 12800;
  for(let y=0;y<28;y++) for(let x=0;x<28;x++){
    let s=0; for(let dy=0;dy<10;dy++) for(let dx=0;dx<10;dx++) s+=px[((y*10+dy)*280+(x*10+dx))*4];
    img[y][x]= s < threshold ? 1 : 0;
  }
  return img;
}

async function send(t){
  let img=getData(); let flat=[];
  for(let y=0;y<28;y++) for(let x=0;x<28;x++) flat.push(img[y][x]);
  let p=new URLSearchParams(); p.append('img',flat.join(','));
  if(t==='train') p.append('label',document.getElementById('label').value);
  let r=await fetch('/'+t+'?t='+Date.now(),{method:'POST',body:p,cache:'no-store'});
  let txt=await r.text();
  document.getElementById('res').innerHTML=txt;
  document.getElementById('progress-card').style.display='block';
  refreshStatus();
}

document.getElementById('trainBtn').onclick=()=>send('train');
document.getElementById('detect').onclick=()=>send('detect');

document.getElementById('trainAll').onclick=async()=>{
  let ep = (new URLSearchParams(window.location.search)).get('epochs');
  if(!confirm('开始全量训练 (' + (ep ? ep : '100') + ' 轮)？训练在后台进行。')) return;
  let r=await fetch('/trainAll?t='+Date.now(),{method:'POST'});
  let txt=await r.text();
  document.getElementById('res').innerHTML=txt;
  document.getElementById('progress-card').style.display='block';
  document.getElementById('stopBtn').style.display='inline-block';
  pollTraining();
};

document.getElementById('stopBtn').onclick=async()=>{
  let r=await fetch('/stop?t='+Date.now());
  let txt=await r.text();
  document.getElementById('res').innerHTML=txt;
  document.getElementById('stopBtn').style.display='none';
};

document.getElementById('clrS').onclick=async()=>{
  if(!confirm('确定清除所有样本？此操作不可恢复！')) return;
  let r=await fetch('/clearS?t='+Date.now());
  let txt=await r.text();
  document.getElementById('res').innerHTML=txt;
  refreshStatus();
};
let pollTimer=null;
function pollTraining(){
  if(pollTimer) clearInterval(pollTimer);
  pollTimer=setInterval(async()=>{
    let r=await fetch('/status?t='+Date.now(),{cache:'no-store'});
    let j=await r.json();
    updateStatusUI(j);
    updateSampleDist(j.counts||[]);
    if(j.state==='done'||j.state==='idle'){
      clearInterval(pollTimer); pollTimer=null;
      document.getElementById('stopBtn').style.display='none';
      document.getElementById('res').innerHTML=j.message||(j.state==='done'?'训练完成':'');
    }
  },1500);
}

function updateStatusUI(j){
  let pct=j.total>0?Math.round(j.epoch/j.total*100):0;
  document.getElementById('pbar').style.width=pct+'%';
  document.getElementById('status-line').innerHTML=
    '进度: '+j.epoch+'/'+j.total+' 轮 | 损失: '+(j.loss||'...')+
    ' | 样本: '+j.samples;
  if(j.state==='running'){
    document.getElementById('stopBtn').style.display='inline-block';
  }
  if(j.temp!==undefined){
    document.getElementById('temp-line').innerHTML='芯片温度: '+j.temp.toFixed(1)+' °C';
  }
}

function updateSampleDist(counts){
  let html='';
  for(let i=0;i<10;i++){
    let cls=counts[i]===0?'sample-chip zero':'sample-chip';
    html+='<span class="'+cls+'">'+i+': '+counts[i]+'</span>';
  }
  document.getElementById('sampleDist').innerHTML=html;
}

async function refreshStatus(){
  try{
    let r=await fetch('/status?t='+Date.now(),{cache:'no-store'});
    let j=await r.json();
    updateSampleDist(j.counts||[]);
    document.getElementById('res-static').innerHTML=
      '共 '+j.samples+' 个样本 | 权重: '+(j.weights_ok?'有效':'无效')+
      ' | 学习率: '+j.lr;
    if(j.temp!==undefined){
      document.getElementById('temp-line').innerHTML='芯片温度: '+j.temp.toFixed(1)+' C';
    }
    if(j.state==='running'){
      updateStatusUI(j);
      document.getElementById('progress-card').style.display='block';
      pollTraining();
    }
    renderSampleList();
  }catch(e){}
}
setInterval(async()=>{
  try{
    let r=await fetch('/status?t='+Date.now(),{cache:'no-store'});
    let j=await r.json();
    if(j.temp!==undefined){
      document.getElementById('temp-line').innerHTML='芯片温度: '+j.temp.toFixed(1)+' C';
    }
  }catch(e){}
},5000);

function renderProbs(probs,pred){
  let html='<div style="font-size:12px;font-weight:600;color:#555;margin-bottom:4px">各类概率</div>';
  for(let i=0;i<10;i++){
    let pct=(probs[i]*100).toFixed(1);
    let w=Math.max(probs[i]*100,0.5); 
    let isPred=i===pred;
    html+='<div style="display:flex;align-items:center;margin:2px 0;font-size:12px">'+
          '<span style="width:16px;font-weight:700;color:'+(isPred?'#111':'#888')+'">'+i+'</span>'+
          '<span style="flex:1;height:14px;background:#e0e0e0;border-radius:4px;overflow:hidden;margin:0 6px">'+
          '<span style="display:block;height:100%;background:'+(isPred?'#222':'#888')+';width:'+w+'%;border-radius:4px;transition:width 0.3s"></span>'+
          '</span>'+
          '<span style="width:42px;text-align:right;font-weight:'+(isPred?'700':'400')+';color:'+(isPred?'#111':'#555')+'">'+pct+'%</span>'+
          '</div>';
  }
  document.getElementById('probBars').innerHTML=html;
}

let origSend=send;
send=async function(t){
  if(t==='detect'){
    document.getElementById('progress-card').style.display='block';
    document.getElementById('res').innerHTML='检测中...';
    document.getElementById('pbar').style.width='0%';
    document.getElementById('probBars').innerHTML='';
    let w=0;
    let anim=setInterval(()=>{w+=15;if(w<=90)document.getElementById('pbar').style.width=w+'%';},50);
    try{
      let img=getData(); let flat=[];
      for(let y=0;y<28;y++) for(let x=0;x<28;x++) flat.push(img[y][x]);
      let p=new URLSearchParams(); p.append('img',flat.join(','));
      let r=await fetch('/detect?t='+Date.now(),{method:'POST',body:p,cache:'no-store'});
      let j=await r.json();
      document.getElementById('res').innerHTML='识别结果: <b>'+j.pred+'</b>';
      renderProbs(j.probs,j.pred);
      document.getElementById('progress-card').style.display='block';
      refreshStatus();
    }finally{
      clearInterval(anim);
      document.getElementById('pbar').style.width='100%';
      setTimeout(()=>{document.getElementById('pbar').style.width='0%';},800);
    }
  }else{
    await origSend(t);
  }
};

refreshStatus();
</script>
</body></html>)raw";

void handleRoot() {
  server.sendHeader("Cache-Control", "no-store");
  server.send_P(200, "text/html", webpage_content);
}

void handleTrain() {
  if (g_async_state == ASYNC_RUNNING) {
    server.send(200, "text/plain", "后台训练进行中，请等待完成或停止训练");
    return;
  }
  if (sample_count >= MAX_SAMPLES) {
    server.send(200, "text/plain", "样本已满 (最大" + String(MAX_SAMPLES) + "个)");
    return;
  }

  int label = server.arg("label").toInt();
  if (label < 0 || label > 9) {
    server.send(200, "text/plain", "标签无效 (需要0-9)");
    return;
  }

  String flat = server.arg("img");
  float img[IMG_H][IMG_W] = {0};
  int st = 0;
  for (int y = 0; y < IMG_H; y++) {
    for (int x = 0; x < IMG_W; x++) {
      int c = flat.indexOf(',', st);
      img[y][x] = flat.substring(st, c == -1 ? flat.length() : c).toFloat();
      st = c + 1;
      if (c == -1 && y < IMG_H - 1 && x < IMG_W - 1) break;
    }
  }

  pack_image(img, train_images[sample_count]);
  train_labels[sample_count] = label;
  sample_count++;
  save_samples();

  train_epochs(5, g_learning_rate);

  if (all_weights_valid()) {
    save_weights();
    String resp = "样本已添加 | 总样本: " + String(sample_count) + "/"
                + String(MAX_SAMPLES) + " | 5轮训练完成";
    server.sendHeader("Cache-Control", "no-store");
    server.send(200, "text/plain", resp);
  } else {
    load_weights();
    server.send(200, "text/plain", "训练异常，权重已回滚");
  }
}

void handleTrainAll() {
  if (g_async_state == ASYNC_RUNNING) {
    server.send(200, "text/plain", "训练已在后台运行中");
    return;
  }
  if (sample_count == 0) {
    server.send(200, "text/plain", "无样本，请先添加样本");
    return;
  }
  start_async_train(g_train_epochs);
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "text/plain", "后台训练已启动 (" + String(g_train_epochs) + "轮)，请等待...");
}

void handleDetect() {
  String flat = server.arg("img");
  float img[IMG_H][IMG_W] = {0};
  int st = 0;
  for (int y = 0; y < IMG_H; y++) {
    for (int x = 0; x < IMG_W; x++) {
      int c = flat.indexOf(',', st);
      img[y][x] = flat.substring(st, c == -1 ? flat.length() : c).toFloat();
      st = c + 1;
      if (c == -1 && y < IMG_H - 1 && x < IMG_W - 1) break;
    }
  }
  forward(img);
  float probs[10];
  memcpy(probs, fcache.logits, sizeof(probs));
  softmax(probs, 10);
  int pred = 0;
  for (int i = 1; i < 10; i++) if (probs[i] > probs[pred]) pred = i;

  String json = "{\"pred\":" + String(pred) + ",\"probs\":[";
  for (int i = 0; i < 10; i++) {
    if (i > 0) json += ",";
    json += String(probs[i], 4);
  }
  json += "]}";
  Serial.println("检测 => " + String(pred));
  server.sendHeader("Cache-Control", "no-store");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void handleClearS() {
  if (g_async_state == ASYNC_RUNNING) {
    server.send(200, "text/plain", "训练进行中，请先停止训练");
    return;
  }
  SPIFFS.remove("/samples.bin");
  sample_count = 0;
  server.send(200, "text/plain", "所有样本已清除");
}

void handleAddSample() {
  if (g_async_state == ASYNC_RUNNING) {
    server.send(200, "text/plain", "后台训练进行中，请等待完成或停止训练");
    return;
  }
  if (sample_count >= MAX_SAMPLES) {
    server.send(200, "text/plain", "样本已满 (最大" + String(MAX_SAMPLES) + "个)");
    return;
  }

  int label = server.arg("label").toInt();
  if (label < 0 || label > 9) {
    server.send(200, "text/plain", "标签无效 (需要0-9)");
    return;
  }

  String flat = server.arg("img");
  float img[IMG_H][IMG_W] = {0};
  int st = 0;
  for (int y = 0; y < IMG_H; y++) {
    for (int x = 0; x < IMG_W; x++) {
      int c = flat.indexOf(',', st);
      img[y][x] = flat.substring(st, c == -1 ? flat.length() : c).toFloat();
      st = c + 1;
      if (c == -1 && y < IMG_H - 1 && x < IMG_W - 1) break;
    }
  }

  pack_image(img, train_images[sample_count]);
  train_labels[sample_count] = label;
  sample_count++;
  save_samples();

  String resp = "样本已添加 | 总样本: " + String(sample_count) + "/" + String(MAX_SAMPLES);
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "text/plain", resp);
}

void handleDelSample() {
  if (g_async_state == ASYNC_RUNNING) {
    server.send(200, "text/plain", "训练进行中，请先停止训练");
    return;
  }
  int idx = server.arg("idx").toInt();
  if (idx < 0 || idx >= sample_count) {
    server.send(200, "text/plain", "无效的样本索引");
    return;
  }
  for (int i = idx; i < sample_count - 1; i++) {
    memcpy(train_images[i], train_images[i + 1], PACKED_SIZE);
    train_labels[i] = train_labels[i + 1];
  }
  sample_count--;
  save_samples();
  String resp = "样本 #" + String(idx) + " 已删除 | 剩余: " + String(sample_count);
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "text/plain", resp);
}

void handleSampleData() {
  int idx = server.arg("idx").toInt();
  if (idx < 0 || idx >= sample_count) {
    server.send(200, "application/json", "{\"error\":\"无效索引\"}");
    return;
  }
  float img[IMG_H][IMG_W];
  unpack_image(train_images[idx], img);
  String json = "{\"label\":" + String(train_labels[idx]) + ",\"pixels\":[";
  for (int y = 0; y < IMG_H; y++) {
    if (y > 0) json += ",";
    json += "[";
    for (int x = 0; x < IMG_W; x++) {
      if (x > 0) json += ",";
      json += String(img[y][x], 0);
    }
    json += "]";
  }
  json += "]}";
  server.sendHeader("Cache-Control", "no-store");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void handleStop() {
  stop_async_train();
  server.send(200, "text/plain", g_async_message);
}

void handleStatus() {
  String json = "{";
  json += "\"state\":\"";
  switch (g_async_state) {
    case ASYNC_IDLE:  json += "idle"; break;
    case ASYNC_RUNNING: json += "running"; break;
    case ASYNC_DONE:  json += "done"; break;
  }
  json += "\",";
  json += "\"epoch\":" + String(g_async_epoch) + ",";
  json += "\"total\":" + String(g_async_total_epochs) + ",";
  json += "\"samples\":" + String(sample_count) + ",";
  json += "\"max_samples\":" + String(MAX_SAMPLES) + ",";
  json += "\"weights_ok\":" + String(all_weights_valid() ? "true" : "false") + ",";
  json += "\"lr\":" + String(g_learning_rate, 5) + ",";
  json += "\"message\":\"" + g_async_message + "\",";

  float last_loss = (g_async_loss_count > 0)
    ? g_async_recent_loss[g_async_loss_count - 1] : -1.0f;
  json += "\"loss\":" + String(last_loss, 4) + ",";

  float temp_c = temperatureRead();
  json += "\"temp\":" + String(temp_c, 1) + ",";

  json += "\"counts\":";
  json += get_sample_stats();

  json += "}";

  server.sendHeader("Cache-Control", "no-store");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void serial_cmd() {
  if (!Serial.available()) return;
  String c = Serial.readStringUntil('\n');
  c.trim();
  c.toLowerCase();

  if (c == "help") {
    Serial.println("  help           - 显示此帮助");
    Serial.println("  heap           - 显示剩余内存");
    Serial.println("  status         - 显示训练参数和样本统计");
    Serial.println("  stats          - 显示每类样本分布");
    Serial.println("  train          - 启动全量训练（同步执行）");
    Serial.println("  clear          - 清除所有样本");
    Serial.println("  clear all      - 清除样本+权重（重置模型）");
    Serial.println("  format         - 格式化 SPIFFS 并重启");
    Serial.println("  stop           - 停止异步训练");
    Serial.println("  set epochs <N> - 设置训练轮数 (1~2000)");
    Serial.println("  set lr <F>     - 设置学习率 (0.0001~0.1)");
  } else if (c == "heap") {
    Serial.printf("剩余内存: %u 字节 (%.1f KB)\n",
                  ESP.getFreeHeap(), ESP.getFreeHeap() / 1024.0f);
    Serial.printf("SPIFFS 总量: %u 字节\n", SPIFFS.totalBytes());
    Serial.printf("SPIFFS 已用: %u 字节\n", SPIFFS.usedBytes());
  } else if (c == "status") {
    Serial.printf("样本数: %d/%d\n", sample_count, MAX_SAMPLES);
    Serial.printf("权重有效: %s\n", all_weights_valid() ? "是" : "否");
    Serial.printf("训练轮数: %d, 学习率: %.5f\n", g_train_epochs, g_learning_rate);
    Serial.printf("异步训练状态: ");
    switch (g_async_state) {
      case ASYNC_IDLE: Serial.println("空闲"); break;
      case ASYNC_RUNNING:
        Serial.printf("运行中 (轮次 %d/%d)\n", g_async_epoch + 1, g_async_total_epochs);
        break;
      case ASYNC_DONE: Serial.println("已完成"); break;
    }
  } else if (c == "stats") {
    Serial.print("样本分布: ");
    Serial.println(get_sample_stats());
  } else if (c == "train") {
    if (g_async_state == ASYNC_RUNNING) {
      Serial.println("异步训练已在运行中");
    } else if (sample_count == 0) {
      Serial.println("无样本");
    } else {
      Serial.printf("开始同步全量训练 (%d轮)...\n", g_train_epochs);
      train_epochs(g_train_epochs, g_learning_rate);
      if (all_weights_valid()) save_weights();
      Serial.println("训练完成");
    }
  } else if (c == "clear") {
    SPIFFS.remove("/samples.bin");
    sample_count = 0;
    Serial.println("样本已清除");
  } else if (c == "clear all") {
    SPIFFS.remove("/samples.bin");
    SPIFFS.remove("/cnn_w.bin");
    sample_count = 0;
    init_weights();
    save_weights();
    Serial.println("模型已完全重置");
  } else if (c == "format") {
    Serial.println("正在格式化 SPIFFS...");
    SPIFFS.format();
    delay(3000);
    ESP.restart();
  } else if (c == "stop") {
    stop_async_train();
  } else if (c.startsWith("set epochs ")) {
    int v = c.substring(11).toInt();
    if (v > 0 && v <= 2000) {
      g_train_epochs = v;
      Serial.printf("训练轮数设为 %d\n", g_train_epochs);
    } else {
      Serial.println("无效值 (范围: 1~2000)");
    }
  } else if (c.startsWith("set lr ")) {
    float v = c.substring(7).toFloat();
    if (v >= 0.0001f && v <= 0.1f) {
      g_learning_rate = v;
      Serial.printf("学习率设为 %.5f\n", g_learning_rate);
    } else {
      Serial.println("无效值 (范围: 0.0001~0.1)");
    }
  } else if (c.length() > 0) {
    Serial.println("未知命令，输入 'help' 查看帮助");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  randomSeed(esp_random());
  Serial.println();

  esp_task_wdt_deinit();

  init_spiffs();
  load_weights();
  load_samples();

  WiFi.begin(ssid, password);
  Serial.print("连接 WiFi");
  int dot_count = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
    dot_count++;
    if (dot_count > 60) {  
      Serial.println("\nWiFi 连接超时，重启...");
      ESP.restart();
    }
  }
  Serial.println();
  Serial.println("WiFi 已连接 | IP: " + WiFi.localIP().toString());

  server.on("/", handleRoot);
  server.on("/train", HTTP_POST, handleTrain);
  server.on("/trainAll", HTTP_POST, handleTrainAll);
  server.on("/detect", HTTP_POST, handleDetect);
  server.on("/clearS", HTTP_GET, handleClearS);
  server.on("/addSample", HTTP_POST, handleAddSample);
  server.on("/delSample", HTTP_POST, handleDelSample);
  server.on("/sampleData", HTTP_GET, handleSampleData);
  server.on("/stop", HTTP_GET, handleStop);
  server.on("/status", HTTP_GET, handleStatus);
  server.begin();

  Serial.println("Web 服务器已启动 | 端口: " + String(WEB_PORT));
  Serial.println("在浏览器中打开: http://" + WiFi.localIP().toString());
  Serial.println("串口输入 'help' 查看命令列表");
}

void loop() {
  server.handleClient();   
  async_train_tick();       
  serial_cmd();           

  if (g_async_state == ASYNC_DONE) {
    delay(100);  
    g_async_state = ASYNC_IDLE;
  }
}
