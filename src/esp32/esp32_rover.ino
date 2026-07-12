#include <esp_now.h>
#include <WiFi.h>
#include <string.h>
uint8_t BASE_MAC[] = {0x68, 0xFE, 0x71, 0x13, 0x14, 0xBC};

#define STM32_SERIAL   Serial2
#define STM32_BAUD     115200
#define STM32_RX_PIN   16
#define STM32_TX_PIN   17

#define MAX_PKT_LEN    80

typedef struct __attribute__((packed)) {
  uint8_t  type;         
  char     payload[72];  
} EspNowPacket;

EspNowPacket tx_pkt;
EspNowPacket rx_pkt;

esp_now_peer_info_t peer_info;

char uart_buf[MAX_PKT_LEN];
int  uart_idx = 0;

volatile bool peer_added = false;

void on_data_sent(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {

}

void on_data_recv(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len) {
  if (len < (int)sizeof(EspNowPacket)) return;
  const EspNowPacket *pkt = (const EspNowPacket *)data;

  if (pkt->type == 'C') {
    STM32_SERIAL.print(pkt->payload);
    STM32_SERIAL.print("\r\n");
  }
}

void setup() {
  Serial.begin(115200);
  STM32_SERIAL.begin(STM32_BAUD, SERIAL_8N1, STM32_RX_PIN, STM32_TX_PIN);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  Serial.print("[ROVER] MAC: ");
  Serial.println(WiFi.macAddress());

  // Kiểm tra BASE_MAC đã được cấu hình chưa
  bool mac_valid = false;
  for (int i = 0; i < 6; i++) {
    if (BASE_MAC[i] != 0xFF) { mac_valid = true; break; }
  }
  if (!mac_valid) {
    Serial.println("[ERROR] BASE_MAC chua duoc cau hinh! Sua BASE_MAC trong code.");
    while (true) delay(1000);
  }

  // Khởi tạo ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ERROR] ESP-NOW init FAIL");
    while (true) delay(1000);
  }

  esp_now_register_send_cb(on_data_sent);
  esp_now_register_recv_cb(on_data_recv);

  // Thêm peer (Base Station)
  memset(&peer_info, 0, sizeof(peer_info));
  memcpy(peer_info.peer_addr, BASE_MAC, 6);
  peer_info.channel = 0;   
  peer_info.encrypt = false;

  if (esp_now_add_peer(&peer_info) == ESP_OK) {
    peer_added = true;
    Serial.println("[ROVER] ESP-NOW peer added OK");
  } else {
    Serial.println("[ERROR] ESP-NOW add peer FAIL");
  }

  Serial.println("[ROVER] Ready — waiting for STM32 DATA...");
}

void loop() {
  while (STM32_SERIAL.available()) {
    char c = (char)STM32_SERIAL.read();

    if (c == '\n') {
      uart_buf[uart_idx] = '\0';

      if (strncmp(uart_buf, "DATA,", 5) == 0 && peer_added) {
        tx_pkt.type = 'D';
        strncpy(tx_pkt.payload, uart_buf, sizeof(tx_pkt.payload) - 1);
        tx_pkt.payload[sizeof(tx_pkt.payload) - 1] = '\0';

        esp_now_send(BASE_MAC, (uint8_t *)&tx_pkt, sizeof(EspNowPacket));
      }
      uart_idx = 0;

    } else if (c != '\r' && uart_idx < MAX_PKT_LEN - 1) {
      uart_buf[uart_idx++] = c;
    }
  }
}