#ifndef CONTROLS_HPP
#define CONTROLS_HPP

#define CHIP_I2C_ADDR 0x0C
#define BUS "/dev/i2c-1"

#include <cstdint>
#include <unordered_map>

/*
void openI2CBus(); 
void closeI2CBus(); 
uint16_t read_word(uint8_t chip_addr, uint8_t reg_addr); 
void write_word(uint8_t chip_addr, uint8_t reg_addr, uint16_t value);
*/

#define OPT_BASE    0x1000
#define OPT_FOCUS   (OPT_BASE | 0x01)
#define OPT_ZOOM    (OPT_BASE | 0x02)
#define OPT_MOTOR_X (OPT_BASE | 0x03)
#define OPT_MOTOR_Y (OPT_BASE | 0x04)
#define OPT_IRCUT   (OPT_BASE | 0x05)

struct OptionInfo {
    uint8_t reg_addr;
    uint16_t min_value;
    uint16_t max_value;
    uint8_t reset_addr;
};

class Controller {
public: 
    Controller(); 
    ~Controller(); 
    void reset(uint8_t opt, bool flag = true);
    uint16_t get(uint8_t opt, bool flag = false);
    void set(uint8_t opt, uint16_t value, bool flag = true);
    
private: 
    uint16_t read_word(uint8_t chip_addr, uint8_t reg_addr); 
    void write_word(uint8_t chip_addr, uint8_t reg_addr, uint16_t value); 
    bool is_busy(); 
    void waiting_for_free(); 

    int m_bus; 
    static const std::unordered_map<uint8_t, OptionInfo> m_opts; 
}; 

#endif // CONTROLS_HPP

