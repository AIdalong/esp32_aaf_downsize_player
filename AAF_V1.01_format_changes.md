# AAF 格式变更说明 V1.01

本文档描述了 V1.01 版本引入的两项压缩优化：
1. **Huffman 字典共享** - 每帧（SBMP）所有 blocks 共享一个 Huffman 字典
2. **全局调色板共享** - 整个 AAF 文件所有帧共享一个调色板

这两项都是无损优化，**不影响图像质量**，可以显著减小文件体积。

---

## 目录
- [1. SBMP 格式变化 (V1.00 vs V1.01)](#1-sbmp-格式变化-v100-vs-v101)
- [2. AAF 文件格式变化](#2-aaf-文件格式变化)
- [3. 解码流程修改步骤](#3-解码流程修改步骤)
  - [3.1 AAF 文件头解码](#31-aaf-文件头解码)
  - [3.2 SBMP 帧解码](#32-sbmp-帧解码)
  - [3.3 Block 解码](#33-block-解码)
- [4. 向后兼容性](#4-向后兼容性)
- [5. 体积收益估算](#5-体积收益估算)

---

## 1. SBMP 格式变化 (V1.00 vs V1.01)

### V1.00 (原始格式)
```
SBMP:
  [Magic: "_S"]                  2 bytes
  [Version: "\x00V1.00\x00"]     8 bytes
  [Bit depth]                    1 byte
  [Width]                        2 bytes (little-endian)
  [Height]                       2 bytes (little-endian)
  [Number of splits]             2 bytes (little-endian)
  [Split height]                 2 bytes (little-endian)
  [Split lengths]                splits × 2 bytes
  [Palette]                      (2^bit_depth × 4) bytes  ← 每个帧都存
  [Split data]                   每个block:
                                  [压缩类型(1B)] + [数据]
                                    - 类型 0: RTE 压缩
                                    - 类型 2: Huffman (独立字典)
                                      - [dict_size(2B)] + [dict_data] + [compressed_data]
```

### V1.01 (共享字典)
```
SBMP:
  [Magic: "_S"]                  2 bytes
  [Version: "\x00V1.01\x00"]     8 bytes
  [Bit depth]                    1 byte
  [Width]                        2 bytes (little-endian)
  [Height]                       2 bytes (little-endian)
  [Number of splits]             2 bytes (little-endian)
  [Split height]                 2 bytes (little-endian)
  [Split lengths]                splits × 2 bytes
  [Shared dictionary size]       2 bytes (little-endian)  ← 新增: 共享字典大小
  [Shared dictionary data]       dict_size bytes          ← 新增: 整个帧的共享字典
  [Palette]                      (2^bit_depth × 4) bytes
  [Split data]                   每个block:
                                  [压缩类型(1B)] + [数据]
                                    - 类型 0: RTE 压缩 (不变)
                                    - 类型 1: Huffman (共享字典)    ← 新类型
                                      - 直接存储 compressed_data
                                      - 不再存储字典!
                                    - 类型 2: Huffman (独立字典)  ← 保留兼容旧格式
```

**压缩类型标识符变更：**
| 值 | 含义 | V1.00 | V1.01 |
|----|------|--------|--------|
| 0 | 纯 RTE 压缩 | ✅ | ✅ |
| 1 | Huffman + 共享字典 | ❌ | ✅ (新增) |
| 2 | Huffman + 独立字典 | ✅ (原标识符1) | ✅ (兼容) |

---

## 2. AAF 文件格式变化

### V1.00 (原始格式)
```
AAF:
  [Total frames]                4 bytes (little-endian)
  [Checksum]                    4 bytes
  [Data length]                 4 bytes
  [Frame map table]             total_frames × (4 + 4) bytes
                                 - each entry: (frame_size, frame_offset)
  [Merged frame data]           each frame: 0x5A5A + SBMP_data
                                 - SBMP_data 包含完整调色板
```

### V1.01 (全局调色板共享)
```
AAF:
  [Flags]                       1 byte  ← 新增
                                 - bit 0: 1 = 有全局调色板 (frame 不包含调色板)
                                 - bit 1: 1 = 所有 frame 使用共享字典 (V1.01)
                                 - bits 2-7: 保留 (0)
  [Total frames]                4 bytes (little-endian)
  [Checksum]                    4 bytes
  [Data length]                 4 bytes
  [Global palette]              (只有 bit 0 = 1 时存在) ← 新增:
                                 - [Bit depth] 1 byte
                                 - [Palette data] (2^bit_depth × 4) bytes
  [Frame map table]             total_frames × (4 + 4) bytes
  [Merged frame data]           each frame: 0x5A5A + SBMP_data
                                 - 如果 bit 0 = 1: SBMP_data **不包含**调色板
                                 - 如果 bit 0 = 0: SBMP_data 包含调色板 (兼容)
```

**校验和计算范围：**
```
checksum = combine_checksum(flags + total_frames + global_palette + mmap_table + merged_data)
```

---

## 3. 解码流程修改步骤

### 3.1 AAF 文件头解码

**原有代码位置：** 读取 `total_frames`、`checksum`、`data_length`，然后读取 mmap_table。

**需要增加：**
```c
// 新的 AAF V1.01 头部
uint8_t flags = read_byte();
uint32_t total_frames = read_u32_le();
uint32_t checksum = read_u32_le();
uint32_t data_length = read_u32_le();

// 解析 flags
bool has_global_palette = (flags & (1 << 0)) != 0;
bool all_shared_dict = (flags & (1 << 1)) != 0;

// 如果有全局调色板，读取它
uint8_t global_bit_depth;
uint32_t global_palette[256][4];  // 最大 256 色
if (has_global_palette) {
    global_bit_depth = read_byte();
    int num_colors = 1 << global_bit_depth;
    for (int i = 0; i < num_colors; i++) {
        global_palette[i][0] = read_byte(); // B
        global_palette[i][1] = read_byte(); // G
        global_palette[i][2] = read_byte(); // R
        global_palette[i][3] = read_byte(); // A
    }
}

// 然后继续读取 mmap_table ...
```

### 3.2 SBMP 帧解码

**原有逻辑：** 每个 SBMP 读完 header 后，直接读取 palette。

**需要修改：**
```c
// 读完 SBMP header 之后:
if (AAF has_global_palette) {
    // 新版本: 调色板已经在 AAF 头部读取过了，直接跳过这里的 palette
    // 不需要在这里读取 palette!
    // palette 使用 AAF 头部的全局调色板
    use_palette = global_palette;
} else {
    // 旧版本兼容: 从 SBMP 读取 palette
    read palette from SBMP...
    use_palette = this_frame_palette;
}

// 检查 SBMP 版本是否是 V1.01
if (SBMP version == V1.01) {
    // 读取共享字典
    uint16_t dict_size = read_u16_le();
    read shared_dict_data(dict_size);
    // 解码并存储共享字典，供所有 blocks 使用
    huffman_shared_dict = decode_dict(shared_dict_data);
}
```

### 3.3 Block 解码

**原有逻辑：**
- 类型 0: RTE 直接解码
- 类型 1: 读取 `dict_size(2B)` → 读取字典 → 解码

**需要增加对类型 1 的处理：**
```c
uint8_t comp_type = read_byte();

switch (comp_type) {
    case 0:
        // RTE 压缩 - 解码逻辑不变
        decode_rte(block_data);
        break;

    case 1:
        // V1.01 新增: Huffman with shared dictionary
        // 字典已经在帧开头读取过了，直接用它解码
        // block 数据就是纯 compressed_data
        decode_huffman(block_data, huffman_shared_dict);
        break;

    case 2:
        // 兼容旧格式: Huffman with independent dictionary
        // 原有解码逻辑不变
        uint16_t dict_size = read_u16_le();
        read dict_data...
        decode_dict(dict_data);
        decode_huffman(block_data, dict);
        break;
}
```

**Huffman 字典解码方法**（和原来独立字典的解码方法一样，只是只需要解码一次而不是每个 block 一次）：
```c
// 字典存储格式（和原来独立字典完全一样）：
//   [padding(1B)] + [entries...]
//   each entry: [byte_val(1B)] + [code_len(1B)] + [code_bytes]
//
// 这个格式和原来每个 block 存储的字典格式完全一样，所以解码代码可以复用！
```

---

## 4. 向后兼容性

### 新解码器可以读旧文件：
- 如果 `flags.bit0 = 0` → 按旧格式解码，每个 frame 读取自己的调色板
- 如果 SBMP 版本是 `V1.00` → 使用原有解码逻辑（每个 block 独立字典）

### 旧解码器不能读新文件：
- 旧解码器不知道 flags 和全局调色板，会解析错误
- 建议版本检测：新生成的文件都是新格式，需要新解码器

---

## 5. 体积收益估算

### 样例：20 帧动画，每帧 80x80，split_height=16 → 5 blocks/frame, 4bit

| 项目 | V1.00 | V1.01 | 节省 |
|------|--------|--------|------|
| 字典存储 | 20 × 5 = 100 个字典 × 50B = **5000B** | 20 个字典 × 80B = **1600B** | **3400B (68%)** |
| 调色板存储 | 20 × 64B = **1280B** | **64B** | **1216B (95%)** |
| **总计节省** | - | - | **~4.5KB** |

### 对于 8bit，收益更大：

| 项目 | V1.00 | V1.01 | 节省 |
|------|--------|--------|------|
| 字典存储 | 100 个 × 200B = 20000B | 20 × 300B = 6000B | **14000B** |
| 调色板存储 | 20 × 1024B = 20480B | 1024B | **19456B** |
| **总计节省** | - | - | **~33KB** |

### 总体收益比例：
- 4bit 动画：**减少 15-25% 体积**
- 8bit 动画：**减少 25-35% 体积**

---

## 6. 压缩类型说明

| 类型 | 压缩方式 | 适用场景 |
|------|----------|----------|
| 0 (RTE) | 纯游程编码 | 颜色变化剧烈，Huffman 不能进一步压缩 |
| 1 (Huffman 共享字典) | RTE + Huffman，字典共享 | V1.01 默认，能压缩就用这个 |
| 2 (Huffman 独立字典) | RTE + Huffman，每个 block 字典 | 兼容 V1.00 |

---

## 7. 代码命令行参数

```bash
# 默认开启两个优化（当 --enable-huffman 启用时）
python gif_to_aaf.py input/ output/ --split 16 --depth 4 --enable-huffman

# 禁用共享字典（回到每个 block 独立字典）
python gif_to_aaf.py input/ output/ --split 16 --depth 4 --enable-huffman --no-shared-dict

# 禁用全局调色板（回到每个帧独立调色板）
python gif_to_aaf.py input/ output/ --split 16 --depth 4 --enable-huffman --no-global-palette

# 完全禁用 Huffman
python gif_to_aaf.py input/ output/ --split 16 --depth 4
```

默认参数（`--enable-huffman` 不指定）：只使用 RTE 压缩，不启用 Huffman，所以字典共享不生效。
