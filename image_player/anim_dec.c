/*
 * SPDX-FileCopyrightText: 2022-2025 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string.h>
#include <stdlib.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "anim_dec.h"

static const char *TAG = "anim_decoder";

static uint16_t read_u16_le(const uint8_t *data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static Node *create_node(void);
static esp_err_t build_huffman_tree(const uint8_t *dict_bytes, size_t dict_len,
                                    Node **out_root, uint8_t *out_padding,
                                    Node **out_pool, size_t *out_pool_nodes);
static esp_err_t decode_huffman_data(const uint8_t *data, size_t data_len,
                                     const uint8_t *dict_bytes, size_t dict_len,
                                     uint8_t *output, size_t *output_len);

uint32_t anim_dec_parse_palette(const image_header_t *header, uint8_t index)
{
    if (!header || !header->palette) {
        return 0;
    }
    if (index >= (uint8_t)header->num_colors) {
        // Corrupted/invalid pixel index: safe fallback to avoid OOB reads.
        return 0;
    }
    const uint8_t *color = &header->palette[index * 4];
    return (color[2] << 16) | (color[1] << 8) | color[0];
}

image_format_t anim_dec_parse_header(const uint8_t *data, size_t data_len,
                                     const uint8_t *global_palette, uint8_t global_bit_depth,
                                     image_header_t *header)
{
    memset(header, 0, sizeof(image_header_t));
    if (!data || data_len < 18) {
        ESP_LOGE(TAG, "Frame too short");
        return IMAGE_FORMAT_INVALID;
    }

    memcpy(header->format, data, 2);
    header->format[2] = '\0';

    if (strncmp(header->format, "_S", 2) == 0) {
        memcpy(header->version, data + 3, 6);
        header->version[6] = '\0';
        header->is_v101 = (strncmp(header->version, "V1.01", 6) == 0);

        header->bit_depth = data[9];
        if (header->bit_depth != 4 && header->bit_depth != 8) {
            ESP_LOGE(TAG, "Invalid bit depth: %d", header->bit_depth);
            return IMAGE_FORMAT_INVALID;
        }

        if (global_palette && global_bit_depth != header->bit_depth) {
            ESP_LOGE(TAG, "Global palette bit depth mismatch: %u != %u", global_bit_depth, header->bit_depth);
            return IMAGE_FORMAT_INVALID;
        }

        header->width = read_u16_le(data + 10);
        header->height = read_u16_le(data + 12);
        header->splits = read_u16_le(data + 14);
        header->split_height = read_u16_le(data + 16);

        if (header->splits == 0) {
            ESP_LOGE(TAG, "Invalid split count");
            return IMAGE_FORMAT_INVALID;
        }

        size_t cursor = 18;
        size_t split_lengths_size = (size_t)header->splits * sizeof(uint16_t);
        if (data_len < cursor + split_lengths_size) {
            ESP_LOGE(TAG, "Frame too short for split lengths");
            return IMAGE_FORMAT_INVALID;
        }

        header->split_lengths = (uint16_t *)malloc(split_lengths_size);
        if (!header->split_lengths) {
            ESP_LOGE(TAG, "Failed to allocate split lengths");
            return IMAGE_FORMAT_INVALID;
        }

        for (int i = 0; i < header->splits; i++) {
            header->split_lengths[i] = read_u16_le(data + cursor + i * 2);
        }
        cursor += split_lengths_size;

        header->num_colors = 1 << header->bit_depth;

        if (header->is_v101) {
            if (data_len < cursor + 2) {
                ESP_LOGE(TAG, "Frame too short for shared dictionary length");
                anim_dec_free_header(header);
                return IMAGE_FORMAT_INVALID;
            }
            header->shared_dict_len = read_u16_le(data + cursor);
            cursor += 2;
            if (data_len < cursor + header->shared_dict_len) {
                ESP_LOGE(TAG, "Frame too short for shared dictionary");
                anim_dec_free_header(header);
                return IMAGE_FORMAT_INVALID;
            }
            if (header->shared_dict_len > 0) {
                // Shared Huffman dictionary is CPU-only data; prefer PSRAM to save internal SRAM.
                header->shared_dict = (uint8_t *)heap_caps_malloc(header->shared_dict_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
                if (!header->shared_dict) {
                    header->shared_dict = (uint8_t *)malloc(header->shared_dict_len);
                }
                if (!header->shared_dict) {
                    ESP_LOGE(TAG, "Failed to allocate shared dictionary");
                    anim_dec_free_header(header);
                    return IMAGE_FORMAT_INVALID;
                }
                memcpy(header->shared_dict, data + cursor, header->shared_dict_len);
                // Build shared Huffman decode tree once per frame using a pooled allocation
                // (avoids many small allocs and reduces internal heap fragmentation).
                if (build_huffman_tree(header->shared_dict, header->shared_dict_len,
                                       &header->shared_tree, &header->shared_padding,
                                       &header->shared_tree_pool, &header->shared_tree_pool_nodes) != ESP_OK) {
                    ESP_LOGE(TAG, "Failed to build shared Huffman tree");
                    anim_dec_free_header(header);
                    return IMAGE_FORMAT_INVALID;
                }
                cursor += header->shared_dict_len;
            }
        }

        if (global_palette) {
            header->palette = (uint8_t *)global_palette;
            header->palette_owned = false;
        } else {
            size_t palette_size = (size_t)header->num_colors * 4;
            if (data_len < cursor + palette_size) {
                ESP_LOGE(TAG, "Frame too short for palette");
                anim_dec_free_header(header);
                return IMAGE_FORMAT_INVALID;
            }
            header->palette = (uint8_t *)malloc(palette_size);
            if (!header->palette) {
                ESP_LOGE(TAG, "Failed to allocate palette");
                anim_dec_free_header(header);
                return IMAGE_FORMAT_INVALID;
            }
            memcpy(header->palette, data + cursor, palette_size);
            header->palette_owned = true;
            cursor += palette_size;
        }

        header->data_offset = (uint32_t)cursor;
        return IMAGE_FORMAT_SBMP;
    }

    if (strncmp(header->format, "_R", 2) == 0) {
        uint8_t file_length = data[2];
        header->palette = (uint8_t *)malloc(file_length + 1);
        if (!header->palette) {
            ESP_LOGE(TAG, "Failed to allocate redirect filename");
            return IMAGE_FORMAT_INVALID;
        }
        memcpy(header->palette, data + 3, file_length);
        header->palette[file_length] = '\0';
        header->palette_owned = true;
        header->num_colors = file_length + 1;
        return IMAGE_FORMAT_REDIRECT;
    }

    ESP_LOGE(TAG, "Invalid format: %s", header->format);
    return IMAGE_FORMAT_INVALID;
}

void anim_dec_calculate_offsets(const image_header_t *header, uint32_t *offsets)
{
    offsets[0] = header->data_offset;
    for (int i = 1; i < header->splits; i++) {
        offsets[i] = offsets[i - 1] + header->split_lengths[i - 1];
    }
}

void anim_dec_free_header(image_header_t *header)
{
    if (header->split_lengths) {
        free(header->split_lengths);
        header->split_lengths = NULL;
    }
    if (header->palette && header->palette_owned) {
        free(header->palette);
        header->palette = NULL;
    }
    if (header->shared_dict) {
        free(header->shared_dict);
        header->shared_dict = NULL;
    }
    header->shared_dict_len = 0;
    if (header->shared_tree) {
        header->shared_tree = NULL;
    }
    if (header->shared_tree_pool) {
        free(header->shared_tree_pool);
        header->shared_tree_pool = NULL;
    }
    header->shared_tree_pool_nodes = 0;
    header->shared_padding = 0;
}

esp_err_t anim_dec_rte_decode(const uint8_t *input, size_t input_len, uint8_t *output, size_t output_len)
{
    size_t in_pos = 0;
    size_t out_pos = 0;

    while (in_pos + 1 <= input_len) {
        uint8_t count = input[in_pos++];
        uint8_t value = input[in_pos++];

        if (out_pos + count > output_len) {
            ESP_LOGE(TAG, "Output buffer overflow, %d > %d", out_pos + count, output_len);
            return ESP_FAIL;
        }

        for (uint8_t i = 0; i < count; i++) {
            output[out_pos++] = value;
        }
    }

    return ESP_OK;
}

static Node *create_node(void)
{
    // Legacy fallback (should not be used by pooled builder).
    return (Node *)calloc(1, sizeof(Node));
}

static esp_err_t build_huffman_tree(const uint8_t *dict_bytes, size_t dict_len,
                                    Node **out_root, uint8_t *out_padding,
                                    Node **out_pool, size_t *out_pool_nodes)
{
    if (!dict_bytes || dict_len == 0 || !out_root || !out_padding || !out_pool || !out_pool_nodes) {
        return ESP_FAIL;
    }

    uint8_t padding = dict_bytes[0];
    size_t dict_pos = 1;

    // First pass: estimate upper bound on node count (1 root + sum(code_len) for all entries).
    size_t scan_pos = dict_pos;
    size_t max_nodes = 1;
    while (scan_pos < dict_len) {
        if (scan_pos + 2 > dict_len) {
            return ESP_FAIL;
        }
        /* uint8_t byte_val = */ (void)dict_bytes[scan_pos++];
        uint8_t code_len = dict_bytes[scan_pos++];
        size_t code_byte_len = (code_len + 7) / 8;
        if (scan_pos + code_byte_len > dict_len) {
            return ESP_FAIL;
        }
        scan_pos += code_byte_len;
        max_nodes += code_len;
    }
    if (max_nodes < 1) {
        return ESP_FAIL;
    }

    // Prefer PSRAM to reduce internal heap pressure; fall back to internal if needed.
    Node *pool = (Node *)heap_caps_calloc(max_nodes, sizeof(Node), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!pool) {
        pool = (Node *)calloc(max_nodes, sizeof(Node));
    }
    if (!pool) {
        return ESP_ERR_NO_MEM;
    }
    size_t pool_used = 0;
    Node *root = &pool[pool_used++];
    Node *current = NULL;

    while (dict_pos < dict_len) {
        if (dict_pos + 2 > dict_len) {
            free(pool);
            return ESP_FAIL;
        }
        uint8_t byte_val = dict_bytes[dict_pos++];
        uint8_t code_len = dict_bytes[dict_pos++];

        size_t code_byte_len = (code_len + 7) / 8;
        if (dict_pos + code_byte_len > dict_len) {
            free(pool);
            return ESP_FAIL;
        }

        uint64_t code = 0;
        for (size_t i = 0; i < code_byte_len; ++i) {
            code = (code << 8) | dict_bytes[dict_pos++];
        }

        current = root;
        for (int bit = code_len - 1; bit >= 0; --bit) {
            int bit_val = (code >> bit) & 1;
            Node **next = bit_val == 0 ? &current->left : &current->right;
            if (!*next) {
                if (pool_used >= max_nodes) {
                    free(pool);
                    return ESP_FAIL;
                }
                *next = &pool[pool_used++];
            }
            current = *next;
        }
        current->is_leaf = 1;
        current->value = byte_val;
    }

    *out_root = root;
    *out_padding = padding;
    *out_pool = pool;
    *out_pool_nodes = max_nodes;
    return ESP_OK;
}

esp_err_t anim_dec_huffman_decode_with_tree(const uint8_t *data, size_t data_len,
                                            const Node *root, uint8_t padding,
                                            uint8_t *output, size_t *output_len)
{
    if (!data || !root || !output || !output_len) {
        return ESP_FAIL;
    }
    if (data_len == 0) {
        *output_len = 0;
        return ESP_OK;
    }

    size_t total_bits = data_len * 8;
    if (padding > total_bits) {
        return ESP_FAIL;
    }
    total_bits -= padding;

    const Node *current = root;
    size_t out_pos = 0;
    for (size_t bit_index = 0; bit_index < total_bits; bit_index++) {
        size_t byte_idx = bit_index / 8;
        int bit_offset = 7 - (bit_index % 8);
        int bit = (data[byte_idx] >> bit_offset) & 1;

        current = bit == 0 ? current->left : current->right;
        if (!current) {
            ESP_LOGE(TAG, "Invalid path in Huffman tree at bit %zu", bit_index);
            return ESP_FAIL;
        }

        if (current->is_leaf) {
            output[out_pos++] = current->value;
            current = root;
        }
    }

    *output_len = out_pos;
    return ESP_OK;
}

static esp_err_t decode_huffman_data(const uint8_t *data, size_t data_len,
                                     const uint8_t *dict_bytes, size_t dict_len,
                                     uint8_t *output, size_t *output_len)
{
    if (!data || !dict_bytes || !output || !output_len || dict_len == 0) {
        return ESP_FAIL;
    }

    Node *root = NULL;
    Node *pool = NULL;
    size_t pool_nodes = 0;
    uint8_t padding = 0;
    esp_err_t ret = build_huffman_tree(dict_bytes, dict_len, &root, &padding, &pool, &pool_nodes);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = anim_dec_huffman_decode_with_tree(data, data_len, root, padding, output, output_len);
    free(pool);
    return ret;
}

esp_err_t anim_dec_huffman_decode(const uint8_t *buffer, size_t buflen, uint8_t *output, size_t *output_len)
{
    if (!buffer || buflen < 3 || !output || !output_len) {
        return ESP_FAIL;
    }

    uint16_t dict_len = (buffer[2] << 8) | buffer[1];
    if (buflen < 3 + dict_len) {
        ESP_LOGE(TAG, "Buffer too short for dictionary");
        return ESP_FAIL;
    }

    size_t data_len = buflen - 3 - dict_len;
    if (data_len == 0) {
        ESP_LOGE(TAG, "No data to decode");
        return ESP_FAIL;
    }

    return decode_huffman_data(buffer + 3 + dict_len, data_len,
                               buffer + 3, dict_len,
                               output, output_len);
}

esp_err_t anim_dec_huffman_decode_shared(const uint8_t *data, size_t data_len,
                                         const uint8_t *dict_bytes, size_t dict_len,
                                         uint8_t *output, size_t *output_len)
{
    if (!data || data_len == 0) {
        return ESP_FAIL;
    }
    return decode_huffman_data(data, data_len, dict_bytes, dict_len, output, output_len);
}
