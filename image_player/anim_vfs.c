/*
 * SPDX-FileCopyrightText: 2022-2025 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string.h>
#include <stdlib.h>
#include "esp_check.h"
#include "esp_log.h"
#include "anim_vfs.h"

static const char *TAG = "anim_vfs";

#define ASSETS_FILE_NUM_OFFSET_OLD   0
#define ASSETS_TABLE_OFFSET_OLD      12
#define ASSETS_FILE_MAGIC_HEAD       0x5A5A
#define ASSETS_FILE_MAGIC_LEN        2

#define AAF_FLAG_GLOBAL_PALETTE      0x01

#pragma pack(1)
typedef struct {
    uint32_t asset_size;
    uint32_t asset_offset;
} asset_table_entry_t;
#pragma pack()

typedef struct {
    const uint8_t *asset_mem;
    const asset_table_entry_t *table;
} asset_entry_t;

typedef struct anim_vfs_t {
    asset_entry_t *entries;
    int total_frames;
    uint8_t flags;
    uint8_t global_bit_depth;
    int global_num_colors;
    uint8_t *global_palette;
} anim_vfs_t;

static uint32_t read_u32_le(const uint8_t *data)
{
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static bool build_entries(const uint8_t *data, size_t data_len, uint32_t total_frames,
                          size_t table_offset, asset_entry_t **out_entries)
{
    if (!data || !out_entries || total_frames == 0) {
        return false;
    }

    size_t table_size = total_frames * sizeof(asset_table_entry_t);
    if (table_offset + table_size > data_len) {
        return false;
    }

    asset_entry_t *entries = (asset_entry_t *)calloc(total_frames, sizeof(asset_entry_t));
    if (!entries) {
        return false;
    }

    const asset_table_entry_t *table = (const asset_table_entry_t *)(data + table_offset);
    size_t data_base = table_offset + table_size;

    for (uint32_t i = 0; i < total_frames; i++) {
        size_t asset_offset = data_base + table[i].asset_offset;
        size_t asset_size = table[i].asset_size;
        if (asset_offset + asset_size > data_len || asset_size < ASSETS_FILE_MAGIC_LEN) {
            free(entries);
            return false;
        }

        const uint16_t *magic_ptr = (const uint16_t *)(data + asset_offset);
        if (*magic_ptr != ASSETS_FILE_MAGIC_HEAD) {
            free(entries);
            return false;
        }

        // Validate per-frame format to avoid mis-parsing AAF header/table.
        // Each frame starts with either "_S" (SBMP) or "_R" (redirect) right after the 0x5A5A magic.
        if (asset_size < ASSETS_FILE_MAGIC_LEN + 2) {
            free(entries);
            return false;
        }
        const uint8_t *fmt = data + asset_offset + ASSETS_FILE_MAGIC_LEN;
        if (fmt[0] != '_' || (fmt[1] != 'S' && fmt[1] != 'R')) {
            free(entries);
            return false;
        }

        entries[i].table = &table[i];
        entries[i].asset_mem = data + asset_offset;
    }

    *out_entries = entries;
    return true;
}

esp_err_t anim_vfs_init(const uint8_t *data, size_t data_len, anim_vfs_handle_t *ret_parser)
{
    esp_err_t ret = ESP_OK;
    anim_vfs_t *parser = NULL;
    asset_entry_t *entries = NULL;

    ESP_GOTO_ON_FALSE(data && ret_parser, ESP_ERR_INVALID_ARG, err, TAG, "invalid args");

    parser = (anim_vfs_t *)calloc(1, sizeof(anim_vfs_t));
    ESP_GOTO_ON_FALSE(parser, ESP_ERR_NO_MEM, err, TAG, "no mem for parser");

    bool parsed = false;

    // Prefer V1.01 (new) parsing when the first byte looks like flags (only bit0/bit1 are used).
    // This avoids mis-parsing V1.01 as the legacy layout (which shifts the frame table and
    // can lead to heap corruption later).
    if (data_len >= 13) {
        uint8_t flags = data[0];
        if ((flags & ~0x03) == 0) {
            uint32_t total_frames_new = read_u32_le(data + 1);
            if (total_frames_new > 0 && total_frames_new < 4096) {
                size_t table_offset = 13;
                uint8_t *global_palette = NULL;
                uint8_t global_bit_depth = 0;
                int global_num_colors = 0;

                if (flags & AAF_FLAG_GLOBAL_PALETTE) {
                    ESP_GOTO_ON_FALSE(data_len > table_offset, ESP_ERR_INVALID_SIZE, err, TAG,
                                      "aaf missing global bit depth");
                    global_bit_depth = data[table_offset++];
                    ESP_GOTO_ON_FALSE(global_bit_depth == 4 || global_bit_depth == 8, ESP_ERR_INVALID_ARG, err, TAG,
                                      "bad global bit depth");
                    global_num_colors = 1 << global_bit_depth;
                    size_t palette_size = (size_t)global_num_colors * 4;
                    ESP_GOTO_ON_FALSE(data_len >= table_offset + palette_size, ESP_ERR_INVALID_SIZE, err, TAG,
                                      "aaf missing global palette");
                    global_palette = (uint8_t *)malloc(palette_size);
                    ESP_GOTO_ON_FALSE(global_palette, ESP_ERR_NO_MEM, err, TAG, "no mem for global palette");
                    memcpy(global_palette, data + table_offset, palette_size);
                    table_offset += palette_size;
                }

                if (build_entries(data, data_len, total_frames_new, table_offset, &entries)) {
                    parser->entries = entries;
                    parser->total_frames = (int)total_frames_new;
                    parser->flags = flags;
                    parser->global_bit_depth = global_bit_depth;
                    parser->global_num_colors = global_num_colors;
                    parser->global_palette = global_palette;
                    global_palette = NULL;
                    parsed = true;
                } else {
                    free(global_palette);
                }
            }
        }
    }

    // Fallback: legacy V1.00 layout.
    if (!parsed && data_len >= ASSETS_TABLE_OFFSET_OLD) {
        uint32_t total_frames_old = read_u32_le(data + ASSETS_FILE_NUM_OFFSET_OLD);
        if (build_entries(data, data_len, total_frames_old, ASSETS_TABLE_OFFSET_OLD, &entries)) {
            parser->entries = entries;
            parser->total_frames = (int)total_frames_old;
            parsed = true;
        }
    }

    ESP_GOTO_ON_FALSE(parsed, ESP_ERR_INVALID_CRC, err, TAG, "failed to parse AAF header");
    *ret_parser = parser;
    return ESP_OK;

err:
    if (entries) {
        free(entries);
    }
    if (parser) {
        free(parser->global_palette);
        free(parser);
    }
    return ret;
}

esp_err_t anim_vfs_deinit(anim_vfs_handle_t handle)
{
    anim_vfs_t *parser = (anim_vfs_t *)handle;
    if (!parser) {
        return ESP_OK;
    }
    free(parser->entries);
    free(parser->global_palette);
    free(parser);
    return ESP_OK;
}

int anim_vfs_get_total_frames(anim_vfs_handle_t handle)
{
    anim_vfs_t *parser = (anim_vfs_t *)handle;
    return parser ? parser->total_frames : 0;
}

const uint8_t *anim_vfs_get_frame_data(anim_vfs_handle_t handle, int index)
{
    anim_vfs_t *parser = (anim_vfs_t *)handle;
    if (!parser || index < 0 || index >= parser->total_frames) {
        ESP_LOGE(TAG, "Invalid index: %d", index);
        return NULL;
    }
    return parser->entries[index].asset_mem + ASSETS_FILE_MAGIC_LEN;
}

int anim_vfs_get_frame_size(anim_vfs_handle_t handle, int index)
{
    anim_vfs_t *parser = (anim_vfs_t *)handle;
    if (!parser || index < 0 || index >= parser->total_frames) {
        ESP_LOGE(TAG, "Invalid index: %d", index);
        return -1;
    }
    return (int)parser->entries[index].table->asset_size - ASSETS_FILE_MAGIC_LEN;
}

bool anim_vfs_get_global_palette(anim_vfs_handle_t handle, const uint8_t **palette, uint8_t *bit_depth, int *num_colors)
{
    anim_vfs_t *parser = (anim_vfs_t *)handle;
    if (!parser || !parser->global_palette) {
        return false;
    }
    if (palette) {
        *palette = parser->global_palette;
    }
    if (bit_depth) {
        *bit_depth = parser->global_bit_depth;
    }
    if (num_colors) {
        *num_colors = parser->global_num_colors;
    }
    return true;
}
