#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h>
#include "esp_err.h"
#include "esp_log.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "anim_player.h"
#include "anim_vfs.h"
#include "anim_dec.h"

static const char *TAG = "anim_player";

// Some build paths may not generate this Kconfig symbol for the local component yet.
// Keep a safe fallback so compilation doesn't fail.
#ifndef CONFIG_ANIM_PLAYER_DEFAULT_FPS
#define CONFIG_ANIM_PLAYER_DEFAULT_FPS 30
#endif

#define NEED_DELETE     BIT0
#define DELETE_DONE     BIT1
#define WAIT_FLUSH_DONE BIT2
#define WAIT_STOP       BIT3
#define WAIT_STOP_DONE  BIT4

#define FPS_TO_MS(fps) (1000 / (fps))

typedef struct {
    player_action_t action;
} anim_player_event_t;

typedef struct {
    EventGroupHandle_t event_group;
    QueueHandle_t event_queue;
} anim_player_events_t;

typedef struct {
    uint32_t start;
    uint32_t end;
    anim_vfs_handle_t file_desc;
} anim_player_info_t;

typedef struct {
    anim_player_info_t info;
    int run_start;
    int run_end;
    bool repeat;
    int fps;
    anim_flush_cb_t flush_cb;
    anim_update_cb_t update_cb;
    void *user_data;
    anim_player_events_t events;
    TaskHandle_t handle_task;
    struct {
        unsigned char swap: 1;
    } flags;
} anim_player_context_t;

typedef struct {
    player_action_t action;
    int run_start;
    int run_end;
    bool repeat;
    int fps;
    uint32_t last_frame_time;
} anim_player_run_ctx_t;

static inline uint16_t rgb888_to_rgb565(uint32_t color)
{
    return (((color >> 16) & 0xF8) << 8) | (((color >> 8) & 0xFC) << 3) | ((color & 0xF8) >> 3);
}

static void *alloc_prefer_psram(size_t size)
{
    void *p = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) {
        p = malloc(size);
    }
    return p;
}

static void *calloc_prefer_psram(size_t n, size_t size)
{
    void *p = heap_caps_calloc(n, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) {
        p = calloc(n, size);
    }
    return p;
}

static esp_err_t anim_player_parse(const uint8_t *data, size_t data_len, image_header_t *header, anim_player_context_t *ctx)
{
    // Offsets and decode buffers are not DMA-critical; prefer PSRAM to save internal heap.
    uint32_t *offsets = (uint32_t *)alloc_prefer_psram(header->splits * sizeof(uint32_t));
    if (!offsets) {
        ESP_LOGE(TAG, "Failed to allocate memory for offsets");
        return ESP_FAIL;
    }

    anim_dec_calculate_offsets(header, offsets);

    void *frame_buffer = alloc_prefer_psram(header->width * header->split_height * sizeof(uint16_t));
    if (!frame_buffer) {
        ESP_LOGE(TAG, "Failed to allocate memory for frame buffer");
        free(offsets);
        return ESP_FAIL;
    }

    uint8_t *decode_buffer = NULL;
    size_t decode_buffer_capacity = 0;
    static bool s_print_4bit_cap = false;
    if (header->bit_depth == 4) {
        // 4-bit: packed two pixels per byte.
        // Some encoders may append padding/alignment bytes at the end of the RTE stream.
        // Your current log shows: "Output buffer overflow, 3929 > 3792" in 4-bit mode,
        // which means the decoder needs noticeably more than our previous +64 headroom.
        const size_t packed_capacity = (size_t)header->width * (header->split_height + (header->split_height % 2)) / 2;
        // Empirically, +256 has enough slack for the reported overflow (3929 vs 3792).
        // Keep this small-ish to avoid excessive heap usage.
        decode_buffer_capacity = packed_capacity + 256;
        // Keep it aligned for malloc efficiency.
        decode_buffer_capacity = (decode_buffer_capacity + 15u) & ~15u; // align16
        if (!s_print_4bit_cap) {
            ESP_LOGI(TAG, "4bit decode_buffer cap=%u (packed=%u) width=%u split_height=%u",
                     (unsigned)decode_buffer_capacity,
                     (unsigned)packed_capacity,
                     (unsigned)header->width,
                     (unsigned)header->split_height);
            s_print_4bit_cap = true;
        }
        decode_buffer = (uint8_t *)alloc_prefer_psram(decode_buffer_capacity);
    } else if (header->bit_depth == 8) {
        decode_buffer_capacity = (size_t)header->width * header->split_height;
        decode_buffer = (uint8_t *)alloc_prefer_psram(decode_buffer_capacity);
    }
    if (!decode_buffer) {
        ESP_LOGE(TAG, "Failed to allocate memory for decode buffer");
        free(offsets);
        free(frame_buffer);
        return ESP_FAIL;
    }

    uint16_t *pixels = (uint16_t *)frame_buffer;
    uint8_t *huffman_buffer = NULL;
    const size_t huffman_buffer_capacity = (size_t)header->width * header->split_height;

    for (int split = 0; split < header->splits; split++) {
        if (offsets[split] + header->split_lengths[split] > data_len) {
            ESP_LOGE(TAG, "Split %d exceeds frame bounds", split);
            continue;
        }

        const uint8_t *compressed_data = data + offsets[split];
        int compressed_len = header->split_lengths[split];
        esp_err_t decode_result = ESP_FAIL;
        int valid_height = (split == header->splits - 1) ?
            (header->height - split * header->split_height) : header->split_height;
        // Safety: corrupted header could make valid_height exceed split_height and overflow buffers.
        if (valid_height < 0) {
            ESP_LOGE(TAG, "Invalid valid_height=%d for split=%d", valid_height, split);
            continue;
        }
        if (valid_height > header->split_height) {
            valid_height = header->split_height;
        }

        uint8_t comp_type = compressed_data[0];

        // IMPORTANT: SBMP V1.01 changed meaning of comp_type=1.
        //   - V1.01: 0=RTE, 1=Huffman(shared dict; no dict payload inside the block), 2=Huffman(independent dict)
        //   - V1.00: 0=RTE, 2=Huffman(independent dict). comp_type=1 is not expected.
        if (comp_type == ENCODING_TYPE_RLE) {
            decode_result = anim_dec_rte_decode(compressed_data + 1, compressed_len - 1,
                                                decode_buffer, decode_buffer_capacity);
        } else if (comp_type == ENCODING_TYPE_HUFFMAN_SHARED) {
            if (header->is_v101) {
                // In V1.01, comp_type=1 means "shared dict" and the block does NOT contain dict_len/dict bytes.
                if (header->shared_tree == NULL) {
                    ESP_LOGE(TAG, "SBMP V1.01 comp_type=1 but shared dict missing (split=%d)", split);
                    continue;
                }
                if (!huffman_buffer) {
                    huffman_buffer = (uint8_t *)alloc_prefer_psram(huffman_buffer_capacity);
                    if (!huffman_buffer) {
                        ESP_LOGE(TAG, "Failed to allocate shared Huffman buffer");
                        continue;
                    }
                }
                size_t huffman_decoded_len = 0;
                // Pass only compressed payload (skip comp_type byte).
                decode_result = anim_dec_huffman_decode_with_tree(compressed_data + 1, compressed_len - 1,
                                                                  header->shared_tree, header->shared_padding,
                                                                  huffman_buffer, &huffman_decoded_len);
                if (decode_result == ESP_OK) {
                    decode_result = anim_dec_rte_decode(huffman_buffer, huffman_decoded_len,
                                                        decode_buffer, decode_buffer_capacity);
                }
            } else {
                // Old frames: comp_type=1 is unexpected, but to be safe, attempt independent decode.
                if (!huffman_buffer) {
                    huffman_buffer = (uint8_t *)alloc_prefer_psram(huffman_buffer_capacity);
                    if (!huffman_buffer) {
                        ESP_LOGE(TAG, "Failed to allocate independent Huffman buffer");
                        continue;
                    }
                }
                size_t huffman_decoded_len = 0;
                decode_result = anim_dec_huffman_decode(compressed_data, compressed_len, huffman_buffer, &huffman_decoded_len);
                if (decode_result == ESP_OK) {
                    decode_result = anim_dec_rte_decode(huffman_buffer, huffman_decoded_len,
                                                    decode_buffer, decode_buffer_capacity);
                }
            }
        } else if (comp_type == ENCODING_TYPE_HUFFMAN_INDEPENDENT) {
            if (!huffman_buffer) {
                huffman_buffer = (uint8_t *)alloc_prefer_psram(huffman_buffer_capacity);
                if (!huffman_buffer) {
                    ESP_LOGE(TAG, "Failed to allocate independent Huffman buffer");
                    continue;
                }
            }
            size_t huffman_decoded_len = 0;
            decode_result = anim_dec_huffman_decode(compressed_data, compressed_len, huffman_buffer, &huffman_decoded_len);
            if (decode_result == ESP_OK) {
                decode_result = anim_dec_rte_decode(huffman_buffer, huffman_decoded_len,
                                                    decode_buffer, decode_buffer_capacity);
            }
        } else {
            ESP_LOGE(TAG, "Unknown encoding type: %02X", comp_type);
            continue;
        }

        if (decode_result != ESP_OK) {
            ESP_LOGE(TAG, "Failed to decode split %d", split);
            continue;
        }

        if (header->bit_depth == 4) {
            for (int y = 0; y < valid_height; y++) {
                for (int x = 0; x < header->width; x += 2) {
                    uint8_t packed_gray = decode_buffer[y * (header->width / 2) + (x / 2)];
                    uint8_t index1 = (packed_gray & 0xF0) >> 4;
                    uint8_t index2 = (packed_gray & 0x0F);
                    uint32_t color1 = anim_dec_parse_palette(header, index1);
                    uint32_t color2 = anim_dec_parse_palette(header, index2);
                    pixels[y * header->width + x] = ctx->flags.swap ? __builtin_bswap16(rgb888_to_rgb565(color1)) : rgb888_to_rgb565(color1);
                    if (x + 1 < header->width) {
                        pixels[y * header->width + x + 1] = ctx->flags.swap ? __builtin_bswap16(rgb888_to_rgb565(color2)) : rgb888_to_rgb565(color2);
                    }
                }
            }
        } else if (header->bit_depth == 8) {
            for (int y = 0; y < valid_height; y++) {
                for (int x = 0; x < header->width; x++) {
                    uint8_t index = decode_buffer[y * header->width + x];
                    uint32_t color = anim_dec_parse_palette(header, index);
                    pixels[y * header->width + x] = ctx->flags.swap ? __builtin_bswap16(rgb888_to_rgb565(color)) : rgb888_to_rgb565(color);
                }
            }
        } else {
            ESP_LOGE(TAG, "Unsupported bit depth: %d", header->bit_depth);
            continue;
        }

        xEventGroupClearBits(ctx->events.event_group, WAIT_FLUSH_DONE);
        if (ctx->flush_cb) {
            ctx->flush_cb(ctx, 0, split * header->split_height, header->width, split * header->split_height + valid_height, pixels);
        }
        xEventGroupWaitBits(ctx->events.event_group, WAIT_FLUSH_DONE, pdTRUE, pdFALSE, pdMS_TO_TICKS(20));
    }

    free(offsets);
    free(frame_buffer);
    free(huffman_buffer);
    free(decode_buffer);
    anim_dec_free_header(header);
    return ESP_OK;
}

static void anim_player_task(void *arg)
{
    image_header_t header;
    anim_player_context_t *ctx = (anim_player_context_t *)arg;
    anim_player_run_ctx_t run_ctx;
    anim_player_event_t player_event;
    bool pending_wait_stop_done = false;

    run_ctx.action = PLAYER_ACTION_STOP;
    run_ctx.run_start = ctx->run_start;
    run_ctx.run_end = ctx->run_end;
    run_ctx.repeat = ctx->repeat;
    run_ctx.fps = ctx->fps;
    run_ctx.last_frame_time = xTaskGetTickCount();

    while (1) {
        EventBits_t bits = xEventGroupWaitBits(ctx->events.event_group,
                                               NEED_DELETE | WAIT_STOP,
                                               pdTRUE, pdFALSE, pdMS_TO_TICKS(10));

        if (bits & NEED_DELETE) {
            xEventGroupSetBits(ctx->events.event_group, DELETE_DONE);
            vTaskDeleteWithCaps(NULL);
        }
        if (bits & WAIT_STOP) {
            // Do not ack immediately. We need to wait until the decode loop is no longer
            // touching ctx->info.file_desc before signaling WAIT_STOP_DONE.
            run_ctx.action = PLAYER_ACTION_STOP;
            pending_wait_stop_done = true;
        }

        if (xQueueReceive(ctx->events.event_queue, &player_event, 0) == pdTRUE) {
            run_ctx.action = player_event.action;
            run_ctx.run_start = ctx->run_start;
            run_ctx.run_end = ctx->run_end;
            run_ctx.repeat = ctx->repeat;
            run_ctx.fps = ctx->fps;
        }

        if (run_ctx.action == PLAYER_ACTION_STOP) {
            if (pending_wait_stop_done) {
                xEventGroupSetBits(ctx->events.event_group, WAIT_STOP_DONE);
                pending_wait_stop_done = false;
            }
            continue;
        }

        do {
            for (int i = run_ctx.run_start; (i <= run_ctx.run_end) && (run_ctx.action != PLAYER_ACTION_STOP); i++) {
                uint32_t current_time = xTaskGetTickCount();
                uint32_t elapsed = current_time - run_ctx.last_frame_time;
                if (elapsed < pdMS_TO_TICKS(FPS_TO_MS(run_ctx.fps))) {
                    vTaskDelay(pdMS_TO_TICKS(FPS_TO_MS(run_ctx.fps)) - elapsed);
                }
                run_ctx.last_frame_time = xTaskGetTickCount();

                bits = xEventGroupWaitBits(ctx->events.event_group, NEED_DELETE | WAIT_STOP,
                                           pdTRUE, pdFALSE, pdMS_TO_TICKS(0));
                if (bits & NEED_DELETE) {
                    xEventGroupSetBits(ctx->events.event_group, DELETE_DONE);
                    vTaskDelete(NULL);
                }
                if (bits & WAIT_STOP) {
                    run_ctx.action = PLAYER_ACTION_STOP;
                    pending_wait_stop_done = true;
                }

                if (xQueueReceive(ctx->events.event_queue, &player_event, 0) == pdTRUE) {
                    run_ctx.action = player_event.action;
                    run_ctx.run_start = ctx->run_start;
                    run_ctx.run_end = ctx->run_end;
                    run_ctx.fps = ctx->fps;
                    run_ctx.repeat = (run_ctx.action == PLAYER_ACTION_STOP) ? false : ctx->repeat;
                    break;
                }

                const void *frame_data = anim_vfs_get_frame_data(ctx->info.file_desc, i);
                size_t frame_size = anim_vfs_get_frame_size(ctx->info.file_desc, i);
                const uint8_t *global_palette = NULL;
                uint8_t global_bit_depth = 0;
                anim_vfs_get_global_palette(ctx->info.file_desc, &global_palette, &global_bit_depth, NULL);

                image_format_t format = anim_dec_parse_header(frame_data, frame_size, global_palette, global_bit_depth, &header);
                if (format == IMAGE_FORMAT_SBMP) {
                    anim_player_parse(frame_data, frame_size, &header, ctx);
                    if (ctx->update_cb) {
                        ctx->update_cb(ctx, PLAYER_EVENT_ONE_FRAME_DONE);
                    }
                }
            }
            if (ctx->update_cb) {
                ctx->update_cb(ctx, PLAYER_EVENT_ALL_FRAME_DONE);
            }
        } while (run_ctx.repeat);

        run_ctx.action = PLAYER_ACTION_STOP;
        if (ctx->update_cb) {
            ctx->update_cb(ctx, PLAYER_EVENT_IDLE);
        }
    }
}

bool anim_player_flush_ready(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (!ctx) {
        return false;
    }

    if (xPortInIsrContext()) {
        BaseType_t pxHigherPriorityTaskWoken = pdFALSE;
        bool result = xEventGroupSetBitsFromISR(ctx->events.event_group, WAIT_FLUSH_DONE, &pxHigherPriorityTaskWoken);
        if (pxHigherPriorityTaskWoken == pdTRUE) {
            portYIELD_FROM_ISR();
        }
        return result;
    }

    return xEventGroupSetBits(ctx->events.event_group, WAIT_FLUSH_DONE);
}

void anim_player_update(anim_player_handle_t handle, player_action_t event)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (!ctx) {
        return;
    }

    anim_player_event_t player_event = {
        .action = event,
    };
    xQueueSend(ctx->events.event_queue, &player_event, pdMS_TO_TICKS(10));
}

esp_err_t anim_player_set_src_data(anim_player_handle_t handle, const void *src_data, size_t src_len)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (!ctx) {
        return ESP_FAIL;
    }

    anim_vfs_handle_t new_desc;
    anim_vfs_init(src_data, src_len, &new_desc);
    if (!new_desc) {
        ESP_LOGE(TAG, "Failed to initialize asset parser");
        return ESP_FAIL;
    }

    anim_player_update(handle, PLAYER_ACTION_STOP);
    xEventGroupSetBits(ctx->events.event_group, WAIT_STOP);
    xEventGroupWaitBits(ctx->events.event_group, WAIT_STOP_DONE, pdTRUE, pdFALSE, portMAX_DELAY);

    if (ctx->info.file_desc) {
        anim_vfs_deinit(ctx->info.file_desc);
        ctx->info.file_desc = NULL;
    }

    ctx->info.file_desc = new_desc;
    ctx->info.start = 0;
    ctx->info.end = anim_vfs_get_total_frames(new_desc) - 1;
    ctx->run_start = ctx->info.start;
    ctx->run_end = ctx->info.end;
    ctx->repeat = true;
    ctx->fps = CONFIG_ANIM_PLAYER_DEFAULT_FPS;

    return ESP_OK;
}

void anim_player_get_segment(anim_player_handle_t handle, uint32_t *start, uint32_t *end)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (!ctx) {
        return;
    }
    *start = ctx->info.start;
    *end = ctx->info.end;
}

void anim_player_set_segment(anim_player_handle_t handle, uint32_t start, uint32_t end, uint32_t fps, bool repeat)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (!ctx) {
        return;
    }
    if (end > ctx->info.end || start > end) {
        ESP_LOGE(TAG, "Invalid segment");
        return;
    }
    ctx->run_start = start;
    ctx->run_end = end;
    ctx->repeat = repeat;
    ctx->fps = fps;
}

void *anim_player_get_user_data(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    return ctx ? ctx->user_data : NULL;
}

anim_player_handle_t anim_player_init(const anim_player_config_t *config)
{
    if (!config) {
        return NULL;
    }

    anim_player_context_t *player = (anim_player_context_t *)malloc(sizeof(anim_player_context_t));
    if (!player) {
        return NULL;
    }

    player->info.file_desc = NULL;
    player->info.start = 0;
    player->info.end = 0;
    player->run_start = 0;
    player->run_end = 0;
    player->repeat = false;
    player->fps = CONFIG_ANIM_PLAYER_DEFAULT_FPS;
    player->flush_cb = config->flush_cb;
    player->update_cb = config->update_cb;
    player->user_data = config->user_data;
    player->flags.swap = config->flags.swap;
    player->events.event_group = xEventGroupCreate();
    player->events.event_queue = xQueueCreate(5, sizeof(anim_player_event_t));

    const uint32_t caps = config->task.task_stack_caps ? config->task.task_stack_caps : MALLOC_CAP_DEFAULT;
    if (config->task.task_affinity < 0) {
        xTaskCreateWithCaps(anim_player_task, "Anim Player", config->task.task_stack, player, config->task.task_priority, &player->handle_task, caps);
    }

    return (anim_player_handle_t)player;
}

void anim_player_deinit(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (!ctx) {
        return;
    }

    if (ctx->events.event_group) {
        xEventGroupSetBits(ctx->events.event_group, NEED_DELETE);
        xEventGroupWaitBits(ctx->events.event_group, DELETE_DONE, pdTRUE, pdFALSE, portMAX_DELAY);
    }
    if (ctx->events.event_group) {
        vEventGroupDelete(ctx->events.event_group);
    }
    if (ctx->events.event_queue) {
        vQueueDelete(ctx->events.event_queue);
    }
    if (ctx->info.file_desc) {
        anim_vfs_deinit(ctx->info.file_desc);
    }
    free(ctx);
}
