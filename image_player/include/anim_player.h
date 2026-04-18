#pragma once

#include <stdbool.h>
#include "esp_err.h"
#include "esp_heap_caps.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ANIM_PLAYER_INIT_CONFIG()                   \
    {                                              \
        .task_priority = 4,                        \
        .task_stack = 7168,                        \
        .task_affinity = -1,                       \
        .task_stack_caps = MALLOC_CAP_DEFAULT,     \
    }

typedef void *anim_player_handle_t;

typedef enum {
    PLAYER_ACTION_STOP = 0,
    PLAYER_ACTION_START,
} player_action_t;

typedef enum {
    PLAYER_EVENT_IDLE = 0,
    PLAYER_EVENT_ONE_FRAME_DONE,
    PLAYER_EVENT_ALL_FRAME_DONE,
} player_event_t;

typedef void (*anim_flush_cb_t)(anim_player_handle_t handle, int x1, int y1, int x2, int y2, const void *data);
typedef void (*anim_update_cb_t)(anim_player_handle_t handle, player_event_t event);

typedef struct {
    anim_flush_cb_t flush_cb;
    anim_update_cb_t update_cb;
    void *user_data;

    struct {
        unsigned char swap:1;
    } flags;

    struct {
        int task_priority;
        int task_stack;
        int task_affinity;
        unsigned task_stack_caps;
    } task;
} anim_player_config_t;

anim_player_handle_t anim_player_init(const anim_player_config_t *config);
void anim_player_deinit(anim_player_handle_t handle);
void anim_player_update(anim_player_handle_t handle, player_action_t event);
bool anim_player_flush_ready(anim_player_handle_t handle);
esp_err_t anim_player_set_src_data(anim_player_handle_t handle, const void *src_data, size_t src_len);
void anim_player_get_segment(anim_player_handle_t handle, uint32_t *start, uint32_t *end);
void anim_player_set_segment(anim_player_handle_t handle, uint32_t start, uint32_t end, uint32_t fps, bool repeat);
void *anim_player_get_user_data(anim_player_handle_t handle);

#ifdef __cplusplus
}
#endif
