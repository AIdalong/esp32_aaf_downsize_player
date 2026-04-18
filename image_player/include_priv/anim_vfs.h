#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct anim_vfs_t *anim_vfs_handle_t;

esp_err_t anim_vfs_init(const uint8_t *data, size_t data_len, anim_vfs_handle_t *ret_parser);
esp_err_t anim_vfs_deinit(anim_vfs_handle_t handle);
int anim_vfs_get_total_frames(anim_vfs_handle_t handle);
int anim_vfs_get_frame_size(anim_vfs_handle_t handle, int index);
const uint8_t *anim_vfs_get_frame_data(anim_vfs_handle_t handle, int index);
bool anim_vfs_get_global_palette(anim_vfs_handle_t handle, const uint8_t **palette, uint8_t *bit_depth, int *num_colors);

#ifdef __cplusplus
}
#endif
