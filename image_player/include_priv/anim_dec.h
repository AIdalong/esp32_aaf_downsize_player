#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IMAGE_FORMAT_SBMP = 0,
    IMAGE_FORMAT_REDIRECT = 1,
    IMAGE_FORMAT_INVALID = 2
} image_format_t;

typedef enum {
    ENCODING_TYPE_RLE = 0,
    ENCODING_TYPE_HUFFMAN_SHARED = 1,
    ENCODING_TYPE_HUFFMAN_INDEPENDENT = 2,
    ENCODING_TYPE_INVALID = 3
} encoding_type_t;

typedef struct {
    char format[3];
    char version[7];
    bool is_v101;
    uint8_t bit_depth;
    uint16_t width;
    uint16_t height;
    uint16_t splits;
    uint16_t split_height;
    uint16_t *split_lengths;
    uint32_t data_offset;
    uint8_t *palette;
    bool palette_owned;
    int num_colors;
    uint8_t *shared_dict;
    uint16_t shared_dict_len;
    uint8_t shared_padding;
    struct Node *shared_tree;
    // Node pool for shared_tree to avoid many small heap allocs (fragmentation).
    // If non-null, shared_tree nodes are allocated from this pool and must be freed with free().
    struct Node *shared_tree_pool;
    size_t shared_tree_pool_nodes;
} image_header_t;

typedef struct Node {
    uint8_t is_leaf;
    uint8_t value;
    struct Node *left;
    struct Node *right;
} Node;

image_format_t anim_dec_parse_header(const uint8_t *data, size_t data_len,
                                     const uint8_t *global_palette, uint8_t global_bit_depth,
                                     image_header_t *header);
uint32_t anim_dec_parse_palette(const image_header_t *header, uint8_t index);
void anim_dec_calculate_offsets(const image_header_t *header, uint32_t *offsets);
void anim_dec_free_header(image_header_t *header);
esp_err_t anim_dec_huffman_decode(const uint8_t *buffer, size_t buflen, uint8_t *output, size_t *output_len);
esp_err_t anim_dec_huffman_decode_shared(const uint8_t *data, size_t data_len,
                                         const uint8_t *dict_bytes, size_t dict_len,
                                         uint8_t *output, size_t *output_len);
esp_err_t anim_dec_huffman_decode_with_tree(const uint8_t *data, size_t data_len,
                                            const Node *root, uint8_t padding,
                                            uint8_t *output, size_t *output_len);
esp_err_t anim_dec_rte_decode(const uint8_t *input, size_t input_len, uint8_t *output, size_t output_len);

#ifdef __cplusplus
}
#endif
