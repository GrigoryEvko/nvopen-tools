/* NVIDIA Fat Binary Unpacker
 * Extracts all CUBIN entries with metadata for repacking.
 *
 * Output: manifest.txt + decompressed .cubin files
 * Build: gcc -O2 -Wall -o fatbin_unpack fatbin_unpack.c -lzstd
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <zstd.h>

#define FATBIN_MAGIC_EXEC   0xBA55ED50

typedef struct {
    uint32_t index;
    uint16_t type;
    uint32_t sm_arch;
    uint64_t original_offset;
    uint32_t data_size;
    uint32_t entry_size;
    uint64_t entry_data_offset;
    char filename[256];
} entry_metadata_t;

typedef struct {
    uint8_t *data;
    size_t size;
    entry_metadata_t *entries;
    int entry_count;
    int entry_capacity;
} context_t;

static int extract_all_entries(context_t *ctx, const char *output_dir) {
    ctx->entry_capacity = 6000;
    ctx->entries = calloc(ctx->entry_capacity, sizeof(entry_metadata_t));
    if (!ctx->entries) {
        fprintf(stderr, "Error: Cannot allocate array\n");
        return -1;
    }

    int container_count = 0;
    uint32_t entry_global_index = 1;
    int total_extracted = 0;

    printf("Extracting fat binary entries...\n");

    for (size_t offset = 0; offset < ctx->size - 4; offset++) {
        uint32_t magic = *(uint32_t *)(ctx->data + offset);

        if (magic != FATBIN_MAGIC_EXEC)
            continue;

        if (offset + 16 > ctx->size)
            continue;

        uint16_t version = *(uint16_t *)(ctx->data + offset + 6);
        uint64_t header_size = *(uint64_t *)(ctx->data + offset + 8);

        if (version != 0x10 || header_size == 0)
            continue;

        if (offset + header_size > ctx->size)
            continue;

        container_count++;

        uint8_t *first_entry = ctx->data + offset + version;
        uint8_t *entry_ptr = first_entry;
        uint8_t *container_end = ctx->data + offset + version + header_size;
        uint8_t *file_end = ctx->data + ctx->size;

        if (container_end > file_end)
            container_end = file_end;

        while (entry_ptr + 64 <= file_end && entry_ptr < container_end) {
            uint16_t type = *(uint16_t *)(entry_ptr + 0);
            uint32_t entry_size = *(uint32_t *)(entry_ptr + 4);
            uint32_t aligned_data_size = *(uint32_t *)(entry_ptr + 8);
            uint32_t data_size = *(uint32_t *)(entry_ptr + 16);
            uint32_t sm_arch = *(uint32_t *)(entry_ptr + 28);

            if (entry_size == 0)
                break;

            size_t entry_rel_offset = entry_ptr - first_entry;
            if (entry_rel_offset + 64 > header_size)
                break;

            if (data_size > 0) {
                if (ctx->entry_count >= ctx->entry_capacity) {
                    ctx->entry_capacity *= 2;
                    entry_metadata_t *new_entries = realloc(ctx->entries,
                                                             ctx->entry_capacity * sizeof(entry_metadata_t));
                    if (!new_entries) {
                        fprintf(stderr, "Error: Cannot expand array\n");
                        return -1;
                    }
                    ctx->entries = new_entries;
                }

                entry_metadata_t *meta = &ctx->entries[ctx->entry_count++];
                meta->index = entry_global_index++;
                meta->type = type;
                meta->sm_arch = sm_arch;
                meta->original_offset = entry_ptr - ctx->data;
                meta->data_size = data_size;
                meta->entry_size = entry_size;
                meta->entry_data_offset = 64;

                const char *type_str = (type == 0x01) ? "ptx" : "elf";
                snprintf(meta->filename, sizeof(meta->filename),
                         "%05d_sm%d_%s.cubin",
                         meta->index, meta->sm_arch, type_str);

                char filepath[4096];
                snprintf(filepath, sizeof(filepath), "%s/%s", output_dir, meta->filename);

                uint8_t *compressed_data = entry_ptr + entry_size;
                size_t decompressed_size = ZSTD_getFrameContentSize(compressed_data, data_size);
                if (decompressed_size == ZSTD_CONTENTSIZE_ERROR ||
                    decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
                    fprintf(stderr, "Warning: Invalid zstd frame in entry %d\n", meta->index);
                    continue;
                }

                uint8_t *decompressed = malloc(decompressed_size);
                if (!decompressed) {
                    fprintf(stderr, "Warning: Out of memory for entry %d\n", meta->index);
                    continue;
                }

                size_t result = ZSTD_decompress(decompressed, decompressed_size,
                                               compressed_data, data_size);
                if (ZSTD_isError(result)) {
                    fprintf(stderr, "Warning: Decompression failed for entry %d: %s\n",
                            meta->index, ZSTD_getErrorName(result));
                    free(decompressed);
                    continue;
                }

                FILE *f = fopen(filepath, "wb");
                if (!f) {
                    fprintf(stderr, "Warning: Cannot create '%s': %s\n", filepath, strerror(errno));
                    free(decompressed);
                } else {
                    fwrite(decompressed, 1, decompressed_size, f);
                    fclose(f);
                    free(decompressed);
                    total_extracted++;
                }
            }

            entry_ptr = entry_ptr + entry_size + aligned_data_size;
        }

        offset += version + header_size - 1;
    }

    printf("Found %d fat binary containers\n", container_count);
    printf("Extracted %d entries\n", total_extracted);

    return 0;
}

static int write_manifest(context_t *ctx, const char *output_dir) {
    char manifest_path[4096];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", output_dir);

    FILE *f = fopen(manifest_path, "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot create manifest: %s\n", strerror(errno));
        return -1;
    }

    fprintf(f, "# NVIDIA Fat Binary Manifest\n");
    fprintf(f, "# Format: index type sm_arch data_size entry_size entry_data_offset filename\n");
    fprintf(f, "# Use with fatbin_pack to rebuild fat binary\n\n");

    for (int i = 0; i < ctx->entry_count; i++) {
        entry_metadata_t *meta = &ctx->entries[i];
        fprintf(f, "%d 0x%02x %d %u %u %lu %s\n",
                meta->index,
                meta->type,
                meta->sm_arch,
                meta->data_size,
                meta->entry_size,
                meta->entry_data_offset,
                meta->filename);
    }

    fclose(f);
    printf("Wrote manifest: %s\n", manifest_path);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("NVIDIA Fat Binary Unpacker\n\n");
        printf("Usage: %s <input.bin> <output_dir>\n\n", argv[0]);
        printf("Extracts all CUBIN entries with metadata for repacking.\n\n");
        printf("Example:\n");
        printf("  %s nv_fatbin.bin /tmp/unpacked\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_dir = argv[2];

    if (mkdir(output_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Error: Cannot create directory '%s': %s\n", output_dir, strerror(errno));
        return 1;
    }

    FILE *f = fopen(input_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open '%s': %s\n", input_path, strerror(errno));
        return 1;
    }

    context_t ctx = {0};
    fseek(f, 0, SEEK_END);
    ctx.size = ftell(f);
    fseek(f, 0, SEEK_SET);

    ctx.data = malloc(ctx.size);
    if (!ctx.data || fread(ctx.data, 1, ctx.size, f) != ctx.size) {
        fprintf(stderr, "Error: Cannot read file: %s\n", strerror(errno));
        fclose(f);
        return 1;
    }
    fclose(f);

    if (extract_all_entries(&ctx, output_dir) != 0) {
        free(ctx.data);
        return 1;
    }

    if (write_manifest(&ctx, output_dir) != 0) {
        free(ctx.entries);
        free(ctx.data);
        return 1;
    }

    printf("\nUnpacking complete!\n");
    printf("Entries: %d\n", ctx.entry_count);
    printf("Output: %s\n", output_dir);

    free(ctx.entries);
    free(ctx.data);
    return 0;
}
