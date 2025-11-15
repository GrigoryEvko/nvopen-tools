/* NVIDIA Fat Binary PTX Extractor
 * Extracts PTX (type 0x01) from fat binaries, handles ZSTD compression.
 * Output organized by SM architecture.
 *
 * Build: gcc -O2 -Wall -o fatbin_extract_ptx fatbin_extract_ptx.c -lzstd
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
#define FATBIN_MAGIC_RELOC  0x466243B1

#define ENTRY_TYPE_PTX      0x01
#define ENTRY_TYPE_ELF      0x02
#define ENTRY_TYPE_ELF_ALT  0x10

#define ZSTD_MAGIC          0xFD2FB528

typedef struct {
    uint32_t global_index;
    uint32_t arch_index;
    uint32_t sm_arch;
    uint64_t file_offset;
    uint32_t comp_size;
    uint32_t uncomp_size;
    int is_compressed;
} ptx_entry_t;

typedef struct {
    uint8_t *data;
    size_t size;
    ptx_entry_t *entries;
    int entry_count;
    int entry_capacity;
    int arch_counters[256];
} context_t;

static int is_zstd_compressed(const uint8_t *data, size_t size) {
    if (size < 4)
        return 0;

    uint32_t magic = *(uint32_t *)data;
    return (magic == ZSTD_MAGIC);
}

static uint8_t *decompress_zstd(const uint8_t *comp_data, size_t comp_size, size_t *out_size) {
    unsigned long long uncomp_size = ZSTD_getFrameContentSize(comp_data, comp_size);

    if (uncomp_size == ZSTD_CONTENTSIZE_ERROR) {
        fprintf(stderr, "Error: Invalid ZSTD frame\n");
        return NULL;
    }

    if (uncomp_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        uncomp_size = comp_size * 4;
    }

    uint8_t *uncomp_data = malloc(uncomp_size);
    if (!uncomp_data) {
        fprintf(stderr, "Error: Cannot allocate %llu bytes for decompression\n", uncomp_size);
        return NULL;
    }

    size_t result = ZSTD_decompress(uncomp_data, uncomp_size, comp_data, comp_size);

    if (ZSTD_isError(result)) {
        fprintf(stderr, "Error: ZSTD decompression failed: %s\n", ZSTD_getErrorName(result));
        free(uncomp_data);
        return NULL;
    }

    *out_size = result;
    return uncomp_data;
}

static int scan_ptx_entries(context_t *ctx) {
    ctx->entry_capacity = 1000;
    ctx->entries = calloc(ctx->entry_capacity, sizeof(ptx_entry_t));
    if (!ctx->entries) {
        fprintf(stderr, "Error: Cannot allocate entry array\n");
        return -1;
    }

    memset(ctx->arch_counters, 0, sizeof(ctx->arch_counters));

    int container_count = 0;
    uint32_t global_index = 1;
    int total_entries_seen = 0;

    printf("Scanning for PTX entries...\n");

    for (size_t offset = 0; offset < ctx->size - 16; offset++) {
        uint32_t magic = *(uint32_t *)(ctx->data + offset);

        if (magic != FATBIN_MAGIC_EXEC && magic != FATBIN_MAGIC_RELOC)
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
        uint8_t *container_end = ctx->data + offset + header_size;
        uint8_t *file_end = ctx->data + ctx->size;

        if (container_end > file_end)
            container_end = file_end;

        while (entry_ptr + 64 <= file_end && entry_ptr < container_end) {
            uint16_t type = *(uint16_t *)(entry_ptr + 0);
            uint32_t entry_size = *(uint32_t *)(entry_ptr + 4);
            uint64_t data_offset = *(uint64_t *)(entry_ptr + 8);
            uint32_t data_size = *(uint32_t *)(entry_ptr + 16);
            uint32_t sm_arch = *(uint32_t *)(entry_ptr + 28);

            if (data_offset == 0 && entry_size == 0)
                break;

            if (entry_size == 0)
                break;

            size_t entry_rel_offset = entry_ptr - first_entry;
            if (entry_rel_offset + 64 > header_size)
                break;

            total_entries_seen++;

            if (type == ENTRY_TYPE_PTX && data_size > 0) {
                if (sm_arch < 50 || sm_arch > 200) {
                    entry_ptr = entry_ptr + data_offset + entry_size;
                    continue;
                }

                if (ctx->entry_count >= ctx->entry_capacity) {
                    ctx->entry_capacity *= 2;
                    ptx_entry_t *new_entries = realloc(ctx->entries,
                                                        ctx->entry_capacity * sizeof(ptx_entry_t));
                    if (!new_entries) {
                        fprintf(stderr, "Error: Cannot expand entry array\n");
                        return -1;
                    }
                    ctx->entries = new_entries;
                }

                ptx_entry_t *entry = &ctx->entries[ctx->entry_count++];

                entry->global_index = global_index++;
                entry->sm_arch = sm_arch;
                entry->file_offset = (entry_ptr - ctx->data) + data_offset;

                if (sm_arch < 256) {
                    ctx->arch_counters[sm_arch]++;
                    entry->arch_index = ctx->arch_counters[sm_arch];
                } else {
                    entry->arch_index = 1;
                }

                uint8_t *ptx_data = ctx->data + entry->file_offset;
                entry->is_compressed = is_zstd_compressed(ptx_data, data_size);

                if (entry->is_compressed) {
                    entry->comp_size = data_size;
                    entry->uncomp_size = 0;
                } else {
                    entry->comp_size = 0;
                    entry->uncomp_size = data_size;
                }
            }

            entry_ptr = entry_ptr + data_offset + entry_size;
        }

        offset += header_size - 1;
    }

    printf("Found %d fat binary containers\n", container_count);
    printf("Processed %d total entries (all types)\n", total_entries_seen);
    printf("Found %d PTX entries (type 0x01)\n\n", ctx->entry_count);

    return 0;
}

static int extract_ptx_entries(context_t *ctx, const char *output_dir) {
    int extracted = 0;
    int failed = 0;

    printf("Extracting PTX entries to: %s\n", output_dir);

    if (mkdir(output_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Error: Cannot create directory '%s': %s\n", output_dir, strerror(errno));
        return -1;
    }

    for (int i = 0; i < ctx->entry_count; i++) {
        ptx_entry_t *entry = &ctx->entries[i];

        char arch_dir[4096];
        snprintf(arch_dir, sizeof(arch_dir), "%s/sm_%u", output_dir, entry->sm_arch);

        if (mkdir(arch_dir, 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "Warning: Cannot create directory '%s': %s\n", arch_dir, strerror(errno));
            failed++;
            continue;
        }

        char output_path[4096];
        snprintf(output_path, sizeof(output_path),
                 "%s/extracted_sm%u_%05u.ptx",
                 arch_dir, entry->sm_arch, entry->arch_index);

        uint8_t *ptx_data = ctx->data + entry->file_offset;
        uint8_t *output_data = NULL;
        size_t output_size = 0;

        if (entry->is_compressed) {
            output_data = decompress_zstd(ptx_data, entry->comp_size, &output_size);
            if (!output_data) {
                fprintf(stderr, "Warning: Failed to decompress PTX at index %u\n", entry->global_index);
                failed++;
                continue;
            }
            entry->uncomp_size = output_size;
        } else {
            output_data = ptx_data;
            output_size = entry->uncomp_size;
        }

        FILE *f = fopen(output_path, "wb");
        if (!f) {
            fprintf(stderr, "Warning: Cannot create '%s': %s\n", output_path, strerror(errno));
            if (entry->is_compressed)
                free(output_data);
            failed++;
            continue;
        }

        size_t written = fwrite(output_data, 1, output_size, f);
        fclose(f);

        if (entry->is_compressed)
            free(output_data);

        if (written != output_size) {
            fprintf(stderr, "Warning: Incomplete write to '%s'\n", output_path);
            failed++;
            continue;
        }

        extracted++;

        if (extracted % 10 == 0 || extracted == ctx->entry_count) {
            printf("  Extracted %d/%d PTX entries...\r", extracted, ctx->entry_count);
            fflush(stdout);
        }
    }

    printf("\n");

    return extracted;
}

static void print_statistics(context_t *ctx) {
    typedef struct {
        uint32_t arch;
        int count;
        int compressed_count;
        int uncompressed_count;
    } arch_stats_t;

    arch_stats_t stats[256];
    int arch_count = 0;

    for (int i = 0; i < ctx->entry_count; i++) {
        ptx_entry_t *entry = &ctx->entries[i];

        int found = -1;
        for (int j = 0; j < arch_count; j++) {
            if (stats[j].arch == entry->sm_arch) {
                found = j;
                break;
            }
        }

        if (found < 0) {
            if (arch_count >= 256)
                continue;
            found = arch_count++;
            stats[found].arch = entry->sm_arch;
            stats[found].count = 0;
            stats[found].compressed_count = 0;
            stats[found].uncompressed_count = 0;
        }

        stats[found].count++;
        if (entry->is_compressed)
            stats[found].compressed_count++;
        else
            stats[found].uncompressed_count++;
    }

    printf("\nPTX Extraction Statistics:\n");
    printf("==========================\n\n");
    printf("Total PTX entries: %d\n\n", ctx->entry_count);

    printf("Architecture Distribution:\n");
    for (int i = 0; i < arch_count; i++) {
        printf("  sm_%-3u : %4d entries (compressed: %d, plain: %d)\n",
               stats[i].arch, stats[i].count,
               stats[i].compressed_count, stats[i].uncompressed_count);
    }

    printf("\n");
}

int main(int argc, char *argv[]) {
    context_t ctx = {0};

    if (argc != 3) {
        printf("NVIDIA Fat Binary PTX Extractor\n\n");
        printf("Usage: %s <input.fatbin> <output_dir>\n\n", argv[0]);
        printf("Extracts PTX code (type 0x01) organized by SM architecture.\n");
        printf("Handles ZSTD compression automatically.\n\n");
        printf("Example:\n");
        printf("  %s nv_fatbin.bin /tmp/ptx\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_dir = argv[2];

    FILE *f = fopen(input_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open '%s': %s\n", input_path, strerror(errno));
        return 1;
    }

    fseek(f, 0, SEEK_END);
    ctx.size = ftell(f);
    fseek(f, 0, SEEK_SET);

    printf("Loading input file: %s (%zu bytes)\n", input_path, ctx.size);

    ctx.data = malloc(ctx.size);
    if (!ctx.data || fread(ctx.data, 1, ctx.size, f) != ctx.size) {
        fprintf(stderr, "Error: Cannot read file: %s\n", strerror(errno));
        fclose(f);
        return 1;
    }
    fclose(f);

    if (scan_ptx_entries(&ctx) != 0) {
        free(ctx.data);
        return 1;
    }

    if (ctx.entry_count == 0) {
        printf("No PTX entries found in input file.\n");
        free(ctx.entries);
        free(ctx.data);
        return 0;
    }

    int extracted = extract_ptx_entries(&ctx, output_dir);

    if (extracted < 0) {
        fprintf(stderr, "Error: Extraction failed\n");
        free(ctx.entries);
        free(ctx.data);
        return 1;
    }

    print_statistics(&ctx);

    printf("Extraction complete!\n");
    printf("Successfully extracted: %d/%d PTX entries\n", extracted, ctx.entry_count);
    printf("Output directory: %s\n", output_dir);

    free(ctx.entries);
    free(ctx.data);
    return 0;
}
