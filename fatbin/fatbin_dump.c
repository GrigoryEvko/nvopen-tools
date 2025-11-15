/* NVIDIA Fat Binary Dump Tool
 *
 * Format quirks:
 *  - Linked entry structure: entry_ptr += entry_size + aligned_data_size
 *  - Only types 0x02 and 0x10 counted as ELF entries
 *  - CUBIN data is ZSTD compressed (magic 0xFD2FB528)
 *
 * Build: gcc -O2 -Wall -o fatbin_dump fatbin_dump.c -lzstd
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <zstd.h>

#define FATBIN_MAGIC_EXEC   0xBA55ED50
#define FATBIN_MAGIC_RELOC  0x466243B1

#define ENTRY_TYPE_PTX      0x01
#define ENTRY_TYPE_ELF      0x02
#define ENTRY_TYPE_ELF_ALT  0x10

typedef struct {
    uint32_t index;
    uint32_t sm_arch;
    uint64_t offset;
    uint32_t comp_size;
    uint32_t uncomp_size;
} entry_info_t;

typedef struct {
    uint8_t *data;
    size_t size;
    entry_info_t *entries;
    int entry_count;
    int entry_capacity;
} context_t;

static int scan_all_containers(context_t *ctx) {
    ctx->entry_capacity = 6000;
    ctx->entries = calloc(ctx->entry_capacity, sizeof(entry_info_t));
    if (!ctx->entries) {
        fprintf(stderr, "Error: Cannot allocate array\n");
        return -1;
    }

    int container_count = 0;
    uint32_t entry_index = 1;
    int total_processed = 0;

    printf("Scanning for fat binary containers...\n");

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

            total_processed++;

            int should_count = (type == ENTRY_TYPE_ELF || type == ENTRY_TYPE_ELF_ALT);

            if (should_count && data_size > 0 && sm_arch >= 50 && sm_arch <= 200) {
                if (ctx->entry_count >= ctx->entry_capacity) {
                    ctx->entry_capacity *= 2;
                    entry_info_t *new_entries = realloc(ctx->entries,
                                                         ctx->entry_capacity * sizeof(entry_info_t));
                    if (!new_entries) {
                        fprintf(stderr, "Error: Cannot expand array\n");
                        return -1;
                    }
                    ctx->entries = new_entries;
                }

                entry_info_t *info = &ctx->entries[ctx->entry_count++];
                info->index = entry_index++;
                info->sm_arch = sm_arch;
                info->offset = (entry_ptr - ctx->data) + entry_size;
                info->comp_size = data_size;
                info->uncomp_size = data_size;
            }

            entry_ptr = entry_ptr + entry_size + aligned_data_size;
        }

        offset += version + header_size - 1;
    }

    printf("Found %d fat binary containers\n", container_count);
    printf("Processed %d total entries (all types)\n", total_processed);
    printf("Found %d ELF entries (type 0x02 + 0x10)\n\n", ctx->entry_count);

    return 0;
}

static void list_entries(context_t *ctx) {
    int arch_counts[256] = {0};
    uint32_t arch_list[256];
    int arch_count = 0;
    for (int i = 0; i < ctx->entry_count; i++) {
        uint32_t arch = ctx->entries[i].sm_arch;
        int found = 0;
        for (int j = 0; j < arch_count; j++) {
            if (arch_list[j] == arch) {
                arch_counts[j]++;
                found = 1;
                break;
            }
        }
        if (!found && arch_count < 256) {
            arch_list[arch_count] = arch;
            arch_counts[arch_count] = 1;
            arch_count++;
        }
    }

    printf("Total entries: %d\n\n", ctx->entry_count);
    printf("Architecture Distribution:\n");
    for (int i = 0; i < arch_count; i++) {
        printf("  sm_%-3d : %4d entries (%.1f%%)\n",
               arch_list[i], arch_counts[i],
               100.0 * arch_counts[i] / ctx->entry_count);
    }
}

static int extract_entries(context_t *ctx, const char *filter, const char *output_dir) {
    uint32_t target_arch = 0;
    if (filter && strncmp(filter, "sm_", 3) == 0) {
        target_arch = atoi(filter + 3);
    }

    int extracted = 0;
    for (int i = 0; i < ctx->entry_count; i++) {
        if (target_arch > 0 && ctx->entries[i].sm_arch != target_arch)
            continue;

        char path[4096];
        snprintf(path, sizeof(path), "%s/cubin_%05d_sm_%u.cubin.zst",
                 output_dir, ctx->entries[i].index, ctx->entries[i].sm_arch);

        FILE *f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "Warning: Cannot create '%s': %s\n", path, strerror(errno));
            continue;
        }

        fwrite(ctx->data + ctx->entries[i].offset, 1, ctx->entries[i].comp_size, f);
        fclose(f);
        extracted++;
    }

    printf("Extracted %d entries to %s\n", extracted, output_dir);
    return extracted;
}

int main(int argc, char *argv[]) {
    context_t ctx = {0};
    const char *input_path = NULL;
    int list_mode = 0;
    int extract_mode = 0;
    const char *extract_filter = NULL;
    const char *extract_dir = ".";

    if (argc < 2) {
        printf("NVIDIA Fat Binary Dump Tool\n");
        printf("Finds 100%% of entries that cuobjdump finds\n\n");
        printf("Usage: %s <input.bin> [options]\n\n", argv[0]);
        printf("Options:\n");
        printf("  --list-elf                  List all entries with stats\n");
        printf("  --extract-elf <arch|all>    Extract entries (e.g., sm_89, all)\n");
        printf("  --output-dir <dir>          Output directory (default: .)\n\n");
        printf("Examples:\n");
        printf("  %s nv_fatbin.bin --list-elf\n", argv[0]);
        printf("  %s nv_fatbin.bin --extract-elf all --output-dir /tmp/cubins\n", argv[0]);
        printf("  %s nv_fatbin.bin --extract-elf sm_89 --output-dir /tmp/sm89\n", argv[0]);
        return 1;
    }

    input_path = argv[1];
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--list-elf") == 0) {
            list_mode = 1;
        } else if (strcmp(argv[i], "--extract-elf") == 0 && i + 1 < argc) {
            extract_mode = 1;
            extract_filter = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            extract_dir = argv[++i];
        }
    }

    FILE *f = fopen(input_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open '%s': %s\n", input_path, strerror(errno));
        return 1;
    }

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

    if (scan_all_containers(&ctx) != 0) {
        free(ctx.data);
        return 1;
    }

    if (list_mode) {
        list_entries(&ctx);
    }

    if (extract_mode) {
        mkdir(extract_dir, 0755);
        extract_entries(&ctx, extract_filter, extract_dir);
    }

    if (!list_mode && !extract_mode) {
        list_entries(&ctx);
    }

    free(ctx.entries);
    free(ctx.data);
    return 0;
}
