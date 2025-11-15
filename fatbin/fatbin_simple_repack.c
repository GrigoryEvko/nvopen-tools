/*
 * fatbin_simple_repack.c - Pack cubin files into NVIDIA fat binary
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <getopt.h>
#include "fatbin_pack.c"

static uint8_t* read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc(*size);
    if (data && fread(data, 1, *size, f) != *size) {
        free(data);
        data = NULL;
    }
    fclose(f);
    return data;
}

static int write_file(const char *path, const void *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t written = fwrite(data, 1, size, f);
    fclose(f);
    return (written == size) ? 0 : -1;
}

static int extract_sm_arch(const char *filename) {
    const char *sm_str = strstr(filename, "sm_");
    if (!sm_str) {
        sm_str = strstr(filename, "sm");
        if (!sm_str) return 0;
    }
    int arch = 0;
    if (sm_str[2] == '_') {
        sscanf(sm_str, "sm_%d", &arch);
    } else {
        sscanf(sm_str, "sm%d", &arch);
    }
    return arch;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [-c LEVEL] <output.bin> <file1.cubin> ...\n", prog);
    fprintf(stderr, "  -c LEVEL  Compression level 1-22 (default: 3)\n");
}

int main(int argc, char **argv) {
    int compression_level = 3;
    int opt;
    while ((opt = getopt(argc, argv, "c:h")) != -1) {
        switch (opt) {
            case 'c':
                compression_level = atoi(optarg);
                if (compression_level < 1 || compression_level > 22) {
                    fprintf(stderr, "Error: Compression level must be 1-22\n");
                    return 1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    if (optind + 2 > argc) {
        fprintf(stderr, "Error: Need output file and at least one input file\n");
        print_usage(argv[0]);
        return 1;
    }
    const char *output_path = argv[optind];
    int first_input = optind + 1;
    int num_inputs = argc - first_input;
    printf("Packing %d files (compression: %d)\n", num_inputs, compression_level);
    nvFatbinHandle handle;
    const char *options[2];
    char level_opt[32];
    snprintf(level_opt, sizeof(level_opt), "-compress-level=%d", compression_level);
    options[0] = level_opt;
    options[1] = NULL;
    nvFatbinResult res = nvFatbinCreate(&handle, options, 1);
    if (res != NVFATBIN_SUCCESS) {
        fprintf(stderr, "nvFatbinCreate failed: %s\n", nvFatbinGetErrorString(res));
        return 1;
    }
    int added = 0;
    size_t total_input_size = 0;
    for (int i = first_input; i < argc; i++) {
        const char *cubin_path = argv[i];
        size_t cubin_size;
        uint8_t *cubin_data = read_file(cubin_path, &cubin_size);
        if (!cubin_data) {
            fprintf(stderr, "Warning: Cannot read %s\n", cubin_path);
            continue;
        }
        total_input_size += cubin_size;
        int sm_arch = extract_sm_arch(cubin_path);
        if (sm_arch == 0) {
            fprintf(stderr, "Warning: Cannot determine SM arch from %s\n", cubin_path);
            free(cubin_data);
            continue;
        }
        char arch_str[16];
        snprintf(arch_str, sizeof(arch_str), "sm_%d", sm_arch);
        int is_ptx = 0;
        if (cubin_size > 4) {
            if (cubin_data[0] == 0x7f && cubin_data[1] == 'E' &&
                cubin_data[2] == 'L' && cubin_data[3] == 'F') {
                is_ptx = 0;
            } else {
                is_ptx = 1;
            }
        }
        if (is_ptx) {
            res = nvFatbinAddPTX(handle, (const char*)cubin_data, cubin_size,
                                arch_str, NULL, NULL);
        } else {
            res = nvFatbinAddCubin(handle, cubin_data, cubin_size,
                                  arch_str, NULL);
        }
        free(cubin_data);
        if (res != NVFATBIN_SUCCESS) {
            fprintf(stderr, "Warning: Failed to add %s: %s\n",
                    cubin_path, nvFatbinGetErrorString(res));
            continue;
        }
        added++;
        if (added % 100 == 0) {
            printf("Added %d entries...\n", added);
        }
    }
    printf("Added %d entries (%.2f MB)\n", added, total_input_size / (1024.0 * 1024.0));
    if (added == 0) {
        fprintf(stderr, "Error: No entries added\n");
        nvFatbinDestroy(&handle);
        return 1;
    }
    size_t fatbin_size;
    res = nvFatbinSize(handle, &fatbin_size);
    if (res != NVFATBIN_SUCCESS) {
        fprintf(stderr, "nvFatbinSize failed: %s\n", nvFatbinGetErrorString(res));
        nvFatbinDestroy(&handle);
        return 1;
    }
    void *fatbin_data = malloc(fatbin_size);
    if (!fatbin_data) {
        fprintf(stderr, "Out of memory for fat binary (%zu bytes)\n", fatbin_size);
        nvFatbinDestroy(&handle);
        return 1;
    }
    res = nvFatbinGet(handle, fatbin_data);
    if (res != NVFATBIN_SUCCESS) {
        fprintf(stderr, "nvFatbinGet failed: %s\n", nvFatbinGetErrorString(res));
        free(fatbin_data);
        nvFatbinDestroy(&handle);
        return 1;
    }
    if (write_file(output_path, fatbin_data, fatbin_size) != 0) {
        fprintf(stderr, "Error: Cannot write output file: %s\n", output_path);
        free(fatbin_data);
        nvFatbinDestroy(&handle);
        return 1;
    }
    double compression_ratio = (1.0 - ((double)fatbin_size / total_input_size)) * 100.0;
    printf("Wrote %s (%.2f MB, %.1f%% compression)\n",
           output_path, fatbin_size / (1024.0 * 1024.0), compression_ratio);
    free(fatbin_data);
    nvFatbinDestroy(&handle);
    return 0;
}
