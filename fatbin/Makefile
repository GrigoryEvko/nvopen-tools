# NVIDIA Fat Binary Tools - Makefile
# ====================================

CC := gcc
CFLAGS := -O2 -Wall -Wextra -std=c11
LDFLAGS := -lzstd

# Targets
TARGET := fatbin_dump

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(TARGET).c
	@echo "Building $(TARGET)..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ Built: $(TARGET)"

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	@echo "✓ Clean complete"

test: $(TARGET)
	@echo "Running basic tests..."
	@if [ ! -f /tmp/test_fatbin.bin ]; then \
		echo "Error: No test file at /tmp/test_fatbin.bin"; \
		exit 1; \
	fi
	./$(TARGET) /tmp/test_fatbin.bin --list-elf
	@echo "✓ Tests passed"

install: $(TARGET)
	@echo "Installing to /usr/local/bin..."
	sudo cp $(TARGET) /usr/local/bin/
	@echo "✓ Installed: /usr/local/bin/$(TARGET)"

help:
	@echo "NVIDIA Fat Binary Tools"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all tools (default)"
	@echo "  clean    - Remove built binaries"
	@echo "  test     - Run basic tests"
	@echo "  install  - Install to /usr/local/bin"
	@echo "  help     - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  make"
	@echo "  ./$(TARGET) nv_fatbin.bin --list-elf"
