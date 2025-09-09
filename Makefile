# Compiler and flags
NVCC       = nvcc
NVCCFLAGS  = -std=c++17 -allow-unsupported-compiler
LDFLAGS    = -lcublas

# Directories
INCLUDE_DIR = include
KERNELS_DIR = kernels
BUILD_DIR   = build

# Source files
KERNEL_SOURCES = $(wildcard $(KERNELS_DIR)/*.cu)
ROOT_SOURCES   = src.cu
CUDA_SOURCES   = $(KERNEL_SOURCES) $(ROOT_SOURCES)

# Object files
CUDA_OBJECTS = $(patsubst $(KERNELS_DIR)/%.cu,$(BUILD_DIR)/%.o,$(KERNEL_SOURCES)) \
               $(BUILD_DIR)/src.o

# Output executable
OUTPUT = matvec.exe

# Default rule
all: $(OUTPUT)

# Link step
$(OUTPUT): $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Compile kernels
$(BUILD_DIR)/%.o: $(KERNELS_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile root source
$(BUILD_DIR)/src.o: src.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean
clean:
	rm -f $(BUILD_DIR)/*.o $(OUTPUT)

.PHONY: all clean
