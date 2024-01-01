CC = gcc
CFLAGS = -Iinclude -fPIC
LDFLAGS = -shared
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
DLL_NAME = $(BIN_DIR)/mylibrary.dll
EXE_NAME = $(BIN_DIR)/testapp.exe
TEST_DIR = test
TEST_SRC = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJ = $(TEST_SRC:$(TEST_DIR)/%.c=$(OBJ_DIR)/%.o)

# Collect all .c files in SRC_DIR and names of corresponding .o files in OBJ_DIR
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Default target
all: $(DLL_NAME) $(EXE_NAME)

# Rule to create the DLL
$(DLL_NAME): $(OBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) -o $@ $^

# Rule to create the test executable
$(EXE_NAME): $(TEST_OBJ) | $(DLL_NAME)
	$(CC) -o $@ $^ -L$(BIN_DIR) -lmylibrary

# Rule to create object files for both DLL and test executable
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to create the object file directory
$(OBJ_DIR):
	mkdir -p $@

# Rule to create the binary directory
$(BIN_DIR):
	mkdir -p $@

# Clean rule
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run target
run: $(EXE_NAME)
	./$(EXE_NAME)

# Debug target
debug: $(EXE_NAME)
	gdb $(EXE_NAME)

# Phony targets
.PHONY: all clean
