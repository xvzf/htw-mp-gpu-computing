.PHONY: build
build:
	bazel build //...

build-cuda:
	bazel build --config=cuda //...

build-debug:
	bazel build --config=debug //...

build-cuda-debug:
	bazel build --config=cuda-debug //...

.PHONY: clean
clean:
	bazel clean --expunge

.PHONY: test
test:
	bazel test //...
