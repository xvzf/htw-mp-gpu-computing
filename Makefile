.PHONY: build
build:
	bazel build //...

build-debug:
	bazel build --config=debug //...

.PHONY: clean
clean:
	bazel clean --expunge

.PHONY: test
test:
	bazel test //...