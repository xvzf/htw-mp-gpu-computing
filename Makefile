.PHONY: build
build:
	bazel build -s //...

build-debug:
	bazel build --config=asan //...

.PHONY: clean
clean:
	bazel clean --expunge

.PHONY: test
test:
	bazel test //...