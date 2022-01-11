.PHONY: build
build:
	bazel build -s //...

.PHONY: clean
clean:
	bazel clean --expunge

.PHONY: test
test:
	bazel test //...