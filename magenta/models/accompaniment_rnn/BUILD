# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description: An RNN model for generating accompanying melodies.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "accompaniment_rnn_config",
    srcs = ["accompaniment_rnn_config.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:accompaniment_rnn
    ],
    deps = [
        ":accompaniment_rnn_encoder_decoder",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "accompaniment_rnn_encoder_decoder",
    srcs = ["accompaniment_rnn_encoder_decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
    ],
)

py_library(
    name = "accompaniment_rnn_graph",
    srcs = ["accompaniment_rnn_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "accompaniment_rnn_graph_test",
    srcs = ["accompaniment_rnn_graph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":accompaniment_rnn_config",
        ":accompaniment_rnn_graph",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "accompaniment_rnn_generator",
    srcs = ["accompaniment_rnn_generator.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/interfaces:generator_interfaces"],
    deps = [
        ":accompaniment_rnn_config",
        ":accompaniment_rnn_graph",
        "//magenta",
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "accompaniment_rnn_generate",
    srcs = ["accompaniment_rnn_generate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":accompaniment_rnn_config",
        ":accompaniment_rnn_generator",
        "//magenta",
        # tensorflow dep
    ],
)