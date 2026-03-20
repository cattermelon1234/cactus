#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../graph/graph.h"
#include <vector>

using namespace cactus::ffi;

struct GraphHandle {
    CactusGraph graph;
};

static GraphHandle* as_graph(cactus_graph_t g) {
    return reinterpret_cast<GraphHandle*>(g);
}

extern "C" {

cactus_graph_t cactus_graph_create(void) {
    try {
        return reinterpret_cast<cactus_graph_t>(new GraphHandle());
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown error creating graph";
        return nullptr;
    }
}

void cactus_graph_destroy(cactus_graph_t graph) {
    delete as_graph(graph);
}

int cactus_graph_hard_reset(cactus_graph_t graph) {
    if (!graph) {
        last_error_message = "Invalid args to cactus_graph_hard_reset";
        return -1;
    }
    try {
        as_graph(graph)->graph.hard_reset();
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_input(cactus_graph_t graph, const size_t* shape, size_t rank, int32_t precision, cactus_node_t* out_node) {
    if (!graph || !shape || rank == 0 || !out_node) {
        last_error_message = "Invalid args to cactus_graph_input";
        return -1;
    }
    try {
        std::vector<size_t> s(shape, shape + rank);
        auto id = as_graph(graph)->graph.input(s, static_cast<Precision>(precision));
        *out_node = static_cast<cactus_node_t>(id);
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_set_input(cactus_graph_t graph, cactus_node_t node, const void* data, int32_t precision) {
    if (!graph || !data) {
        last_error_message = "Invalid args to cactus_graph_set_input";
        return -1;
    }
    try {
        as_graph(graph)->graph.set_input(static_cast<size_t>(node), data, static_cast<Precision>(precision));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_add(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_add";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.add(static_cast<size_t>(a), static_cast<size_t>(b)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_subtract(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_subtract";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.subtract(static_cast<size_t>(a), static_cast<size_t>(b)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_multiply(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_multiply";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.multiply(static_cast<size_t>(a), static_cast<size_t>(b)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_divide(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_divide";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.divide(static_cast<size_t>(a), static_cast<size_t>(b)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_abs(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_abs";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.abs(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_pow(cactus_graph_t graph, cactus_node_t x, float exponent, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_pow";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.pow(static_cast<size_t>(x), exponent));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_view(cactus_graph_t graph, cactus_node_t x, const size_t* shape, size_t rank, cactus_node_t* out) {
    if (!graph || !shape || rank == 0 || !out) {
        last_error_message = "Invalid args to cactus_graph_view";
        return -1;
    }
    try {
        std::vector<size_t> s(shape, shape + rank);
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.view(static_cast<size_t>(x), s));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_flatten(cactus_graph_t graph, cactus_node_t x, int32_t start_dim, int32_t end_dim, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_flatten";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.flatten(static_cast<size_t>(x), start_dim, end_dim));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_concat(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, int32_t axis, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_concat";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.concat(static_cast<size_t>(a), static_cast<size_t>(b), axis));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_cat(cactus_graph_t graph, const cactus_node_t* nodes, size_t count, int32_t axis, cactus_node_t* out) {
    if (!graph || !nodes || !out || count == 0) {
        last_error_message = "Invalid args to cactus_graph_cat";
        return -1;
    }
    try {
        size_t acc = static_cast<size_t>(nodes[0]);
        for (size_t i = 1; i < count; ++i) {
            acc = as_graph(graph)->graph.concat(acc, static_cast<size_t>(nodes[i]), axis);
        }
        *out = static_cast<cactus_node_t>(acc);
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_execute(cactus_graph_t graph) {
    if (!graph) {
        last_error_message = "Graph is null";
        return -1;
    }
    try {
        as_graph(graph)->graph.execute();
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_get_output_ptr(cactus_graph_t graph, cactus_node_t node, void** out_ptr) {
    if (!graph || !out_ptr) {
        last_error_message = "Invalid args to cactus_graph_get_output_ptr";
        return -1;
    }
    try {
        *out_ptr = as_graph(graph)->graph.get_output(static_cast<size_t>(node));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_get_output_info(cactus_graph_t graph, cactus_node_t node, cactus_tensor_info_t* out_info) {
    if (!graph || !out_info) {
        last_error_message = "Invalid args to cactus_graph_get_output_info";
        return -1;
    }
    try {
        const auto& buf = as_graph(graph)->graph.get_output_buffer(static_cast<size_t>(node));
        out_info->precision = static_cast<int32_t>(buf.precision);
        out_info->rank = buf.shape.size();
        if (out_info->rank > 8) {
            last_error_message = "Rank exceeds cactus_tensor_info_t shape capacity";
            return -1;
        }
        for (size_t i = 0; i < out_info->rank; ++i) {
            out_info->shape[i] = buf.shape[i];
        }
        for (size_t i = out_info->rank; i < 8; ++i) {
            out_info->shape[i] = 0;
        }
        out_info->num_elements = buf.total_size;
        out_info->byte_size = buf.byte_size;
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}
}
