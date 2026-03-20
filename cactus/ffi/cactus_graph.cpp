#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../graph/graph.h"

using namespace cactus::ffi; // for last_error_message

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

    int cactus_graph_input(cactus_graph_t graph, const size_t* shape, size_t rank,
    int32_t precision, cactus_node_t* out_node) {
        if (!graph || !shape || rank == 0 || !out_node) {
            last_error_message = "Invalid args to cactus_graph_input";
            return -1;
        }
        try {
            std::vector<size_t> s(shape, shape + rank);
            auto id = as_graph(graph)->graph.input(s,
    static_cast<Precision>(precision));
            *out_node = static_cast<cactus_node_t>(id);
            return 0;
        } catch (const std::exception& e) {
            last_error_message = e.what();
            return -1;
        }
    }

    int cactus_graph_abs(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
        if (!graph || !out) { last_error_message = "Invalid args to cactus_graph_abs"; return -1; }
        try {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.abs(static_cast<size_t>(x)));
            return 0;
        } catch (const std::exception& e) {
            last_error_message = e.what();
            return -1;
        }
    }

    int cactus_graph_pow(cactus_graph_t graph, cactus_node_t x, float exponent,
    cactus_node_t* out) {
        if (!graph || !out) { last_error_message = "Invalid args to cactus_graph_pow"; return -1; }
        try {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.pow(static_cast<size_t>(x), exponent));
            return 0;
        } catch (const std::exception& e) {
            last_error_message = e.what();
            return -1;
        }
    }

    int cactus_graph_execute(cactus_graph_t graph) {
        if (!graph) { last_error_message = "Graph is null"; return -1; }
        try {
            as_graph(graph)->graph.execute();
            return 0;
        } catch (const std::exception& e) {
            last_error_message = e.what();
            return -1;
        }
    }

    int cactus_graph_get_output_ptr(cactus_graph_t graph, cactus_node_t node, void** out_ptr) {
        if (!graph || !out_ptr) { last_error_message = "Invalid args to cactus_graph_get_output_ptr"; return -1; }
        try {
            *out_ptr = as_graph(graph)->graph.get_output(static_cast<size_t>(node));
            return 0;
        } catch (const std::exception& e) {
            last_error_message = e.what();
            return -1;
        }
    }
}
