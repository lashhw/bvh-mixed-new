#include <iostream>
#include <mpfr.h>
#include <bvh/triangle.hpp>
#include <bvh/bvh.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>
#include "happly.h"
#include "mark.hpp"

typedef bvh::Bvh<float> bvh_t;
typedef bvh::Triangle<float> triangle_t;
typedef bvh::Vector3<float> vector_t;
typedef bvh::Ray<float> ray_t;
typedef bvh::BoundingBox<float> bbox_t;
typedef bvh_t::Node node_t;
typedef bvh::SweepSahBuilder<bvh_t> builder_t;
typedef bvh::SingleRayTraverser<bvh_t> traverser_t;
typedef bvh::ClosestPrimitiveIntersector<bvh_t, triangle_t> intersector_t;
typedef traverser_t::Statistics statistics_t;

const mpfr_prec_t mantissa_width = 7;
const mpfr_exp_t exponent_width = 8;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "usage: ./bvh_mixed_new MODEL_FILE RAY_FILE" << std::endl;
        exit(EXIT_FAILURE);
    }

    char *model_file = argv[1];
    char *ray_file = argv[2];

    std::cout << "model_file = " << model_file << std::endl;
    std::cout << "ray_file = " << ray_file << std::endl;

    happly::PLYData ply_data(model_file);
    std::vector<std::array<double, 3>> v_pos = ply_data.getVertexPositions();
    std::vector<std::vector<size_t>> f_idx = ply_data.getFaceIndices<size_t>();

    std::vector<triangle_t> triangles;
    for (auto &face : f_idx) {
        triangles.emplace_back(vector_t(v_pos[face[0]][0], v_pos[face[0]][1], v_pos[face[0]][2]),
                               vector_t(v_pos[face[1]][0], v_pos[face[1]][1], v_pos[face[1]][2]),
                               vector_t(v_pos[face[2]][0], v_pos[face[2]][1], v_pos[face[2]][2]));
    }

    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());
    std::cout << "global_bbox = ("
              << global_bbox.min[0] << ", " << global_bbox.min[1] << ", " << global_bbox.min[2] << "), ("
              << global_bbox.max[0] << ", " << global_bbox.max[1] << ", " << global_bbox.max[2] << ")" << std::endl;

    std::cout << "building..." << std::endl;
    bvh_t bvh;
    builder_t builder(bvh);
    builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());
}
