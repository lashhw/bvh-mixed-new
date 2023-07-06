#ifndef BVH_MIXED_NEW_MARKER_HPP
#define BVH_MIXED_NEW_MARKER_HPP

typedef bvh::Bvh<float> bvh_t;
typedef bvh::BoundingBox<float> bbox_t;

struct Marker {
    mpfr_t tmp{};
    mpfr_exp_t exp_min;
    mpfr_exp_t exp_max;

    Marker(mpfr_prec_t mantissa_width, mpfr_exp_t exponent_width) {
        mpfr_init2(tmp, mantissa_width + 1);
        exp_min = -(1 << (exponent_width - 1)) + 2;
        exp_max = (1 << (exponent_width - 1));
    }

    ~Marker() {
        mpfr_clear(tmp);
    }

    void check_exponent_and_set_inf(mpfr_t &num) const {
        if (mpfr_number_p(num)) {
            if (mpfr_get_exp(num) < exp_min) mpfr_set_zero(num, mpfr_signbit(num) ? -1 : 1);
            else if (mpfr_get_exp(num) > exp_max) mpfr_set_inf(num, mpfr_signbit(num) ? -1 : 1);
        }
    }

    float low_half_area(const bvh::BoundingBox<float> &bbox, int r) {  // r -> (MSB)zmax-zmin-ymax-ymin-xmax-xmin(LSB)
        bvh::BoundingBox<float> bbox_low{};
        for (int i = 0; i < 3; i++) {
            if ((r >> (i * 2)) & 1) {
                mpfr_set_flt(tmp, bbox.min[i], MPFR_RNDD);
                check_exponent_and_set_inf(tmp);
                bbox_low.min[i] = mpfr_get_flt(tmp, MPFR_RNDD);
            }
            if ((r >> (i * 2 + 1)) & 1) {
                mpfr_set_flt(tmp, bbox.max[i], MPFR_RNDU);
                check_exponent_and_set_inf(tmp);
                bbox_low.max[i] = mpfr_get_flt(tmp, MPFR_RNDU);
            }
        }
        return bbox_low.half_area();
    }

    void mark(bvh_t &bvh, float t_bbox_hp, float t_bbox_lp) {
        std::vector<int> r(bvh.node_count);  // r[child]: which planes are shared b/t parent and child
        std::vector<float> c_hp(bvh.node_count);  // c_hp[node]: optimal cost when node is in hp
        std::vector<std::array<float, 64>> c_lp(bvh.node_count);  // c_lp[node][bitmask]: optimal cost when node is in
                                                                  // lp and which planes are in hp is indicated by bitmask
        std::vector<std::array<bool, 2>> p_hp(bvh.node_count);  // c_hp[node]: (left_lp, right_lp)
        std::vector<std::array<std::array<bool, 2>, 64>> p_lp(bvh.node_count);  // c_lp[node][bitmask]: (left_lp, right_lp)

        std::stack<std::pair<size_t, bool>> stk_1;
        stk_1.emplace(0, true);
        while (!stk_1.empty()) {
            auto [curr_idx, first] = stk_1.top();
            stk_1.pop();

            size_t left_idx = bvh.nodes[curr_idx].first_child_or_primitive;
            bbox_t bbox = bvh.nodes[curr_idx].bounding_box_proxy().to_bounding_box();

            if (first) {
                if (bvh.nodes[curr_idx].is_leaf()) {
                    c_hp[curr_idx] = bbox.half_area() * (float)bvh.nodes[curr_idx].primitive_count;
                    for (int i = 0; i < 64; i++)
                        c_lp[curr_idx][i] = low_half_area(bbox, i) * (float)bvh.nodes[curr_idx].primitive_count;
                } else {
                    stk_1.emplace(curr_idx, false);
                    for (int i = 0; i <= 1; i++) {
                        size_t child_idx = left_idx + i;
                        stk_1.emplace(child_idx, true);
                        r[child_idx] = 0;
                        for (int j = 0; j < 6; j++)
                            if (bvh.nodes[child_idx].bounds[j] == bvh.nodes[curr_idx].bounds[j])
                                r[child_idx] |= (1 << j);
                    }
                }
            } else {
                // update c_hp
                c_hp[curr_idx] = 0;
                for (int i = 0; i <= 1; i++) {
                    size_t child_idx = left_idx + i;
                    float c_hp_hp = bbox.half_area() * t_bbox_hp + c_hp[child_idx];
                    float c_hp_lp = bbox.half_area() * t_bbox_lp + c_lp[child_idx][r[child_idx]];
                    if (c_hp_lp < c_hp_hp) {
                        c_hp[curr_idx] += c_hp_lp;
                        p_hp[curr_idx][i] = true;
                    } else {
                        c_hp[curr_idx] += c_hp_hp;
                        p_hp[curr_idx][i] = false;
                    }
                }

                // update c_lp
                for (int i = 0; i < 64; i++) {
                    c_lp[curr_idx][i] = 0;
                    for (int j = 0; j <= 1; j++) {
                        size_t child_idx = left_idx + j;
                        float c_lp_hp = low_half_area(bbox, i) * t_bbox_hp + c_hp[child_idx];
                        float c_lp_lp = low_half_area(bbox, i) * t_bbox_lp + c_lp[child_idx][i & r[child_idx]];
                        if (c_lp_lp < c_lp_hp) {
                            c_lp[curr_idx][i] += c_lp_lp;
                            p_lp[curr_idx][i][j] = true;
                        } else {
                            c_lp[curr_idx][i] += c_lp_hp;
                            p_lp[curr_idx][i][j] = false;
                        }
                    }
                }
            }
        }

        std::stack<std::tuple<size_t, bool, int>> stk_2;
        stk_2.emplace(0, false, 0b111111);
        while (!stk_2.empty()) {
            auto [curr_idx, lp, bitmask] = stk_2.top();  // bitmask: which plane is in hp
            stk_2.pop();

            bvh.nodes[curr_idx].low_precision = lp;
            if (bvh.nodes[curr_idx].is_leaf())
                continue;

            size_t left_idx = bvh.nodes[curr_idx].first_child_or_primitive;

            for (int i = 0; i <= 1; i++) {
                size_t child_idx = left_idx + i;
                if (lp) {
                    if (p_lp[curr_idx][bitmask][i])
                        stk_2.emplace(child_idx, true, bitmask & r[child_idx]);
                    else
                        stk_2.emplace(child_idx, false, 0b111111);
                } else {
                    if (p_hp[curr_idx][i])
                        stk_2.emplace(child_idx, true, r[child_idx]);
                    else
                        stk_2.emplace(child_idx, false, 0b111111);
                }
            }
        }
    }
};

#endif //BVH_MIXED_NEW_MARKER_HPP
