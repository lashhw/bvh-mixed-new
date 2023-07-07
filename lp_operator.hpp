#ifndef BVH_MIXED_NEW_LP_OPERATOR_HPP
#define BVH_MIXED_NEW_LP_OPERATOR_HPP

typedef bvh::Bvh<float> bvh_t;
typedef bvh::BoundingBox<float> bbox_t;

struct LPOperator {
    mpfr_t tmp{};
    mpfr_exp_t exp_min;
    mpfr_exp_t exp_max;

    LPOperator(mpfr_prec_t mantissa_width, mpfr_exp_t exponent_width) {
        mpfr_init2(tmp, mantissa_width + 1);
        exp_min = -(1 << (exponent_width - 1)) + 2;
        exp_max = (1 << (exponent_width - 1));
    }

    ~LPOperator() {
        mpfr_clear(tmp);
    }

    void check_exponent_and_set_inf(mpfr_t &num) const {
        if (mpfr_number_p(num)) {
            if (mpfr_get_exp(num) < exp_min) mpfr_set_zero(num, mpfr_signbit(num) ? -1 : 1);
            else if (mpfr_get_exp(num) > exp_max) mpfr_set_inf(num, mpfr_signbit(num) ? -1 : 1);
        }
    }

    bbox_t lp_bbox(const bbox_t &bbox, int bitmask) {  // bitmask: (MSB)zmax-zmin-ymax-ymin-xmax-xmin(LSB), true if hp
        bbox_t lp_bbox_{};
        for (int i = 0; i < 3; i++) {
            if ((bitmask >> (i * 2)) & 1) {
                lp_bbox_.min[i] = bbox.min[i];
            } else {
                mpfr_set_flt(tmp, bbox.min[i], MPFR_RNDD);
                check_exponent_and_set_inf(tmp);
                lp_bbox_.min[i] = mpfr_get_flt(tmp, MPFR_RNDD);
            }
            if ((bitmask >> (i * 2 + 1)) & 1) {
                lp_bbox_.max[i] = bbox.max[i];
            } else {
                mpfr_set_flt(tmp, bbox.max[i], MPFR_RNDU);
                check_exponent_and_set_inf(tmp);
                lp_bbox_.max[i] = mpfr_get_flt(tmp, MPFR_RNDU);
            }
        }
        return lp_bbox_;
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
                        c_lp[curr_idx][i] = lp_bbox(bbox, i).half_area() * (float)bvh.nodes[curr_idx].primitive_count;
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
                        float c_lp_hp = lp_bbox(bbox, i).half_area() * t_bbox_hp + c_hp[child_idx];
                        float c_lp_lp = lp_bbox(bbox, i).half_area() * t_bbox_lp + c_lp[child_idx][i & r[child_idx]];
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

    float sah_cost(bvh_t &bvh, float t_bbox_hp, float t_bbox_lp) {
        // fill actual half data
        std::vector<float> actual_half_area(bvh.node_count);
        std::queue<std::pair<int, bbox_t>> q_1;
        q_1.emplace(0, bbox_t::full());
        while (!q_1.empty()) {
            auto [idx, parent_bbox] = q_1.front();
            q_1.pop();

            bbox_t tmp_bbox;
            if (bvh.nodes[idx].low_precision) {
                tmp_bbox = parent_bbox;
                tmp_bbox.shrink(lp_bbox(bvh.nodes[idx].bounding_box_proxy().to_bounding_box(), 0b000000));
                actual_half_area[idx] = tmp_bbox.half_area();
            } else {
                tmp_bbox = bvh.nodes[idx].bounding_box_proxy();
                actual_half_area[idx] = tmp_bbox.half_area();
            }
            if (!bvh.nodes[idx].is_leaf()) {
                q_1.emplace(bvh.nodes[idx].first_child_or_primitive, tmp_bbox);
                q_1.emplace(bvh.nodes[idx].first_child_or_primitive + 1, tmp_bbox);
            }
        }

        float cost = 0;
        std::queue<size_t> q_2;
        q_2.push(0);
        while (!q_2.empty()) {
            size_t curr = q_2.front();
            q_2.pop();

            bvh::Bvh<float>::Node &node = bvh.nodes[curr];
            if (node.is_leaf())
                cost += actual_half_area[curr] * (float)node.primitive_count;
            else {
                for (int i = 0; i <= 1; i++) {  // 0 for left child, 1 for right child
                    size_t child_idx = node.first_child_or_primitive + i;
                    q_2.push(child_idx);
                    bvh::Bvh<float>::Node &child_node = bvh.nodes[child_idx];
                    if (child_node.low_precision)
                        cost += actual_half_area[curr] * t_bbox_lp;
                    else
                        cost += actual_half_area[curr] * t_bbox_hp;
                }
            }
        }

        return cost / bvh.nodes[0].bounding_box_proxy().to_bounding_box().half_area();
    }
};

#endif //BVH_MIXED_NEW_LP_OPERATOR_HPP
