#include "engine_helper.h"
#include "consts.h"

Eigen::Vector3d engine_helper::project_point(const Eigen::Vector3d p1,
                                             const Eigen::Vector3d cam_u,
                                             const Eigen::Vector3d cam_v,
                                             const Eigen::Vector3d cam_w,
                                             const Eigen::Vector3d origin) {
    Eigen::Vector3d translated = p1 - origin;
    double x_cam = translated.dot(cam_u);
    double y_cam = translated.dot(cam_v);
    double z_cam = -translated.dot(cam_w);
    return Eigen::Vector3d(x_cam, y_cam, z_cam);
}

raw_tri engine_helper::proj_tri(const raw_tri &tri, const Eigen::Vector3d cam_u,
                                const Eigen::Vector3d cam_v,
                                const Eigen::Vector3d cam_w,
                                const Eigen::Vector3d origin) {
    Eigen::Vector3d p1 = tri.p1;
    Eigen::Vector3d p2 = tri.p2;
    Eigen::Vector3d p3 = tri.p3;
    Eigen::Vector3d proj_1 = project_point(p1, cam_u, cam_v, cam_w, origin);
    Eigen::Vector3d proj_2 = project_point(p2, cam_u, cam_v, cam_w, origin);
    Eigen::Vector3d proj_3 = project_point(p3, cam_u, cam_v, cam_w, origin);
    return raw_tri{proj_1, proj_2, proj_3};
};

bound_box<double> engine_helper::w_box(const Eigen::Vector3d p1,
                                       const Eigen::Vector3d p2,
                                       const Eigen::Vector3d p3,
                                       const double aspect_ratio) {
    double hor_min = std::min({p1[0], p2[0], p3[0]});
    double ver_min = std::min({p1[1], p2[1], p3[1]});
    double hor_max = std::max({p1[0], p2[0], p3[0]});
    double ver_max = std::max({p1[1], p2[1], p3[1]});
    // image planes are bound by the box: [-aspect_ratio, aspect_ratio, -1, 1]
    if (hor_min < -aspect_ratio) {
        hor_min = -aspect_ratio;
    }
    if (hor_max > aspect_ratio) {
        hor_max = aspect_ratio;
    }
    if (ver_min < -1) {
        ver_min = -1;
    }
    if (ver_max > 1) {
        ver_max = 1;
    }
    bound_box<double> bbox{hor_min, hor_max, ver_min, ver_max};
    return bbox;
}

bound_box<int>
engine_helper::create_box(const Eigen::Vector3d p1, const Eigen::Vector3d p2,
                          const Eigen::Vector3d p3, const double aspect_ratio,
                          const int img_length, const int img_width) {
    bound_box<double> w_bbox = w_box(p1, p2, p3, aspect_ratio);
    int left_pixel =
        std::floor(0.5 * (w_bbox.min_x / aspect_ratio + 1.0) * img_length);
    int right_pixel =
        std::ceil(0.5 * (w_bbox.max_x / aspect_ratio + 1.0) * img_length);
    int top_pixel = std::floor((1.0 - 0.5 * (w_bbox.max_y + 1.0)) * img_width);
    int bot_pixel = std::ceil((1.0 - 0.5 * (w_bbox.min_y + 1.0)) * img_width);
    return bound_box<int>{left_pixel, right_pixel, top_pixel, bot_pixel};
}

double engine_helper::edge_func(const Eigen::Vector3d &a,
                                const Eigen::Vector3d &b,
                                const Eigen::Vector3d &p) {
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0]);
}

Eigen::Vector3d engine_helper::get_bary(const Eigen::Vector3d &p1,
                                        const Eigen::Vector3d &p2,
                                        const Eigen::Vector3d &p3,
                                        const Eigen::Vector3d &test_pt) {
    double w1 = edge_func(p2, p3, test_pt);
    double w2 = edge_func(p3, p1, test_pt);
    double w3 = edge_func(p1, p2, test_pt);
    double total_area = w1 + w2 + w3;
    if (total_area == 0) {
        return {-1.0, -1.0, -1.0};
    }
    bool all_pos = w1 > 0.0 && w2 > 0.0 && w3 > 0.0;
    bool all_neg = w1 < 0.0 && w2 < 0.0 && w3 < 0.0;
    if (all_neg || all_pos) {
        double inv_area = 1.0 / total_area;
        w1 = std::abs(w1);
        w2 = std::abs(w2);
        w3 = std::abs(w3);
        return {w1 * inv_area, w2 * inv_area, w3 * inv_area};
    }
    return {-1.0, -1.0, -1.0};
}

template <typename T>
void engine_helper::pull(buffer<T> &src, buffer<T> &dest, const int offset_x,
                         const int offset_y) {
    const int sqrt_tile = dest.get_length_p();
    const int tile_w = dest.get_width_p();
    if (sqrt_tile != tile_w) {
        std::runtime_error("src is a tile, it must be square");
    }
    const int sqrt_samples = dest.get_sqrt_samples();
    const int buff_sqrt_samples = src.get_sqrt_samples();
    if (sqrt_samples != buff_sqrt_samples) {
        std::runtime_error("src and dest must have same num_samples");
    }
    const int buff_len_p = src.get_length_p();
    const int buff_wid_p = src.get_width_p();
    if (offset_x > buff_len_p || offset_y > buff_len_p) {
        std::runtime_error("offset out of range");
    }
    if (buff_len_p < sqrt_tile || buff_wid_p < sqrt_tile) {
        std::runtime_error("src must be smaller than buffer");
    }
    const int s_offset_x = offset_x * sqrt_samples;
    const int s_offset_y = offset_y * sqrt_samples;
    const int rem_x = buff_len_p - offset_x;
    const int rem_y = buff_wid_p - offset_y;
    if (rem_x < sqrt_tile || rem_y < sqrt_tile) {
        for (int i = 0; i < rem_x * sqrt_samples; i++) {
            for (int j = 0; j < rem_y * sqrt_samples; j++) {
                dest.set(i, j, src.get(s_offset_x + i, s_offset_y + j));
            }
        }
    }
    for (int i = 0; i < sqrt_tile * sqrt_samples; i++) {
        for (int j = 0; j < sqrt_tile * sqrt_samples; j++) {
            dest.set(i, j, src.get(s_offset_x + i, s_offset_y + j));
        }
    }
}

template <typename T>
void push(buffer<T> &src, buffer<T> &dest, const int offset_x,
          const int offset_y) {
    const int sqrt_tile = src.get_length_p();
    const int tile_w = src.get_width_p();
    constexpr int TILE_SIZE = consts::TILE_SIZE;
    if (sqrt_tile != tile_w || sqrt_tile == TILE_SIZE) {
        std::runtime_error("src is a tile, it must be square");
    }
    const int sqrt_samples = src.get_sqrt_samples();
    const int buff_sqrt_samples = dest.get_sqrt_samples();
    if (sqrt_samples != buff_sqrt_samples) {
        std::runtime_error("src and dest must have same num_samples");
    }
    const int buff_len_p = dest.get_length_p();
    const int buff_wid_p = dest.get_width_p();
    if (offset_x > buff_len_p || offset_y > buff_len_p) {
        std::runtime_error("offset out of range");
    }
    if (buff_len_p < sqrt_tile || buff_wid_p < sqrt_tile) {
        std::runtime_error("src must be smaller than buffer");
    }
    const int s_offset_x = offset_x * sqrt_samples;
    const int s_offset_y = offset_y * sqrt_samples;
    const int rem_x = buff_len_p - offset_x;
    const int rem_y = buff_wid_p - offset_y;
    if (rem_x < sqrt_tile || rem_y < sqrt_tile) {
        for (int i = 0; i < rem_x * sqrt_samples; i++) {
            for (int j = 0; j < rem_y * sqrt_samples; j++) {
                dest.set(s_offset_x + i, s_offset_y + j, src.get(i, j));
            }
        }
    }
    for (int i = 0; i < sqrt_tile * sqrt_samples; i++) {
        for (int j = 0; j < sqrt_tile * sqrt_samples; j++) {
            dest.set(s_offset_x + i, s_offset_y + j, src.get(i, j));
        }
    }
}

// these functions kind of got hard to work with, if the function is simple
// enough I avoid using these functions
template <typename Func, typename BuffType, typename ArgType>
void with_tiles(Func &&job, bound_box<int> &b_box, BuffType &&buffs,
                ArgType &&args) {
    constexpr int sqrt_tile = consts::TILE_SIZE;
    int sqrt_samples = buffs.get_sqrt_samples();
    int box_len = b_box.max_x - b_box.min_x;
    int box_wid = b_box.max_y - b_box.min_y;
    auto tiles = buffs.make_tiles(sqrt_tile);
    int x_tiles = (box_len + sqrt_tile - 1) / sqrt_tile;
    int y_tiles = (box_wid + sqrt_tile - 1) / sqrt_tile;
    for (int i = 0; i < x_tiles; ++i) {
        int offset_x = b_box.min_x + (i * sqrt_tile);
        for (int j = 0; j < y_tiles; ++j) {
            int offset_y = b_box.min_y + (j * sqrt_tile);
            tiles.pull_to_tile(buffs, offset_x, offset_y);
            job(b_box, std::forward<BuffType>(tiles),
                std::forward<ArgType>(args));
            tiles.push_to_buff(buffs, offset_x, offset_y);
        }
    }
}

template <typename Func, typename BuffType, typename ArgType>
void engine_helper::with_buff(Func &&job, const bound_box<int> &b_box,
                              BuffType &&buffs, ArgType &&args) {
    job(b_box, std::forward<BuffType>(buffs), std::forward<ArgType>(args));
}

template <typename T>
void engine_helper::rast_tri(const bound_box<int> &b_box,
                             ra_tri_buffs<T> &buffs, ra_tri_args<T> &args,
                             const int off_x, const int off_y) {
    const int paren_len = args.paren_len;
    const int paren_wid = args.paren_wid;
    const double a_ratio = (double)paren_len / paren_wid;
    const int sqrt_samples = buffs.buff.get_sqrt_samples();
    const int left = b_box.min_x * sqrt_samples;
    const int right = b_box.max_x * sqrt_samples;
    const int top = b_box.min_y * sqrt_samples;
    const int bot = b_box.max_y * sqrt_samples;
    const double s_pix_to_world_x = 2.0 * a_ratio / paren_len;
    const double s_pix_to_world_y = 2.0 / paren_wid;
    const raw_tri &p_tri = args.p_tri;
    const double p1_z = p_tri.p1[2];
    const double p2_z = p_tri.p2[2];
    const double p3_z = p_tri.p3[2];
    const double inv_p1_z = 1.0 / p1_z;
    const double inv_p2_z = 1.0 / p2_z;
    const double inv_p3_z = 1.0 / p3_z;
    const double near_plane = 0.1;
    if (p1_z < near_plane || p2_z < near_plane || p3_z < near_plane) {
        return;
    }
    for (int k = left; k < right; ++k) {
        double world_x = k * s_pix_to_world_x - a_ratio;
        for (int l = top; l < bot; ++l) {
            double world_y = l * s_pix_to_world_y - 1;
            Eigen::Vector3d test{world_x, world_y, 0};
            Eigen::Vector3d bary = get_bary(p_tri.p1, p_tri.p2, p_tri.p3, test);
            if (bary[0] < 0.0 || bary[1] < 0.0 || bary[2] < 0.0) {
                continue;
            }
            double z_rep =
                bary[0] * inv_p1_z + bary[1] * inv_p2_z + bary[2] * inv_p3_z;
            double z_sub = 1.0 / z_rep;
            int x = k - off_x;
            int y = l - off_y;
            if (buffs.z_buff.get(x, y) > z_sub) {
                buffs.z_buff.set(x, y, z_sub);
                buffs.buff.set(x, y, args.val);
            }
        }
    }
}
