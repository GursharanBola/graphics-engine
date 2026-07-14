#include "engine_helper.h"
#include "consts.h"

Eigen::Vector3d engine_helper::project_point(const Eigen::Vector3d &p1,
                                             const Eigen::Vector3d &cam_u,
                                             const Eigen::Vector3d &cam_v,
                                             const Eigen::Vector3d &cam_w,
                                             const Eigen::Vector3d &origin,
                                             const double focal_len) {
    Eigen::Vector3d translated = p1 - origin;
    double x_cam = translated.dot(cam_u);
    double y_cam = translated.dot(cam_v);
    double z_cam = -translated.dot(cam_w);
    if (std::abs(z_cam) < 1e-6) {
        z_cam = 1e-6;
    }
    double ratio = focal_len / z_cam;
    return Eigen::Vector3d(x_cam * ratio, y_cam * ratio, z_cam);
}

raw_tri engine_helper::proj_tri(const raw_tri &tri,
                                const Eigen::Vector3d &cam_u,
                                const Eigen::Vector3d &cam_v,
                                const Eigen::Vector3d &cam_w,
                                const Eigen::Vector3d &origin,
                                const double focal_len) {
    return raw_tri{
        project_point(tri.p1, cam_u, cam_v, cam_w, origin, focal_len),
        project_point(tri.p2, cam_u, cam_v, cam_w, origin, focal_len),
        project_point(tri.p3, cam_u, cam_v, cam_w, origin, focal_len)};
}

bound_box<double> engine_helper::w_box(const Eigen::Vector3d &p1,
                                       const Eigen::Vector3d &p2,
                                       const Eigen::Vector3d &p3,
                                       const double aspect_ratio) {
    double hor_min = std::min({p1[0], p2[0], p3[0]});
    double ver_min = std::min({p1[1], p2[1], p3[1]});
    double hor_max = std::max({p1[0], p2[0], p3[0]});
    double ver_max = std::max({p1[1], p2[1], p3[1]});
    hor_min = std::clamp(hor_min, -aspect_ratio, aspect_ratio);
    hor_max = std::clamp(hor_max, -aspect_ratio, aspect_ratio);
    ver_min = std::clamp(ver_min, -1.0, 1.0);
    ver_max = std::clamp(ver_max, -1.0, 1.0);
    return bound_box<double>{hor_min, hor_max, ver_min, ver_max};
}

bound_box<int>
engine_helper::create_box(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                          const Eigen::Vector3d &p3, const double aspect_ratio,
                          const int img_length, const int img_width) {
    bound_box<double> w_bbox = w_box(p1, p2, p3, aspect_ratio);
    int left =
        std::floor(0.5 * (w_bbox.min_x / aspect_ratio + 1.0) * img_length);
    int right =
        std::ceil(0.5 * (w_bbox.max_x / aspect_ratio + 1.0) * img_length);
    int y_top = std::floor((1.0 - w_bbox.max_y) * 0.5 * img_width);
    int y_bottom = std::ceil((1.0 - w_bbox.min_y) * 0.5 * img_width);
    return bound_box<int>{
        std::clamp(left, 0, img_length), std::clamp(right, 0, img_length),
        std::clamp(y_top, 0, img_width), std::clamp(y_bottom, 0, img_width)};
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
    double area = edge_func(p1, p2, p3);
    if (std::abs(area) < 1e-9) {
        return {-1.0, -1.0, -1.0};
    }
    double w1 = edge_func(p2, p3, test_pt);
    double w2 = edge_func(p3, p1, test_pt);
    double w3 = edge_func(p1, p2, test_pt);
    double inv_area = 1.0 / area;
    double b1 = w1 * inv_area;
    double b2 = w2 * inv_area;
    double b3 = w3 * inv_area;
    if (b1 >= 0.0 && b2 >= 0.0 && b3 >= 0.0) {
        return {b1, b2, b3};
    }
    return {-1.0, -1.0, -1.0};
}

template <typename T>
void engine_helper::pull(buffer<T> &src, buffer<T> &dest, const int offset_x,
                         const int offset_y) {
    const int sqrt_tile = dest.get_length_p();
    const int tile_w = dest.get_width_p();
    if (sqrt_tile != tile_w) {
        throw std::runtime_error("dest is a tile, it must be square");
    }
    const int sqrt_samples = dest.get_sqrt_samples();
    const int buff_sqrt_samples = src.get_sqrt_samples();
    if (sqrt_samples != buff_sqrt_samples) {
        throw std::runtime_error("src and dest must have same num_samples");
    }
    int rem_x = (src.get_length_p() - offset_x) * sqrt_samples;
    int rem_y = (src.get_width_p() - offset_y) * sqrt_samples;
    int bound_x = std::min(sqrt_tile * sqrt_samples, rem_x);
    int bound_y = std::min(sqrt_tile * sqrt_samples, rem_y);
    if (bound_x <= 0 || bound_y <= 0) {
        return;
    }
    const int s_offset_x = offset_x * sqrt_samples;
    const int s_offset_y = offset_y * sqrt_samples;
    for (int j = 0; j < bound_y; ++j) {
        for (int i = 0; i < bound_x; ++i) {
            dest.set(i, j, src.get(s_offset_x + i, s_offset_y + j));
        }
    }
}

template <typename T>
void push(buffer<T> &src, buffer<T> &dest, const int offset_x,
          const int offset_y) {
    const int sqrt_tile = src.get_length_p();
    const int tile_w = src.get_width_p();
    if (sqrt_tile != tile_w) {
        throw std::runtime_error("src is a tile, it must be square");
    }
    const int sqrt_samples = dest.get_sqrt_samples();
    const int buff_sqrt_samples = src.get_sqrt_samples();
    if (sqrt_samples != buff_sqrt_samples) {
        throw std::runtime_error("src and dest must have same num_samples");
    }
    int rem_x = (dest.get_length_p() - offset_x) * sqrt_samples;
    int rem_y = (dest.get_width_p() - offset_y) * sqrt_samples;
    int bound_x = std::min(sqrt_tile * sqrt_samples, rem_x);
    int bound_y = std::min(sqrt_tile * sqrt_samples, rem_y);
    if (bound_x <= 0 || bound_y <= 0) {
        return;
    }
    const int s_offset_x = offset_x * sqrt_samples;
    const int s_offset_y = offset_y * sqrt_samples;
    for (int j = s_offset_y; j < s_offset_y + bound_y; ++j) {
        for (int i = s_offset_x; i < s_offset_x + bound_x; ++i) {
            dest.set(i, j, src.get(i - s_offset_x, j - s_offset_y));
        }
    }
}

// these functions kind of got hard to work with, if the function is simple
// enough I avoid using these functions
template <typename Func, typename BuffType, typename ArgType>
void with_tiles(Func &&job, bound_box<int> &b_box, BuffType &buffs,
                ArgType &args) {
    constexpr int sqrt_tile = consts::TILE_SIZE;
    int box_len = b_box.max_x - b_box.min_x;
    int box_wid = b_box.max_y - b_box.min_y;
    auto tiles = make_tiles_for(buffs, sqrt_tile);
    int x_tiles = (box_len + sqrt_tile - 1) / sqrt_tile;
    int y_tiles = (box_wid + sqrt_tile - 1) / sqrt_tile;
    for (int j = 0; j < y_tiles; ++j) {
        int offset_y = b_box.min_y + (j * sqrt_tile);
        for (int i = 0; i < x_tiles; ++i) {
            int offset_x = b_box.min_x + (i * sqrt_tile);
            tiles.view.pull_to_tile(buffs, offset_x, offset_y);
            job(b_box, tiles.view, args, offset_x, offset_y);
            tiles.view.push_to_buff(buffs, offset_x, offset_y);
        }
    }
}

template <typename Func, typename BuffType, typename ArgType>
void engine_helper::with_buff(Func &&job, const bound_box<int> &b_box,
                              BuffType &buffs, ArgType &args) {
    job(b_box, buffs, args, 0, 0);
}

template <typename T>
void engine_helper::rast_tri(const bound_box<int> &b_box,
                             ra_tri_buffs<T> &buffs, ra_tri_args<T> &args,
                             const int off_x, const int off_y) {
    const int paren_len = args.paren_len; // in sub_pixels
    const int paren_wid = args.paren_wid; // in sub_pixels
    const int buf_w = buffs.buff->get_length_p();
    const int buf_h = buffs.buff->get_width_p();
    const double a_ratio = (double)paren_len / paren_wid;
    const int sqrt_samples = buffs.buff->get_sqrt_samples();

    const int s_off_x = off_x * sqrt_samples;
    const int s_off_y = off_y * sqrt_samples;

    const int left = std::max(b_box.min_x * sqrt_samples, s_off_x);
    const int right =
        std::min(b_box.max_x * sqrt_samples, s_off_x + (buf_w * sqrt_samples));
    const int top = std::max(b_box.min_y * sqrt_samples, s_off_y);
    const int bot =
        std::min(b_box.max_y * sqrt_samples, s_off_y + (buf_h * sqrt_samples));

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
    for (int l = top; l < bot; ++l) {
        double world_y = 1.0 - (l + 0.5) + s_pix_to_world_y;
        for (int k = left; k < right; ++k) {
            double world_x = (k + 0.5) * s_pix_to_world_x - a_ratio;
            Eigen::Vector3d test{world_x, world_y, 0};
            Eigen::Vector3d bary = get_bary(p_tri.p1, p_tri.p2, p_tri.p3, test);
            if (bary[0] < 0.0 || bary[1] < 0.0 || bary[2] < 0.0) {
                continue;
            }
            double z_rep =
                bary[0] * inv_p1_z + bary[1] * inv_p2_z + bary[2] * inv_p3_z;
            double z_sub = 1.0 / z_rep;
            int x = k - s_off_x;
            int y = l - s_off_y;
            if (buffs.z_buff->get(x, y) > z_sub) {
                buffs.z_buff->set(x, y, z_sub);
                buffs.buff->set(x, y, args.val);
            }
        }
    }
}
