#include "engine.h"
#include "Eigen/src/Core/Matrix.h"
#include "buffer.h"
#include "consts.h"
#include "ds.h"
#include "engine_helper.h"
#include "material.h"
#include "mesh.h"
#include "projector.h"
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>

void engine::render() {
    const int img_length = scene.get_img_length();
    const int img_height = scene.get_img_height();
    const int sqrt_samples = scene.get_sqrt_samples();
    const int samples = sqrt_samples * sqrt_samples;
    color_buffer &color_buff = scene.col_buffer;
    image_buffer &img = scene.img;
    // TODO: complete this function later not a focus right now
}

void engine::fill_z_s(const projector &projector,
                      const std::vector<shape> &meshes,
                      const ds::e_cache_map<triangle> &list_of_tris,
                      z_buffer &z_buff, seen_buffer &s_buff) const {
    const Eigen::Vector3d &cam_u = projector.get_u();
    const Eigen::Vector3d &cam_v = projector.get_v();
    const Eigen::Vector3d &cam_w = projector.get_w();
    const Eigen::Vector3d &cam_o = projector.get_o();
    const double f_len = projector.get_f_len();
    const int length = z_buff.get_length();
    const int width = z_buff.get_width();
    const int length_p = z_buff.get_length_p();
    const int width_p = z_buff.get_width_p();
    const int sqrt_samples = z_buff.get_sqrt_samples();
    const double a_ratio = static_cast<double>(length) / width;
    const int num_meshes = meshes.size();
    if (length != s_buff.get_length() || width != s_buff.get_width() ||
        s_buff.get_sqrt_samples() != sqrt_samples) {
        throw std::runtime_error(
            "s_buff must have same width, length, and sqrt_samples");
    }
    std::vector<int> sizes{}; // this is not changed by e_c_map constructor
    for (int m_id = 0; m_id < num_meshes; ++m_id) {
        sizes.emplace_back(list_of_tris.mesh_size(m_id));
    }
    ds::e_cache_map<cached_tri> map(sizes);
    constexpr int NUM_THREADS = consts::NUM_THREADS;
    int stride = std::max(1, num_meshes / NUM_THREADS);
    std::vector<std::thread> threads;
    for (int mesh_start = 0; mesh_start < num_meshes; mesh_start += stride) {
        int mesh_end = mesh_start + stride;
        mesh_end = mesh_end > num_meshes ? num_meshes : mesh_end;
        threads.emplace_back([&, mesh_start, mesh_end]() {
            for (int mesh_id = mesh_start; mesh_id < mesh_end; ++mesh_id) {
                const shape &c_mesh = meshes[mesh_id];
                const int num_tris = sizes[mesh_id];
                for (int tri_index = 0; tri_index < num_tris; ++tri_index) {
                    const triangle &tri = list_of_tris.get(mesh_id, tri_index);
                    triangle p_tri = engine_helper::proj_tri(
                        tri, cam_u, cam_v, cam_w, cam_o, f_len);
                    if (engine_helper::edge_func(p_tri.point1, p_tri.point2,
                                                 p_tri.point3) < 0) {
                        continue; // backface cull
                    }
                    bound_box<int> p_b_box = engine_helper::create_box(
                        p_tri.point1, p_tri.point2, p_tri.point3, a_ratio,
                        length_p, width_p);
                    cached_tri c_tri = map.claim_next_slot(mesh_id);
                    c_tri = cached_tri{tri_index, p_tri, p_b_box};
                }
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    threads.clear();
    std::vector<int> &initial = map.initial;
    std::vector<int> &offsets = map.offsets;
    std::vector<cached_tri> &seen_tris = map.data;
    engine_helper::ra_tri_buffs<tri_ref> buffs{s_buff, z_buff};
    threads.reserve(NUM_THREADS);
    const int t_height = (width + NUM_THREADS - 1) / NUM_THREADS;
    for (int top = 0; top < width; top += t_height) {
        int next_row = top + t_height;
        int bot = next_row > width_p ? width_p : next_row;
        bound_box thread_box = bound_box<int>{0, length_p, top, bot};
        threads.emplace_back([&, thread_box]() {
            for (int mesh_id = 0; mesh_id < num_meshes; ++mesh_id) {
                const int start = initial[mesh_id];
                const int end = offsets[mesh_id];
                for (int ith_tri = start; ith_tri < end; ++ith_tri) {
                    const int tri_index = seen_tris[ith_tri].tri_index;
                    const triangle &p_tri = seen_tris[ith_tri].p_tri;
                    const bound_box<int> &p_b_box = seen_tris[ith_tri].b_box;
                    int b_left = std::max(p_b_box.min_x, thread_box.min_x);
                    int b_right = std::min(p_b_box.max_x, thread_box.max_x);
                    int b_top = std::max(p_b_box.min_y, thread_box.min_y);
                    int b_bot = std::min(p_b_box.max_y, thread_box.max_y);
                    if (b_left > b_right || b_top > b_bot) {
                        continue;
                    }
                    bound_box t_b_box =
                        bound_box<int>{b_left, b_right, b_top, b_bot};
                    tri_ref current_val = tri_ref{mesh_id, tri_index};
                    engine_helper::ra_tri_args<tri_ref> args{
                        length, width, p_tri, current_val};
                    engine_helper::with_buff(engine_helper::rast_tri_fn{},
                                             t_b_box, buffs, args);
                }
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
}

void engine::color_buff(const camera &cam, const bool is_map,
                        const seen_buffer &cam_s_buff, color_buffer &col_buff) {
    const int length_p = scene.get_img_length();
    const int width_p = scene.get_img_height();
    const int sqrt_samples = scene.get_sqrt_samples();
    const int length = length_p * sqrt_samples;
    const int width = width_p * sqrt_samples;
    const double shadow_world_x_to_pix = length / 2.0;
    const double shadow_world_y_to_pix = width / 2.0;
    const double inv_pi = 1.0 / EIGEN_PI;
    const Eigen::Vector3d ones = Eigen::Vector3d::Ones();
    const Eigen::Vector3d &cam_o = cam.get_o();
    const Eigen::Vector3d &cam_u = cam.get_u();
    const Eigen::Vector3d &cam_v = cam.get_v();
    const Eigen::Vector3d &cam_w = cam.get_w();
    const double cam_focal_len = cam.get_f_len();
    const color &ambient = scene.ambient_color;
    const int num_meshes = scene.meshes.size();
    const int num_lights = scene.lights.size();
    double x_max = static_cast<double>(length_p) / width_p;
    double y_max = 1.0;
    double s_pix_to_world_x = 2.0 * x_max / length;
    double s_pix_to_world_y = 2.0 * y_max / width;
    double world_y_to_s_pix = width / (2.0 * y_max);
    double world_x_to_s_pix = length / (2.0 * x_max);
    if (is_map) {
        x_max = 1.0;
        y_max = 1.0;
        s_pix_to_world_x = 2.0 * x_max / length;
        s_pix_to_world_y = 2.0 * y_max / width;
        world_y_to_s_pix = width / (2.0 * y_max);
        world_x_to_s_pix = length / (2.0 * x_max);
    }
    const std::vector<shape> &meshes = scene.meshes;
    const std::vector<light> &lights = scene.lights;
    const std::vector<material> mats = scene.mats;
    const std::vector<seen_buffer> &l_s_buffs = scene.s_buffer_lights;
    const ds::e_cache_map<triangle> &list_of_tri = scene.list_of_tri;

    for (int j = 0; j < width; ++j) {
        const double world_y = (j + 0.5) * s_pix_to_world_y - y_max;
        for (int i = 0; i < length; ++i) {
            const double world_x = (i + 0.5) * s_pix_to_world_x - x_max;
            const Eigen::Vector3d test{world_x, world_y, 0};
            const tri_ref tri_r = cam_s_buff.get(i, j);
            const int mesh_id = tri_r.mesh_id;
            const int tri_index = tri_r.tri_index;
            const shape &c_mesh = meshes[mesh_id];
            const material &mat = mats[mesh_id];
            const color &base_color = mat.col;
            const bool is_metal = mat.is_metal;
            const double metal = mat.metalic;
            const double shine = mat.shine;

            const triangle &tri = list_of_tri.get(mesh_id, tri_index);
            const Eigen::Vector3d &p1 = tri.point1;
            const Eigen::Vector3d &p2 = tri.point2;
            const Eigen::Vector3d &p3 = tri.point3;
            const Eigen::Vector3d n1 = find_normal_at(c_mesh, p1);
            const Eigen::Vector3d n2 = find_normal_at(c_mesh, p2);
            const Eigen::Vector3d n3 = find_normal_at(c_mesh, p3);

            const triangle p_tri = engine_helper::proj_tri(
                tri, cam_u, cam_v, cam_w, cam_o, cam_focal_len);

            const double inv_area =
                1.0 / engine_helper::edge_func(p_tri.point1, p_tri.point2,
                                               p_tri.point3);

            const Eigen::Vector3d bary = engine_helper::get_bary(
                p_tri.point1, p_tri.point2, p_tri.point3, test, inv_area);

            const double alpha = bary[0];
            const double beta = bary[1];
            const double gamma = bary[2];

            const double z1 = p_tri.point1.z();
            const double z2 = p_tri.point2.z();
            const double z3 = p_tri.point3.z();

            const double inv_z_pixel =
                (alpha / z1) + (beta / z2) + (gamma / z3);
            const double pixel_z = 1.0 / inv_z_pixel;

            const double pc_alpha = (alpha / z1) * pixel_z;
            const double pc_beta = (beta / z2) * pixel_z;
            const double pc_gamma = (gamma / z3) * pixel_z;

            const Eigen::Vector3d world_pos =
                pc_alpha * p1 + pc_beta * p2 + pc_gamma * p3;

            const Eigen::Vector3d inter_norm =
                (pc_alpha * n1 + pc_beta * n2 + pc_gamma * n3).normalized();

            const Eigen::Vector3d view = (cam_o - world_pos).normalized();

            const Eigen::Vector3d &reflectance = mat.reflectance;
            const Eigen::Vector3d metal_color = (1 - metal) * base_color.val;
            const Eigen::Vector3d ambient_term =
                metal_color.cwiseProduct(ambient.val);
            const Eigen::Vector3d norm_metal_color = metal_color * inv_pi;
            const Eigen::Vector3d F0 =
                (1.0 - metal) * reflectance + metal * base_color.val;
            const double scaled_shine = (mat.shine + 2.0) * 0.5 * inv_pi;
            const double n_dot_v = std::max(0.0, inter_norm.dot(view));
            const double cam_trans = engine_helper::f_pow(1.0 - n_dot_v, 5);

            Eigen::Vector3d tot_col = ambient_term;

            if (is_metal && !is_map) {
                const double env_trans = engine_helper::f_pow(1.0 - n_dot_v, 5);
                const Eigen::Vector3d env_frensel =
                    F0 + (ones - F0) * env_trans;

                const double v_dot_n = view.dot(inter_norm);
                const Eigen::Vector3d reflect_dir =
                    ((2.0 * v_dot_n * inter_norm) - view).normalized();

                const Eigen::Vector3d refl_col =
                    ref_col(c_mesh, inter_norm, world_pos);
                tot_col += env_frensel.cwiseProduct(refl_col);
            }

            for (size_t ith_light = 0; ith_light < num_lights; ++ith_light) {
                const light &c_light = lights[ith_light];
                const seen_buffer &c_l_s_buff = l_s_buffs[ith_light];
                const Eigen::Vector3d &light_u = c_light.get_u();
                const Eigen::Vector3d &light_v = c_light.get_v();
                const Eigen::Vector3d &light_w = c_light.get_w();
                const Eigen::Vector3d &light_o = c_light.get_o();
                const color &l_color = c_light.get_color();
                const double light_f_len = c_light.get_f_len();

                const Eigen::Vector3d proj_point = engine_helper::project_point(
                    world_pos, light_u, light_v, light_w, light_o, light_f_len);

                if (proj_point.z() <= 0.0) {
                    continue; // cull triangles behind camera
                }

                const int s_pixel_x = static_cast<int>((proj_point[0] + 1.0) *
                                                       shadow_world_x_to_pix);
                const int s_pixel_y = static_cast<int>((proj_point[1] + 1.0) *
                                                       shadow_world_y_to_pix);

                if (s_pixel_x < 0 || s_pixel_x >= length || s_pixel_y < 0 ||
                    s_pixel_y >= width) {
                    continue;
                }

                const tri_ref &l_tri = c_l_s_buff.get(s_pixel_x, s_pixel_y);
                if (l_tri.mesh_id != mesh_id || l_tri.tri_index != tri_index) {
                    continue;
                }
                const Eigen::Vector3d L = (light_o - world_pos).normalized();
                const Eigen::Vector3d half = (view + L).normalized();
                const double n_dot_h = std::max(0.0, inter_norm.dot(half));
                const double n_dot_l = std::max(0.0, inter_norm.dot(L));
                if (n_dot_l <= 0.0) {
                    continue;
                }

                const double v_dot_h = std::max(0.0, view.dot(half));
                const double distro =
                    scaled_shine * engine_helper::f_pow(n_dot_h, shine);
                const double trans = engine_helper::f_pow(1.0 - v_dot_h, 5);
                const Eigen::Vector3d frensel = F0 + (ones - F0) * trans;
                const double facet = 2.0 * n_dot_h / std::max(0.0001, v_dot_h);
                const double dir_1 = facet * n_dot_v;
                const double dir_2 = facet * n_dot_l;
                const double G = std::min<double>({1, dir_1, dir_2});

                const Eigen::Vector3d diff_brdf =
                    (Eigen::Vector3d::Ones() - frensel)
                        .cwiseProduct(norm_metal_color) *
                    n_dot_l;
                Eigen::Vector3d spec_brdf =
                    (distro * frensel * G * 0.25) / std::max(0.0001, n_dot_v);

                tot_col += (diff_brdf + spec_brdf).cwiseProduct(l_color.val);
            }
            col_buff.set(i, j, color{tot_col[0], tot_col[1], tot_col[2]});
        }
    }
}

void engine::make_all_maps() {
    const std::vector<shape> &meshes = scene.meshes;
    const std::vector<material> &mats = scene.mats;
    const int num_meshes = meshes.size();
    for (size_t m_id = 0; m_id < num_meshes; ++m_id) {
        if (!mats[m_id].is_metal) {
            continue;
        } else {
            make_cubemap(meshes[m_id]);
        }
    }
}

void engine::make_cubemap(const shape &mesh) {
    const int m_id = get_id(mesh);
    const int index = scene.mats[m_id].metal_data;
    const double f_len = 1.0;
    const Eigen::Vector3d &origin = get_origin_of(mesh);
    const Eigen::Vector3d cam_u{1, 0, 0};
    const Eigen::Vector3d cam_v{0, 1, 0};
    const Eigen::Vector3d cam_w{0, 0, 1};
    const std::vector<seen_buffer> &c_s_buffs = scene.cubemaps_s;
    std::vector<color_buffer> &c_c_buffs = scene.cubemaps;

    color_buff(camera{origin, cam_u, cam_v, cam_w, f_len}, true,
               c_s_buffs[index + 0], c_c_buffs[index + 0]);

    color_buff(camera{origin, -cam_u, cam_v, -cam_w, f_len}, true,
               c_s_buffs[index + 1], c_c_buffs[index + 1]);

    color_buff(camera{origin, cam_u, -cam_w, cam_v, f_len}, true,
               c_s_buffs[index + 2], c_c_buffs[index + 2]);

    color_buff(camera{origin, cam_u, cam_w, -cam_v, f_len}, true,
               c_s_buffs[index + 3], c_c_buffs[index + 3]);

    color_buff(camera{origin, cam_w, cam_v, -cam_u, f_len}, true,
               c_s_buffs[index + 4], c_c_buffs[index + 4]);

    color_buff(camera{origin, -cam_w, cam_v, cam_u, f_len}, true,
               c_s_buffs[index + 5], c_c_buffs[index + 5]);
}

void engine::fill_all_z_s() {
    z_buffer &z_buffer_cam = scene.z_buffer_cam;
    seen_buffer &s_buffer_cam = scene.s_buffer_cam;
    ds::e_cache_map<triangle> &list_of_tri = scene.list_of_tri;
    std::vector<z_buffer> &z_buffer_lights = scene.z_buffer_lights;
    std::vector<seen_buffer> &s_buffer_lights = scene.s_buffer_lights;
    const std::vector<shape> &meshes = scene.meshes;
    const std::vector<light> &lights = scene.lights;
    fill_z_s(scene_cam, meshes, list_of_tri, z_buffer_cam, s_buffer_cam);
    int z_buffer_lights_size = z_buffer_lights.size();
    for (size_t i = 0; i < z_buffer_lights_size; ++i) {
        fill_z_s(lights[i], meshes, list_of_tri, z_buffer_lights[i],
                 s_buffer_lights[i]);
    }
}

void engine::fill_map_v_s() {
    std::vector<seen_buffer> &cubemaps_s = scene.cubemaps_s;
    std::vector<z_buffer> &cubemaps_z = scene.cubemaps_z;
    const std::vector<shape> &meshes = scene.meshes;
    const std::vector<material> &mats = scene.mats;
    const ds::e_cache_map<triangle> &list_of_tris = scene.list_of_tri;
    const int num_meshes = meshes.size();
    const double f_len = 1.0; // associated w/ 90* angle fov
    for (size_t ith_mesh = 0; ith_mesh < num_meshes; ++ith_mesh) {
        const material &mat = mats[ith_mesh];
        const int data_index = mat.metal_data;
        if (!mat.is_metal) {
            continue;
        }
        const Eigen::Vector3d &origin = get_origin_of(meshes[ith_mesh]);

        const Eigen::Vector3d cam_u{1, 0, 0};
        const Eigen::Vector3d cam_v{0, 1, 0};
        const Eigen::Vector3d cam_w{0, 0, 1};
        fill_z_s(camera{origin, cam_u, cam_v, cam_w, f_len}, meshes,
                 list_of_tris, cubemaps_z[data_index], cubemaps_s[data_index]);

        fill_z_s(camera{origin, -cam_u, cam_v, -cam_w, f_len}, meshes,
                 list_of_tris, cubemaps_z[data_index + 1],
                 cubemaps_s[data_index + 1]);

        fill_z_s(camera{origin, cam_u, -cam_w, cam_v, f_len}, meshes,
                 list_of_tris, cubemaps_z[data_index + 2],
                 cubemaps_s[data_index + 2]);

        fill_z_s(camera{origin, cam_u, cam_w, -cam_v, f_len}, meshes,
                 list_of_tris, cubemaps_z[data_index + 3],
                 cubemaps_s[data_index + 3]);

        fill_z_s(camera{origin, cam_w, cam_v, -cam_u, f_len}, meshes,
                 list_of_tris, cubemaps_z[data_index + 4],
                 cubemaps_s[data_index + 4]);

        fill_z_s(camera{origin, -cam_w, cam_v, cam_u, f_len}, meshes,
                 list_of_tris, cubemaps_z[data_index + 5],
                 cubemaps_s[data_index + 5]);
    }
}

Eigen::Vector3d engine::ref_col(const shape &mesh, const Eigen::Vector3d &r_dir,
                                const Eigen::Vector3d &r_origin) {
    const int m_id = get_id(mesh);
    const int side_len = consts::MAP_SIDE_LEN;

    if (std::holds_alternative<quad>(mesh)) {
        const quad &q = std::get<quad>(mesh);
        const Eigen::Vector3d &q_origin = q.get_origin();
        const Eigen::Vector3d box_min = q.b_cube[0];
        const Eigen::Vector3d box_max = q.b_cube[1];
        const double u_norm = q.u_norm;
        const double v_norm = q.v_norm;

        const double half_wid_x = std::abs(box_max.x() - box_min.x()) * 0.5;
        const double half_wid_y = std::abs(box_max.y() - box_min.y()) * 0.5;
        const double half_wid_z = std::abs(box_max.z() - box_min.z()) * 0.5;

        const double t_x_exit = (r_dir.x() >= 0.0)
                                    ? (box_max.x() - r_origin.x()) / r_dir.x()
                                    : (box_min.x() - r_origin.x()) / r_dir.x();
        const double t_y_exit = (r_dir.y() >= 0.0)
                                    ? (box_max.y() - r_origin.y()) / r_dir.y()
                                    : (box_min.y() - r_origin.y()) / r_dir.y();
        const double t_z_exit = (r_dir.z() >= 0.0)
                                    ? (box_max.z() - r_origin.z()) / r_dir.z()
                                    : (box_min.z() - r_origin.z()) / r_dir.z();

        const double t_exit = std::min({t_x_exit, t_y_exit, t_z_exit});

        // center of the bounding box
        const Eigen::Vector3d cen = (box_min + box_max) * 0.5;
        const Eigen::Vector3d cen_point_on_box =
            t_exit * r_dir + r_origin - cen;

        const auto [face_idx, per_x, per_y] = engine_helper::get_face(
            cen_point_on_box, half_wid_x, half_wid_y, half_wid_z);

        const int index = scene.mats[m_id].metal_data + face_idx;
        const color_buffer &col_buff = scene.cubemaps[index];

        const int max_val = side_len - 1;
        const int s_pix_x = std::clamp(
            static_cast<int>(std::floor(per_x * side_len)), 0, max_val);
        const int s_pix_y = std::clamp(
            static_cast<int>(std::floor(per_y * side_len)), 0, max_val);

        return col_buff.get(s_pix_x, s_pix_y).val;
    } else {
        const sphere &s = std::get<sphere>(mesh);
        const Eigen::Vector3d &s_origin = s.get_origin();
        const double radius = s.get_radius();
        const Eigen::Vector3d box_min = s.b_cube[0];
        const Eigen::Vector3d box_max = s.b_cube[1];

        const double t_x_exit = (r_dir.x() >= 0.0)
                                    ? (box_max.x() - r_origin.x()) / r_dir.x()
                                    : (box_min.x() - r_origin.x()) / r_dir.x();
        const double t_y_exit = (r_dir.y() >= 0.0)
                                    ? (box_max.y() - r_origin.y()) / r_dir.y()
                                    : (box_min.y() - r_origin.y()) / r_dir.y();
        const double t_z_exit = (r_dir.z() >= 0.0)
                                    ? (box_max.z() - r_origin.z()) / r_dir.z()
                                    : (box_min.z() - r_origin.z()) / r_dir.z();

        double t_exit = std::min({t_x_exit, t_y_exit, t_z_exit});

        Eigen::Vector3d sur = t_exit * r_dir + r_origin - get_origin_of(mesh);
        auto [face_idx, per_x, per_y] =
            engine_helper::get_face(sur, radius, radius, radius);

        const int index = scene.mats[m_id].metal_data + face_idx;
        const color_buffer &col_buff = scene.cubemaps[index];
        const int max_val = side_len - 1;
        const int s_pix_x = std::clamp(
            static_cast<int>(std::floor(per_x * side_len)), 0, max_val);
        const int s_pix_y = std::clamp(
            static_cast<int>(std::floor(per_y * side_len)), 0, max_val);

        return col_buff.get(s_pix_x, s_pix_y).val;
    }
}
