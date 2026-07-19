#include "engine.h"
#include "Eigen/src/Core/Matrix.h"
#include "buffer.h"
#include "consts.h"
#include "engine_helper.h"
#include "projector.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

// TODO: double check the cam_w convention, which is cam_w faces backward

void engine::fill_z_s(const projector &projector,
                      const std::vector<std::shared_ptr<mesh>> &meshes,
                      const vertex_buffer &v_buff, z_buffer &z_buff,
                      seen_buffer &s_buff) const {
    const Eigen::Vector3d &cam_u = projector.get_u();
    const Eigen::Vector3d &cam_v = projector.get_v();
    const Eigen::Vector3d &cam_w = projector.get_w();
    const Eigen::Vector3d &cam_o = projector.get_o();
    const double f_len = projector.get_f_len();
    const int length = z_buff.get_length();
    const int width = z_buff.get_width();
    const int length_p = z_buff.get_width_p();
    const int width_p = z_buff.get_width_p();
    const int sqrt_samples = z_buff.get_sqrt_samples();
    const double a_ratio = static_cast<double>(length) / width;
    if (length != s_buff.get_length() || width != s_buff.get_width() ||
        s_buff.get_sqrt_samples() != sqrt_samples) {
        throw std::runtime_error(
            "s_buff must have same width, length, and sqrt_samples");
    }
    constexpr int NUM_THREADS = consts::NUM_THREADS;
    engine_helper::ra_tri_buffs<tri_ref> buffs{&s_buff, &z_buff};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    const int t_height = (width + NUM_THREADS - 1) / NUM_THREADS;
    for (int top = 0; top < width; top += t_height) {
        int next_row = top + t_height;
        int bot = next_row > width_p ? width_p : next_row;
        bound_box thread_box = bound_box<int>{0, length_p, top, bot};
        threads.emplace_back([&, thread_box]() {
            for (const auto &mesh : meshes) {
                int tri_index = 0;
                for (const triangle &tri : mesh->list_of_triangles) {
                    raw_tri new_tri =
                        raw_tri{v_buff.get(tri.point1), v_buff.get(tri.point2),
                                v_buff.get(tri.point3)};
                    raw_tri p_tri = engine_helper::proj_tri(
                        new_tri, cam_u, cam_v, cam_w, cam_o, f_len);
                    bound_box<int> p_b_box =
                        engine_helper::create_box(p_tri.p1, p_tri.p2, p_tri.p3,
                                                  a_ratio, length_p, width_p);
                    int left = std::max(p_b_box.min_x, thread_box.min_x);
                    int right = std::min(p_b_box.max_x, thread_box.max_x);
                    int top = std::max(p_b_box.min_y, thread_box.min_y);
                    int bot = std::min(p_b_box.max_y, thread_box.max_y);
                    if (left > right || bot > top) {
                        continue;
                    }
                    bound_box t_b_box = bound_box<int>{left, right, top, bot};
                    tri_ref current_val = tri_ref{mesh->get_id(), tri_index};
                    engine_helper::ra_tri_args<tri_ref> args{
                        length, width, p_tri, current_val};
                    engine_helper::with_buff(engine_helper::rast_tri_fn{},
                                             t_b_box, buffs, args);
                    tri_index++;
                }
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
}

void engine::color_buffs() {
    const int length_p = scene.get_img_length();
    const int width_p = scene.get_img_height();
    const int sqrt_samples = scene.get_sqrt_samples();
    const int length = length_p * sqrt_samples;
    const int width = width_p * sqrt_samples;
    const Eigen::Vector3d &cam_o = camera.get_o();
    const Eigen::Vector3d &cam_u = camera.get_u();
    const Eigen::Vector3d &cam_v = camera.get_v();
    const Eigen::Vector3d &cam_w = camera.get_w();
    const double cam_focal_len = camera.get_f_len();
    const color &ambient = scene.ambient_color;
    const int num_meshes = scene.meshes.size();
    const int num_lights = scene.lights.size();
    const double a_ratio = static_cast<double>(length_p) / width_p;
    const double s_pix_to_world_x = 2.0 * a_ratio / length;
    const double s_pix_to_world_y = 2.0 / width;
    const double world_y_to_s_pix = width / 2.0;
    const double world_x_to_s_pix = length / (2.0 * a_ratio);
    const double inv_pi = 1.0 / EIGEN_PI;
    const std::vector<seen_buffer> &l_s_buffs = scene.s_buffer_lights;
    const std::vector<std::unique_ptr<mesh>> &meshes = scene.meshes;
    const vertex_buffer &v_buff = scene.v_buffer;
    const seen_buffer &cam_s_buff = scene.s_buffer_cam;
    color_buffer &col_buff = scene.col_buffer;
    std::vector<int> sizes{num_meshes};
    for (auto &mesh : meshes) {
        sizes.emplace_back(mesh->list_of_triangles.size());
    }
    e_cache_map map(sizes);
    for (int j = 0; j < width; ++j) {
        for (int i = 0; i < length; ++i) {
            tri_ref new_tri = cam_s_buff.get(i, j);
            map.add_tri(new_tri);
        }
    }
    map.post_process();
    std::vector<int> &initial = map.initial;
    std::vector<int> &offsets = map.offsets;
    std::vector<int> &seen_tris = map.seen_tris;
    for (int mesh_id = 0; mesh_id < num_meshes; ++mesh_id) {
        const auto &mesh = meshes[mesh_id];
        const auto &list_of_tris = mesh->list_of_triangles;
        const color &base_color = mesh->get_color();
        const double metalic = mesh->mat->metalic;
        const double shine = mesh->mat->shine;
        const Eigen::Vector3d &reflectance = mesh->mat->reflectance;
        const Eigen::Vector3d metal_color = base_color.val * (1 - metalic);
        const Eigen::Vector3d ambient_term =
            metal_color.cwiseProduct(ambient.val);
        const Eigen::Vector3d norm_metal_color = metal_color * inv_pi;
        const Eigen::Vector3d F0 =
            reflectance * (1.0 - metalic) + base_color.val * metalic;
        const int start = initial[mesh_id];
        const int end = offsets[mesh_id];
        for (int ith_tri = start; ith_tri < end; ++ith_tri) {
            const int tri_index = seen_tris[tri_index];
            triangle tri = list_of_tris[tri_index];
            const Eigen::Vector3d &p1 = v_buff.get(tri.point1);
            const Eigen::Vector3d &p2 = v_buff.get(tri.point2);
            const Eigen::Vector3d &p3 = v_buff.get(tri.point3);
            const Eigen::Vector3d n1 = mesh->find_normal(p1);
            const Eigen::Vector3d n2 = mesh->find_normal(p2);
            const Eigen::Vector3d n3 = mesh->find_normal(p3);
            const raw_tri new_tri = raw_tri{p1, p2, p3};
            const raw_tri p_tri = engine_helper::proj_tri(
                new_tri, cam_u, cam_v, cam_w, cam_o, cam_focal_len);
            const bound_box<int> p_b_box = engine_helper::create_box(
                p_tri.p1, p_tri.p2, p_tri.p3, a_ratio, length_p, width_p);
            const int left = p_b_box.min_x * sqrt_samples;
            const int right = p_b_box.max_x * sqrt_samples;
            const int top = p_b_box.min_y * sqrt_samples;
            const int bot = p_b_box.max_y * sqrt_samples;
            // NOTE: multithreading if uses goes here
            for (int j = top; j < bot; ++j) {
                double world_y = (j + 0.5) * s_pix_to_world_y - 1;
                for (int i = left; i < right; ++i) {
                    double world_x = (i + 0.5) * s_pix_to_world_x - a_ratio;
                    const Eigen::Vector3d test{world_x, world_y, 0};
                    const Eigen::Vector3d bary = engine_helper::get_bary(
                        p_tri.p1, p_tri.p2, p_tri.p3, test);
                    const double alpha = bary[0];
                    const double beta = bary[1];
                    const double gamma = bary[2];
                    const Eigen::Vector3d inter_norm =
                        alpha * n1 + beta * n2 + gamma * n3;
                    const Eigen::Vector3d world_pos =
                        alpha * p1 + beta * p2 + gamma * p3;
                    const Eigen::Vector3d view =
                        (cam_o - world_pos).normalized();
                    Eigen::Vector3d total_color = ambient_term;
                    for (size_t l_index = 0; l_index < num_lights; ++l_index) {
                        auto &c_light = scene.lights[l_index];
                        const seen_buffer &curr_l_s_buff = l_s_buffs[l_index];
                        const Eigen::Vector3d &light_o = c_light->get_o();
                        const Eigen::Vector3d &light_u = c_light->get_u();
                        const Eigen::Vector3d &light_v = c_light->get_v();
                        const Eigen::Vector3d &light_w = c_light->get_w();
                        const double light_f_len = c_light->get_f_len();
                        Eigen::Vector3d proj_point =
                            engine_helper::project_point(world_pos, light_u,
                                                         light_v, light_w,
                                                         light_o, light_f_len);
                        double x = (proj_point[0] + a_ratio) * world_x_to_s_pix;
                        double y = (proj_point[1] + 1.0) * world_y_to_s_pix;
                        int s_pixel_x = static_cast<int>(std::floor(x));
                        int s_pixel_y = static_cast<int>(std::floor(y));
                        if (s_pixel_x < 0 || s_pixel_x >= length ||
                            s_pixel_y < 0 || s_pixel_y >= width) {
                            continue;
                        }
                        tri_ref l_tri = curr_l_s_buff.get(s_pixel_x, s_pixel_y);
                        if (l_tri.mesh_id != mesh_id ||
                            l_tri.tri_index != tri_index) {
                            continue;
                        }
                        const color &l_color = c_light->get_color();
                        Eigen::Vector3d half = (view - light_w).normalized();
                        double n_dot_h = std::max(0.0, inter_norm.dot(half));
                        double n_dot_v = std::max(0.0, inter_norm.dot(view));
                        double n_dot_l =
                            std::max(0.0, inter_norm.dot(-light_w));
                        if (n_dot_l <= 0.0) {
                            continue;
                        }
                        double v_dot_h = std::max(0.0, view.dot(half));
                        double distro = (shine + 2.0) * 0.5 * inv_pi *
                                        std::pow(n_dot_h, shine);
                        double transmission = 1.0 - v_dot_h;
                        double sq_trans = transmission * transmission;
                        double pent_trans = sq_trans * sq_trans * transmission;
                        Eigen::Vector3d frensel =
                            F0.array() + (1.0 - F0.array()) * pent_trans;
                        double facet =
                            2.0 * n_dot_h / std::max(0.0001, v_dot_h);
                        double dir_1 = facet * n_dot_v;
                        // light_w is backwards by convention
                        double dir_2 = facet * n_dot_l;
                        double G = std::min<double>({1, dir_1, dir_2});
                        Eigen::Vector3d spec_brdf =
                            (distro * frensel * G * 0.25) /
                            std::max(0.0001, n_dot_v);
                        Eigen::Vector3d diff_brdf =
                            (Eigen::Vector3d::Ones() - frensel)
                                .cwiseProduct(norm_metal_color) *
                            n_dot_l;
                        Eigen::Vector3d added_light =
                            (diff_brdf + spec_brdf).cwiseProduct(l_color.val);
                        total_color += added_light;
                    }
                    col_buff.set(
                        i, j,
                        color{total_color[0], total_color[1], total_color[2]});
                }
            }
        }
    }
}

void engine::render() {
    const int img_length = scene.get_img_length();
    const int img_height = scene.get_img_height();
    const int sqrt_samples = scene.get_sqrt_samples();
    const int samples = sqrt_samples * sqrt_samples;
    const int inv_samples = 1 / samples;
    color_buffer &color_buff = scene.col_buffer;
    image_buffer &img = scene.img;
    z_buffer &z_buffer_cam = scene.z_buffer_cam;
    seen_buffer &s_buffer_cam = scene.s_buffer_cam;
    std::vector<std::unique_ptr<mesh>> &meshes = scene.meshes;
    vertex_buffer &v_buff = scene.v_buffer;
    std::vector<z_buffer> &z_buffer_lights = scene.z_buffer_lights;
    std::vector<seen_buffer> &s_buffer_lights = scene.s_buffer_lights;
    std::vector<std::unique_ptr<light>> &lights = scene.lights;
    fill_z_s(camera, meshes, v_buff, z_buffer_cam, s_buffer_cam);
    int z_buffer_lights_size = z_buffer_lights.size();
    for (size_t i = 0; i < z_buffer_lights_size; ++i) {
        fill_z_s(*lights[i], meshes, v_buff, z_buffer_lights[i],
                 s_buffer_lights[i]);
    }
    color_buffs();
    // take the average, written this way to be quick
    std::vector<Eigen::Vector3d> run_tot(img_length);
    int img_height_s = img_height * sqrt_samples;
    int img_length_s = img_length * sqrt_samples;
    int pixel_y = 0;
    int sub_y = 0;
    for (int j = 0; j < img_height_s; ++j) {
        int pixel_x = 0;
        int sub_x = 0;
        for (int i = 0; i < img_length_s; ++i) {
            run_tot[pixel_x] += color_buff.get(i, j).val;
            sub_x++;
            if (sub_x == sqrt_samples) {
                pixel_x++;
                sub_x = 0;
            }
        }
        sub_y++;
        if (sub_y == sqrt_samples) {
            for (int x = 0; x < run_tot.size(); ++x) {
                Eigen::Vector3d avg_col =
                    (run_tot[x] * inv_samples).cwiseMin(1.0).cwiseMax(0.0);
                img.set_color(x, pixel_y, avg_col);
            }
            std::fill(run_tot.begin(), run_tot.end(), Eigen::Vector3d::Zero());
            pixel_y++;
            sub_y = 0;
        }
    }
}

void engine::make_cube_map(const mesh &metal_mesh, const int side_len) {}

void engine::make_cube_map_face(const mesh &metal_mesh, const int side_len,
                                Eigen::Vector3d &cam_u, Eigen::Vector3d &cam_v,
                                Eigen::Vector3d &cam_w,
                                const int cube_maps_index) {}

Eigen::Vector3d engine::get_cube_color(const int metal_data,
                                       const Eigen::Vector3d &r_dir,
                                       const Eigen::Vector3d &r_origin) const {
    return Eigen::Vector3d{};
}

void engine::make_quad_map(const mesh &metal_quad) {}
