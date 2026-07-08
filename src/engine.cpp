#include "engine.h"
#include "buffer.h"
#include "consts.h"
#include "engine_helper.h"
#include "projector.h"
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

void engine::fill_z_s(const projector &projector,
                      const std::vector<std::unique_ptr<mesh>> &meshes,
                      const vertex_buffer &v_buff, z_buffer &z_buff,
                      seen_buffer &s_buff) const {
    const Eigen::Vector3d cam_u = projector.get_u();
    const Eigen::Vector3d cam_v = projector.get_v();
    const Eigen::Vector3d cam_w = projector.get_w();
    const Eigen::Vector3d cam_o = projector.get_o();
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
    engine_helper::ra_tri_buffs<tri_ref> buffs{s_buff, z_buff};
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
                        new_tri, cam_u, cam_v, cam_w, cam_o);
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
    Eigen::Vector3d cam_o = camera.get_o();
    Eigen::Vector3d cam_u = camera.get_u();
    Eigen::Vector3d cam_v = camera.get_v();
    Eigen::Vector3d cam_w = camera.get_w();
    const int length_p = scene.get_img_length();
    const int width_p = scene.get_img_height();
    const int sqrt_samples = scene.get_sqrt_samples();
    const int length = length_p * sqrt_samples;
    const int width = width_p * sqrt_samples;
    const double a_ratio = static_cast<double>(length_p) / width_p;
    const double s_pix_to_world_x = 2.0 * a_ratio / length;
    const double s_pix_to_world_y = 2.0 / width;
    const double world_y_to_s_pix = width / 2.0;
    const double world_x_to_s_pix = length / (2.0 * a_ratio);
    const double ambient = scene.ambient_light;
    vertex_buffer &v_buff = scene.v_buffer;
    seen_buffer &cam_s_buff = scene.s_buffer_cam;
    color_buffer &col_buff = scene.col_buffer;
    std::vector<seen_buffer> &l_s_buffs = scene.s_buffer_lights;
    int mesh_id = 0;

    for (auto &mesh : scene.meshes) {
        const double k_a = mesh->mat->k_a();
        const double k_d = mesh->mat->k_d();
        const double k_s = mesh->mat->k_s();
        const color base_color = mesh->get_color();
        int tri_index = 0;
        for (const triangle &tri : mesh->list_of_triangles) {
            const Eigen::Vector3d p1 = v_buff.get(tri.point1);
            const Eigen::Vector3d p2 = v_buff.get(tri.point2);
            const Eigen::Vector3d p3 = v_buff.get(tri.point3);
            const Eigen::Vector3d n1 = mesh->find_normal(p1);
            const Eigen::Vector3d n2 = mesh->find_normal(p2);
            const Eigen::Vector3d n3 = mesh->find_normal(p3);
            const double shine = mesh->mat->shine;
            raw_tri new_tri = raw_tri{p1, p2, p3};
            raw_tri p_tri =
                engine_helper::proj_tri(new_tri, cam_u, cam_v, cam_w, cam_o);
            bound_box<int> p_b_box = engine_helper::create_box(
                p_tri.p1, p_tri.p2, p_tri.p3, a_ratio, length_p, width_p);
            const int left = p_b_box.min_x * sqrt_samples;
            const int right = p_b_box.max_x * sqrt_samples;
            const int top = p_b_box.min_y * sqrt_samples;
            const int bot = p_b_box.max_y * sqrt_samples;

            for (int i = left; i < right; ++i) {
                double world_x = i * s_pix_to_world_x - a_ratio;
                for (int j = top; j < bot; ++j) {
                    tri_ref seen_tri = cam_s_buff.get(i, j);
                    if (seen_tri.mesh_id != mesh_id ||
                        seen_tri.tri_index != tri_index) {
                        continue;
                    }
                    double world_y = j * s_pix_to_world_y - 1;
                    Eigen::Vector3d test{world_x, world_y, 0};
                    Eigen::Vector3d bary = engine_helper::get_bary(
                        p_tri.p1, p_tri.p2, p_tri.p3, test);
                    double alpha = bary[0];
                    double beta = bary[1];
                    double gamma = bary[2];
                    Eigen::Vector3d inter_norm =
                        alpha * n1 + beta * n2 + gamma * n3;
                    Eigen::Vector3d world_pos =
                        alpha * p1 + beta * p2 + gamma * p3;
                    Eigen::Vector3d ambient_comp =
                        k_a * ambient * base_color.val;
                    Eigen::Vector3d total_color =
                        scene.l_color.val + ambient_comp;

                    for (size_t l_index = 0; l_index < scene.lights.size();
                         ++l_index) {
                        auto &c_light = scene.lights[l_index];
                        seen_buffer &curr_l_s_buff = l_s_buffs[l_index];
                        const Eigen::Vector3d light_o = c_light->get_o();
                        const Eigen::Vector3d light_u = c_light->get_u();
                        const Eigen::Vector3d light_v = c_light->get_v();
                        const Eigen::Vector3d light_w = c_light->get_w();
                        Eigen::Vector3d proj_point =
                            engine_helper::project_point(
                                world_pos, light_u, light_v, light_w, light_o);
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
                        const double I_d = c_light->get_I_d();
                        const double I_s = c_light->get_I_s();
                        const Eigen::Vector3d &l_cam_w = c_light->get_w();
                        const color &l_color = c_light->get_color();
                        Eigen::Vector3d reflect =
                            2 * inter_norm.dot(l_cam_w) * inter_norm - l_cam_w;
                        Eigen::Vector3d v = (cam_o - world_pos).normalized();
                        double dot_w_l = std::max(inter_norm.dot(l_cam_w), 0.0);
                        double dot_w_v =
                            std::pow(std::max(reflect.dot(v), 0.0), shine);
                        Eigen::Vector3d diffuse_comp =
                            k_d * I_d * dot_w_l *
                            (l_color.val).cwiseProduct(base_color.val);
                        Eigen::Vector3d specular_comp =
                            k_s * I_s * dot_w_v * l_color.val;
                        total_color += diffuse_comp + specular_comp;
                    }
                    col_buff.set(
                        i, j,
                        color{total_color[0], total_color[1], total_color[2]});
                }
            }
            tri_index++;
        }
        mesh_id++;
    }
};

void engine::render() {}
