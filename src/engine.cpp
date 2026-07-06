#include "engine.h"
#include "buffer.h"
#include "consts.h"
#include "engine_helper.h"
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
                    bound_box<int> p_b_box = engine_helper::create_box(
                        p_tri.p1, p_tri.p2, p_tri.p3, a_ratio, length, width);
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

// TODO: debug everything that was changed reltaed to implimenting this function
void engine::shade() {
    const int length_p = scene.get_img_length();
    const int width_p = scene.get_img_height();
    const int sqrt_samples = scene.get_sqrt_samples();
    const int num_lights = scene.s_buffer_lights.size();
    const int sqrt_tile = consts::TILE_SIZE;
    seen_buffer s_cam_tile = seen_buffer(sqrt_tile, sqrt_tile, sqrt_samples);
    color_buffer color_tile = color_buffer(sqrt_tile, sqrt_tile, sqrt_samples);

    constexpr int NUM_THREADS = consts::NUM_THREADS;
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    // TODO: add thread loop for multithreading
    const int num_col_tiles = (length_p * sqrt_tile - 1) / sqrt_tile;
    const int num_row_tiles = (width_p * sqrt_tile - 1) / sqrt_tile;
    for (int off_x = 0; off_x < num_col_tiles; ++off_x) {
        for (int off_y = 0; off_y < num_row_tiles; ++off_y) {
            engine_helper::pull(scene.s_buffer_cam, s_cam_tile, off_x, off_y);
            engine_helper::pull(scene.col_buffer, color_tile, off_x, off_y);
            engine_helper::shade_buff(scene.s_buffer_lights, scene.lights,
                                      s_cam_tile, color_tile,
                                      scene.get_ambient_light(), off_x, off_y);
            engine_helper::push(scene.s_buffer_cam, s_cam_tile, off_x, off_y);
            engine_helper::pull(scene.col_buffer, color_tile, off_x, off_y);
        }
    }
}
