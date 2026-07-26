#ifndef SCENE_H
#define SCENE_H

#include "buffer.h"
#include "consts.h"
#include "ds.h"
#include "material.h"
#include "mesh.h"
#include "projector.h"

/*
 * Scenes are a container for the buffers that the program uses. Also note
 * that since the program renders s_pixel at a time after all preprocessing is
 * done.
 */

/*
 makes a cubemap for any mesh with a volume. Cubemaps look like:
 [front, back, top, bot, left, right] w/ front being -cam_w, top being
 cam_v, and right being cam_u.
 note -cam_w is right, cam_v is up, cam_u is out of page
*/

class scene {
  private:
    int img_length;
    int img_height;
    int num_channels;
    int sqrt_samples;
    friend class engine;

  public:
    scene(const int img_length, const int img_height, const int num_channels,
          const int sqrt_samples, const color l_color,
          const color &ambient_color)
        : img_length(img_length), img_height(img_height),
          num_channels(num_channels), sqrt_samples(sqrt_samples),
          img(img_length, img_height, num_channels),
          col_buffer(img_length, img_height, sqrt_samples),
          z_buffer_cam(img_length, img_height, sqrt_samples),
          s_buffer_cam(img_length, img_height, sqrt_samples),
          ambient_color(ambient_color) {};

    void add_sphere(const Eigen::Vector3d &center, const double radius,
                    const color &mesh_color, material &mat,
                    const int num_samples) {
        if (mat.is_metal) {
            mat.metal_data = cubemaps.size();
            mat.metal_faces = 6;

            const int side_len = 2 * radius * consts::CUBE_MAP_PIXEL_DENSITY;
            for (int i = 0; i < 6; ++i) {
                cubemaps.emplace_back(color_buffer{side_len, side_len, 1});
                cubemaps_z.emplace_back(z_buffer{side_len, side_len, 1});
                cubemaps_s.emplace_back(seen_buffer{side_len, side_len, 1});
            }
        }
        mats.emplace_back(std::move(mat));
        meshes.emplace_back(sphere{(int)meshes.size(), center, radius,
                                   num_samples, list_of_tri});
    }

    void add_quad(const Eigen::Vector3d &origin, const Eigen::Vector3d &u,
                  const Eigen::Vector3d &v, const color &mesh_color,
                  material &mat) {
        if (mat.is_metal) {
            mat.metal_data = cubemaps.size();
            mat.metal_faces = 2;
            int len_p = std::ceil(u.norm() * consts::CUBE_MAP_PIXEL_DENSITY);
            int wid_p = std::ceil(v.norm() * consts::CUBE_MAP_PIXEL_DENSITY);
            for (int i = 0; i < 2; ++i) {
                cubemaps.emplace_back(color_buffer{len_p, wid_p, 1});
                cubemaps_z.emplace_back(z_buffer{len_p, wid_p, 1});
                cubemaps_s.emplace_back(seen_buffer{len_p, wid_p, 1});
            }
        }
        mats.emplace_back(std::move(mat));
        meshes.emplace_back(
            quad{(int)meshes.size(), origin, u, v, list_of_tri});
    }

    void add_light(const Eigen::Vector3d &origin, const Eigen::Vector3d &cam_u,
                   const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
                   const color &light_color, const double focal_len) {
        lights.emplace_back(origin, cam_u, cam_v, cam_w, light_color,
                            focal_len);
        s_buffer_lights.emplace_back(img_length, img_height, sqrt_samples);
        z_buffer_lights.emplace_back(img_length, img_height, sqrt_samples);
    }

    void clear_scene() {
        meshes.clear();
        lights.clear();
        z_buffer_cam.clear();
        s_buffer_cam.clear();
        img.clear();
        z_buffer_lights.clear();
        s_buffer_lights.clear();
        ambient_color = color{0, 0, 0};
    }

    int get_img_length() const { return img_length; }
    int get_img_height() const { return img_height; }
    int get_num_channels() const { return num_channels; }
    int get_sqrt_samples() const { return sqrt_samples; }

  protected:
    std::vector<shape> meshes;
    // all triangles are in this e_cache_map
    ds::e_cache_map<triangle> list_of_tri;
    std::vector<material> mats;
    std::vector<light> lights;

    image_buffer img;
    color_buffer col_buffer;
    z_buffer z_buffer_cam;
    seen_buffer s_buffer_cam;
    std::vector<z_buffer> z_buffer_lights;
    std::vector<seen_buffer> s_buffer_lights;
    std::vector<color_buffer> cubemaps;
    std::vector<z_buffer> cubemaps_z;
    std::vector<seen_buffer> cubemaps_s;
    color ambient_color;
};

#endif
