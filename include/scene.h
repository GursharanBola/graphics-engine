#ifndef SCENE_H
#define SCENE_H

#include "buffer.h"
#include "material.h"
#include "mesh.h"
#include "projector.h"
#include <memory>

/*
 * Scenes are a container for the buffers that the program uses. Also note
 * that since the program renders one color at a time and one mesh at a time
 * only one colorbuffer is needed
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
                    const color &mesh_color,
                    const std::shared_ptr<material> &mat,
                    const int num_samples) {
        std::shared_ptr<mesh> new_sphere = std::make_shared<sphere>(
            meshes.size(), center, radius, mesh_color, mat, num_samples);

        new_sphere->build(v_buffer);
        meshes.push_back(new_sphere);
    }

    void add_quad(const Eigen::Vector3d &origin, const Eigen::Vector3d &u,
                  const Eigen::Vector3d &v, const color &mesh_color,
                  const std::shared_ptr<material> &mat) {
        std::shared_ptr<mesh> new_quad = std::make_shared<quad>(
            meshes.size(), origin, u, v, mesh_color, mat);

        new_quad->build(v_buffer);
        meshes.push_back(new_quad);
    }

    void add_light(const Eigen::Vector3d &origin, const Eigen::Vector3d &cam_u,
                   const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
                   const color &light_color, const double I_d, const double I_s,
                   const double focal_dist) {
        std::shared_ptr<light> new_light = std::make_shared<light>(
            origin, cam_u, cam_v, cam_w, light_color, focal_dist);

        z_buffer_lights.emplace_back(img_length, img_height, sqrt_samples);
        s_buffer_lights.emplace_back(img_length, img_height, sqrt_samples);
        lights.push_back(new_light);
    }

    void add_light_s_buff() {
        s_buffer_lights.emplace_back(img_length, img_height, sqrt_samples);
    }

    void add_light_z_buff() {
        z_buffer_lights.emplace_back(img_length, img_height, sqrt_samples);
    }

    void clear_scene() {
        meshes.clear();
        lights.clear();
        z_buffer_cam.clear();
        s_buffer_cam.clear();
        img.clear();
        v_buffer.clear();
        z_buffer_lights.clear();
        s_buffer_lights.clear();
        ambient_color = color{0, 0, 0};
    }

    int get_img_length() const { return img_length; }
    int get_img_height() const { return img_height; }
    int get_num_channels() const { return num_channels; }
    int get_sqrt_samples() const { return sqrt_samples; }

  protected:
    image_buffer img;
    color_buffer col_buffer;
    vertex_buffer v_buffer;
    z_buffer z_buffer_cam;
    seen_buffer s_buffer_cam;
    std::vector<z_buffer> z_buffer_lights;
    std::vector<seen_buffer> s_buffer_lights;
    std::vector<std::shared_ptr<mesh>> meshes;
    std::vector<std::shared_ptr<light>> lights;
    color ambient_color;
};

#endif
