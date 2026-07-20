#ifndef ENGINE_H
#define ENGINE_H

#include "buffer.h"
#include "projector.h"
#include "scene.h"
#include <memory>
#include <vector>
// TODO: add cube mapped reflections and maybe file parsing

/*
 * Resource for PBR for rasterization
 * http://www.thetenthplanet.de/archives/255
 */

class engine {
  public:
    engine(scene &scene, camera &camera) : scene(scene), camera(camera) {};

    // user will use this to render their scene, calls all children in private
    void render();

  private:
    scene &scene;
    camera &camera;

    // calls child to fill all buffers in the scene
    void fill_all_z_s();

    // child used to fill in the depth and visibility buffers
    void fill_z_s(const projector &projector,
                  const std::vector<std::unique_ptr<mesh>> &meshes,
                  const vertex_buffer &v_buff, z_buffer &z_buff,
                  seen_buffer &s_buff) const;

    // phong normal interpolaton, muliple samples per pixel, and BRDF
    // this function assumes that fill_v_s run on all buffers as well as
    // all cube mapped reflections are finished
    void color_buffs();

    // makes a cube map for any mesh with a volume
    void make_cube_map(const mesh &metal_mesh, const int side_len);

    // makes a projection plane for a quad, simply uses make_cube_map_face
    void make_quad_map(const mesh &metal_quad);

    // makes a single face of the cube map
    // this program assumes that it's seen_buffer and depth buffers have already
    // been finished
    void make_cube_map_face(const mesh &metal_mesh, const int side_len,
                            Eigen::Vector3d &cam_u, Eigen::Vector3d &cam_v,
                            Eigen::Vector3d &cam_w, const int cube_maps_index);

    // gets the color on a cube map associated mesh with a volume given a ray
    Eigen::Vector3d get_cube_color(const int metal_data,
                                   const Eigen::Vector3d &r_dir,
                                   const Eigen::Vector3d &r_origin) const;
};
#endif
