#ifndef ENGINE_H
#define ENGINE_H

#include "buffer.h"
#include "ds.h"
#include "projector.h"
#include "scene.h"
#include <vector>

// TODO: add cube mapped reflections and maybe file parsing

/*
 * Resource for PBR for rasterization
 * http://www.thetenthplanet.de/archives/255
 */

class engine {
  public:
    engine(scene &scene, camera &scene_cam)
        : scene(scene), scene_cam(scene_cam) {};

    // user will use this to render their scene, calls all children in private
    void render();

  private:
    scene &scene;
    camera &scene_cam;

    // gets the color on a reflection

    // for cubemaps: r_origin is on surface of mesh, r_dir is the surface normal
    Eigen::Vector3d ref_col(const shape &mesh, const Eigen::Vector3d &r_dir,
                            const Eigen::Vector3d &r_origin);

    // makes all of cubemaps for reflective materials
    void make_all_maps();

    // makes a cubemap for any mesh.
    // Cubemaps look like:
    // [front, back, top, bot, left, right] with front being z, top being y
    // and right being x
    // x, cam_u goes right; y, cam_v goes up; z, cam_w comes out of the page
    void make_cubemap(const shape &mesh);

    // calls child to fill all projectors' buffers in the scene
    void fill_all_z_s();

    // calls child to fill all v and s buffers for cubemaps
    void fill_map_v_s();

    // child used to fill in the depth and visibility buffers
    void fill_z_s(const projector &projector, const std::vector<shape> &meshes,
                  const ds::e_cache_map<triangle> &list_of_tris,
                  z_buffer &z_buff, seen_buffer &s_buff) const;

    // fill all color_buffers in the program
    void color_all_buffs();

    // phong normal interpolaton, muliple samples per pixel, and BRDF
    // this function assumes that fill_v_s run on all buffers.
    // colors a buffer, can be cubemap or final buffer
    void color_buff(const camera &cam, const bool is_map,
                    const seen_buffer &cam_s_buff, color_buffer &col_buff);
};
#endif
