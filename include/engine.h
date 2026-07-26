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

    // gets the color on a reflection, handles both cubemaps and quadmaps.
    // in the case for quad maps, r_dir is either the surface normal
    // or -surface normal, and r_origin is the pt in world_coords (not at
    // origin).
    // for cube maps r_origin must be within bounding bcube of mesh
    Eigen::Vector3d ref_col(const shape &mesh, const Eigen::Vector3d &r_dir,
                            const Eigen::Vector3d &r_origin);

    // makes all of cube / quad maps for reflective materials
    void make_all_maps();

    // makes a cubemap for any mesh with a volume. Cubemaps look like:
    // [front, back, top, bot, left, right] with front being x, top being y
    // and left being z
    // x, cam_u is out of page; y, cam_v is up; z, cam_w is left
    void make_cubemap(const shape &mesh);

    // makes projection planes for two quads
    // [font, back] is the way this stored
    void make_quadmap(const shape &quad);

    // calls child to fill all projectors' buffers in the scene
    void fill_all_z_s();

    // calls child to fill all v and s buffers for cube/quadmaps
    void fill_ref_v_s();

    // child used to fill in the depth and visibility buffers
    void fill_z_s(const projector &projector, const std::vector<shape> &meshes,
                  const ds::e_cache_map<triangle> &list_of_tris,
                  z_buffer &z_buff, seen_buffer &s_buff) const;

    // color all of the buffers cube/quad maps and the final output buffer
    void color_all_buffs();

    // phong normal interpolaton, muliple samples per pixel, and BRDF
    // this function assumes that fill_v_s run on all buffers.
    // colors a buffer, can be cubemap or final buffer
    void color_buff(const camera &cam, const bool is_map,
                    const seen_buffer &cam_s_buff, color_buffer &col_buff);
};
#endif
