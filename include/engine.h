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
    engine(scene &scene, camera &cam) : scene(scene), cam(cam) {};

    // user will use this to render their scene, calls all children in private
    void render();

  private:
    scene &scene;
    camera &cam;

    // gets the color on a reflection, handles both cubemaps and quadmaps
    double ref_col(const shape &mesh, const Eigen::Vector3d &r_dir,
                   const Eigen::Vector3d &r_origin);

    // makes a cubemap for any mesh with a volume. Cubemaps look like:
    // [front, back, top, bot, left, right] w/ front being -cam_w, top being
    // cam_v, and right being cam_u.
    // note -cam_w is right, cam_v is up, cam_u is out of page
    void make_cubemap(const mesh &mesh, const int side_len);

    // makes projection planes for two quads
    void make_quadmap(const mesh &quad);

    // makes a single face of the cubemap assumes v, s are populted
    void make_face(const mesh &metal_mesh, const int side_len,
                   Eigen::Vector3d &cam_u, Eigen::Vector3d &cam_v,
                   Eigen::Vector3d &cam_w, const int);

    // calls child to fill all projectors' buffers in the scene
    void fill_all_z_s();

    // calls child to fill all v and s buffers for cube/quadmaps
    void fill_ref_v_s();

    // child used to fill in the depth and visibility buffers
    void fill_z_s(const projector &projector, const std::vector<shape> &meshes,
                  const ds::e_cache_map<triangle> &list_of_tris,
                  z_buffer &z_buff, seen_buffer &s_buff) const;

    // phong normal interpolaton, muliple samples per pixel, and BRDF
    // this function assumes that fill_v_s run on all buffers as well as
    // all cube mapped reflections are finished
    void color_buffs();
};
#endif
