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

    // child used to fill in the depth and visibility buffers
    void fill_z_s(const projector &projector,
                  const std::vector<std::shared_ptr<mesh>> &meshes,
                  const vertex_buffer &v_buff, z_buffer &z_buff,
                  seen_buffer &s_buff) const;

    // child runs phong shading as well as determing shadows
    void color_buffs();

    // adds cubes mapped reflections
    void add_cube();

    // user will use this to render their scene, calls all children
    void render();

  private:
    scene &scene;
    camera &camera;
};
#endif
