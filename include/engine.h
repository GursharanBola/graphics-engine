#ifndef ENGINE_H
#define ENGINE_H

#include "buffer.h"
#include "projector.h"
#include "scene.h"
#include <memory>
#include <vector>
// TODO: add file parsing and video rendering using interpolation

class engine {
  public:
    engine(scene &scene, camera &camera) : scene(scene), camera(camera) {};
    void fill_z_s(const projector &projector,
                  const std::vector<std::unique_ptr<mesh>> &meshes,
                  const vertex_buffer &v_buff, z_buffer &z_buff,
                  seen_buffer &s_buff) const;
    void color_buffs();
    void render();

  private:
    scene &scene;
    camera &camera;
};
#endif
