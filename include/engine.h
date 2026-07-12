#ifndef ENGINE_H
#define ENGINE_H

#include "buffer.h"
#include "projector.h"
#include "scene.h"
#include <memory>
#include <vector>
// TODO: add cube based reflections and file parsing

class engine {
  public:
    engine(scene &scene, camera &camera) : scene(scene), camera(camera) {};
    void fill_z_s(const projector &projector,
                  const std::vector<std::shared_ptr<mesh>> &meshes,
                  const vertex_buffer &v_buff, z_buffer &z_buff,
                  seen_buffer &s_buff) const;
    void color_buffs();
    void render();

  private:
    scene &scene;
    camera &camera;
};
#endif
