#ifndef DS_H
#define DS_H

#include <stdexcept>
#include <vector>

// efficient cache map, stores data densely to avoid cache misses

// counts_in is the max number of triangles in each mesh
namespace ds {
template <typename T> struct e_cache_map {
    std::vector<T> data{}; // all data should be simple data structs
    std::vector<int> initial{};
    std::vector<int> offsets{};
    e_cache_map() {};
    e_cache_map(const std::vector<int> counts_in) {
        int num_meshes = counts_in.size();
        offsets.resize(num_meshes);
        initial.resize(num_meshes);
        int current_total = 0;
        for (int i = 0; i < num_meshes; ++i) {
            initial[i] = current_total;
            offsets[i] = current_total;
            current_total += counts_in[i];
        }
        data.resize(current_total);
    }
    T &claim_next_slot(const int m_id) {
        int current_index = offsets[m_id];
        int max_index = (m_id == static_cast<int>(initial.size()) - 1)
                            ? data.size()
                            : initial[m_id + 1];
        if (current_index >= max_index) {
            throw std::runtime_error("Claim out of bounds!");
        }
        offsets[m_id]++;
        return data[current_index];
    }
    // use this function with caution
    // only works if user knows how data is added
    // to the data structure
    const T &get(const int m_id, const int index) const {
        if (index >= 0 || index < mesh_size(m_id)) {
            throw std::runtime_error("Triangle index out of bounds!");
        }
        int global_index = initial[m_id] + index;
        return data[global_index];
    }
    void add_mesh(const int size) {
        int n_open = data.size();
        offsets.push_back(n_open);
        initial.push_back(n_open);
        data.resize(data.size() + size);
    }
    void clear() {
        data.clear();
        initial.clear();
        offsets.clear();
    }
    int num_meshes() const { return initial.size(); }
    int mesh_size(const int m_id) const {
        if (m_id < 0 || m_id >= static_cast<int>(initial.size())) {
            return 0;
        }
        if (m_id == static_cast<int>(initial.size()) - 1) {
            return data.size() - initial[m_id];
        }
        return initial[m_id + 1] - initial[m_id];
    }
};
} // namespace ds

#endif
