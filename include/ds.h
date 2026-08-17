#ifndef DS_H
#define DS_H

#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ds {
template <typename T> struct job {
    job(const T job_data) : job_data(job_data), completed(false) {}
    void mark_completed() { completed = true; }
    T job_data;
    bool completed;
};

// efficient cache map, stores data densely to avoid cache misses

// counts_in is the max number of triangles in each mesh
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
    T &claim_next_slot(const int id) {
        int current_index = offsets[id];
        int max_index = (id == static_cast<int>(initial.size()) - 1)
                            ? data.size()
                            : initial[id + 1];
        if (current_index >= max_index) {
            throw std::runtime_error("Claim out of bounds!");
        }
        offsets[id]++;
        return data[current_index];
    }
    // use this function with caution
    // only works if user knows how data is added
    // to the data structure
    const T &get(const int id, const int index) const {
        if (index >= 0 || index < obj_size(id)) {
            throw std::runtime_error("Triangle index out of bounds!");
        }
        int global_index = initial[id] + index;
        return data[global_index];
    }
    void add_obj(const int size) {
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
    int num_objs() const { return initial.size(); }
    int obj_size(const int id) const {
        if (id < 0 || id >= static_cast<int>(initial.size())) {
            return 0;
        }
        if (id == static_cast<int>(initial.size()) - 1) {
            return data.size() - initial[id];
        }
        return initial[id + 1] - initial[id];
    }
    bool is_full(const int id) const {
        if (id == static_cast<int>(initial.size()) - 1) {
            return offsets[id] == data.size();
        }
        return offsets[id] == initial[id + 1];
    }
};

// this is a thread pool implimentation without work stealing

// load imbalances will kill this optimization but works well
// on dense scenes, user should consider writing a load balancer
template <typename T> struct thread_pool {
  public:
    std::optional<e_cache_map<job<T>>> jobs;
    std::optional<std::vector<std::thread>> threads;
    thread_pool(const int num_threads, const int num_jobs)
        : num_threads(num_threads), num_jobs(num_jobs),
          jobs_per_thread(num_jobs / num_threads), open_thread(0) {

        std::vector<int> counts_in(num_threads, jobs_per_thread);
        if (num_jobs % num_threads != 0) {
            counts_in.back() += num_jobs % num_threads;
        }

        jobs = e_cache_map<job<T>>(counts_in);
        threads = std::vector<std::thread>(num_threads);
    }

    void add_job(T &job_data) {
        if (jobs->is_full(open_thread)) {
            open_thread += 1;
        }
        auto &new_job = jobs->claim_next_slot(open_thread);
        new_job = job<T>(job_data);
    }

    template <typename F> void execute(F &&work_function) {
        for (int t = 0; t < num_threads; ++t) {
            (*threads)[t] = std::thread([this, t, work_function]() {
                int start_idx = jobs->initial[t];
                int end_idx = jobs->offsets[t];

                for (int i = start_idx; i < end_idx; ++i) {
                    work_function(jobs->data[i].job_data);
                    jobs->data[i].mark_completed();
                }
            });
        }

        for (auto &th : *threads) {
            if (th.joinable()) {
                th.join();
            }
        }
    }

  private:
    int num_threads;
    int num_jobs;
    int jobs_per_thread;

    int open_thread;
};

} // namespace ds

#endif
