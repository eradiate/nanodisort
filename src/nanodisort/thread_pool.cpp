// SPDX-FileCopyrightText: 2025 Rayference
//
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * nanodisort - Thread pool for parallel batch solving
 */

#include "thread_pool.h"

ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty())
                        return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_)
        worker.join();
}

std::future<void> ThreadPool::submit(std::function<void()> task) {
    auto packaged = std::make_shared<std::packaged_task<void()>>(std::move(task));
    auto future = packaged->get_future();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        tasks_.emplace([packaged] { (*packaged)(); });
    }
    cv_.notify_one();
    return future;
}
