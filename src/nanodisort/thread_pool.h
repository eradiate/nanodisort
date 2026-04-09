// SPDX-FileCopyrightText: 2026 Rayference
//
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * nanodisort - Thread pool for parallel batch solving
 */

#ifndef NANODISORT_THREAD_POOL_H
#define NANODISORT_THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::future<void> submit(std::function<void()> task);

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

#endif // NANODISORT_THREAD_POOL_H
