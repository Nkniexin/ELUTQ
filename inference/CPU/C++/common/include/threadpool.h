
//
// Adapated from fastllm
//
#pragma once
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <thread>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <cstring>

static void barrier() {
#ifdef __aarch64__
        asm volatile("dmb ish");
#elif defined(_WIN32) || defined(_WIN64)
        MemoryBarrier();
#else
        __asm__ __volatile__("": : :"memory");
#endif
}


struct MultiThreadBaseOp {
    virtual void Run() = 0;
};

struct AliveThreadTask {
    int signal;
    MultiThreadBaseOp *op;

    AliveThreadTask () {
        signal = 0;
        op = nullptr;
    }
};

struct AliveThreadLoop {
    int id;
    AliveThreadTask realTask;
    volatile AliveThreadTask *task;

    AliveThreadLoop(int id)  {
        this->id = id;
        this->task = &this->realTask;
    }

    void operator()() {
        int cnt = 0;
        auto lastRunTime = std::chrono::system_clock::now();
        while (true) {
            barrier();
            if (task->signal == 1) {
                task->op->Run();
                task->signal = 0;
                barrier();
                lastRunTime = std::chrono::system_clock::now();
            }

            cnt = (cnt + 1) & ((1 << 16) - 1);
            // if (cnt == 0) {
            //     auto duration = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::system_clock::now() - lastRunTime);
            //     double gap = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
            //     if (gap > 3) {
            //         std::this_thread::sleep_for(std::chrono::microseconds(2000));
            //     }
            // }
        }
    }

    void PushOp(MultiThreadBaseOp *op) {
        this->task->op = op;
        barrier();
        this->task->signal = 1;
        barrier();
    }

    void Wait() {
        while (true) {
            int a = task->signal;
            if (a == 0) {
                break;
            }
        }
    }
};

struct AliveThreadPool {
    std::pair <int, int> curActivateThreadInterval; // 设定当前激活 [curActivateThreadInterval.first, curActivateThreadInterval.second) 的线程

    std::vector <AliveThreadLoop*> loops;
    std::vector <std::thread*> threads;
    
    AliveThreadPool (int threadNum) {
        for (int i = 0; i < threadNum; i++) {
            this->loops.push_back(new AliveThreadLoop(i));
            this->threads.push_back(new std::thread(*(this->loops[i])));
        }
        curActivateThreadInterval = std::make_pair(0, threadNum);
    }

    void PushOp(int tid, MultiThreadBaseOp *op) {
        this->loops[tid]->PushOp(op);
    }

    void Wait(int tid) {
        this->loops[tid]->Wait();
    }

    void Shutdown() {
        /// TODO: shutdown
    }
};

struct MultiThreadMultiOps : MultiThreadBaseOp {
    std::vector <MultiThreadBaseOp*> ops;

    void Run() {
        for (int i = 0; i < ops.size(); i++) {
            ops[i]->Run();
        }
    }

    ~MultiThreadMultiOps() {
        for (int i = 0; i < ops.size(); i++) {
            delete[] ops[i];
        }
    }
};

#endif // THREADPOOL_H