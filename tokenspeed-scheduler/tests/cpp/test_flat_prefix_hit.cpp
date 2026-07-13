// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

// Prefix-cache HIT coverage for the real 397B flat pool shape. Every other flat
// scheduler test runs with disable_prefix_cache=true, so cross-request reuse on
// the flat path was never exercised. These tests use the actual published group
// shape and assert the exact hit boundary for the serving scenarios that matter.
//
// Real 397B shape (paged_cache_spec.group_specs_from_layer_types):
//   - one History "full" group (retention=FullHistory)
//   - k State shards "state_shard{i}" (retention=FullHistory, family=State),
//     the linear_attention_shard* mamba-state groups. Real 397B uses k=4.
//   (Pure-GDN 397B has no sliding-window group: GDN state rides full_history.)
//
// Hit-boundary rule the flat coordinator implements (mirrors radix):
//   * the last prompt token is always recomputed for logits, so a length-L
//     prompt caps the match at floor((L-1)/block)*block tokens;
//   * an unaligned tail (partial final page) is never registered, so it is
//     simply excluded -- the aligned pages below it still hit.

#if TOKENSPEED_FLAT_KVCACHE

#include <algorithm>
#include <string>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace flat_prefix_hit {

constexpr std::int32_t kNumShards = 4;  // real 397B state-shard fan-out
constexpr std::int32_t kBlock = 2;      // small block keeps hit math legible

const FlatForwardOperation* FindFlatOp(const ExecutionPlan& plan) {
    for (const auto& op : plan.Operations()) {
        if (const auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
    }
    return nullptr;
}

PagedCacheGroupConfig MakeGroup(const std::string& id, std::int32_t block_size, std::int32_t total_pages,
                                PagedCacheGroupConfig::Retention retention, PagedCacheGroupFamily family) {
    PagedCacheGroupConfig g;
    g.group_id = id;
    g.rows_per_page = block_size;
    g.entry_stride_tokens = 1;
    g.total_pages = total_pages;
    g.retention = retention;
    g.family = family;
    return g;
}

// full(History) + k x state_shard{i}(State/FullHistory), prefix caching ON.
class FlatRealShapeHitSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.block_size = kBlock;
        cfg.device_allocator.total_pages = 128;
        cfg.host_allocator.total_pages = 128;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = false;
        cfg.paged_cache_groups = {MakeGroup("full", cfg.block_size, cfg.device_allocator.total_pages,
                                            PagedCacheGroupConfig::Retention::FullHistory,
                                            PagedCacheGroupFamily::History)};
        for (std::int32_t i = 0; i < kNumShards; ++i) {
            cfg.paged_cache_groups.push_back(MakeGroup("state_shard" + std::to_string(i), cfg.block_size,
                                                       cfg.device_allocator.total_pages,
                                                       PagedCacheGroupConfig::Retention::FullHistory,
                                                       PagedCacheGroupFamily::State));
        }
        return cfg;
    }

    // Prefill -> finalize (registers page hashes) -> one decode -> finish. Finish
    // reaps the request but its page hashes stay matchable for a later request.
    void RunLifecycle(const RequestSpec& spec) {
        Submit(spec);
        ASSERT_NE(FindFlatOp(PlanOnce()), nullptr);
        SendForwardDone(spec.request_id, {9001});
        PlanOnce();
        SendForwardDone(spec.request_id, {9002});
        SendFinish(spec.request_id);
        PlanOnce();
    }

    // The single flat prefill op's prefix-hit length (tokens) for one request.
    // The FlatForwardOperation is owned by its ExecutionPlan, so the plan must
    // outlive the op pointer -- keep it in a local, never a temporary.
    std::int32_t SubmitAndGetHit(const RequestSpec& spec) {
        Submit(spec);
        ExecutionPlan plan = PlanOnce();
        const FlatForwardOperation* op = FindFlatOp(plan);
        EXPECT_NE(op, nullptr);
        if (op == nullptr) return -1;
        if (op->extend_prefix_lens.empty()) return 0;  // no extend recorded == no hit
        return op->extend_prefix_lens.at(0);
    }

    // Expected hit for an identical resubmit of a length-L prompt: all but the
    // last token, floored to a whole page (last token recomputed for logits).
    static std::int32_t ExpectedIdenticalHit(std::int32_t len_tokens) {
        return ((len_tokens - 1) / kBlock) * kBlock;
    }
};

// A cached model server's most common hit: the exact same request arrives again
// (retry, identical system prompt, etc.). Must reuse all but the recompute tail.
TEST_F(FlatRealShapeHitSuite, IdenticalResubmitAlignedPrompt) {
    const std::int32_t pages = 8;  // 16 tokens, page-aligned
    RunLifecycle(MakeRequestSpec("r1", pages));
    const std::int32_t hit = SubmitAndGetHit(MakeRequestSpec("r2", pages));
    EXPECT_EQ(hit, ExpectedIdenticalHit(pages * kBlock))
        << "identical aligned resubmit must reuse every page but the recompute tail";
}

// The exact 1032-token engine scenario: prompt length is NOT a page multiple
// (N full pages + a partial tail). The aligned pages must still hit; only the
// partial tail is excluded. A "whole end must be page-aligned to register" bug
// collapses the state group to zero here.
TEST_F(FlatRealShapeHitSuite, IdenticalResubmitUnalignedPrompt) {
    token_vec_t prompt = MakeAlignedTokens(/*num_pages=*/8, PageSize());  // 16 tok aligned
    prompt.push_back(777);                                                // + partial tail -> 17 (unaligned)
    RunLifecycle(RequestSpec{.request_id = "r1", .tokens = prompt});
    const std::int32_t hit = SubmitAndGetHit(RequestSpec{.request_id = "r2", .tokens = prompt});
    EXPECT_EQ(hit, ExpectedIdenticalHit(static_cast<std::int32_t>(prompt.size())))
        << "unaligned-length resubmit must still reuse the aligned prefix pages "
           "(the 1032-token cached_tokens=0 regression collapses this to 0)";
}

// Batched serving: many requests share a long common prefix (shared system
// prompt) then diverge in their user turn. The shared pages must be reused.
TEST_F(FlatRealShapeHitSuite, SharedSystemPromptThenDivergingTurn) {
    const std::int32_t shared_pages = 8;
    RunLifecycle(MakeRequestSpec("r1", shared_pages));  // register the shared prefix

    token_vec_t r2 = MakeAlignedTokens(shared_pages, PageSize());  // same 8 pages
    const token_vec_t user_turn = MakeTokens(/*count=*/6, /*start=*/901);
    r2.insert(r2.end(), user_turn.begin(), user_turn.end());
    const std::int32_t hit = SubmitAndGetHit(RequestSpec{.request_id = "r2", .tokens = r2});
    EXPECT_EQ(hit, shared_pages * kBlock)
        << "shared-prefix request must reuse the whole shared prefix (it is not "
           "the request's own tail, so no recompute cap applies to it)";
}

// A later request that is a strict PREFIX of the registered one (shorter). It
// reuses up to its own recompute cap, never past its own length.
TEST_F(FlatRealShapeHitSuite, ShorterPrefixOfRegistered) {
    RunLifecycle(MakeRequestSpec("r1", /*pages=*/12));  // 24 tokens registered
    const std::int32_t short_pages = 5;                 // 10-token prefix of r1
    const std::int32_t hit = SubmitAndGetHit(MakeRequestSpec("r2", short_pages));
    EXPECT_EQ(hit, ExpectedIdenticalHit(short_pages * kBlock))
        << "a shorter identical prefix reuses up to its own recompute cap";
}

// Two requests sharing a prefix admitted in the SAME batch: the second still
// reuses the first's just-registered pages (in-flight claim is hit-safe).
TEST_F(FlatRealShapeHitSuite, TwoRequestsOneBatchShareInFlightPrefix) {
    const std::int32_t pages = 8;
    // r1 registers and finishes first.
    RunLifecycle(MakeRequestSpec("r1", pages));
    // r2 and r3 both identical to r1, submitted together.
    Submit(MakeRequestSpec("r2", pages));
    Submit(MakeRequestSpec("r3", pages));
    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_GE(op->extend_prefix_lens.size(), 1u);
    for (std::size_t i = 0; i < op->extend_prefix_lens.size(); ++i) {
        EXPECT_EQ(op->extend_prefix_lens.at(i), ExpectedIdenticalHit(pages * kBlock))
            << "batched request " << i << " must reuse the shared registered prefix";
    }
}

// Cold request (nothing registered) must report zero prefix hit -- the cache is
// only a reuse path, never fabricates a hit.
TEST_F(FlatRealShapeHitSuite, ColdRequestNoHit) {
    const std::int32_t hit = SubmitAndGetHit(MakeRequestSpec("r1", /*pages=*/8));
    EXPECT_EQ(hit, 0) << "a cold request with an empty cache must have no prefix hit";
}

}  // namespace flat_prefix_hit
}  // namespace tokenspeed::test

#endif  // TOKENSPEED_FLAT_KVCACHE
