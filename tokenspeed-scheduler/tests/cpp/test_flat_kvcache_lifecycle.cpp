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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// End-to-end lifecycle tests for the flat KV-cache FSM path
// (TOKENSPEED_FLAT_KVCACHE=ON; the whole file compiles to nothing otherwise).
// Config: two paged-cache groups (full + sliding-window), no L2/L3.

#if TOKENSPEED_FLAT_KVCACHE

#include <algorithm>
#include <optional>
#include <type_traits>

#include "cache/mamba_state_manager.h"
#include "integration_test_helper.h"

namespace tokenspeed::test {

class FlatKvCacheLifecycleTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.block_size = 2;
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        PagedCacheGroupConfig full_grp;
        full_grp.group_id = "full";
        full_grp.rows_per_page = cfg.block_size;
        full_grp.entry_stride_tokens = 1;
        full_grp.total_pages = cfg.device_allocator.total_pages;
        full_grp.retention = PagedCacheGroupConfig::Retention::FullHistory;
        full_grp.family = PagedCacheGroupFamily::History;

        PagedCacheGroupConfig swa_grp;
        swa_grp.group_id = "swa";
        swa_grp.rows_per_page = cfg.block_size;
        swa_grp.entry_stride_tokens = 1;
        swa_grp.total_pages = cfg.device_allocator.total_pages;
        swa_grp.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        swa_grp.sliding_window_tokens = 4;
        swa_grp.family = PagedCacheGroupFamily::State;

        cfg.paged_cache_groups = {full_grp, swa_grp};
        return cfg;
    }

    static const FlatForwardOperation* FindFlatOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (const auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }
};

class FlatStatePageTestSuite : public FlatKvCacheLifecycleTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatKvCacheLifecycleTestSuite::MakeConfig();
        PagedCacheGroupConfig state;
        state.group_id = "state";
        state.rows_per_page = cfg.block_size;
        state.entry_stride_tokens = 1;
        state.total_pages = cfg.device_allocator.total_pages;
        state.retention = PagedCacheGroupConfig::Retention::FullHistory;
        state.family = PagedCacheGroupFamily::State;
        cfg.paged_cache_groups.push_back(state);
        return cfg;
    }
};

class FlatStateChunkAlignmentTestSuite : public FlatStatePageTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatStatePageTestSuite::MakeConfig();
        cfg.max_scheduled_tokens = 8;
        return cfg;
    }
};

class FlatSpeculativeStatePageTestSuite : public FlatStatePageTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatStatePageTestSuite::MakeConfig();
        cfg.decode_input_tokens = 3;
        return cfg;
    }
};

class FlatSpeculativeStateAdmissionTestSuite : public FlatSpeculativeStatePageTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatSpeculativeStatePageTestSuite::MakeConfig();
        // 13 usable pages: the 12 canonical pages for prompt + decode reserve fit,
        // but the three request-owned speculative State pages do not.
        cfg.device_allocator.total_pages = 14;
        for (auto& group : cfg.paged_cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }
};

void AcquireRequestSpeculativeStateBlocks(Request& request, MambaStateManager& manager, BlockPool& pool,
                                          std::size_t state_group_index, std::int32_t target_num_blocks) {
    bool acquired = false;
    request.Apply([&](auto&& state) -> std::remove_cvref_t<decltype(state)> {
        using State = std::remove_cvref_t<decltype(state)>;
        if constexpr (std::derived_from<State, fsm::ForwardState>) {
            acquired = manager.AcquireSpeculativeBlocks(pool, state.BlockTables().at(state_group_index),
                                                        target_num_blocks);
        }
        return std::move(state);
    });
    ASSERT_TRUE(acquired);
}

void ScheduleFlatStateRequest(Request& request, KvCacheCoordinator& coordinator, ReqPoolAllocator& req_pool) {
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        /*tokens_this_round=*/4, /*decode_input_tokens=*/1, /*device_allocator=*/nullptr, &req_pool, MatchResult{},
        Role::kFused, /*kv_prefix_cache=*/nullptr, /*disable_l2_cache=*/true, /*loadback_diff=*/{},
        /*hybrid_prefix_cache=*/nullptr, /*mamba_allocator=*/nullptr, /*mamba_loadback_nodes=*/{}, &coordinator});
    EXPECT_TRUE(request.Is<fsm::PrefillDone>());
}

TEST(FlatSpeculativeStateLifecycleTest, FinishEventReleasesSpeculativeStateBlocks) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 2, 0}, {AttnKind::kMambaState, 2, 0}};
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool{4};
    MambaStateManager manager(/*block_size=*/2);
    const std::int32_t free_at_start = pool.NumFreeBlocks();
    RequestSpec spec{.request_id = "finish", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2)};
    Request request{spec, /*page_size=*/2, Role::kFused};
    ScheduleFlatStateRequest(request, coordinator, req_pool);
    AcquireRequestSpeculativeStateBlocks(request, manager, pool, /*state_group_index=*/1, /*target_num_blocks=*/2);
    ASSERT_EQ(pool.NumFreeBlocks(), free_at_start - 6);  // four canonical + two speculative

    request.Apply(fsm::FinishEvent{/*kv_prefix_cache=*/nullptr, /*host_allocator=*/nullptr, /*page_hashes=*/{},
                                   /*disable_l2_cache=*/true, /*hybrid_prefix_cache=*/nullptr, &coordinator});

    EXPECT_TRUE(request.Is<fsm::Finished>());
    EXPECT_EQ(pool.NumFreeBlocks(), free_at_start);
}

TEST(FlatSpeculativeStateLifecycleTest, AbortEventReleasesSpeculativeStateBlocks) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 2, 0}, {AttnKind::kMambaState, 2, 0}};
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool{4};
    MambaStateManager manager(/*block_size=*/2);
    const std::int32_t free_at_start = pool.NumFreeBlocks();
    RequestSpec spec{.request_id = "abort", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2)};
    Request request{spec, /*page_size=*/2, Role::kFused};
    ScheduleFlatStateRequest(request, coordinator, req_pool);
    AcquireRequestSpeculativeStateBlocks(request, manager, pool, /*state_group_index=*/1, /*target_num_blocks=*/2);

    request.Apply(fsm::AbortEvent{&coordinator});

    EXPECT_TRUE(request.Is<fsm::Finished>());
    EXPECT_EQ(pool.NumFreeBlocks(), free_at_start);
}

TEST(FlatSpeculativeStateLifecycleTest, FlatRetractEventReleasesSpeculativeStateBlocks) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 2, 0}, {AttnKind::kMambaState, 2, 0}};
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool{4};
    MambaStateManager manager(/*block_size=*/2);
    const std::int32_t free_at_start = pool.NumFreeBlocks();
    RequestSpec spec{.request_id = "retract", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2)};
    Request request{spec, /*page_size=*/2, Role::kFused};
    ScheduleFlatStateRequest(request, coordinator, req_pool);
    AcquireRequestSpeculativeStateBlocks(request, manager, pool, /*state_group_index=*/1, /*target_num_blocks=*/2);

    request.Apply(fsm::FlatRetractEvent{&coordinator});

    EXPECT_TRUE(request.Is<fsm::Submitted>());
    EXPECT_EQ(pool.NumFreeBlocks(), free_at_start);
}

TEST(FlatSpeculativeStateAdmissionTest, CombinedAcquireDoesNotMutateWhenOnlyCanonicalPagesFit) {
    BlockPool pool(/*total_blocks=*/5);  // four usable pages after the null placeholder
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 2, 0}, {AttnKind::kMambaState, 2, 0}};
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::int32_t free_at_start = pool.NumFreeBlocks();

    ASSERT_EQ(coordinator.BlocksNeededFor(tables, /*num_tokens=*/2, /*decode_width=*/0), 2);
    EXPECT_FALSE(coordinator.Acquire(tables, /*num_tokens=*/2, /*decode_width=*/3));

    EXPECT_EQ(pool.NumFreeBlocks(), free_at_start);
    for (const BlockTable& table : tables) {
        EXPECT_EQ(table.NumBlocks(), 0);
        EXPECT_TRUE(table.SpeculativeBlockIds().empty());
    }
}

TEST_F(FlatSpeculativeStateAdmissionTestSuite, DefersWhenCanonicalPagesFitButSpeculativeStatePagesDoNot) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));

    ExecutionPlan plan = PlanOnce();

    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_TRUE(op->empty());
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

TEST_F(FlatSpeculativeStatePageTestSuite, DecodeEmitsDistinctSpecPagesAndReusesThemAcrossRounds) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    const FlatForwardOperation* prefill = FindFlatOp(PlanOnce());
    ASSERT_NE(prefill, nullptr);
    EXPECT_TRUE(prefill->flat_state_spec_pages.empty());

    SendForwardDone("r1", {42, 43, 44});
    ExecutionPlan first_decode_plan = PlanOnce();
    const FlatForwardOperation* first_decode = FindFlatOp(first_decode_plan);
    ASSERT_NE(first_decode, nullptr);
    const auto& spec_pages = first_decode->flat_state_spec_pages.at("state").at(0);
    ASSERT_EQ(spec_pages.size(), 3u);
    EXPECT_TRUE(std::ranges::all_of(spec_pages, [](std::int32_t id) { return id > 0; }));
    std::vector<std::int32_t> sorted_spec_pages = spec_pages;
    std::ranges::sort(sorted_spec_pages);
    EXPECT_EQ(std::ranges::adjacent_find(sorted_spec_pages), sorted_spec_pages.end());

    const auto& state_row = first_decode->flat_block_tables.at("state").at(0);
    EXPECT_TRUE(first_decode->flat_state_in_pages.empty());
    EXPECT_TRUE(first_decode->flat_state_out_pages.empty());
    // Runtime derives canonical destinations from the GPU frontier. The C++
    // allocation contract is only that every reachable slot already exists.
    ASSERT_GE(state_row.size(), 4u);
    EXPECT_GT(state_row.at(2), 0);
    EXPECT_GT(state_row.at(3), 0);

    SendForwardDone("r1", {45, 46, 47});
    ExecutionPlan second_decode_plan = PlanOnce();
    const FlatForwardOperation* second_decode = FindFlatOp(second_decode_plan);
    ASSERT_NE(second_decode, nullptr);
    EXPECT_EQ(second_decode->flat_state_spec_pages.at("state").at(0), spec_pages);
    EXPECT_TRUE(second_decode->flat_state_in_pages.empty());
    EXPECT_TRUE(second_decode->flat_state_out_pages.empty());
}

TEST_F(FlatStatePageTestSuite, PrefillResolvesPagesAndDecodePublishesAllocationTable) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ExecutionPlan prefill_plan = PlanOnce();
    const FlatForwardOperation* prefill = FindFlatOp(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    const auto& prefill_row = prefill->flat_block_tables.at("state").at(0);
    EXPECT_EQ(prefill->flat_state_in_pages.at("state").at(0), 0);
    EXPECT_EQ(prefill->flat_state_out_pages.at("state").at(0).at(0), prefill_row.at(1));

    SendForwardDone("r1", {42});
    ExecutionPlan decode_plan = PlanOnce();
    const FlatForwardOperation* decode = FindFlatOp(decode_plan);
    ASSERT_NE(decode, nullptr);
    const auto& decode_row = decode->flat_block_tables.at("state").at(0);
    EXPECT_TRUE(decode->flat_state_in_pages.empty());
    EXPECT_TRUE(decode->flat_state_out_pages.empty());
    ASSERT_GE(decode_row.size(), 3u);
    EXPECT_GT(decode_row.at(1), 0);
    EXPECT_GT(decode_row.at(2), 0);
    EXPECT_TRUE(decode->flat_state_spec_pages.empty());
}

TEST_F(FlatStatePageTestSuite, BoundaryPlusOnePromptKeepsStateInputPageForFirstDecode) {
    token_vec_t prompt = MakeAlignedTokens(/*num_pages=*/2, PageSize());
    prompt.push_back(99);  // 5 tokens with P=2: k*P+1 is the reclaim edge case.
    Submit(RequestSpec{.request_id = "r1", .tokens = prompt});
    ASSERT_NE(FindFlatOp(PlanOnce()), nullptr);

    // The real overlapped serving path can acknowledge prefill without adding
    // a sampled token before scheduling the first decode. The conservative
    // reclaim boundary must keep the page supplying that decode's input state.
    SendForwardDone("r1", {});
    ExecutionPlan decode_plan = PlanOnce();
    const FlatForwardOperation* decode = FindFlatOp(decode_plan);
    ASSERT_NE(decode, nullptr);
    const auto& state_row = decode->flat_block_tables.at("state").at(0);
    ASSERT_GE(state_row.size(), 3u);
    EXPECT_TRUE(decode->flat_state_in_pages.empty());
    EXPECT_TRUE(decode->flat_state_out_pages.empty());
    EXPECT_GT(state_row.at(1), 0);
    EXPECT_GT(state_row.at(2), 0);
}

TEST_F(FlatStateChunkAlignmentTestSuite, PartialChunkEndsOnStateCheckpointGrid) {
    token_vec_t prompt = MakeAlignedTokens(/*num_pages=*/2, PageSize());
    prompt.push_back(99);  // 5 tokens per request with an 8-token round budget.
    Submit(RequestSpec{.request_id = "r1", .tokens = prompt});
    Submit(RequestSpec{.request_id = "r2", .tokens = prompt});

    ExecutionPlan first_round = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(first_round);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 2u);
    const auto r2 = std::find(op->request_ids.begin(), op->request_ids.end(), "r2");
    ASSERT_NE(r2, op->request_ids.end());
    const std::size_t index = static_cast<std::size_t>(r2 - op->request_ids.begin());

    // r1 consumes 5 tokens, leaving a nominal budget of 3. Sending all 3
    // would leave r2 at absolute position 3, so the next chunk would cross
    // the P=2 state boundary off the kernel's chunk-relative checkpoint grid.
    EXPECT_EQ(op->input_lengths.at(index), 2);
    EXPECT_EQ((op->extend_prefix_lens.at(index) + op->input_lengths.at(index)) % PageSize(), 0);
    EXPECT_LT(op->extend_prefix_lens.at(index) + op->input_lengths.at(index), op->prefill_lengths.at(index));
}

TEST_F(FlatKvCacheLifecycleTestSuite, Construct_AndSubmit_Waiting) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

TEST_F(FlatKvCacheLifecycleTestSuite, SingleRequest_PrefillDecodeFinish) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ExecutionPlan prefill_plan = PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    const FlatForwardOperation* prefill = FindFlatOp(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->flat_block_tables.count("full"), 1u);
    ASSERT_EQ(prefill->flat_block_tables.count("swa"), 1u);
    EXPECT_EQ(prefill->flat_block_tables.at("full").size(), 1u);
    EXPECT_EQ(prefill->flat_block_tables.at("swa").size(), 1u);

    SendForwardDone("r1", {42});
    EXPECT_EQ(scheduler_->PrefillSize(), 1u);

    // Swa null hole first appears at decode step 1 (window=4 tokens = 2 pages).
    // last_plan must outlive the loop: the FlatForwardOperation is owned by its plan.
    std::optional<ExecutionPlan> last_plan;
    int tok = 43;
    for (int step = 0; step < 4; ++step) {
        last_plan = PlanOnce();
        ASSERT_NE(FindFlatOp(*last_plan), nullptr);
        EXPECT_EQ(scheduler_->DecodingSize(), 1u);
        SendForwardDone("r1", {tok++});
    }
    const FlatForwardOperation* last_decode = FindFlatOp(*last_plan);
    ASSERT_NE(last_decode, nullptr);

    const auto& full_row = last_decode->flat_block_tables.at("full").at(0);
    for (std::int32_t id : full_row) {
        EXPECT_GT(id, 0) << "full row should keep history with no null/padding hole";
    }
    const auto& swa_row = last_decode->flat_block_tables.at("swa").at(0);
    EXPECT_NE(std::find(swa_row.begin(), swa_row.end(), 0), swa_row.end())
        << "swa row should contain a null hole after the sliding window slides";

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// AvailableKvPages() must report the flat shared BlockPool, not the radix
// device_allocator_. TODO(radix-removal): collapses to the only accessor.
TEST_F(FlatKvCacheLifecycleTestSuite, AvailableKvPagesReportsFlatPool) {
    const std::size_t idle = scheduler_->AvailableKvPages();
    EXPECT_EQ(idle, static_cast<std::size_t>(scheduler_->FlatPoolFreeBlocks()));
    // 32 total pages, block 0 is the never-allocated null placeholder.
    EXPECT_EQ(idle, 31u);

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), static_cast<std::size_t>(scheduler_->FlatPoolFreeBlocks()));
    EXPECT_LT(scheduler_->AvailableKvPages(), idle)
        << "prefill draws from the flat pool and the bound accessor must see it";

    SendForwardDone("r1", {42});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), idle);
}

TEST_F(FlatKvCacheLifecycleTestSuite, TwoRequests_BatchedFlatBlockTables) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan prefill_plan = PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    const FlatForwardOperation* prefill = FindFlatOp(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->request_ids.size(), 2u);

    ASSERT_EQ(prefill->flat_block_tables.count("full"), 1u);
    ASSERT_EQ(prefill->flat_block_tables.count("swa"), 1u);
    const auto& full = prefill->flat_block_tables.at("full");
    const auto& swa = prefill->flat_block_tables.at("swa");
    ASSERT_EQ(full.size(), 2u);
    ASSERT_EQ(swa.size(), 2u);

    EXPECT_EQ(full.at(0).size(), full.at(1).size());
    EXPECT_EQ(swa.at(0).size(), swa.at(1).size());
    const bool any_pad = std::any_of(full.at(0).begin(), full.at(0).end(), [](std::int32_t id) { return id == -1; }) ||
                         std::any_of(full.at(1).begin(), full.at(1).end(), [](std::int32_t id) { return id == -1; });
    EXPECT_TRUE(any_pad) << "unequal prompt lengths should force -1 padding in one full row";

    auto assert_no_page_collision = [](const std::vector<std::vector<std::int32_t>>& group) {
        std::vector<std::int32_t> real;
        for (const auto& row : group) {
            for (std::int32_t id : row) {
                if (id > 0) real.push_back(id);
            }
        }
        std::vector<std::int32_t> sorted = real;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
            << "two requests must not be handed the same physical page";
    };
    assert_no_page_collision(full);
    assert_no_page_collision(swa);

    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendFinish("r1");
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

}  // namespace tokenspeed::test

#endif  // TOKENSPEED_FLAT_KVCACHE
