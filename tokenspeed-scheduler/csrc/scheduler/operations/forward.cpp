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

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#include "cache/forward_cache_ops.h"
#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/radix_tree/node_range.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/page_hasher.h"
#include "scheduler/request.h"
#include "scheduler/request_spec.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"
#include "utils.h"

namespace tokenspeed {

namespace {

constexpr std::int32_t kLocalMambaSlotsPerRequest = 2;

std::int32_t DecodePagedCacheReservationEnd(std::int32_t first_pos, std::int32_t verify_width,
                                            std::int32_t overlap_depth) {
    if (first_pos < 0 || verify_width < 0 || overlap_depth < 0 || overlap_depth > 1) {
        throw std::invalid_argument("invalid paged-cache decode reservation arguments");
    }
    const std::int64_t reservation_end =
        static_cast<std::int64_t>(first_pos) + static_cast<std::int64_t>(overlap_depth + 1) * verify_width;
    if (reservation_end > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("paged-cache decode reservation exceeds int32 range");
    }
    return static_cast<std::int32_t>(reservation_end);
}

std::int32_t CountMambaDeviceLoadBackSlots(const std::vector<TreeNode*>& nodes) {
    std::int32_t slots = 0;
    for (TreeNode* node : nodes) {
        if (node != nullptr && node->HasMambaOnHost() && !node->HasMamba()) {
            ++slots;
        }
    }
    return slots;
}

void AddUniqueNode(std::vector<TreeNode*>& nodes, TreeNode* node) {
    if (node == nullptr) return;
    if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
        nodes.push_back(node);
    }
}

// Copy the request's live local-mamba working (and checkpoint, if any) slot
// indices into the op. No-op when the request holds no local mamba allocation.
void FillMambaWorkingIndices(ForwardOperationBase& op, Request* request) {
    auto* mamba = request->GetLocalMambaAllocator();
    if (mamba != nullptr && mamba->HasWorking()) {
        op.mamba_working_idx = mamba->WorkingIndex();
        if (mamba->HasCheckpoint()) {
            op.mamba_checkpoint_dst_idx = mamba->CheckpointIndex();
        }
    }
}

template <typename Op>
void FillFlatCacheInputs(Op& op, Request* request, std::span<const std::string> flat_group_ids,
                         const std::map<std::string, std::int32_t>& state_group_block_sizes,
                         std::int32_t tokens_before, std::int32_t tokens_after, bool per_step_out_pages,
                         bool resolve_state_execution_pages) {
    if (!request->FlatBlockTablesEmpty()) {
        op.flat_block_tables = BuildFlatBlockTables(request->FlatBlockTablesRef(), flat_group_ids);
    }

    for (const auto& [group_id, block_size] : state_group_block_sizes) {
        if (resolve_state_execution_pages) {
            // Phase 1 keeps prefill/extend on scheduler-resolved pages. Decode
            // and target verify instead resolve from GPU sequence lengths.
            const auto& row = op.flat_block_tables.at(group_id);
            const std::int32_t in_slot = std::max(tokens_before - 1, 0) / block_size;
            _assert(tokens_after > 0, "state token range must be non-empty");
            _assert(in_slot < static_cast<std::int32_t>(row.size()), "state page slot out of range");
            const std::int32_t in_page = tokens_before > 0 ? row[static_cast<std::size_t>(in_slot)] : 0;
            _assert(tokens_before == 0 || in_page > 0, "state page is missing");
            op.flat_state_in_pages.emplace(group_id, in_page);

            const std::int32_t first_step = per_step_out_pages ? tokens_before + 1 : tokens_after;
            std::vector<std::int32_t> out_pages;
            out_pages.reserve(static_cast<std::size_t>(std::max(1, tokens_after - first_step + 1)));
            for (std::int32_t pos = first_step; pos <= tokens_after; ++pos) {
                const std::int32_t out_slot = (pos - 1) / block_size;
                _assert(out_slot < static_cast<std::int32_t>(row.size()), "state page slot out of range");
                const std::int32_t out_page = row[static_cast<std::size_t>(out_slot)];
                _assert(out_page > 0, "state page is missing");
                out_pages.push_back(out_page);
            }
            op.flat_state_out_pages.emplace(group_id, std::move(out_pages));
        }

        const auto group_it = std::ranges::find(flat_group_ids, group_id);
        _assert(group_it != flat_group_ids.end(), "state group is missing from flat group ids");
        const std::size_t group_index = static_cast<std::size_t>(group_it - flat_group_ids.begin());
        std::vector<std::int32_t> spec_pages = request->FlatBlockTablesRef().at(group_index).SpeculativeBlockIds();
        if (!spec_pages.empty()) {
            op.flat_state_spec_pages.emplace(group_id, std::move(spec_pages));
        }
    }
}

}  // namespace

#if TOKENSPEED_FLAT_KVCACHE
namespace {

// Slide credit is registration-aware: with candidate collection on, blocks the op will register
// get pinned and do not free (count_uncached=false path).
[[nodiscard]] std::int32_t FlatSlideCredit(const KvCacheCoordinator& coordinator, std::span<const BlockTable> tables,
                                           std::int32_t num_computed_tokens,
                                           std::int32_t pending_state_input_tokens) {
    std::int32_t total_freed = 0;
    for (std::int32_t i = 0; i < coordinator.NumGroups(); ++i) {
        const KvCacheManager& manager = coordinator.GroupManager(i);
        const std::int32_t boundary =
            manager.SafeReclaimBoundary(num_computed_tokens, pending_state_input_tokens);
        total_freed += manager.BlocksReclaimableAt(tables[static_cast<std::size_t>(i)], boundary,
                                                   /*count_uncached=*/!coordinator.HasHostTier());
    }
    return total_freed;
}

// Intermediate state snapshots are emitted on a chunk-relative page grid. Keep
// every non-final prefill end on the same absolute grid so the next chunk does
// not cross a cacheable state boundary before its first kernel checkpoint.
[[nodiscard]] std::int32_t AlignFlatPartialPrefillChunk(const KvCacheCoordinator& coordinator,
                                                        std::int32_t start_tokens,
                                                        std::int32_t requested_tokens,
                                                        std::int32_t unscheduled_tokens) {
    _assert(requested_tokens > 0, "prefill chunk must make progress");
    _assert(unscheduled_tokens > 0, "remaining prefill tokens must be positive");
    _assert(requested_tokens <= unscheduled_tokens, "prefill chunk exceeds remaining prefill tokens");
    if (requested_tokens == unscheduled_tokens) {
        return requested_tokens;
    }
    if (!coordinator.RequiresAlignedPrefillChunks()) {
        return requested_tokens;
    }
    const std::int32_t alignment = coordinator.LcmBlockSize();
    const std::int32_t aligned_end = ((start_tokens + requested_tokens) / alignment) * alignment;
    return std::max(aligned_end - start_tokens, 0);
}

// Decoding/PrefillDone requests hold pool pages a flat retract can release.
[[nodiscard]] bool IsFlatHolder(const Request* req) {
    return req->Is<fsm::Decoding>() || req->Is<fsm::PrefillDone>();
}

// Deferred = schedulable states the forward loop skipped this round for lack of pool pages.
// Superset of IsFlatHolder plus the not-yet-running schedulable states.
[[nodiscard]] bool IsFlatDeferred(const Request* req) {
    return IsFlatHolder(req) || req->Is<fsm::Submitted>() || req->Is<fsm::PrefetchDone>() || req->Is<fsm::Prefilling>();
}

}  // namespace

// One match, one hash pass at admission: the device match plus its host-tier extension share the
// token math, the gate charge and window.begin. Claiming in-flight pages is stream-ordering safe
// (forward_cache_ops.h).
Scheduler::FlatAdmissionMatch Scheduler::matchFlatPrefixAtAdmission(Request* request) {
    if (config_.disable_prefix_cache) {
        return {};
    }
    // Hash input must be byte-identical to the REGISTRATION form (GetFullPagedTokens(false)); radix's
    // except_last rule (last prompt token recomputed for logits) becomes the page cap, also bounding SWA.
    // These offsets index the base-granular hash array (GetFullPagedTokens is cut at base).
    const std::int32_t base_block_size = coordinator_.BaseBlockSize();
    const std::int32_t cap_pages = std::max((request->PrefillSize() - 1) / base_block_size, 0);
    std::vector<std::span<const std::int32_t>> paged_tokens = request->GetFullPagedTokens(/*except_last=*/false);
    if (static_cast<std::size_t>(cap_pages) < paged_tokens.size()) {
        paged_tokens.resize(cap_pages);
    }
    const std::vector<std::string> flat_hashes = ComputePagedHashes(paged_tokens, "");
    FlatAdmissionMatch match;
    auto [device, host] = coordinator_.MatchPrefix(flat_hashes);
    match.device = std::move(device);
    match.host = std::move(host);
    // Boundaries are in tokens; the extension hash offsets are in base pages (the granularity the
    // hashes were computed at). No host pool -> host boundary 0 -> empty slice.
    const std::int32_t ext_pages =
        std::max(match.host.num_common_tokens - match.device.num_common_tokens, 0) / base_block_size;
    const auto ext_begin = flat_hashes.begin() + match.device.num_common_tokens / base_block_size;
    match.ext_hashes.assign(ext_begin, ext_begin + ext_pages);
    return match;
}

// Returns the decode-reserve pages to record when admitted (0 unless this chunk completes prefill); nullopt = defer.
std::optional<std::int32_t> Scheduler::flatAdmitFirstChunk(Request* request, const CoordinatorMatch& hit,
                                                           std::int32_t ext_real_pages, std::int32_t chunk_tokens,
                                                           std::int32_t decode_reserve_tokens) const {
    // Charge chunk + reserve in one query: unreserved, an exactly-filling prompt's own decode defers forever.
    // ext_real_pages composes exactly: extension pages are FULL, so they leave tail_avail 0.
    // When present, the reserve IS the first decode step -> its token count doubles as the state groups' width.
    const std::int32_t reserve_decode_width = decode_reserve_tokens;
    const std::int32_t blocks_needed =
        coordinator_.BlocksNeededFor(chunk_tokens + decode_reserve_tokens, reserve_decode_width);
    // Exact since gate and apply run back to back.
    const std::int32_t claim_blocks = coordinator_.BlocksConsumedByClaim(hit);
    if (blocks_needed + claim_blocks + ext_real_pages > flatFreeBudget(request->Id())) {
        return std::nullopt;
    }
    // Reserve need is computed on the post-prefill table shape now, never recomputed against drifted state.
    return decode_reserve_tokens > 0
               ? blocks_needed - coordinator_.BlocksNeededFor(chunk_tokens, /*decode_width=*/0)
               : 0;
}

// Same contract as flatAdmitFirstChunk, on live tables; num_computed_tokens matches the transition's slide.
std::optional<std::int32_t> Scheduler::flatAdmitPrefillChunk(Request* request, std::int32_t chunk_tokens,
                                                             std::int32_t decode_reserve_tokens,
                                                             std::int32_t num_computed_tokens) const {
    const std::int32_t slide_credit = FlatSlideCredit(coordinator_, request->FlatBlockTablesRef(),
                                                      num_computed_tokens, /*pending_state_input_tokens=*/0);
    // When present, the reserve IS the first decode step -> its token count doubles as the state groups' width.
    const std::int32_t reserve_decode_width = decode_reserve_tokens;
    const std::int32_t blocks_needed = coordinator_.BlocksNeededFor(
        request->FlatBlockTablesRef(), chunk_tokens + decode_reserve_tokens, reserve_decode_width);
    if (blocks_needed > flatFreeBudget(request->Id()) + slide_credit) {
        return std::nullopt;
    }
    // The pending slide cannot drift the reserve: ReclaimExpired punches front holes, BlocksNeededFor reads tail_avail.
    return decode_reserve_tokens > 0
               ? blocks_needed - coordinator_.BlocksNeededFor(request->FlatBlockTablesRef(), chunk_tokens,
                                                              /*decode_width=*/0)
               : 0;
}

// Gate for the PrefillDone reserve Acquire and each DecodeStep, composed from the transition's own primitives.
bool Scheduler::flatAdmitDecode(Request* request) const {
    // At first decode, ordinary KV has computed the full prefill while state still reads the pending tail.
    // Later decode steps use the same pending-tail boundary for every group.
    const bool first_decode = request->Is<fsm::PrefillDone>();
    const std::int32_t num_computed_tokens =
        first_decode ? request->PrefillSize() : request->TokenSize() - config_.decode_input_tokens;
    const std::int32_t pending_state_input_tokens = first_decode ? config_.decode_input_tokens : 0;
    const std::int32_t slide_credit = FlatSlideCredit(coordinator_, request->FlatBlockTablesRef(),
                                                      num_computed_tokens, pending_state_input_tokens);
    const std::int32_t blocks_needed = coordinator_.BlocksNeededFor(
        request->FlatBlockTablesRef(), request->GetReserveNumTokensInNextScheduleEvent(), config_.decode_input_tokens);
    return blocks_needed <= flatFreeBudget(request->Id()) + slide_credit;
}

// True when this round can never unwedge itself; fused-only (PD requests hold pages outside both ledgers).
bool Scheduler::flatPoolWedged(const std::vector<Request*>& candidates) const {
    const bool any_deferred = std::any_of(candidates.begin(), candidates.end(), IsFlatDeferred);
    // Block 0 is the null placeholder, never allocated.
    const bool pool_pages_held = block_pool_.NumFreeBlocks() < block_pool_.TotalBlocks() - 1;
    // An in-flight D2H store OR H2D load still holds pool pages its Done event will free; both
    // ledgers must be empty. Dispatched mid-prefill chunk ops are invisible here and safe (they
    // free no pool pages, emit no event, and page reuse under them is stream-ordering safe).
    const bool nothing_in_flight = pending_forward_results_.empty() && cache_op_tracker_.empty() &&
                                   flat_store_ops_.Empty() && flat_load_ops_.empty();
    return config_.role == Role::kFused && any_deferred && pool_pages_held && nothing_in_flight;
}

// Wedge resolution, escalating on the SECOND consecutive wedged round (an in-flight Finish fakes one):
// flat-retract the largest Decoding/PrefillDone holder, or -- with no holder -- OOM-terminalize the
// head-of-line deferred non-holder. Sole owner of flat_starved_rounds_.
void Scheduler::resolveFlatStarvation(const std::vector<Request*>& candidates, bool made_progress) {
    if (made_progress || !flatPoolWedged(candidates)) {
        flat_starved_rounds_ = 0;
        return;
    }
    if (++flat_starved_rounds_ < 2) {
        return;
    }
    flat_starved_rounds_ = 0;
    std::vector<Request*> holders;
    for (Request* req : candidates) {
        if (IsFlatHolder(req)) {
            holders.push_back(req);
        }
    }
    if (!holders.empty()) {
        Request* victim = *std::ranges::max_element(holders, {}, &Request::TokenSize);
        // Wedge-gate guarantee (flatPoolWedged: pending_forward_results_.empty()): the runtime owes
        // the victim nothing, so its re-prefill can never race a stale ExtendResult. If the gate is
        // ever relaxed, fail loud here instead of corrupting the rebased prefill.
        _assert(pending_forward_results_.find(victim->Id()) == pending_forward_results_.end(),
                "retract victim must not owe a forward result");
        flat_reserved_pages_.erase(victim->Id());
        victim->Apply(fsm::FlatRetractEvent{&coordinator_});
        spdlog::info("[Scheduler] flat retract: released request {} ({} tokens) to unwedge the pool", victim->Id(),
                     victim->TokenSize());
        return;
    }
    // No retract victim: the pages are held mid-prefill (a wedged holder, or a mutual wedge among
    // prefilling requests). Terminalize the head-of-line deferred non-holder to unwedge the pool.
    for (Request* req : candidates) {
        if (IsFlatDeferred(req) && !IsFlatHolder(req)) {
            req->Apply(fsm::AbortEvent{&coordinator_});
            flat_oom_request_ids_.push_back(req->Id());
            spdlog::warn(
                "[Scheduler] flat OOM: pool wedged by unretractable mid-prefill holders; terminalized request {}",
                req->Id());
            return;
        }
    }
    _assert(false, "wedged with no holder and no deferred non-holder");
}
#endif

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, bool disable_l2_cache,
    std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};
    MatchResult match_result = hybrid_prefix_cache_ ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true))
                                                    : kv_prefix_cache_.Match(request->GetFullPagedTokens(true));
    std::int32_t loadback_tokens = 0;
    std::int32_t unscheduled = 0;
    std::vector<TreeNode*> loadback_diff;
    std::vector<TreeNode*> mamba_loadback_nodes;

    const std::int32_t device_matched = match_result.device.DepthInPage();
    const std::int32_t host_matched = match_result.host.DepthInPage();
    if (disable_l2_cache) {
        unscheduled = request->PrefillSize() - device_matched * config_.block_size;
    } else {
        loadback_diff = match_result.NodesWithout<ResourceType::Device>();
        if (host_matched > device_matched) {
            loadback_tokens = config_.block_size * (host_matched - device_matched);
        }
        unscheduled = request->PrefillSize() - std::max(device_matched, host_matched) * config_.block_size;
    }

    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && match_result.mamba_branching_seqlen == -1) {
        const std::int32_t aligned = hybrid_prefix_cache_->AlignMambaCacheSeqlen(tokens_this_round);
        if (aligned > 0) {
            match_result.mamba_branching_seqlen = aligned;
        }
    }

    std::int32_t num_tokens = loadback_tokens + tokens_this_round + decode_input_tokens;
    std::int32_t device_pages_needed = (num_tokens + config_.block_size - 1) / config_.block_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);

    // Evict unlocked prefix-cache nodes before allocating request-local pages.
    if (!(kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed))) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    FlatAdmissionMatch flat_match = matchFlatPrefixAtAdmission(request);
    // Overwrite the radix-sourced locals: the radix tree is never written on flat builds.
    const std::int32_t hit_num_tokens =
        std::max(flat_match.device.num_common_tokens, flat_match.host.num_common_tokens);
    unscheduled = request->PrefillSize() - hit_num_tokens;
    tokens_this_round = std::min(remaining, unscheduled);
    tokens_this_round =
        AlignFlatPartialPrefillChunk(coordinator_, hit_num_tokens, tokens_this_round, unscheduled);
    if (tokens_this_round <= 0) {
        return {};
    }

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t flat_decode_reserve = completes_prefill ? decode_input_tokens : 0;
    // One pin per real (non-hole) extension slot across all groups = the new device pages the load needs.
    std::int32_t flat_ext_real_pages = 0;
    for (const PrefixMatch& g : flat_match.host.per_group) {
        flat_ext_real_pages += g.num_hit_blocks;
    }
    const std::optional<std::int32_t> flat_reserve_pages =
        flatAdmitFirstChunk(request, flat_match.device, flat_ext_real_pages, tokens_this_round, flat_decode_reserve);
    if (!flat_reserve_pages) {
        return {};
    }
#endif

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && match_result.mamba_host_src_index >= 0 &&
        match_result.mamba_cow_src_index < 0) {
        TreeNode* host_mamba_node = hybrid_prefix_cache_->FindLastMambaHostNode(match_result.host.last_node);
        if (host_mamba_node != nullptr && host_mamba_node->HasMambaOnHost() && !host_mamba_node->HasMamba()) {
            AddUniqueNode(mamba_loadback_nodes, host_mamba_node);
        }
    }
    const bool needs_mamba_loadback = !mamba_loadback_nodes.empty();
    const std::int32_t mamba_loadback_slots_needed =
        needs_mamba_loadback ? CountMambaDeviceLoadBackSlots(mamba_loadback_nodes) : 0;
    const std::int32_t mamba_slots_needed = kLocalMambaSlotsPerRequest + mamba_loadback_slots_needed;
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(mamba_slots_needed)) {
        return {};
    }

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    if (hybrid_prefix_cache_ &&
        !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, match_result.paged_cache)) {
        return {};
    }
    if (needs_mamba_loadback) {
        hybrid_prefix_cache_->PrepareMambaDeviceLoadBack(mamba_loadback_nodes);
        TreeNode* mamba_node = hybrid_prefix_cache_->FindLastMambaNode(match_result.host.last_node);
        if (mamba_node != nullptr) {
            match_result.mamba_cow_src_index = mamba_node->MambaSlotIndex();
        }
    }
    if (mamba_allocator_ && mamba_allocator_->AvailableSlots() < kLocalMambaSlotsPerRequest) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    // Role kD reaches PrefillDone via RemotePrefillDoneEvent with reserve 0: recording would leave a phantom entry.
    if (*flat_reserve_pages > 0 && config_.role != Role::kD) {
        flat_reserved_pages_[request->Id()] = *flat_reserve_pages;
    }
#endif
    return fsm::SchedulePrefillFirstChunkEvent{
        tokens_this_round,
        decode_input_tokens,
        &device_allocator_,
        &req_pool_allocator_,
        match_result,
        config_.role,
        &kv_prefix_cache_,
        disable_l2_cache,
        std::move(loadback_diff),
        hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr,
        mamba_allocator_ ? &*mamba_allocator_ : nullptr,
        std::move(mamba_loadback_nodes),
#if TOKENSPEED_FLAT_KVCACHE
        &coordinator_,
        std::move(flat_match.device),
        std::move(flat_match.host),
        std::move(flat_match.ext_hashes),
#endif
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event,
    std::map<std::string, std::int32_t>& simulated_free) {
    std::int32_t unscheduled = request->UnScheduledPrefillSize();
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);

    std::int32_t pages_needed = (tokens_this_round + config_.block_size - 1) / config_.block_size;

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    const std::int32_t num_computed_tokens = request->PrefillSize() - unscheduled;
    tokens_this_round =
        AlignFlatPartialPrefillChunk(coordinator_, num_computed_tokens, tokens_this_round, unscheduled);
    if (tokens_this_round <= 0) {
        return {};
    }
    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t flat_decode_reserve = completes_prefill ? reserve_num_tokens_in_next_schedule_event : 0;
    const std::optional<std::int32_t> flat_reserve_pages =
        flatAdmitPrefillChunk(request, tokens_this_round, flat_decode_reserve, num_computed_tokens);
    if (!flat_reserve_pages) {
        return {};
    }
#endif

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(1)) {
        return {};
    }

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    if (hybrid_prefix_cache_) {
        const std::int32_t commit_target = (first_pos / config_.block_size) * config_.block_size;
        const auto commit_token_pages = request->GetFullPagedTokens(false);
        if (!hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, {}, commit_target,
                                              commit_token_pages)) {
            return {};
        }
    }

#if TOKENSPEED_FLAT_KVCACHE
    // No kD gate needed: the planning loop never calls schedulePrefill for role kD.
    if (*flat_reserve_pages > 0) {
        flat_reserved_pages_[request->Id()] = *flat_reserve_pages;
    }
#endif

    return fsm::SchedulePrefillEvent{tokens_this_round, reserve_num_tokens_in_next_schedule_event,
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr
#if TOKENSPEED_FLAT_KVCACHE
                                     ,
                                     &coordinator_
#endif
    };
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(Request* request,
                                                                  std::map<std::string, std::int32_t>& simulated_free) {
    std::int32_t tail_available = request->TailPageAvailableTokens();
    std::int32_t extra_tokens = std::max(0, request->GetReserveNumTokensInNextScheduleEvent() - tail_available);
    std::int32_t pages_needed = (extra_tokens + config_.block_size - 1) / config_.block_size;

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    if (!flatAdmitDecode(request)) {
        return {};
    }
#endif

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && mamba_allocator_ &&
        request->Is<fsm::PrefillDone>() && request->GetLocalMambaAllocator() != nullptr &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(1)) {
        return {};
    }

    const std::int32_t first_pos = request->TokenSize();
    const std::int32_t target =
        DecodePagedCacheReservationEnd(first_pos, config_.decode_input_tokens, config_.overlap_schedule_depth);
    if (hybrid_prefix_cache_) {
        std::optional<std::int32_t> commit_target;
        std::vector<std::span<const std::int32_t>> commit_token_pages;
        if (request->Is<fsm::PrefillDone>()) {
            commit_target = (request->PrefillSize() / config_.block_size) * config_.block_size;
            commit_token_pages = request->GetFullPagedTokens(false);
        }
        if (!hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, {}, commit_target,
                                              commit_token_pages)) {
            return {};
        }
    }

    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr
#if TOKENSPEED_FLAT_KVCACHE
                                    ,
                                    &coordinator_
#endif
    };
}

std::optional<fsm::ScheduleDecodeFromRetractedEvent> Scheduler::scheduleDecodeFromRetracted(
    Request* request, std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};

    MatchResult match_result =
        hybrid_prefix_cache_
            ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery)
            : kv_prefix_cache_.Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery);
    std::vector<TreeNode*> loadback_diff = match_result.NodesWithout<ResourceType::Device>();
    std::vector<TreeNode*> mamba_loadback_nodes;
    TreeNode* mamba_recovery_node = nullptr;
    bool needs_mamba_loadback = false;
    if (hybrid_prefix_cache_ && mamba_allocator_) {
        mamba_recovery_node = hybrid_prefix_cache_->FindLastMambaNode(match_result.host.last_node);
        if (mamba_recovery_node == nullptr) {
            mamba_recovery_node = hybrid_prefix_cache_->FindLastMambaHostNode(match_result.host.last_node);
            needs_mamba_loadback = mamba_recovery_node != nullptr;
            if (needs_mamba_loadback && !mamba_recovery_node->HasMamba()) {
                AddUniqueNode(mamba_loadback_nodes, mamba_recovery_node);
            }
        }
        if (mamba_recovery_node == nullptr) {
            spdlog::warn("[Scheduler] Retracted request {} lost tree-owned Mamba state, aborting request",
                         request->Id());
            request->Apply(fsm::AbortEvent{
#if TOKENSPEED_FLAT_KVCACHE
                &coordinator_
#endif
            });
            return {};
        }
        if (!needs_mamba_loadback) {
            match_result.mamba_cow_src_index = mamba_recovery_node->MambaSlotIndex();
        }
    }

    const std::int32_t device_matched = match_result.device.DepthInPage();
    const std::int32_t host_matched = match_result.host.DepthInPage();
    std::int32_t num_tokens = 0;
    if (host_matched > device_matched) {
        num_tokens += (config_.block_size * (host_matched - device_matched)) + config_.decode_input_tokens;
    } else {
        num_tokens += config_.decode_input_tokens;
    }
    std::int32_t device_pages_needed = (num_tokens + config_.block_size - 1) / config_.block_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed)) {
        return {};
    }
    if (hybrid_prefix_cache_ && mamba_allocator_) {
        // Protect the COW source node only for this allocation; retracted Mamba states stay normal evictable entries.
        const std::int32_t mamba_slots_needed =
            kLocalMambaSlotsPerRequest + CountMambaDeviceLoadBackSlots(mamba_loadback_nodes);
        if (!hybrid_prefix_cache_->EnsureMambaCapacityByEvict(mamba_slots_needed, mamba_recovery_node)) {
            return {};
        }
    }

    const std::int32_t first_pos = request->TokenSize() - 1;
    const std::int32_t target = std::max(
        request->TokenSize(),
        DecodePagedCacheReservationEnd(first_pos, config_.decode_input_tokens, config_.overlap_schedule_depth));
    if (hybrid_prefix_cache_ &&
        !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, match_result.paged_cache)) {
        return {};
    }
    if (needs_mamba_loadback) {
        hybrid_prefix_cache_->PrepareMambaDeviceLoadBack(mamba_loadback_nodes);
        if (mamba_recovery_node->HasMamba()) {
            match_result.mamba_cow_src_index = mamba_recovery_node->MambaSlotIndex();
        }
    }
    if (mamba_allocator_ && mamba_allocator_->AvailableSlots() < kLocalMambaSlotsPerRequest) {
        return {};
    }

    return fsm::ScheduleDecodeFromRetractedEvent{
        config_.decode_input_tokens,
        &device_allocator_,
        &req_pool_allocator_,
        &kv_prefix_cache_,
        std::move(match_result),
        loadback_diff,
        mamba_allocator_ ? &*mamba_allocator_ : nullptr,
        std::move(mamba_loadback_nodes),
    };
}

std::optional<fsm::ScheduleRetractEvent> Scheduler::scheduleRetract(Request* request) {
    auto full_paged_tokens = request->GetFullPagedTokens(true);
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(request->GetDeviceNode());
    std::int32_t total_available = static_cast<std::int32_t>(request->GetOccupiedPages().size());

    // Overlap scheduling: ExtendResult may grow the token container early; clamp to the pages we actually have.
    if (total_available < static_cast<std::int32_t>(full_paged_tokens.size())) {
        full_paged_tokens.resize(total_available);
    }

    std::int32_t alloc_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    // Skip when alloc_count <= 0: a prefix deeper than total_available would make TakeFirstPages negative.
    if (alloc_count > 0) {
        OwnedPages alloc_pages = request->TakeFirstPages(alloc_count);
        kv_prefix_cache_.Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages));
    }

    MatchResult match_result = kv_prefix_cache_.Match(full_paged_tokens, MatchIntent::StateRecovery);

    std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(match_result.host.last_node);
    const std::int32_t device_matched = match_result.device.DepthInPage();
    const std::int32_t host_matched = match_result.host.DepthInPage();
    std::int32_t host_pages_needed = 0;
    if (device_matched > host_matched) {
        host_pages_needed = device_matched - host_matched;
    }

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(host_pages_needed)) {
        return {};
    }
    return fsm::ScheduleRetractEvent{&kv_prefix_cache_, &host_allocator_, match_result,
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}

LoadBackOperation GenerateLoadBackOp(const std::vector<TreeNode*>& diff, const std::vector<TreeNode*>& mamba_nodes,
                                     cache_op_id op_id) {
    std::vector<TransferPair> transfers;

    for (TreeNode* node : diff) {
        const auto& host_pages = node->Host().Pages();
        const auto& device_pages = node->Device().Pages();
        for (std::size_t i = 0; i < host_pages.size(); ++i) {
            transfers.push_back(TransferPair{CacheKind::kKV, host_pages[i], device_pages[i]});
        }
    }
    for (TreeNode* node : mamba_nodes) {
        if (node != nullptr && node->HasMambaOnHost() && node->HasMamba()) {
            transfers.push_back(TransferPair{CacheKind::kMamba, node->MambaHostSlotIndex(), node->MambaSlotIndex()});
        }
    }
    return LoadBackOperation{op_id, std::move(transfers)};
}

std::optional<WriteBackOperation> Scheduler::applyEventAndGenerateOp(Request* request,
                                                                     fsm::ScheduleRetractEvent event) {
    request->Apply(std::move(event));

    const auto& pages_to_transfer = request->GetPagesToTransfer<fsm::Retracting>();
    if (pages_to_transfer.empty()) {
        // No copy needed; advance Retracting to Retracted without an op_id.
        request->Apply(
            fsm::WriteBackDoneEvent{&kv_prefix_cache_, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr});
        return std::nullopt;
    }
    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
    CacheOpSpec spec;
    spec.request_id = request->Id();
    cache_op_tracker_[op_id] = std::move(spec);
    return WriteBackOperation{op_id, std::vector<TransferPair>(pages_to_transfer.begin(), pages_to_transfer.end()),
                              true};
}

std::optional<WriteBackOperation> Scheduler::newRetractOperation(Request* retract_request) {
    if (auto event = scheduleRetract(retract_request)) {
        if (auto op = applyEventAndGenerateOp(retract_request, std::move(*event))) {
            return std::move(*op);
        }
    } else {
        spdlog::warn("[Scheduler] Retract failed for request {}: host capacity exhausted, aborting request",
                     retract_request->Id());
        retract_request->Apply(fsm::AbortEvent{
#if TOKENSPEED_FLAT_KVCACHE
            &coordinator_
#endif
        });
    }
    return std::nullopt;
}

// By-reference so the first-chunk caller can harvest the transition's flat load pairs afterwards.
template <typename Event>
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> || std::same_as<Event, fsm::SchedulePrefillEvent>)
static PrefillOperation applyPrefillEvent(Request* request, Event& event, std::span<const std::string> flat_group_ids,
                                          const std::map<std::string, std::int32_t>& state_group_block_sizes) {
    // begin/size are PAGE-space: the occupied_pages slice new this round (Python copies it into req_to_page).
    // A first-chunk prefix hit enters during the event, so begin stays 0 and size counts the hit rows too;
    // the op's token-space INPUT window intentionally starts past the hit.
    std::int32_t begin = static_cast<std::int32_t>(request->GetOccupiedPages().size());
    request->Apply(event);
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

    auto info = request->GetPrefillInfo();
    auto op = PrefillOperation{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = info.extend_len,
        .occupied_pages = std::move(all_pages),
        .begin = begin,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};
    op.input_ids = std::vector<std::int32_t>(info.input_ids.begin(), info.input_ids.end());
    op.shifted_input_ids = std::move(info.shifted_input_ids);
    op.extend_prefix_len = info.already_scheduled_len;

    FillMambaWorkingIndices(op, request);

    FillFlatCacheInputs(op, request, flat_group_ids, state_group_block_sizes, info.already_scheduled_len,
                        info.already_scheduled_len + info.extend_len, /*per_step_out_pages=*/false,
                        /*resolve_state_execution_pages=*/true);

    return op;
}

// TODO(radix-removal): the #if !TOKENSPEED_FLAT_KVCACHE publishing arms in these op-builders go with the radix path.
PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                    std::vector<LoadBackOperation>& loadback_ops) {
#if !TOKENSPEED_FLAT_KVCACHE
    auto match = event.GetMatchResult();
#endif
    auto op = applyPrefillEvent(request, event, FlatGroupIds(), FlatStateGroupBlockSizes());
#if TOKENSPEED_FLAT_KVCACHE
    // Host-loaded pages ride the same LoadBackOperation channel as radix loadbacks.
    std::vector<std::pair<CacheBlock*, CacheBlock*>> load_pairs = event.TakeFlatLoadPairs();
    if (!load_pairs.empty()) {
        std::vector<TransferPair> transfers;
        transfers.reserve(load_pairs.size());
        FlatLoadTicket ticket;
        ticket.host_pins.reserve(load_pairs.size());
        ticket.device_blocks.reserve(load_pairs.size());
        for (const auto& [host_block, device_block] : load_pairs) {
            // Matches are read-only; this ticket is where both sides get pinned. Safe because the
            // sink drain -- the only host-pool evictor -- runs after op building every round.
            _assert(host_block->IsCached(), "host page evicted between match and load emission");
            transfers.push_back(TransferPair{CacheKind::kKV, host_block->BlockId(), device_block->BlockId()});
            ticket.host_pins.push_back(BlockRef::Share(flat_host_pool_, host_block));
            ticket.device_blocks.push_back(BlockRef::Share(block_pool_, device_block));
        }
        const cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
        flat_load_ops_.emplace(op_id, std::move(ticket));
        loadback_ops.push_back(LoadBackOperation{op_id, std::move(transfers)});
    }
#else
    (void)loadback_ops;
#endif
#if !TOKENSPEED_FLAT_KVCACHE
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct()) {
        op.mamba_cow_src_idx = match.mamba_cow_src_index;
        op.mamba_branching_seqlen = match.mamba_branching_seqlen;
    }
    // CommitChunk before acquire: prior-chunk tail pages must commit into snapshots before ReleaseSkipped frees them.
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.extend_prefix_len,
                                                op.extend_prefix_len + op.input_length, match.paged_cache);
        hybrid_prefix_cache_->PopulateOp(op);
    }
#endif
    return op;
}

PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event) {
    auto op = applyPrefillEvent(request, event, FlatGroupIds(), FlatStateGroupBlockSizes());
#if !TOKENSPEED_FLAT_KVCACHE
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.extend_prefix_len,
                                                op.extend_prefix_len + op.input_length);
        hybrid_prefix_cache_->PopulateOp(op);
    }
#endif
    return op;
}

template <typename Event>
    requires(std::same_as<Event, fsm::ScheduleDecodeEvent> ||
             std::same_as<Event, fsm::ScheduleDecodeFromRetractedEvent>)
static DecodeOperation applyDecodeEvent(Request* request, Event event, std::int32_t decode_input_tokens,
                                        std::span<const std::string> flat_group_ids,
                                        const std::map<std::string, std::int32_t>& state_group_block_sizes) {
    std::int32_t begin = static_cast<std::int32_t>(request->GetOccupiedPages().size());
    request->Apply(std::move(event));
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

    auto op = DecodeOperation{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = decode_input_tokens,
        .occupied_pages = std::move(all_pages),
        .begin = begin,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};

    FillMambaWorkingIndices(op, request);

    const std::int32_t tokens_after = request->TokenSize();
    FillFlatCacheInputs(op, request, flat_group_ids, state_group_block_sizes,
                        tokens_after - decode_input_tokens, tokens_after, /*per_step_out_pages=*/true,
                        /*resolve_state_execution_pages=*/false);

    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event) {
    const bool need_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    std::int32_t bootstrap_token = need_bootstrap_token ? request->GetLastToken() : -1;
    const bool came_from_prefill_done = request->Is<fsm::PrefillDone>();
#if !TOKENSPEED_FLAT_KVCACHE
    const std::int32_t first_pos = request->TokenSize();
#endif

    auto op = applyDecodeEvent(request, std::move(event), config_.decode_input_tokens, FlatGroupIds(),
                               FlatStateGroupBlockSizes());
    if (need_bootstrap_token) {
        op.decode_input_id = bootstrap_token;
    }
#if TOKENSPEED_FLAT_KVCACHE
    // FinalizePrefillAndReserveDecode just acquired the promised decode reserve: retire the ledger entry.
    if (came_from_prefill_done) {
        flat_reserved_pages_.erase(op.request_id);
    }
#endif
#if !TOKENSPEED_FLAT_KVCACHE
    if (hybrid_prefix_cache_) {
        if (came_from_prefill_done) {
            hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        }
        const std::int32_t target =
            DecodePagedCacheReservationEnd(first_pos, op.input_length, config_.overlap_schedule_depth);
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, first_pos, target);
        hybrid_prefix_cache_->PopulateOp(op);
    }
#endif
    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event) {
    const std::int32_t mamba_cow_src_index = event.GetMatchResult().mamba_cow_src_index;
#if !TOKENSPEED_FLAT_KVCACHE
    auto paged_cache_hit = event.GetMatchResult().paged_cache;
#endif
    request->Apply(std::move(event));
    if (!request->Is<fsm::Decoding>()) {
        throw std::logic_error(
            "Scheduler::applyEventAndGenerateOp: expected state=Decoding after loadback recovery; got state=" +
            request->StateName());
    }
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size());
    DecodeOperation op{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = config_.decode_input_tokens,
        .occupied_pages = std::move(all_pages),
        .begin = 0,
        .size = sz,
    }};
    op.decode_input_id = request->GetLastToken();
    op.hist_token_len = request->TokenSize() - 1;
    op.mamba_cow_src_idx = mamba_cow_src_index;

    FillMambaWorkingIndices(op, request);

#if !TOKENSPEED_FLAT_KVCACHE
    if (hybrid_prefix_cache_) {
        const std::int32_t target = std::max(
            request->TokenSize(),
            DecodePagedCacheReservationEnd(op.hist_token_len, op.input_length, config_.overlap_schedule_depth));
        // Preserve the existing table across retraction. Its request-local
        // tail contains state after the last published prefix checkpoint and
        // cannot be reconstructed by importing that older snapshot alone.
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.hist_token_len, target, paged_cache_hit);
        hybrid_prefix_cache_->PopulateOp(op);
    }
#endif

    FillFlatCacheInputs(op, request, FlatGroupIds(), FlatStateGroupBlockSizes(), op.hist_token_len,
                        op.hist_token_len + op.input_length, /*per_step_out_pages=*/true,
                        /*resolve_state_execution_pages=*/false);

    return op;
}

std::tuple<std::vector<ForwardOperation>, std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
Scheduler::newForwardOperation(std::vector<Request*> candidates) {
    bool allow_mixed_prefill_decode = config_.enable_mixed_prefill_decode;
#if TOKENSPEED_FLAT_KVCACHE
    // Flat recurrent target verify currently has one uniform N-token kernel
    // shape per request. Keep admission and execution unified in the C++
    // scheduler until mixed prefill + verify has an explicit backend contract.
    if (config_.decode_input_tokens > 1 && !FlatStateGroupBlockSizes().empty()) {
        allow_mixed_prefill_decode = false;
    }
#endif
    auto priority = [&](const Request* req) -> int {
        if (req->Is<fsm::Prefilling>()) return 1;
        if (req->Is<fsm::Submitted>()) return 2;
        if (req->Is<fsm::Decoding>() || req->Is<fsm::PrefillDone>()) {
            // Decode-first if mixed-batch is enabled; prefill-first otherwise.
            return allow_mixed_prefill_decode ? 0 : 3;
        }
        if (req->Is<fsm::Retracted>()) return 4;
        return 9;
    };
    // TP-determinism: tie-break on Id() so every rank schedules the same subset (a rank-varying op deadlocks NCCL).
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        int pa = priority(a);
        int pb = priority(b);
        return pa != pb ? pa < pb : a->Id() < b->Id();
    });

    std::vector<ForwardOperation> ops;
    std::int32_t token_budget = config_.max_scheduled_tokens;
    bool pushed_prefill = false;
    auto push_op = [&](auto op) {
        if (config_.role != Role::kD) {
            token_budget -= op.input_length;
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PrefillOperation>) {
            pushed_prefill = true;
        }
        ops.push_back(std::move(op));
    };
#if TOKENSPEED_FLAT_KVCACHE
    // Mid-prefill chunk ops emit no ExtendResult; only decode and prefill-completing ops owe one.
    auto note_result_owed = [&](Request* request) {
        if (!request->Is<fsm::Prefilling>()) {
            ++pending_forward_results_[request->Id()];
        }
    };
#else
    auto note_result_owed = [](Request*) {};
#endif
    std::vector<LoadBackOperation> loadback_ops;
    auto simulated_free =
        hybrid_prefix_cache_ ? hybrid_prefix_cache_->InitialSimulatedFree() : std::map<std::string, std::int32_t>{};
    for (Request* request : candidates) {
        if (token_budget <= 0 || config_.max_batch_size == ops.size()) break;

        if (request->Is<fsm::Prefilling>() && config_.role != Role::kD) {
            std::int32_t reserve_num_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            if (auto ev = schedulePrefill(request, token_budget, reserve_num_tokens, simulated_free)) {
                push_op(applyEventAndGenerateOp(request, *ev));
                note_result_owed(request);
            }
        } else if (request->Is<fsm::Submitted>() || request->Is<fsm::PrefetchDone>()) {
            // PrefetchDone: host cache populated; treat same as Submitted for forward scheduling.
            std::int32_t decode_input_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;

            if (auto ev = schedulePrefillFirstChunk(request, token_budget, decode_input_tokens,
                                                    config_.disable_l2_cache, simulated_free)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                std::vector<TreeNode*> mamba_loadback_nodes = ev->GetMambaLoadbackNodes();
                push_op(applyEventAndGenerateOp(request, std::move(*ev), loadback_ops));
                note_result_owed(request);
                // will be empty when disable_l2_cache
                if (!loadback_diff.empty() || !mamba_loadback_nodes.empty()) {
                    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, mamba_loadback_nodes, op_id));
                }
            }
        } else if (request->Is<fsm::PrefillDone>() || (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            // Mixed-batch disabled: skip ALL decode once a prefill was scheduled.
            if (!allow_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecode(request, simulated_free)) {
                push_op(applyEventAndGenerateOp(request, *ev));
                note_result_owed(request);
            }
        } else if (request->Is<fsm::Retracted>() && config_.role != Role::kP) {
            if (!allow_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecodeFromRetracted(request, simulated_free)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                std::vector<TreeNode*> mamba_loadback_nodes = ev->GetMambaLoadbackNodes();
                push_op(applyEventAndGenerateOp(request, std::move(*ev)));
                note_result_owed(request);
                if (!loadback_diff.empty() || !mamba_loadback_nodes.empty()) {
                    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, mamba_loadback_nodes, op_id));
                }
            }
        }
    }

#if TOKENSPEED_FLAT_KVCACHE
    resolveFlatStarvation(candidates, /*made_progress=*/!ops.empty());
#else
    // If all active decode requests failed, device memory is exhausted: retract the longest one.
    if (ops.empty() && !candidates.empty()) {
        std::vector<Request*> retract_candidates;
        for (Request* req : candidates) {
            if ((req->Is<fsm::Decoding>() || (req->Is<fsm::PrefillDone>() && config_.role != Role::kD)) &&
                config_.role != Role::kP) {
                retract_candidates.push_back(req);
            }
        }
        if (!retract_candidates.empty()) {
            Request* victim =
                *std::max_element(retract_candidates.begin(), retract_candidates.end(),
                                  [](const Request* a, const Request* b) { return a->TokenSize() < b->TokenSize(); });
            std::vector<WriteBackOperation> wb_ops;
            if (auto op = newRetractOperation(victim)) {
                wb_ops.push_back(std::move(*op));
            }
            return {std::vector<ForwardOperation>{}, std::move(wb_ops)};
        }
    }
#endif

    return {std::move(ops), std::move(loadback_ops)};
}

}  // namespace tokenspeed
