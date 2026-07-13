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

#include "cache/forward_cache_ops.h"

#include "resource/allocator/paged_cache_group.h"
#include "scheduler/types.h"

namespace tokenspeed {

bool PrefillFirstChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables, const CoordinatorMatch& hit,
                       std::int32_t num_new_tokens) {
    coordinator.ClaimCommonPrefix(tables, hit);
    return coordinator.Acquire(tables, num_new_tokens);
}

std::vector<std::pair<CacheBlock*, CacheBlock*>> LoadHostExtension(KvCacheCoordinator& coordinator,
                                                                   std::vector<BlockTable>& tables,
                                                                   const CoordinatorMatch& host) {
    return coordinator.LoadHostExtension(tables, host);
}

bool PrefillChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                  std::span<const std::string> content_hashes, std::int32_t num_tokens,
                  std::int32_t num_computed_tokens) {
    return DecodeStep(coordinator, tables, content_hashes, /*first_base_block=*/0, num_tokens, num_computed_tokens);
}

bool DecodeStep(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                std::span<const std::string> content_hashes, std::int32_t first_base_block, std::int32_t num_tokens,
                std::int32_t num_computed_tokens, std::int32_t decode_width) {
    // CacheFullBlocks before ReclaimExpired: registration skips null holes, so
    // the reverse order would lose the punched pages' hashes forever.
    // ReclaimExpired before Acquire: the slide's freed pages fund this chunk
    // (admission gates credit them via BlocksReclaimableAt in lockstep).
    // num_computed_tokens is the chunk end: state groups register every complete emitted snapshot below it.
    coordinator.CacheFullBlocks(tables, content_hashes, first_base_block, num_computed_tokens);
    coordinator.ReclaimExpired(tables, num_computed_tokens);
    return coordinator.Acquire(tables, num_tokens, decode_width);
}

bool FinalizePrefillAndReserveDecode(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                                     std::span<const std::string> content_hashes, std::int32_t reserve_tokens,
                                     std::int32_t registration_end_tokens, std::int32_t decode_width) {
    // KV and recurrent State were both computed through registration_end_tokens. Unlike KV,
    // recurrent State must retain an earlier snapshot: the page that supplies the pending
    // first decode's input State.
    coordinator.CacheFullBlocks(tables, content_hashes, /*first_base_block=*/0, registration_end_tokens);
    coordinator.ReclaimExpiredForFirstDecode(tables, registration_end_tokens, decode_width);
    return coordinator.Acquire(tables, reserve_tokens, decode_width);
}

std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config) {
    std::vector<KvCacheSpec> specs;
    specs.reserve(config.paged_cache_groups.size());
    for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
        const std::int32_t block_size = group.block_size > 0 ? group.block_size : config.block_size;
        // family=State marks trailing-window prefix reuse and covers both SWA and
        // linear-attention groups; only a State group WITHOUT SlidingWindow
        // retention is a mamba-style state group.
        if (group.family == PagedCacheGroupFamily::State &&
            group.retention != PagedCacheGroupConfig::Retention::SlidingWindow) {
            specs.push_back(KvCacheSpec{
                .kind = AttnKind::kMambaState,
                .block_size = block_size,
                .sliding_window = 0,
            });
            continue;
        }
        const bool is_swa = group.retention == PagedCacheGroupConfig::Retention::SlidingWindow;
        specs.push_back(KvCacheSpec{
            .kind = is_swa ? AttnKind::kSlidingWindow : AttnKind::kFull,
            .block_size = block_size,
            .sliding_window = is_swa ? group.sliding_window_tokens.value_or(0) : 0,
        });
    }
    return specs;
}

void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables) {
    if (tables.empty()) {
        return;  // request never got tables, or a failure path already released them
    }
    coordinator.Free(tables);
}

std::map<std::string, std::vector<std::int32_t>> BuildFlatBlockTables(const std::vector<BlockTable>& tables,
                                                                      std::span<const std::string> group_ids) {
    _assert(tables.size() == group_ids.size(), "BuildFlatBlockTables: tables/group_ids size mismatch");
    std::map<std::string, std::vector<std::int32_t>> out;
    for (std::size_t i = 0; i < tables.size(); ++i) {
        out.emplace(group_ids[i], BlockTablePageIds(tables[i]));
    }
    return out;
}

}  // namespace tokenspeed
