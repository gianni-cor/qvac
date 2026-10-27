#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <llama.h>

#include "CacheLedger.hpp"

namespace qvac_lib_inference_addon_llama::cache {

// Entries of `prompt` a state described by `state`, with its `checkpoints`,
// can serve without decoding them again. When the memory can be trimmed back
// to any position it is the whole shared prefix. Otherwise the whole state
// serves only when it is a prefix of the prompt, and a shorter shared prefix
// is served by the longest checkpoint within it.
inline size_t reusablePrefix(
    const Ledger& state, const Checkpoints* checkpoints, const Ledger& prompt,
    bool trimmable) {
  const size_t shared = commonPrefix(state, prompt);
  if (trimmable || shared == state.entries.size()) {
    return shared;
  }
  size_t best = 0;
  if (checkpoints != nullptr) {
    for (const Checkpoint& checkpoint : *checkpoints) {
      const size_t size = checkpoint.ledger.entries.size();
      if (size > best && size <= shared &&
          commonPrefix(checkpoint.ledger, prompt) == size) {
        best = size;
      }
    }
  }
  return best;
}

// Index of the largest of `reusable`, the earliest on a tie, when it exceeds
// `minimum`. Candidates are listed cheapest first, so a tie keeps the one
// that costs least to bring in.
inline std::optional<size_t>
longestReusable(std::span<const size_t> reusable, size_t minimum) {
  std::optional<size_t> best;
  for (size_t i = 0; i < reusable.size(); ++i) {
    if (reusable[i] > minimum &&
        (!best.has_value() || reusable[i] > reusable[*best])) {
      best = i;
    }
  }
  return best;
}

// A conversation state another request can start from: a parked batch
// sequence, a RAM-tier entry or a cacheKey file. Valid only for the call that
// receives it; the conversation it belongs to is never changed.
struct DonorState {
  // Puts the state into `seqId`, whose memory is empty. False when it could
  // not; the caller then clears the sequence.
  std::function<bool(llama_context*, llama_seq_id)> copyInto;
  // Ledger words of the state, in the format `serialize` writes.
  const std::vector<llama_token>* ledgerWords = nullptr;
  // Checkpoints of the state, or null when it has none at hand.
  const Checkpoints* checkpoints = nullptr;
  // Where the state lives, for logs.
  std::string source;
};

// Adopts `donor` into the requesting sequence; true when it did.
using DonorAdopter = std::function<bool(const DonorState&)>;

// Finds, among the states kept outside a request's own sequence, the one
// that serves the most of its prompt. Implemented by the owner of those
// states: the single-prompt `CacheManager` and the `ContinuousBatchScheduler`.
class CacheSourceProvider {
public:
  CacheSourceProvider() = default;
  CacheSourceProvider(const CacheSourceProvider&) = delete;
  CacheSourceProvider& operator=(const CacheSourceProvider&) = delete;
  CacheSourceProvider(CacheSourceProvider&&) = delete;
  CacheSourceProvider& operator=(CacheSourceProvider&&) = delete;
  virtual ~CacheSourceProvider() = default;

  // Hands `adopt` the state with the longest `reusablePrefix` of `prompt`
  // when it exceeds `minimum` (what the request's own sequence already
  // serves). Returns what `adopt` returned, or false when no state qualified.
  virtual bool offerDonor(
      llama_seq_id seqId, const Ledger& prompt, bool trimmable, size_t minimum,
      const DonorAdopter& adopt) = 0;
};

} // namespace qvac_lib_inference_addon_llama::cache
