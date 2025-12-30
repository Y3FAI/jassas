# Jassas Benchmark Results

> Last updated: 2025-12-30

## Test Environment

- **Machine:** MacOS Darwin 24.6.0
- **Python:** 3.13
- **Documents:** 1,003
- **Vocabulary:** 28,926 tokens
- **Index Entries:** 404,689

---

## Optimization History

### Phase 1: Baseline (3 docs)

| Metric | Value |
|--------|-------|
| Cold Start | 5.92s |
| BM25 | 1.80ms (3.7%) |
| Vector | 44.74ms (92.7%) |
| Throughput | 70.71 QPS |
| Avg Latency | 14.14ms |

### Phase 2: Scale to 1000 docs (No Optimizations)

| Metric | Value | Change |
|--------|-------|--------|
| Cold Start | 6.27s | +6% |
| BM25 | 30.70ms (38.4%) | **17x slower** |
| Vector | 44.41ms (55.6%) | Same |
| Throughput | 50.27 QPS | -29% |
| Avg Latency | 19.89ms | +41% |

**Bottleneck Identified:** BM25 SQL query doing table scans for `frequency` column.

### Phase 3: Covering Index Applied

```sql
CREATE INDEX idx_inverted_search
ON inverted_index(vocab_id, doc_id, frequency);
```

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| BM25 Avg | 30.70ms | 8.33ms | **3.7x faster** |
| BM25 Worst (`government services`) | 90.92ms | 18.05ms | **5x faster** |
| BM25 % of Total | 38.4% | 14.7% | -24% |

### Phase 4: Parallelization Applied

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    future_bm25 = executor.submit(self._bm25_search, ...)
    future_vector = executor.submit(self._vector_search, ...)
```

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Throughput | 45.66 QPS | 53.33 QPS | +17% |
| Avg Latency | 21.90ms | 18.75ms | -14% |

---

## Final Results (After All Optimizations)

### Latency Breakdown

```
                        Latency Breakdown (ms)
╭─────────────────────┬──────┬───────┬────────┬──────┬───────┬────────╮
│ Query               │ Norm │  BM25 │ Vector │  RRF │ Fetch │  Total │
├─────────────────────┼──────┼───────┼────────┼──────┼───────┼────────┤
│ government services │ 0.34 │ 18.05 │ 147.50 │ 0.04 │  6.48 │ 168.68 │
│ renew passport      │ 0.03 │  2.65 │  29.72 │ 0.02 │  6.37 │  38.82 │
│ الأمن السيبراني     │ 0.27 │ 10.57 │  12.28 │ 0.03 │  6.36 │  29.36 │
│ national platform   │ 0.02 │  5.72 │  10.63 │ 0.03 │  6.95 │  23.35 │
│ cybersecurity       │ 0.02 │  3.55 │  11.02 │ 0.03 │  6.33 │  20.94 │
╰─────────────────────┴──────┴───────┴────────┴──────┴───────┴────────╯
```

### Component Distribution

| Component | Avg (ms) | % of Total |
|-----------|----------|------------|
| Normalization | 0.07 | 0.1% |
| BM25 Search | 7.14 | 14.9% |
| Vector Search | 33.77 | 70.6% |
| RRF Merge | 0.03 | 0.1% |
| Result Fetch | 6.81 | 14.2% |

### Throughput

| Metric | Value |
|--------|-------|
| Total Time (50 queries) | 0.94s |
| Throughput | **53.33 QPS** |
| Avg Latency | **18.75ms** |
| Min Latency | 17.41ms |
| Max Latency | 26.52ms |
| P95 Latency | 20.60ms |

### Cold Start

| Component | Time |
|-----------|------|
| Model + Index Load | 6.55s |

---

## Remaining Bottlenecks

| Component | % of Time | Potential Fix |
|-----------|-----------|---------------|
| Vector Search | 70.6% | ONNX Runtime, quantization |
| Result Fetch | 14.2% | Lazy load `clean_text` |
| BM25 Search | 14.9% | ✅ Optimized |
| Cold Start | 6.55s | Lazy model loading |

---

## API Benchmarks

*TODO: Add API response time measurements*
