# Quantifying the Operating Envelope of Each Architecture Concept

## Key Message

The proposed mechanisms do not provide a fixed gain under all workloads.  
Each mechanism removes a different source of inefficiency, so its benefit depends on workload shape, batching, precision, SLO slack, and speculation quality.

We therefore define an **operating envelope** for each mechanism:

\[
\boxed{
\text{Workload / system conditions}
\rightarrow
\text{mechanism effectiveness}
\rightarrow
\text{runtime selection}
}
\]

Outside the high-benefit region, the design should gracefully fall back to the baseline rather than force an inefficient configuration.

---

## 1. P-Core: Prefill-Dominated Workloads

### Main observation

Prefill processes many prompt tokens in parallel and is typically **compute intensive**, whereas decode processes one new token per request per iteration and is much more likely to be limited by memory bandwidth and low compute utilization [1,2]. :contentReference[oaicite:0]{index=0}

Define the prefill energy fraction as

\[
\phi_P =
\frac{E_{\text{prefill}}}
{E_{\text{prefill}}+E_{\text{decode}}}.
\]

If the P-Core improves prefill efficiency by \(g_P\), the maximum request-level saving is approximately

\[
\boxed{
\Delta E_{\text{req}}
\approx
\phi_P
\left(
1-\frac{1}{g_P}
\right)
}
\]

Therefore, the P-Core becomes useful only when prefill occupies a sufficiently large fraction of total request energy.

### Proposed operating envelope

- **High-benefit region:** prefill contributes at least **20% of request energy** **[Our characterization, Fig. X]**.
- On our evaluated platform, this crossover occurs around **2K prompt tokens** **[Our characterization, Fig. X]**.
- Splitwise independently shows that prompt and decode phases have strongly different compute/memory characteristics, and its batching characterization explicitly studies prompt batches from **128 to 8192 tokens** [1]. :contentReference[oaicite:1]{index=1}
- Production workloads can have very different prompt/output ratios; therefore, P-Core benefit should be evaluated as a function of the **P/D ratio**, rather than assuming one fixed gain [1]. :contentReference[oaicite:2]{index=2}

For our representative workloads:

- **RAG: 105K input / 17 output tokens** → strongly prefill dominated **[Our workload characterization, Fig. X]**.
- **Chat: 512 input / 4096 output tokens** → decode dominated **[Our workload characterization, Fig. X]**.
- In the latter case, prefill contributes only **10.8%** of request energy **[Our characterization, Fig. X]**; even a **3.5×** P-Core improvement translates to only about **8%** request-level energy reduction **[Our analytical model, Eq./Fig. X]**.

### Design implication

\[
\boxed{
\text{Long prompt / high P:D ratio}
\rightarrow
\text{P-Core}
}
\]

\[
\boxed{
\text{Decode-heavy workload}
\rightarrow
\text{D-Core / baseline decode path}
}
\]

AFlex provides additional evidence that Attention/FFN components can have different energy-optimal operating points and reports up to **49% lower energy per token** through disaggregation and independent frequency control [3]. :contentReference[oaicite:3]{index=3}

---

## 2. P-Core Token Window: Speculative Acceptance Determines the Gain

A speculative window amortizes one target-model weight read over multiple candidate tokens.

However, rejected draft tokens provide no useful output, and the draft model itself incurs overhead.

Let

- \(k\): speculative window size,
- \(\alpha\): token acceptance probability,
- \(c=T_{\text{draft}}/T_{\text{target}}\): relative draft-model cost.

A standard speculative-decoding performance model is

\[
\boxed{
S(\alpha,k,c)
=
\frac{1-\alpha^{k+1}}
{(1-\alpha)(1+kc)}
}
\]

[4]. :contentReference[oaicite:4]{index=4}

The equation shows that the useful region is determined jointly by

\[
\boxed{
(\alpha,k,c)
}
\]

rather than by \(k\) alone.

Speculative decoding has demonstrated approximately **2–3×** wall-clock acceleration in the original study while preserving the target model's output distribution [4]. :contentReference[oaicite:5]{index=5}

### Proposed operating envelope

For our architecture:

- Robust high-benefit region:

\[
\boxed{
\alpha \gtrsim 0.5{-}0.6
}
\]

**[Our speculative-execution sweep, Fig. X]**.

- The runtime should adapt \(k\) based on measured acceptance rather than fixing one large window for every request **[Our runtime policy, Fig. X]**.
- Large \(k\) becomes unattractive as acceptance decreases because more draft work is discarded **[Our analytical model, Fig. X]**.

### Design implication

\[
\boxed{
\text{High acceptance}
\rightarrow
\text{larger token window}
}
\]

\[
\boxed{
\text{Low acceptance}
\rightarrow
\text{smaller window / normal decode}
}
\]

---

## 3. D-Core: Small-Batch, Memory-Dominated Decode

### Main observation

Decode has low arithmetic intensity at small batch sizes because each request contributes only one new token per iteration. Sarathi-Serve shows that prefill can saturate GPU compute even at small batch sizes, whereas decode exhibits low compute utilization and benefits strongly from batching [2]. :contentReference[oaicite:6]{index=6}

This motivates a smaller, memory-oriented D-Core that avoids paying for compute resources that are poorly utilized during low-batch decode.

Low precision further reduces both weight traffic and KV-cache traffic:

- Atom reports up to **7.73×** end-to-end throughput over FP16 with W4A4 quantization under the same latency target [5]. :contentReference[oaicite:7]{index=7}
- KIVI's 2-bit KV cache reduces peak memory by **2.6×**, enables up to **4× larger batches**, and improves throughput by **2.35–3.47×** on real LLM inference workloads [6]. :contentReference[oaicite:8]{index=8}

### Proposed operating envelope

Our architecture model predicts:

- **Batch ≤ 8 + W4/KV4:** highest D-Core benefit, approximately **8–16×** **[Our architecture model, Fig. X]**.
- **Batch 8–64:** intermediate benefit of approximately **4–8×** **[Our architecture model, Fig. X]**.
- **Very large batch or W8/KV8:** benefit approaches the underlying memory-byte reduction, approximately **2.5–3.5×** **[Our architecture model, Fig. X]**.

These are **architecture-specific projected gains**, not universal literature values.

### Long-context crossover

KV-cache storage grows approximately as

\[
\boxed{
M_{\text{KV}}
=
2
\times
L
\times
B
\times
S
\times
N_{\text{KV-head}}
\times
d_{\text{head}}
\times
b_{\text{KV}}
}
\]

where \(L\) is layer count, \(B\) batch size, \(S\) context length, and \(b_{\text{KV}}\) bytes per element.

KIVI confirms that KV memory grows with both batch size and context length and can become the inference bottleneck [6]. :contentReference[oaicite:9]{index=9}

For our reference long-context case, KV traffic/storage exceeds weight traffic at approximately **4K context and batch 128** **[Our memory model, Fig. X]**.

Therefore:

\[
\boxed{
\text{Short context}
\rightarrow
\text{weight precision is dominant}
}
\]

\[
\boxed{
\text{Long context + high concurrency}
\rightarrow
\text{KV precision becomes dominant}
}
\]

### Design implication

Use multiple small D-Cores for low-batch decode, while sufficiently large decode batches can be redirected to the more compute-capable P-Core **[Our Task-1 architecture design]**.

---

## 4. Fabric: Benefit Comes from Hiding Waiting, Not Only Moving KV Faster

P/D disaggregation requires transferring the KV state between phases.

The transfer time can be approximated by

\[
\boxed{
T_{\text{KV}}
\approx
\frac{S_{\text{KV}}}
{BW_{\text{link}}}
}
\]

while the fabric is beneficial when communication can overlap computation and eliminate idle waiting.

Splitwise demonstrates that optimized P/D phase splitting can transfer KV state over high-speed cluster interconnects with very small user-visible latency impact [1]. :contentReference[oaicite:10]{index=10}

More generally, the fabric is worthwhile when

\[
\boxed{
E_{\text{saved waiting}}
>
E_{\text{KV transfer}}
}
\]

or approximately

\[
\boxed{
P_{\text{wait}}
T_{\text{hidden}}
>
E_{\text{fabric}}
}
\]

### Proposed operating envelope

Our current fabric model gives the high-benefit region as:

- at least **2 in-flight requests per P–D pair** **[Our queueing analysis, Fig. X]**;
- link utilization below approximately **60%** **[Our network sweep, Fig. X]**.

When the link approaches saturation, communication can no longer be hidden and the design falls back toward software P/D disaggregation **[Our fabric characterization, Fig. X]**.

The broader importance of workload-, network-, and resource-dependent disaggregation is also supported by recent AFD design-space studies spanning input/output lengths, KV reuse, SLOs, and interconnect conditions [7]. :contentReference[oaicite:11]{index=11}

---

## 5. Runtime: Exploit SLO Slack and Workload Heterogeneity

The runtime does not create additional compute efficiency by itself.  
Its role is to route each request to the mechanism that lies inside its effective operating envelope.

Define normalized SLO slack as

\[
\boxed{
s_i
=
\frac{
T_{\text{SLO},i}
-
\hat T_i
}{
T_{\text{SLO},i}
}
}
\]

When

\[
s_i\approx0,
\]

there is little freedom to reduce precision, frequency, resources, or speculative verification effort.

As slack increases, the scheduler gains freedom to trade performance headroom for energy efficiency.

DynamoLLM demonstrates that exploiting workload heterogeneity while dynamically adjusting instance count, model parallelism, and GPU frequency can reduce serving energy by up to **53%** while meeting latency SLOs [8]. :contentReference[oaicite:12]{index=12}

### Proposed operating envelope

Our runtime characterization identifies the main useful region as:

- median SLO slack

\[
\boxed{
s_{p50}\gtrsim30\%
}
\]

**[Our runtime sweep, Fig. X]**;

- at least **two distinguishable workload classes** **[Our runtime sweep, Fig. X]**;
- request-length prediction error below approximately **50%** **[Our predictor sensitivity study, Fig. X]**.

Within this region, the projected runtime-level gain is approximately **1.3–1.8×** **[Our runtime evaluation, Fig. X]**.

For a homogeneous workload under a tight SLO, the benefit falls toward approximately **1.1×** **[Our runtime evaluation, Fig. X]**.

### Design implication

\[
\boxed{
\text{Runtime}
=
f(
P/D\ ratio,
B,
L_{\text{ctx}},
L_{\text{out}},
\alpha,
\text{precision},
\text{SLO slack}
)
}
\]

The runtime should not force every request onto the specialized hardware. It should keep requests inside the operating envelope of each architecture component.

---

# Overall Operating-Envelope View

| Concept | Main control variable | High-benefit region | Graceful fallback |
|---|---|---|---|
| **P-Core** | P/D ratio, prompt length | Prefill-heavy requests; \(\phi_P\ge20\%\) **[Our Fig. X]** | Decode-oriented path |
| **Token Window** | \(k,\alpha,c\) | High speculative acceptance; \(\alpha\gtrsim0.5{-}0.6\) **[Our Fig. X]** | Reduce \(k\) or normal decode |
| **D-Core** | Batch, precision, context | Small-batch W4/KV4 decode **[Our Fig. X]** | Route large batches to P-Core |
| **Fabric** | Concurrency, KV size, link load | ≥2 in-flight/P–D pair and <60% link utilization **[Our Fig. X]** | Software P/D disaggregation |
| **Runtime** | SLO slack, workload heterogeneity | \(s_{p50}\ge30\%\), heterogeneous requests **[Our Fig. X]** | Baseline scheduling |

The key architectural argument is therefore not that every mechanism always achieves its peak gain, but that:

\[
\boxed{
\text{Different workload regions expose different forms of waste}
}
\]

and

\[
\boxed{
\text{The runtime maps each request to the architecture mechanism
that removes the dominant waste.}
}
\]

---

# References

**[1]** P. Patel et al., **“Splitwise: Efficient Generative LLM Inference Using Phase Splitting,”** ISCA 2024.  
Supports: prefill/decode asymmetry, compute- vs. memory-intensive phases, P/D disaggregation, KV transfer, heterogeneous serving. :contentReference[oaicite:13]{index=13}

**[2]** A. Agrawal et al., **“Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve,”** OSDI 2024.  
Supports: compute-saturating prefill, low-utilization small-batch decode, batching dependence. :contentReference[oaicite:14]{index=14}

**[3]** C. Hu et al., **“Energy-Efficient LLM Serving via Disaggregated Attention–FFN and Flexible Frequency Scaling,”** 2026.  
Supports: component-specific frequency sensitivity and up to **49% energy/token reduction**. :contentReference[oaicite:15]{index=15}

**[4]** Y. Leviathan et al., **“Fast Inference from Transformers via Speculative Decoding,”** ICML 2023.  
Supports: speculative-decoding acceptance/window cost model and **2–3×** measured acceleration. :contentReference[oaicite:16]{index=16}

**[5]** Y. Zhao et al., **“Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving,”** MLSys 2024.  
Supports: W4A4 inference and up to **7.73×** throughput improvement over FP16. :contentReference[oaicite:17]{index=17}

**[6]** Z. Liu et al., **“KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache,”** ICML 2024.  
Supports: KV-cache memory bottleneck, **2.6×** lower peak memory, up to **4×** larger batch, and **2.35–3.47×** throughput improvement. :contentReference[oaicite:18]{index=18}

**[7]** H. Wu et al., **“How Far Can Disaggregation Go? A Design-Space Exploration of Attention-FFN Disaggregation for Efficient MoE LLM Serving,”** 2026.  
Supports: workload-dependent operating envelopes across input/output lengths, KV reuse, SLOs, resource allocation, and interconnect conditions. :contentReference[oaicite:19]{index=19}

**[8]** J. Stojkovic et al., **“DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency,”** 2024.  
Supports: SLO-aware runtime reconfiguration under heterogeneous workloads and up to **53% energy reduction**. :contentReference[oaicite:20]{index=20}