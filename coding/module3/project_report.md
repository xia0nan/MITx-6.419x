## Research question (2 pts)

Did police disruptions produce durable shifts in leadership and structure in the CAVIAR network? I test whether the first major seizure and subsequent pivot (around phases 4→5) led to: (i) a redistribution of leadership from the initial organizer(s) to new actors; (ii) persistent changes in the global structure (clustering, assortativity, core–periphery). Sub‑questions: How do hub/authority roles (directed) and betweenness/eigenvector (undirected) evolve across phases? Are the rank changes brief turbulence or regime shifts that persist beyond one phase?

## Methodology (2 pts)

Data are 11 two‑month phases of the CAVIAR wiretap network. I analyze both undirected binary graphs (ties if any communication in either direction) and directed weighted graphs (call counts). For roles I compute, per phase: degree, betweenness and eigenvector centralities; and HITS hubs/authorities. I summarize leadership using top‑k rank series (k=10) and quantify stability as Kendall’s τ between consecutive phases. I conduct an event study around 4→5 and run a permutation test of phase labels for actor n12 (pre 1–4 vs post 5–11; 5,000 shuffles) to assess whether its post‑event mean betweenness exceeds the pre‑event mean beyond chance. For structure I compute per‑phase transitivity, average clustering, degree assortativity, and maximum k‑core index and size. These measures are within the module’s toolkit; they jointly capture local cohesion, degree mixing, and core–periphery.

## Results (2 pts)

Leadership reconfigures at 4→5 and then partially stabilizes. n12’s betweenness jumps by ≈0.26 and remains high post‑event; the permutation test yields p≈0.013, indicating a statistically meaningful increase. n1 stays extremely central, but its directed role flips: strong hub early, dominant authority mid‑course, then hub again—consistent with temporary inward consolidation after the seizure and later outward coordination. Rank stability is modest: Kendall τ for top hubs shows several weak or negative values (e.g., 5→6 ≈ −0.59; 9→10 ≈ −0.88), quantifying churn during reconfiguration windows. Structural series show persistent disassortativity (≈ −0.65 to −0.38), moderate clustering with peaks during reorganization (avg clustering ≈ 0.53 at phase 6), and a mostly stable max k‑core index of 3 whose membership size varies; the final phase drops to index 2 with a larger core, compatible with diffusion under pressure. Overall, shocks triggered a pivot and leadership redistribution anchored by n12 while retaining a small operational core.

Figures:
- Structural trends over phases (transitivity, clustering, assortativity, max k‑core): `fig_struct_series.png`.
- HITS hub/authority trajectories for n1 and n3: `fig_hits_track.png`.
- Event study (betweenness Δ, phases 4→5 for n1/n3/n12): `fig_event_bet_4_5.png`.

## Discussion (2 pts)

The evidence supports a resilience‑through‑reassignment narrative: enforcement shocks do not decapitate the network; instead, authority migrates toward logistics actors (n12) while the long‑standing organizer (n1) alternates between outward broadcasting and inward aggregation. Disassortative mixing and a stable core index imply a centralized, role‑differentiated structure that can rewire spokes quickly. Practical implication: focusing solely on high‑degree hubs risks missing emerging authorities; combining directed (hub/authority) with undirected (betweenness/eigenvector) metrics better identifies succession paths. Limitations include partial observation (wiretap coverage), phase granularity (two‑month bins), and reliance on shortest‑path centrality. As robustness, one could add time‑decayed centralities or bootstrap the rank trajectories; nonetheless, the permutation evidence for n12 and the repeated role flips of n1/n3 make the reorganization claim credible within the course toolkit.


