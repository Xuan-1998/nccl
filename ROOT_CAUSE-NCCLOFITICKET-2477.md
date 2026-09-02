# Root cause: B300 large-message AllToAll regression (NCCL 2.28.9 -> 2.29.x)

Symptom (p6-b300.48xlarge, 8 EFA NICs/node, alltoall_perf mask 0x0):

| busbw GB/s | 1 GiB | 2 GiB | 4 GiB | 8 GiB |
|---|---|---|---|---|
| NCCL 2.28.9            | ~96  | ~98  | ~100 | ~100 |
| NCCL 2.29.7 - 2.31.2   | ~91  | ~76  | ~69  | ~66  |
| 2.31.2 + `NCCL_NCHANNELS_PER_NET_PEER=2` | 104 | 105 | 106 | 106 |

The culprit is the change to the NET branch of `ncclTopoGetNchannels()`
(src/graph/paths.cc) between v2.28.9-1 and v2.29.2-1.

2.28.9 (rank-count-scaled):
```c
nNetChannels = 2;
int netCountByBw = 1, nChannelsMax = nNetChannels;
NCCLCHECK(getLocalNetCountByBw(system, g, &netCountByBw));
// Avoid overloading channels with 8+ operations as we loose the sync warp
while (nChannelsMax*comm->nRanks > comm->p2pnChannels*4 && nChannelsMax > 1) nChannelsMax /= 2;
nNetChannels = std::max(netCountByBw, nChannelsMax);
```

2.29.2+ (NIC-bandwidth-driven, no rank scaling):
```c
NCCLCHECK(ncclTopoGetLocalNetCountByBw(system, g, &netCount, &netBw));
nNetChannels = 2;
if (netCount > 0) nNetChannels = std::max(netCount, divUp((int)netBw, (int)ncclParamP2pPerChannelNetBw()));
```

On B300 (per-GPU net bw ~50 GB/s, NCCL_P2P_PER_CHANNEL_NET_BW default 14):
`divUp(50,14) = 4` channels/peer, independent of communicator size.
Under 2.28.9 at 64 ranks: `2*64 > 32*4` is false, so 2 channels/peer.

Why 4 channels/peer hurts: NCCL INIT/NET logs show every channel of a given
peer maps to the SAME NIC (rail-aligned p2p). With 8 ranks/node, each NIC
serves 8 remote-peer connections; 4 channels/peer means 32 concurrent proxy
streams per NIC vs 16 with 2 channels/peer. Beyond ~8 MiB/peer the per-stream
pipelining (NCCL_P2P_NET_CHUNKSIZE x NCCL_STEPS in-flight) stops hiding the
extra fragmentation and busbw collapses ~35%.

Controls (measured, 8-node B300, NCCL 2.31.2):
- `NCCL_NCHANNELS_PER_NET_PEER=4` == baseline (confirms baseline picks 4)
- `NCCL_NCHANNELS_PER_NET_PEER=8`: worse than 2 (partial recovery only)
- `NCCL_MAX_P2P_NCHANNELS=64`: no effect (clamp not binding at 64 ranks)
- `NCCL_P2P_NET_CHUNKSIZE=1M/2M`: partial (77 GB/s at 8 GiB)
- mask 0x7 (one GPU/node comms): flat ~94 GB/s on ALL versions -> transport healthy
- P6-B200 (lower per-GPU net bw): formula change is benign there; flat ~50 GB/s

The commit on this branch reverts the formula to the 2.28.9 behavior for
demonstration. See branch `fix-alltoall-p2p-channel-scaling` for the
upstream-ready fix.
