# Runbook: Live Buffering / Rebuffering Spikes (CDN Path)
## Symptoms
- Live streams show frequent buffering; VOD may be fine.
- Issues concentrated in a region or ISP.
## Immediate Checks
1. Compare Live vs VOD behavior on same device/network.
2. Verify CDN health for affected region (edge errors, cache hit ratio).
3. Check Origin/Packager latency; confirm manifests are being served.
4. Validate player download metrics: segment fetch time, throughput.
## Likely Causes
- Regional CDN degradation
- Origin saturation
- Bad cache behavior (low hit ratio)
## Mitigations
- Failover CDN (if configured)
- Reduce segment size / adjust ABR ladder (temporary)
## Evidence to Collect
- Timestamp range, region, ISP, device type
- Segment URLs, manifest URL, response codes
