# Runbook: Packager / Origin Manifest Errors
## Symptoms
- Manifest fetch returns 404/5xx or malformed playlist/MPD
- Players fail before requesting segments
## Checks
1. Verify packager job status and origin storage availability.
2. Confirm manifest URL path and asset identifiers.
3. Inspect manifest for missing EXT-X-KEY (HLS) or ContentProtection (DASH).
4. Ensure renditions are present and aligned (audio/video tracks).
## Mitigations
- Restart packager job; re-publish asset
- Validate CMS metadata and publishing pipeline
## Evidence to Collect
- Manifest URL + response headers
- Asset ID, publish time, ingest pipeline logs
