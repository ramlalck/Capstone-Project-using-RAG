# Runbook: SSAI Creative Miss Flow (Ad missing)
## Context
When an ad decision is returned but the referenced creative is missing/unavailable, playback may show filler or skip ad.
## Symptoms
- Ad break starts, then content resumes immediately or plays slate.
- Prisma/manifest manipulator logs show creative not found / missing media.
## Checks
1. Verify ADS decision response contains creative IDs.
2. Confirm creative availability in packager/origin for ad assets.
3. Check manifest manipulation logs for ad segment insertion failures.
4. Validate CDN availability for ad segments.
## Evidence to Collect
- Ad decision response snippet (creative id)
- Ad manifest URL and segment URLs
- Prisma/SSAI logs around the ad break timestamp
