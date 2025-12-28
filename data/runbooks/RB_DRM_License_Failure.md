# Runbook: DRM License Failure (Widevine / PlayReady / FairPlay)
## Symptoms
- Playback fails at start; error dialogs may mention license.
- Black screen with audio sometimes indicates video decryption issues.
## Checks
1. Determine DRM type by device: Android -> Widevine, iOS/tvOS -> FairPlay, some STB -> PlayReady.
2. Confirm license server reachability (DNS, TLS handshake).
3. Check entitlement/token validity (expired JWT can fail license acquisition).
4. Validate KID/content key presence in manifest and license request.
## Common Causes
- Expired auth token
- Wrong DRM configuration for asset
- License server outage or TLS mismatch
## Evidence to Collect
- Device model, OS version, app version
- License request/response (status code, error body)
- Asset ID and manifest URL
