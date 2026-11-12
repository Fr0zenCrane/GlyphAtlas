# Licensing and attribution

This is a release candidate prepared for maintainer review. A repository-wide license for the original GlyphAtlas integration has not yet been selected by the maintainer. Apache-2.0 is proposed; no license grant is fabricated by this preparation step. Select the license and add its full text before labeling or publishing this candidate as an open-source release.

Third-party components remain separately governed:

| Component | Treatment |
| --- | --- |
| FLUX-Text implementation | Pinned source fetched from [AMAP-ML/FluxText](https://github.com/AMAP-ML/FluxText), excluded from this archive. The pinned snapshot contains no root LICENSE file; this project does not relicense it. |
| FLUX.1-Fill-dev weights | Downloaded separately under the [upstream model terms](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev). No weights redistributed. |
| OCR weights and Arial fonts | Fetched separately from pinned upstream assets, with original attribution and hashes. Availability for download is not a new license grant. |
| DejaVu fallback font | Original notice in `assets/fonts/DejaVu-LICENSE.txt`. |
| Noto fallback fonts | SIL Open Font License notices shipped alongside each font. |
| FreeSerif fallback font | Original FreeFont/GPL notices and font exception shipped in `assets/fonts/`. |
| Datasets and checkpoints | Not included. Providers' terms apply independently of code licensing. |

`NOTICE` and `provenance/` preserve source identities. `vendor/FluxText/` is a local installed dependency ignored by Git and excluded by the release packager. Review upstream redistribution permissions before changing that packaging policy.
