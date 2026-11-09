# Geographic Boundary Data for NFHS-4 Analysis

## Source & Lineage
- **Dataset**: India District Boundaries GeoJSON (594 polygons)
- **Upstream Repository**: [`geohacker/india`](https://github.com/geohacker/india) (`district/india_district.geojson`)
- **Direct Download URL**: `https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson`
- **Origin & Baseline**: Derived from GADM (Database of Global Administrative Areas) level 2 administrative geometry. With **594 polygons**, this file reflects an **older administrative baseline, exact vintage unverified** (rather than a 2011 census layer which contains 640 districts, e.g. DataMeet's `Districts/Census_2011/2011_Dist.shp`).
- **GADM License & Redistribution Compliance**: GADM license terms prohibit redistribution without explicit authorization. To strictly comply with these terms, `india_districts.geojson` is excluded from the Git repository (`.gitignore`) and is downloaded automatically on first run to `Data/geo/india_districts.geojson`.
- **Implication**: This boundary file predates multiple subsequent district reorganizations (for example, Yadgir in Karnataka was created in 2009 and the 8 J&K districts were created in 2006–07, so they do not exist as distinct polygons in this geometry).
- **Local File Path**: `Data/geo/india_districts.geojson` (downloaded dynamically on first run)

## Matching Methodology & Override Resolution
- **Total NFHS-4 Districts**: 607 districts across 29 States and UTs.
- **Matched Districts**: **570 of 607 (93.90%)** successfully linked to GeoJSON polygons.
- **Match Strategy**:
  1. State-bounded exact matching on standardized names.
  2. Manual spelling and naming overrides from `Data/geo/name_overrides.csv`.
  3. High-confidence RapidFuzz string matching (threshold $\ge 90\%$) within candidate states.
- **Manual Overrides**: 108 entries in `Data/geo/name_overrides.csv` resolving alternative spellings and naming variations (e.g., `Mahesena` $\to$ `Mahesana`, `Sonipat` $\to$ `Sonepat`, `Budaun` $\to$ `Badaun`, `Kodarma` $\to$ `Koderma`, `Coddalore` $\to$ `Cuddalore`, `Paschim Medinipur` $\to$ `West Midnapore`, `Garhwal` $\to$ `Pauri Garhwal`).
- **Unmapped Districts (37 districts)**: Documented in `Outputs/reports/unmatched_districts.csv`. These 37 districts are not linked in this 594-polygon boundary set; many correspond to districts created or reorganized after this older baseline (e.g., Yadgir in Karnataka, Bodoland BTAD districts in Assam, 8 J&K districts), though some may also reflect other boundary divergences or unmapped naming variations. In the dashboard, these 37 districts are surfaced transparently in an interactive unmapped district expander.

## Attribution
- DataMeet Community (http://datameet.org/)
- GADM / geohacker/india community datasets
