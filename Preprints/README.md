# Preprints for AeroFlyway (`crvscience-aeroflyway`)

**Author:** Bryan P. White (ORCID: [0000-0003-4191-3511](https://orcid.org/0000-0003-4191-3511))  
**Hub:** https://crvscience.com/apps

This directory contains the self-contained LaTeX source (`.tex`) and single-column PDF (`.pdf`) editions for both:

1. **Companion Application Preprint (`APP-PREPRINT-13`)**
   - **Title:** AeroFlyway: Tracking Bird Migration with Weather Radar, Skyglow Maps, and Nighttime Flight Calls
   - **Files:** `aeroflyway-avian-migration-radar.tex`, `aeroflyway-avian-migration-radar.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222366
   - **Figshare Mirror:** https://figshare.com/articles/preprint/AeroFlyway_Tracking_Bird_Migration_with_Weather_Radar_Skyglow_Maps_and_Nighttime_Flight_Calls/35013739
   - **Web Preprint:** https://crvscience.com/apps/preprints/aeroflyway-avian-migration-radar

2. **CRV Science Public Apps Network Overview (`APP-PREPRINT-01`)**
   - **Title:** The CRV Science App Network: Eighteen Open-Source Rust Programs for Science, Data, and Simulation
   - **Files:** `crv-science-public-apps-network.tex`, `crv-science-public-apps-network.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222398
   - **Figshare Mirror:** https://figshare.com/articles/preprint/The_CRV_Science_App_Network_Seventeen_Open-Source_Rust_Programs_for_Science_Data_and_Simulation/35013736
   - **Web Preprint:** https://crvscience.com/apps/preprints/crv-science-public-apps-network

## Compiling Standalone LaTeX

Both `.tex` files inline their bibliography so each compiles standalone with standard `pdflatex`:

```bash
pdflatex aeroflyway-avian-migration-radar.tex
pdflatex crv-science-public-apps-network.tex
```
