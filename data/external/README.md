# Observed rainfall for external validation

`nasa_power_bd_daily_rain_1991_2024.csv` — daily corrected precipitation
(PRECTOTCORR, mm/day, MERRA-2) from the NASA POWER daily API v2.10,
community AG, 1 Jan 1991 – 31 Dec 2024, 12,419 days, no missing values.

| Column | District | Lat | Lon |
|---|---|---|---|
| sylhet | Sylhet | 24.90 | 91.87 |
| sunamganj | Sunamganj | 25.07 | 91.40 |
| moulvibazar | Moulvibazar | 24.48 | 91.78 |
| netrokona | Netrokona | 24.88 | 90.73 |
| feni | Feni | 23.01 | 91.40 |
| kurigram | Kurigram | 25.81 | 89.64 |

Each column was downloaded with a request of the form

```
https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=AG&latitude=24.90&longitude=91.87&start=19910101&end=20241231&format=JSON
```

No account or API key is needed.

Acknowledgement requested by the data provider: *These data were obtained from
the NASA Langley Research Center (LaRC) POWER Project funded through the NASA
Earth Science Directorate Applied Science Program.*

Note: MERRA-2 corrected precipitation is not homogeneous in time over this
region — seasonal means in 2015–2024 are about twice those of 1991–2014. The
experiment fits all conversions on 1991–2014 only, so this appears as part of
the distribution shift in the test years (see the paper, Sec. IV-F).
