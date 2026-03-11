import math
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class LinkBudget:
    frequency_hz: float
    bandwidth_hz: float
    tx_power_dbw: float
    tx_antenna_gain_dbi: float
    rx_antenna_gain_dbi: float
    distance_km: float
    system_noise_temp_k: float
    atmospheric_loss_db: float = 2.0
    misc_losses_db: float = 1.0

    @property
    def wavelength_m(self) -> float:
        return 299_792_458 / self.frequency_hz

    @property
    def free_space_path_loss_db(self) -> float:
        d_m = self.distance_km * 1000
        return 20 * math.log10(4 * math.pi * d_m / self.wavelength_m)

    @property
    def eirp_dbw(self) -> float:
        return self.tx_power_dbw + self.tx_antenna_gain_dbi

    @property
    def received_power_dbw(self) -> float:
        return (self.eirp_dbw
                + self.rx_antenna_gain_dbi
                - self.free_space_path_loss_db
                - self.atmospheric_loss_db
                - self.misc_losses_db)

    @property
    def noise_power_dbw(self) -> float:
        k = 1.380649e-23
        n_w = k * self.system_noise_temp_k * self.bandwidth_hz
        return 10 * math.log10(n_w)

    @property
    def snr_db(self) -> float:
        return self.received_power_dbw - self.noise_power_dbw

    @property
    def snr_linear(self) -> float:
        return 10 ** (self.snr_db / 10)

    @property
    def shannon_capacity_bps(self) -> float:
        return self.bandwidth_hz * math.log2(1 + self.snr_linear)

    @property
    def spectral_efficiency(self) -> float:
        return self.shannon_capacity_bps / self.bandwidth_hz

    @property
    def eb_no_db(self) -> float:
        eb_no_lin = self.snr_linear / self.spectral_efficiency
        return 10 * math.log10(eb_no_lin)

    @property
    def shannon_limit_eb_no_db(self) -> float:
        return 10 * math.log10(math.log(2))  # -1.59 dB

    def report(self) -> dict:
        return {
            "frequency_ghz": self.frequency_hz / 1e9,
            "bandwidth_mhz": self.bandwidth_hz / 1e6,
            "distance_km": self.distance_km,
            "eirp_dbw": round(self.eirp_dbw, 2),
            "fspl_db": round(self.free_space_path_loss_db, 2),
            "rx_power_dbw": round(self.received_power_dbw, 2),
            "noise_power_dbw": round(self.noise_power_dbw, 2),
            "snr_db": round(self.snr_db, 2),
            "shannon_capacity_mbps": round(self.shannon_capacity_bps / 1e6, 4),
            "spectral_efficiency_bps_hz": round(self.spectral_efficiency, 4),
            "eb_no_db": round(self.eb_no_db, 2),
            "shannon_limit_eb_no_db": round(self.shannon_limit_eb_no_db, 2),
            "margin_above_shannon_db": round(self.eb_no_db - self.shannon_limit_eb_no_db, 2),
        }


def shannon_capacity_sweep(link: LinkBudget,
                           snr_range_db: tuple = (-5, 30),
                           step: float = 1.0) -> list[dict]:
    results = []
    snr = snr_range_db[0]
    while snr <= snr_range_db[1]:
        snr_lin = 10 ** (snr / 10)
        cap = link.bandwidth_hz * math.log2(1 + snr_lin)
        results.append({
            "snr_db": snr,
            "capacity_mbps": round(cap / 1e6, 4),
            "spectral_eff": round(math.log2(1 + snr_lin), 4),
        })
        snr += step
    return results

STAC_CATALOGS = {
    "earth_search": {
        "name": "Earth Search (Sentinel/Landsat via AWS)",
        "url": "https://earth-search.aws.element84.com/v1",
    },
}


@dataclass
class ImagerySearchParams:
    bbox: list[float]                       # [west, south, east, north]
    datetime_range: str                     # ISO 8601 interval e.g. "2024-01-01/2024-06-01"
    collections: Optional[list[str]] = None # e.g. ["sentinel-2-l2a"]
    max_cloud_cover: Optional[float] = None # 0-100
    limit: int = 5


def search_stac(params: ImagerySearchParams,
                catalog_key: str = "earth_search") -> dict:
    catalog = STAC_CATALOGS[catalog_key]
    search_url = catalog["url"] + "/search"

    body: dict = {
        "bbox": params.bbox,
        "datetime": params.datetime_range,
        "limit": params.limit,
    }
    if params.collections:
        body["collections"] = params.collections
    if params.max_cloud_cover is not None:
        body["query"] = {
            "eo:cloud_cover": {"lte": params.max_cloud_cover}
        }

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        search_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:500]}
    except Exception as e:
        return {"error": str(e)}


def summarize_stac_results(geojson: dict) -> list[dict]:
    if "error" in geojson:
        return [geojson]
    features = geojson.get("features", [])
    summaries = []
    for f in features:
        props = f.get("properties", {})
        summaries.append({
            "id": f.get("id"),
            "datetime": props.get("datetime"),
            "collection": props.get("collection"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "platform": props.get("platform"),
            "gsd_m": props.get("gsd"),
            "bbox": f.get("bbox"),
            "thumbnail": (f.get("assets", {})
                           .get("thumbnail", {})
                           .get("href")),
        })
    return summaries

def estimate_image_downlink_time(image_size_mb: float, link: LinkBudget) -> dict:
    cap = link.shannon_capacity_bps
    bits = image_size_mb * 8 * 1e6
    seconds = bits / cap
    return {
        "image_size_mb": image_size_mb,
        "channel_capacity_mbps": round(cap / 1e6, 4),
        "min_download_seconds": round(seconds, 2),
        "min_download_minutes": round(seconds / 60, 3),
    }

def print_section(title: str):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def demo_shannon():
    print_section("Shannon Limit — LEO Satellite Downlink Example")

    link = LinkBudget(
        frequency_hz=8.2e9,        # X-band (8.2 GHz)
        bandwidth_hz=150e6,        # 150 MHz channel
        tx_power_dbw=10,           # 10 W transmitter
        tx_antenna_gain_dbi=6,
        rx_antenna_gain_dbi=34,    # 3 m ground dish
        distance_km=600,           # LEO altitude
        system_noise_temp_k=135,
    )

    rpt = link.report()
    for k, v in rpt.items():
        label = k.replace("_", " ").title()
        print(f"  {label:.<40} {v}")

    print_section("Capacity vs SNR Sweep")
    sweep = shannon_capacity_sweep(link, snr_range_db=(0, 25), step=5)
    print(f"  {'SNR (dB)':>10}  {'Capacity (Mbps)':>16}  {'Spec Eff (b/s/Hz)':>18}")
    for row in sweep:
        print(f"  {row['snr_db']:>10.1f}  {row['capacity_mbps']:>16.4f}  {row['spectral_eff']:>18.4f}")


SAMPLE_STAC_RESPONSE = {
    "type": "FeatureCollection",
    "features": [
        {
            "id": "S2B_10SEG_20250215_0_L2A",
            "bbox": [-122.5, 37.5, -122.0, 37.9],
            "properties": {
                "datetime": "2025-02-15T18:42:11Z",
                "collection": "sentinel-2-l2a",
                "eo:cloud_cover": 8.3,
                "platform": "sentinel-2b",
                "gsd": 10,
            },
            "assets": {
                "thumbnail": {"href": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2B_10SEG_20250215_0_L2A/thumbnail"}
            },
        },
        {
            "id": "S2A_10SEG_20250210_0_L2A",
            "bbox": [-122.5, 37.5, -122.0, 37.9],
            "properties": {
                "datetime": "2025-02-10T18:42:08Z",
                "collection": "sentinel-2-l2a",
                "eo:cloud_cover": 12.1,
                "platform": "sentinel-2a",
                "gsd": 10,
            },
            "assets": {
                "thumbnail": {"href": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_10SEG_20250210_0_L2A/thumbnail"}
            },
        },
        {
            "id": "S2B_10SEG_20250125_0_L2A",
            "bbox": [-122.5, 37.5, -122.0, 37.9],
            "properties": {
                "datetime": "2025-01-25T18:41:55Z",
                "collection": "sentinel-2-l2a",
                "eo:cloud_cover": 4.7,
                "platform": "sentinel-2b",
                "gsd": 10,
            },
            "assets": {
                "thumbnail": {"href": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2B_10SEG_20250125_0_L2A/thumbnail"}
            },
        },
    ],
}


def demo_imagery_search():
    print_section("Satellite Imagery Search — San Francisco Bay Area")

    params = ImagerySearchParams(
        bbox=[-122.5, 37.5, -122.0, 37.9],
        datetime_range="2025-01-01/2025-03-01",
        collections=["sentinel-2-l2a"],
        max_cloud_cover=20,
        limit=5,
    )

    print("  Querying Earth Search STAC API ...")
    raw = search_stac(params)
    results = summarize_stac_results(raw)

    if not results or "error" in results[0]:
        print("  Live API unavailable — using cached sample results.")
        raw = SAMPLE_STAC_RESPONSE
        results = summarize_stac_results(raw)

    if not results:
        print("  No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r.get('id', 'N/A')}")
        print(f"      Date ........... {r.get('datetime', '?')}")
        print(f"      Collection ..... {r.get('collection', '?')}")
        print(f"      Cloud Cover .... {r.get('cloud_cover', '?')}%")
        print(f"      Platform ....... {r.get('platform', '?')}")
        print(f"      GSD ............ {r.get('gsd_m', '?')} m")
        print(f"      Thumbnail ...... {r.get('thumbnail', 'N/A')}")


def demo_downlink():
    print_section("Image Downlink Time Estimate")

    link = LinkBudget(
        frequency_hz=8.2e9,
        bandwidth_hz=150e6,
        tx_power_dbw=10,
        tx_antenna_gain_dbi=6,
        rx_antenna_gain_dbi=34,
        distance_km=600,
        system_noise_temp_k=135,
    )

    for size_mb in [100, 500, 1000, 5000]:
        dl = estimate_image_downlink_time(size_mb, link)
        print(f"  {size_mb:>5} MB  ->  {dl['min_download_seconds']:>8.2f} s  "
              f"({dl['min_download_minutes']:.3f} min)  "
              f"@ {dl['channel_capacity_mbps']:.2f} Mbps Shannon limit")


if __name__ == "__main__":
    print("\n  SATELLITE IMAGERY & SHANNON LIMIT ANALYZER")
    print("  " + "~" * 46)

    demo_shannon()
    demo_downlink()

    print()
    try:
        demo_imagery_search()
    except Exception as e:
        print(f"\n  [!] Imagery search unavailable: {e}")
        print("      (Requires network access to earth-search.aws.element84.com)")

    print()
