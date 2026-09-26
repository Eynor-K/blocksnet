"""Render an HTML map of the optimized blocks, one card per block.

Each card carries what a planner needs to judge the proposal for that block:
the land use it has now and the one the run assumed, the population before and
after, the size of the territory, the floor area the optimizer spent on
services, and every service placed there with its count and the capacity of a
single unit.
"""

import argparse
import html
import json
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import folium
import geopandas as gpd
import pandas as pd

from blocksnet.config import service_types_config
from blocksnet.optimization.services.acl.provision_adapter import BFA_COEF, LIVING_DEMAND

BLOCKS_PATH = Path("data/blocks_with_buildings_and_services.gpkg")
LANDUSE_PATH = Path("data/landuse.geojson")
LANDUSE_NAME_COLUMN = "full_name"
FSI_MAX = 3.0
LAND_USE_TITLES = {"RESIDENTIAL": "Жилая застройка"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory holding the run's artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/optimization/services/17blocks_project_cards"))
    return parser.parse_args()


def number(value: float, digits: int = 0) -> str:
    return f"{float(value):,.{digits}f}".replace(",", " ")


def floor_area_budget(site_area: float) -> float:
    """Floor area the optimizer may hand out in a block."""
    return site_area * FSI_MAX * BFA_COEF


def free_floor_area(site_area: float, used_bfa: float) -> float:
    """What is left of that budget once services have taken ``used_bfa``."""
    return max(floor_area_budget(site_area) - used_bfa, 0.0)


def housing_floor_area(site_area: float, used_bfa: float) -> float:
    """Floor area the housing gets.

    Services may take up to ``BFA_COEF`` of what can be built, and the housing
    takes the rest, so every square metre left in the budget carries
    ``1 / BFA_COEF - 1`` square metres of flats with it.
    """
    return free_floor_area(site_area, used_bfa) * (1 / BFA_COEF - 1)


def block_population(site_area: float, used_bfa: float) -> int:
    """Residents a block houses once services have taken ``used_bfa``."""
    return math.floor(housing_floor_area(site_area, used_bfa) / LIVING_DEMAND)


def existing_land_use(blocks: gpd.GeoDataFrame, block_ids: list) -> dict:
    """Dominant functional zone under each block, by area of overlap."""
    if not LANDUSE_PATH.exists():
        return {}
    zones = gpd.read_file(LANDUSE_PATH).to_crs(blocks.crs)
    if LANDUSE_NAME_COLUMN not in zones.columns:
        return {}
    zones = zones[[LANDUSE_NAME_COLUMN, "geometry"]].dropna(subset=[LANDUSE_NAME_COLUMN])
    targets = blocks.loc[block_ids, ["geometry"]].reset_index(names="block_id")
    overlap = gpd.overlay(targets, zones, how="intersection", keep_geom_type=False)
    if overlap.empty:
        return {}
    overlap["area"] = overlap.geometry.area
    ranked = overlap.groupby(["block_id", LANDUSE_NAME_COLUMN])["area"].sum().reset_index()
    ranked = ranked.sort_values("area", ascending=False).drop_duplicates("block_id")
    return dict(zip(ranked["block_id"], ranked[LANDUSE_NAME_COLUMN]))


def service_rows(services: pd.DataFrame) -> list:
    """Placed services of one block, grouped by type and unit size."""
    rows = []
    for service_type, group in services.groupby("service_type", sort=True):
        variants = []
        for _, unit in group.sort_values("capacity", ascending=False).iterrows():
            variants.append(
                {
                    "count": int(unit["count"]),
                    "capacity": int(unit["capacity"]),
                    "site_area": float(unit["site_area"]),
                    "build_floor_area": float(unit["build_floor_area"]),
                }
            )
        rows.append(
            {
                "service_type": service_type,
                "name_ru": service_types_config[service_type]["name_ru"],
                "count": int(group["count"].sum()),
                "capacity": int((group["capacity"] * group["count"]).sum()),
                "variants": variants,
            }
        )
    return sorted(rows, key=lambda row: -row["count"])


def services_text(services: list) -> str:
    """The card's service table flattened into one cell of the CSV."""
    parts = []
    for service in services:
        units = " + ".join(f"{v['count']}x{v['capacity']}" for v in service["variants"])
        parts.append(f"{service['name_ru']} ({service['count']} шт., {units})")
    return "; ".join(parts)


def card_html(block_id: int, facts: dict) -> str:
    def section(title):
        return (
            "<tr><td colspan='2' style='padding:7px 0 2px;font-weight:600;"
            f"border-bottom:1px solid #eee'>{html.escape(title)}</td></tr>"
        )

    def row(label, value):
        return (
            f"<tr><td style='padding:2px 10px 2px 0;color:#555;white-space:nowrap'>{html.escape(label)}</td>"
            f"<td style='padding:2px 0;text-align:right;font-weight:600'>{value}</td></tr>"
        )

    # Against the population living there now, not against the hypothetical
    # maximum: the blocks could hold more if nothing but housing were built,
    # and comparing with that made a fourfold increase read as a loss.
    delta = facts["population_after"] - facts["population_now"]
    delta_text = f"<span style='color:{'#c0392b' if delta < 0 else '#27ae60'}'>{delta:+,}</span>".replace(",", " ")

    head = (
        f"<div style='font:13px/1.45 system-ui,sans-serif;max-width:430px'>"
        f"<div style='font-size:15px;font-weight:700;margin-bottom:6px'>Квартал {block_id}</div>"
        "<table style='border-collapse:collapse;width:100%'>"
        + row("Землепользование сейчас", html.escape(facts["land_use_existing"]))
        + row("Землепользование проектное", html.escape(facts["land_use_project"]))
        + section("Участок, м²")
        + row("всего", number(facts["site_area"]))
        + row("под жильё", number(facts["site_area_housing"]))
        + row("под сервисы", number(facts["site_area_services"]))
        + section("Поэтажная площадь, м²")
        + row("жильё", number(facts["bfa_housing"]))
        + row("встроенные сервисы", number(facts["bfa_built_in_services"]))
        + row("отдельно стоящие сервисы", number(facts["bfa_standalone_services"]))
        + section("Население")
        + row("сейчас", number(facts["population_now"]))
        + row("по проекту", f"{number(facts['population_after'])} ({delta_text})")
        + row("потенциал без сервисов", number(facts["population_potential"]))
        + row("объектов размещено", number(facts["units"]))
        + "</table>"
    )

    if not facts["services"]:
        return head + "<div style='margin-top:8px;color:#777'>Сервисы не размещены</div></div>"

    body = [
        "<div style='margin-top:10px;font-weight:600'>Размещённые сервисы</div>",
        "<table style='border-collapse:collapse;width:100%;margin-top:4px;font-size:12px'>",
        "<tr style='color:#555;text-align:left'>"
        "<th style='padding:3px 8px 3px 0'>Сервис</th>"
        "<th style='padding:3px 8px;text-align:right'>Шт.</th>"
        "<th style='padding:3px 0;text-align:right'>Вместимость юнита</th></tr>",
    ]
    for service in facts["services"]:
        units = "<br>".join(
            f"{variant['count']} × {number(variant['capacity'])}" for variant in service["variants"]
        )
        body.append(
            "<tr style='border-top:1px solid #eee'>"
            f"<td style='padding:3px 8px 3px 0'>{html.escape(service['name_ru'])}</td>"
            f"<td style='padding:3px 8px;text-align:right'>{service['count']}</td>"
            f"<td style='padding:3px 0;text-align:right'>{units}</td></tr>"
        )
    body.append("</table></div>")
    return head + "".join(body)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    solution = pd.read_csv(run_dir / "solution_services.csv")
    block_ids = list(summary["block_ids"])

    blocks = gpd.read_file(BLOCKS_PATH)
    blocks["site_area"] = blocks.geometry.area
    existing = existing_land_use(blocks, block_ids)
    project_land_use = LAND_USE_TITLES.get(summary["land_use"], summary["land_use"])

    solution["total_bfa"] = solution["build_floor_area"] * solution["count"]
    solution["total_site"] = solution["site_area"] * solution["count"]
    used_bfa = solution.groupby("block_id")["total_bfa"].sum()
    used_site = solution.groupby("block_id")["total_site"].sum()
    # A service either stands on a plot of its own or is built into the housing.
    built_in = solution[solution["site_area"] == 0]
    standalone = solution[solution["site_area"] > 0]
    built_in_bfa = built_in.groupby("block_id")["total_bfa"].sum()
    standalone_bfa = standalone.groupby("block_id")["total_bfa"].sum()

    cards = {}
    for block_id in block_ids:
        site_area = float(blocks.loc[block_id, "site_area"])
        block_services = solution[solution["block_id"] == block_id]
        block_bfa = float(used_bfa.get(block_id, 0.0))
        service_plots = float(used_site.get(block_id, 0.0))
        cards[block_id] = {
            "land_use_existing": existing.get(block_id, "нет данных"),
            "land_use_project": project_land_use,
            "site_area": site_area,
            "site_area_services": service_plots,
            "site_area_housing": max(site_area - service_plots, 0.0),
            "bfa_housing": housing_floor_area(site_area, block_bfa),
            "bfa_built_in_services": float(built_in_bfa.get(block_id, 0.0)),
            "bfa_standalone_services": float(standalone_bfa.get(block_id, 0.0)),
            "population_now": int(blocks.loc[block_id, "population"]),
            "population_potential": block_population(site_area, 0.0),
            "population_after": block_population(site_area, block_bfa),
            "units": int(block_services["count"].sum()) if not block_services.empty else 0,
            "services": service_rows(block_services) if not block_services.empty else [],
        }

    targets = blocks.loc[block_ids].to_crs(4326)
    centre = [targets.geometry.centroid.y.mean(), targets.geometry.centroid.x.mean()]
    fmap = folium.Map(location=centre, zoom_start=12, tiles="cartodbpositron")

    for block_id, geometry in zip(block_ids, targets.geometry):
        facts = cards[block_id]
        served = facts["units"] > 0
        folium.GeoJson(
            geometry,
            style_function=lambda _f, served=served: {
                "fillColor": "#2ca02c" if served else "#999999",
                "color": "#1a1a1a",
                "weight": 1.2,
                "fillOpacity": 0.45,
            },
            highlight_function=lambda _f: {"weight": 3, "fillOpacity": 0.65},
            tooltip=f"Квартал {block_id}: {facts['units']} объектов",
            popup=folium.Popup(card_html(block_id, facts), max_width=470),
        ).add_to(fmap)

    total_now = sum(card["population_now"] for card in cards.values())
    total_after = sum(card["population_after"] for card in cards.values())
    total_potential = sum(card["population_potential"] for card in cards.values())
    legend = (
        "<div style='position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;"
        "padding:10px 14px;border:1px solid #ccc;border-radius:6px;font:12px/1.5 system-ui,sans-serif'>"
        f"<b>Кварталов:</b> {len(block_ids)}<br>"
        f"<b>Объектов:</b> {int(solution['count'].sum())}<br>"
        f"<b>Население:</b> {number(total_now)} сейчас → {number(total_after)} по проекту<br>"
        f"<span style='color:#666'>потенциал без сервисов: {number(total_potential)}</span><br>"
        "<span style='color:#666'>Нажмите на квартал, чтобы открыть карточку</span></div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend))

    output = output_dir / "project_cards_map.html"
    fmap.save(str(output))
    pd.DataFrame(
        [
            {
                "block_id": block_id,
                "land_use_existing": card["land_use_existing"],
                "land_use_project": card["land_use_project"],
                "site_area": round(card["site_area"]),
                "site_area_housing": round(card["site_area_housing"]),
                "site_area_services": round(card["site_area_services"]),
                "bfa_housing": round(card["bfa_housing"]),
                "bfa_built_in_services": round(card["bfa_built_in_services"]),
                "bfa_standalone_services": round(card["bfa_standalone_services"]),
                "population_now": card["population_now"],
                "population_after": card["population_after"],
                "population_potential": card["population_potential"],
                "units": card["units"],
                "services": services_text(card["services"]),
            }
            for block_id, card in cards.items()
        ]
    ).to_csv(output_dir / "project_cards.csv", index=False)
    pd.DataFrame(
        [
            {
                "block_id": block_id,
                "service_type": service["service_type"],
                "name_ru": service["name_ru"],
                "count": variant["count"],
                "capacity_per_unit": variant["capacity"],
                "capacity_total": variant["count"] * variant["capacity"],
                "site_area_per_unit": round(variant["site_area"]),
                "build_floor_area_per_unit": round(variant["build_floor_area"]),
                "built_in": variant["site_area"] == 0,
            }
            for block_id, card in cards.items()
            for service in card["services"]
            for variant in service["variants"]
        ]
    ).to_csv(output_dir / "project_services.csv", index=False)
    print(f"Карта: {output}")
    print(f"Таблица: {output_dir / 'project_cards.csv'}")
    print(f"Сервисы: {output_dir / 'project_services.csv'}")


if __name__ == "__main__":
    main()
