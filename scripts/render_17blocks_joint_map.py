"""Render one interactive HTML map for the 17-block joint optimization run."""

import argparse
import html
import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd


SERVICE_NAMES = {
    "kindergarten": "Детский сад",
    "school": "Школа",
    "pharmacy": "Аптека",
    "convenience": "Магазин у дома",
    "playground": "Детская площадка",
    "hairdresser": "Парикмахерская",
    "fuel": "АЗС",
}
SERVICE_COLORS = {
    "kindergarten": "#2ca02c",
    "school": "#e377c2",
    "pharmacy": "#d62728",
    "convenience": "#ff7f0e",
    "playground": "#8c564b",
    "hairdresser": "#17becf",
    "fuel": "#1f77b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/optimization/services/17blocks_joint_optimization"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <run-dir>/17blocks_joint_services_map.html",
    )
    return parser.parse_args()


def format_number(value: float, digits: int = 0) -> str:
    return f"{float(value):,.{digits}f}".replace(",", " ")


def aggregate_services(services: pd.DataFrame) -> pd.DataFrame:
    if services.empty:
        return pd.DataFrame(
            columns=["service_type", "count", "total_capacity", "total_site_area", "total_build_floor_area", "variants"]
        )
    rows = []
    for service_type, group in services.groupby("service_type", sort=True):
        variants = ", ".join(
            f"{int(row['count'])} × {int(row['capacity'])}"
            for _, row in group.sort_values(["capacity", "build_floor_area"]).iterrows()
        )
        rows.append(
            {
                "service_type": service_type,
                "count": int(group["count"].sum()),
                "total_capacity": int((group["capacity"] * group["count"]).sum()),
                "total_site_area": float((group["site_area"] * group["count"]).sum()),
                "total_build_floor_area": float((group["build_floor_area"] * group["count"]).sum()),
                "variants": variants,
            }
        )
    return pd.DataFrame(rows)


def service_rows_html(services: pd.DataFrame) -> str:
    aggregated = aggregate_services(services)
    if aggregated.empty:
        return "<tr><td colspan='6'><i>Сервисы не предложены</i></td></tr>"
    rows = []
    for _, row in aggregated.iterrows():
        service_type = str(row["service_type"])
        name = html.escape(SERVICE_NAMES.get(service_type, service_type))
        color = SERVICE_COLORS.get(service_type, "#555")
        anomaly = service_type == "fuel" and int(row["count"]) > 1
        row_style = "background:#fff0f0" if anomaly else ""
        warning = " ⚠" if anomaly else ""
        rows.append(
            f"<tr style='{row_style}'>"
            f"<td><span style='display:inline-block;width:10px;height:10px;background:{color};"
            f"border-radius:50%;margin-right:5px'></span>{name}{warning}</td>"
            f"<td style='text-align:right'><b>{int(row['count'])}</b></td>"
            f"<td style='text-align:right'>{int(row['total_capacity'])}</td>"
            f"<td style='text-align:right'>{format_number(row['total_site_area'])}</td>"
            f"<td style='text-align:right'>{format_number(row['total_build_floor_area'])}</td>"
            f"<td style='font-size:10px;color:#666'>{html.escape(str(row['variants']))}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_popup(
    block_id: int,
    source: pd.Series,
    result: pd.Series,
    services: pd.DataFrame,
    run_summary: dict,
    neighbours: list[tuple[int, float]],
) -> str:
    proposed_units = int(services["count"].sum()) if not services.empty else 0
    proposed_types = int(services["service_type"].nunique()) if not services.empty else 0
    total_capacity = int((services["capacity"] * services["count"]).sum()) if not services.empty else 0
    site_budget = float(source["site_area"])
    site_used = float(result["service_site_area"])
    bfa_budget = float(result["build_floor_area"])
    bfa_used = float(result["service_build_floor_area"])
    residual_bfa = bfa_budget - bfa_used
    source_population = float(source.get("population", 0))
    calculated_population = float(result.get("population", 0))
    neighbour_text = ", ".join(f"{bid} ({minutes:.1f} мин)" for bid, minutes in neighbours) or "—"
    services_html = service_rows_html(services)
    anomaly_note = ""
    fuel_count = int(services.loc[services["service_type"] == "fuel", "count"].sum()) if not services.empty else 0
    if fuel_count > 1:
        anomaly_note = (
            "<div style='margin:8px 0;padding:6px;background:#fff0f0;border-left:4px solid #d62728;"
            "font-size:11px'><b>Диагностическая аномалия:</b> количество АЗС требует ручной проверки.</div>"
        )

    return f"""
    <div style="font-family:Inter,Arial,sans-serif;font-size:12px;min-width:620px;max-width:760px">
      <h3 style="margin:0 0 3px">Квартал {block_id}</h3>
      <div style="color:#666;font-size:11px;margin-bottom:8px">
        Совместная оптимизация 17 кварталов · best objective {run_summary['best_value']:.6f}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
        <div>
          <div style="font-weight:700;margin-bottom:3px">Исходные атрибуты квартала</div>
          <table style="width:100%;border-collapse:collapse">
            <tr><td>Население</td><td style="text-align:right">{format_number(source_population)}</td></tr>
            <tr><td>Площадь пятна (footprint)</td><td style="text-align:right">{format_number(source.get('footprint_area', 0))} м²</td></tr>
            <tr><td>Общая BFA</td><td style="text-align:right">{format_number(source.get('build_floor_area', 0))} м²</td></tr>
            <tr><td>Жилая площадь</td><td style="text-align:right">{format_number(source.get('living_area', 0))} м²</td></tr>
            <tr><td>Нежилая площадь</td><td style="text-align:right">{format_number(source.get('non_living_area', 0))} м²</td></tr>
          </table>
        </div>
        <div>
          <div style="font-weight:700;margin-bottom:3px">Результат оптимизации</div>
          <table style="width:100%;border-collapse:collapse">
            <tr><td>Предложено объектов</td><td style="text-align:right"><b>{proposed_units}</b></td></tr>
            <tr><td>Типов сервисов</td><td style="text-align:right">{proposed_types}</td></tr>
            <tr><td>Суммарная вместимость</td><td style="text-align:right">{format_number(total_capacity)}</td></tr>
            <tr><td>Расчётное доп. население</td><td style="text-align:right">{format_number(calculated_population)}</td></tr>
            <tr><td>Site area: использовано / бюджет</td><td style="text-align:right">{format_number(site_used)} / {format_number(site_budget)} м²</td></tr>
            <tr><td>BFA сервисов / бюджет</td><td style="text-align:right">{format_number(bfa_used)} / {format_number(bfa_budget)} м²</td></tr>
            <tr><td>Остаток BFA</td><td style="text-align:right">{format_number(residual_bfa)} м²</td></tr>
          </table>
        </div>
      </div>
      {anomaly_note}
      <div style="font-weight:700;margin:8px 0 3px">Предложенные сервисы</div>
      <table style="width:100%;border-collapse:collapse;font-size:11px">
        <thead style="background:#f2f2f2">
          <tr><th style="text-align:left">Сервис</th><th>Кол-во</th><th>Вместимость</th>
          <th>Site, м²</th><th>BFA, м²</th><th style="text-align:left">Юниты: кол-во × вместимость</th></tr>
        </thead>
        <tbody>{services_html}</tbody>
      </table>
      <div style="font-weight:700;margin:8px 0 2px">Ближайшие кварталы по матрице доступности</div>
      <div style="font-size:11px;color:#555">{neighbour_text}</div>
      <div style="margin-top:8px;color:#777;font-size:10px">
        Геометрия и исходные атрибуты: blocks_with_buildings_and_services.gpkg. Предложения: solution_services.csv.
        Размещение сервисов показано на уровне квартала; координаты отдельных объектов оптимизатор не формирует.
      </div>
    </div>
    """


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    output = args.output or run_dir / "17blocks_joint_services_map.html"
    output = (project_root / output).resolve() if not output.is_absolute() else output.resolve()

    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    block_ids = [int(block_id) for block_id in run_summary["block_ids"]]
    source = pd.read_csv(run_dir / "input_blocks.csv").set_index("block_id")
    results = pd.read_csv(run_dir / "per_block_summary.csv").set_index("block_id")
    services = pd.read_csv(run_dir / "solution_services.csv")
    acc_mx = pd.read_pickle(project_root / "data/acc_mx.pickle")

    blocks = gpd.read_file(project_root / "data/blocks_with_buildings_and_services.gpkg")
    blocks = blocks.copy()
    blocks["block_id"] = blocks.index.astype(int)
    missing = sorted(set(block_ids) - set(blocks["block_id"]))
    if missing:
        raise ValueError(f"Target geometries are missing: {missing}")
    blocks_wgs = blocks[["block_id", "geometry"]].to_crs(epsg=4326)
    targets = blocks_wgs[blocks_wgs["block_id"].isin(block_ids)].copy()
    context = blocks_wgs[~blocks_wgs["block_id"].isin(block_ids)].copy()

    bounds = targets.total_bounds
    fmap = folium.Map(
        location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
        tiles="CartoDB Positron",
        control_scale=True,
        zoom_start=12,
    )
    fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]], padding=(25, 25))

    context_layer = folium.FeatureGroup(name="Другие кварталы", show=True)
    folium.GeoJson(
        context.__geo_interface__,
        style_function=lambda _feature: {
            "color": "#777",
            "weight": 0.45,
            "fillColor": "#eeeeee",
            "fillOpacity": 0.25,
        },
        tooltip=folium.GeoJsonTooltip(fields=["block_id"], aliases=["Квартал"]),
        control=False,
    ).add_to(context_layer)
    context_layer.add_to(fmap)

    targets_layer = folium.FeatureGroup(name="Оптимизированные кварталы", show=True)
    render_rows = []
    for block_id in block_ids:
        geometry = targets.loc[targets["block_id"] == block_id, "geometry"].iloc[0]
        point = geometry.representative_point()
        block_services = services[services["block_id"] == block_id].copy()
        nearest = [
            (int(neighbour_id), float(minutes))
            for neighbour_id, minutes in acc_mx.loc[block_id].sort_values().items()
            if int(neighbour_id) != block_id
        ][:5]
        popup_html = build_popup(
            block_id,
            source.loc[block_id],
            results.loc[block_id],
            block_services,
            run_summary,
            nearest,
        )
        unit_count = int(block_services["count"].sum())
        fuel_count = int(block_services.loc[block_services["service_type"] == "fuel", "count"].sum())
        fill_color = "#ff9f9f" if fuel_count > 1 else "#ffd34d"

        feature = {
            "type": "Feature",
            "properties": {"block_id": block_id, "objects": unit_count},
            "geometry": geometry.__geo_interface__,
        }
        folium.GeoJson(
            feature,
            style_function=lambda _feature, color=fill_color: {
                "color": "#c62828",
                "weight": 2.8,
                "fillColor": color,
                "fillOpacity": 0.55,
            },
            highlight_function=lambda _feature: {"weight": 4.0, "fillOpacity": 0.75},
            tooltip=folium.Tooltip(f"Квартал {block_id}: {unit_count} объектов — нажмите для деталей"),
            popup=folium.Popup(popup_html, max_width=820),
        ).add_to(targets_layer)

        marker_html = (
            "<div style='font-family:Inter,Arial;font-size:10px;background:white;border:2px solid #c62828;"
            "border-radius:5px;padding:2px 4px;text-align:center;white-space:nowrap;box-shadow:0 1px 3px #777'>"
            f"<b>{block_id}</b> · {unit_count}"
            "</div>"
        )
        folium.Marker(
            location=[point.y, point.x],
            icon=folium.DivIcon(html=marker_html, icon_size=(0, 0), icon_anchor=(25, 10)),
            tooltip=folium.Tooltip(f"Квартал {block_id}"),
            popup=folium.Popup(popup_html, max_width=820),
        ).add_to(targets_layer)

        aggregated = aggregate_services(block_services)
        render_rows.append(
            {
                "block_id": block_id,
                "source_population": float(source.loc[block_id, "population"]),
                "calculated_additional_population": float(results.loc[block_id, "population"]),
                "total_objects": unit_count,
                "total_capacity": int((block_services["capacity"] * block_services["count"]).sum()),
                "service_types": aggregated[["service_type", "count", "total_capacity"]].to_dict(orient="records"),
                "service_site_area": float(results.loc[block_id, "service_site_area"]),
                "service_build_floor_area": float(results.loc[block_id, "service_build_floor_area"]),
            }
        )

    targets_layer.add_to(fmap)

    total_objects = int(services["count"].sum())
    total_capacity = int((services["capacity"] * services["count"]).sum())
    global_summary = aggregate_services(services)
    legend_rows = "".join(
        f"<tr><td><span style='display:inline-block;width:9px;height:9px;background:{SERVICE_COLORS.get(row.service_type, '#555')};"
        f"border-radius:50%;margin-right:5px'></span>{html.escape(SERVICE_NAMES.get(row.service_type, row.service_type))}</td>"
        f"<td style='text-align:right'>{int(row['count'])}</td></tr>"
        for _, row in global_summary.iterrows()
    )
    panel = f"""
    <div style="position:fixed;left:12px;bottom:28px;z-index:9999;background:white;border:2px solid #555;
                border-radius:6px;padding:9px 11px;font-family:Inter,Arial;font-size:11px;box-shadow:0 1px 6px #777;
                max-height:45vh;overflow:auto">
      <div style="font-size:13px;font-weight:700">Совместная оптимизация 17 кварталов</div>
      <div>Объектов: <b>{total_objects}</b> · Вместимость: <b>{format_number(total_capacity)}</b></div>
      <div>Best objective: {run_summary['best_value']:.6f} · Trials: {run_summary['completed_trials']} + {run_summary['pruned_trials']} pruned</div>
      <table style="width:100%;margin-top:5px">{legend_rows}</table>
      <div style="margin-top:5px;color:#a00000">Красная заливка: аномальное количество АЗС.</div>
      <div style="color:#666">Нажмите на выделенный квартал для полной детализации.</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(panel))
    folium.LayerControl(collapsed=False).add_to(fmap)

    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output))
    render_summary = {
        "map": str(output),
        "run_dir": str(run_dir),
        "block_count": len(block_ids),
        "total_objects": total_objects,
        "total_capacity": total_capacity,
        "blocks": render_rows,
    }
    (run_dir / "map_render_summary.json").write_text(
        json.dumps(render_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output}")
    print(f"Wrote {run_dir / 'map_render_summary.json'}")


if __name__ == "__main__":
    main()
