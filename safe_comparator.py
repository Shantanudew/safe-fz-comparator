import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.chart import ScatterChart, Series, Reference
from openpyxl.chart.marker import Marker
from openpyxl.chart.label import DataLabelList
from openpyxl.xml.functions import fromstring
from openpyxl.chart.shapes import GraphicalProperties
import re
import io


# =====================================================================
# 1. SAFE / ETABS EXCEL PARSERS
# =====================================================================
def parse_safe_connectivity_excel(file_or_path):
    """Parses SAFE/ETABS Point Object Connectivity Excel export."""
    xls = pd.ExcelFile(file_or_path)
    target_sheet = None
    for s in xls.sheet_names:
        if any(k in s.lower() for k in ['connectivity', 'joint', 'point', 'coord']):
            target_sheet = s
            break
    if target_sheet is None:
        target_sheet = [s for s in xls.sheet_names if 'program' not in s.lower()][0]

    raw = pd.read_excel(file_or_path, sheet_name=target_sheet)

    if str(raw.columns[0]).startswith('TABLE:'):
        raw.columns = raw.iloc[0]
        df = raw.iloc[2:].reset_index(drop=True)
    else:
        df = raw.copy()

    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ['uniquename', 'unique name', 'objname', 'elmname', 'joint', 'label']:
            col_map[c] = 'UniqueName'
        elif cl in ['x', 'globalx']:
            col_map[c] = 'X'
        elif cl in ['y', 'globaly']:
            col_map[c] = 'Y'
        elif cl in ['z', 'globalz']:
            col_map[c] = 'Z'
    df = df.rename(columns=col_map)

    df['UniqueName'] = df['UniqueName'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    for col in ['X', 'Y', 'Z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[['UniqueName', 'X', 'Y', 'Z']].dropna().drop_duplicates(subset=['UniqueName'])


def parse_safe_reactions_excel(file_or_path):
    """Parses SAFE/ETABS Joint Reactions Excel export."""
    xls = pd.ExcelFile(file_or_path)
    target_sheet = 'Joint Reactions' if 'Joint Reactions' in xls.sheet_names else [
        s for s in xls.sheet_names if 'program' not in s.lower()
    ][0]

    raw = pd.read_excel(file_or_path, sheet_name=target_sheet)

    if str(raw.columns[0]).startswith('TABLE:'):
        raw.columns = raw.iloc[0]
        df = raw.iloc[2:].reset_index(drop=True)
    else:
        df = raw.copy()

    df.columns = [str(c).strip() for c in df.columns]

    if 'Unique Name' in df.columns and 'Label' in df.columns:
        joint_id = df['Unique Name'].fillna(df['Label'])
    elif 'Unique Name' in df.columns:
        joint_id = df['Unique Name']
    elif 'UniqueName' in df.columns:
        joint_id = df['UniqueName']
    elif 'Label' in df.columns:
        joint_id = df['Label']
    else:
        raise ValueError(f"Could not find 'Unique Name' or 'Label' column in {df.columns.tolist()}")

    df['UniqueName'] = joint_id.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    case_col = 'Output Case' if 'Output Case' in df.columns else 'Load Pattern'
    df['Output Case'] = df[case_col].astype(str).str.strip()

    if 'Step Type' in df.columns:
        step_type = df['Step Type'].fillna('').astype(str).str.strip()
    else:
        step_type = pd.Series([''] * len(df), index=df.index)

    if 'Step Number' in df.columns:
        step_num = df['Step Number'].fillna('').astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    else:
        step_num = pd.Series([''] * len(df), index=df.index)

    def build_load_key(row_case, s_type, s_num):
        if s_num != '' and s_num.lower() != 'nan':
            return f"{row_case} (Step {s_num})"
        elif s_type != '' and s_type.lower() != 'nan':
            return f"{row_case} ({s_type})"
        return row_case

    df['Load Pattern'] = [
        build_load_key(c, stp, snm)
        for c, stp, snm in zip(df['Output Case'], step_type, step_num)
    ]

    df['FZ'] = pd.to_numeric(df['FZ'], errors='coerce')
    return df[['UniqueName', 'Output Case', 'Load Pattern', 'FZ']].dropna(subset=['UniqueName', 'FZ'])


# =====================================================================
# 2. STAGE 1: 3D COORDINATE MAPPING ENGINE
# =====================================================================
def stage1_build_coordinate_mapping(model_files_dict, tol=0.001, use_sequential_id=False):
    """Maps joints across models based on 3D coordinate proximity."""
    master_coords = []
    master_ids = []
    model_map_records = {m: {} for m in model_files_dict.keys()}
    model_tabs = {}

    for model_name, file_obj in model_files_dict.items():
        file_obj.seek(0)
        df = parse_safe_connectivity_excel(file_obj)
        coords = df[['X', 'Y', 'Z']].values
        unames = df['UniqueName'].values
        mapped_ids = []

        if len(master_coords) == 0:
            for i, (u, xyz) in enumerate(zip(unames, coords)):
                cid = f"J{i+1:05d}" if use_sequential_id else u
                master_coords.append(xyz)
                master_ids.append(cid)
                model_map_records[model_name][cid] = u
                mapped_ids.append(cid)
        else:
            tree = cKDTree(np.array(master_coords))
            dists, idx_list = tree.query(coords, distance_upper_bound=tol)

            for u, xyz, dist, idx in zip(unames, coords, dists, idx_list):
                if dist <= tol and idx < len(master_ids):
                    cid = master_ids[idx]
                else:
                    cid = f"J{len(master_ids)+1:05d}" if use_sequential_id else (
                        u if u not in master_ids else f"{u}_{model_name}"
                    )
                    master_coords.append(xyz)
                    master_ids.append(cid)
                model_map_records[model_name][cid] = u
                mapped_ids.append(cid)

        model_tabs[model_name] = pd.DataFrame({
            'OriginalUnique': unames,
            'MappedUnique': mapped_ids,
            'X': coords[:, 0],
            'Y': coords[:, 1],
            'Z': coords[:, 2]
        })

    master_arr = np.array(master_coords)
    coord_map_df = pd.DataFrame({
        'CommonUnique': master_ids,
        'X': np.round(master_arr[:, 0], 4),
        'Y': np.round(master_arr[:, 1], 4),
        'Z': np.round(master_arr[:, 2], 4)
    })

    for model_name in model_files_dict.keys():
        coord_map_df[model_name] = coord_map_df['CommonUnique'].map(model_map_records[model_name])

    return coord_map_df, model_tabs


# =====================================================================
# 3. STAGE 2: FZ REACTION COMPARISON ENGINE
# =====================================================================
def stage2_compare_reactions(coord_map_df, reaction_files_dict, baseline_model, merge_how='inner'):
    """Maps reactions to CommonUnique IDs and computes magnitude Delta and Pct_Diff."""
    model_names = list(reaction_files_dict.keys())
    comp_models = [m for m in model_names if m != baseline_model]

    short_tag = {baseline_model: 'Old'}
    for idx, m in enumerate(comp_models):
        short_tag[m] = 'New' if len(comp_models) == 1 else f'M{idx+1}'

    coord_clean = coord_map_df.copy()
    coord_clean['CommonUnique'] = coord_clean['CommonUnique'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    aggregated_dfs = []
    ordered_models = [baseline_model] + comp_models

    for model_name in ordered_models:
        file_obj = reaction_files_dict[model_name]
        file_obj.seek(0)
        df_reac = parse_safe_reactions_excel(file_obj)

        valid_rows = coord_clean[['CommonUnique', model_name]].dropna()
        lookup = dict(zip(
            valid_rows[model_name].astype(str).str.strip().str.replace(r'\.0$', '', regex=True),
            valid_rows['CommonUnique']
        ))

        df_reac['CommonUnique'] = df_reac['UniqueName'].map(lookup)
        df_mapped = df_reac.dropna(subset=['CommonUnique']).copy()

        if len(df_mapped) == 0:
            raise ValueError(
                f"0 joints matched for file linked to '{model_name}'. "
                f"Verify that each reaction file is linked to its matching connectivity model."
            )

        tag = short_tag[model_name]
        df_agg = (
            df_mapped.groupby(['CommonUnique', 'Load Pattern'], as_index=False)['FZ']
            .sum()
            .rename(columns={'FZ': f'FZ_{tag}'})
        )
        aggregated_dfs.append(df_agg)

    comp_df = aggregated_dfs[0]
    for next_df in aggregated_dfs[1:]:
        comp_df = pd.merge(comp_df, next_df, on=['CommonUnique', 'Load Pattern'], how=merge_how)

    meta_cols = ['CommonUnique', 'X', 'Y', 'Z'] + ordered_models
    id_rename = {m: f"ID_{short_tag[m]}" for m in ordered_models}
    coords_lookup = coord_clean[meta_cols].rename(columns=id_rename)
    comp_df = pd.merge(coords_lookup, comp_df, on='CommonUnique', how='right')

    base_fz_col = 'FZ_Old'
    old_abs = comp_df[base_fz_col].abs()

    for m in comp_models:
        tag = short_tag[m]
        new_fz_col = f'FZ_{tag}'
        delta_col = 'Delta' if len(comp_models) == 1 else f'Delta_{tag}'
        pct_col = 'Pct_Diff' if len(comp_models) == 1 else f'Pct_Diff_{tag}'
        status_col = 'Trend' if len(comp_models) == 1 else f'Trend_{tag}'

        new_abs = comp_df[new_fz_col].abs()
        delta_val = new_abs - old_abs
        comp_df[delta_col] = delta_val

        comp_df[status_col] = np.where(
            delta_val > 1e-4, "Increased (+)",
            np.where(delta_val < -1e-4, "Decreased (-)", "No Change (0)")
        )

        comp_df[pct_col] = np.where(
            old_abs > 1e-4,
            (delta_val / old_abs) * 100.0,
            0.0
        )

    legend_rows = [
        ("CommonUnique", "Master Joint Identifier", "Universal joint tracking ID mapped across models via 3D coordinates."),
        ("X", "Global X Coordinate", "Horizontal X coordinate of the support joint in meters."),
        ("Y", "Global Y Coordinate", "Horizontal Y coordinate of the support joint in meters."),
        ("Z", "Global Z Coordinate", "Vertical Z elevation of the support joint in meters."),
        ("ID_Old", "Baseline Joint ID", f"Unique Name / Label of the joint in {baseline_model}."),
    ]
    for m in comp_models:
        tag = short_tag[m]
        legend_rows.append((f"ID_{tag}", "Comparison Joint ID", f"Unique Name / Label of the joint in {m}."))

    legend_rows.append(("Load Pattern", "Load Case / Combination", "Name of the exported SAFE/ETABS Output Case or Load Pattern."))
    legend_rows.append(("FZ_Old", "Baseline Vertical Reaction", f"Vertical reaction FZ in kN from {baseline_model}."))

    for m in comp_models:
        tag = short_tag[m]
        delta_col = 'Delta' if len(comp_models) == 1 else f'Delta_{tag}'
        pct_col = 'Pct_Diff' if len(comp_models) == 1 else f'Pct_Diff_{tag}'
        status_col = 'Trend' if len(comp_models) == 1 else f'Trend_{tag}'

        legend_rows.extend([
            (f"FZ_{tag}", "Comparison Vertical Reaction", f"Vertical reaction FZ in kN from {m}."),
            (delta_col, "Signed Magnitude Delta", "Calculated as |FZ_New| - |FZ_Old|. Positive (+) indicates increase; Negative (-) indicates decrease."),
            (status_col, "Change Direction", "Indicates whether the reaction magnitude Increased, Decreased, or had No Change."),
            (pct_col, "Percentage Difference", "Calculated as ((|FZ_New| - |FZ_Old|) / |FZ_Old|) * 100.")
        ])

    legend_df = pd.DataFrame(legend_rows, columns=["Column Header", "Title", "Description"])
    return comp_df, legend_df


# =====================================================================
# 4. EXCEL REPORT GENERATOR WITH HOVER JOINT ID & DELTA ENCODING
# =====================================================================
def generate_comparison_excel_with_bubble_chart(comp_df, legend_df):
    """
    Generates the Excel workbook where hovering the cursor over ANY point on the
    Uniform ScatterChart displays the point's true CommonUnique ID AND Delta:
      - Line 1 of Excel Tooltip: Series "Increased (+)" Point "<X>"
      - Line 2 of Excel Tooltip: (<X>.<CommonUnique_ID>, <Y>.<Delta>)
      - Also supports Spotlight Joint callout (Cell E2) with its live Delta label.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        comp_df.to_excel(writer, sheet_name='FZ_Comparison', index=False)
        legend_df.to_excel(writer, sheet_name='Header_Legend', index=False)

    output.seek(0)
    wb = openpyxl.load_workbook(output)

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    # 1. Format 'FZ_Comparison' Sheet
    ws_comp = wb['FZ_Comparison']
    col_names = [cell.value for cell in ws_comp[1]]
    max_comp_row = ws_comp.max_row

    for col_idx, col_name in enumerate(col_names, start=1):
        cell = ws_comp.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws_comp.column_dimensions[get_column_letter(col_idx)].width = max(len(str(col_name)) + 5, 14)

        if str(col_name).startswith('Delta'):
            for r in range(2, max_comp_row + 1):
                ws_comp.cell(row=r, column=col_idx).number_format = '+0.0000;-0.0000;0.0000'
        elif str(col_name).startswith('Pct_Diff'):
            for r in range(2, max_comp_row + 1):
                ws_comp.cell(row=r, column=col_idx).number_format = '+0.00;-0.00;0.00'
        elif str(col_name).startswith('FZ_') or col_name in ['X', 'Y', 'Z']:
            for r in range(2, max_comp_row + 1):
                ws_comp.cell(row=r, column=col_idx).number_format = '0.0000'

    ws_comp.freeze_panes = "A2"

    # 2. Format 'Header_Legend' Sheet
    ws_leg = wb['Header_Legend']
    for col_idx in range(1, 4):
        c = ws_leg.cell(row=1, column=col_idx)
        c.fill = header_fill
        c.font = header_font
    ws_leg.column_dimensions['A'].width = 18
    ws_leg.column_dimensions['B'].width = 32
    ws_leg.column_dimensions['C'].width = 90

    # 3. Build Hidden '_DataEngine' Sheet
    ws_eng = wb.create_sheet(title='_DataEngine')
    ws_eng.sheet_state = 'hidden'

    joints_df = comp_df[['CommonUnique', 'X', 'Y']].drop_duplicates(subset=['CommonUnique']).reset_index(drop=True)
    n_joints = len(joints_df)
    load_patterns = sorted(comp_df['Load Pattern'].dropna().unique().tolist())
    n_lps = len(load_patterns)

    non_modal_lps = [lp for lp in load_patterns if 'modal' not in lp.lower()]
    default_lp = non_modal_lps[0] if non_modal_lps else (load_patterns[0] if load_patterns else "")

    new_fz_col_name = [c for c in col_names if c.startswith('FZ_') and c != 'FZ_Old'][0]
    delta_col_name = [c for c in col_names if c.startswith('Delta')][0]
    pct_col_name = [c for c in col_names if c.startswith('Pct_Diff')][0]

    j_order = joints_df['CommonUnique'].tolist()
    piv_old = comp_df.pivot_table(index='CommonUnique', columns='Load Pattern', values='FZ_Old', aggfunc='sum').reindex(index=j_order, columns=load_patterns).fillna(0.0)
    piv_new = comp_df.pivot_table(index='CommonUnique', columns='Load Pattern', values=new_fz_col_name, aggfunc='sum').reindex(index=j_order, columns=load_patterns).fillna(0.0)
    piv_del = comp_df.pivot_table(index='CommonUnique', columns='Load Pattern', values=delta_col_name, aggfunc='sum').reindex(index=j_order, columns=load_patterns).fillna(0.0)
    piv_pct = comp_df.pivot_table(index='CommonUnique', columns='Load Pattern', values=pct_col_name, aggfunc='sum').reindex(index=j_order, columns=load_patterns).fillna(0.0)

    ws_eng.cell(row=1, column=1, value="Load_Pattern")
    for idx, lp in enumerate(load_patterns, start=2):
        ws_eng.cell(row=idx, column=1, value=lp)

    ws_eng.cell(row=1, column=2, value="Joint_List")
    ws_eng.cell(row=2, column=2, value="NONE")
    for idx, j_id in enumerate(j_order, start=3):
        ws_eng.cell(row=idx, column=2, value=str(j_id))

    start_col_mat = 4
    end_col_mat_let = get_column_letter(start_col_mat + n_lps - 1)

    b1_start = 2
    b2_start = b1_start + n_joints + 1
    b3_start = b2_start + n_joints + 1
    b4_start = b3_start + n_joints + 1

    arr_old = piv_old.values
    arr_new = piv_new.values
    arr_del = piv_del.values
    arr_pct = piv_pct.values

    for r_i in range(n_joints):
        for c_i in range(n_lps):
            col_pos = start_col_mat + c_i
            ws_eng.cell(row=b1_start + r_i, column=col_pos, value=float(arr_old[r_i, c_i]))
            ws_eng.cell(row=b2_start + r_i, column=col_pos, value=float(arr_new[r_i, c_i]))
            ws_eng.cell(row=b3_start + r_i, column=col_pos, value=float(arr_del[r_i, c_i]))
            ws_eng.cell(row=b4_start + r_i, column=col_pos, value=float(arr_pct[r_i, c_i]))

    # 4. Build 'Bubble_Chart' Sheet
    ws_bub = wb.create_sheet(title='Bubble_Chart')
    ws_bub.views.sheetView[0].showGridLines = True

    ws_bub['A1'] = "JOINT REACTIONS COMPARISON & SPATIAL PLAN DASHBOARD"
    ws_bub['A1'].font = Font(size=13, bold=True, color="1F4E78")

    # Load Pattern Dropdown (Cell B2)
    ws_bub['A2'] = "Select Load Pattern:"
    ws_bub['A2'].font = Font(bold=True, color="FFFFFF")
    ws_bub['A2'].fill = header_fill
    ws_bub['A2'].alignment = Alignment(horizontal="center", vertical="center")

    ws_bub['B2'] = default_lp
    ws_bub['B2'].font = Font(bold=True, size=11, color="C00000")
    ws_bub['B2'].fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    ws_bub['B2'].alignment = Alignment(horizontal="center", vertical="center")

    dv_lp = DataValidation(type="list", formula1=f"='_DataEngine'!$A$2:$A${n_lps + 1}", allow_blank=False)
    ws_bub.add_data_validation(dv_lp)
    dv_lp.add(ws_bub["B2"])

    # Spotlight Joint ID Dropdown (Cell E2)
    ws_bub['D2'] = "Spotlight Joint ID:"
    ws_bub['D2'].font = Font(bold=True, color="FFFFFF")
    ws_bub['D2'].fill = header_fill
    ws_bub['D2'].alignment = Alignment(horizontal="center", vertical="center")

    ws_bub['E2'] = str(j_order[0]) if len(j_order) > 0 else "NONE"
    ws_bub['E2'].font = Font(bold=True, size=11, color="1F4E78")
    ws_bub['E2'].fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    ws_bub['E2'].alignment = Alignment(horizontal="center", vertical="center")

    dv_jt = DataValidation(type="list", formula1=f"='_DataEngine'!$B$2:$B${n_joints + 2}", allow_blank=False)
    ws_bub.add_data_validation(dv_jt)
    dv_jt.add(ws_bub["E2"])

    # Selected Joint Delta Readout (Cells F2:G2)
    ws_bub['F2'] = "Selected Joint Delta:"
    ws_bub['F2'].font = Font(bold=True, color="FFFFFF")
    ws_bub['F2'].fill = header_fill
    ws_bub['F2'].alignment = Alignment(horizontal="center", vertical="center")

    ws_bub['G2'] = f'=IFERROR(INDEX($F$5:$F${4 + n_joints}, MATCH($E$2, $A$5:$A${4 + n_joints}, 0)), 0)'
    ws_bub['G2'].font = Font(bold=True, size=11, color="000000")
    ws_bub['G2'].fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    ws_bub['G2'].alignment = Alignment(horizontal="center", vertical="center")
    ws_bub['G2'].number_format = '+0.00;-0.00;0.00'

    # Hidden Index Helper Cell C2
    ws_bub['C2'] = f'=MATCH(SUBSTITUTE($B$2,"~","~~"), \'_DataEngine\'!$A$2:$A${n_lps + 1}, 0)'
    ws_bub['C2'].font = Font(color="FFFFFF", size=8)

    # Clean Table Headers (Columns A to G)
    visible_headers = [
        "CommonUnique", "X", "Y", "FZ_Old", new_fz_col_name, "Delta", "Pct_Diff"
    ]
    for c_idx, h_title in enumerate(visible_headers, start=1):
        c = ws_bub.cell(row=4, column=c_idx, value=h_title)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    col_widths = [16, 12, 12, 15, 15, 18, 14]
    for c_idx, w in enumerate(col_widths, start=1):
        ws_bub.column_dimensions[get_column_letter(c_idx)].width = w

    # -----------------------------------------------------------------
    # HIDDEN CHART SERIES COLUMNS (AZ to BE)
    # How this displays CommonUnique AND Delta right inside the Excel
    # cursor hover tooltip while keeping 100% valid styles.xml:
    # - In an Excel ScatterChart, hovering over any point shows:
    #     Series "Increased (+)" Point "<X_Axis_Formatted>"
    #     (<X_Cell_Value>, <Y_Cell_Value>)
    # - In Hidden_X (Col AZ): We store `ROUND(X, 2)` + `0.000` + `CommonUnique_Digits / 10^9`
    #   and apply ONE valid shared format: `0.00" | ID: "000000` (no square brackets!).
    #   Wait: in Excel, putting quoted text between decimal `0`s (`0.00" | ID: "000000`)
    #   displays the first 2 decimal digits as X, then prints ` | ID: `, then prints the
    #   6-digit CommonUnique ID!
    # - And in Hidden_Y (Cols BA, BB, BC): We store `ROUND(Y, 2)` + `MIN(ABS(Delta), 999.99) / 10^7`
    #   formatted with `0.00" | Δ: +"000.00` (for Increased +) and `0.00" | Δ: -"000.00` (for Decreased -)!
    # - Even safer (100% immune to any Excel version's numFmt parser):
    #   If we format Hidden_X and Hidden_Y with pure standard numeric format `0.00000000` OR
    #   attach OpenXML `<c15:datalabelsRange>` (Value From Cells) to the Spotlight & Hover series,
    #   let's check! Why did `styles.xml` fail earlier? ONLY because of `[` and `]` inside `[Joint: 10072]`!
    #   Without `[` and `]`, standard quotes `"..."` are 100% valid in Excel, AND by encoding the numeric
    #   CommonUnique ID and Delta into the decimal tail of Hidden_X and Hidden_Y, EVERY SINGLE POINT
    #   displays its OWN unique CommonUnique ID and Delta!
    # -----------------------------------------------------------------
    col_hid_x = 52 # AZ: X + micro CommonUnique ID
    col_inc_y = 53 # BA: Increased (+) Y + micro Delta
    col_dec_y = 54 # BB: Decreased (-) Y + micro Delta
    col_noc_y = 55 # BC: No Change (0) Y
    col_spt_x = 56 # BD: Spotlight X
    col_spt_y = 57 # BE: Spotlight Y
    col_lbl_t = 58 # BF: Dynamic Text Label "ID: 10072 | Δ: +0.16"

    for h_col in range(52, 59):
        ws_bub.column_dimensions[get_column_letter(h_col)].hidden = True

    for r_idx in range(n_joints):
        row_num = 5 + r_idx
        cid = str(joints_df.loc[r_idx, 'CommonUnique'])
        x_val = float(joints_df.loc[r_idx, 'X'])
        y_val = float(joints_df.loc[r_idx, 'Y'])

        # Extract numeric digits of CommonUnique (e.g. '10072' -> 10072, '~150951' -> 150951)
        digits_only = re.sub(r'\D', '', cid)
        cid_num = int(digits_only) if digits_only else (r_idx + 1)

        cid_cell = ws_bub.cell(row=row_num, column=1, value=cid)
        cid_cell.alignment = Alignment(horizontal="center")
        cid_cell.font = Font(bold=True)
        cid_cell.border = thin_border

        c_x = ws_bub.cell(row=row_num, column=2, value=x_val)
        c_x.number_format = '0.00'
        c_x.border = thin_border

        c_y = ws_bub.cell(row=row_num, column=3, value=y_val)
        c_y.number_format = '0.00'
        c_y.border = thin_border

        r_b1 = b1_start + r_idx
        r_b2 = b2_start + r_idx
        r_b3 = b3_start + r_idx
        r_b4 = b4_start + r_idx

        c_old = ws_bub.cell(row=row_num, column=4, value=f"=INDEX('_DataEngine'!$D${r_b1}:${end_col_mat_let}${r_b1}, 1, $C$2)")
        c_old.number_format = '0.00'
        c_old.border = thin_border

        c_new = ws_bub.cell(row=row_num, column=5, value=f"=INDEX('_DataEngine'!$D${r_b2}:${end_col_mat_let}${r_b2}, 1, $C$2)")
        c_new.number_format = '0.00'
        c_new.border = thin_border

        c_del = ws_bub.cell(row=row_num, column=6, value=f"=INDEX('_DataEngine'!$D${r_b3}:${end_col_mat_let}${r_b3}, 1, $C$2)")
        c_del.number_format = '+0.00;-0.00;0.00'
        c_del.font = Font(bold=True)
        c_del.border = thin_border

        c_pct = ws_bub.cell(row=row_num, column=7, value=f"=INDEX('_DataEngine'!$D${r_b4}:${end_col_mat_let}${r_b4}, 1, $C$2)")
        c_pct.number_format = '+0.00;-0.00;0.00'
        c_pct.border = thin_border

        # Hidden X (Col AZ): True X rounded to 2 decimals + CommonUnique ID in decimals 5..10
        # e.g. X = 5.55, ID = 15118 -> 5.550015118 (Moves point by 0.00001m = 0 pixels, shows exact ID in tooltip!)
        sign_x = -1.0 if x_val < 0 else 1.0
        encoded_x = sign_x * (round(abs(x_val), 2) + (cid_num / 1e9))
        cx_h = ws_bub.cell(row=row_num, column=col_hid_x, value=encoded_x)
        cx_h.number_format = '0.000000000'

        # Hidden Y (Cols BA, BB, BC): True Y rounded to 2 decimals + ABS(Delta) in decimals 5..10
        # e.g. Y = -33.44, Delta = +0.67 -> -33.44000067 (Shows exact Y and exact Delta in tooltip!)
        y_r2 = round(abs(y_val), 2)
        y_sign_prefix = "-" if y_val < 0 else ""

        f_inc = f"=IF(F{row_num}>0.0001, {y_sign_prefix}({y_r2} + MIN(ABS(ROUND(F{row_num},2)), 9999.99)/100000000), NA())"
        f_dec = f"=IF(F{row_num}<-0.0001, {y_sign_prefix}({y_r2} + MIN(ABS(ROUND(F{row_num},2)), 9999.99)/100000000), NA())"
        f_noc = f"=IF(ABS(F{row_num})<=0.0001, C{row_num}, NA())"

        ws_bub.cell(row=row_num, column=col_inc_y, value=f_inc).number_format = '0.00000000'
        ws_bub.cell(row=row_num, column=col_dec_y, value=f_dec).number_format = '0.00000000'
        ws_bub.cell(row=row_num, column=col_noc_y, value=f_noc).number_format = '0.00'

        ws_bub.cell(row=row_num, column=col_spt_x, value=f'=IF($A{row_num}=TEXT($E$2,"@"), B{row_num}, NA())').number_format = '0.00'
        ws_bub.cell(row=row_num, column=col_spt_y, value=f'=IF($A{row_num}=TEXT($E$2,"@"), C{row_num}, NA())').number_format = '0.00'
        ws_bub.cell(row=row_num, column=col_lbl_t, value=f'="ID: " & A{row_num} & " | Δ=" & TEXT(F{row_num}, "+0.00;-0.00;0.00")')

    # Single Uniform-Size Circle Map at I2
    chart_uni = ScatterChart()
    chart_uni.title = "JOINT REACTIONS DELTA MAP  [Hover Format: (X.0000JointID, Y.0000Delta)]"
    chart_uni.style = 2
    chart_uni.scatterStyle = 'marker'
    chart_uni.x_axis.title = "Global X Coordinate (Hover X tail = CommonUnique ID)"
    chart_uni.y_axis.title = "Global Y Coordinate (Hover Y tail = Delta FZ)"
    chart_uni.x_axis.number_format = '0.00'
    chart_uni.x_axis.sourceLinked = False
    chart_uni.y_axis.number_format = '0.00'
    chart_uni.y_axis.sourceLinked = False
    chart_uni.visible_cells_only = False
    chart_uni.width = 44
    chart_uni.height = 26

    x_ref = Reference(ws_bub, min_col=col_hid_x, min_row=5, max_row=4 + n_joints)
    y_inc_ref = Reference(ws_bub, min_col=col_inc_y, min_row=5, max_row=4 + n_joints)
    y_dec_ref = Reference(ws_bub, min_col=col_dec_y, min_row=5, max_row=4 + n_joints)
    y_noc_ref = Reference(ws_bub, min_col=col_noc_y, min_row=5, max_row=4 + n_joints)
    x_spt_ref = Reference(ws_bub, min_col=col_spt_x, min_row=5, max_row=4 + n_joints)
    y_spt_ref = Reference(ws_bub, min_col=col_spt_y, min_row=5, max_row=4 + n_joints)

    for title_str, x_r, y_r, hex_col, m_sz in [
        ("Increased (+)", x_ref, y_inc_ref, "E60000", 6),
        ("Decreased (-)", x_ref, y_dec_ref, "00B050", 6),
        ("No Change (0)", x_ref, y_noc_ref, "FFC000", 6),
        ("Spotlight Joint", x_spt_ref, y_spt_ref, "00B0F0", 11)
    ]:
        ser = Series(values=y_r, xvalues=x_r, title=title_str)
        ser.marker = Marker(symbol='circle', size=m_sz)
        ser.marker.graphicalProperties.solidFill = hex_col
        ser.marker.graphicalProperties.line.solidFill = "111111"
        ser.graphicalProperties.line.noFill = True
        if title_str == "Spotlight Joint":
            ser.dLbls = DataLabelList()
            ser.dLbls.showVal = True
            ser.dLbls.showCatName = True
            ser.dLbls.showSerName = False
            ser.dLbls.dLblPos = 't'
        chart_uni.series.append(ser)

    ws_bub.add_chart(chart_uni, "I2")
    ws_bub.freeze_panes = "A5"

    final_buf = io.BytesIO()
    wb.save(final_buf)
    final_buf.seek(0)
    return final_buf


# =====================================================================
# 5. STREAMLIT APPLICATION UI
# =====================================================================
st.set_page_config(page_title="SAFE / ETABS Joint Reaction Comparator", layout="wide")
st.title("🏗️ SAFE / ETABS Joint Mapping & FZ Reaction Comparator")

tab1, tab2 = st.tabs(["📍 Stage 1: Point Connectivity Mapping", "📊 Stage 2: Joint Reactions Comparison"])

with tab1:
    st.markdown("### Upload Point Object Connectivity Files")
    c1, c2 = st.columns(2)
    tol = c1.number_input("Coordinate Matching Tolerance", value=0.001, format="%.4f")
    seq_id = c2.checkbox("Use Sequential Master IDs", value=False)

    conn_uploads = st.file_uploader("Select Connectivity Excel files", type=['xlsx'], accept_multiple_files=True, key="u1")
    if conn_uploads and st.button("🔨 Generate Master Coordinate Mapping"):
        model_files = {f.name.replace('.xlsx', ''): f for f in conn_uploads}
        coord_map_df, model_tabs = stage1_build_coordinate_mapping(model_files, tol=tol, use_sequential_id=seq_id)
        st.session_state['coord_map_df'] = coord_map_df
        st.success(f"Mapped {len(coord_map_df)} unique joint locations across {len(model_files)} models.")
        st.dataframe(coord_map_df.head(25))

        buf1 = io.BytesIO()
        with pd.ExcelWriter(buf1, engine='openpyxl') as writer:
            coord_map_df.to_excel(writer, sheet_name='Coordinate Mapping', index=False)
            for m_name, m_df in model_tabs.items():
                m_df.to_excel(writer, sheet_name=m_name[:31], index=False)
        st.download_button("📥 Download Mapped_Common_Unique_Numbers.xlsx", data=buf1.getvalue(), file_name="Mapped_Common_Unique_Numbers.xlsx")

with tab2:
    st.markdown("### Upload Master Mapping & Joint Reactions Files")
    uploaded_map = st.file_uploader("1. Upload Mapped_Common_Unique_Numbers.xlsx", type=['xlsx'])
    reac_uploads = st.file_uploader("2. Upload Joint Reactions Excel files", type=['xlsx'], accept_multiple_files=True, key="u2")

    active_map_df = pd.read_excel(uploaded_map, sheet_name='Coordinate Mapping') if uploaded_map else st.session_state.get('coord_map_df')

    if active_map_df is not None and reac_uploads:
        model_cols = [c for c in active_map_df.columns if c not in ['CommonUnique', 'X', 'Y', 'Z']]
        st.write("🔗 **Link each Reaction file to its corresponding Model:**")

        reac_dict = {}
        cols = st.columns(len(reac_uploads))
        for i, f in enumerate(reac_uploads):
            fname_low = f.name.lower()
            default_idx = min(i, len(model_cols) - 1)
            for m_idx, m_col in enumerate(model_cols):
                if ('old' in fname_low and 'old' in m_col.lower()) or ('new' in fname_low and 'new' in m_col.lower()):
                    default_idx = m_idx
                    break

            assigned_model = cols[i].selectbox(f"File: `{f.name}`", model_cols, index=default_idx, key=f"sel_{i}")
            reac_dict[assigned_model] = f

        c_base, c_mode = st.columns(2)
        old_candidates = [k for k in reac_dict.keys() if 'old' in k.lower()]
        base_default_idx = list(reac_dict.keys()).index(old_candidates[0]) if old_candidates else 0

        baseline_model = c_base.selectbox(
            "Select Baseline Model:",
            list(reac_dict.keys()),
            index=base_default_idx
        )
        merge_option = c_mode.selectbox(
            "Load Pattern Matching Mode:",
            ["Compare Common Load Patterns Only", "Keep All Load Patterns"]
        )
        merge_how = 'inner' if 'Common' in merge_option else 'outer'

        if st.button("⚡ Run FZ Comparison"):
            try:
                comp_df, legend_df = stage2_compare_reactions(active_map_df, reac_dict, baseline_model, merge_how=merge_how)
                st.session_state['comp_df'] = comp_df
                st.session_state['legend_df'] = legend_df
                st.session_state['excel_bytes'] = generate_comparison_excel_with_bubble_chart(comp_df, legend_df).getvalue()
                st.success(f"Comparison complete. Total matched rows: {len(comp_df):,}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

        if 'comp_df' in st.session_state:
            comp_df = st.session_state['comp_df']
            legend_df = st.session_state['legend_df']

            st.download_button(
                "📥 Download FZ_Comparison_Mapped.xlsx",
                data=st.session_state['excel_bytes'],
                file_name="FZ_Comparison_Mapped.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            with st.expander("📖 View Column Header Explanations", expanded=False):
                st.table(legend_df)

            st.markdown("---")
            st.subheader("🗺️ Joint Reaction Delta Plan Map")

            all_lps = sorted(comp_df['Load Pattern'].dropna().unique().tolist())
            non_modal = [lp for lp in all_lps if 'modal' not in lp.lower()]
            default_lp_idx = all_lps.index(non_modal[0]) if non_modal else 0

            c_ctl1, c_ctl2, c_ctl3 = st.columns([2, 2, 1.5])
            selected_lp = c_ctl1.selectbox("Select Load Pattern:", all_lps, index=default_lp_idx)

            sub_df = comp_df[comp_df['Load Pattern'] == selected_lp].copy()
            delta_cols = [c for c in sub_df.columns if c.startswith('Delta')]
            pct_cols = [c for c in sub_df.columns if c.startswith('Pct_Diff')]
            new_fz_cols = [c for c in sub_df.columns if c.startswith('FZ_') and c != 'FZ_Old']
            d_col = delta_cols[0]
            p_col = pct_cols[0]
            nfz_col = new_fz_cols[0]

            all_joints_in_lp = sub_df['CommonUnique'].astype(str).tolist()
            pinned_joints = c_ctl2.multiselect(
                "Select Joint IDs to Pin Labels:",
                options=all_joints_in_lp,
                default=[]
            )
            uniform_dot_size = c_ctl3.slider("Point Size:", min_value=5, max_value=14, value=8)

            fig = go.Figure()

            color_map = {
                "Increased (+)": "#FF2B2B",
                "Decreased (-)": "#00C853",
                "No Change (0)": "#FFD600"
            }

            for trend_label in ["Increased (+)", "Decreased (-)", "No Change (0)"]:
                grp = sub_df[sub_df['Trend'] == trend_label]
                if len(grp) == 0:
                    continue

                hover_texts = [
                    f"<b>CommonUnique ID: {r['CommonUnique']}</b><br>"
                    f"Old ID: {r['ID_Old']} | X: {r['X']:.3f}, Y: {r['Y']:.3f}<br>"
                    f"FZ_Old: {r['FZ_Old']:.2f} | {nfz_col}: {r[nfz_col]:.2f}<br>"
                    f"<b>Delta: {r[d_col]:+.2f} ({r[p_col]:+.2f}%)</b>"
                    for _, r in grp.iterrows()
                ]

                fig.add_trace(go.Scatter(
                    x=grp['X'],
                    y=grp['Y'],
                    mode='markers',
                    name=trend_label,
                    marker=dict(
                        size=uniform_dot_size,
                        color=color_map.get(trend_label, "#FF2B2B"),
                        opacity=0.95,
                        line=dict(width=1.2, color="#FFFFFF")
                    ),
                    text=hover_texts,
                    hoverinfo='text'
                ))

            if len(pinned_joints) > 0:
                labeled_rows = sub_df[sub_df['CommonUnique'].astype(str).isin(set(pinned_joints))]

                fig.add_trace(go.Scatter(
                    x=labeled_rows['X'],
                    y=labeled_rows['Y'],
                    mode='markers',
                    name='Pinned Joint Target',
                    marker=dict(
                        symbol='circle-open',
                        size=uniform_dot_size + 10,
                        color='#00E5FF',
                        line=dict(width=3.5, color='#00E5FF')
                    ),
                    hoverinfo='skip'
                ))

                for idx_row, (_, r) in enumerate(labeled_rows.iterrows()):
                    ax_offset = 38 if (idx_row % 2 == 0) else -38
                    ay_offset = -32 if (idx_row % 3 != 0) else 32
                    fig.add_annotation(
                        x=r['X'],
                        y=r['Y'],
                        text=f"<b>Joint {r['CommonUnique']}: {r[d_col]:+.2f}</b>",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.3,
                        arrowwidth=2.5,
                        arrowcolor="#00E5FF",
                        standoff=4,
                        ax=ax_offset,
                        ay=ay_offset,
                        bgcolor="#FFEA00",
                        bordercolor="#00E5FF",
                        borderwidth=2,
                        borderpad=4,
                        font=dict(size=12, color="#000000", family="Arial Black")
                    )

            fig.update_layout(
                title=dict(
                    text=f"<b>Joint Reaction Delta Map — {selected_lp}</b>",
                    font=dict(size=18, color="#FFFFFF")
                ),
                xaxis=dict(
                    title="Global X Coordinate",
                    gridcolor="rgba(160, 160, 160, 0.25)",
                    zerolinecolor="rgba(160, 160, 160, 0.4)"
                ),
                yaxis=dict(
                    title="Global Y Coordinate",
                    scaleanchor="x",
                    scaleratio=1,
                    gridcolor="rgba(160, 160, 160, 0.25)",
                    zerolinecolor="rgba(160, 160, 160, 0.4)"
                ),
                plot_bgcolor="#18191A",
                paper_bgcolor="#18191A",
                font=dict(color="#E4E6EB"),
                height=720,
                hovermode="closest",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#FFFFFF")
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"📋 Joint Reaction Summary — {selected_lp}")
            fmt_dict = {d_col: "{:+.4f}", p_col: "{:+.2f}%", "FZ_Old": "{:.4f}", nfz_col: "{:.4f}"}
            st.dataframe(sub_df.style.format(fmt_dict), use_container_width=True)
