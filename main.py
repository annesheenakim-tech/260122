import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="기온 비교(같은 날짜 대비)", layout="wide")

DEFAULT_CSV_PATH = "ta_20260122174530.csv"

REQUIRED_COLUMNS = ["날짜", "지점", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
TEMP_COLS = ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]

# -----------------------------
# Utilities
# -----------------------------
def read_csv_robust(file_like) -> pd.DataFrame:
    """
    KMA 계열 CSV는 종종 cp949/euc-kr/utf-8-sig 등이 섞여 있어서
    몇 가지 인코딩을 순차 시도.
    """
    encodings_to_try = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings_to_try:
        try:
            if isinstance(file_like, (str, bytes)):
                df = pd.read_csv(file_like, encoding=enc)
            else:
                # Uploaded file: streamlit UploadedFile (BytesIO)
                raw = file_like.read()
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            return df
        except Exception as e:
            last_err = e
            # 업로드 파일은 read()를 소모하므로, 실패 시 다시 세팅 필요:
            if hasattr(file_like, "seek"):
                file_like.seek(0)
            continue
    raise last_err

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼명 좌우 공백/탭 제거
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 날짜 컬럼이 탭/공백을 포함하는 경우가 있어 값도 strip
    if "날짜" in df.columns:
        df["날짜"] = df["날짜"].astype(str).str.strip()

    # 필요한 컬럼 존재 확인
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")

    # 타입 변환
    df["지점"] = pd.to_numeric(df["지점"], errors="coerce").astype("Int64")

    # 날짜 파싱
    df["날짜_dt"] = pd.to_datetime(df["날짜"], errors="coerce")
    # 기온 파싱
    for c in TEMP_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # month-day 키
    df["월일"] = df["날짜_dt"].dt.strftime("%m-%d")
    df["연도"] = df["날짜_dt"].dt.year

    # 완전 필수(날짜, 지점) 없는 행 제거
    df = df.dropna(subset=["날짜_dt", "지점"])
    df["지점"] = df["지점"].astype(int)

    return df

def merge_datasets(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    # 중복 제거 기준: (날짜_dt, 지점) 동일하면 extra가 우선하도록 뒤에 concat 후 drop_duplicates keep='last'
    merged = pd.concat([base, extra], ignore_index=True)
    merged = merged.sort_values(["날짜_dt", "지점"])
    merged = merged.drop_duplicates(subset=["날짜_dt", "지점"], keep="last")
    return merged

def compute_day_stats(df_station: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    """
    같은 월일(예: 01-22)의 과거 분포 대비 선택 날짜가 얼마나 다른지 계산.
    """
    if pd.isna(target_date):
        return {}

    md = target_date.strftime("%m-%d")
    day_pool = df_station[df_station["월일"] == md].copy()

    # target row(해당 날짜 정확히)
    target_row = df_station[df_station["날짜_dt"] == target_date].copy()

    out = {"month_day": md, "pool_n": int(day_pool.shape[0])}

    for col in TEMP_COLS:
        pool_vals = day_pool[col].dropna().values
        if pool_vals.size == 0:
            out[col] = None
            continue

        # 선택 날짜 값
        tval = target_row[col].dropna().values
        tval = float(tval[0]) if tval.size > 0 else np.nan

        mean = float(np.mean(pool_vals))
        std = float(np.std(pool_vals, ddof=1)) if pool_vals.size >= 2 else np.nan
        median = float(np.median(pool_vals))

        # 퍼센타일(선택값이 분포에서 어느 위치인지)
        if np.isfinite(tval):
            pct = float((pool_vals <= tval).mean() * 100.0)
            # 랭크(덥->1, 춥->1 둘 다 보고 싶으면 2개)
            rank_hot = int(np.sum(pool_vals > tval) + 1)   # 큰 값일수록 더 덥다고 가정
            rank_cold = int(np.sum(pool_vals < tval) + 1)  # 작은 값일수록 더 춥다고 가정
            delta = float(tval - mean)
            z = float(delta / std) if (np.isfinite(std) and std > 0) else np.nan
        else:
            pct, rank_hot, rank_cold, delta, z = np.nan, None, None, np.nan, np.nan

        out[col] = {
            "target": tval,
            "mean": mean,
            "median": median,
            "std": std,
            "delta": delta,
            "z": z,
            "percentile": pct,
            "rank_hot": rank_hot,
            "rank_cold": rank_cold,
            "pool_min": float(np.min(pool_vals)),
            "pool_max": float(np.max(pool_vals)),
            "pool_count": int(pool_vals.size),
        }

    return out

def make_distribution_plot(df_station: pd.DataFrame, target_date: pd.Timestamp, temp_col: str):
    md = target_date.strftime("%m-%d")
    pool = df_station[df_station["월일"] == md].copy()
    pool = pool.dropna(subset=[temp_col])

    fig = go.Figure()

    # 바이올린(분포)
    fig.add_trace(go.Violin(
        y=pool[temp_col],
        name=f"{md} 분포",
        box_visible=True,
        meanline_visible=True,
        points="all",
        jitter=0.3,
        scalemode="width"
    ))

    # 선택값 마커
    target_row = df_station[df_station["날짜_dt"] == target_date].dropna(subset=[temp_col])
    if not target_row.empty:
        tval = float(target_row.iloc[0][temp_col])
        fig.add_trace(go.Scatter(
            x=[f"{md} 분포"],
            y=[tval],
            mode="markers",
            name="선택 날짜",
            marker=dict(size=14, symbol="diamond")
        ))

    fig.update_layout(
        title=f"{temp_col} — 같은 월일({md}) 과거 분포 vs 선택 날짜",
        yaxis_title=temp_col,
        xaxis_title="",
        height=450
    )
    return fig

def make_window_timeseries_plot(df_station: pd.DataFrame, target_date: pd.Timestamp, temp_col: str, window_days: int = 30):
    start = target_date - pd.Timedelta(days=window_days)
    end = target_date + pd.Timedelta(days=window_days)

    win = df_station[(df_station["날짜_dt"] >= start) & (df_station["날짜_dt"] <= end)].copy()
    win = win.sort_values("날짜_dt")

    # 같은 월일 기준 과거 평균(전후 창 전체에 대해 해당 월일 평균을 매핑)
    md_mean_map = (
        df_station.dropna(subset=[temp_col])
        .groupby("월일")[temp_col].mean()
        .to_dict()
    )
    win["clim_mean_same_md"] = win["월일"].map(md_mean_map)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=win["날짜_dt"], y=win[temp_col],
        mode="lines+markers",
        name="관측치"
    ))

    fig.add_trace(go.Scatter(
        x=win["날짜_dt"], y=win["clim_mean_same_md"],
        mode="lines",
        name="같은 월일 장기평균"
    ))

    # 선택일 vertical line
    fig.add_vline(x=target_date, line_width=2, line_dash="dash")

    fig.update_layout(
        title=f"{temp_col} — 선택일 전후 {window_days}일(지점별)",
        xaxis_title="날짜",
        yaxis_title=temp_col,
        height=450
    )
    return fig

# -----------------------------
# Load & App UI
# -----------------------------
st.title("🌡️ 기온 비교 대시보드 — 같은 날짜(월-일) 대비 얼마나 춥/덥?")
st.caption("기본 데이터는 자동 탑재되고, 같은 형식의 CSV를 업로드하면 합쳐서 분석합니다. (Plotly 인터랙티브 그래프)")

with st.sidebar:
    st.header("데이터")
    uploaded = st.file_uploader("추가 CSV 업로드(같은 형식)", type=["csv"])

@st.cache_data(show_spinner=False)
def load_base() -> pd.DataFrame:
    df0 = read_csv_robust(DEFAULT_CSV_PATH)
    df0 = normalize_columns(df0)
    return df0

base_df = load_base()

if uploaded is not None:
    extra_df = read_csv_robust(uploaded)
    extra_df = normalize_columns(extra_df)
    df = merge_datasets(base_df, extra_df)
    st.sidebar.success(f"추가 데이터 병합 완료: +{extra_df.shape[0]:,}행 (총 {df.shape[0]:,}행)")
else:
    df = base_df

# 결측치 요약(간단)
with st.expander("🔎 결측치 요약 보기", expanded=False):
    miss = df[REQUIRED_COLUMNS].isna().sum().rename("결측치 개수").to_frame()
    st.dataframe(miss, use_container_width=True)

# 지점 선택
stations = sorted(df["지점"].dropna().unique().tolist())
default_station = stations[0] if stations else None

colA, colB, colC = st.columns([1, 1, 2], vertical_alignment="top")

with colA:
    station = st.selectbox("지점 선택", options=stations, index=0)

df_station = df[df["지점"] == station].copy()
df_station = df_station.sort_values("날짜_dt")

# 기본 날짜: 최신 날짜
latest_date = df_station["날짜_dt"].max()

with colB:
    # Streamlit date_input은 date를 반환하므로 Timestamp로 변환
    chosen_date = st.date_input(
        "비교할 날짜(기본=최신)",
        value=(latest_date.date() if pd.notna(latest_date) else None)
    )
    target_date = pd.to_datetime(chosen_date)

# target_date가 데이터에 없으면 가장 가까운 날짜로 스냅
if target_date not in set(df_station["날짜_dt"]):
    # 가장 가까운 날짜
    diffs = (df_station["날짜_dt"] - target_date).abs()
    nearest = df_station.loc[diffs.idxmin(), "날짜_dt"]
    st.info(f"선택한 날짜({target_date.date()})가 데이터에 없어 가장 가까운 날짜({nearest.date()})로 비교합니다.")
    target_date = nearest

stats = compute_day_stats(df_station, target_date)

# -----------------------------
# KPI Cards
# -----------------------------
with colC:
    st.subheader("요약(같은 월-일 과거 대비)")
    md = stats.get("month_day", "")
    pool_n = stats.get("pool_n", 0)
    st.write(f"- 비교 기준: **월-일 {md}** (해당 지점에서 관측치 **{pool_n:,}개**)")
    target_row = df_station[df_station["날짜_dt"] == target_date][["날짜_dt"] + TEMP_COLS].head(1)
    if not target_row.empty:
        st.dataframe(target_row.rename(columns={"날짜_dt": "날짜"}), use_container_width=True)

def metric_block(colname: str, label: str):
    d = stats.get(colname)
    if not isinstance(d, dict) or not np.isfinite(d.get("target", np.nan)):
        st.warning(f"{label}: 선택 날짜 값이 없거나 비교 불가")
        return

    t = d["target"]
    delta = d["delta"]
    pct = d["percentile"]
    z = d["z"]

    # 해석 문장(간단)
    # pct 낮으면 춥고, 높으면 덥다고 해석
    if np.isfinite(pct):
        if pct <= 10:
            interp = "역대적으로 꽤 추운 편(하위 10%)"
        elif pct >= 90:
            interp = "역대적으로 꽤 더운 편(상위 10%)"
        else:
            interp = "대체로 평년 범위"
    else:
        interp = "상대적 위치 계산 불가"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label} (선택일)", f"{t:.1f}℃")
    c2.metric("같은 월-일 평균", f"{d['mean']:.1f}℃", f"{delta:+.1f}℃")
    c3.metric("퍼센타일", f"{pct:.1f}%" if np.isfinite(pct) else "NA")
    c4.metric("z-score", f"{z:.2f}" if np.isfinite(z) else "NA")
    st.caption(f"해석: {interp}")

st.divider()
st.subheader("① 평균/최저/최고 각각 비교")

tabs = st.tabs(["평균기온", "최저기온", "최고기온"])
tab_cols = [("평균기온(℃)", "평균기온"), ("최저기온(℃)", "최저기온"), ("최고기온(℃)", "최고기온")]

for tab, (cname, label) in zip(tabs, tab_cols):
    with tab:
        metric_block(cname, label)

        left, right = st.columns(2)
        with left:
            fig1 = make_distribution_plot(df_station, target_date, cname)
            st.plotly_chart(fig1, use_container_width=True)

        with right:
            fig2 = make_window_timeseries_plot(df_station, target_date, cname, window_days=30)
            st.plotly_chart(fig2, use_container_width=True)

st.divider()

# -----------------------------
# Download merged dataset (optional)
# -----------------------------
with st.expander("📦 병합된 데이터 다운로드", expanded=False):
    out_csv = df.drop(columns=["날짜_dt"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
    st.download_button("병합 데이터 CSV 다운로드(utf-8-sig)", data=out_csv, file_name="merged_temperature.csv", mime="text/csv")
