import pandas as pd
import streamlit as st

st.set_page_config(page_title="Road Accidents in India", page_icon="🚦", layout="wide")
st.title("🚦 Road Accident Analysis - India")


@st.cache_data
def load(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Year"] = df["Date"].dt.year
    return df


df = load("data/accidents.csv")

st.sidebar.header("Filters")
states = st.sidebar.multiselect("States", sorted(df["State"].unique()), default=sorted(df["State"].unique()))
years = st.sidebar.slider("Years", int(df["Year"].min()), int(df["Year"].max()),
                          (int(df["Year"].min()), int(df["Year"].max())))
causes = st.sidebar.multiselect("Causes", sorted(df["Cause"].unique()), default=sorted(df["Cause"].unique()))

view = df[df["State"].isin(states) & df["Year"].between(*years) & df["Cause"].isin(causes)]
if view.empty:
    st.warning("No data for these filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accidents", f"{len(view):,}")
c2.metric("Killed", f"{view['Killed'].sum():,}")
c3.metric("Injured", f"{view['Injured'].sum():,}")
c4.metric("Deaths per 100 accidents", f"{view['Killed'].sum() / len(view) * 100:.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Accidents by hour of day")
    st.bar_chart(view.groupby("Hour").size())
    st.subheader("Main causes")
    st.bar_chart(view["Cause"].value_counts())
with right:
    st.subheader("Deaths by state")
    st.bar_chart(view.groupby("State")["Killed"].sum().sort_values(ascending=False))
    st.subheader("Fatality rate by road type (deaths per 100 accidents)")
    st.bar_chart(view.groupby("Road_Type")["Killed"].sum() / view.groupby("Road_Type").size() * 100)

st.subheader("Monthly accidents")
st.line_chart(view.set_index("Date").resample("MS").size())

with st.expander("See the data"):
    st.dataframe(view.head(500))

st.caption("If you are using the bundled sample file, the data is synthetic practice data. "
           "Replace data/accidents.csv with real data for real conclusions.")
