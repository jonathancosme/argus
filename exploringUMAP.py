import streamlit as st
import sqlalchemy
import pandas as pd
from umap import UMAP
import altair as alt
import gc
rawDataUrl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
colNames = ['id', 
'diagnosis',
'radius_mean',
'texture_mean',
'perimeter_mean',
'area_mean',
'smoothness_mean', 
'compactness_mean', 
'concavity_mean',
'concave_points_mean',
'symmetry_mean', 
'fractal_dimension_mean', 
'radius_se',
'texture_se', 
'perimeter_se', 
'area_se',
'smoothness_se', 
'compactness_se',
'concavity_se',
'concave_points_se',
'symmetry_se',
'fractal_dimension_se',
'radius_worst', 
'texture_worst',
'perimeter_worst', 
'area_worst', 
'smoothness_worst',
'compactness_worst',
'concavity_worst', 
'concave_points_worst',
'symmetry_worst', 
'fractal_dimension_worst', 
]

st.title('Exploring UMAP')

"UMAP dimensionality reduction uses a stochastic process. This means each time UMAP is run on a dataset, the results will be different"

"Click the re-run button to try it out yourself. Notice the data points, and the chart, change."

"You can change the UMAP parameters in the sidebar, to explore how they affect the results."

@st.cache
def loadInitialDataSplitsSQL():
	rawDataDf = pd.read_csv(rawDataUrl, header=None, names=colNames, index_col=0)
	return rawDataDf

data = loadInitialDataSplitsSQL()


min_dist = st.sidebar.slider('min_dist', 0.0, 1.0, 0.1)

n_neighbors = st.sidebar.slider('n_neighbors', 0, 50, 15)

metric = st.sidebar.selectbox( "metric", ("euclidean", "manhattan", "chebyshev", "minkowski"))

# @st.cache()
def getUmap(min_dist, n_neighbors, metric, data):
	thisData = data.copy().drop(['diagnosis'], axis=1)
	umap  = UMAP(min_dist=min_dist, n_neighbors=n_neighbors, metric=metric)
	umapfit = umap.fit_transform(thisData)
	umapResults = pd.DataFrame()
	umapResults['diagnosis'] = data['diagnosis']
	umapResults['x'] = umapfit[...,0]
	umapResults['y'] = umapfit[...,1]
	# umapResults['x'] = umapResults['x'] + umapResults['x'].max()
	# umapResults['y'] = umapResults['y'] + umapResults['y'].max()
	return umapResults

UMAPdata = getUmap(min_dist, n_neighbors, metric, data)


if st.button('re-run'):
	UMAPdata2 = getUmap(min_dist, n_neighbors, metric, data)

c = alt.Chart(UMAPdata).mark_circle(size=3, opacity=0.5).encode(x='x', y='y', size=alt.Size('diagnosis', scale=alt.Scale(range=[5, 50])), ).interactive()
st.altair_chart(c, use_container_width=True)

st.subheader("UMAP Data")
st.write(UMAPdata)

# if st.checkbox('view UMAP data'):
# 	st.subheader("UMAP Data")
# 	st.write(UMAPdata)

if st.checkbox('view raw data'):
	st.subheader("Raw Data")
	st.write(data)
