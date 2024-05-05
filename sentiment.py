
import streamlit as st
import nltk
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster, HeatMap
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from geopy.geocoders import Nominatim
from streamlit_folium import folium_static
# Initialize NLTK's Vader
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Initialize geocoder
geolocator = Nominatim(user_agent="sentiment_analysis_app")

# Function to analyze sentiment
def analyze_sentiment(text):
    score = sid.polarity_scores(text)
    return score

# Function to plot sentiment distribution
def plot_sentiment_distribution(score):
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [score['pos'], score['neu'], score['neg']]
    explode = (0.1, 0, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# Function to display map with markers
def display_map(locations, clustering=True, heatmap=False):
    mymap = folium.Map(location=[0, 0], zoom_start=2)
    if clustering:
        marker_cluster = MarkerCluster().add_to(mymap)
    for location in locations:
        if clustering:
            folium.Marker(location).add_to(marker_cluster)
        else:
            folium.Marker(location).add_to(mymap)
    if heatmap:
        HeatMap(locations).add_to(mymap)
    return mymap

# Streamlit UI
def main():
    st.title("Sentiment Analyzer with Geospatial Visualization")
    st.write("Enter text below:")

    text_input = st.text_area("Input Text", "")
    if st.button("Analyze"):
        score = analyze_sentiment(text_input)
        st.write("### Sentiment Analysis Results:")
        st.write(f"Positive: {score['pos']}")
        st.write(f"Neutral: {score['neu']}")
        st.write(f"Negative: {score['neg']}")
        
        # Plot sentiment distribution
        plot_sentiment_distribution(score)
    
    st.write("### Geospatial Data Visualization")
    st.write("Enter country or city names separated by semicolons (e.g., Paris, France):")
    location_input = st.text_input("Input Locations", "")
    clustering = st.checkbox("Enable Marker Clustering")
    heatmap = st.checkbox("Enable Heatmap Overlay")
    if st.button("Visualize Map"):
        locations = []
        for loc in location_input.split(';'):
            try:
                location = geolocator.geocode(loc)
                if location:
                    locations.append((location.latitude, location.longitude))
                else:
                    st.error(f"Location '{loc}' not found.")
            except Exception as e:
                st.error(f"Error: {e}")
                return
        if locations:
            mymap = display_map(locations, clustering, heatmap)
            folium_static(mymap)

if __name__ == "__main__":
    main()
