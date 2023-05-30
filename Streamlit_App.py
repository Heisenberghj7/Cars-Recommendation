import streamlit as st
from Cars_Recommendation import get_recommendations,New_Cars


def main():
    selected_value = st.sidebar.selectbox('Select a value:', [''] + list(New_Cars['name']))
    if selected_value:
        recommendations = get_recommendations(selected_value)
        st.write("Recommendations:")
        for recommendation in recommendations:
            st.write(recommendation)


if __name__ == "__main__":
    st.title("Cars Recommendation")
    st.write("Select a value from the sidebar.")
    main()