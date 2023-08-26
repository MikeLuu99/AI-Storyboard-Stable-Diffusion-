import streamlit as st
from multiapp import MultiApp
import storyboard
import autostorygenerate # import your app modules here
st.set_page_config(
        page_title="AI Storyboard",
        page_icon="media_file/ai.png",
        initial_sidebar_state="expanded"
    )
st.image('media_file/ai.png', width=100)
st.title('AI Storyboard')
app = MultiApp()

# Add all your application here
app.add_app("Create your own Storyboard with AI", storyboard.main)
app.add_app("Create AI-generated movie scenes with your prompt", autostorygenerate.main)

# The main app
app.run()