import streamlit as st
from generate_function import clear_cuda_memory


class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

        
    def run(self):
        app = st.selectbox(
            'Go To',
            self.apps,
            format_func=lambda app: app['title'],
        )

        app['function']()