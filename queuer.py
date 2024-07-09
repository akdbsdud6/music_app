import os
import sys
from typing import List, Any, Dict
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, OPENAI_API_KEY
import webbrowser
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QTextEdit, QLineEdit

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QLineEdit
from PySide6.QtCore import Signal

class SpotifyAuthDialog(QDialog):
    auth_completed = Signal(str)

    def __init__(self, auth_url):
        super().__init__()
        self.setWindowTitle("Spotify Authentication")
        self.setGeometry(150, 150, 400, 200)

        layout = QVBoxLayout()

        label = QLabel(f"Please authenticate in the browser and paste the redirect URL here:")
        layout.addWidget(label)

        self.url_input = QLineEdit()
        layout.addWidget(self.url_input)

        auth_button = QPushButton("Open Auth URL")
        auth_button.clicked.connect(lambda: webbrowser.open(auth_url))
        layout.addWidget(auth_button)

        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit_url)
        layout.addWidget(submit_button)

        self.setLayout(layout)

    def submit_url(self):
        redirect_url = self.url_input.text()
        self.auth_completed.emit(redirect_url)
        self.accept()

def get_spotify_client(parent_widget):
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri="http://localhost:8888/callback",
        scope="user-modify-playback-state user-read-playback-state"
    )
    
    auth_url = auth_manager.get_authorize_url()
    
    auth_dialog = SpotifyAuthDialog(auth_url)
    if auth_dialog.exec() == QDialog.Accepted:
        redirect_url = auth_dialog.url_input.text()
        code = auth_manager.parse_response_code(redirect_url)
        token_info = auth_manager.get_access_token(code)
        return spotipy.Spotify(auth=token_info['access_token'])
    else:
        return None
    
#sp = get_spotify_client()

llm = OpenAI(temperature=0.7)

# MusicTools class remains the same

# tools list remains the same

# Music Agent Tools
class MusicTools:
    def __init__(self, spotify_client):
        self.sp = spotify_client

    def search_track(self, query):
        results = self.sp.search(q=query, type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            return f"Found: {track['name']} by {track['artists'][0]['name']} (ID: {track['id']})"
        return "No track found"

    def get_spotify_recommendations(self, track_id):
        if not track_id.startswith("spotify:track:"):
            results = self.sp.search(q=track_id, type='track', limit=1)
            if results['tracks']['items']:
                track_id = results['tracks']['items'][0]['id']
            else:
                return "No track found to base recommendations on"
        
        recommendations = self.sp.recommendations(seed_tracks=[track_id], limit=5)
        return [f"{track['name']} by {track['artists'][0]['name']}" for track in recommendations['tracks']]

    def get_mood_recommendations(self, mood):
        prompt = f"Suggest 5 popular songs that match the mood: {mood}. Just list the songs and artists, no explanation."
        return llm.invoke(prompt).strip().split('\n')
    
    def add_to_queue(self, track_id):
        if not track_id.startswith("spotify:track:"):
            results = self.sp.search(q=track_id, type='track', limit=1)
            if results['tracks']['items']:
                track_id = results['tracks']['items'][0]['id']
            else:
                return "No track found to add to queue"
        
        try:
            self.sp.add_to_queue(track_id)
            track = self.sp.track(track_id)
            return f"Added to queue: {track['name']} by {track['artists'][0]['name']}"
        except spotipy.exceptions.SpotifyException as e:
            return f"Error adding to queue: {str(e)}"

# Define tools



def get_prompt_template() -> PromptTemplate:
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If asked for recommendations based on a mood or genre, use GetSpotifyRecommendations with a relevant search term.

When recommending songs, always try to add them to the user's queue using the AddToQueue tool.
Also, try to recommend the original version of the songs. List of versions NOT TO RECOMMEND: [Acoustic, remix, cover, any sort of lowering or increasing key]
Go through each track in your final response and add them one by one to the queue.

If you successfully added all tracks in your final response to the queue using AddToQueue, your action is done for that iteration. 
Don't you dare ever try to use something called "RemoveFromQueue", that thing doesn't exist and it is not your action to perform.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    return PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )


class SpotifyAssistantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spotify AI Assistant")
        self.setGeometry(100, 100, 600, 400)

        self.sp = None
        self.agent_executor = None
        self.init_ui()
        self.authenticate_spotify()

    def init_ui(self):
        layout = QVBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def authenticate_spotify(self):
        self.sp = get_spotify_client(self)
        if self.sp is None:
            QMessageBox.critical(self, "Authentication Failed", "Failed to authenticate with Spotify. The application will now close.")
            self.close()
        else:
            self.chat_display.append("Successfully authenticated with Spotify!")
            self.setup_agent()

    def setup_agent(self):
        music_tools = MusicTools(self.sp)
        tools = [
            Tool(
                func=music_tools.search_track,
                name="SearchTrack",
                description="Searches for a track on Spotify. Input should be the song name."
            ),
            Tool(
                func=music_tools.get_spotify_recommendations,
                name="GetSpotifyRecommendations",
                description="Gets track recommendations from Spotify based on a seed track ID."
            ),
            Tool(
                func=music_tools.get_mood_recommendations,
                name="GetMoodRecommendations",
                description="Gets song recommendations based on a mood using AI. Input should be a mood or feeling."
            ),
            Tool(
                func=music_tools.add_to_queue,
                name="AddToQueue",
                description="Adds a track to the Spotify queue. Input should be the Spotify track ID."
            )
        ]

        prompt = get_prompt_template()
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", input_key="input")
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=self.memory
        )

    def send_message(self):
        if not self.agent_executor:
            self.chat_display.append("Error: Agent not initialized. Please try restarting the application.")
            return

        user_input = self.input_field.text()
        self.chat_display.append(f"You: {user_input}")
        self.input_field.clear()

        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in self.agent_executor.tools])
        tool_names = ", ".join([tool.name for tool in self.agent_executor.tools])
        
        response = self.agent_executor.invoke({
            "input": user_input,
            "tools": tool_strings,
            "tool_names": tool_names,
            "chat_history": self.memory.load_memory_variables({})["chat_history"]
        })
        
        self.chat_display.append(f"Assistant: {response['output']}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpotifyAssistantGUI()
    window.show()
    sys.exit(app.exec())