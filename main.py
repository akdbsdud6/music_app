import os
from typing import List, Any, Dict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET))

# Initialize OpenAI
llm = OpenAI(temperature=2)

# Music Agent Tools
class MusicTools:
    @staticmethod
    def search_track(query):
        results = sp.search(q=query, type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            return f"Found: {track['name']} by {track['artists'][0]['name']} (ID: {track['id']})"
        return "No track found"

    @staticmethod
    def get_spotify_recommendations(track_id):
        recommendations = sp.recommendations(seed_tracks=[track_id], limit=5)
        return [f"{track['name']} by {track['artists'][0]['name']}" for track in recommendations['tracks']]

    @staticmethod
    def get_mood_recommendations(mood):
        prompt = f"Suggest 5 popular songs that match the mood: {mood}. Just list the songs and artists, no explanation."
        return llm.invoke(prompt).strip().split('\n')

# Define tools
tools = [
    Tool(
        func=MusicTools.search_track,
        name="SearchTrack",
        description="Searches for a track on Spotify. Input should be the song name."
    ),
    Tool(
        func=MusicTools.get_spotify_recommendations,
        name="GetSpotifyRecommendations",
        description="Gets track recommendations from Spotify based on a seed track ID."
    ),
    Tool(
        func=MusicTools.get_mood_recommendations,
        name="GetMoodRecommendations",
        description="Gets song recommendations based on a mood using AI. Input should be a mood or feeling."
    )
]

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

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    return PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )

# Initialize the agent
prompt = get_prompt_template()
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", input_key="input")

# Create React agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

# Main execution
def main():
    print("Welcome to your AI-powered Music Assistant!")
    while True:
        user_input = input("What would you like to do? (Type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Music Assistant. Goodbye!")
            break
        
        # Prepare the input for the agent
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        
        response = agent_executor.invoke({
            "input": user_input,
            "tools": tool_strings,
            "tool_names": tool_names,
            "chat_history": memory.load_memory_variables({})["chat_history"]
        })
        print(f"Assistant: {response['output']}")

if __name__ == "__main__":
    main()