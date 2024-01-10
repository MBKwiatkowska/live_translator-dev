import os

import pyaudio
import queue
import logging
from dotenv import load_dotenv
import openai

load_dotenv()
client = openai.Client()

frame_queue = queue.Queue()
transcription_queue = queue.Queue()
moderated_queue = queue.Queue()
file_queue = queue.Queue()

# Audio configuration
RATE = 16000  # Sample rate
CHUNK = 1024  # Chunk size
BUFFER_DURATION = 5  # Buffer duration in seconds
BUFFER_MAX_SIZE = int(RATE / CHUNK * BUFFER_DURATION)

file_id = 0
text = ''
regex_pattern = r'\b(thanks?\s*(you\s*|for\s*)?|bye|hello).*'

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize logging
logging.basicConfig(level=logging.INFO)

AUDIO_PATH = os.path.join('audios')
os.makedirs(AUDIO_PATH, exist_ok=True)
logging.info(os.path.abspath(AUDIO_PATH))


from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import (
    CombinedMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
)
from langchain.prompts import PromptTemplate

load_dotenv()

presentation_summary = """
- Project Introduction: The project, SubtitleME.live, focuses on real-time event translation using AI.
- Team: Magda Kurlanc and Bartłomiej Marek lead the project.
- Problem Statement: Current lack of effective real-time translation solutions for events.
- AI in Translation: Utilization of AI and large language models like GPT-4 for translation.
- Translation Challenges: Addressing issues of translation quality, cost, and adaptability.
- Technical Aspects: Incorporating AI models such as Whisper for speech-to-text conversion.
- AI Fine-Tuning: Emphasis on fine-tuning AI models for better performance.
- Ethical AI Use: Implementing guardrails for ethical AI operation.
- Initial Language Pair: Starting with Polish to English translations.
- Customization Feature: Offering customizable translation solutions.
- Affordability and Quality: Balancing cost and translation quality.
- Limitations of Current Technology: Acknowledging and addressing existing technological constraints.
- Contextual Understanding: Enhancing AI's understanding of context for accurate translations.
- Managing Incorrect Translations: Strategies for handling translation errors.
- Future Prospects: Potential expansion and application of the technology.
- Market Analysis: Evaluating the demand and potential market for such a service.
- Competitive Landscape: Overview of existing solutions and competitors.
- Business Model: Discussion on how the project will generate revenue.
- Funding and Investment: Strategies for securing funding and investment.
- Technical Roadmap: Detailed timeline and milestones for technical development.
- Marketing Strategy: Approaches to market and promote the service.
- User Experience Design: Focus on creating a seamless user interface.
- Feedback Mechanism: Implementing user feedback for continuous improvement.
- Pilot Testing: Plans for initial testing and pilot programs.
- Data Privacy and Security: Ensuring user data protection and security.
- Regulatory Compliance: Adhering to relevant laws and regulations.
- Sustainability Goals: Incorporating sustainable practices in the project.
- Community Engagement: Engaging with potential users and communities for input.
- Collaboration Opportunities: Exploring partnerships and collaborations.
- Vision for the Future: Long-term vision and goals of the project.
"""


# The following is a fragment of a transcription of the presentation about {presentation_summary}.
_DEFAULT_TEMPLATE = """


### RULES: 
- Your task is to analyze the input message and determine its relevance to the ongoing presentation and previous sentences. 
- If it is relevant, return grammatically correct output 
- Utilize context analysis algorithms, keep in mind last messages and context of the presentation. 
- You can fix the grammar, typos or other errors. 
- The output must be JUST A SENTENCE in English.
- If the text is already in fixed English, no translation is necessary.
- Do not analyze or respond to the content of the text in any way.
- Focus solely on translating the text to English.
- Filter out any "Thank you for watching", "Thank you", "Enjoy your meal" etc. -> Rememeber that is a translation of the 
presentation. Consider that noone eat or wish anyone to enjoy the meal 

### context
{presentation_summary}
#### Examples:
input: "Hello,  I am from Warszawa" -> output: "Hello, I am from Warsaw."
input: "Enjoy your meal" -> output ""
(conversation is related to presentation topics, but the input is not) input: "Thank you for watching. Subscribe my channel." -> output: ""
(conversation is related to presentation topics and last messages does not seems to be the end) input: "Thank you for watching." -> output: ""
(conversation is related to presentation topics or you have no context) input: "Today we are talking about cybersecurity threats" -> output: "Hello, I am from Warsaw."
""".format(presentation_summary=presentation_summary) + '\n' + """
######
Current speech history (last 3 sentences):
{chat_history_lines}
Current sentence: {input}
AI:
"""

conv_memory = ConversationBufferWindowMemory(k=3,
                                             memory_key="chat_history_lines",
                                             input_key="input"
                                             )
prompt = PromptTemplate(input_variables=["input", "chat_history_lines"],
                        template=_DEFAULT_TEMPLATE)




llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=False, memory=conv_memory, prompt=prompt)






