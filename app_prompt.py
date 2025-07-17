# app_prompt.py

import os
import traceback
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.tools import Tool

load_dotenv()

# Custom OpenRouter-based ChatOpenAI
class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", None),
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        resolved_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        if resolved_api_key:
            os.environ["OPENAI_API_KEY"] = resolved_api_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=resolved_api_key,
            **kwargs
        )

# Dummy tool
def get_company_policy_info(topic: str) -> str:
    return f"Kebijakan untuk topik '{topic}' akan disusun berdasarkan ketentuan perusahaan yang berlaku."

tools = [
    Tool(
        name="get_company_policy_info",
        func=get_company_policy_info,
        description="Get Company Policy"
    )
]

# Style Examples
style_examples = [
    {
        "input": "direktur utama harus mengawasi pegawai di cxo office",
        "output": "Direktur Utama wajib mengawasi kinerja seluruh pegawai di CXO Office, memberikan dukungan penuh dalam pelaksanaan tugas, dan segera melaporkan setiap temuan kesalahan, ketidaksesuaian, atau pelanggaran kepada komite etik perusahaan untuk ditindaklanjuti sesuai prosedur yang berlaku."
    },
    {
        "input": "vp harus mengawasi bawahan mulai dari band 2 hingga band 6",
        "output": "Wakil Presiden wajib mengawasi seluruh bawahan dengan rentang grade 2 hingga grade 6, menetapkan tugas secara jelas, dan memastikan kinerja setiap bawahan mencapai standar optimal yang telah ditetapkan perusahaan."
    }
]

# Prompt Templates
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=style_examples,
)

agent_prompt_template = ChatPromptTemplate(
    messages=[
        ("system", """Anda adalah konsultan perusahaan yang ingin membuat kebijakan perusahaan dalam bentuk tulisan dengan gaya formal dan tegas dengan kalimat perintah yang jelas untuk ditulis dalam peraturan perusahaan "hanya satu paragraph saja" juga memiliki banyak sekali action words yang menegaskan tugas seseorang atau suatu fungsi di perusahaan, Anda memiliki akses ke alat berikut:
{tools}

Nama-nama alat yang tersedia: {tool_names}

Ikuti format ini dengan ketat:

Thought: Anda harus selalu memikirkan apa yang harus dilakukan.
Action: Nama alat yang harus dipanggil. Harus salah satu dari [{tool_names}].
Action Input: Masukan ke alat (berupa string JSON).
Observation: Hasil dari alat.
... (ini Thought/Action/Action Input/Observation bisa berulang beberapa kali)
Thought: Saya tahu jawaban akhirnya.
Final Answer: Jawaban akhir untuk pertanyaan asli.
         
format jawaban:
tidak boleh menggunakan tanda " atau # @ ¥ *+=
"""),
        few_shot_prompt,
        ("human", "{input}\n\n{agent_scratchpad}"),
    ],
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)

# Main callable function
def invoke_agent(user_question: str, temperature: float = 0.5, model_name: str = "google/gemma-3-27b-it:free") -> str:
    try:
        llm = ChatOpenRouter(model_name=model_name, temperature=temperature)
        agent = create_react_agent(llm, tools, agent_prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        result = agent_executor.invoke({"input": user_question})
        return result["output"]
    except Exception as e:
        tb = traceback.format_exc()
        return f"❌ Terjadi kesalahan:\n{e}\n\nTraceback:\n{tb}"