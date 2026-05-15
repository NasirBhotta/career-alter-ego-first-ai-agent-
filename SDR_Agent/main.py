import asyncio
import os
from typing import Dict
from openai import AsyncOpenAI
import sendgrid
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput
from dotenv import load_dotenv
from sendgrid.helpers.mail import Content, Email, Mail, To

load_dotenv(override=True)


instructions1 = (
    "You are a sales agent working for ComplAI, "
    "a company that provides a SaaS tool for ensuring SOC2 compliance and "
    "preparing for audits, powered by AI. "
    "You write professional, serious cold emails."
)

instructions2 = (
    "You are a humorous, engaging sales agent working for ComplAI, "
    "a company that provides a SaaS tool for ensuring SOC2 compliance and "
    "preparing for audits, powered by AI. "
    "You write witty, engaging cold emails that are likely to get a response."
)

instructions3 = (
    "You are a busy sales agent working for ComplAI, "
    "a company that provides a SaaS tool for ensuring SOC2 compliance and "
    "preparing for audits, powered by AI. "
    "You write concise, to the point cold emails."
)


# lets say we have to use other models here other than the openai, how we will do it, described here

KIMI_BASE_URL = "https://integrate.api.nvidia.com/v1"
kimi_client = AsyncOpenAI(base_url=KIMI_BASE_URL, api_key=os.environ.get("KIMI_API_KEY"))
kimi_model = OpenAIChatCompletionsModel(openai_client=kimi_client, model="moonshotai/kimi-k2.6")



sales_agent1 = Agent(
    name="Professional Sales Agent",
    instructions=instructions1,
    model="gpt-4o-mini",
)

sales_agent2 = Agent(
    name="Engaging Sales Agent",
    instructions=instructions2,
    model="gpt-4o-mini",
)

sales_agent3 = Agent(
    name="Busy Sales Agent",
    instructions=instructions3,
    model="gpt-4o-mini",
)


sales_picker = Agent(
    name="sales_picker",
    instructions=(
        "You pick the best cold sales email from the given options. "
        "Imagine you are a customer and pick the one you are most likely to "
        "respond to. Do not give an explanation; reply with the selected "
        "email only."
    ),
    model="gpt-4o-mini",
)


@function_tool
def send_email(body: str) -> Dict[str, str]:
    """Send out an email with the given body to all sales prospects."""
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email("me.bhotta@gmail.com")
    to_email = To("nasirbhotta@gmail.com")
    content = Content("text/plain", body)
    mail = Mail(from_email, to_email, "Sales email", content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}


draft_description = "Write a cold sales email"

tool1 = sales_agent1.as_tool(
    tool_name="sales_agent1",
    tool_description=draft_description,
)
tool2 = sales_agent2.as_tool(
    tool_name="sales_agent2",
    tool_description=draft_description,
)
tool3 = sales_agent3.as_tool(
    tool_name="sales_agent3",
    tool_description=draft_description,
)


plain_tools = [tool1, tool2, tool3, send_email]

plain_manager_instructions = """
You are a Sales Manager at ComplAI. Your goal is to find the single best cold
sales email using the sales_agent tools.

Follow these steps carefully:
1. Generate Drafts: Use all three sales_agent tools to generate three
different email drafts. Do not proceed until all three drafts are ready.

2. Evaluate and Select: Review the drafts and choose the single best email
using your judgment of which one is most effective.

3. Use the send_email tool to send the best email, and only the best email,
to the user.

Crucial Rules:
- You must use the sales agent tools to generate the drafts; do not write
  them yourself.
- You must send ONE email using the send_email tool, never more than one.
"""

plain_sales_manager = Agent(
    name="Sales Manager",
    instructions=plain_manager_instructions,
    tools=plain_tools,
    model="gpt-4o-mini",
)


@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """Send out an email with the given subject and HTML body to prospects."""
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email("me.bhotta@gmail.com")
    to_email = To("nasirbhotta@gmail.com")
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}


subject_instructions = (
    "You can write a subject for a cold sales email. "
    "You are given a message and you need to write a subject for an email "
    "that is likely to get a response."
)

html_instructions = (
    "You can convert a text email body to an HTML email body. "
    "You are given a text email body which might have some markdown and you "
    "need to convert it to an HTML email body with simple, clear, compelling "
    "layout and design."
)

subject_writer = Agent(
    name="Email subject writer",
    instructions=subject_instructions,
    model="gpt-4o-mini",
)
subject_tool = subject_writer.as_tool(
    tool_name="subject_writer",
    tool_description="Write a subject for a cold sales email",
)

html_converter = Agent(
    name="HTML email body converter",
    instructions=html_instructions,
    model="gpt-4o-mini",
)
html_tool = html_converter.as_tool(
    tool_name="html_converter",
    tool_description="Convert a text email body to an HTML email body",
)


email_manager_tools = [subject_tool, html_tool, send_html_email]

email_manager_instructions = (
    "You are an email formatter and sender. You receive the body of an email "
    "to be sent. You first use the subject_writer tool to write a subject for "
    "the email, then use the html_converter tool to convert the body to HTML. "
    "Finally, you use the send_html_email tool to send the email with the "
    "subject and HTML body."
)

emailer_agent = Agent(
    name="Email Manager",
    instructions=email_manager_instructions,
    tools=email_manager_tools,
    model="gpt-4o-mini",
    handoff_description="Convert an email to HTML and send it",
)


handoff_tools = [tool1, tool2, tool3]
handoffs = [emailer_agent]

sales_manager_instructions = """
You are a Sales Manager at ComplAI. Your goal is to find the single best cold
sales email using the sales_agent tools.

Follow these steps carefully:
1. Generate Drafts: Use all three sales_agent tools to generate three
different email drafts. Do not proceed until all three drafts are ready.

2. Evaluate and Select: Review the drafts and choose the single best email
using your judgment of which one is most effective. You can use the tools
multiple times if you're not satisfied with the results from the first try.

3. Handoff for Sending: Pass ONLY the winning email draft to the 'Email
Manager' agent. The Email Manager will take care of formatting and sending.

Crucial Rules:
- You must use the sales agent tools to generate the drafts; do not write
  them yourself.
- You must hand off exactly ONE email to the Email Manager, never more than
  one.
"""

handoff_sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=handoff_tools,
    handoffs=handoffs,
    model="gpt-4o-mini",
)


async def pick_best_sales_email() -> str:
    draft_prompt = "Write a cold sales email"
    results = await asyncio.gather(
        Runner.run(sales_agent1, draft_prompt),
        Runner.run(sales_agent2, draft_prompt),
        Runner.run(sales_agent3, draft_prompt),
    )
    outputs = [result.final_output for result in results]
    emails = "Cold sales emails:\n\n" + "\n\nEmail:\n\n".join(outputs)
    best = await Runner.run(sales_picker, emails)
    return best.final_output


async def send_plaintext_sales_email() -> str:
    prompt = "Send a cold sales email addressed to 'Dear CEO'"
    result = await Runner.run(plain_sales_manager, prompt)
    return result.final_output


async def send_html_sales_email() -> str:
    prompt = "Send out a cold sales email addressed to Dear CEO from Alice"
    result = await Runner.run(handoff_sales_manager, prompt)
    return result.final_output


async def main() -> None:
    best_email = await pick_best_sales_email()
    print(f"Best sales email:\n{best_email}")

    plain_result = await send_plaintext_sales_email()
    print(f"\nPlaintext sales manager result:\n{plain_result}")

    html_result = await send_html_sales_email()
    print(f"\nHTML sales manager result:\n{html_result}")


if __name__ == "__main__":
    asyncio.run(main())
