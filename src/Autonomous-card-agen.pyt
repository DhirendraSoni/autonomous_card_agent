from typing import TypedDict, List, Dict, Optional from langgraph.graph import StateGraph, END

""" Autonomous Card Replacement Agent (Multi-Card, Interactive)

What this script does

Detects intent: card replacement (assumed by starting the flow)

Asks for: (1) reason for replacement, (2) card selection (if user has multiple), (3) confirms delivery address from stored records (with ability to update), (4) asks for final confirmation, then (5) cancels old card and places new card delivery.


How interaction works

The agent node decides what to ask next based on state fields.

The outer loop prints the agent's prompt and reads your input from stdin.

The loop updates state according to what the agent is currently "awaiting".


Replace the mock USER_DB and tool functions with real integrations in production. """

-----------------------------

1) State definition

-----------------------------

class State(TypedDict, total=False): # latest user message query: str # running transcript / events (optional) history: List[Dict] # message for the user from the agent answer: str

# conversational slots the agent fills
selected_card: Optional[str]   # card last4 or id
reason: Optional[str]
address: Optional[str]
address_confirmed: bool
final_confirmed: bool

# what the agent expects next from the user
awaiting: Optional[str]  # one of: None, "card_selection", "reason", "address_confirmation", "new_address", "final_confirmation"

-----------------------------

2) Mock data store

-----------------------------

USER_ID = "user-001" USER_DB: Dict[str, Dict] = { USER_ID: { "cards": [ {"id": "card_visa_1234", "last4": "1234", "product": "VISA Platinum", "address": "221B Baker Street, London"}, {"id": "card_mc_5678", "last4": "5678", "product": "MasterCard Gold", "address": "742 Evergreen Terrace, Springfield"}, ] } }

-----------------------------

3) Tools (mock implementations)

-----------------------------

def list_user_cards(user_id: str) -> List[Dict]: return USER_DB.get(user_id, {}).get("cards", [])

def fetch_card_address(card_last4: str, user_id: str) -> Optional[str]: for c in list_user_cards(user_id): if c["last4"] == card_last4: return c["address"] return None

def update_card_address(card_last4: str, new_address: str, user_id: str) -> bool: for c in list_user_cards(user_id): if c["last4"] == card_last4: c["address"] = new_address return True return False

def cancel_and_replace_card(card_last4: str, delivery_address: str, user_id: str) -> str: # In real life: call issuer APIs (cancel card, order replacement, set shipment address) and persist audit trail. return ( f"Card ending {card_last4} cancelled successfully. " f"A new card will be delivered to: {delivery_address}." )

-----------------------------

4) Agent logic (single-node, autonomous slot-filling)

-----------------------------

def agent_node(state: State): """Single agent node that decides what to ask/do next based on state.

It returns updated state and the next hop key ("agent" to continue loop, or END).
"""
# Unpack state with defaults
query = state.get("query", "").strip()
history = state.get("history", [])
selected_card = state.get("selected_card")
reason = state.get("reason")
address = state.get("address")
address_confirmed = state.get("address_confirmed", False)
final_confirmed = state.get("final_confirmed", False)
awaiting = state.get("awaiting")

# 0) Welcome if nothing asked yet
if not awaiting and not (selected_card or reason or address or final_confirmed):
    cards = list_user_cards(USER_ID)
    if not cards:
        state["answer"] = "I couldn't find any cards on your profile. Please contact support."
        return "END", state
    # ask reason first as per requirement (1)
    state["answer"] = (
        "I can help replace your card. First, what's the reason for replacement? "
        "(lost, damaged, stolen, expired, name change, other)"
    )
    state["awaiting"] = "reason"
    return "agent", state

# 1) If we still need the reason
if not reason:
    state["answer"] = (
        "Please share the reason for replacement (lost, damaged, stolen, expired, name change, other)."
    )
    state["awaiting"] = "reason"
    return "agent", state

# 2) If user has multiple cards and hasn't selected one yet
if not selected_card:
    cards = list_user_cards(USER_ID)
    if len(cards) == 1:
        selected_card = cards[0]["last4"]
        state["selected_card"] = selected_card
    else:
        # Ask the user to choose by last4
        options = ", ".join([c["last4"] for c in cards])
        state["answer"] = f"Which card would you like to replace? Please reply with last 4 digits: [{options}]"
        state["awaiting"] = "card_selection"
        return "agent", state

# 3) Confirm delivery address for the selected card
if not address:
    addr = fetch_card_address(state["selected_card"], USER_ID)
    if not addr:
        state["answer"] = (
            "I couldn't find an address for that card. Please provide the full delivery address."
        )
        state["awaiting"] = "new_address"
        return "agent", state
    state["address"] = addr
    state["answer"] = (
        f"The current delivery address on file is:\n{addr}\n"
        "Do you confirm delivery to this address? (yes/no)"
    )
    state["awaiting"] = "address_confirmation"
    return "agent", state

# 4) If address not yet confirmed
if not address_confirmed:
    state["answer"] = "Please confirm delivery address: reply 'yes' to confirm or 'no' to enter a new address."
    state["awaiting"] = "address_confirmation"
    return "agent", state

# 5) Final confirmation step
if not final_confirmed:
    state["answer"] = (
        f"Final confirmation: Replace card ending {state['selected_card']} and deliver the new card to\n"
        f"{state['address']}\n"
        "Reply 'confirm' to proceed or 'cancel' to abort."
    )
    state["awaiting"] = "final_confirmation"
    return "agent", state

# 6) If final confirmed, perform action and END
result = cancel_and_replace_card(state["selected_card"], state["address"], USER_ID)
state["answer"] = result
return "END", state

-----------------------------

5) Build graph

-----------------------------

workflow = StateGraph(State) workflow.add_node("agent", agent_node) workflow.add_conditional_edges( "agent", lambda s: agent_node(s)[0], {"agent": "agent", "END": END}, ) workflow.set_entry_point("agent") app = workflow.compile()

-----------------------------

6) Helper: reducer to apply user input to state based on what the agent asked

-----------------------------

def apply_user_input(state: State, user_text: str) -> State: user_text_norm = (user_text or "").strip() awaiting = state.get("awaiting")

# Keep transcript (optional)
state.setdefault("history", []).append({"role": "user", "content": user_text_norm})

if awaiting == "reason":
    state["reason"] = user_text_norm
    state["awaiting"] = None

elif awaiting == "card_selection":
    # accept last4 only if present in user's cards
    last4s = {c["last4"] for c in list_user_cards(USER_ID)}
    if user_text_norm in last4s:
        state["selected_card"] = user_text_norm
        state["awaiting"] = None
    else:
        state["answer"] = (
            "That doesn't match any of your cards. Please reply with one of the valid last 4 digits."
        )
        # keep awaiting the same

elif awaiting == "address_confirmation":
    if user_text_norm.lower() in ("yes", "y"):  # confirmed
        state["address_confirmed"] = True
        state["awaiting"] = None
    elif user_text_norm.lower() in ("no", "n"):
        state["answer"] = "Okay, please enter the new delivery address in full."
        state["awaiting"] = "new_address"
    else:
        state["answer"] = "Please reply with 'yes' to confirm or 'no' to change the address."

elif awaiting == "new_address":
    if user_text_norm:
        # update address and persist in store
        sel = state.get("selected_card")
        if sel:
            update_card_address(sel, user_text_norm, USER_ID)
        state["address"] = user_text_norm
        state["address_confirmed"] = True
        state["awaiting"] = None
    else:
        state["answer"] = "Please provide a non-empty address."

elif awaiting == "final_confirmation":
    if user_text_norm.lower() in ("confirm", "yes", "y"):
        state["final_confirmed"] = True
        state["awaiting"] = None
    elif user_text_norm.lower() in ("cancel", "no", "n"):
        state["final_confirmed"] = False
        state["answer"] = "Okay, the request has been cancelled. No changes were made."
        return state
    else:
        state["answer"] = "Please reply 'confirm' to proceed or 'cancel' to abort."

# set latest user message in state
state["query"] = user_text_norm
return state

-----------------------------

7) Demo REPL (can be removed in production)

-----------------------------

if name == "main": # Initialize state state: State = { "query": "I want to replace my card", "history": [], "answer": "", "selected_card": None, "reason": None, "address": None, "address_confirmed": False, "final_confirmed": False, "awaiting": None, }

# Start the loop
while True:
    # Invoke agent
    next_state = app.invoke(state)
    print(f"Agent: {next_state['answer']}")

    # If we reached a terminal success message, stop
    if next_state.get("answer", "").lower().startswith("card ending"):
        break
    if next_state.get("answer", "").lower().startswith("i couldn't find any cards"):
        break
    if next_state.get("answer", "").lower().startswith("okay, the request has been cancelled"):
        break

    # Read user input and apply
    user_text = input("You: ")
    next_state = apply_user_input(next_state, user_text)
    state = next_state

print("\nSession ended.")

