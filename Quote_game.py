# Quote_game.py
# ------------------------------------------------------------
# Usage:  streamlit run Quote_game.py
# Requires: streamlit, pandas, your Word_similarity helper
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import random

from Word_similarity import do_it_all_similarity_df_out_both

# ------------------------------------------------------------
# 1. Load dictionary data  (CSV must have columns: word, definition)
# ------------------------------------------------------------
DICT_FILE   = "dict.csv"          # adjust path if needed
df_dict     = pd.read_csv(DICT_FILE)[["word", "definition"]]
WORDS       = df_dict["word"].tolist()
DEFINITIONS = df_dict["definition"].tolist()
N           = len(WORDS)

MAX_GUESSES = 1000
SIM_THRESHOLD = 0.70              # score to auto-win a round

# ------------------------------------------------------------
# 2. Streamlit session-state  (one state per browser tab)
# ------------------------------------------------------------
if "idx" not in st.session_state:
    st.session_state.idx          = 0
    st.session_state.order        = random.sample(range(N), k=N)  # shuffle order
    st.session_state.guesses      = []        # current round guesses
    st.session_state.finished_one = False
    st.session_state.best_row     = None      # best row of current round

def new_word():
    """Advance to next word and reset round state."""
    st.session_state.idx         += 1
    st.session_state.guesses      = []
    st.session_state.finished_one = False
    st.session_state.best_row     = None

# ------------------------------------------------------------
# 3. End-of-game check
# ------------------------------------------------------------
if st.session_state.idx >= N:
    st.success("🎉 All words attempted – refresh to replay.")
    st.stop()

# Current word / answer
current_i      = st.session_state.order[st.session_state.idx]
current_word   = WORDS[current_i]
current_answer = DEFINITIONS[current_i]

# ------------------------------------------------------------
# 4. UI – prompt + buttons
# ------------------------------------------------------------
st.title(f"What is the definition of **{current_word}**?")
guess_text = st.text_input("Your guess", key="current_guess")

col1, col2 = st.columns(2)
submit_btn = col1.button("Submit guess")
give_up_btn= col2.button("Give up")

# ------------------------------------------------------------
# 5. Handle submit / give-up
# ------------------------------------------------------------
if submit_btn and guess_text:
    st.session_state.guesses.append(guess_text)
    st.rerun()                     # refresh the page to show updated table

if give_up_btn:
    st.session_state.finished_one = True

# ------------------------------------------------------------
# 6. When there are guesses, build similarity DataFrame & display
# ------------------------------------------------------------
if st.session_state.guesses:
    guess_arr = [current_answer] + st.session_state.guesses
    df = do_it_all_similarity_df_out_both(
            guess_arr,
            columnnames=['Answer', 'Guess', 'Sent_1_ID', 'Sent_2_ID',
                         'Sentence_Sim', 'Word_Sim'])
    df = (df.query("Sent_1_ID==0 and Sent_2_ID>0")
            .drop(columns=["Sent_1_ID", "Sent_2_ID", "Answer"])
            .loc[:, ["Guess", "Word_Sim", "Sentence_Sim"]])

    # helper column to find overall best similarity per row
    df["MaxSim"] = df[["Sentence_Sim", "Word_Sim"]].max(axis=1)

    # store the single best row in session_state for later display
    st.session_state.best_row = df.loc[df["MaxSim"].idxmax()]

    # show current guess table (without MaxSim column)
    st.dataframe(df.drop(columns="MaxSim")
                   .style.format({"Sentence_Sim": "{:.3f}",
                                  "Word_Sim":      "{:.3f}"}))

    # auto-win if best similarity crosses threshold
    best_score = st.session_state.best_row["MaxSim"]
    if best_score >= SIM_THRESHOLD:
        st.balloons()
        st.success(f"Great! You matched with score {best_score:.3f}")
        st.session_state.finished_one = True

# ------------------------------------------------------------
# 7. Round-end section (win, give-up, or out of guesses)
# ------------------------------------------------------------
if (st.session_state.finished_one or
        len(st.session_state.guesses) >= MAX_GUESSES):

    st.info(f"**Correct definition:**  _{current_answer}_")

    # show best guess row if it exists
    if st.session_state.best_row is not None:
        st.info("**Your best guess:**")
        st.table(st.session_state.best_row.drop("MaxSim"))

    if st.button("Next word"):
        new_word()
        st.rerun()
