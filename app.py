# import libraries
import streamlit as st
from screens import screen_1, screen_2


# main function
def main():
    """

    :return:
    """

    # Dashboard Title
    st.title("Fuzzy Record Matching Demo!")

    # create select box on sidebar
    activities = ['Introduction', 'Solution', 'Parameter Fine-tuning']
    selected_screen = st.sidebar.selectbox("Select Screen", activities)

    # render screens
    if selected_screen == "Introduction":
        st.sidebar.markdown('---')
        # reading markdown
        st.markdown(open("./markdowns/home.md").read())
    elif selected_screen == "Solution":
        screen_1.render()
    elif selected_screen == "Parameter Fine-tuning":
        screen_2.render()

    # about on side bar
    st.sidebar.markdown("---")
    st.sidebar.markdown("# About \n This app has been developed by [Pavan] (https://github.com/pavancs) \
     using [Streamlit](https://streamlit.io/).")


if __name__ == '__main__':
    main()
