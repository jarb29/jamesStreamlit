import streamlit as st

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
#
# def login():
#     if st.button("Log in"):
#         st.session_state.logged_in = True
#         st.rerun()
#
# def logout():
#     if st.button("Log out"):
#         st.session_state.logged_in = False
#         st.rerun()

# login_page = st.Page(login, title="Log in", icon=":material/login:")
# logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

dashboard = st.Page(
    "reports/tiempo_inicio.py", title="Tiempo de inicio", icon=":material/dashboard:", default=True
)
bugs = st.Page("reports/tiempo_entre_cortes.py", title="Tiempo entre inicio de cortes", icon=":material/bug_report:")
alerts = st.Page(
    "reports/tiempo_diario.py", title="Tiempo/Dia", icon=":material/notification_important:"
)
tiempo = st.Page(
    "reports/tiempo_programa.py", title="Tiempo/Programa", icon=":material/notification_important:"
)
reporte = st.Page(
    "reports/reporte.py", title="Reporte", icon=":material/history:"
)
search = st.Page("tools/search.py", title="Search", icon=":material/search:")
history = st.Page("tools/history.py", title="History", icon=":material/history:")

# if st.session_state.logged_in:
pg = st.navigation(
        {
            # "Account": [logout_page],
            "Informe": [dashboard, bugs, alerts, tiempo, reporte],
            # "Tools": [search, history],
        }
    )
# else:
#     pg = st.navigation([login_page])

pg.run()