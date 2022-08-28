"""
Web application for the apartment price estimate

* uses streamlit to render the GUI
* uses two states:
  * get_input : get input variables and show a price estimate live
  * render_feedback: to inform about successfully having sent feedback
* loads regression model via inference.py which also does the predictions
* appends feedback with suggested pricing to a CSV file on a MinIO S3 bucket via storage.py
"""

import streamlit as st
import numpy as np
import apartment_price_estimate.inference as inference
import apartment_price_estimate.storage as storage


def _calculate_apartment():
    """
    Calculate apartment estimate . inference.py does the heavy lifting.

    :return: price for the apartment via session state
    """
    apartment_features = np.array([[int(st.session_state.size),
                                    st.session_state.room_nr,
                                    int(st.session_state.sleeping_room_nr),
                                    int(st.session_state.bathroom_nr),
                                    int(st.session_state.lift),
                                    int(st.session_state.balcony),
                                    int(st.session_state.monument),
                                    int(st.session_state.parking),
                                    energy_efficency_classes_dict[st.session_state.energy_efficiency_class]
                                    ]])

    st.session_state.price = inference.predict_price_on_regression_model(
        regression_model,
        apartment_features
    )


def _get_input():
    """
    First state in the UI state model:
    Load language model and vectorization layer from the inference functions.
    Provide text box for user to input text sample.
    Provide button to start language detection.
    Wait for on_click event on button to proceed, then start .
    """

    st.slider(
        label="Wohnfläche in m²",
        min_value=30,
        max_value=230,
        step=5,
        disabled=False,
        key="size"
    )

    st.slider(
        label="Anzahl der Räume",
        min_value=1.0,
        max_value=8.0,
        step=0.5,
        disabled=False,
        key="room_nr"
    )

    st.slider(
        label="Anzahl der Schlafzimmer",
        min_value=0,
        max_value=5,
        step=1,
        disabled=False,
        key="sleeping_room_nr"
    )

    st.slider(
        label="Anzahl der Badezimmer",
        min_value=1,
        max_value=3,
        step=1,
        disabled=False,
        key="bathroom_nr"
    )

    st.checkbox(
        label="Aufzug",
        value=False,
        key="lift"
    )

    st.checkbox(
        label="Balkon",
        value=False,
        key="balcony"
    )

    st.checkbox(
        label="Denkmalgeschütztes Objekt",
        value=False,
        key="monument"
    )

    st.checkbox(
        label="Parkplatz",
        value=False,
        key="parking"
    )

    st.select_slider(
        label="Energieeffizienzklasse (A = 30-50 kWh/m²*a (geringe Heizkosten), H = 250 kWh/m²*a (hohe Heizkosten))",
        options=(energy_efficency_classes_dict.keys()),
        on_change=None,
        key="energy_efficiency_class"
    )

    _calculate_apartment()

    st.write(f"""Wohnungen mit diesen Eigenschaften kosten etwa:""")

    def pretty_number(price: int) -> str:
        first_part, residue = divmod(price, 1000)
        return str(first_part) + " " + str(f"{residue:03d}")

    st.markdown(f"""### {pretty_number(st.session_state.price)} €""")

    st.subheader("Entspricht der berechnete Preis dem, was du gedacht hattest?")

    st.button('Ja', on_click=_submit_feedback)

    # Collapsed container for submitting feedback
    with st.expander("Nein"):
        st.slider(
            label="Realistischer Preis der Wohnung (in Tausend EUR)?",
            min_value=20,
            max_value=1400,
            value=int(st.session_state.price / 1000),
            step=10,
            disabled=False,
            key="price_feedback"
        )
        st.button('Feedback übermitteln', on_click=_submit_feedback)


def _submit_feedback():
    """
    Insert a feedback record into a table in on an S3 bucket. With "training.py -f"
    this feedback can be queried from the CSV and put into data/feedback
    for retraining the model on this additional data.
    """

    if st.session_state.price_feedback > 0:
        # get feedback file from MinIO to local file
        storage.get_feedback_from_minio(client)

        # Open local file for writing, append mode to append current feedback record
        with open("feedback.csv", "a") as csv_file:
            # assemble row of current feedback record in appropriate order
            csv_file.write(
                f"""{st.session_state.price_feedback * 1000},"""
                f"""{int(st.session_state.size)},"""
                f"""{st.session_state.room_nr},"""  # is of type "float" to match apartments with 1.5 or 2.5 rooms
                f"""{int(st.session_state.sleeping_room_nr)},"""
                f"""{int(st.session_state.bathroom_nr)},"""
                f"""{int(st.session_state.lift)},"""
                f"""{int(st.session_state.balcony)},"""
                f"""{int(st.session_state.monument)},"""
                f"""{int(st.session_state.parking)},"""
                f"""{energy_efficency_classes_dict[st.session_state.energy_efficiency_class]}\n"""
            )

        # put file back to file storage
        storage.upload_to_minio(client)

    # turn UI state model to next state
    st.session_state.ui_state = "render_feedback"


def _render_feedback():
    """
    Just say thank you. And wait for button click to restart with newly initialized session.
    """
    st.subheader("Deine Rückmeldung")
    st.success("Danke für deine Rückmeldung. Damit können wir das Modell verbessern.")
    st.session_state.clear()
    st.button('Neue Schätzung beginnen!')


def main():
    """
    Main Loop of streamlit, where it acts as a one-page application.
    Uses a simple state machine model for the UI: get_input -> render_feedback.
    Uses session variables to handle state.
    """

    # common UI elements for all screens
    st.title("Wohnungspreisschätzer - Leipzig")

    # initialize session state variables
    if "size" not in st.session_state:
        st.session_state.size = 75
    if "room_nr" not in st.session_state:
        st.session_state.room_nr = 3.0
    if "sleeping_room_nr" not in st.session_state:
        st.session_state.sleeping_room_nr = 1
    if "bathroom_nr" not in st.session_state:
        st.session_state.bathroom_nr = 1
    if "lift" not in st.session_state:
        st.session_state.lift = False
    if "balcony" not in st.session_state:
        st.session_state.balcony = False
    if "monument" not in st.session_state:
        st.session_state.monument = False
    if "parking" not in st.session_state:
        st.session_state.parking = True
    if "energy_efficiency_class" not in st.session_state:
        st.session_state.energy_efficiency_class = "D"
    if "price" not in st.session_state:
        st.session_state.price = 0
    if "price_feedback" not in st.session_state:
        st.session_state.price_feedback = 184865 / 1000
    if "ui_state" not in st.session_state:
        st.session_state.ui_state = "get_input"

    with st.sidebar:
        """
        Diese einfache streamlit Applikation schätzt Preise von Eigentumswohnungen in Leipzig mit einem 
        einfachen linearen Regressionsmodell.
                
        Es nutzt dafür alle Verkaufsanzeigen von Eigentumswohnungen von ImmobilienScout24 aus den Jahren 2020 und 2021, 
        aus denen es das lineare Modell abgeleitet hat. Diese Verkaufsanzeigen von Eigentumswohnungen haben im 
        Public Use File zum Zwecke der Anonymisierung nur eine Zuordnung zur Stadt Leipzig an sich, aber nicht zu 
        einem Stadtteil oder einer Adresse in Leipzig.

        Die Nutzerin oder der Nutzer sind selbst aufgefordert, sinnvolle Kombinationen der Parameter 
        einzustellen.
        
        Einschränkungen: Selbst für sinnvolle Kombinationen ist der Schätzwert des linearen Modells teils nicht
        plausibel. An den Rändern der Bereiche passt das lineare Modell an sich nicht. Für kleine Wohnungen
        unterschätzt das Modell die Preise deutlich. Die binären Optionen wie Aufzug, Balkon, Denkmalobjekt, 
        Parkplatz bekommen durch das Modell einen festen Preiseinfluss.
        Der Heizkosteneinfluss bekommt durch das Modell auch einen festen Preiseinfluss pro Stufe, ganz gleich
        ob 1-Raum-Wohnung oder sehr große 6-Raum-Wohnung.
        Der Lageeinfluss einer Wohnung innerhalb der Stadt Leipzig bleibt völlig unberücksichtigt, weil
        ImmobilienScout24 keine Lageinformationen in den Public Use Files bereitstellt.
        Die reale Preisbildung von Eigentumswohnungen ist deutlich komplexer und kann durch das Modell höchstens 
        näherungsweise nachgebildet werden.
        """
        # st.write(st.session_state)

    # simple two-state state machine
    if st.session_state.ui_state == "get_input":
        _get_input()
    elif st.session_state.ui_state == "render_feedback":
        _render_feedback()


if __name__ == "__main__":
    # regression_model = inference.load_regression_model_from_file(
    #     "apartment_price_estimate/model/lpz_apt_prices_regression_model.pickle")
    regression_model = inference.load_regression_model_from_model_store()

    # create client for minio server
    client = storage.create_client()

    # translate energy efficiency class labels to numeric values
    energy_efficency_classes_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
    main()
