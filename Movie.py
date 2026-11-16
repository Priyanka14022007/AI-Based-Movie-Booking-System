import streamlit as st
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

colors = {
    "Blue": ("#E3F2FD", "#1565C0"),
    "Green": ("#E8F5E9", "#2E7D32"),
    "Pink": ("#FCE4EC", "#AD1457"),
    "Orange": ("#FFF3E0", "#E65100"),
    "Purple": ("#F3E5F5", "#6A1B9A"),
    "Beige": ("#F5F5DC", "#8D6E63"),
    "White": ("#FFFFFF", "#424242")
}

with st.sidebar:
    theme = st.radio(" Choose Theme", colors)

bg, side = colors[theme]

st.markdown(f"""
<style>
.stApp {{background-color: {bg};}}
[data-testid="stSidebar"] {{background-color: {side};}}
[data-testid="stSidebar"] * {{color: white !important;}}
</style>
""", unsafe_allow_html=True)




tab1,tab2,tab3,tab4=st.tabs(["Home","Page2","Bookings","Profile"])
with tab1:
   st.sidebar.title("Please Login for Ticket Booking")
   if st.sidebar.button("Signup"):
      USERNAME=st.text_input("Please Enter your Username:")
      PASSWORD1=st.text_input("Please secure your account with a password:")
      PASSWORD2=st.text_input("Please again enter your password")
   if st.sidebar.button("Ticket Booking"):
        username=st.text_input("USERNAME")
        password=st.text_input("PASSWORD1",type="password")
        if st.button("Login"):
           if username==USERNAME and password==PASSWORD1:
            st.session_state.logged_in=True
            st.success("Login Successsful")
           else:
            st.error("Invalid username or password")
           if st.session_state.get("logged_in"):
            st.write("Welcome for booking the show")
           if st.button("Logout"):
            st.session_state.logged_in=False
            st.info("Logged out Successfully")
          
        
with tab1:
    st.header("Movies You need to watch")
    st.image("1a.jpeg")
    st.image("1b.jpg")  
# Movie Data Collection:
  
    data = {
    "Movie_Name": [
        "3 Idiots", "Dangal", "Jawan", "Pathaan",
        "Zindagi Na Milegi Dobara", "Chhichhore",
        "Inception", "Interstellar", "Titanic",
        "Avengers: Endgame", "The Dark Knight", "Frozen",
        "Drishyam", "Stree", "Taare Zameen Par",
        "Yeh Jawaani Hai Deewani", "Pushpa", "RRR",
        "Shershaah", "Munna Bhai M.B.B.S",

        "Sholay", "Lagaan", "Bajrangi Bhaijaan", "PK",
        "Barfi", "Queen", "Andhadhun", "Kahaani",
        "Article 15", "Pink", "Swades", "Chak De! India",
        "Rang De Basanti", "Gully Boy", "Dil Chahta Hai",
        "Tamasha", "Rockstar", "Jab We Met",
        "Kal Ho Naa Ho", "Veer-Zaara",

        "War", "Tiger Zinda Hai", "URI: The Surgical Strike",
        "Baby", "Special 26", "Raid", "Badla", "Talvar",
        "Kahaani 2", "Bhool Bhulaiyaa", "Tumbbad",
        "Bulbbul", "Pari", "Raaz", "Welcome", "Hera Pheri",
        "Phir Hera Pheri", "Dhamaal",
        "Golmaal: Fun Unlimited", "Chup Chup Ke",
        "The Shawshank Redemption",

        "Forrest Gump", "The Matrix", "Gladiator",
        "Parasite", "Joker", "Avatar", "Jurassic Park",
        "Coco", "Finding Nemo"
    ],

    "Genre": [
        "Comedy, Drama", "Sports, Drama", "Action, Thriller",
        "Action, Thriller", "Comedy, Drama", "Comedy, Drama",
        "Sci-Fi, Thriller", "Sci-Fi, Adventure",
        "Romance, Drama", "Action, Sci-Fi", "Action, Crime",
        "Animation, Adventure", "Mystery, Thriller",
        "Horror, Comedy", "Drama", "Romance, Drama",
        "Action, Crime", "Action, Drama", "Biography, War",
        "Comedy, Drama",

        "Action, Adventure", "Drama, Sports",
        "Adventure, Drama", "Comedy, Drama",
        "Comedy, Romance", "Comedy, Drama",
        "Thriller, Crime", "Mystery, Thriller",
        "Crime, Drama", "Drama, Thriller", "Drama",
        "Sports, Drama", "Drama, History", "Drama, Music",
        "Drama, Music", "Romance, Drama", "Romance, Music",
        "Romance, Comedy", "Romance, Drama",
        "Romance, Drama",

        "Action, Thriller", "Action, Adventure",
        "Action, War", "Action, Thriller",
        "Crime, Thriller", "Crime, Drama",
        "Mystery, Thriller", "Crime, Mystery",
        "Mystery, Thriller", "Comedy, Horror",
        "Horror, Fantasy", "Horror, Drama",
        "Horror, Thriller", "Horror, Mystery",
        "Comedy", "Comedy, Crime", "Comedy, Crime",
        "Comedy, Adventure", "Comedy", "Comedy, Romance",
        "Drama",

        "Drama, Romance", "Sci-Fi, Action",
        "Action, Drama", "Thriller, Drama",
        "Crime, Drama", "Sci-Fi, Adventure",
        "Adventure, Sci-Fi", "Animation, Family",
        "Animation, Adventure"
    ],

    "Rating": [
        8.4, 8.3, 7.0, 5.8, 8.2, 8.3, 8.8, 8.7,
        7.9, 8.4, 9.0, 7.4, 8.2, 7.5, 8.3, 7.2,
        7.6, 7.8, 8.3, 8.1,

        8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.2, 8.1,
        8.1, 8.1, 8.2, 8.1, 8.1, 7.9, 8.1, 7.3,
        7.7, 7.9, 7.9, 7.8,

        6.5, 5.9, 8.2, 7.9, 8.0, 7.4, 7.7, 8.1,
        6.7, 7.4, 8.2, 6.5, 6.6, 6.6, 7.1, 8.2,
        7.1, 7.5, 7.4, 6.9,

        9.3, 8.8, 8.7, 8.5, 8.5, 8.3, 7.9, 8.2,
        8.4, 8.2
    ]
}
# Create DataFrame
df=pd.DataFrame(data)
st.write(df)

st.subheader("Recommended Movies for You")
genre = st.selectbox(
    "Choose your favourite genre",
    df["Genre"].unique()
)

if st.button("Recommend Movies"):
    result = df[df["Genre"] == genre]
    result1 = result.sort_values(by="Movie_Name", ascending=False)
    st.write(result1)



with tab2:
  st.subheader("Select Your Location:")

  location = st.selectbox(
    "Select your location:",
    ["Ghatkopar", "Vikhroli", "Andheri", "Kurla"]
)
  st.write("Your Selected Location:", location)
  Language=st.radio("Choose the language you want:",("Hindi","English","Marathi","Telugu","Kannada","Malayalam","Tamil"))
  if Language=="Hindi":
      st.write("a")
    #For English Movies
  elif Language=="English":
      st.write("b")
  elif Language=="Marathi":
      st.write("")
  elif Language=="Marathi":
      st.write("")
  elif Language=="Telugu":
      st.write("")
  elif Language=="Kannada":
      st.write("")
  elif Language=="Malayalam":
      st.write("")
  elif  Language=="Tamil":
      st.write("")
  st.header("Search Your Favourite Movies Here:")
  value=st.text_input("Enter the name of the movie you want to search:")
  if st.button("Search"):
      if value in df["Movie_Name"].values:
        st.success(f"{value} is available in our database.")
      else:
        st.error(f"{value} is not available in our database.")
with tab3:
  st.subheader(" Select Your Seats")
  movie = st.selectbox(
      "Select Movie",
      df["Movie_Name"].values
  )
  seats = st.multiselect(
      "Choose available seats",
      [f"{row}{number}" for row in "ABCDE"
       for number in range(1, 6)]
  )
  show_time = st.selectbox(
      "Select Show Time",
      ["10:00 AM", "1:00 PM", "4:00 PM", "7:00 PM"])
#Price per section of seats

  price1= 200 #Price for section A
  price2= 150 #Price for section B
  price3= 200 #Price for section C
  price4= 300 #Price for section D
  price5= 250 #Price for section
  price=price1+price2+price3+price4+price5
  
  total = len(seats) * price
  if st.button("Confirm Booking"):
      if not seats:
          st.warning("Please select at least one seat!")
      else:
          booking = {
              "Movie": movie,
              "Time": show_time,
              "Seats": seats,
              "Total": total
          }
          if "bookings" not in st.session_state:
              st.session_state.bookings = []
          st.session_state.bookings.append(booking)
          st.success(" Booking Successful!")
          st.write("Movie:", movie)
          st.write("Show Time:", show_time)
          st.write("Seats:", ", ".join(seats))
          st.write("Total Amount: ₹", total)
# Add Online Payement Option
  if st.button("Pay Now"):
    #Select your options for ticket booking
    payment_method = st.radio(
        "Select Payment Method",
        ["Credit Card", "Debit Card", "Net Banking", "UPI"]
    ) 
    #Option No 1 for payment method
    if payment_method == "Credit Card":
        card_number = st.text_input("Enter Credit Card Number")
        expiry_date = st.text_input("Enter Expiry Date (MM/YY)")
        cvv = st.text_input("Enter CVV", type="password")
    #Option No 2 for payment method
    elif payment_method == "Debit Card":
        card_number = st.text_input("Enter Debit Card Number")
        expiry_date = st.text_input("Enter Expiry Date (MM/YY)")
        cvv = st.text_input("Enter CVV", type="password")
    #Option No 3 for payment method
    elif payment_method == "Net Banking":
        bank_name = st.selectbox(
            "Select Your Bank",
            ["HDFC", "ICICI", "SBI", "Axis Bank", "Kotak Mahindra"]
        )
        user_id = st.text_input("Enter User ID")
        password = st.text_input("Enter Password", type="password")
    #Option No 4 for payment method
    elif payment_method == "UPI":
        upi_id = st.text_input("Enter UPI ID")

      
        st.success("Payment Successful! Thank you for your booking.")