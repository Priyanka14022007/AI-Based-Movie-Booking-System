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
    data=[[
         "https://m.media-amazon.com/images/M/MV5BNzc4ZWQ3NmYtODE0Ny00YTQ4LTlkZWItNTBkMGQ0MmUwMmJlXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
    "https://tse1.mm.bing.net/th/id/OIP.pVCzZi2NLIm1ICXs3ELdDQHaJ4?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
    "https://assets.gadgets360cdn.com/pricee/assets/product/202206/Jawan-poster_1655912386.jpg",
    "https://www.bing.com/images/search?view=detailV2&ccid=3dErbj2p&id=CC751B4FBAE793138E7D2DA331F1755166F614AD&thid=OIP.3dErbj2pyzbYY-7_GA91PwHaJ4&mediaurl=https%3a%2f%2ffilmfare.wwmindia.com%2fcontent%2f2022%2fdec%2fpathaan21669878743.jpg&exph=1200&expw=900&q=Pathan&FORM=IRPRST&ck=830CC1B030D847393F5EF653EA7BE8B5&selectedIndex=0&itb=0",
    "https://th.bing.com/th/id/OIP.PcEPrrOvliLyat_xAcz0_wHaKs?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.5RYlXAzbMsOs8Ss4fFJ9ogHaEK?w=320&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.vnJImFIy1GEoBBAjyZ-tfQHaK-?w=132&h=187&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
     "https://th.bing.com/th/id/OIP.uiaj_IMaC7h3NoieAhcmVwHaLG?w=132&h=181&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
     "https://www.bing.com/images/search?view=detailV2&ccid=yISCAt9X&id=3E98E2EFD124D45318AD968A1CF20F97315431C5&thid=OIP.yISCAt9X1_31N4axTrzosAHaLH&mediaurl=https%3a%2f%2fimages.plex.tv%2fphoto%3fsize%3dlarge-720%26scale%3d2%26url%3dhttps%3a%252F%252Fmetadata-static.plex.tv%252F7%252Fgracenote%252F755f50f15f46112eb0c882e8437a4d2e.jpg&exph=2160&expw=1440&q=Titanic+1997+movie&FORM=IMVNXT&ck=983B21D7C961931E2E15235773CA7F04&selectedIndex=1&itb=0&rtpu=%2fimages%2fsearch%3fq%3dTitanic",
     "https://www.bing.com/images/search?view=detailV2&ccid=AsVeA2uT&id=7E5EC048FAA0CE3521C0E182D28EFFB5F95B54DA&thid=OIP.AsVeA2uTH8DIHIezO_yDGAHaK-&mediaurl=https%3a%2f%2fpics.filmaffinity.com%2favengers_endgame-135478227-large.jpg&exph=1200&expw=810&q=avengers+endgame&FORM=IRPRST&ck=3FEBDFDBB638DE189ADA9EB8FF5521DA&selectedIndex=7&itb=0",
     "https://th.bing.com/th/id/OIP.NN9rKH-vZbFgtH4FuoW7OwHaLH?w=133&h=199&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
     "https://th.bing.com/th/id/OIP.ECQhs5CQEAmtAaCATnXIPgHaLH?w=115&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
     "https://th.bing.com/th/id/OIP.3dZzZddEONuMnu9FJ94v7AHaKs?w=207&h=300&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3v",
     "https://th.bing.com/th/id/OIP.eR4n4m1BAKAinVFHUGjgBQHaLH?w=132&h=186&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
     "https://th.bing.com/th/id/OIP.xiGBZTMBhUHrfm2_ueC_TgHaKd?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
     "https://th.bing.com/th/id/OIP.pRgxSaJEtTAPdqAH3xJuIwHaJ3?w=143&h=191&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
     "https://th.bing.com/th/id/OIP.MCgKiaSPTZphX2CDJdHpDwAAAA?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
     "https://th.bing.com/th/id/OIP.DsyE699dbPH9mT4ZpWKh7gHaLG?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
     "https://tse2.mm.bing.net/th/id/OIP.vxGKU0yuI-w7EEumE_NfsgHaHa?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
     "https://th.bing.com/th/id/OIP.2iuURApiETIDyU-kfZl-gQHaEK?w=278&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    #2 Set of Movie Images from Sholay Movie

    "https://images.hindustantimes.com/rf/image_size_960x540/HT/p2/2020/08/13/Pictures/amitabh-bachchan-film-actor-in-sholay_1fd1c202-dd5d-11ea-b1b0-f6d177b03796.jpg",
    "https://images.ottplay.com/images/lagaan-358.jpg",
    "https://images.ottplay.com/images/bajrangi-bhaijaan-486.jpg",
    "https://th.bing.com/th/id/OIP.VXwYF9mUDKEYQcyUwY7ZLwHaKv?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.gF8D64gDH_CbP8K3KRcMcAHaKs?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://tse2.mm.bing.net/th/id/OIP.0FGqYvZm7rjJKFJO53x8AAAAAA?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
    "https://th.bing.com/th/id/OIP.u5_RjgfDZZzqPZWuME2IVQHaEK?w=332&h=187&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.fSRNRwu2yEBC-ALgAQ8dmwHaKr?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.gaqoSDZ_4EGmKbvVaBfexQHaEK?w=297&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.PRDYMhjMEYGCfweK29twnQHaJ3?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.a_BeH8iItoxOEVoTE2_jyQHaKs?w=132&h=181&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.eb-d6JwRCAJPWdJN36zpLgHaLH?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://images.indianexpress.com/2015/01/rang-de-basanti-759.jpg",
    "https://th.bing.com/th/id/OIP.I8sWetd-qk2W4tjrzK9rzQHaJv?w=133&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.v9KvJut1gSHx29AFp1_efAHaHa?w=182&h=185&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.GMItp1yH-aiQNqBai6qKBwHaKJ?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://tse4.mm.bing.net/th/id/OIP.fe6Rem5Vys2Sz9rYq6l-ggHaHa?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
    "https://th.bing.com/th/id/OIP.PZUa3N6VGamU186ndxNtSgHaKs?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3v",
    "data:image/webp;base64,UklGRoQXAABXRUJQVlA4IHgXAAAQXgCdASqrAOoAPp1EnEolo6KkqlP72LATiU2ub4cUbaodjDfTEzgvqVnT5ZAcvmD8P/E9ZvlO9F7zT/YB7qf/H/3P+d+AfnZ9Up/XvUj/Vrriv3J9rG/+OR35r8qfQPzhfJ5fPgfqg9z+e/+e8G+Ai9PtC+/vgi6sXhrot/63jHfi/+r+wHwC/z3+7/+D2h/87/6f6/0Q/sf+89g7y2/YV+6f//91H9lG6K+CVW4wurhtsnVzOy9Sxkxns+yjsFjhEP5UjSxfr3WIOuF+hPKzSQ13rp6Z94n5KrWuKaNcvbggvwYtDRtvARkw/A7DDSVNWSZ1ZstTe0688ChkDj9t/DVIgbOa5zPoCAxFVzR5+fkZrvxSDNqoTxnu8U+vEGjN0M6FVyIxCB9hwj9IM7RanF7NPECpUSTo5T5GXSayHAy+O9VhgUX3hJm0WLvjj8ZsuRv1MtKbvdFrA/J0gGXrsEHAFjf41NFOlJOM3KKXDAwluxY8nkk0o84UEZmcrE2Jkfr0E/wNyoMf/030oAN59fd3UmRqi+FAkhjVnU3liwNnnLysLwDGpna7KrZJAtIpPJVPRwbDZdvwy4r1l4o34Ny+v5XICgGbmbbGwVvWP8kpZHfhvsRNpaFffYx6lJwU8YHXluQZip6jVxj6GL5lj0n6tqMdmV5wLffakp8mr1g3iFY7o25MOWWmLfuCwOwY33jxZ66UtF643tPmKUwWjaFsdSCZLwhQKmViiJxpgp9DfkZLyHV7OAPsodsTHe8SmiYs3sVMc1lPCr0n/4vO7S0JqKwux8kBO7+Xc7LY6F3xn5ZmRJYEJWdChM4tfcG5pkJ7UGaF+sVvccHbzGIzcKsTzROjI3yMeziugxIGiD3sVXiZtvXFr9BT0rWf8scag1AV++5utU3b/TA1DJ8iMU2aClHjdZ6Kg1F7YJADJq2tVsI441wYPLkmqc9R04fW1rYC/iltmHq0PH2mDm8obdB986ha2u2wJ30fWQV2waDukYH1tK338AD+/FnXdIaHUNQgxagFJM8vgoYS3Zt78NUUqClaX7EcqZSM18GcFV5u++QcOboiaN5hWrdrhmDzGW8GpQio8apYx8InEdCriw8bhASzBuvOnIq5Al3/eveauoeBpxooX8qc1Xlrvx4MNNRsyM4bWJrpdps1K0fUtFVeQRgeWXZBrX3Jj0Vkf8JkcQJE/rK/V1LIqQllwUro5CmNk02116WEOjUZVAb+lbjfG/OyO3SJNvhM24bpQ34L7B1o3mgifG+dyZ3KPoh1rxXs27VV78oH4mur9xNc8IvmrmXN6ry6TzdnEh/4Y4TPSH67YAcKzFKEpJsUcTBmvCt2bM/PgN4S8F/btji9tn/HkbWH5mVjzpbwFi48GUlYzwDgt1tJ1JIFfh+d6/Dvs6HhhWC3/Kf11O+K9lw5KHANk0eUCR7K0ihbgHdMrmMMA1u98Ofo/M/nJ2Ay3ysf9/0jjDsIqVNSdpQB8eGSf7QVTZUWDSQj785fM0Ey6H4hZdNmVAvsd9P6TDyEXXJyxlcKNnxzxVxlO4PDS6OLmzzWDlZ36SiMZUOz63aWV+LXDjyYBVQCdZbwbqqougZpBB6DYz0nv97aUArJwLmWBteKK8iCjBPlysGABpzilcNXLEn/e0ux+fT8QDRAAAAD+mdbLXC7lJ3ESTWpNUmMBnDuiG9SgJ4uiLIXYpYoYGa8z3ObBX8XMre/bYTrU2Wsjx3w7jZ6kwDpWZsbB6o44vztEdoEksq/QIodsY9HuEVnnZNDkbw66MRpulO2oRzSgJxzef5U8Ju3MiV2luajj5LJgAEvD9kwQc/7UuWVRo6DWc6GpcVROGd4iYgsqEqxXLWMWJFzQty7H15o+IF3xeaaLCkyjt23PLVCKbxvGyC212P/iiuHCw/Z9xWchYDHMORMkm7BSphnrw5+qbMk56D9SwgiOoN444VfsJ9pVC7+LLu647cgETSyAUwD6WDsnFQhnTmSD/OVWDO0OmauaU8mDOs8n/1fClfnel8g0KbSooEIAq6LxB79qsmWSoJ7mx3qlWhc+2/zb6u0RVhahZrvq9f5+O7mqvfkhQ9FHx226LWoOM+fAY5iyOSWzeS5h6s7D6Rl8P1kqvHj9j+UUuYAfmqmISmehWRs4K18R+OMc+IxHWZqhiVBMrtj+NeBLvMWJ8ygo/kLHV9VnBWhvSwACg3YE1YQw1sr+F0QtMQn689Ztd3h7u6UvRro2qYrEVvBouLJLqBvZcZ8iukch1i2UkrphOmUU5QGDm2H/QlliAoUXizvgcasMMhh+SXKNdCA37dc3+Sli1QqvWT7eXvw2pzAxAVJ4lwyUPh8MO4/lADzPLy0ldh1uwB6S3sK1FBbjVfe/EcAg/VtVlwfUGzTrY6TOvFbCJyC0ih2NtEpEyTBizfp5OjFLpLMqkAEnyMX0xcElMRnyxKvoN4JU0edxG7rSY1sD5U7GSnDHZWHnUcUj8UfH8e4qpDT5ZMCuEulUPpRwOcZPjiidV11TqjiDTEQiFdX+jvfxuK9QZGOrX6gxrEew5kV6Xs2mgdSEE17Ws3/nvESByq58jOMGXOqfT7jelBPdBPl/ZyuvvMA6urbXSUpz0qy9A/zC6UIqgDLIsj7zBwDDadu8VOFg//fjkbu+XzUWRXZ5pxzZcfiw2bUtR9L9vERpIOrlgMPe4+WDxq9M2zE13UuG1h/30rcvegJ+q7aVIlq5JLFxrG/+7L1W2f3bp3yaLjXu375jVMflr+phgDb6c7oP3WkWIKmdpmKNRWbNMmLs5PHyct5PFEsCcXgi+saO4M2o64Iz2x2QHJczTCBL51tBETlHzVDRx4eEoLxKgG6+Bk/WxFwZSgoJyccXP3ewtyqeke7o4lAWVjJts4RbPTEYyuEpQQry6iTSUQFpk4Yxf9YhoV6wnleF7F+6QCKViaKJMnYriHouJcHKrZDhT3wp9X0eaA74XUY16YXMlHciktFs2kK947OaqtPG2TOvajtt0sbN1btW4GQ7xaeNaWRw/MlTmtCS5acUWwTvk4mlLzkTpPb36pHnV1pmDM/JO0+KDmJw1fWwtZ77G/UuPqaG9B2G25KrdCoaxx4eBGff62CBScme645nYggWV4sEhGEE4yc7XVdBS8wE2rXYNFKztAiRjAjfhBIKn2FRVWpaVZ57TLuBPNk2ckoAQxdgnIUfl/W0+FDTYHOSovNIC8u3GyKBD2zVqTy/xISjUW2EXPwuDpair6I0Gj98pcRsrTn2YAK3K+C/7Wa59YQjTkJLbb7ec2V3WRyra/bi6wJDmuKbpw0mz26/Wn9VFaP1mFWyYyeRi3OQHT4dOYQsCWvIo4iV6S49WeJepY2LC9QsBDmZsLx+FzMsjfA2JtsKItYztO52LAmiCkjIJUFXvlStbPTbjtA7ponzj7tj7L/oEIOBja2H+2X4sUobB706O2OCpVBPdxMqFwK8m+wKkAFycR3bF2Kh2ouhfCDMgCHktFkDE2dJE9e/RxfnfU2tNIq6l/5sPowMmlJWwxCuqsW2t22l1FXmUx81ubPozQSRcJ0g1KmBdbGeHIYHUW+uI/R52klbhAB2bgqodwP7ezhYfrxbirlEfTqC5VzDDnIGBB7eUvtr+673hVHZNE06SOJMLQhwI3h14JUkAwGBHh3/L55aBIWL+rbq7Q+Gu6wWy2aNswaFhMAJVsMLZKEJvldbYhPlgg/yzr9Wao+RRoeqqyFlcUKcSrdEVvKAX1BLB2wlYFipTLzpXIREjvcx3ikxo2TcMZ3ZQIj4L+E7xGXeeDhJdiJkltwHBzjIuHAdb8RCXoo257VcOtUoc7UYG37dQ4mHYGknLJHaFj/z6+Xq6KwBI7nw9kY3Prlq5FDGVEIg1eOK+faRWXVE1LbZCtt+vOi8/nvQWI+WqOYcbRNsW7ncG8WxrosUrOxmmJunM9tgZ0qUmEnhuzI+FAtLkgudDTaJKH9KCuJ/2MbQ8WzE6aTUP7/J5tHcUddA1O64UY0wp1ew7gpOD4812aKYus7KYOxSk/46Bsgwkk3E1qUI5Cxl7K0Y7i8rKFlSFNXMeM0nb9Q5ctoJN8L+tocDd9VkqE70pHZzwZ8YtnS1+57KuQw+VqX9gz5u3rrJPjmua/6tQhpoqfJpkMLri5H3UNxWp+o+2Iixkbjl2aSwe+kMC3rRbUaGhNgkHUV828Niw9ayNRRuX5ebIklELWVfJ2twvdiOIYEv9G/upepaDt3HnJ/Cp+UvjzTB/GjF8MrowqBn1eQXizXIlWyiI6c4msbXWs5oyIVfQmfQk5GMhxPiWEszTkCcfEMR9Moy///IC/mt4S733oG9GNWJDSEvqnRTrPUwm9A5sgGzYcCS22n40MnwJP6jxzRyQfsQMZUjt771ptsY6MKX2FwKkQLf/jdh9opaaxgjep8yDm0jZ5bGYzpRaMxH9Y6wwRIrBF3ySWk5sHWuZ9PDmmZ5yMgTRbWo4MZQBuL5OcNkqbhda+uaHERVqqBEhhHQmz9zEtyHjQ9ljXmJ/Tc11SMHzBiH5D1Z5CromIm9C3iLozDOYH2TXHvXGoWfsep1bPwB3gFriWsRIYWUcI6lsVY5pkFy8m6NtPGovx0OiV9sB6+DCWZCDlqFbV9sm7RoUfetosq7ey5AuJJX46QhEUAIWCWtS/Y0fCHdPrAgsxf7JsegZTN1eU161Lz+S7NfF74STP6yDNbd9wP60/torAOsSRZ/USDaNbCoi5lXw87XwgJIXR73GBRgUmag5F4QP4I+q8MyTpbQT3iofTk0jwEoZnttpPBdsbDpzVoazKVK0XAX3AVDUHffefs02GMosdIN858g22rkjlcIji4WFTQ5AhN7ogLLtISw6gmBFSABJ4ib58Q4gfCtWET13spHhr4Ae27g2vMdtDNCH9rB/FOZw/d00QGA2Ot9Zvh6ur4pwApnHbMH7DY/t2Bd0s8Ze4H+O5Q/Iq1JHAmj8Soc6qfDleBN9jFZIz2URyNJ4SY4XEIv4KZOUj4jE1F9G1KcAQsDDV7TOLvDEDH1ySTD7U5z7DTwy3mHrRibWeD8wKiswVMFs8WFb4lfVtPjGjprTsPe5kxRToDhtWEKUE4gyk7K2xpMJQVUwLp0wqmBMXkB5Ylcms5hoLyjx1VZYDUrekApOHdgkpKiSMl1NVxdUfCUnp/bTKiRyyNMG1rWvLUX4x2QUgdwnvPaam4rlN5dc2US9TxqsW7yLfhOHmvWXmcgD46+I9BnvcJdplJ9ZF8zeZUraQjEvGVOZDMbEIIjrDUAnlFzGeKI9GT3bt6bE20W8uloybrYOQo2+wMkgGd/rpakCDlNjpa2Csy5vxi48n/0r5v4MbVQ+ruIlwv9ilPttBmBKTV64JjOd/tAa183lOHcUC1hfJ7xJ3C3bZLFIsk3g65bpJAS+JS+lAqYLnlUI8RPmOMAvQ/En8IpD6LJbl599svmCc+v6NDJV14dMmAHonxPqlaYL09R0R5j8O3F0hbIkS7fI5K5rtvNYnUF0F2KSwgqhjwuKTYXjuUYrMpJUNSNwOD4L50Q6JnupJLqKdQgQNTuwGDKAidbbxsfps5THVdTHNShqTzf6ZVVgGm/b1Vth42YifqSGJfggshaJs1q0oCqRGPRUbhxuXpt2omZrG9eaYP6uwMoViwFpCPz1ywMpSls0ymJucGWFMBdnUjvvrZQhlFT7PrUg+RbSUpmvPwZkHXeKgVWdnStbYnm7a4PoWOyMpdIdJs9lr8kfqnrbyCJIdci/pMQ9NZQ/x7EhsnLGWD1FdHtZyYDZ4H/O/uBYaYuXTyAvDI9U0y3bNMb/vLFzrU5GutwKvlck08EsiAXc3gI4+F3BIgoRx5pe+6nLOCK9T9wjS0F6gSHiFvhDPpy1/cWzqejVC+HxKMaxZPoGQco+jLF00NBldIpLzZSW1YE1DMvY7Bs43/4mIIrcrF1BsaUq1qWatwvbiLt3LZhZJZOUm3Q7jsk2fGit10+Tk/igPv0TElJMgDos9ElA3shXZTX/7Ys4x4HPQXwNBWAAb40FBOKPymGJNQ5C9rjc5JZaj6USMQddc14V7BMOmpBC0Kqr/YRSmxiJBFeHnSPK/pPIoeqPgascarTo3/SrvC0Nh+bPNfMSwsyE4syPZvR0dY1zbxtKHtD1q2i4LPKVjbqhZgVU+anGaZ1ezBivfCG36j1ePX7dqDcu2PhkwGuxUkpjuNMKgcn+zW82IKikQvNYYPOfs+m9ziwsVQyjruDbpV+4YwKeU/3zZAcr9ApEB53EPoQWK4qOgJhNE2CJwIZHjvRhL85nFzVUmtAQNDQUZzzB4o5dyiBXHxNIkgK+Nv1N/6NheUM+BaHcMbl4BI8vu8YHcuWGwYeowgmAvh6QsGORouIx5Y5TZFTMlfdCc+ziDcAu0UGSGAaVFEyFZ0cga3dN1XdznpWPNRTpLKhJMTAceWdz6gbinyYIAv/NnIOxF6XQDaTOKcLwtyPywN3ojS+lE8oWLr/jWM581ev8/jNG+NDqI/k0nHVHEQreQwpEjCaJXik5uP/2EV0rCWfAWFZCByopXOrEMvaUntVSEfciGModooKAIVAINYBDtW1JTH9TA/qRlX1EfCFM/CeDI7gVsXQj1IsZ68NcyiG4uhxumE4Ln8K0tp8ODAGJm7+MTnakMgDDH+Nz1y2mO3fJf3m9uXd6onouuTDXE9pGDLVlgTRq2KhFA6d+AuGYQ2+cfSigPEyDHUfDnCR45Ewy9S+D4K1r45knD4RKjEWCBPy0kVbcVrMfSOCbUzOsPjYjBkp8EbGORejBjqIv0qX63OBEoJ/JA/HQASwBMEEiBQyK437CYEB7dI7MT3+GHTuFTJJ8Y6xvO7/c1f7ZQjJ/5fFcF82BUBf9tN8mB+OC6PeXo1OHP9AX/lKT9uBx95aQqIfDXuW3Mkmlktww+n98RVge+5hBkTL57ZQvNnTs2JFkNlBQbhQFXS4ej4sVzFO12ImPQaSme/XTZ5Bu2Af4rLP46ePuCQNHcszwqqO8/xb3qtldItY12Ivuse6BfSjhfjAtbYEdW7qTm/AVe5F70fv9ORItm1241kDBUjQIiqIW67uJOTnynECHQkfZ8Ibnfew5Vag1G/I7X/wSGB+2qylbBQNO/WU9Re/j414bBa8eWGlxhIby4DucQgMZCo+BigQ9T0ivPZPLGLibwZETeyNQWpBzbMKvy2GK5zK24JhSMOxGwpRM1taZFLDu4yHYeKV2OQ6kr87XRDMbuQGUSHmzQvPfRPa2rm0cYMLUbxO2qA64d0nATSgb7wkhGjh7IyN0u+EDjPl66A3WydDUn3BVVN1289kDsIJxzFQUWpuvSY47OGpiY5lAN83vx5oXOdpiIjKLSDcTtpqJO39vnouEuP9pBHMHSoOWUuXAzwz9OKf9BYcfEVwQUllNsTuB2t8FavPjDKqQiDqUmUFCiPyEqsDUnoioGJkMjd0PTxf3ndTC/6iVK7cfnFdkF4PEj26T/RaoPMlOLMKxdufWieoWmyQ0bbH4cNffj1h9HxTzMrUWIaNYcqvNDrQ6+JRKl4I9NAHnfJLSMcdkTr3MzmGGZpeG5wAn670+LWBj4ekJwmqeFquqyMhUzj9pd1hSJtzz+14zWH/M/rsddty3yJoc7zH0Ll6y1n4eut5nQXSdufD/LFt7jsmjszR1HxI3VDttpKZkrgtuKRS+/bsxPntcqlED7vz9bd81dVOWSbNYC8H/KgrO8qvqdLmmfzXzsnAaEd7bJMIgBYEnKcZ3qIiHRLj2GCJP3dvrG3/ikfCMUg2mZwqkLIoovrMKYDqwszN/hABq+mNHBdrUSf0avmxxfOf4VcNqitFbp+GR7kH568fUeptvmc/tcnB7wL9pA4Swsv/7+f0ryrLvQuFTnvHxJcbvlgmeZ5PEDXYyRH/oKyE1Gziap3l7+zBD2gB8xsGo0zsZfOzJvF2Toa3h7ODxNvtVRQxLUAEw3Kul/rWOexjNK13YH/8S392Ir1PI01gXtgAA==",
    "https://th.bing.com/th/id/OIP.uWIpq74jMmeuryvhtiyzyAHaJ4?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    # Section 3
    "https://th.bing.com/th/id/OIP.FL2oEXsInvzoZh2-Bh_iCQHaJ3?w=137&h=186&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.eNAKIMeW4bHx5lslb3PjowHaE8?w=244&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.EkrVzPsagFeu0q2XsGmeNAHaKs?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://tse4.mm.bing.net/th/id/OIP._h9gTd2SgI6Ip4-_LytDAgHaKu?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
    "https://th.bing.com/th/id/OIP.blgag7qoTsXuZ7GI5NjqtAHaKs?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://www.bing.com/images/search?view=detailV2&ccid=BcR8rLu9&id=39762E091756FB8609FC3199B7F091DB49E58903&thid=OIP.BcR8rLu9BzpQ5dvfjsXgTQHaKs&mediaurl=https%3a%2f%2fm.media-amazon.com%2fimages%2fM%2fMV5BZDcwMWZiYjgtYWE5Zi00NDk0LTkyMDctYzgzMzA4YTZlZjBlXkEyXkFqcGc%40._V1_FMjpg_UX1000_.jpg&exph=1445&expw=1000&q=Raid+movie&FORM=IRPRST&ck=1F7EADF93F4BEE864D526590FCBA7511&selectedIndex=1&itb=0",
    "https://th.bing.com/th/id/OIP.d5CV9HHjIecMCVlPqY4O5gHaE0?w=270&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.QIXf6d_1XZP6q0ofwFskKgHaI4?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://images.ottplay.com/images/big/kahaani-2-poster-1715840710.jpeg?impolicy=ottplay-20210210&width=1200&height=675",
    "https://tse4.mm.bing.net/th/id/OIP.GzLZRsYVTfAEHvoDRfz9hQHaLH?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
    "https://tse1.mm.bing.net/th/id/OIP.oSBzrHv4PUYvWbDjhS3p7wHaKs?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
    "https://th.bing.com/th/id/OIP.gMFiEpUBERSC8p9LvwEO_wHaKk?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://media.gqindia.com/wp-content/uploads/2020/02/Pari.jpg",
    "https://th.bing.com/th/id/OIP.E6X_7W45HaOPAtIxwQGv9AHaKu?w=134&h=194&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.G4I697Ejw8-QUaugjEcX2gHaKv?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.klzufoaxdAjF2ySrTOV4yQHaEK?w=308&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://www.bing.com/images/search?view=detailV2&ccid=fKw9fV4R&id=A45D930AC809C5B79A711B49C07B09ADEA075F8D&thid=OIP.fKw9fV4RSI9I65w_QR9RZQHaFj&mediaurl=https%3a%2f%2fstat4.bollywoodhungama.in%2fwp-content%2fuploads%2f2016%2f03%2f50368184.jpg&exph=768&expw=1024&q=Hera+pheri+movie&FORM=IRPRST&ck=B15DB5D7AE9A79741B42B369EADD81B3&selectedIndex=5&itb=0",
    "https://th.bing.com/th/id/OIP.fOWhfhDpg8Z82ODtG00ucQHaFj?w=238&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.bDrmSv7zX05TcFacdBzaqwHaI5?w=147&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.dUtzWxaDTUDOptZZH7rjAgHaFj?w=237&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.B1GfnOoTppaZgPdTPzh0pwHaLG?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.hEWrJSlzUXiQMoa3sYqT-AHaLH?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://tse2.mm.bing.net/th/id/OIP.arDLjtjGXpEXCJbfbHkpIAHaKv?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
    "https://th.bing.com/th/id/OIP.ltqtxgMmvu6dE8lHKjKXXQHaLH?w=132&h=192&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.4SzxxHd6AhrJIjmE39oKZwHaKe?w=132&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://www.bing.com/images/search?view=detailV2&ccid=ELbV2pfm&id=90CF0A80D7C1570D6139608F32220DF55296CA5C&thid=OIP.ELbV2pfmaMqaj6xKi-A8nwHaK-&mediaurl=https%3a%2f%2ffr.web.img4.acsta.net%2fpictures%2f22%2f11%2f02%2f14%2f49%2f4565071.jpg&exph=1600&expw=1080&q=avatar+movie&FORM=IRPRST&ck=DDE64B1FA87CF3B7C84852227AB43E55&selectedIndex=40&itb=0",
    "https://th.bing.com/th/id/OIP.uwwwrpjl09QdgsMpLEvDZAHaE7?w=266&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.dxB1AvQgPlbgdUblXe3R2wHaKj?w=132&h=187&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    "https://th.bing.com/th/id/OIP.5dCdOjU5wusg1c6nU3ViEAHaLH?w=204&h=306&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3"
    ]]
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