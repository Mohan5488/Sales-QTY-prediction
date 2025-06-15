import streamlit as st
import numpy as np
import joblib
from gensim.models import Word2Vec

word2vec_model = Word2Vec.load("product_word2vec.model")
model = joblib.load("quantity_predictor.pkl")

product_list = ["Select Product",'BOYS KURTA PYJAMA', 'BOYS NIGHT DRESS', 'BOYS SHORTS', 'BOYS C.SHIRTS ', 'BOYS TSHIRT', 'BOYS 3/4TH', 'BOYS PAIRS', 'BOYS TROUSERS', 'BOYS BRIEFS', 'BOYS VESTS ', 'BOOKS', 'DEO', 'FACE MASK', 'GOOGLES', 'HANDKERCHIEF', 'KIDS BELT', 'SANITIZER', 'BELTS', 'BOTTLE', 'PENCIL BOX', 'SCHOOL BAGS', "GIRLS KURTI''S", 'GIRLS W.TOP', 'GIRLS CAPRIS', 'GIRLS JEANS', 'GIRLS LEGGINGS', 'GIRLS SHORTS', 'GIRLS C.FROCKS', 'BEDSHEET (DOUBLE)', 'NAPKINS', 'BABY PANTY', 'BABY BOYS PAIRS', 'BABY C.FROCKS', 'BABY FROCK', 'BABY GIRLS PAIRS', 'BABY KURTA PYJAMA', 'BABY SHORTS', 'BABY TRACKS', 'BABY TSHIRT', 'BABY VEST', 'JABLA', 'POWDER', 'TIC-TAC', 'EARINGS', 'HAIR BAND', 'HAIR CLIP', 'LADIES SOCKS', 'RUBBER BAND', 'MENS KURTA', 'MENS KURTA PYJAMA', 'MENS 3/4THS', 'MENS SHORTS', 'MENS SHORTS (C)', 'MENS TRACKS', 'MENS CASUAL SHIRT', 'MENS F.SHIRT', 'MENS TSHIRT', 'MENS TSHIRT (C)', 'MENS C. TROUSERS', 'MENS JEANS', 'MENS BRIEFS', 'MENS VESTS', 'UNIFORM SUITING', 'LAB COAT', 'SCHOOL HALF PANT', 'SCHOOL SHIRT', 'SCHOOL SOCKS', 'SHOES', 'CHUNI', 'LADIES KURTIS', 'LADIES LEGGINGS', 'LADIES NIGHT DRESS', 'LADIES NIGHTY', 'LADIES PETTICOAT', 'BLOOMERS', 'BRA', 'PANTIES', 'SLIPS', 'SPORTS BRA', 'LADIES HARAM PANTS', 'LADIES TSHIRTS', 'LADIES W.TOPS', 'BOYS TRACKS', 'PERFUME', 'SOCKS', 'WALLETS', 'TIFFEN BAGS', 'GIRLS GHAGRA CHOLI', 'GIRLS FROCK', 'MATRESS COVER', 'BABY TROUSERS', 'NAIL POLISH', 'MENS 3/4THS (C)', 'GIRLS SWEATER ', 'UNIFORM DUPATTA', 'SCHOOL FULL PANT', 'SCHOOL TUNIC', 'UNIFORM BOTTOM', 'UNIFORM SALWAR', 'DUPATTA SET', 'GIRLS NIGHT DRESS', 'PANT BITS', 'CAPS', 'TIFFIN BOX', 'GIRLS PAIRS', 'WINDOW', 'COSMETIC', 'HAND BAGS', 'LADIES BELTS', 'LADIES WALLET', 'SCHOOL BELT', 'SCHOOL SKIRT', 'SCHOOL TIE', 'LADIES DRESS MATERIAL', 'LADIES TRACKS', 'H SHIRTING', 'H SUITING', 'LADIES SALWARS SUIT', 'LADIES 3/4THS', 'BEDSHEET(SINGLE)', 'TOWELS (COTTON)', 'SCARFS/STOLES', 'MENS F.TROUSERS', 'UNIFORM SHIRTING', 'YOGA MAT', 'SCHOOL JACKET', 'LADIES JEANS', 'GIRLS SALWAR SUIT', 'DEEWAN SETS', 'DOOR MATS', 'BABY GIFT SETS', 'BABY 3/4TH', 'BATH MATS', 'DOHAR', 'DOOR CURTAINS', 'BABY SLIPS', 'GENERAL ITEM', 'KAJAL', 'LADIES HANDKERCHIEF', 'SHRUG', 'BABY W.TOPS', 'MENS SWEATER', 'SWIMMING CAPS', 'GIRLS TRACK', 'BABY ACCESSORIES', 'BABY SALWAR SUIT', 'PILLOW COVERS', 'HAND TOWEL', 'LADIES SHORTS', 'BOYS TRUNK', 'BABY NIGHT DRESS', 'KIDS SOCKS', 'MOSQUITO NETS', 'BABY BEDS', 'BABY SHIRTS', 'BABY TOWELS/NAPKIN', 'BOYS BLAZER', 'SHAWLS', 'PILLOW', 'MENS PYJAMA', 'NIGHTY SLIPS', 'SCHOOL FLAPS', 'TABLE MATS', 'PLASTIC SHEETS', 'BANGLES', 'MENS LUNGI', 'BINDI', 'ROOM FRESHNER', 'BABY SWEATERS', 'SWIMMING GOGGLES', 'LOGO', 'CUSHION', 'TABLE COVERS', 'GIRLS SWIM COSTUME', 'KITCHEN APPRON', 'FRIDGE TOP', 'DRIVING GLOVES', 'SAREE PIN', 'TRAVELLING BAG', 'SCHOOL FROCK', 'O.T DRESS', 'TOWELS (TURKEY)', 'MENS RAINCOAT', 'BOYS SWEATER ', 'UMBRELLA', 'LADIES SWIMMING COSTUME', 'SHOE POLISH', 'TIES', 'SOFA SETS', 'LADIES SHIRT', 'KIDS RAINCOAT', 'GIRLS SKIRTS', 'CUSHION COVERS', 'MENS JACKET', 'BABY SHOES', 'SCHOOL T-SHIRT', 'BLANKET', 'SCHOOL TRACKS', 'SCHOOL SWEATER', 'BANNANA CLIP', 'SOFA COVER', 'FREE GIFT', 'WOLLEN CAP', 'MONKEY CAPS', 'WOOLEN SCARFS', 'BED COVER', 'LADIES SWEATER ', 'LIP BALM', 'WOOLEN GLOVES', 'WOOLEN SOCKS', 'KIDS CAP', 'MENS COAT', 'SAREE', 'STATIONERY', 'BOYS SHORTS (C)', 'STITCHING CHARGES', 'H UNIFORM', 'STOCKINGS', 'SHAPE WEAR', 'DHOTI SET', 'FRIDGE HC', 'ADVOCATE GOWNS', 'GIRLS TSHIRT', 'BOYS TSHIRT (C)', 'FANCY DRESS', 'SCHOOL FULLPANT BE', 'PENCIL POUCH']
price_buckets = ['Select Price Bucket','0-500', '500-1000', '1000-1500', '1500-2000', '2000-2500', '3500+']
years = ['Select Year',2022, 2023, 2024, 2025, 2026, 2027]
months = ['Select Month',"January", "February", "March", "April", "May", "June",
"July", "August", "September", "October", "November", "December"]

def price_bucket_to_mid(bucket):
    if '+' in bucket:
        return int(bucket.replace('+', '')) + 250
    else:
        low, high = map(int, bucket.split('-'))
        return (low + high) / 2

def get_product_vector(tokens):
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)
def get_month(month):
    month_dict = {
        "December": 12,
        "November": 11,
        "October": 10,
        "September": 9,
        "August": 8,
        "July": 7,
        "June": 6,
        "May": 5,
        "April": 4,
        "March": 3,
        "February": 2,
        "January": 1
    }
    for key, val in month_dict.items():
        if key == month:
            return val

def predict_quantity(input_data):
    mid_price = price_bucket_to_mid(input_data['price_bucket'])
    tokens = input_data['product'].lower().split()
    product_vec = get_product_vector(tokens)
    numeric_features = np.array([input_data['year'], input_data['month'], mid_price]).reshape(1, -1)
    input_vector = np.hstack((numeric_features, product_vec.reshape(1, -1)))
    return model.predict(input_vector)[0]

st.title("📦 Product Quantity Predictor")
col1, col2 = st.columns([5, 5])
with col1:
    product = st.selectbox("Select Product", product_list)
with col2:
    price_bucket = st.selectbox("Select Price Bucket", price_buckets)

col3, col4 = st.columns([5, 5])
with col3:
    year = st.selectbox("Select Year", years)
with col4:
    month = st.selectbox("Select Month", months)
if st.button("Predict Quantity"):
    input_data = {
        'year': year,
        'month': get_month(month),
        'product': product,
        'price_bucket': price_bucket
    }
    predicted_qty = predict_quantity(input_data)
    st.success(f"Predicted Quantity: {round(predicted_qty)}")
