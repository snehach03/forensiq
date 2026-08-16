# Base image: "slim" variant kyunki humein sirf Python runtime chahiye -
# full python:3.13 image ~1GB hai, slim ~150MB. Faster build, kam space.
FROM python:3.13-slim
# Container ke andar working directory - saare aage ke commands (COPY,
# RUN) isi folder ke relative honge.
WORKDIR /app
# Poora code copy karo (pyproject.toml + src/ dono chahiye, kyunki
# ye project "src layout" use karta hai - pip install ke liye src/
# folder present hona zaroori hai, isliye caching-optimization wala
# "pehle sirf pyproject.toml" approach yahan kaam nahi karta)
COPY . .

# Dependencies install karo
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .
# Streamlit ka default port - container ke bahar se access karne ke
# liye ye port expose karna zaroori hai (docker run ke time -p flag
# se map hoga host ke kisi port pe).
EXPOSE 8501
# Container start hote hi ye command chalega - dashboard turant
# accessible hoga bina manually kuch chalaye.
# --server.address=0.0.0.0 zaroori hai (localhost nahi) - warna
# container ke bahar se (host machine se) dashboard access nahi hoga,
# sirf container ke andar se hi dikhega.
CMD streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
