from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from utils.model import MockForcastModel
from utils.cron_job import CronJob

app = FastAPI()

config = {
    "history_endpoint": "",
    "forecase_model": "uit-aqi" 
}

# Khai báo thư mục chứa template (HTML)
templates = Jinja2Templates(directory="templates")
# Mount thư mục static nếu bạn muốn tách file css/js riêng (tùy chọn)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Render file HTML. Bạn có thể truyền dữ liệu từ Python sang HTML tại đây qua biến context
    return templates.TemplateResponse("index.html", {"request": request})

# API lấy dữ liệu (nếu bạn muốn tách data ra khỏi HTML sau này)
@app.get("/api/air-quality")
async def get_air_quality():
    return {
        "location": "Hà Nội, Việt Nam",
        "aqi": 156,
        "status": "Kém"
    }