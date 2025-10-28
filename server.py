from fastapi import FastAPI, APIRouter, Depends, HTTPException, UploadFile, File, status, Form, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import base64
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from google.cloud import storage
import datetime as dt
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import logging
import uuid
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path
import jwt
from passlib.context import CryptContext
# import magic  # Temporarily disabled
from PIL import Image
import io
import socketio

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Google Cloud Storage setup - Make optional for Railway deployment
try:
    # Check if GCS credentials are available
    if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') or os.path.exists('/app/backend/service-account.json'):
        if os.path.exists('/app/backend/service-account.json'):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/backend/service-account.json'
        storage_client = storage.Client()
        bucket_name = os.environ.get('GOOGLE_CLOUD_BUCKET', 'default-bucket')
        bucket = storage_client.bucket(bucket_name)
        GCS_AVAILABLE = True
        logger.info("✅ Google Cloud Storage initialized successfully")
    else:
        storage_client = None
        bucket = None
        GCS_AVAILABLE = False
        logger.warning("⚠️ Google Cloud Storage disabled - no credentials found")
except Exception as e:
    storage_client = None
    bucket = None
    GCS_AVAILABLE = False
    logger.error(f"⚠️ Google Cloud Storage initialization failed: {e}")

# Authentication Configuration
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'changeme')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

# JWT Models
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenRefresh(BaseModel):
    refresh_token: str

class LoginRequest(BaseModel):
    username: str
    password: str

# JWT Utility Functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    })
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user with username/password"""
    if not secrets.compare_digest(username, ADMIN_USERNAME):
        return False
    if not secrets.compare_digest(password, ADMIN_PASSWORD):
        return False
    return True

def verify_token(token: str, token_type: str = "access") -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Check token type
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.now(timezone.utc) > datetime.fromtimestamp(exp, tz=timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = verify_token(credentials.credentials, "access")
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return username
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Create FastAPI app
app = FastAPI(title="Dental Clinic Signage API")
api_router = APIRouter(prefix="/api")

# Socket.IO setup
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    logger=True
)
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app)

# Models
class Location(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    slug: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LocationCreate(BaseModel):
    name: str
    slug: str

class Patient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location_id: str
    name: str
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PatientCreate(BaseModel):
    name: str
    note: Optional[str] = None

class Staff(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location_id: str
    role: str  # "doctor" or "nurse"
    name: str
    photo_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StaffCreate(BaseModel):
    role: str
    name: str

class LiveState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location_id: str
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    nurse_id: Optional[str] = None
    music_url: Optional[str] = None
    music_muted: bool = True
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LiveStateUpdate(BaseModel):
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    nurse_id: Optional[str] = None

class MusicRequest(BaseModel):
    youtube_url: Optional[str] = None
    muted: Optional[bool] = None

class LiveStateResponse(BaseModel):
    patient: Optional[Patient] = None
    doctor: Optional[Staff] = None
    nurse: Optional[Staff] = None
    music_url: Optional[str] = None
    music_muted: bool = True
    updated_at: datetime

# Helper functions
def prepare_for_mongo(data):
    """Convert datetime objects to ISO strings for MongoDB storage"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
    return data

def parse_from_mongo(item):
    """Convert ISO strings back to datetime objects from MongoDB"""
    if isinstance(item, dict):
        for key, value in item.items():
            if key in ['created_at', 'updated_at'] and isinstance(value, str):
                try:
                    item[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass
    return item

async def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    # Check file size (5MB limit)
    content = await file.read()
    await file.seek(0)  # Reset file pointer
    
    if len(content) > 5 * 1024 * 1024:  # 5MB
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB")
    
    # Check file type using content type header (simplified validation)
    if file.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed")
    
    # Validate image using PIL
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    return True

async def upload_to_gcs(file: UploadFile, filename: str) -> str:
    """Upload file to Google Cloud Storage"""
    blob = bucket.blob(filename)
    content = await file.read()
    await file.seek(0)  # Reset file pointer
    
    blob.upload_from_string(content, content_type=file.content_type)
    
    # Generate a signed URL that's valid for 1 year
    expiration_time = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=365)
    signed_url = blob.generate_signed_url(expiration=expiration_time, method='GET')
    
    return signed_url

# Socket.IO events
@sio.event
async def connect(sid, environ):
    logger.info(f"Client {sid} connected")

@sio.event
async def disconnect(sid):
    logger.info(f"Client {sid} disconnected")

@sio.event 
async def join_room(sid, data):
    location_slug = data.get('locationSlug')
    if location_slug:
        room = f"room:{location_slug}"
        await sio.enter_room(sid, room)
        
        # Send current live state
        live_state = await get_live_state_data(location_slug)
        if live_state:
            await sio.emit('live:init', live_state.dict(), room=sid)

async def get_live_state_data(location_slug: str) -> Optional[LiveStateResponse]:
    """Get complete live state data for a location"""
    # Get location
    location = await db.locations.find_one({"slug": location_slug})
    if not location:
        return None
    
    # Get live state
    live_state = await db.live_states.find_one({"location_id": location["id"]})
    if not live_state:
        return LiveStateResponse(
            patient=None,
            doctor=None,
            nurse=None,
            updated_at=datetime.now(timezone.utc)
        )
    
    parse_from_mongo(live_state)
    
    # Get patient, doctor, nurse data
    patient = None
    doctor = None
    nurse = None
    
    if live_state.get('patient_id'):
        patient_data = await db.patients.find_one({"id": live_state['patient_id']})
        if patient_data:
            parse_from_mongo(patient_data)
            patient = Patient(**patient_data)
    
    if live_state.get('doctor_id'):
        doctor_data = await db.staff.find_one({"id": live_state['doctor_id'], "role": "doctor"})
        if doctor_data:
            parse_from_mongo(doctor_data)
            doctor = Staff(**doctor_data)
    
    if live_state.get('nurse_id'):
        nurse_data = await db.staff.find_one({"id": live_state['nurse_id'], "role": "nurse"})
        if nurse_data:
            parse_from_mongo(nurse_data)
            nurse = Staff(**nurse_data)
    
    return LiveStateResponse(
        patient=patient,
        doctor=doctor,
        nurse=nurse,
        music_url=live_state.get('music_url'),
        music_muted=live_state.get('music_muted', True),
        updated_at=live_state.get('updated_at', datetime.now(timezone.utc))
    )

# API Routes

# Authentication Routes
@api_router.post("/auth/login", response_model=Token)
async def login(login_data: LoginRequest):
    """Login and receive JWT tokens"""
    if not authenticate_user(login_data.username, login_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": login_data.username})
    refresh_token = create_refresh_token(data={"sub": login_data.username})
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    )

@api_router.post("/auth/refresh", response_model=Token)
async def refresh_token(token_data: TokenRefresh):
    """Refresh access token using refresh token"""
    try:
        payload = verify_token(token_data.refresh_token, "refresh")
        username = payload.get("sub")
        
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new tokens
        access_token = create_access_token(data={"sub": username})
        refresh_token = create_refresh_token(data={"sub": username})
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@api_router.post("/auth/logout")
async def logout(current_user: str = Depends(get_current_user)):
    """Logout endpoint (JWT tokens are stateless, so this is mainly for client-side cleanup)"""
    return {"message": "Logged out successfully"}

@api_router.get("/auth/me")
async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """Get current authenticated user information"""
    return {"username": current_user, "authenticated": True}

# Locations
@api_router.get("/locations", response_model=List[Location])
async def get_locations():
    locations = await db.locations.find().to_list(100)
    return [Location(**parse_from_mongo(loc)) for loc in locations]

@api_router.post("/locations", response_model=Location)
async def create_location(location: LocationCreate, _: str = Depends(get_current_user)):
    location_dict = location.dict()
    location_obj = Location(**location_dict)
    
    # Check if slug already exists
    existing = await db.locations.find_one({"slug": location.slug})
    if existing:
        raise HTTPException(status_code=400, detail="Location slug already exists")
    
    await db.locations.insert_one(prepare_for_mongo(location_obj.dict()))
    
    # Create initial live state
    live_state = LiveState(location_id=location_obj.id)
    await db.live_states.insert_one(prepare_for_mongo(live_state.dict()))
    
    return location_obj

# Patients
@api_router.get("/patients", response_model=List[Patient])
async def get_patients(location: str):
    # Get location by slug
    location_doc = await db.locations.find_one({"slug": location})
    if not location_doc:
        raise HTTPException(status_code=404, detail="Location not found")
    
    patients = await db.patients.find({"location_id": location_doc["id"]}).to_list(100)
    return [Patient(**parse_from_mongo(p)) for p in patients]

@api_router.post("/patients", response_model=Patient)
async def create_patient(
    locationSlug: str = Form(...),
    name: str = Form(...),
    note: str = Form(None),
    _: str = Depends(get_current_user)
):
    # Get location by slug
    location_doc = await db.locations.find_one({"slug": locationSlug})
    if not location_doc:
        raise HTTPException(status_code=404, detail="Location not found")
    
    patient = Patient(
        location_id=location_doc["id"],
        name=name.strip(),
        note=note.strip() if note else None
    )
    
    await db.patients.insert_one(prepare_for_mongo(patient.dict()))
    return patient

@api_router.delete("/patients/{patient_id}")
async def delete_patient(patient_id: str, _: str = Depends(get_current_user)):
    result = await db.patients.delete_one({"id": patient_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Remove from live state if active
    await db.live_states.update_many(
        {"patient_id": patient_id},
        {"$unset": {"patient_id": ""}, "$set": {"updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    return {"message": "Patient deleted"}

# Staff
@api_router.get("/staff", response_model=List[Staff])
async def get_staff(location: str, role: str):
    # Get location by slug
    location_doc = await db.locations.find_one({"slug": location})
    if not location_doc:
        raise HTTPException(status_code=404, detail="Location not found")
    
    if role not in ["doctor", "nurse"]:
        raise HTTPException(status_code=400, detail="Role must be 'doctor' or 'nurse'")
    
    staff = await db.staff.find({"location_id": location_doc["id"], "role": role}).to_list(100)
    return [Staff(**parse_from_mongo(s)) for s in staff]

@api_router.post("/staff", response_model=Staff)
async def create_staff(
    locationSlug: str = Form(...),
    role: str = Form(...),
    name: str = Form(...),
    _: str = Depends(get_current_user)
):
    # Get location by slug
    location_doc = await db.locations.find_one({"slug": locationSlug})
    if not location_doc:
        raise HTTPException(status_code=404, detail="Location not found")
    
    if role not in ["doctor", "nurse"]:
        raise HTTPException(status_code=400, detail="Role must be 'doctor' or 'nurse'")
    
    staff = Staff(
        location_id=location_doc["id"],
        role=role,
        name=name.strip()
    )
    
    await db.staff.insert_one(prepare_for_mongo(staff.dict()))
    return staff

@api_router.post("/staff/{staff_id}/photo", response_model=Staff)
async def upload_staff_photo(
    staff_id: str,
    file: UploadFile = File(...),
    _: str = Depends(get_current_user)
):
    # Get staff member
    staff_doc = await db.staff.find_one({"id": staff_id})
    if not staff_doc:
        raise HTTPException(status_code=404, detail="Staff member not found")
    
    # Validate image
    await validate_image_file(file)
    
    # Generate unique filename
    file_extension = file.filename.split('.')[-1].lower()
    filename = f"staff/{staff_id}_{uuid.uuid4()}.{file_extension}"
    
    # Upload to GCS
    photo_url = await upload_to_gcs(file, filename)
    
    # Update staff record
    await db.staff.update_one(
        {"id": staff_id},
        {"$set": {"photo_url": photo_url, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    # Get updated staff
    updated_staff = await db.staff.find_one({"id": staff_id})
    return Staff(**parse_from_mongo(updated_staff))

@api_router.delete("/staff/{staff_id}")
async def delete_staff(staff_id: str, _: str = Depends(get_current_user)):
    staff_doc = await db.staff.find_one({"id": staff_id})
    if not staff_doc:
        raise HTTPException(status_code=404, detail="Staff member not found")
    
    # Delete from database
    await db.staff.delete_one({"id": staff_id})
    
    # Remove from live state
    role = staff_doc["role"]
    field_name = f"{role}_id"
    await db.live_states.update_many(
        {field_name: staff_id},
        {"$unset": {field_name: ""}, "$set": {"updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    return {"message": "Staff member deleted"}

# Live State
@api_router.get("/live/{location_slug}", response_model=LiveStateResponse)
async def get_live_state(location_slug: str):
    live_state = await get_live_state_data(location_slug)
    if not live_state:
        raise HTTPException(status_code=404, detail="Location not found")
    return live_state

@api_router.post("/live/{location_slug}")
async def update_live_state(
    location_slug: str,
    update: LiveStateUpdate,
    _: str = Depends(get_current_user)
):
    # Get location
    location_doc = await db.locations.find_one({"slug": location_slug})
    if not location_doc:
        raise HTTPException(status_code=404, detail="Location not found")
    
    location_id = location_doc["id"]
    
    # Get or create live state
    live_state = await db.live_states.find_one({"location_id": location_id})
    if not live_state:
        live_state = LiveState(location_id=location_id)
        await db.live_states.insert_one(prepare_for_mongo(live_state.dict()))
    
    # Update fields
    update_dict = {"updated_at": datetime.now(timezone.utc).isoformat()}
    
    if update.patient_id is not None:
        if update.patient_id == "":
            update_dict["$unset"] = {"patient_id": ""}
        else:
            update_dict["patient_id"] = update.patient_id
    
    if update.doctor_id is not None:
        if update.doctor_id == "":
            update_dict["$unset"] = update_dict.get("$unset", {})
            update_dict["$unset"]["doctor_id"] = ""
        else:
            update_dict["doctor_id"] = update.doctor_id
    
    if update.nurse_id is not None:
        if update.nurse_id == "":
            update_dict["$unset"] = update_dict.get("$unset", {})
            update_dict["$unset"]["nurse_id"] = ""
        else:
            update_dict["nurse_id"] = update.nurse_id
    
    # Handle unset operations
    if "$unset" in update_dict:
        unset_dict = update_dict.pop("$unset")
        await db.live_states.update_one(
            {"location_id": location_id},
            {"$unset": unset_dict, "$set": update_dict}
        )
    else:
        await db.live_states.update_one(
            {"location_id": location_id},
            {"$set": update_dict}
        )
    
    # Get updated live state and broadcast
    updated_live_state = await get_live_state_data(location_slug)
    room = f"room:{location_slug}"
    await sio.emit('live:update', updated_live_state.dict(), room=room)
    
    return {"message": "Live state updated"}

# Clear operations
@api_router.post("/live/{location_slug}/clear-name")
async def clear_patient_name(location_slug: str, _: str = Depends(get_current_user)):
    update = LiveStateUpdate(patient_id="")
    await update_live_state(location_slug, update, _)
    return {"message": "Patient name cleared"}

@api_router.post("/live/{location_slug}/reset-photos")
async def reset_photos(location_slug: str, _: str = Depends(get_current_user)):
    update = LiveStateUpdate(doctor_id="", nurse_id="")
    await update_live_state(location_slug, update, _)
    return {"message": "Photos reset"}

@api_router.post("/live/{location_slug}/set-music")
async def set_music(
    location_slug: str,
    music_request: MusicRequest,
    _: str = Depends(get_current_user)
):
    # Get location
    location_doc = await db.locations.find_one({"slug": location_slug})
    if not location_doc:
        raise HTTPException(status_code=404, detail="Location not found")
    
    location_id = location_doc["id"]
    
    # Get or create live state
    live_state = await db.live_states.find_one({"location_id": location_id})
    if not live_state:
        live_state = LiveState(location_id=location_id)
        await db.live_states.insert_one(prepare_for_mongo(live_state.dict()))
    
    # Update music state
    update_dict = {"updated_at": datetime.now(timezone.utc).isoformat()}
    
    if music_request.youtube_url:
        update_dict["music_url"] = music_request.youtube_url
    
    if music_request.muted is not None:
        update_dict["music_muted"] = music_request.muted
    
    await db.live_states.update_one(
        {"location_id": location_id},
        {"$set": update_dict}
    )
    
    # Get updated live state and broadcast
    updated_live_state = await get_live_state_data(location_slug)
    room = f"room:{location_slug}"
    await sio.emit('music:update', {
        'music_url': updated_live_state.music_url,
        'music_muted': updated_live_state.music_muted
    }, room=room)
    
    return {"message": "Music updated"}

@api_router.post("/live/{location_slug}/clear-music")
async def clear_music(
    location_slug: str,
    _: str = Depends(get_current_user)
):
    # Get location
    location_doc = await db.locations.find_one({"slug": location_slug})
    if not location_doc:
        raise HTTPException(status_code=404, detail="Location not found")
    
    location_id = location_doc["id"]
    
    # Update music state
    update_dict = {
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.live_states.update_one(
        {"location_id": location_id},
        {"$set": update_dict, "$unset": {"music_url": ""}}
    )
    
    # Get updated live state and broadcast
    updated_live_state = await get_live_state_data(location_slug)
    room = f"room:{location_slug}"
    await sio.emit('music:update', {
        'music_url': None,
        'music_muted': updated_live_state.music_muted
    }, room=room)
    
    return {"message": "Music cleared"}

# Health check
@api_router.get("/")
async def root():
    return {"message": "Dental Clinic Signage API"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}

# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data
@app.on_event("startup")
async def startup_event():
    """Initialize default location if none exists"""
    try:
        # Check if SAKO location exists
        existing = await db.locations.find_one({"slug": "sako"})
        if not existing:
            # Create SAKO location
            sako_location = Location(
                name="PUTIH Dental SAKO",
                slug="sako"
            )
            await db.locations.insert_one(prepare_for_mongo(sako_location.dict()))
            
            # Create initial live state
            live_state = LiveState(location_id=sako_location.id)
            await db.live_states.insert_one(prepare_for_mongo(live_state.dict()))
            
            # Create default patient
            default_patient = Patient(
                location_id=sako_location.id,
                name="Ms. Melinda",
                note="Default patient for display"
            )
            await db.patients.insert_one(prepare_for_mongo(default_patient.dict()))
            
            # Set default patient as live
            await db.live_states.update_one(
                {"location_id": sako_location.id},
                {"$set": {"patient_id": default_patient.id, "updated_at": datetime.now(timezone.utc).isoformat()}}
            )
            
            logger.info("Initialized SAKO location with default data")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
