import numpy as np
import librosa
import librosa.display
import librosa.feature
from pygame import mixer
import taichi as ti

filename = r'audio\audio2.mp3'

start = 18

def get_spectrum(filename):
    time_series, sample_rate = librosa.load(filename, sr=44100, mono=True, offset=start, res_type='kaiser_best')
    spec = librosa.feature.melspectrogram(time_series, sr=44100, n_fft=16384, n_mels=1024, hop_length=735)[:128, :]
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db

ti.init(arch=ti.gpu)  # Try to run on GPU
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 1e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 6
E = 400

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


# intensity in dB is usually between -30 and 0
spec_db = get_spectrum(filename)
spec_db = np.gradient(np.gradient(spec_db, axis=1), axis=1)
print(spec_db.shape, np.min(spec_db), np.max(spec_db))
n_frames = spec_db.shape[1]

spectrum = ti.field(dtype=ti.f32, shape=spec_db.shape)
frame = ti.field(dtype=int, shape=())



@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        gravity = max(-0.02, min(spectrum[i - bound, frame[None]], 0.02)) * 18000.0 - 10.0
        grid_v[i, j][1] += dt * gravity  # gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1


spectrum.from_numpy(spec_db)
init()
frame[None] = 0
gui = ti.GUI("MPM88", res=512, background_color=0x112F41)
mixer.init()
mixer.music.load(filename)
mixer.music.play(start=start)
mixer.music.pause()

for i in range(n_frames):
    for s in range(int(2e-3 // dt)):
        substep()
    
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
    #gui.show(f'frames_musicmpm88/{i:06d}.png')
    mixer.music.unpause()
    frame[None] += 1