import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sympy
import os
import math  # добавлено для sqrt и sin/cos в энергии кандидатов

# ================== Конфигурация ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

MIN_BITS = 12
MAX_BITS = 16
MAX_P_VAL = 2 ** MAX_BITS
MAX_N_VAL = 2 ** (MAX_BITS * 2)

DATASET_FILE = "rsa_massive_test.pt"

DT = 0.1
NUM_STEPS = 800
BATCH_SIZE = 16
GRAD_NORM_THRESH = 1.0
NUM_ATTEMPTS = 1000

NUM_MIXTURES = 20
HIDDEN_DIM = 512

TOP_K = 5

# ================== Генерация датасета ==================
def generate_dataset(num_samples, min_bits, max_bits, file_path):
    if os.path.exists(file_path):
        print(f"Датасет уже существует: {file_path}")
        return
    print("Генерация датасета...")
    data = []
    for _ in range(num_samples):
        bits = np.random.randint(min_bits, max_bits + 1)
        p = sympy.randprime(2**(bits-1), 2**bits)
        q = sympy.randprime(2**(bits-1), 2**bits)
        while p == q:
            q = sympy.randprime(2**(bits-1), 2**bits)
        n = p * q
        data.append((int(p), int(q), int(n)))
    data = torch.tensor(data, dtype=torch.float64)
    torch.save(data, file_path)
    print(f"Датасет сохранён: {file_path}, размер: {len(data)}")

class RSADataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path, weights_only=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        p, q, n = self.data[idx]
        return p, q, n

# ================== MDN модель ==================
class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=HIDDEN_DIM, num_mixtures=NUM_MIXTURES):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, num_mixtures)
        self.sigma_head = nn.Linear(hidden_dim, num_mixtures)
        self.pi_head = nn.Linear(hidden_dim, num_mixtures)

    def forward(self, x):
        h = self.hidden(x)
        mu = torch.sigmoid(self.mu_head(h))
        sigma = torch.exp(self.sigma_head(h)) + 1e-4
        pi = torch.softmax(self.pi_head(h), dim=-1)
        return mu, sigma, pi

# ================== Нормализованный ландшафт ==================
class CryptoLandscape:
    def __init__(self, n):
        self.n = n.view(-1, 1)

    def energy(self, x, y, omega_x=None, omega_y=None, correction_active=1.0):
        # Нормализация: делим на sqrt(n), чтобы уравновесить градиенты для больших n
        base = ((self.n - x * y) / torch.sqrt(self.n)) ** 2
        # Усиливаем штраф за нецелость, чтобы сетка была "видна"
        int_pen = 100.0 * (torch.sin(np.pi * x)**2 + torch.sin(np.pi * y)**2)
        energy = base + int_pen
        if omega_x is not None and omega_y is not None:
            energy = energy + correction_active * (omega_x * x + omega_y * y)
        return energy

    def gradient(self, x, y, create_graph=False, **kwargs):
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        E = self.energy(x, y, **kwargs)
        grad_x, grad_y = torch.autograd.grad(E.sum(), (x, y), create_graph=create_graph)
        return grad_x, grad_y

# ================== MDN Оракул ==================
class MDNOracle:
    def __init__(self, model_p_path, model_q_path, device):
        self.model_p = MixtureDensityNetwork().to(device)
        self.model_q = MixtureDensityNetwork().to(device)
        self.model_p.load_state_dict(torch.load(model_p_path, map_location=device, weights_only=True))
        self.model_q.load_state_dict(torch.load(model_q_path, map_location=device, weights_only=True))
        self.model_p.eval()
        self.model_q.eval()
        self.device = device
        self.max_p_val = MAX_P_VAL
        self.max_n_val = MAX_N_VAL

    def get_top_candidates(self, n, x, y, top_k=TOP_K, num_attempts=NUM_ATTEMPTS):
        batch_size = n.shape[0]
        n_np = n.cpu().numpy().flatten()
        all_p = []
        all_q = []
        all_energies = []
        for i in range(batch_size):
            n_val = n_np[i]
            n_norm = torch.tensor([[n_val / self.max_n_val]], device=self.device)
            with torch.no_grad():
                mu_p, sigma_p, pi_p = self.model_p(n_norm)
                mu_q, sigma_q, pi_q = self.model_q(n_norm)
            mu_p = mu_p.cpu().numpy().flatten()
            pi_p = pi_p.cpu().numpy().flatten()
            mu_q = mu_q.cpu().numpy().flatten()
            pi_q = pi_q.cpu().numpy().flatten()
            sigma_p = sigma_p.cpu().numpy().flatten()
            sigma_q = sigma_q.cpu().numpy().flatten()
            candidates_p = []
            candidates_q = []
            energies = []
            for _ in range(num_attempts):
                comp_p = np.random.choice(NUM_MIXTURES, p=pi_p)
                comp_q = np.random.choice(NUM_MIXTURES, p=pi_q)
                p_pred = np.random.normal(mu_p[comp_p], sigma_p[comp_p])
                q_pred = np.random.normal(mu_q[comp_q], sigma_q[comp_q])
                p_pred = np.clip(p_pred, 0, 1) * self.max_p_val
                q_pred = np.clip(q_pred, 0, 1) * self.max_p_val
                if p_pred > q_pred:
                    p_pred, q_pred = q_pred, p_pred
                E_cand = (n_val - p_pred * q_pred)**2 + 100.0 * (np.sin(np.pi * p_pred)**2 + np.sin(np.pi * q_pred)**2)
                candidates_p.append(p_pred)
                candidates_q.append(q_pred)
                energies.append(E_cand)
            sorted_idx = np.argsort(energies)[:top_k]
            top_p = [candidates_p[idx] for idx in sorted_idx]
            top_q = [candidates_q[idx] for idx in sorted_idx]
            top_energies = [energies[idx] for idx in sorted_idx]
            all_p.append(top_p)
            all_q.append(top_q)
            all_energies.append(top_energies)
        return all_p, all_q, all_energies

# ================== Симуляция с нормализацией ==================
def simulate_rays(landscape, oracle, p_true, q_true):
    batch_size = landscape.n.shape[0]
    n = landscape.n.detach()

    # Старт
    x = torch.sqrt(n) * (1.0 + 0.2 * torch.randn(batch_size, 1, device=DEVICE))
    y = n / x
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)

    active = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
    positions = [(x.clone().detach(), y.clone().detach())]
    oracle_used = False
    steps_since_last_oracle = 0

    for step in range(NUM_STEPS):
        if not active.any():
            break

        x.requires_grad_(True)
        y.requires_grad_(True)

        grad_x, grad_y = landscape.gradient(x, y, create_graph=False)
        grad_norm = torch.sqrt(grad_x**2 + grad_y**2)

        # Активация оракула
        if (grad_norm.mean() < GRAD_NORM_THRESH or steps_since_last_oracle >= 100) and active.any():
            candidates_p, candidates_q, _ = oracle.get_top_candidates(
                n, x, y, top_k=TOP_K, num_attempts=NUM_ATTEMPTS
            )
            oracle_used = True
            steps_since_last_oracle = 0

            with torch.no_grad():
                for i in range(batch_size):
                    if not active[i]:
                        continue

                    best_energy = float('inf')
                    best_coords = (x[i].item(), y[i].item())
                    n_val_i = n[i].item()

                    for p_cand, q_cand in zip(candidates_p[i], candidates_q[i]):
                        E_cand = ((n_val_i - p_cand * q_cand) / math.sqrt(n_val_i))**2 + 100.0 * (np.sin(np.pi * p_cand)**2 + np.sin(np.pi * q_cand)**2)
                        if E_cand < best_energy:
                            best_energy = E_cand
                            best_coords = (p_cand, q_cand)

                    x[i] = best_coords[0]
                    y[i] = best_coords[1]

            if active[0]:
                print(f"    Step {step}: ORACLE JUMP (Best of Top {TOP_K}), grad_norm = {grad_norm.mean().item():.4f}")
                with torch.no_grad():
                    print(f"      Debug: n={n[0].item():.0f}, jumped to=({x[0].item():.1f},{y[0].item():.1f}), "
                          f"true=({p_true[0].item():.1f},{q_true[0].item():.1f})")
        else:
            steps_since_last_oracle += 1

        # Проверка решения (нейросимволический коллапс)
        with torch.no_grad():
            solved_mask = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
            for i in range(batch_size):
                if not active[i]:
                    continue
                n_val = int(n[i].item())
                x_center = int(round(x[i].item()))
                search_radius = 5000
                start_x = max(2, x_center - search_radius)
                end_x = min(int(math.sqrt(n_val)) + 1, x_center + search_radius)
                for test_x in range(start_x, end_x + 1):
                    if n_val % test_x == 0:
                        x[i] = float(test_x)
                        y[i] = float(n_val // test_x)
                        solved_mask[i] = True
                        break
            if solved_mask.any():
                active = active & ~solved_mask
                if not active.any():
                    print(f"    [!!!] ВСЕ КЛЮЧИ В БАТЧЕ УСПЕШНО ВЗЛОМАНЫ НА ШАГЕ {step}!")
                    break

        # Градиентный спуск для активных
        if active.any():
            E = landscape.energy(x, y)
            grad_x, grad_y = torch.autograd.grad(E.sum(), (x, y), create_graph=False)
            grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            step_x = -DT * grad_x / grad_norm
            step_y = -DT * grad_y / grad_norm
            with torch.no_grad():
                x[active] += step_x[active]
                y[active] += step_y[active]

        positions.append((x.detach().clone(), y.detach().clone()))

    # Финальное расстояние с учётом симметрии
    if positions:
        final_x, final_y = positions[-1]
        dist1 = torch.sqrt((final_x - p_true)**2 + (final_y - q_true)**2)
        dist2 = torch.sqrt((final_x - q_true)**2 + (final_y - p_true)**2)
        dist = torch.minimum(dist1, dist2).mean().item()
    else:
        dist = 0.0

    return positions, dist, oracle_used


# ================== Тестирование ==================
def test():
    print(f"Используется устройство: {DEVICE}")
    generate_dataset(200, MIN_BITS, MAX_BITS, DATASET_FILE)
    dataset = RSADataset(DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    oracle = MDNOracle("mdn_p_massive.pth", "mdn_q_massive.pth", DEVICE)

    distances = []
    oracle_used_count = 0

    for idx, (p_batch, q_batch, n_batch) in enumerate(dataloader):
        p_batch = p_batch.to(DEVICE).view(-1,1)
        q_batch = q_batch.to(DEVICE).view(-1,1)
        n_batch = n_batch.to(DEVICE).view(-1,1)

        landscape = CryptoLandscape(n_batch)
        positions, dist, used = simulate_rays(landscape, oracle, p_batch, q_batch)

        distances.append(dist)
        if used:
            oracle_used_count += 1

        if idx % 5 == 0:
            print(f"Batch {idx}: Dist = {dist:.2f}, Oracle used = {used}")

    avg_dist = np.mean(distances)
    print(f"\nСреднее расстояние до истинных (p,q): {avg_dist:.2f}")
    print(f"Оракул использовался в {oracle_used_count} из {len(dataloader)} батчей ({oracle_used_count/len(dataloader)*100:.1f}%)")

    # Визуализация последней траектории
    pos_x = [p[0][0].item() for p in positions]
    pos_y = [p[1][0].item() for p in positions]
    plt.figure(figsize=(6,6))
    plt.plot(pos_x, pos_y, 'b-', alpha=0.6)
    plt.scatter(pos_x[0], pos_y[0], c='g', s=50, label='Start')
    plt.scatter(pos_x[-1], pos_y[-1], c='r', s=50, label='End')
    true_p, true_q = p_batch[0].item(), q_batch[0].item()
    if abs(pos_x[-1] - true_q) < abs(pos_x[-1] - true_p):
        true_p, true_q = true_q, true_p
    plt.scatter(true_p, true_q, c='k', marker='*', s=200, label='True')
    plt.xlabel('p')
    plt.ylabel('q')
    plt.legend()
    plt.title(f'Trajectory with MDN correction (Dist={dist:.2f})')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test()
