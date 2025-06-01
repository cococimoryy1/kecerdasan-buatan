import numpy as np
import matplotlib.pyplot as plt
import random

"""
PSEUDOCODE ALGORITMA PSO:

1. INISIALISASI
   - Set parameter: w (inersia), c1 (kognitif), c2 (sosial)
   - Set jumlah partikel, iterasi maksimum, batas pencarian
   - Inisialisasi posisi dan kecepatan partikel secara acak
   - Set personal best dan global best

2. UNTUK setiap iterasi:
   a. UNTUK setiap partikel:
      - Hitung nilai fitness f(x) = x²
      - Update personal best jika fitness lebih baik
   b. Update global best dari semua personal best
   c. UNTUK setiap partikel:
      - Update kecepatan: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
      - Update posisi: x = x + v
      - Batasi posisi dalam range yang ditentukan
   d. Simpan global best untuk plotting

3. RETURN global best position dan value
"""

class PSO:
    def __init__(self, num_particles=10, max_iterations=50, w=0.5, c1=1.5, c2=1.5, 
                 x_min=-10, x_max=10):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w  # inersia
        self.c1 = c1  # koefisien kognitif
        self.c2 = c2  # koefisien sosial
        self.x_min = x_min
        self.x_max = x_max
        
        # Inisialisasi partikel
        self.positions = np.random.uniform(x_min, x_max, num_particles)
        self.velocities = np.random.uniform(-1, 1, num_particles)
        
        # Personal best
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.array([self.objective_function(x) for x in self.positions])
        
        # Global best
        best_particle_index = np.argmin(self.personal_best_values)
        self.global_best_position = self.personal_best_positions[best_particle_index]
        self.global_best_value = self.personal_best_values[best_particle_index]
        
        # Untuk tracking
        self.best_values_history = []
        
    def objective_function(self, x):
        """Fungsi objektif: f(x) = x²"""
        return x**2
    
    def optimize(self):
        print("=== PARTICLE SWARM OPTIMIZATION ===")
        print(f"Fungsi objektif: f(x) = x²")
        print(f"Jumlah partikel: {self.num_particles}")
        print(f"Iterasi maksimum: {self.max_iterations}")
        print(f"Parameter - w: {self.w}, c1: {self.c1}, c2: {self.c2}")
        print(f"Batas pencarian: [{self.x_min}, {self.x_max}]")
        print("-" * 50)
        
        for iteration in range(self.max_iterations):
            # Update personal best
            for i in range(self.num_particles):
                current_value = self.objective_function(self.positions[i])
                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.positions[i]
            
            # Update global best
            best_particle_index = np.argmin(self.personal_best_values)
            if self.personal_best_values[best_particle_index] < self.global_best_value:
                self.global_best_value = self.personal_best_values[best_particle_index]
                self.global_best_position = self.personal_best_positions[best_particle_index]
            
            # Simpan untuk plotting
            self.best_values_history.append(self.global_best_value)
            
            # Update velocities dan positions
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()
                
                # Update velocity
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = (self.w * self.velocities[i] + 
                                    cognitive_component + social_component)
                
                # Update position
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # Batasi posisi dalam range
                self.positions[i] = np.clip(self.positions[i], self.x_min, self.x_max)
            
            # Print progress setiap 10 iterasi
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iterasi {iteration + 1:2d}: Best Value = {self.global_best_value:.6f}, "
                      f"Best Position = {self.global_best_position:.6f}")
        
        return self.global_best_position, self.global_best_value
    
    def plot_convergence(self):
        """Plot grafik konvergensi"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.best_values_history) + 1), self.best_values_history, 'b-', linewidth=2)
        plt.title('Konvergensi PSO untuk f(x) = x²', fontsize=14, fontweight='bold')
        plt.xlabel('Iterasi', fontsize=12)
        plt.ylabel('Nilai Terbaik', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, len(self.best_values_history))
        plt.ylim(0, max(self.best_values_history[0], 1))
        
        # Tambahkan informasi pada plot
        plt.text(0.7, 0.8, f'Minimum: {self.global_best_value:.6f}\nPosisi: {self.global_best_position:.6f}', 
                transform=plt.gca().transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def plot_function_and_solution(self):
        """Plot fungsi objektif dan solusi yang ditemukan"""
        x = np.linspace(self.x_min, self.x_max, 1000)
        y = x**2
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
        plt.plot(self.global_best_position, self.global_best_value, 'ro', 
                markersize=10, label=f'Minimum ditemukan\n({self.global_best_position:.6f}, {self.global_best_value:.6f})')
        plt.plot(0, 0, 'g*', markersize=15, label='Minimum teoritis (0, 0)')
        
        plt.title('Fungsi f(x) = x² dan Solusi PSO', fontsize=14, fontweight='bold')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(0, 20)
        plt.tight_layout()
        plt.show()

# Jalankan optimasi PSO
if __name__ == "__main__":
    # Inisialisasi PSO dengan parameter yang ditentukan
    pso = PSO(num_particles=10, max_iterations=50, w=0.5, c1=1.5, c2=1.5, 
              x_min=-10, x_max=10)
    
    # Jalankan optimasi
    best_position, best_value = pso.optimize()
    
    # Cetak hasil akhir
    print("\n" + "="*50)
    print("HASIL OPTIMASI PSO")
    print("="*50)
    print(f"Nilai minimum yang ditemukan: {best_value:.8f}")
    print(f"Posisi x terbaik: {best_position:.8f}")
    print(f"Error dari minimum teoritis (0): {abs(best_position):.8f}")
    print("="*50)
    
    # Buat grafik
    print("\nMembuat grafik konvergensi...")
    pso.plot_convergence()
    
    print("Membuat grafik fungsi dan solusi...")
    pso.plot_function_and_solution()