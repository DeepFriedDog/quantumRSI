"""
Quantum RSI - Core Quantum Model Implementation

This module implements the quantum mechanics-inspired model for RSI in trading.
It models RSI as a quantum particle in a potential well, with trend reversals
analogous to quantum tunneling events.
"""

import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any


class QuantumWellPotential:
    """
    Implements a double-well potential that models bullish (high RSI) and
    bearish (low RSI) market states separated by a barrier.
    """
    
    def __init__(self, v0: float = 1.0, a: float = 0.4, symmetric: bool = True):
        """
        Initialize the potential well parameters.
        
        Args:
            v0: Strength of the potential (affects barrier height)
            a: Position of the minima (wells), e.g., 0.4 corresponds to RSI ~30/70
            symmetric: If True, wells are symmetric; otherwise can be asymmetric
        """
        self.v0 = v0
        self.a = a
        self.symmetric = symmetric
        # Track the dynamic barrier adjustments from Maxwell's Demon
        self.barrier_adjustment = 0.0
    
    def v(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the potential at position x.
        
        V(x) = V0 * (x^2 - a^2)^2 + barrier_adjustment
        
        Args:
            x: Normalized position array (RSI mapped to [-1, 1])
            
        Returns:
            Potential energy values at positions x
        """
        # Base double-well potential: V(x) = V0 * (x^2 - a^2)^2
        potential = self.v0 * (x**2 - self.a**2)**2
        
        # Apply dynamic barrier adjustment from Maxwell's Demon
        # This will affect the region near x=0 (the barrier)
        barrier_effect = self.barrier_adjustment * np.exp(-(x**2) / 0.01)
        
        return potential + barrier_effect
    
    def adjust_barrier(self, adjustment: float):
        """
        Dynamically adjust the barrier height based on market information.
        
        Args:
            adjustment: Positive values raise the barrier (harder to reverse),
                       negative values lower it (easier reversal)
        """
        self.barrier_adjustment = adjustment
    
    def reset_barrier(self):
        """Reset any dynamic barrier adjustments."""
        self.barrier_adjustment = 0.0
    
    def plot(self, x_range: Tuple[float, float] = (-1, 1), num_points: int = 1000):
        """
        Plot the potential well for visualization.
        
        Args:
            x_range: Range of x values to plot
            num_points: Number of points to evaluate
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        potential = self.v(x)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, potential)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Mark potential wells and barriers
        plt.plot(-self.a, 0, 'ro', label='Bearish Well (Low RSI)')
        plt.plot(self.a, 0, 'go', label='Bullish Well (High RSI)')
        plt.plot(0, self.v0 * self.a**4, 'bo', label='Barrier (Neutral RSI)')
        
        # Add RSI scale
        ax2 = plt.twiny()
        rsi_values = np.array([0, 30, 50, 70, 100])
        x_positions = (rsi_values - 50) / 50  # Transform to x scale
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(rsi_values)
        ax2.set_xlabel('RSI Value')
        
        plt.title('Quantum Potential Well Model for RSI')
        plt.xlabel('Normalized Position (x)')
        plt.ylabel('Potential Energy V(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()


class SchrodingerSolver:
    """
    Solves the time-dependent Schrödinger equation for a given potential.
    
    This represents the evolution of the RSI "wavefunction" over time.
    """
    
    def __init__(self, potential: QuantumWellPotential, 
                 x_min: float = -1.0, x_max: float = 1.0, 
                 nx: int = 1000, dt: float = 0.01, hbar: float = 1.0, m: float = 1.0):
        """
        Initialize the Schrödinger equation solver.
        
        Args:
            potential: The potential well function
            x_min, x_max: Spatial domain limits
            nx: Number of spatial grid points
            dt: Time step for evolution
            hbar: Reduced Planck constant (kept for generality)
            m: Effective mass parameter (higher = more inertia)
        """
        self.potential = potential
        self.x_min = x_min
        self.x_max = x_max
        self.nx = nx
        self.dt = dt
        self.hbar = hbar
        self.m = m
        
        # Initialize spatial grid
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = (x_max - x_min) / (nx - 1)
        
        # Initialize wavefunction as empty
        self.psi = None
        
    def initialize_wavefunction(self, center: float, width: float = 0.1):
        """
        Initialize a Gaussian wavefunction centered at a specific position.
        
        Args:
            center: Center position of the wavefunction (e.g., initial RSI state)
            width: Width of the Gaussian packet
        """
        # Create a Gaussian wavepacket: psi(x) = exp(-(x-center)^2 / (2*width^2))
        self.psi = np.exp(-(self.x - center)**2 / (2 * width**2))
        
        # Normalize the wavefunction
        norm = np.sqrt(simpson(np.abs(self.psi)**2, self.x))
        self.psi = self.psi / norm
        
    def evolve_step(self):
        """
        Evolve the wavefunction for one time step using the split-operator method.
        """
        if self.psi is None:
            raise ValueError("Wavefunction not initialized. Call initialize_wavefunction first.")
        
        # Step 1: Apply half-step of potential evolution
        self.psi = self.psi * np.exp(-1j * self.dt * self.potential.v(self.x) / (2 * self.hbar))
        
        # Step 2: Go to momentum space (FFT), apply kinetic evolution, and go back (IFFT)
        psi_k = np.fft.fft(self.psi)
        k = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        psi_k = psi_k * np.exp(-1j * self.dt * (self.hbar * k**2) / (2 * self.m))
        self.psi = np.fft.ifft(psi_k)
        
        # Step 3: Apply second half-step of potential evolution
        self.psi = self.psi * np.exp(-1j * self.dt * self.potential.v(self.x) / (2 * self.hbar))
        
        # Ensure normalization (should be preserved by split-operator, but numerical precision)
        norm = np.sqrt(simpson(np.abs(self.psi)**2, self.x))
        self.psi = self.psi / norm
    
    def evolve(self, steps: int) -> np.ndarray:
        """
        Evolve the wavefunction for multiple time steps.
        
        Args:
            steps: Number of time steps to evolve
            
        Returns:
            Probability density after evolution
        """
        for _ in range(steps):
            self.evolve_step()
            
        # Return probability density |ψ|²
        return np.abs(self.psi)**2
    
    def calculate_well_probabilities(self) -> Dict[str, float]:
        """
        Calculate the probability of finding the particle in each well.
        
        Returns:
            Dictionary with probabilities for bearish, neutral, and bullish regions
        """
        if self.psi is None:
            raise ValueError("Wavefunction not initialized")
            
        # Define regions: bearish (x < -a/2), neutral (-a/2 < x < a/2), bullish (x > a/2)
        bearish_mask = self.x < -self.potential.a/2
        bullish_mask = self.x > self.potential.a/2
        neutral_mask = ~bearish_mask & ~bullish_mask
        
        # Calculate probability in each region
        prob_density = np.abs(self.psi)**2
        
        p_bearish = simpson(prob_density[bearish_mask], self.x[bearish_mask]) if np.any(bearish_mask) else 0
        p_neutral = simpson(prob_density[neutral_mask], self.x[neutral_mask]) if np.any(neutral_mask) else 0
        p_bullish = simpson(prob_density[bullish_mask], self.x[bullish_mask]) if np.any(bullish_mask) else 0
        
        return {
            "bearish": p_bearish,
            "neutral": p_neutral,
            "bullish": p_bullish
        }


class QuantumTunneling:
    """
    Calculates tunneling probabilities through the potential barrier (RSI trend reversal).
    """
    
    def __init__(self, potential: QuantumWellPotential, hbar: float = 1.0, m: float = 1.0):
        """
        Initialize the tunneling calculator.
        
        Args:
            potential: The potential well model
            hbar: Reduced Planck constant
            m: Effective mass
        """
        self.potential = potential
        self.hbar = hbar
        self.m = m
    
    def find_turning_points(self, energy: float, x_range: Tuple[float, float] = (-1, 1),
                           num_points: int = 1000) -> Tuple[float, float]:
        """
        Find the classical turning points for a given energy.
        
        Args:
            energy: Energy level to find turning points for
            x_range: Range to search within
            num_points: Resolution of search
            
        Returns:
            Tuple of (left_turning_point, right_turning_point)
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        v = self.potential.v(x)
        
        # Find points where potential crosses energy level
        crossings = []
        for i in range(1, len(x)):
            if (v[i-1] - energy) * (v[i] - energy) <= 0:
                # Linear interpolation to find the exact crossing
                t = (energy - v[i-1]) / (v[i] - v[i-1])
                crossing = x[i-1] + t * (x[i] - x[i-1])
                crossings.append(crossing)
        
        if len(crossings) < 2:
            raise ValueError(f"Couldn't find two turning points for energy {energy}")
            
        # Return the leftmost and rightmost turning points
        return min(crossings), max(crossings)
    
    def calculate_wkb_tunneling(self, energy: float) -> float:
        """
        Calculate tunneling probability using WKB approximation.
        
        Args:
            energy: Energy level (analogous to momentum against the trend)
            
        Returns:
            Tunneling probability
        """
        try:
            # Find turning points for this energy
            x1, x2 = self.find_turning_points(energy)
            
            # Create grid for numerical integration
            x_grid = np.linspace(x1, x2, 1000)
            
            # Integrand for WKB approximation: sqrt(2m(V(x) - E))
            v_values = self.potential.v(x_grid)
            integrand = np.sqrt(2 * self.m * (v_values - energy))
            
            # Handle any NaN or negative values (shouldn't occur but just in case)
            integrand = np.nan_to_num(integrand, nan=0.0)
            integrand = np.maximum(integrand, 0)
            
            # Calculate action integral
            action = simpson(integrand, x_grid)
            
            # WKB tunneling probability: T ≈ exp(-2*action/ħ)
            tunneling_prob = np.exp(-2 * action / self.hbar)
            
            # Ensure reasonable bounds
            return min(max(tunneling_prob, 0.0), 1.0)
            
        except Exception as e:
            print(f"Error calculating tunneling probability: {e}")
            return 0.0
    
    def calculate_tunneling_rates(self, energy_range: Tuple[float, float] = (0.0, 1.0), 
                                 num_points: int = 20) -> Dict[float, float]:
        """
        Calculate tunneling probabilities across a range of energies.
        
        Args:
            energy_range: Energy range to calculate over
            num_points: Number of energy levels to evaluate
            
        Returns:
            Dictionary mapping energy levels to tunneling probabilities
        """
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        tunneling_rates = {}
        
        for energy in energies:
            tunneling_rates[float(energy)] = self.calculate_wkb_tunneling(energy)
            
        return tunneling_rates


class MaxwellsDemon:
    """
    Maxwell's Demon mechanism that adjusts the quantum model based on market information.
    """
    
    def __init__(self, potential: QuantumWellPotential, 
                 volume_threshold: float = 2.0,
                 vwap_threshold: float = 0.01):
        """
        Initialize the Maxwell's Demon filter.
        
        Args:
            potential: The quantum potential model to adjust
            volume_threshold: Threshold for considering volume as significant
            vwap_threshold: Threshold for price deviation from VWAP
        """
        self.potential = potential
        self.volume_threshold = volume_threshold
        self.vwap_threshold = vwap_threshold
        
    def process_information(self, current_rsi: float, 
                           volume_ratio: float, 
                           price_to_vwap_ratio: float,
                           delta_volume: float = 0.0) -> Dict[str, Any]:
        """
        Process market information and adjust potential barrier accordingly.
        
        Args:
            current_rsi: Current RSI value
            volume_ratio: Current volume / average volume
            price_to_vwap_ratio: Current price / VWAP
            delta_volume: Net buying vs selling pressure (optional)
            
        Returns:
            Dictionary with demon's decision and reasoning
        """
        # Initialize default response
        response = {
            "barrier_adjustment": 0.0,
            "allow_signal": True,
            "confidence": "Medium",
            "reason": ""
        }
        
        # Normalized RSI position (transform RSI from 0-100 to -1 to 1)
        x_position = (current_rsi - 50) / 50
        
        # --- Volume Analysis ---
        volume_signal = "None"
        if volume_ratio > self.volume_threshold:
            # Significant volume detected
            if current_rsi > 70:  # Overbought
                # High volume at top can be distribution or momentum
                if delta_volume < 0:  # More selling than buying
                    volume_signal = "Distribution"
                    response["barrier_adjustment"] = -0.5  # Lower barrier (easier reversal)
                    response["reason"] += "High volume with selling pressure at overbought level suggests distribution. "
                    response["confidence"] = "High"
                else:  # More buying than selling
                    volume_signal = "Strong_Momentum"
                    response["barrier_adjustment"] = 0.5  # Raise barrier (harder reversal)
                    response["reason"] += "High volume with buying pressure at overbought level suggests strong momentum. "
                    
            elif current_rsi < 30:  # Oversold
                # High volume at bottom can be accumulation or panic
                if delta_volume > 0:  # More buying than selling
                    volume_signal = "Accumulation"
                    response["barrier_adjustment"] = -0.5  # Lower barrier (easier reversal)
                    response["reason"] += "High volume with buying pressure at oversold level suggests accumulation. "
                    response["confidence"] = "High"
                else:  # More selling than buying
                    volume_signal = "Panic_Selling"
                    # For panic selling, could go either way - might be capitulation
                    # We'll lean toward allowing reversal signals here
                    response["barrier_adjustment"] = -0.3  # Slightly lower barrier
                    response["reason"] += "High volume with continued selling at oversold level could be capitulation. "
        
        # --- VWAP Analysis ---
        if abs(price_to_vwap_ratio - 1.0) > self.vwap_threshold:
            if price_to_vwap_ratio > 1.0:  # Price above VWAP
                if current_rsi > 60:  # Already high RSI
                    response["reason"] += "Price significantly above VWAP with high RSI suggests extended move. "
                    if volume_signal == "Distribution":
                        response["barrier_adjustment"] -= 0.2  # Strengthen reversal signal
                        response["confidence"] = "Very High"
                    elif volume_signal == "Strong_Momentum":
                        # Conflicting signals - momentum vs extension
                        response["barrier_adjustment"] += 0.1  # Slight bias toward trend
                        response["reason"] += "But strong momentum still present. "
            else:  # Price below VWAP
                if current_rsi < 40:  # Already low RSI
                    response["reason"] += "Price significantly below VWAP with low RSI suggests extended move. "
                    if volume_signal == "Accumulation":
                        response["barrier_adjustment"] -= 0.2  # Strengthen reversal signal
                        response["confidence"] = "Very High"
                    elif volume_signal == "Panic_Selling":
                        # Conflicting signals - momentum vs extension
                        response["barrier_adjustment"] += 0.1  # Slight bias toward trend
                        response["reason"] += "But continued selling pressure present. "
        
        # --- Apply the barrier adjustment to the potential ---
        self.potential.adjust_barrier(response["barrier_adjustment"])
        
        # Final decision based on confidence
        if response["barrier_adjustment"] > 0.3:
            response["allow_signal"] = False  # Block reversal signal (trend likely continues)
            response["reason"] += "SIGNAL BLOCKED: Strong evidence of trend continuation. "
        elif response["barrier_adjustment"] < -0.3:
            response["allow_signal"] = True   # Allow reversal signal (reversal likely)
            response["reason"] += "SIGNAL CONFIRMED: Strong evidence supports trend reversal. "
            
        return response


# For testing/visualization
if __name__ == "__main__":
    # Create a potential well with default parameters
    potential = QuantumWellPotential(v0=1.0, a=0.4)
    
    # Initialize a Schrödinger solver
    solver = SchrodingerSolver(potential)
    
    # Initialize a wavefunction in the bullish well
    solver.initialize_wavefunction(center=0.4)  # Start in bullish region (high RSI)
    
    # Plot initial state
    plt.figure(figsize=(10, 6))
    plt.plot(solver.x, np.abs(solver.psi)**2, label='Initial State')
    plt.plot(solver.x, potential.v(solver.x)/10, label='Potential (scaled)')
    plt.legend()
    plt.title('Initial Wavefunction in Bullish Well')
    plt.xlabel('Position (RSI state)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Evolve for 100 time steps
    solver.evolve(100)
    
    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.plot(solver.x, np.abs(solver.psi)**2, label='After Evolution')
    plt.plot(solver.x, potential.v(solver.x)/10, label='Potential (scaled)')
    plt.legend()
    plt.title('Wavefunction After Time Evolution')
    plt.xlabel('Position (RSI state)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate well probabilities
    probs = solver.calculate_well_probabilities()
    print(f"Probability in bearish well: {probs['bearish']:.2f}")
    print(f"Probability in neutral region: {probs['neutral']:.2f}")
    print(f"Probability in bullish well: {probs['bullish']:.2f}")
    
    # Calculate tunneling probabilities for different energy levels
    tunneling = QuantumTunneling(potential)
    tunneling_rates = tunneling.calculate_tunneling_rates()
    
    # Plot tunneling probabilities
    plt.figure(figsize=(10, 6))
    energies, probs = zip(*tunneling_rates.items())
    plt.semilogy(energies, probs, 'o-')
    plt.title('Tunneling Probability vs Energy')
    plt.xlabel('Energy Level')
    plt.ylabel('Tunneling Probability (log scale)')
    plt.grid(True)
    plt.show() 