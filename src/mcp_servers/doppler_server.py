#!/usr/bin/env python3
"""
MCP Server for Doppler Calculations

Provides tools for wavelength shifts and radial velocity measurements.
"""

import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Doppler Tools", log_level="WARNING")
C_KMS = 299792.458  # Speed of light in km/s


@mcp.tool()
def doppler_shift(
    rest_wavelength_angstroms: float,
    velocity_kms: float
) -> dict:
    """
    Calculate observed wavelength given rest wavelength and radial velocity.
    
    Uses the relativistic Doppler formula for accuracy at all velocities.
    Positive velocity means the source is receding (redshift).
    
    Args:
        rest_wavelength_angstroms: Laboratory wavelength in Angstroms
                                  (e.g., H-alpha = 6562.8 Å)
        velocity_kms: Radial velocity in km/s (positive = receding)
        
    Returns:
        Observed wavelength and shift direction
    """
    if rest_wavelength_angstroms <= 0:
        return {"error": "Wavelength must be positive"}
    
    beta = velocity_kms / C_KMS
    if abs(beta) >= 1:
        return {"error": "Velocity cannot exceed speed of light"}
    
    doppler_factor = np.sqrt((1 + beta) / (1 - beta))
    observed = rest_wavelength_angstroms * doppler_factor
    shift = observed - rest_wavelength_angstroms
    
    direction = "redshift" if velocity_kms > 0 else "blueshift" if velocity_kms < 0 else "none"
    
    return {
        "observed_wavelength_angstroms": round(float(observed), 4),
        "shift_angstroms": round(float(shift), 4),
        "direction": direction
    }


@mcp.tool()
def wavelength_to_velocity(
    rest_wavelength_angstroms: float,
    observed_wavelength_angstroms: float
) -> dict:
    """
    Calculate radial velocity from observed wavelength shift.
    
    This is the inverse of doppler_shift—given where a line should be
    and where you observe it, calculate how fast the source is moving.
    
    Args:
        rest_wavelength_angstroms: Laboratory wavelength
        observed_wavelength_angstroms: Measured wavelength in spectrum
        
    Returns:
        Radial velocity in km/s and redshift z
    """
    if rest_wavelength_angstroms <= 0 or observed_wavelength_angstroms <= 0:
        return {"error": "Wavelengths must be positive"}
    
    z = (observed_wavelength_angstroms - rest_wavelength_angstroms) / rest_wavelength_angstroms
    z_factor = (1 + z) ** 2
    beta = (z_factor - 1) / (z_factor + 1)
    velocity = beta * C_KMS
    
    direction = "receding" if velocity > 0 else "approaching" if velocity < 0 else "stationary"
    
    return {
        "velocity_kms": round(float(velocity), 3),
        "redshift_z": round(float(z), 6),
        "direction": direction
    }


if __name__ == "__main__":
    mcp.run()
