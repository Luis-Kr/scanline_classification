import numpy as np 
from typing import Tuple
from numba import njit, prange, int64, float64


def sort_pcd(pcd: np.ndarray, 
             col: int = -2) -> np.ndarray:
    return pcd[pcd[:, col].argsort()]


def find_knickpoints(pcd: np.ndarray, 
                     threshold: int = 100, 
                     col: int = -2) -> Tuple[np.ndarray, np.ndarray]:
    # Sort the pcd based on the second last column
    pcd = sort_pcd(pcd, col=col)
    
    # Calculate the absolute difference of the last column
    vertical_ang_diff = abs(np.diff(pcd[:, -1]))
    
    # Append the last value of vertical_ang_diff to itself
    vertical_ang_diff = np.append(vertical_ang_diff, vertical_ang_diff[-1])
    
    # Find the indices where vertical_ang_diff is greater than the threshold
    knickpoints = np.where(vertical_ang_diff > threshold)[0]
    
    return pcd, knickpoints


@njit([(int64, float64[:], int64[:])], parallel=True)
def scanline_extraction(n, scanlines, knickpoints):
   for i in prange(n):
      # For each i, find the index in the sorted knickpoints array where i should be inserted to maintain sorted order.
      # 'side=left' means that the first suitable location is given.
      scanlines[i] = np.searchsorted(knickpoints, i, side='left')
   
   # Increment all elements in scanlines by 1
   scanlines += 1
   
   return scanlines


def append_scanlines(pcd: np.ndarray, scanlines: np.ndarray) -> np.ndarray:
    # Use np.c_ to concatenate pcd and scanlines along the second axis (columns)
    return np.c_[pcd, scanlines]