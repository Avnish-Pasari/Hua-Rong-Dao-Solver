# Hua Rong Dao Puzzle Solver 🎲

A Python program that solves the **Hua Rong Dao sliding puzzle** using **Depth-First Search (DFS)** and **A* Search** with heuristics.  

---

## 📖 About the Project

The **Hua Rong Dao** is a classic Chinese sliding block puzzle. The objective is to move the 2x2 block (representing Cao Cao) to the bottom opening of the board by sliding other pieces out of the way.

This solver supports:
- **DFS (Depth-First Search)**: Finds a solution (not guaranteed optimal).  
- **A* Search with Manhattan Distance Heuristic**: Finds the optimal solution.  
- **Custom Advanced Heuristic**: Improves efficiency while maintaining admissibility.

---

## 🧩 Puzzle Rules

- Board size: **4 columns × 5 rows**  
- Pieces:  
  - One 2×2 block  
  - Five 1×2 blocks (vertical or horizontal)  
  - Four 1×1 blocks  
- Two empty spaces  
- Moves: Slide pieces horizontally or vertically into empty spaces (no rotation or diagonal moves).  
- Goal: Move the 2×2 block to the bottom opening.

---

## ⚙️ Features

- Implements **DFS** and **A\*** search algorithms  
- Supports **multi-path pruning** for efficiency  
- Uses **Manhattan distance** and an **advanced admissible heuristic**  
- Reads puzzles from plain-text input files  
- Outputs complete solution sequences to text files  

---

## 🛠️ Tech Stack

- **Language**: Python 3  
- **Core Modules**: `heapq`, `argparse`, `sys`  
- **Algorithm Design**: DFS, A*, heuristic search  
- **Environment**: Linux
