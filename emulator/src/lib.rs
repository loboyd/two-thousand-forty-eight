use pyo3::prelude::*;
use rand::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass]
pub struct Batch {
    games: [Game; 512],
    move_masks: [u8; 512],
}

#[pymethods]
impl Batch {
    #[new]
    pub fn new() -> Self {
        Self {
            games: [Game::new(); 512],
            move_masks: [13; 512],
        }
    }

    pub fn r#move(moves: Vec<bool>) {
        //let mut moves = Vec::new();
        //for _ in 0..64 {
        //    moves.push(match m0 && 0x11 {
        //        0 => Direction.Up,
        //        1 => Direction.Right,
        //        2 => Direction.Down,
        //        3 => Direction.Left,
        //    })
        //    m0 >>= 2;
        //}
    }
}

#[derive(Clone, Copy)]
#[pyclass]
pub struct Game {
    // note: 2^17 is technically possible to achieve, but cmon now...
    #[pyo3(get)]
    pub board: [[u16; 4]; 4],
    #[pyo3(get)]
    pub score: u64,
}

// note: take 8 u128s and the just process them as 2-bit pairs each describing one of 512 moves

#[pymethods]
impl Game {
    #[new]
    pub fn new() -> Self {
        let mut s = Self {
            board: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            score: 0,
        };
        s.place_random_tile().unwrap();
        s.place_random_tile().unwrap();
        s
    }

    pub fn r#move(&mut self, direction: Direction) {
        match direction {
            Direction::Right => {
                  for i in 0..4 {
                      self.collapse_unit([(i, 3), (i, 2), (i, 1), (i, 0)]);
                      self.merge_unit([(i, 3), (i, 2), (i, 1), (i, 0)]);
                      self.collapse_unit([(i, 3), (i, 2), (i, 1), (i, 0)]);
                  }
            },
            Direction::Down => {
                  for i in 0..4 {
                      self.collapse_unit([(3, i), (2, i), (1, i), (0, i)]);
                      self.merge_unit([(3, i), (2, i), (1, i), (0, i)]);
                      self.collapse_unit([(3, i), (2, i), (1, i), (0, i)]);
                  }
            },
            Direction::Up => {
                  for i in 0..4 {
                      self.collapse_unit([(0, i), (1, i), (2, i), (3, i)]);
                      self.merge_unit([(0, i), (1, i), (2, i), (3, i)]);
                      self.collapse_unit([(0, i), (1, i), (2, i), (3, i)]);
                  }
            },
            Direction::Left => {
                  for i in 0..4 {
                      self.collapse_unit([(i, 0), (i, 1), (i, 2), (i, 3)]);
                      self.merge_unit([(i, 0), (i, 1), (i, 2), (i, 3)]);
                      self.collapse_unit([(i, 0), (i, 1), (i, 2), (i, 3)]);
                  }
            },
        }

        self.place_random_tile();
    }

    pub fn get_move_mask(&self) -> [bool; 4] {
        let mut moves = [false; 4]; // right, down, up, left
        for i in 0..4 {
            for j in 0..3 {
                if self.board[i][j] != 0 && self.board[i][j] == self.board[i][j+1] {
                    moves[0] = true; // right is available
                    moves[3] = true; // left is available
                }
                if self.board[i][j] == 0 && self.board[i][j+1] > 0 {
                    moves[3] = true; // left is available
                }
                if self.board[i][j] > 0 && self.board[i][j+1] == 0 {
                    moves[0] = true; // right is available
                }
            }
        }

        for i in 0..3 {
            for j in 0..4 {
                if self.board[i][j] != 0 && self.board[i][j] == self.board[i+1][j] {
                    moves[1] = true; // down is available
                    moves[2] = true; // up is available
                }
                if self.board[i][j] == 0 && self.board[i+1][j] > 0 {
                    moves[2] = true; // up is available
                }
                if self.board[i][j] > 0 && self.board[i+1][j] == 0 {
                    moves[1] = true; // down is available
                }
            }
        }
        moves
    }

    fn collapse_unit(&mut self, tiles: [(usize, usize); 4]) {
        for i in 0..4 {
            let mut j = i;
            while j < 4 && self.board[tiles[j].0][tiles[j].1]  == 0 {
                j += 1;
            }
            if i < j && j < 4 {
                self.board[tiles[i].0][tiles[i].1] = self.board[tiles[j].0][tiles[j].1];
                self.board[tiles[j].0][tiles[j].1] = 0;
            }
        }
    }

    fn merge_unit(&mut self, tiles: [(usize, usize); 4]) {
        for i in 0..3 {
            let tile = self.board[tiles[i].0][tiles[i].1];
            let next = self.board[tiles[i+1].0][tiles[i+1].1];
            if tile != 0 && tile == next {
                // sadly you can't take two mutable refs, so we have to do this via indexing
                self.board[tiles[i].0][tiles[i].1] *= 2; // update tile
                self.score += self.board[tiles[i].0][tiles[i].1] as u64;
                self.board[tiles[i+1].0][tiles[i+1].1] = 0; // update next
            }
        }
    }

    fn find_empties(&mut self) -> Vec<(usize, usize)> {
        let mut empties = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                if self.board[i][j] == 0 {
                    empties.push((i, j));
                }
            }
        }
        empties
    }

    fn place_random_tile(&mut self) -> Option<()> {
        let mut rng = rand::thread_rng();
        let empties = self.find_empties();
        if empties.len() == 0 {
            return None;
        } else {
            let (r, c) = empties[rng.gen_range(0..empties.len())];
            self.board[r][c] = if rng.gen::<f64>() < 0.1 { 4 } else { 2 };
            return Some(());
        }
    }
}
#[pyclass]
#[derive(Clone)]
pub enum Direction {
    Right,
    Down,
    Up,
    Left,
}

#[pymethods]
impl Direction {
    #[new]
    fn new(which: usize) -> Self {
        match which {
            0 => Self::Right,
            1 => Self::Down,
            2 => Self::Up,
            _ => Self::Left,
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn emulator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Game>()?;
    m.add_class::<Direction>()?;
    Ok(())
}

