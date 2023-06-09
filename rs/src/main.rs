use rand::prelude::*;

pub struct Game {
    // note: 2^17 is technically possible to achieve, but cmon now...
    pub board: [[u16; 4]; 4],
    pub score: u64,
}

impl Game {
    pub fn new() -> Self {
        let mut s = Self {
            board: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            score: 0,
        };
        s.place_random_tile();
        s.place_random_tile();
        s
    }

    pub fn slide(&mut self, direction: Direction) {
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

    pub fn get_available_moves(&self) -> [bool; 4] {
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

    // todo: this can fail (when there are no empty tiles left)
    fn place_random_tile(&mut self) {
        let mut rng = rand::thread_rng();
        let empties = self.find_empties();
        let (r, c) = empties[rng.gen_range(0..empties.len())];
        self.board[r][c] = if rng.gen::<f64>() < 0.1 { 4 } else { 2 };
    }
}

pub enum Direction {
    Right,
    Down,
    Up,
    Left,
}

fn print_game(game: &Game) {
    // print the board
    for row in game.board {
        for tile in row {
            print!("{} ", tile);
        }
        println!();
    }
    println!("score: {}", game.score);
    println!();
}

//let mut moves = [false; 4]; // right, down, up, left

fn main() {
    let mut game = Game::new();
    print_game(&game);
    loop {
        let available_moves = game.get_available_moves();
        if available_moves[1] {
            game.slide(Direction::Down);
        } else if available_moves[0] {
            game.slide(Direction::Right);
        } else if available_moves[2] {
            game.slide(Direction::Up);
            game.slide(Direction::Down);
        } else if available_moves[3] {
            game.slide(Direction::Left);
            game.slide(Direction::Right);
        } else {
            break;
        }
        print_game(&game);
    }
}

