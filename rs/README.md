The intention here is to write an extremely high performance 2048 emulator. What is "high
performance"? idk, just as fast as I can get it I guess.

This is the public interface
```
pub struct Game {
    pub board: [[u16; 4]; 4],
    pub score: u64,
}

impl Game {
    pub fn slide(direction: Direction);

    pub fn get_possible_moves(direction: Direction);
}
```

