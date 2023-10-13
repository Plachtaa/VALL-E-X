# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

Symbol = TypeVar('Symbol')


# Disable __repr__ otherwise it could freeze e.g. Jupyter.
@dataclass(repr=False)
class SymbolTable(Generic[Symbol]):
    '''SymbolTable that maps symbol IDs, found on the FSA arcs to
    actual objects. These objects can be arbitrary Python objects
    that can serve as keys in a dictionary (i.e. they need to be
    hashable and immutable).

    The SymbolTable can only be read to/written from disk if the
    symbols are strings.
    '''
    _id2sym: Dict[int, Symbol] = field(default_factory=dict)
    '''Map an integer to a symbol.
    '''

    _sym2id: Dict[Symbol, int] = field(default_factory=dict)
    '''Map a symbol to an integer.
    '''

    _next_available_id: int = 1
    '''A helper internal field that helps adding new symbols
    to the table efficiently.
    '''

    eps: Symbol = '<eps>'
    '''Null symbol, always mapped to index 0.
    '''

    def __post_init__(self):
        for idx, sym in self._id2sym.items():
            assert self._sym2id[sym] == idx
            assert idx >= 0

        for sym, idx in self._sym2id.items():
            assert idx >= 0
            assert self._id2sym[idx] == sym

        if 0 not in self._id2sym:
            self._id2sym[0] = self.eps
            self._sym2id[self.eps] = 0
        else:
            assert self._id2sym[0] == self.eps
            assert self._sym2id[self.eps] == 0

        self._next_available_id = max(self._id2sym) + 1

    @staticmethod
    def from_str(s: str) -> 'SymbolTable':
        '''Build a symbol table from a string.

        The string consists of lines. Every line has two fields separated
        by space(s), tab(s) or both. The first field is the symbol and the
        second the integer id of the symbol.

        Args:
          s:
            The input string with the format described above.
        Returns:
          An instance of :class:`SymbolTable`.
        '''
        id2sym: Dict[int, str] = dict()
        sym2id: Dict[str, int] = dict()

        for line in s.split('\n'):
            fields = line.split()
            if len(fields) == 0:
                continue  # skip empty lines
            assert len(fields) == 2, \
                    f'Expect a line with 2 fields. Given: {len(fields)}'
            sym, idx = fields[0], int(fields[1])
            assert sym not in sym2id, f'Duplicated symbol {sym}'
            assert idx not in id2sym, f'Duplicated id {idx}'
            id2sym[idx] = sym
            sym2id[sym] = idx

        eps = id2sym.get(0, '<eps>')

        return SymbolTable(_id2sym=id2sym, _sym2id=sym2id, eps=eps)

    @staticmethod
    def from_file(filename: str) -> 'SymbolTable':
        '''Build a symbol table from file.

        Every line in the symbol table file has two fields separated by
        space(s), tab(s) or both. The following is an example file:

        .. code-block::

            <eps> 0
            a 1
            b 2
            c 3

        Args:
          filename:
            Name of the symbol table file. Its format is documented above.

        Returns:
          An instance of :class:`SymbolTable`.

        '''
        with open(filename, 'r', encoding='utf-8') as f:
            return SymbolTable.from_str(f.read().strip())

    def to_str(self) -> str:
        '''
        Returns:
          Return a string representation of this object. You can pass
          it to the method ``from_str`` to recreate an identical object.
        '''
        s = ''
        for idx, symbol in sorted(self._id2sym.items()):
            s += f'{symbol} {idx}\n'
        return s

    def to_file(self, filename: str):
        '''Serialize the SymbolTable to a file.

        Every line in the symbol table file has two fields separated by
        space(s), tab(s) or both. The following is an example file:

        .. code-block::

            <eps> 0
            a 1
            b 2
            c 3

        Args:
          filename:
            Name of the symbol table file. Its format is documented above.
        '''
        with open(filename, 'w', encoding='utf-8') as f:
            for idx, symbol in sorted(self._id2sym.items()):
                print(symbol, idx, file=f)

    def add(self, symbol: Symbol, index: Optional[int] = None) -> int:
        '''Add a new symbol to the SymbolTable.

        Args:
            symbol:
                The symbol to be added.
            index:
                Optional int id to which the symbol should be assigned.
                If it is not available, a ValueError will be raised.

        Returns:
            The int id to which the symbol has been assigned.
        '''
        # Already in the table? Return its ID.
        if symbol in self._sym2id:
            return self._sym2id[symbol]
        # Specific ID not provided - use next available.
        if index is None:
            index = self._next_available_id
        # Specific ID provided but not available.
        if index in self._id2sym:
            raise ValueError(f"Cannot assign id '{index}' to '{symbol}' - "
                             f"already occupied by {self._id2sym[index]}")
        self._sym2id[symbol] = index
        self._id2sym[index] = symbol

        # Update next available ID if needed
        if self._next_available_id <= index:
            self._next_available_id = index + 1

        return index

    def get(self, k: Union[int, Symbol]) -> Union[Symbol, int]:
        '''Get a symbol for an id or get an id for a symbol

        Args:
          k:
            If it is an id, it tries to find the symbol corresponding
            to the id; if it is a symbol, it tries to find the id
            corresponding to the symbol.

        Returns:
          An id or a symbol depending on the given `k`.
        '''
        if isinstance(k, int):
            return self._id2sym[k]
        else:
            return self._sym2id[k]

    def merge(self, other: 'SymbolTable') -> 'SymbolTable':
        '''Create a union of two SymbolTables.
        Raises an AssertionError if the same IDs are occupied by
        different symbols.

        Args:
            other:
                A symbol table to merge with ``self``.

        Returns:
            A new symbol table.
        '''
        self._check_compatible(other)

        id2sym = {**self._id2sym, **other._id2sym}
        sym2id = {**self._sym2id, **other._sym2id}

        return SymbolTable(_id2sym=id2sym, _sym2id=sym2id, eps=self.eps)

    def _check_compatible(self, other: 'SymbolTable') -> None:
        # Epsilon compatibility
        assert self.eps == other.eps, f'Mismatched epsilon symbol: ' \
                                      f'{self.eps} != {other.eps}'
        # IDs compatibility
        common_ids = set(self._id2sym).intersection(other._id2sym)
        for idx in common_ids:
            assert self[idx] == other[idx], f'ID conflict for id: {idx}, ' \
                                            f'self[idx] = "{self[idx]}", ' \
                                            f'other[idx] = "{other[idx]}"'
        # Symbols compatibility
        common_symbols = set(self._sym2id).intersection(other._sym2id)
        for sym in common_symbols:
            assert self[sym] == other[sym], f'ID conflict for id: {sym}, ' \
                                            f'self[sym] = "{self[sym]}", ' \
                                            f'other[sym] = "{other[sym]}"'

    def __getitem__(self, item: Union[int, Symbol]) -> Union[Symbol, int]:
        return self.get(item)

    def __contains__(self, item: Union[int, Symbol]) -> bool:
        if isinstance(item, int):
            return item in self._id2sym
        else:
            return item in self._sym2id

    def __len__(self) -> int:
        return len(self._id2sym)

    def __eq__(self, other: 'SymbolTable') -> bool:
        if len(self) != len(other):
            return False

        for s in self.symbols:
            if self[s] != other[s]:
                return False

        return True

    @property
    def ids(self) -> List[int]:
        '''Returns a list of integer IDs corresponding to the symbols.
        '''
        ans = list(self._id2sym.keys())
        ans.sort()
        return ans

    @property
    def symbols(self) -> List[Symbol]:
        '''Returns a list of symbols (e.g., strings) corresponding to
        the integer IDs.
        '''
        ans = list(self._sym2id.keys())
        ans.sort()
        return ans
