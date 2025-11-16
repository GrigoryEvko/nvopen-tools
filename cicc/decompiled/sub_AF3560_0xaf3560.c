// Function: sub_AF3560
// Address: 0xaf3560
//
__int64 __fastcall sub_AF3560(__int64 a1, __int64 a2)
{
  switch ( a2 )
  {
    case 12LL:
      if ( (*(_QWORD *)a1 != 0x67616C4650534944LL || *(_DWORD *)(a1 + 8) != 1869768026)
        && *(_QWORD *)a1 == 0x67616C4650534944LL
        && *(_DWORD *)(a1 + 8) == 1701999952 )
      {
        return 32;
      }
      return 0;
    case 15LL:
      if ( *(_QWORD *)a1 == 0x67616C4650534944LL
        && *(_DWORD *)(a1 + 8) == 1953655126
        && *(_WORD *)(a1 + 12) == 24949
        && *(_BYTE *)(a1 + 14) == 108 )
      {
        return 1;
      }
      if ( *(_QWORD *)a1 != 0x67616C4650534944LL
        || *(_DWORD *)(a1 + 8) != 1701602628
        || *(_WORD *)(a1 + 12) != 25972
        || *(_BYTE *)(a1 + 14) != 100 )
      {
        return 0;
      }
      return 512;
    case 19LL:
      if ( *(_QWORD *)a1 ^ 0x67616C4650534944LL | *(_QWORD *)(a1 + 8) ^ 0x7472695665727550LL
        || *(_WORD *)(a1 + 16) != 24949
        || *(_BYTE *)(a1 + 18) != 108 )
      {
        if ( !(*(_QWORD *)a1 ^ 0x67616C4650534944LL | *(_QWORD *)(a1 + 8) ^ 0x556F546C61636F4CLL)
          && *(_WORD *)(a1 + 16) == 26990
          && *(_BYTE *)(a1 + 18) == 116 )
        {
          return 4;
        }
        return 0;
      }
      return 2;
    case 18LL:
      if ( !(*(_QWORD *)a1 ^ 0x67616C4650534944LL | *(_QWORD *)(a1 + 8) ^ 0x6974696E69666544LL)
        && *(_WORD *)(a1 + 16) == 28271 )
      {
        return 8;
      }
      if ( *(_QWORD *)a1 ^ 0x67616C4650534944LL | *(_QWORD *)(a1 + 8) ^ 0x65726944436A624FLL
        || *(_WORD *)(a1 + 16) != 29795 )
      {
        return 0;
      }
      return 2048;
    case 17LL:
      if ( *(_QWORD *)a1 ^ 0x67616C4650534944LL | *(_QWORD *)(a1 + 8) ^ 0x657A696D6974704FLL
        || *(_BYTE *)(a1 + 16) != 100 )
      {
        if ( *(_QWORD *)a1 ^ 0x67616C4650534944LL | *(_QWORD *)(a1 + 8) ^ 0x61746E656D656C45LL
          || *(_BYTE *)(a1 + 16) != 108 )
        {
          if ( !(*(_QWORD *)a1 ^ 0x67616C4650534944LL | *(_QWORD *)(a1 + 8) ^ 0x7669737275636552LL)
            && *(_BYTE *)(a1 + 16) == 101 )
          {
            return 128;
          }
          return 0;
        }
        return 64;
      }
      else
      {
        return 16;
      }
    default:
      if ( a2 != 22
        || *(_QWORD *)a1 ^ 0x67616C4650534944LL | *(_QWORD *)(a1 + 8) ^ 0x706275536E69614DLL
        || *(_DWORD *)(a1 + 16) != 1919381362
        || *(_WORD *)(a1 + 20) != 28001 )
      {
        return 0;
      }
      return 256;
  }
}
