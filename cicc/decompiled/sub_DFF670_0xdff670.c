// Function: sub_DFF670
// Address: 0xdff670
//
bool __fastcall sub_DFF670(__int64 a1)
{
  bool result; // al

  result = (*(_BYTE *)(a1 - 16) & 2) != 0;
  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
  {
    result = 0;
    if ( *(_DWORD *)(a1 - 24) > 2u )
      return (unsigned __int8)(***(_BYTE ***)(a1 - 32) - 5) <= 0x1Fu;
  }
  else if ( ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) > 2 )
  {
    return (unsigned __int8)(**(_BYTE **)(a1 - 8LL * ((*(_BYTE *)(a1 - 16) >> 2) & 0xF) - 16) - 5) <= 0x1Fu;
  }
  return result;
}
