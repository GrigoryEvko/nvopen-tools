// Function: sub_DFF600
// Address: 0xdff600
//
bool __fastcall sub_DFF600(__int64 a1)
{
  unsigned __int8 v1; // al

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
  {
    if ( (unsigned __int8)(***(_BYTE ***)(a1 - 32) - 5) <= 0x1Fu )
      return *(_DWORD *)(a1 - 24) > 2u;
    return 0;
  }
  if ( (unsigned __int8)(**(_BYTE **)(a1 - 8LL * ((v1 >> 2) & 0xF) - 16) - 5) > 0x1Fu )
    return 0;
  return ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) > 2;
}
