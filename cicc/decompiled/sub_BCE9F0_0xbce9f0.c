// Function: sub_BCE9F0
// Address: 0xbce9f0
//
char __fastcall sub_BCE9F0(__int64 a1)
{
  char i; // al
  char result; // al

  for ( i = *(_BYTE *)(a1 + 8); i == 16; i = *(_BYTE *)(a1 + 8) )
    a1 = *(_QWORD *)(a1 + 24);
  if ( i == 15 )
  {
    result = 1;
    if ( (*(_DWORD *)(a1 + 8) & 0x1000) == 0 )
    {
      if ( ((*(_DWORD *)(a1 + 8) >> 8) & 0x20) != 0 )
        return 0;
      else
        return sub_BCEAB0();
    }
  }
  else
  {
    return i == 18 || *(_BYTE *)(a1 + 8) == 20 && *(_BYTE *)(sub_BCE9B0(a1) + 8) == 18;
  }
  return result;
}
