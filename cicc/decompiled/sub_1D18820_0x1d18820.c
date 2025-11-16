// Function: sub_1D18820
// Address: 0x1d18820
//
bool __fastcall sub_1D18820(__int64 a1)
{
  bool result; // al
  __int64 v2; // rbx
  char v3; // dl
  __int64 v4; // rbx
  __int64 v5; // rbx

  result = *(_WORD *)(a1 + 24) == 11 || *(_WORD *)(a1 + 24) == 33;
  if ( result )
  {
    v2 = *(_QWORD *)(a1 + 88);
    if ( *(void **)(v2 + 32) != sub_16982C0() )
    {
      v3 = *(_BYTE *)(v2 + 50);
      result = 0;
      v4 = v2 + 32;
      if ( (v3 & 7) != 3 )
        return result;
      return ((*(_BYTE *)(v4 + 18) >> 3) ^ 1) & 1;
    }
    v5 = *(_QWORD *)(v2 + 40);
    result = 0;
    if ( (*(_BYTE *)(v5 + 26) & 7) == 3 )
    {
      v4 = v5 + 8;
      return ((*(_BYTE *)(v4 + 18) >> 3) ^ 1) & 1;
    }
  }
  return result;
}
