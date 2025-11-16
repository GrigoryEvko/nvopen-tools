// Function: sub_1D18140
// Address: 0x1d18140
//
bool __fastcall sub_1D18140(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rbx
  __int64 v4; // rbx

  result = 1;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 792LL) & 8) == 0 )
  {
    result = (*(_BYTE *)(a2 + 80) & 0x10) != 0;
    if ( (*(_BYTE *)(a2 + 80) & 0x10) == 0 )
    {
      result = *(_WORD *)(a2 + 24) == 11 || *(_WORD *)(a2 + 24) == 33;
      if ( result )
      {
        v3 = *(_QWORD *)(a2 + 88);
        if ( *(void **)(v3 + 32) == sub_16982C0() )
          v4 = *(_QWORD *)(v3 + 40) + 8LL;
        else
          v4 = v3 + 32;
        return (*(_BYTE *)(v4 + 18) & 7) != 1;
      }
    }
  }
  return result;
}
