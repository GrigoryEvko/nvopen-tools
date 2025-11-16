// Function: sub_FFFFF0
// Address: 0xfffff0
//
__int64 __fastcall sub_FFFFF0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rsi

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v2 = *(_QWORD **)(a2 - 8);
    if ( *v2 != *a1 )
      return 0;
  }
  else
  {
    v2 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *v2 != *a1 )
      return 0;
  }
  return sub_FFFE90(v2[8]);
}
