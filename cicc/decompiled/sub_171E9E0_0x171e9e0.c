// Function: sub_171E9E0
// Address: 0x171e9e0
//
__int64 __fastcall sub_171E9E0(_QWORD **a1, __int64 a2)
{
  int v2; // eax
  int v4; // eax
  _QWORD *v5; // rsi

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 > 0x17u )
  {
    v4 = v2 - 24;
  }
  else
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    v4 = *(unsigned __int16 *)(a2 + 18);
  }
  if ( v4 != 38 )
    return 0;
  v5 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
     ? *(_QWORD **)(a2 - 8)
     : (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !*v5 )
    return 0;
  **a1 = *v5;
  return 1;
}
