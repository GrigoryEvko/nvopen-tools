// Function: sub_13D7710
// Address: 0x13d7710
//
__int64 __fastcall sub_13D7710(_QWORD *a1, __int64 a2)
{
  int v2; // eax
  int v4; // eax
  unsigned int v5; // r8d
  _QWORD *v6; // rsi

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
  v5 = 0;
  if ( v4 != 45 )
    return 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(_QWORD **)(a2 - 8);
  else
    v6 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  LOBYTE(v5) = *v6 == *a1;
  return v5;
}
