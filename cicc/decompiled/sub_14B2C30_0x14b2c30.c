// Function: sub_14B2C30
// Address: 0x14b2c30
//
__int64 __fastcall sub_14B2C30(_QWORD *a1, __int64 a2)
{
  int v2; // eax
  int v3; // eax
  unsigned int v4; // r8d
  _QWORD *v5; // rsi
  _QWORD *v7; // rdx

  if ( a2 == *a1 )
    return 1;
  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 <= 0x17u )
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 45 )
      goto LABEL_17;
  }
  else if ( (_BYTE)v2 != 69 )
  {
LABEL_4:
    v3 = v2 - 24;
    goto LABEL_5;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v7 = *(_QWORD **)(a2 - 8);
  else
    v7 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v4 = 1;
  if ( *v7 == a1[1] )
    return v4;
  if ( (unsigned __int8)v2 > 0x17u )
    goto LABEL_4;
LABEL_17:
  v3 = *(unsigned __int16 *)(a2 + 18);
LABEL_5:
  v4 = 0;
  if ( v3 != 47 )
    return v4;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v5 = *(_QWORD **)(a2 - 8);
  else
    v5 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  LOBYTE(v4) = *v5 == a1[2];
  return v4;
}
