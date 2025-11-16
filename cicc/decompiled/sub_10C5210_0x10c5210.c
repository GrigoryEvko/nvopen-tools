// Function: sub_10C5210
// Address: 0x10c5210
//
__int64 __fastcall sub_10C5210(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v5; // r12
  char v6; // al
  __int64 v7; // rsi
  _BYTE *v8; // rax
  __int64 v9; // rax

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 57 )
    return 0;
  v5 = *(_BYTE **)(a2 - 64);
  if ( *v5 != 59 )
    return 0;
  v6 = sub_995B10(a1, *((_QWORD *)v5 - 8));
  v7 = *((_QWORD *)v5 - 4);
  if ( v6 && v7 )
  {
    *a1[1] = v7;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10(a1, v7) )
      return 0;
    v9 = *((_QWORD *)v5 - 8);
    if ( !v9 )
      return 0;
    *a1[1] = v9;
  }
  v8 = *(_BYTE **)(a2 - 32);
  if ( *v8 <= 0x15u )
  {
    *a1[2] = v8;
    return 1;
  }
  return 0;
}
