// Function: sub_138EBD0
// Address: 0x138ebd0
//
char __fastcall sub_138EBD0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // r13
  unsigned __int64 v4; // r14
  unsigned __int8 v7; // al
  __int64 v9; // rsi
  __int64 *v10; // rax
  unsigned __int8 v11; // al

  v3 = *a2;
  if ( *(_BYTE *)(*(_QWORD *)*a2 + 8LL) != 15 )
    return 0;
  v4 = *a3;
  if ( *(_BYTE *)(*(_QWORD *)*a3 + 8LL) != 15 )
    return 0;
  v7 = *(_BYTE *)(v3 + 16);
  if ( v7 <= 0x17u )
  {
    if ( v7 != 17 )
      goto LABEL_10;
    v9 = *(_QWORD *)(v3 + 24);
  }
  else
  {
    v9 = *(_QWORD *)(*(_QWORD *)(v3 + 40) + 56LL);
  }
  if ( v9 )
  {
LABEL_7:
    v10 = sub_138E440(a1, v9);
    return sub_13833D0((__int64)v10, v3, a2[1], v4, a3[1]);
  }
LABEL_10:
  v11 = *(_BYTE *)(v4 + 16);
  if ( v11 <= 0x17u )
  {
    if ( v11 != 17 )
      return 1;
    v9 = *(_QWORD *)(v4 + 24);
  }
  else
  {
    v9 = *(_QWORD *)(*(_QWORD *)(v4 + 40) + 56LL);
  }
  if ( v9 )
    goto LABEL_7;
  return 1;
}
