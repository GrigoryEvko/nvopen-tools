// Function: sub_1277390
// Address: 0x1277390
//
__int64 __fastcall sub_1277390(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rsi
  __int64 *v10; // r14
  __int64 v11; // rsi
  __int64 result; // rax
  _BYTE v13[33]; // [rsp+Fh] [rbp-21h] BYREF

  v5 = sub_1269DC0((__int64)a1, a2, v13);
  if ( (*(_BYTE *)(a2 + 156) & 2) != 0 )
  {
    v10 = (__int64 *)sub_127D2C0(a1, *(_QWORD *)(a2 + 120));
LABEL_15:
    if ( a3 )
      goto LABEL_16;
    goto LABEL_12;
  }
  v9 = v5;
  if ( v13[0] == 2 )
  {
    if ( v5 )
      goto LABEL_11;
    goto LABEL_14;
  }
  if ( !v13[0] || v13[0] == 3 )
  {
LABEL_14:
    v10 = (__int64 *)sub_127D2A0(a1, *(_QWORD *)(a2 + 120));
    goto LABEL_15;
  }
  if ( !v5 )
    goto LABEL_8;
  if ( v13[0] != 1 )
  {
    v9 = v5 + 64;
LABEL_8:
    sub_127B630("unsupported initialization variant!", v9);
  }
LABEL_11:
  v10 = (__int64 *)sub_127F610(a1, v5, *(_QWORD *)(a2 + 120), v6, v7, v8);
  if ( a3 )
    goto LABEL_16;
LABEL_12:
  if ( v10 )
    a3 = *v10;
  a3 = sub_1277310(a1, a2, a3);
LABEL_16:
  sub_126A090((__int64)a1, a3, (__int64)v10, a2);
  v11 = (unsigned int)sub_127C800(a2);
  sub_15E4CC0(a3, v11);
  result = sub_127B250(a2);
  if ( (_BYTE)result )
  {
    if ( (*(_BYTE *)(a2 + 156) & 2) == 0 )
      return sub_1273CD0((__int64)a1, a3);
  }
  return result;
}
