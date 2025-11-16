// Function: sub_17A4C80
// Address: 0x17a4c80
//
_QWORD *__fastcall sub_17A4C80(__int64 *a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v5; // rsi
  __int64 v6; // r14
  __int64 v7; // rsi
  unsigned __int8 *v8; // rsi
  __int64 v9; // rcx
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = a3[6];
  v11[0] = v5;
  if ( v5 )
  {
    v6 = (__int64)(a2 + 6);
    sub_1623A60((__int64)v11, v5, 2);
    v7 = a2[6];
    if ( !v7 )
      goto LABEL_4;
  }
  else
  {
    v7 = a2[6];
    v6 = (__int64)(a2 + 6);
    if ( !v7 )
      goto LABEL_6;
  }
  sub_161E7C0(v6, v7);
LABEL_4:
  v8 = (unsigned __int8 *)v11[0];
  a2[6] = v11[0];
  if ( v8 )
    sub_1623210((__int64)v11, v8, v6);
LABEL_6:
  sub_157E9D0(a3[5] + 40LL, (__int64)a2);
  v9 = a3[3];
  a2[4] = a3 + 3;
  v9 &= 0xFFFFFFFFFFFFFFF8LL;
  a2[3] = v9 | a2[3] & 7LL;
  *(_QWORD *)(v9 + 8) = a2 + 3;
  a3[3] = a3[3] & 7LL | (unsigned __int64)(a2 + 3);
  sub_170B990(*a1, (__int64)a2);
  return a2;
}
