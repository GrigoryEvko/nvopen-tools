// Function: sub_11A3480
// Address: 0x11a3480
//
_QWORD *__fastcall sub_11A3480(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // rsi
  __int64 v9; // rdx
  unsigned __int8 *v10; // rsi
  __int64 v11; // rdi
  __int64 v13; // [rsp-50h] [rbp-50h]
  __int64 v14[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( !a3 )
    BUG();
  v7 = *(_QWORD *)(a3 + 24);
  v14[0] = v7;
  if ( v7 )
  {
    sub_B96E90((__int64)v14, v7, 1);
    v8 = a2[6];
    v9 = (__int64)(a2 + 6);
    if ( !v8 )
      goto LABEL_5;
  }
  else
  {
    v8 = a2[6];
    v9 = (__int64)(a2 + 6);
    if ( !v8 )
      goto LABEL_7;
  }
  v13 = v9;
  sub_B91220(v9, v8);
  v9 = v13;
LABEL_5:
  v10 = (unsigned __int8 *)v14[0];
  a2[6] = v14[0];
  if ( v10 )
    sub_B976B0((__int64)v14, v10, v9);
LABEL_7:
  sub_B44220(a2, a3, a4);
  v11 = *(_QWORD *)(a1 + 40);
  v14[0] = (__int64)a2;
  sub_11A2F60(v11 + 2096, v14);
  return a2;
}
