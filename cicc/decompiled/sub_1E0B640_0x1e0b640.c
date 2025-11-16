// Function: sub_1E0B640
// Address: 0x1e0b640
//
_QWORD *__fastcall sub_1E0B640(__int64 a1, __int64 a2, __int64 *a3, unsigned __int8 a4)
{
  __int64 v6; // rsi
  _QWORD *v7; // r13
  __int64 v9[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = *a3;
  v9[0] = v6;
  if ( v6 )
    sub_1623A60((__int64)v9, v6, 2);
  v7 = *(_QWORD **)(a1 + 224);
  if ( v7 )
    *(_QWORD *)(a1 + 224) = *v7;
  else
    v7 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 120), 72, 8);
  sub_1E1C450(v7, a1, a2, v9, a4);
  if ( v9[0] )
    sub_161E7C0((__int64)v9, v9[0]);
  return v7;
}
