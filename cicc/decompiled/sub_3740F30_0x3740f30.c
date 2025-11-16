// Function: sub_3740F30
// Address: 0x3740f30
//
_QWORD *__fastcall sub_3740F30(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v7; // rsi
  _QWORD *v8; // r15
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  unsigned __int8 *v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(unsigned __int8 **)a3;
  v8 = *(_QWORD **)(a1 + 32);
  v15[0] = v7;
  if ( v7 )
    sub_B96E90((__int64)v15, (__int64)v7, 1);
  v9 = (__int64)sub_2E7B380(v8, a4, v15, 0);
  if ( v15[0] )
    sub_B91220((__int64)v15, (__int64)v15[0]);
  sub_2E31040((__int64 *)(a1 + 40), v9);
  v10 = *a2;
  v11 = *(_QWORD *)v9;
  *(_QWORD *)(v9 + 8) = a2;
  v10 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v9 = v10 | v11 & 7;
  *(_QWORD *)(v10 + 8) = v9;
  *a2 = v9 | *a2 & 7;
  v12 = *(_QWORD *)(a3 + 8);
  if ( v12 )
    sub_2E882B0(v9, (__int64)v8, v12);
  v13 = *(_QWORD *)(a3 + 16);
  if ( v13 )
    sub_2E88680(v9, (__int64)v8, v13);
  return v8;
}
