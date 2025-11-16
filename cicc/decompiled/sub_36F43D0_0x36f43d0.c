// Function: sub_36F43D0
// Address: 0x36f43d0
//
_QWORD *__fastcall sub_36F43D0(_QWORD *a1, __int64 a2, __int64 a3, __int32 a4)
{
  unsigned __int8 *v7; // rsi
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rdx
  unsigned __int8 *v13; // [rsp+8h] [rbp-68h] BYREF
  __m128i v14; // [rsp+10h] [rbp-60h] BYREF
  __int64 v15; // [rsp+20h] [rbp-50h]
  __int64 v16; // [rsp+28h] [rbp-48h]
  __int64 v17; // [rsp+30h] [rbp-40h]

  v7 = *(unsigned __int8 **)a2;
  v13 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v13, (__int64)v7, 1);
  v8 = sub_2E7B380(a1, a3, &v13, 0);
  v9 = *(_QWORD *)(a2 + 8);
  v10 = (__int64)v8;
  if ( v9 )
    sub_2E882B0((__int64)v8, (__int64)a1, v9);
  v11 = *(_QWORD *)(a2 + 16);
  if ( v11 )
    sub_2E88680(v10, (__int64)a1, v11);
  v14.m128i_i32[2] = a4;
  v14.m128i_i64[0] = 0x10000000;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  sub_2E8EAD0(v10, (__int64)a1, &v14);
  if ( v13 )
    sub_B91220((__int64)&v13, (__int64)v13);
  return a1;
}
