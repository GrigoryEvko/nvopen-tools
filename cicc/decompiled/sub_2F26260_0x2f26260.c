// Function: sub_2F26260
// Address: 0x2f26260
//
_QWORD *__fastcall sub_2F26260(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int32 a5)
{
  __int64 v8; // rsi
  _QWORD *v9; // r13
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __m128i v17; // [rsp+10h] [rbp-60h] BYREF
  __int64 v18; // [rsp+20h] [rbp-50h]
  __int64 v19; // [rsp+28h] [rbp-48h]
  __int64 v20; // [rsp+30h] [rbp-40h]

  v8 = *a3;
  v9 = *(_QWORD **)(a1 + 32);
  v17.m128i_i64[0] = v8;
  if ( v8 )
    sub_B96E90((__int64)&v17, v8, 1);
  v10 = (__int64)sub_2E7B380(v9, a4, (unsigned __int8 **)&v17, 0);
  if ( v17.m128i_i64[0] )
    sub_B91220((__int64)&v17, v17.m128i_i64[0]);
  sub_2E31040((__int64 *)(a1 + 40), v10);
  v11 = *a2;
  v12 = *(_QWORD *)v10;
  *(_QWORD *)(v10 + 8) = a2;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v10 = v11 | v12 & 7;
  *(_QWORD *)(v11 + 8) = v10;
  *a2 = v10 | *a2 & 7;
  v13 = a3[1];
  if ( v13 )
    sub_2E882B0(v10, (__int64)v9, v13);
  v14 = a3[2];
  if ( v14 )
    sub_2E88680(v10, (__int64)v9, v14);
  v17.m128i_i64[0] = 0x10000000;
  v17.m128i_i32[2] = a5;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  sub_2E8EAD0(v10, (__int64)v9, &v17);
  return v9;
}
