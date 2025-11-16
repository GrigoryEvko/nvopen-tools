// Function: sub_356E080
// Address: 0x356e080
//
_QWORD *__fastcall sub_356E080(
        unsigned int a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int32 v9; // eax
  _QWORD *v10; // r13
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int32 v17; // [rsp+Ch] [rbp-94h]
  __int64 v18; // [rsp+18h] [rbp-88h] BYREF
  __int64 v19; // [rsp+20h] [rbp-80h] BYREF
  __int64 v20; // [rsp+28h] [rbp-78h]
  __int64 v21; // [rsp+30h] [rbp-70h]
  __m128i v22; // [rsp+40h] [rbp-60h] BYREF
  __int64 v23; // [rsp+50h] [rbp-50h]
  __int64 v24; // [rsp+58h] [rbp-48h]
  __int64 v25; // [rsp+60h] [rbp-40h]

  v9 = sub_2EC0720(a6, a4, a5, byte_3F871B3, 0, a6);
  v10 = *(_QWORD **)(a2 + 32);
  v17 = v9;
  v11 = *(_QWORD *)(a7 + 8);
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22.m128i_i64[0] = 0;
  v12 = (__int64)sub_2E7B380(v10, v11 - 40LL * a1, (unsigned __int8 **)&v22, 0);
  if ( v22.m128i_i64[0] )
    sub_B91220((__int64)&v22, v22.m128i_i64[0]);
  sub_2E31040((__int64 *)(a2 + 40), v12);
  v13 = *a3;
  v14 = *(_QWORD *)v12;
  *(_QWORD *)(v12 + 8) = a3;
  v13 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v12 = v13 | v14 & 7;
  *(_QWORD *)(v13 + 8) = v12;
  v15 = v20;
  *a3 = v12 | *a3 & 7;
  if ( v15 )
    sub_2E882B0(v12, (__int64)v10, v15);
  if ( v21 )
    sub_2E88680(v12, (__int64)v10, v21);
  v22.m128i_i64[0] = 0x10000000;
  v23 = 0;
  v22.m128i_i32[2] = v17;
  v24 = 0;
  v25 = 0;
  sub_2E8EAD0(v12, (__int64)v10, &v22);
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v10;
}
