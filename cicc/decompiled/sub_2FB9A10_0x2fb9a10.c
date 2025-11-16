// Function: sub_2FB9A10
// Address: 0x2fb9a10
//
unsigned __int64 __fastcall sub_2FB9A10(
        __int64 a1,
        __int32 a2,
        __int32 a3,
        __int64 a4,
        unsigned __int64 *a5,
        __int16 a6,
        int a7,
        char a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v11; // r12
  _QWORD *v12; // r13
  __int64 v13; // r15
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned int v16; // ebx
  unsigned __int64 v21; // [rsp+28h] [rbp-98h]
  __int64 v22; // [rsp+38h] [rbp-88h] BYREF
  __int64 v23; // [rsp+40h] [rbp-80h] BYREF
  __int64 v24; // [rsp+48h] [rbp-78h]
  __int64 v25; // [rsp+50h] [rbp-70h]
  __m128i v26; // [rsp+60h] [rbp-60h] BYREF
  __int64 v27; // [rsp+70h] [rbp-50h]
  __int64 v28; // [rsp+78h] [rbp-48h]
  __int64 v29; // [rsp+80h] [rbp-40h]

  v11 = a9;
  v12 = *(_QWORD **)(a4 + 32);
  v21 = a9 & 0xFFFFFFFFFFFFFFF8LL;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26.m128i_i64[0] = 0;
  v13 = (__int64)sub_2E7B380(v12, a10, (unsigned __int8 **)&v26, 0);
  if ( v26.m128i_i64[0] )
    sub_B91220((__int64)&v26, v26.m128i_i64[0]);
  sub_2E31040((__int64 *)(a4 + 40), v13);
  v14 = *a5;
  v15 = *(_QWORD *)v13;
  *(_QWORD *)(v13 + 8) = a5;
  v14 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v13 = v14 | v15 & 7;
  *(_QWORD *)(v14 + 8) = v13;
  *a5 = v13 | *a5 & 7;
  if ( v24 )
    sub_2E882B0(v13, (__int64)v12, v24);
  if ( v25 )
    sub_2E88680(v13, (__int64)v12, v25);
  v27 = 0;
  v16 = (a6 & 0xFFF) << 8;
  v28 = 0;
  v29 = 0;
  v26.m128i_i32[2] = a3;
  v26.m128i_i32[1] = (v21 == 0) | (-2 * (v21 == 0) + 2) & 2;
  v26.m128i_i32[0] = v16 | 0x10000000;
  sub_2E8EAD0(v13, (__int64)v12, &v26);
  v27 = 0;
  v26.m128i_i32[2] = a2;
  v28 = 0;
  v29 = 0;
  v26.m128i_i64[0] = v16;
  sub_2E8EAD0(v13, (__int64)v12, &v26);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  if ( !v21 )
    return sub_2E192D0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL), v13, a8) & 0xFFFFFFFFFFFFFFF8LL | 4;
  sub_2E89030((__int64 *)v13);
  return v11;
}
