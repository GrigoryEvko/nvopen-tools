// Function: sub_2E8FEC0
// Address: 0x2e8fec0
//
_QWORD *__fastcall sub_2E8FEC0(
        _QWORD *a1,
        unsigned __int8 **a2,
        __int64 a3,
        char a4,
        __int32 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int8 *v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // r13
  unsigned __int8 *v15; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int8 *v16; // [rsp+20h] [rbp-80h] BYREF
  __int64 v17; // [rsp+28h] [rbp-78h]
  __int64 v18; // [rsp+30h] [rbp-70h]
  __m128i v19; // [rsp+40h] [rbp-60h] BYREF
  __int64 v20; // [rsp+50h] [rbp-50h]
  __int64 v21; // [rsp+58h] [rbp-48h]
  __int64 v22; // [rsp+60h] [rbp-40h]

  v10 = *a2;
  v15 = v10;
  if ( !v10 )
  {
    v16 = 0;
    goto LABEL_20;
  }
  sub_B96E90((__int64)&v15, (__int64)v10, 1);
  v16 = v15;
  if ( !v15 )
  {
LABEL_20:
    v17 = 0;
    v18 = 0;
    v19.m128i_i64[0] = 0;
    goto LABEL_5;
  }
  sub_B976B0((__int64)&v15, v15, (__int64)&v16);
  v15 = 0;
  v17 = 0;
  v18 = 0;
  v19.m128i_i64[0] = (__int64)v16;
  if ( v16 )
    sub_B96E90((__int64)&v19, (__int64)v16, 1);
LABEL_5:
  v11 = sub_2E7B380(a1, a3, (unsigned __int8 **)&v19, 0);
  v12 = (__int64)v11;
  if ( v17 )
    sub_2E882B0((__int64)v11, (__int64)a1, v17);
  if ( v18 )
    sub_2E88680(v12, (__int64)a1, v18);
  if ( v19.m128i_i64[0] )
    sub_B91220((__int64)&v19, v19.m128i_i64[0]);
  v19.m128i_i32[2] = a5;
  v19.m128i_i64[0] = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  sub_2E8EAD0(v12, (__int64)a1, &v19);
  if ( v16 )
    sub_B91220((__int64)&v16, (__int64)v16);
  if ( v15 )
    sub_B91220((__int64)&v15, (__int64)v15);
  if ( a4 )
  {
    v19.m128i_i64[0] = 1;
    v20 = 0;
    v21 = 0;
  }
  else
  {
    v19 = 0u;
    v20 = 0;
    v21 = 0;
    v22 = 0;
  }
  sub_2E8EAD0(v12, (__int64)a1, &v19);
  v21 = a6;
  v19.m128i_i64[0] = 14;
  v20 = 0;
  sub_2E8EAD0(v12, (__int64)a1, &v19);
  v19.m128i_i64[0] = 14;
  v21 = a7;
  v20 = 0;
  sub_2E8EAD0(v12, (__int64)a1, &v19);
  return a1;
}
