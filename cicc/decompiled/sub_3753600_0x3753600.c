// Function: sub_3753600
// Address: 0x3753600
//
__int64 __fastcall sub_3753600(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  unsigned __int8 *v7; // r15
  _QWORD *v8; // rax
  __int64 v9; // r12
  __int64 v11; // rax
  _QWORD *v12; // [rsp+8h] [rbp-A8h]
  _QWORD *v13; // [rsp+8h] [rbp-A8h]
  _QWORD *v14; // [rsp+8h] [rbp-A8h]
  __int64 v16; // [rsp+18h] [rbp-98h]
  __int64 v17; // [rsp+20h] [rbp-90h]
  __int64 v18; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v19; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int8 *v20; // [rsp+38h] [rbp-78h] BYREF
  unsigned __int8 *v21; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v22; // [rsp+48h] [rbp-68h]
  __m128i v23; // [rsp+50h] [rbp-60h] BYREF
  __int64 v24; // [rsp+60h] [rbp-50h]
  __int64 v25; // [rsp+68h] [rbp-48h]

  v16 = *(_QWORD *)(a2 + 32);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(unsigned __int8 **)(a2 + 48);
  v17 = v4;
  v19 = v5;
  if ( !v5 )
  {
    v11 = a1[2];
    v20 = 0;
    v18 = *(_QWORD *)(v11 + 8) - 600LL;
    goto LABEL_20;
  }
  sub_B96E90((__int64)&v19, (__int64)v5, 1);
  v6 = *(_QWORD *)(a1[2] + 8);
  v20 = v19;
  v18 = v6 - 600;
  if ( !v19 )
  {
LABEL_20:
    v23.m128i_i64[0] = 0;
    goto LABEL_21;
  }
  sub_B96E90((__int64)&v20, (__int64)v19, 1);
  v23.m128i_i64[0] = (__int64)v20;
  if ( !v20 )
  {
LABEL_21:
    v23.m128i_i64[1] = 0;
    v7 = (unsigned __int8 *)*a1;
    v24 = 0;
    v21 = 0;
    goto LABEL_6;
  }
  sub_B976B0((__int64)&v20, v20, (__int64)&v23);
  v20 = 0;
  v23.m128i_i64[1] = 0;
  v7 = (unsigned __int8 *)*a1;
  v24 = 0;
  v21 = (unsigned __int8 *)v23.m128i_i64[0];
  if ( v23.m128i_i64[0] )
    sub_B96E90((__int64)&v21, v23.m128i_i64[0], 1);
LABEL_6:
  v8 = sub_2E7B380(v7, v18, &v21, 0);
  if ( v23.m128i_i64[1] )
  {
    v12 = v8;
    sub_2E882B0((__int64)v8, (__int64)v7, v23.m128i_i64[1]);
    v8 = v12;
  }
  if ( v24 )
  {
    v13 = v8;
    sub_2E88680((__int64)v8, (__int64)v7, v24);
    v8 = v13;
  }
  if ( v21 )
  {
    v14 = v8;
    sub_B91220((__int64)&v21, (__int64)v21);
    v8 = v14;
  }
  v21 = v7;
  v22 = v8;
  if ( v23.m128i_i64[0] )
    sub_B91220((__int64)&v23, v23.m128i_i64[0]);
  if ( v20 )
    sub_B91220((__int64)&v20, (__int64)v20);
  v23.m128i_i64[0] = 14;
  v24 = 0;
  v25 = v16;
  sub_2E8EAD0((__int64)v22, (__int64)v21, &v23);
  v23.m128i_i64[0] = 14;
  v24 = 0;
  v25 = v17;
  sub_2E8EAD0((__int64)v22, (__int64)v21, &v23);
  sub_3753340(a1, (__int64 *)&v21, v18, *(unsigned int **)(a2 + 8), *(_QWORD *)a2, a3);
  v9 = (__int64)v22;
  if ( v19 )
    sub_B91220((__int64)&v19, (__int64)v19);
  return v9;
}
