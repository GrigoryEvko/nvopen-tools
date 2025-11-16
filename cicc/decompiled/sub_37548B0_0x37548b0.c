// Function: sub_37548B0
// Address: 0x37548b0
//
__int64 __fastcall sub_37548B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  unsigned __int8 *v3; // rsi
  __int64 v4; // r12
  __int64 v5; // r12
  _QWORD *v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // r12
  __int64 v10; // rax
  unsigned __int8 *v11; // [rsp+8h] [rbp-78h] BYREF
  unsigned __int8 *v12; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int8 *v13; // [rsp+18h] [rbp-68h] BYREF
  __m128i v14; // [rsp+20h] [rbp-60h] BYREF
  __int64 v15; // [rsp+30h] [rbp-50h]
  __int64 v16; // [rsp+38h] [rbp-48h]

  v2 = *a2;
  v3 = (unsigned __int8 *)a2[1];
  v11 = v3;
  if ( !v3 )
  {
    v10 = *(_QWORD *)(a1 + 16);
    v12 = 0;
    v5 = *(_QWORD *)(v10 + 8) - 720LL;
    goto LABEL_20;
  }
  sub_B96E90((__int64)&v11, (__int64)v3, 1);
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL);
  v12 = v11;
  v5 = v4 - 720;
  if ( !v11 )
  {
LABEL_20:
    v14.m128i_i64[0] = 0;
    goto LABEL_21;
  }
  sub_B96E90((__int64)&v12, (__int64)v11, 1);
  v14.m128i_i64[0] = (__int64)v12;
  if ( !v12 )
  {
LABEL_21:
    v14.m128i_i64[1] = 0;
    v6 = *(_QWORD **)a1;
    v15 = 0;
    v13 = 0;
    goto LABEL_6;
  }
  sub_B976B0((__int64)&v12, v12, (__int64)&v14);
  v12 = 0;
  v14.m128i_i64[1] = 0;
  v6 = *(_QWORD **)a1;
  v15 = 0;
  v13 = (unsigned __int8 *)v14.m128i_i64[0];
  if ( v14.m128i_i64[0] )
    sub_B96E90((__int64)&v13, v14.m128i_i64[0], 1);
LABEL_6:
  v7 = sub_2E7B380(v6, v5, &v13, 0);
  v8 = (__int64)v7;
  if ( v14.m128i_i64[1] )
    sub_2E882B0((__int64)v7, (__int64)v6, v14.m128i_i64[1]);
  if ( v15 )
    sub_2E88680(v8, (__int64)v6, v15);
  if ( v13 )
    sub_B91220((__int64)&v13, (__int64)v13);
  if ( v14.m128i_i64[0] )
    sub_B91220((__int64)&v14, v14.m128i_i64[0]);
  if ( v12 )
    sub_B91220((__int64)&v12, (__int64)v12);
  v16 = v2;
  v14.m128i_i64[0] = 14;
  v15 = 0;
  sub_2E8EAD0(v8, (__int64)v6, &v14);
  if ( v11 )
    sub_B91220((__int64)&v11, (__int64)v11);
  return v8;
}
