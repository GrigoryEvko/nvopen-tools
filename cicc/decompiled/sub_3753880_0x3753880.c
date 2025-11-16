// Function: sub_3753880
// Address: 0x3753880
//
__int64 __fastcall sub_3753880(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rax
  const __m128i *v8; // roff
  __m128i v9; // xmm0
  unsigned __int8 *v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // r12
  unsigned int *v14; // rax
  __int64 v15; // rdx
  _QWORD *v16; // [rsp+0h] [rbp-E0h]
  _QWORD *v17; // [rsp+0h] [rbp-E0h]
  _QWORD *v18; // [rsp+0h] [rbp-E0h]
  _QWORD *v20; // [rsp+18h] [rbp-C8h]
  __int64 v21; // [rsp+20h] [rbp-C0h]
  _QWORD *v22; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v23; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int8 *v24; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int8 *v25; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD *v26; // [rsp+48h] [rbp-98h]
  __m128i *v27; // [rsp+50h] [rbp-90h]
  __int64 v28; // [rsp+58h] [rbp-88h]
  __m128i v29; // [rsp+60h] [rbp-80h] BYREF
  __int64 v30; // [rsp+70h] [rbp-70h]
  __m128i v31; // [rsp+80h] [rbp-60h] BYREF
  __int64 v32; // [rsp+90h] [rbp-50h]
  _QWORD *v33; // [rsp+98h] [rbp-48h]
  __int64 v34; // [rsp+A0h] [rbp-40h]

  v20 = *(_QWORD **)(a2 + 32);
  v4 = *(_QWORD **)(a2 + 40);
  v5 = *(unsigned __int8 **)(a2 + 48);
  v22 = v4;
  v23 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v23, (__int64)v5, 1);
  v6 = a1[2];
  v27 = &v29;
  v7 = *(_QWORD *)(v6 + 8);
  v28 = 0x100000000LL;
  v21 = v7 - 560;
  v8 = *(const __m128i **)(a2 + 8);
  v9 = _mm_loadu_si128(v8);
  LODWORD(v28) = 1;
  v29 = v9;
  v30 = v8[1].m128i_i64[0];
  if ( v22 && v29.m128i_i32[0] == 1 && *(_BYTE *)v29.m128i_i64[1] == 17 )
  {
    v22 = (_QWORD *)sub_B0E7F0(v22, (unsigned __int64 *)v29.m128i_i64[1]);
    v14 = (unsigned int *)v27;
    v27->m128i_i32[0] = 1;
    *((_QWORD *)v14 + 1) = v15;
  }
  v24 = v23;
  if ( !v23 )
  {
    v31.m128i_i64[0] = 0;
    goto LABEL_30;
  }
  sub_B96E90((__int64)&v24, (__int64)v23, 1);
  v31.m128i_i64[0] = (__int64)v24;
  if ( !v24 )
  {
LABEL_30:
    v31.m128i_i64[1] = 0;
    v10 = (unsigned __int8 *)*a1;
    v32 = 0;
    v25 = 0;
    goto LABEL_9;
  }
  sub_B976B0((__int64)&v24, v24, (__int64)&v31);
  v10 = (unsigned __int8 *)*a1;
  v24 = 0;
  v31.m128i_i64[1] = 0;
  v32 = 0;
  v25 = (unsigned __int8 *)v31.m128i_i64[0];
  if ( v31.m128i_i64[0] )
    sub_B96E90((__int64)&v25, v31.m128i_i64[0], 1);
LABEL_9:
  v11 = sub_2E7B380(v10, v21, &v25, 0);
  if ( v31.m128i_i64[1] )
  {
    v16 = v11;
    sub_2E882B0((__int64)v11, (__int64)v10, v31.m128i_i64[1]);
    v11 = v16;
  }
  if ( v32 )
  {
    v17 = v11;
    sub_2E88680((__int64)v11, (__int64)v10, v32);
    v11 = v17;
  }
  if ( v25 )
  {
    v18 = v11;
    sub_B91220((__int64)&v25, (__int64)v25);
    v11 = v18;
  }
  v25 = v10;
  v26 = v11;
  if ( v31.m128i_i64[0] )
    sub_B91220((__int64)&v31, v31.m128i_i64[0]);
  if ( v24 )
    sub_B91220((__int64)&v24, (__int64)v24);
  sub_3753340(a1, (__int64 *)&v25, v21, (unsigned int *)v27, (unsigned int)v28, a3);
  if ( *(_BYTE *)(a2 + 60) )
  {
    v31.m128i_i64[0] = 1;
    v32 = 0;
    v33 = 0;
  }
  else
  {
    v31 = 0u;
    v32 = 0;
    v33 = 0;
    v34 = 0;
  }
  sub_2E8EAD0((__int64)v26, (__int64)v25, &v31);
  v31.m128i_i64[0] = 14;
  v32 = 0;
  v33 = v20;
  sub_2E8EAD0((__int64)v26, (__int64)v25, &v31);
  v31.m128i_i64[0] = 14;
  v32 = 0;
  v33 = v22;
  sub_2E8EAD0((__int64)v26, (__int64)v25, &v31);
  v12 = (__int64)v26;
  if ( v27 != &v29 )
    _libc_free((unsigned __int64)v27);
  if ( v23 )
    sub_B91220((__int64)&v23, (__int64)v23);
  return v12;
}
