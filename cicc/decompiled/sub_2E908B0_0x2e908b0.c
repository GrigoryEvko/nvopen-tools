// Function: sub_2E908B0
// Address: 0x2e908b0
//
_QWORD *__fastcall sub_2E908B0(
        _QWORD *a1,
        unsigned __int8 **a2,
        _WORD *a3,
        char a4,
        const __m128i *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  const __m128i *v9; // rbx
  unsigned __int8 *v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // r13
  const __m128i *v13; // r15
  __int32 v14; // eax
  const __m128i *v15; // rdx
  __int64 v18; // rdx
  __int8 v19; // al
  unsigned __int8 *v20; // rsi
  _QWORD *v21; // rax
  __int64 v22; // r13
  unsigned __int8 *v24; // [rsp+10h] [rbp-A0h] BYREF
  unsigned __int8 *v25; // [rsp+18h] [rbp-98h] BYREF
  __m128i v26[2]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v27; // [rsp+40h] [rbp-70h]
  __m128i v28; // [rsp+50h] [rbp-60h] BYREF
  __int64 v29; // [rsp+60h] [rbp-50h]
  __int64 v30; // [rsp+68h] [rbp-48h]
  __int64 v31; // [rsp+70h] [rbp-40h]

  v9 = a5;
  if ( *a3 == 14 )
  {
    v18 = a5[2].m128i_i64[0];
    v19 = a5->m128i_i8[0];
    v26[0] = _mm_loadu_si128(a5);
    v27 = v18;
    v26[1] = _mm_loadu_si128(a5 + 1);
    if ( !v19 )
      return sub_2E8FEC0(a1, a2, (__int64)a3, a4, v26[0].m128i_i32[2], a7, a8);
    v20 = *a2;
    v24 = v20;
    if ( v20 )
    {
      sub_B96E90((__int64)&v24, (__int64)v20, 1);
      v28.m128i_i64[0] = (__int64)v24;
      if ( v24 )
      {
        sub_B976B0((__int64)&v24, v24, (__int64)&v28);
        v24 = 0;
        v28.m128i_i64[1] = 0;
        v29 = 0;
        v25 = (unsigned __int8 *)v28.m128i_i64[0];
        if ( v28.m128i_i64[0] )
          sub_B96E90((__int64)&v25, v28.m128i_i64[0], 1);
        goto LABEL_29;
      }
    }
    else
    {
      v28.m128i_i64[0] = 0;
    }
    v28.m128i_i64[1] = 0;
    v29 = 0;
    v25 = 0;
LABEL_29:
    v21 = sub_2E7B380(a1, (__int64)a3, &v25, 0);
    v22 = (__int64)v21;
    if ( v28.m128i_i64[1] )
      sub_2E882B0((__int64)v21, (__int64)a1, v28.m128i_i64[1]);
    if ( v29 )
      sub_2E88680(v22, (__int64)a1, v29);
    if ( v25 )
      sub_B91220((__int64)&v25, (__int64)v25);
    sub_2E8EAD0(v22, (__int64)a1, v26);
    if ( v28.m128i_i64[0] )
      sub_B91220((__int64)&v28, v28.m128i_i64[0]);
    if ( v24 )
      sub_B91220((__int64)&v24, (__int64)v24);
    if ( a4 )
    {
      v28.m128i_i64[0] = 1;
      v29 = 0;
      v30 = 0;
    }
    else
    {
      v28 = 0u;
      v29 = 0;
      v30 = 0;
      v31 = 0;
    }
    sub_2E8EAD0(v22, (__int64)a1, &v28);
    v28.m128i_i64[0] = 14;
    v30 = a7;
    v29 = 0;
    sub_2E8EAD0(v22, (__int64)a1, &v28);
    v28.m128i_i64[0] = 14;
    v30 = a8;
    v29 = 0;
    sub_2E8EAD0(v22, (__int64)a1, &v28);
    return a1;
  }
  v10 = *a2;
  v25 = v10;
  if ( v10 )
  {
    sub_B96E90((__int64)&v25, (__int64)v10, 1);
    v28.m128i_i64[0] = (__int64)v25;
    if ( v25 )
    {
      sub_B976B0((__int64)&v25, v25, (__int64)&v28);
      v25 = 0;
      v28.m128i_i64[1] = 0;
      v29 = 0;
      v26[0].m128i_i64[0] = v28.m128i_i64[0];
      if ( v28.m128i_i64[0] )
        sub_B96E90((__int64)v26, v28.m128i_i64[0], 1);
      goto LABEL_6;
    }
  }
  else
  {
    v28.m128i_i64[0] = 0;
  }
  v28.m128i_i64[1] = 0;
  v29 = 0;
  v26[0].m128i_i64[0] = 0;
LABEL_6:
  v11 = sub_2E7B380(a1, (__int64)a3, (unsigned __int8 **)v26, 0);
  v12 = (__int64)v11;
  if ( v28.m128i_i64[1] )
    sub_2E882B0((__int64)v11, (__int64)a1, v28.m128i_i64[1]);
  if ( v29 )
    sub_2E88680(v12, (__int64)a1, v29);
  if ( v26[0].m128i_i64[0] )
    sub_B91220((__int64)v26, v26[0].m128i_i64[0]);
  if ( v28.m128i_i64[0] )
    sub_B91220((__int64)&v28, v28.m128i_i64[0]);
  if ( v25 )
    sub_B91220((__int64)&v25, (__int64)v25);
  v28.m128i_i64[0] = 14;
  v30 = a7;
  v29 = 0;
  sub_2E8EAD0(v12, (__int64)a1, &v28);
  v28.m128i_i64[0] = 14;
  v30 = a8;
  v29 = 0;
  sub_2E8EAD0(v12, (__int64)a1, &v28);
  v13 = (const __m128i *)((char *)v9 + 40 * a6);
  while ( v13 != v9 )
  {
    while ( !v9->m128i_i8[0] )
    {
      v14 = v9->m128i_i32[2];
      v9 = (const __m128i *)((char *)v9 + 40);
      v28.m128i_i64[0] = 0;
      v29 = 0;
      v28.m128i_i32[2] = v14;
      v30 = 0;
      v31 = 0;
      sub_2E8EAD0(v12, (__int64)a1, &v28);
      if ( v13 == v9 )
        return a1;
    }
    v15 = v9;
    v9 = (const __m128i *)((char *)v9 + 40);
    sub_2E8EAD0(v12, (__int64)a1, v15);
  }
  return a1;
}
