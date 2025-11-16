// Function: sub_2BEC920
// Address: 0x2bec920
//
__int8 *__fastcall sub_2BEC920(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  __m128i v3; // xmm2
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  unsigned __int64 v6; // rdi
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rcx
  __m128i *v10; // rax
  __int8 *result; // rax
  unsigned __int64 *v12; // r12
  __m128i *v13; // rsi
  __m128i v14; // xmm0
  unsigned __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rax
  const __m128i *v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __m128i v22; // xmm1
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __m128i v30; // [rsp+0h] [rbp-50h] BYREF
  __m128i v31; // [rsp+10h] [rbp-40h] BYREF
  __m128i v32; // [rsp+20h] [rbp-30h] BYREF

  if ( (unsigned __int8)sub_2BEE440() )
  {
LABEL_2:
    v2 = a1[44];
    if ( v2 == a1[45] )
    {
      v18 = *(const __m128i **)(a1[47] - 8LL);
      v30 = _mm_loadu_si128(v18 + 30);
      v31.m128i_i64[0] = v18[31].m128i_i64[0];
      j_j___libc_free_0(v2);
      v19 = (__int64 *)(a1[47] - 8LL);
      a1[47] = v19;
      v20 = *v19;
      v21 = *v19 + 504;
      a1[45] = v20;
      a1[46] = v21;
      a1[44] = v20 + 480;
    }
    else
    {
      v3 = _mm_loadu_si128((const __m128i *)(v2 - 24));
      v4 = v2 - 24;
      v30 = v3;
      v5 = *(_QWORD *)(v4 + 16);
      a1[44] = v4;
      v31.m128i_i64[0] = v5;
    }
    sub_2BEC920(a1);
    v6 = a1[44];
    if ( v6 == a1[45] )
    {
      v26 = *(_QWORD *)(a1[47] - 8LL);
      v7 = *(_QWORD *)(v26 + 488);
      v8 = *(_QWORD *)(v26 + 496);
      j_j___libc_free_0(v6);
      v27 = (__int64 *)(a1[47] - 8LL);
      a1[47] = v27;
      v28 = *v27;
      v29 = *v27 + 504;
      a1[45] = v28;
      a1[46] = v29;
      a1[44] = v28 + 480;
    }
    else
    {
      v7 = *(_QWORD *)(v6 - 16);
      v8 = *(_QWORD *)(v6 - 8);
      a1[44] = v6 - 24;
    }
    *(_QWORD *)(*(_QWORD *)(v30.m128i_i64[0] + 56) + 48 * v31.m128i_i64[0] + 8) = v7;
    v9 = a1[46];
    v10 = (__m128i *)a1[44];
    v31.m128i_i64[0] = v8;
    if ( v10 == (__m128i *)(v9 - 24) )
      return (__int8 *)sub_2BE3350(a1 + 38, &v30);
    if ( v10 )
    {
      *v10 = _mm_loadu_si128(&v30);
      v10[1].m128i_i64[0] = v31.m128i_i64[0];
      v10 = (__m128i *)a1[44];
    }
    result = &v10[1].m128i_i8[8];
    a1[44] = result;
    return result;
  }
  if ( (unsigned __int8)sub_2BEC0F0((__int64)a1) )
  {
    while ( (unsigned __int8)sub_2BE6C70((__int64)a1) && (unsigned __int8)sub_2BE6C70((__int64)a1) )
      ;
    goto LABEL_2;
  }
  v12 = (unsigned __int64 *)a1[32];
  v30.m128i_i32[0] = 10;
  v30.m128i_i64[1] = -1;
  v13 = (__m128i *)v12[8];
  if ( v13 == (__m128i *)v12[9] )
  {
    sub_2BE00E0(v12 + 7, v13, &v30);
    v15 = v12[8];
  }
  else
  {
    if ( v13 )
    {
      *v13 = _mm_loadu_si128(&v30);
      v14 = _mm_loadu_si128(&v31);
      v13[1] = v14;
      v13[2] = _mm_loadu_si128(&v32);
      if ( v30.m128i_i32[0] == 11 )
      {
        v13[2].m128i_i64[0] = 0;
        v22 = _mm_loadu_si128(&v31);
        v31 = v14;
        v13[1] = v22;
        v23 = v32.m128i_i64[0];
        v32.m128i_i64[0] = 0;
        v24 = v13[2].m128i_i64[1];
        v13[2].m128i_i64[0] = v23;
        v25 = v32.m128i_i64[1];
        v32.m128i_i64[1] = v24;
        v13[2].m128i_i64[1] = v25;
      }
      v13 = (__m128i *)v12[8];
    }
    v15 = (unsigned __int64)&v13[3];
    v12[8] = v15;
  }
  v16 = v15 - v12[7];
  if ( (unsigned __int64)v16 > 0x493E00 )
    abort();
  if ( v30.m128i_i32[0] == 11 && v32.m128i_i64[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v32.m128i_i64[0])(&v31, &v31, 3);
  v17 = a1[32];
  v30.m128i_i64[1] = 0xAAAAAAAAAAAAAAABLL * (v16 >> 4) - 1;
  v31.m128i_i64[0] = v30.m128i_i64[1];
  v30.m128i_i64[0] = v17;
  return sub_2BE3490(a1 + 38, &v30);
}
