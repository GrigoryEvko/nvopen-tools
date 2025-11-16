// Function: sub_2265100
// Address: 0x2265100
//
unsigned __int64 __fastcall sub_2265100(unsigned __int64 *a1, __m128i *a2)
{
  char *v4; // r14
  char *v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rcx
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  __m128i *v10; // rax
  __m128i v11; // xmm1
  __int64 v12; // rcx
  __int64 v13; // rdx
  __m128i v14; // xmm0
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 *v17; // rdx
  unsigned __int64 result; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned __int64 v22; // r14
  __int64 v23; // rax
  const void *v24; // rsi
  _QWORD *v25; // r15
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  char *v29; // r14
  size_t v30; // rdx
  unsigned __int64 v31; // [rsp+8h] [rbp-38h]

  v4 = (char *)a1[9];
  v5 = (char *)a1[5];
  v6 = v4 - v5;
  v7 = (v4 - v5) >> 3;
  if ( 0xCCCCCCCCCCCCCCCDLL * ((__int64)(a1[6] - a1[7]) >> 3)
     + 4 * (3 * v7 - 3)
     - 0x3333333333333333LL * ((__int64)(a1[4] - a1[2]) >> 3) == 0x333333333333333LL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  v8 = *a1;
  v9 = a1[1];
  if ( v9 - ((__int64)&v4[-*a1] >> 3) <= 1 )
  {
    v20 = v7 + 2;
    if ( v9 > 2 * (v7 + 2) )
    {
      v29 = v4 + 8;
      v25 = (_QWORD *)(v8 + 8 * ((v9 - v20) >> 1));
      v30 = v29 - v5;
      if ( v5 <= (char *)v25 )
      {
        if ( v5 != v29 )
          memmove((char *)v25 + v6 + 8 - v30, v5, v30);
      }
      else if ( v5 != v29 )
      {
        memmove(v25, v5, v30);
      }
    }
    else
    {
      v21 = 1;
      if ( v9 )
        v21 = a1[1];
      v22 = v9 + v21 + 2;
      if ( v22 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v8, v5, v9);
      v23 = sub_22077B0(8 * v22);
      v24 = (const void *)a1[5];
      v31 = v23;
      v25 = (_QWORD *)(v23 + 8 * ((v22 - v20) >> 1));
      v26 = a1[9] + 8;
      if ( (const void *)v26 != v24 )
        memmove(v25, v24, v26 - (_QWORD)v24);
      j_j___libc_free_0(*a1);
      a1[1] = v22;
      *a1 = v31;
    }
    a1[5] = (unsigned __int64)v25;
    v27 = *v25;
    v4 = (char *)v25 + v6;
    a1[9] = (unsigned __int64)v25 + v6;
    a1[3] = v27;
    a1[4] = v27 + 480;
    v28 = *(_QWORD *)((char *)v25 + v6);
    a1[7] = v28;
    a1[8] = v28 + 480;
  }
  *((_QWORD *)v4 + 1) = sub_22077B0(0x1E0u);
  v10 = (__m128i *)a1[6];
  if ( v10 )
  {
    v11 = _mm_loadu_si128(v10);
    v12 = v10[1].m128i_i64[1];
    v10[1].m128i_i64[0] = 0;
    v13 = a2[1].m128i_i64[0];
    v14 = _mm_loadu_si128(a2);
    a2[1].m128i_i64[0] = 0;
    *a2 = v11;
    v10[1].m128i_i64[0] = v13;
    v15 = a2[1].m128i_i64[1];
    *v10 = v14;
    v10[1].m128i_i64[1] = v15;
    v16 = a2[2].m128i_i64[0];
    a2[1].m128i_i64[1] = v12;
    v10[2].m128i_i64[0] = v16;
  }
  v17 = (unsigned __int64 *)(a1[9] + 8);
  a1[9] = (unsigned __int64)v17;
  result = *v17;
  v19 = *v17 + 480;
  a1[7] = result;
  a1[8] = v19;
  a1[6] = result;
  return result;
}
