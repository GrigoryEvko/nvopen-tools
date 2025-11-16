// Function: sub_D52090
// Address: 0xd52090
//
__m128i *__fastcall sub_D52090(__int64 *a1, const __m128i *a2)
{
  __m128i *v4; // rax
  __m128i *result; // rax
  char *v6; // r14
  char *v7; // rsi
  __int64 v8; // r13
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rdi
  __m128i *v12; // rax
  __m128i **v13; // rdx
  __int64 m128i_i64; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned __int64 v17; // r14
  __int64 v18; // rax
  const void *v19; // rsi
  _QWORD *v20; // r15
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  char *v24; // r14
  size_t v25; // rdx
  __int64 v26; // [rsp+8h] [rbp-38h]

  v4 = (__m128i *)a1[6];
  if ( v4 == (__m128i *)(a1[8] - 32) )
  {
    v6 = (char *)a1[9];
    v7 = (char *)a1[5];
    v8 = v6 - v7;
    v9 = (v6 - v7) >> 3;
    if ( (((__int64)v4->m128i_i64 - a1[7]) >> 5) + 16 * (v9 - 1) + ((a1[4] - a1[2]) >> 5) == 0x3FFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    v10 = a1[1];
    v11 = *a1;
    if ( v10 - ((__int64)&v6[-v11] >> 3) <= 1 )
    {
      v15 = v9 + 2;
      if ( v10 > 2 * (v9 + 2) )
      {
        v24 = v6 + 8;
        v20 = (_QWORD *)(v11 + 8 * ((v10 - v15) >> 1));
        v25 = v24 - v7;
        if ( v7 <= (char *)v20 )
        {
          if ( v7 != v24 )
            memmove((char *)v20 + v8 + 8 - v25, v7, v25);
        }
        else if ( v7 != v24 )
        {
          memmove(v20, v7, v25);
        }
      }
      else
      {
        v16 = 1;
        if ( v10 )
          v16 = v10;
        v17 = v10 + v16 + 2;
        if ( v17 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v11, v7, v10);
        v18 = sub_22077B0(8 * v17);
        v19 = (const void *)a1[5];
        v26 = v18;
        v20 = (_QWORD *)(v18 + 8 * ((v17 - v15) >> 1));
        v21 = a1[9] + 8;
        if ( (const void *)v21 != v19 )
          memmove(v20, v19, v21 - (_QWORD)v19);
        j_j___libc_free_0(*a1, 8 * a1[1]);
        a1[1] = v17;
        *a1 = v26;
      }
      a1[5] = (__int64)v20;
      v22 = *v20;
      v6 = (char *)v20 + v8;
      a1[9] = (__int64)v20 + v8;
      a1[3] = v22;
      a1[4] = v22 + 512;
      v23 = *(_QWORD *)((char *)v20 + v8);
      a1[7] = v23;
      a1[8] = v23 + 512;
    }
    *((_QWORD *)v6 + 1) = sub_22077B0(512);
    v12 = (__m128i *)a1[6];
    if ( v12 )
    {
      *v12 = _mm_loadu_si128(a2);
      v12[1] = _mm_loadu_si128(a2 + 1);
    }
    v13 = (__m128i **)(a1[9] + 8);
    a1[9] = (__int64)v13;
    result = *v13;
    m128i_i64 = (__int64)(*v13)[32].m128i_i64;
    a1[7] = (__int64)result;
    a1[8] = m128i_i64;
    a1[6] = (__int64)result;
  }
  else
  {
    if ( v4 )
    {
      *v4 = _mm_loadu_si128(a2);
      v4[1] = _mm_loadu_si128(a2 + 1);
      v4 = (__m128i *)a1[6];
    }
    result = v4 + 2;
    a1[6] = (__int64)result;
  }
  return result;
}
