// Function: sub_2555870
// Address: 0x2555870
//
__m128i *__fastcall sub_2555870(_DWORD *a1, __m128i *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v7; // rbx
  __m128i *result; // rax
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // rdi
  __m128i *v13; // r8
  __int64 v14; // rax
  __int64 *m128i_i64; // rdx
  const __m128i *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax

  v7 = a2;
  LODWORD(a2) = a1[2];
  if ( (_DWORD)a2 )
  {
    result = *(__m128i **)a1;
    if ( **(_QWORD **)a1 == 0x7FFFFFFF || result->m128i_i64[1] == 0x7FFFFFFF )
      return result;
  }
  v9 = *a3;
  if ( *a3 == 0x7FFFFFFF || (v10 = a3[1], v10 == 0x7FFFFFFF) )
  {
    a1[2] = 0;
    v17 = 0;
    if ( !a1[3] )
    {
      sub_C8D5F0((__int64)a1, a1 + 4, 1u, 0x10u, a5, a6);
      v17 = 16LL * (unsigned int)a1[2];
    }
    *(__m128i *)(*(_QWORD *)a1 + v17) = _mm_load_si128((const __m128i *)&xmmword_438A6D0);
    result = *(__m128i **)a1;
    ++a1[2];
    return result;
  }
  v11 = *(_QWORD *)a1;
  v12 = 16LL * (unsigned int)a2;
  v13 = (__m128i *)(*(_QWORD *)a1 + v12);
  v14 = v13 - v7;
  if ( (char *)v13 - (char *)v7 > 0 )
  {
    do
    {
      while ( 1 )
      {
        a4 = v14 >> 1;
        m128i_i64 = v7[v14 >> 1].m128i_i64;
        if ( v9 <= *m128i_i64 )
          break;
        v7 = (__m128i *)(m128i_i64 + 2);
        v14 = v14 - a4 - 1;
LABEL_12:
        if ( v14 <= 0 )
          goto LABEL_13;
      }
      if ( v9 == *m128i_i64 && v10 > m128i_i64[1] )
      {
        v7 = (__m128i *)(m128i_i64 + 2);
        v14 = v14 - a4 - 1;
        goto LABEL_12;
      }
      v14 >>= 1;
    }
    while ( a4 > 0 );
  }
LABEL_13:
  if ( v13 == v7 )
  {
    sub_2555810((__int64)a1, v9, v10, a4, (__int64)v13, (unsigned int)a2);
    return (__m128i *)(*(_QWORD *)a1 + 16LL * (unsigned int)a1[2] - 16);
  }
  if ( v7->m128i_i64[0] != v9 )
  {
    if ( (unsigned __int64)(unsigned int)a2 + 1 > (unsigned int)a1[3] )
    {
      sub_C8D5F0((__int64)a1, a1 + 4, (unsigned int)a2 + 1LL, 0x10u, (__int64)v13, (unsigned int)a2);
      a2 = (__m128i *)(unsigned int)a1[2];
      v12 = 16LL * (_QWORD)a2;
      v7 = (__m128i *)((char *)v7 + *(_QWORD *)a1 - v11);
      v11 = *(_QWORD *)a1;
      v13 = (__m128i *)(*(_QWORD *)a1 + 16LL * (_QWORD)a2);
    }
    v16 = (const __m128i *)(v11 + v12 - 16);
    if ( v13 )
    {
      *v13 = _mm_loadu_si128(v16);
      v11 = *(_QWORD *)a1;
      a2 = (__m128i *)(unsigned int)a1[2];
      v12 = 16LL * (_QWORD)a2;
      v16 = (const __m128i *)(*(_QWORD *)a1 + 16LL * (_QWORD)a2 - 16);
    }
    if ( v7 != v16 )
    {
      memmove((void *)(v11 + v12 - ((char *)v16 - (char *)v7)), v7, (char *)v16 - (char *)v7);
      LODWORD(a2) = a1[2];
    }
    a1[2] = (_DWORD)a2 + 1;
    v7->m128i_i64[0] = v9;
    v7->m128i_i64[1] = v10;
    return v7;
  }
  v18 = v7->m128i_i64[1];
  if ( v9 == 0xFFFFFFFF80000000LL )
  {
    v20 = v7->m128i_i64[1];
  }
  else
  {
    if ( v18 == 0x7FFFFFFF )
    {
LABEL_35:
      a1[2] = 0;
      sub_2555810((__int64)a1, 0x7FFFFFFF, 0x7FFFFFFF, a4, (__int64)v13, (unsigned int)a2);
      return *(__m128i **)a1;
    }
    v19 = v9 + v10;
    a4 = v9 + v18;
    if ( v9 + v10 < v9 + v18 )
      v19 = v9 + v18;
    v20 = v19 - v9;
    v7->m128i_i64[1] = v20;
  }
  if ( v20 == 0x7FFFFFFF )
    goto LABEL_35;
  return v7;
}
