// Function: sub_B17F60
// Address: 0xb17f60
//
__int64 __fastcall sub_B17F60(__int64 a1, __m128i *a2)
{
  const __m128i *v3; // rdi
  __int64 result; // rax
  const __m128i *v5; // rcx
  const __m128i *v6; // rax
  __int64 m128i_i64; // rdx
  unsigned __int64 v8; // rdi
  const __m128i *v9; // rcx
  __m128i v10; // xmm0
  __int64 v11; // rcx
  const __m128i *v12; // r12
  const __m128i *v13; // rbx
  const __m128i *v14; // rdi

  v3 = *(const __m128i **)a1;
  result = *(unsigned int *)(a1 + 8);
  v5 = &v3[5 * result];
  if ( v3 != v5 )
  {
    v6 = v3 + 3;
    m128i_i64 = (__int64)v3[1].m128i_i64;
    v8 = (unsigned __int64)v3[8].m128i_u64 + (((char *)v5 - (char *)v3 - 80) & 0xFFFFFFFFFFFFFFF0LL);
    do
    {
      if ( a2 )
      {
        a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
        v11 = v6[-3].m128i_i64[0];
        if ( v11 == m128i_i64 )
        {
          a2[1] = _mm_loadu_si128(v6 - 2);
        }
        else
        {
          a2->m128i_i64[0] = v11;
          a2[1].m128i_i64[0] = v6[-2].m128i_i64[0];
        }
        a2->m128i_i64[1] = v6[-3].m128i_i64[1];
        v6[-3].m128i_i64[0] = m128i_i64;
        v6[-3].m128i_i64[1] = 0;
        v6[-2].m128i_i8[0] = 0;
        a2[2].m128i_i64[0] = (__int64)a2[3].m128i_i64;
        v9 = (const __m128i *)v6[-1].m128i_i64[0];
        if ( v9 == v6 )
        {
          a2[3] = _mm_loadu_si128(v6);
        }
        else
        {
          a2[2].m128i_i64[0] = (__int64)v9;
          a2[3].m128i_i64[0] = v6->m128i_i64[0];
        }
        a2[2].m128i_i64[1] = v6[-1].m128i_i64[1];
        v10 = _mm_loadu_si128(v6 + 1);
        v6[-1].m128i_i64[0] = (__int64)v6;
        v6[-1].m128i_i64[1] = 0;
        v6->m128i_i8[0] = 0;
        a2[4] = v10;
      }
      v6 += 5;
      a2 += 5;
      m128i_i64 += 80;
    }
    while ( v6 != (const __m128i *)v8 );
    result = *(unsigned int *)(a1 + 8);
    v12 = *(const __m128i **)a1;
    v13 = (const __m128i *)(*(_QWORD *)a1 + 80 * result);
    while ( v13 != v12 )
    {
      while ( 1 )
      {
        v13 -= 5;
        v14 = (const __m128i *)v13[2].m128i_i64[0];
        if ( v14 != &v13[3] )
          j_j___libc_free_0(v14, v13[3].m128i_i64[0] + 1);
        result = (__int64)v13[1].m128i_i64;
        if ( (const __m128i *)v13->m128i_i64[0] == &v13[1] )
          break;
        result = j_j___libc_free_0(v13->m128i_i64[0], v13[1].m128i_i64[0] + 1);
        if ( v13 == v12 )
          return result;
      }
    }
  }
  return result;
}
