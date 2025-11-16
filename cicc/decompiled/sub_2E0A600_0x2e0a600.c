// Function: sub_2E0A600
// Address: 0x2e0a600
//
__int64 __fastcall sub_2E0A600(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __m128i *v6; // rdx
  const __m128i *v7; // rsi
  signed __int64 v8; // rax
  __m128i *v9; // rdi
  const __m128i *v10; // rax
  const __m128i *v11; // rbx

  result = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)result )
  {
    v3 = 24 * result;
    v6 = *(__m128i **)a1;
    v7 = (const __m128i *)(*(_QWORD *)a1 + v3);
    v8 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
    if ( v8 >> 2 )
    {
      v9 = *(__m128i **)a1;
      while ( a2 != v9[1].m128i_i64[0] )
      {
        if ( a2 == v9[2].m128i_i64[1] )
        {
          v9 = (__m128i *)((char *)v9 + 24);
          break;
        }
        if ( a2 == v9[4].m128i_i64[0] )
        {
          v9 += 3;
          break;
        }
        if ( a2 == v9[5].m128i_i64[1] )
        {
          v9 = (__m128i *)((char *)v9 + 72);
          break;
        }
        v9 += 6;
        if ( &v6[6 * (v8 >> 2)] == v9 )
        {
          v8 = 0xAAAAAAAAAAAAAAABLL * (((char *)v7 - (char *)v9) >> 3);
          goto LABEL_21;
        }
      }
LABEL_9:
      if ( v7 != v9 )
      {
        v10 = (__m128i *)((char *)v9 + 24);
        if ( v7 == (const __m128i *)&v9[1].m128i_u64[1] )
        {
          v11 = v9;
        }
        else
        {
          do
          {
            if ( a2 != v10[1].m128i_i64[0] )
            {
              v9 = (__m128i *)((char *)v9 + 24);
              *(__m128i *)((char *)v9 - 24) = _mm_loadu_si128(v10);
              v9[-1].m128i_i64[1] = v10[1].m128i_i64[0];
            }
            v10 = (const __m128i *)((char *)v10 + 24);
          }
          while ( v7 != v10 );
          v6 = *(__m128i **)a1;
          v11 = (__m128i *)((char *)v9 + *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8) - (_QWORD)v7);
          if ( v7 != (const __m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8)) )
          {
            memmove(v9, v7, *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8) - (_QWORD)v7);
            v6 = *(__m128i **)a1;
          }
        }
        goto LABEL_16;
      }
      goto LABEL_24;
    }
    v9 = *(__m128i **)a1;
LABEL_21:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        if ( v8 != 1 )
        {
LABEL_24:
          v11 = v7;
LABEL_16:
          *(_DWORD *)(a1 + 8) = -1431655765 * (((char *)v11 - (char *)v6) >> 3);
          return sub_2E0A2D0(a1, a2);
        }
        goto LABEL_29;
      }
      if ( a2 == v9[1].m128i_i64[0] )
        goto LABEL_9;
      v9 = (__m128i *)((char *)v9 + 24);
    }
    if ( a2 == v9[1].m128i_i64[0] )
      goto LABEL_9;
    v9 = (__m128i *)((char *)v9 + 24);
LABEL_29:
    v11 = v7;
    if ( a2 != v9[1].m128i_i64[0] )
      goto LABEL_16;
    goto LABEL_9;
  }
  return result;
}
