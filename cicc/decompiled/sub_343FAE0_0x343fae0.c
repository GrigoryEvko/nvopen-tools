// Function: sub_343FAE0
// Address: 0x343fae0
//
__int64 __fastcall sub_343FAE0(unsigned __int64 *a1, const __m128i **a2, int a3)
{
  unsigned __int64 v4; // rdi
  const __m128i *v5; // r12
  __m128i *v6; // rax

  if ( a3 == 1 )
  {
    *a1 = (unsigned __int64)*a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v4 = *a1;
      if ( v4 )
        j_j___libc_free_0(v4);
    }
    return 0;
  }
  v5 = *a2;
  v6 = (__m128i *)sub_22077B0(0x68u);
  if ( v6 )
  {
    *v6 = _mm_loadu_si128(v5);
    v6[1] = _mm_loadu_si128(v5 + 1);
    v6[2] = _mm_loadu_si128(v5 + 2);
    v6[3] = _mm_loadu_si128(v5 + 3);
    v6[4] = _mm_loadu_si128(v5 + 4);
    v6[5] = _mm_loadu_si128(v5 + 5);
    v6[6].m128i_i64[0] = v5[6].m128i_i64[0];
  }
  *a1 = (unsigned __int64)v6;
  return 0;
}
