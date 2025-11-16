// Function: sub_ED0500
// Address: 0xed0500
//
__int64 __fastcall sub_ED0500(__m128i **a1, const __m128i **a2, int a3)
{
  __m128i *v4; // rdi
  const __m128i *v5; // r12
  __m128i *v6; // rax

  if ( a3 == 1 )
  {
    *a1 = (__m128i *)*a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v4 = *a1;
      if ( v4 )
        j_j___libc_free_0(v4, 24);
    }
    return 0;
  }
  v5 = *a2;
  v6 = (__m128i *)sub_22077B0(24);
  if ( v6 )
  {
    *v6 = _mm_loadu_si128(v5);
    v6[1].m128i_i64[0] = v5[1].m128i_i64[0];
  }
  *a1 = v6;
  return 0;
}
