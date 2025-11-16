// Function: sub_305D0A0
// Address: 0x305d0a0
//
__int64 __fastcall sub_305D0A0(unsigned __int64 *a1, const __m128i **a2, int a3)
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
  v6 = (__m128i *)sub_22077B0(0x34u);
  if ( v6 )
  {
    *v6 = _mm_loadu_si128(v5);
    v6[1] = _mm_loadu_si128(v5 + 1);
    v6[2] = _mm_loadu_si128(v5 + 2);
    v6[3].m128i_i32[0] = v5[3].m128i_i32[0];
  }
  *a1 = (unsigned __int64)v6;
  return 0;
}
