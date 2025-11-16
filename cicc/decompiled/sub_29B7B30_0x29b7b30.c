// Function: sub_29B7B30
// Address: 0x29b7b30
//
void __fastcall sub_29B7B30(const __m128i **a1, unsigned __int64 a2)
{
  const __m128i *v2; // r8
  const __m128i *v4; // rax
  unsigned __int64 v5; // rdi
  __m128i *v6; // r13
  unsigned __int64 v7; // r12
  __int64 v8; // rax
  __m128i *v9; // rdx

  if ( a2 > 0x333333333333333LL )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *a1;
  v4 = *a1;
  if ( a2 > 0xCCCCCCCCCCCCCCCDLL * (((char *)a1[2] - (char *)*a1) >> 3) )
  {
    v5 = (unsigned __int64)a1[1];
    v6 = 0;
    v7 = v5 - (_QWORD)v2;
    if ( a2 )
    {
      v8 = sub_22077B0(40 * a2);
      v2 = *a1;
      v5 = (unsigned __int64)a1[1];
      v6 = (__m128i *)v8;
      v4 = *a1;
    }
    if ( v2 != (const __m128i *)v5 )
    {
      v9 = v6;
      do
      {
        if ( v9 )
        {
          *v9 = _mm_loadu_si128(v4);
          v9[1] = _mm_loadu_si128(v4 + 1);
          v9[2].m128i_i64[0] = v4[2].m128i_i64[0];
        }
        v4 = (const __m128i *)((char *)v4 + 40);
        v9 = (__m128i *)((char *)v9 + 40);
      }
      while ( (const __m128i *)v5 != v4 );
      v5 = (unsigned __int64)v2;
    }
    if ( v5 )
      j_j___libc_free_0(v5);
    *a1 = v6;
    a1[1] = (__m128i *)((char *)v6 + v7);
    a1[2] = (__m128i *)((char *)v6 + 40 * a2);
  }
}
