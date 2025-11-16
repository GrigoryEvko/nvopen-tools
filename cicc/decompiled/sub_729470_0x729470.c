// Function: sub_729470
// Address: 0x729470
//
void __fastcall sub_729470(__int64 a1, const __m128i *a2)
{
  __m128i *v2; // rax

  if ( a2 )
  {
    v2 = *(__m128i **)(a1 + 72);
    if ( v2 )
    {
      *v2 = _mm_loadu_si128(a2 + 1);
      v2[1] = _mm_loadu_si128(a2 + 2);
      v2[2] = _mm_loadu_si128(a2 + 3);
      v2[3].m128i_i64[0] = a2[5].m128i_i64[0];
    }
  }
}
