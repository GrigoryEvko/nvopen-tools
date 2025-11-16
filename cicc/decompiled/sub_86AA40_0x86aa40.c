// Function: sub_86AA40
// Address: 0x86aa40
//
void __fastcall sub_86AA40(__int64 a1)
{
  const __m128i *v1; // r12
  __m128i *v2; // rax
  __int8 v3; // dl
  char v4; // dl

  v1 = *(const __m128i **)(a1 + 96);
  if ( v1 )
  {
    if ( v1[1].m128i_i8[0] != 53 )
    {
      v2 = (__m128i *)sub_727110();
      v2[1] = _mm_loadu_si128(v1 + 1);
      v1[1].m128i_i64[1] = (__int64)v2;
      v1[1].m128i_i8[0] = 53;
      v2->m128i_i64[0] = *(_QWORD *)(a1 + 64);
      v2[2].m128i_i64[0] = *(_QWORD *)(a1 + 264);
      v3 = *(_BYTE *)(a1 + 173);
      *(_QWORD *)(a1 + 264) = 0;
      v2[3].m128i_i8[8] = v3;
      v4 = *(_BYTE *)(a1 + 202) >> 7;
      *(_BYTE *)(a1 + 173) = 0;
      v2[3].m128i_i8[9] = (4 * v4) | v2[3].m128i_i8[9] & 0xFB;
      *(_BYTE *)(a1 + 202) &= ~0x80u;
    }
  }
}
