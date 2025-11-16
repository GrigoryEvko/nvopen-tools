// Function: sub_86ACD0
// Address: 0x86acd0
//
void __fastcall sub_86ACD0(__int64 a1)
{
  const __m128i *v1; // r12
  __int64 v2; // rbx
  __m128i *v3; // rax
  __int64 v4; // rdx
  __int64 *v5; // rax

  v1 = *(const __m128i **)(a1 + 96);
  if ( v1 )
  {
    v2 = a1;
    if ( (*(_BYTE *)(a1 + 177) & 0x12) == 2 )
    {
      sub_869630(*(_QWORD *)(a1 + 96), 0);
      v5 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 96LL);
      do
        v5 = (__int64 *)*v5;
      while ( v5 && (*((_BYTE *)v5 + 16) != 53 || a1 != *(_QWORD *)(v5[3] + 24)) );
      *(_QWORD *)(a1 + 96) = v5;
    }
    else
    {
      sub_869630(*(_QWORD *)(a1 + 96), 1);
      v3 = (__m128i *)sub_727110();
      v3[1] = _mm_loadu_si128(v1 + 1);
      v1[1].m128i_i64[1] = (__int64)v3;
      v1[1].m128i_i8[0] = 53;
      v4 = *(_QWORD *)(a1 + 64);
      v3[3].m128i_i8[9] |= 1u;
      v3->m128i_i64[0] = v4;
      v3[2].m128i_i64[0] = a1;
      if ( *(_BYTE *)(a1 + 140) == 12 )
      {
        do
          v2 = *(_QWORD *)(v2 + 160);
        while ( *(_BYTE *)(v2 + 140) == 12 );
      }
      v3[3].m128i_i8[9] = (32 * (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v2 + 96LL) + 180LL) >> 7)) | v3[3].m128i_i8[9] & 0xDF;
    }
  }
}
