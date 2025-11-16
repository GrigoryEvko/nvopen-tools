// Function: sub_86A990
// Address: 0x86a990
//
void __fastcall sub_86A990(__int64 a1)
{
  const __m128i *v1; // r12
  __int64 *v2; // rax
  __m128i *v3; // rax
  __int8 v4; // dl

  v1 = *(const __m128i **)(a1 + 96);
  if ( v1 )
  {
    if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    {
      sub_86A080(*(_QWORD **)(a1 + 96));
      v2 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 96LL);
      do
        v2 = (__int64 *)*v2;
      while ( v2 && (*((_BYTE *)v2 + 16) != 53 || a1 != *(_QWORD *)(v2[3] + 24)) );
      *(_QWORD *)(a1 + 96) = v2;
    }
    else
    {
      v3 = (__m128i *)sub_727110();
      v3[1] = _mm_loadu_si128(v1 + 1);
      v1[1].m128i_i64[1] = (__int64)v3;
      v1[1].m128i_i8[0] = 53;
      v3->m128i_i64[0] = *(_QWORD *)(a1 + 64);
      v3[2].m128i_i64[0] = *(_QWORD *)(a1 + 256);
      v4 = *(_BYTE *)(a1 + 137);
      *(_QWORD *)(a1 + 256) = 0;
      v3[3].m128i_i8[8] = v4;
      *(_BYTE *)(a1 + 137) = 0;
    }
  }
}
