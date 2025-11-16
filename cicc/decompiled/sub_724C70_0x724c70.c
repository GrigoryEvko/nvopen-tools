// Function: sub_724C70
// Address: 0x724c70
//
void __fastcall sub_724C70(__int64 a1, char a2)
{
  __m128i si128; // xmm1
  __m128i v3; // xmm2
  __m128i v4; // xmm3
  __m128i v5; // xmm4
  __m128i v6; // xmm5
  __m128i v7; // xmm6
  __int64 v8; // rax
  __int64 v9; // rax

  si128 = _mm_load_si128((const __m128i *)&xmmword_4F079B0);
  v3 = _mm_load_si128((const __m128i *)&xmmword_4F079C0);
  v4 = _mm_load_si128((const __m128i *)&xmmword_4F079D0);
  v5 = _mm_load_si128((const __m128i *)&xmmword_4F079E0);
  *(__m128i *)a1 = _mm_load_si128((const __m128i *)&xmmword_4F079A0);
  v6 = _mm_load_si128((const __m128i *)&xmmword_4F079F0);
  v7 = _mm_load_si128((const __m128i *)&xmmword_4F07A00);
  *(__m128i *)(a1 + 16) = si128;
  v8 = unk_4D03FA0;
  *(__m128i *)(a1 + 32) = v3;
  *(__m128i *)(a1 + 48) = v4;
  *(__m128i *)(a1 + 64) = v5;
  *(__m128i *)(a1 + 80) = v6;
  *(__m128i *)(a1 + 96) = v7;
  if ( v8 )
    *(_QWORD *)(a1 + 56) = *(_QWORD *)(v8 + 8);
  *(_BYTE *)(a1 + 172) &= 0xE0u;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  v9 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 112) = v9;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  sub_724A80(a1, a2);
}
