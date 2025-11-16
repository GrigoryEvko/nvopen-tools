// Function: sub_7258C0
// Address: 0x7258c0
//
void __fastcall sub_7258C0(__int64 a1, char a2)
{
  __m128i si128; // xmm1
  __m128i v3; // xmm2
  __m128i v4; // xmm3
  __m128i v5; // xmm4
  __m128i v6; // xmm5
  __m128i v7; // xmm6
  __int64 v8; // rax

  si128 = _mm_load_si128((const __m128i *)&xmmword_4F079B0);
  v3 = _mm_load_si128((const __m128i *)&xmmword_4F079C0);
  v4 = _mm_load_si128((const __m128i *)&xmmword_4F079D0);
  v5 = _mm_load_si128((const __m128i *)&xmmword_4F079E0);
  v6 = _mm_load_si128((const __m128i *)&xmmword_4F079F0);
  v7 = _mm_load_si128((const __m128i *)&xmmword_4F07A00);
  *(__m128i *)a1 = _mm_load_si128((const __m128i *)&xmmword_4F079A0);
  v8 = unk_4D03FA0;
  *(__m128i *)(a1 + 16) = si128;
  *(__m128i *)(a1 + 32) = v3;
  *(__m128i *)(a1 + 48) = v4;
  *(__m128i *)(a1 + 64) = v5;
  *(__m128i *)(a1 + 80) = v6;
  *(__m128i *)(a1 + 96) = v7;
  if ( v8 )
    *(_QWORD *)(a1 + 56) = *(_QWORD *)(v8 + 8);
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 1;
  *(_BYTE *)(a1 + 141) = 32 * (a2 == 1 || (unsigned __int8)(a2 - 8) <= 3u);
  sub_7258B0(a1);
  *(_WORD *)(a1 + 142) &= 0x403u;
  *(_BYTE *)(a1 + 144) &= 0xF0u;
  *(_QWORD *)(a1 + 152) = 0;
  sub_725570(a1, a2);
}
