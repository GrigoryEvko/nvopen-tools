// Function: sub_AE1D50
// Address: 0xae1d50
//
__int64 __fastcall sub_AE1D50(__int64 a1)
{
  __m128i si128; // xmm1
  __m128i v2; // xmm2
  __m128i v3; // xmm3
  __m128i v4; // xmm4
  __m128i v5; // xmm0
  __m128i v6; // xmm5

  si128 = _mm_load_si128((const __m128i *)&xmmword_3F28F80);
  *(_BYTE *)a1 = 0;
  *(_QWORD *)(a1 + 32) = a1 + 56;
  v2 = _mm_load_si128((const __m128i *)&xmmword_3F28F90);
  *(_QWORD *)(a1 + 64) = a1 + 80;
  v3 = _mm_load_si128((const __m128i *)&xmmword_3F28F60);
  *(_QWORD *)(a1 + 72) = 0x600000005LL;
  v4 = _mm_load_si128((const __m128i *)&xmmword_3F28F70);
  *(_QWORD *)(a1 + 128) = a1 + 144;
  v5 = _mm_load_si128((const __m128i *)&xmmword_3F28FB0);
  *(_QWORD *)(a1 + 136) = 0x400000004LL;
  v6 = _mm_load_si128((const __m128i *)&xmmword_3F28F30);
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 112) = 0x30200000040LL;
  *(_QWORD *)(a1 + 184) = 0xA00000002LL;
  *(_QWORD *)(a1 + 272) = a1 + 288;
  *(_DWORD *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 17) = 0;
  *(_QWORD *)(a1 + 19) = 0;
  *(_BYTE *)(a1 + 27) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 8;
  *(_DWORD *)(a1 + 304) = 0;
  *(__m128i *)(a1 + 80) = si128;
  *(__m128i *)(a1 + 96) = v2;
  *(__m128i *)(a1 + 144) = v3;
  *(__m128i *)(a1 + 160) = v4;
  *(__m128i *)(a1 + 192) = v5;
  *(__m128i *)(a1 + 288) = v6;
  *(_QWORD *)(a1 + 280) = 0x800000001LL;
  *(_QWORD *)(a1 + 448) = a1 + 464;
  *(_QWORD *)(a1 + 456) = 0;
  *(_BYTE *)(a1 + 464) = 0;
  *(_WORD *)(a1 + 480) = 768;
  *(_QWORD *)(a1 + 488) = 0;
  return 768;
}
