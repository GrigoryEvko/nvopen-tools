// Function: sub_2EE7BD0
// Address: 0x2ee7bd0
//
__int64 __fastcall sub_2EE7BD0(__int64 a1)
{
  __m128i v1; // xmm0
  __m128i v2; // xmm4
  __m128i v3; // xmm3
  __m128i v4; // xmm2
  __m128i v5; // xmm1

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 16) = &unk_502234C;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  *(_QWORD *)a1 = &unk_4A2A2B8;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  v1 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
  v2 = _mm_loadu_si128(xmmword_3F8F0C0);
  *(_QWORD *)(a1 + 408) = 0;
  v3 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
  v4 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
  *(_QWORD *)(a1 + 416) = 0;
  v5 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
  *(__m128i *)(a1 + 304) = v1;
  *(_QWORD *)(a1 + 448) = a1 + 464;
  *(_QWORD *)(a1 + 456) = 0x1000000000LL;
  *(_QWORD *)(a1 + 536) = a1 + 552;
  *(_QWORD *)(a1 + 544) = 0x400000000LL;
  *(__m128i *)(a1 + 384) = v1;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 584) = a1 + 600;
  *(_QWORD *)(a1 + 592) = 0;
  *(__m128i *)(a1 + 240) = v2;
  *(__m128i *)(a1 + 256) = v3;
  *(__m128i *)(a1 + 272) = v4;
  *(__m128i *)(a1 + 288) = v5;
  *(__m128i *)(a1 + 320) = v2;
  *(__m128i *)(a1 + 336) = v3;
  *(__m128i *)(a1 + 352) = v4;
  *(__m128i *)(a1 + 368) = v5;
  *(_OWORD *)(a1 + 600) = 0;
  return a1 + 600;
}
