// Function: sub_1E7FB20
// Address: 0x1e7fb20
//
__int64 __fastcall sub_1E7FB20(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  __m128i v4; // xmm3
  __m128i v5; // xmm2
  __m128i v6; // xmm1
  __m128i v7; // xmm0

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 16) = &unk_4FC820C;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)a1 = &unk_49FB790;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 176) = 8;
  v1 = (_QWORD *)malloc(8u);
  if ( !v1 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v1 = 0;
  }
  *(_QWORD *)(a1 + 160) = v1;
  *(_QWORD *)(a1 + 168) = 1;
  *v1 = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 8;
  v2 = (_QWORD *)malloc(8u);
  if ( !v2 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v2 = 0;
  }
  *(_QWORD *)(a1 + 184) = v2;
  *(_QWORD *)(a1 + 192) = 1;
  *v2 = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 8;
  v3 = (_QWORD *)malloc(8u);
  if ( !v3 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v3 = 0;
  }
  *(_QWORD *)(a1 + 208) = v3;
  *v3 = 0;
  *(_QWORD *)(a1 + 216) = 1;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)a1 = &unk_49FCD40;
  *(_QWORD *)(a1 + 240) = 0;
  v4 = _mm_loadu_si128(xmmword_452E800);
  v5 = _mm_loadu_si128(&xmmword_452E800[1]);
  *(_QWORD *)(a1 + 248) = 0;
  v6 = _mm_loadu_si128(&xmmword_452E800[2]);
  v7 = _mm_loadu_si128(&xmmword_452E800[3]);
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 336) = unk_452E840;
  *(_QWORD *)(a1 + 408) = unk_452E840;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(__m128i *)(a1 + 272) = v4;
  *(__m128i *)(a1 + 288) = v5;
  *(__m128i *)(a1 + 304) = v6;
  *(__m128i *)(a1 + 320) = v7;
  *(__m128i *)(a1 + 344) = v4;
  *(__m128i *)(a1 + 360) = v5;
  *(__m128i *)(a1 + 376) = v6;
  *(__m128i *)(a1 + 392) = v7;
  *(_QWORD *)(a1 + 464) = a1 + 480;
  *(_QWORD *)(a1 + 472) = 0x1000000000LL;
  *(_QWORD *)(a1 + 552) = a1 + 568;
  *(_QWORD *)(a1 + 560) = 0x400000000LL;
  *(_QWORD *)(a1 + 600) = a1 + 616;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  return a1 + 616;
}
