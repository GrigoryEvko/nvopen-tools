// Function: sub_2E98450
// Address: 0x2e98450
//
__int64 __fastcall sub_2E98450(__int64 a1, char a2, __int64 a3, __int64 a4)
{
  __m128i v4; // xmm4
  __m128i v5; // xmm3
  __m128i v6; // xmm2
  __m128i v7; // xmm1
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  _DWORD *v10; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v4 = _mm_loadu_si128(xmmword_3F8F0C0);
  v5 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
  *(_QWORD *)(a1 + 16) = 0;
  v6 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
  v7 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
  *(_QWORD *)(a1 + 24) = 0;
  v8 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
  *(__m128i *)(a1 + 40) = v4;
  *(_QWORD *)(a1 + 248) = a1 + 264;
  *(_QWORD *)(a1 + 256) = 0x1000000000LL;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_BYTE *)(a1 + 336) = a2;
  *(_BYTE *)(a1 + 337) = 0;
  *(__m128i *)(a1 + 56) = v5;
  *(__m128i *)(a1 + 72) = v6;
  *(__m128i *)(a1 + 88) = v7;
  *(__m128i *)(a1 + 104) = v8;
  *(__m128i *)(a1 + 120) = v4;
  *(__m128i *)(a1 + 136) = v5;
  *(__m128i *)(a1 + 152) = v6;
  *(__m128i *)(a1 + 168) = v7;
  *(__m128i *)(a1 + 184) = v8;
  *(_QWORD *)(a1 + 344) = a3;
  *(_QWORD *)(a1 + 352) = a4;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 1;
  *(_WORD *)(a1 + 392) = 0;
  v9 = (_QWORD *)(a1 + 416);
  do
  {
    if ( v9 )
      *v9 = -4096;
    v9 += 2;
  }
  while ( v9 != (_QWORD *)(a1 + 480) );
  *(_QWORD *)(a1 + 480) = 0;
  v10 = (_DWORD *)(a1 + 528);
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_DWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 1;
  do
  {
    if ( v10 )
      *v10 = -1;
    ++v10;
  }
  while ( (_DWORD *)(a1 + 544) != v10 );
  *(_QWORD *)(a1 + 1424) = 0;
  *(_QWORD *)(a1 + 544) = a1 + 560;
  *(_QWORD *)(a1 + 552) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = 0x800000000LL;
  *(_QWORD *)(a1 + 640) = a1 + 656;
  *(_QWORD *)(a1 + 592) = a1 + 608;
  *(_QWORD *)(a1 + 648) = 0x1000000000LL;
  *(_QWORD *)(a1 + 1432) = 0;
  *(_QWORD *)(a1 + 1440) = 0;
  *(_DWORD *)(a1 + 1448) = 0;
  *(_DWORD *)(a1 + 1456) = 2;
  return 0x1000000000LL;
}
