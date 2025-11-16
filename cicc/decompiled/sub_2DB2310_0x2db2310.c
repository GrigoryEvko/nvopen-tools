// Function: sub_2DB2310
// Address: 0x2db2310
//
__int64 sub_2DB2310()
{
  __int64 result; // rax
  __m128i v1; // xmm4
  __m128i v2; // xmm3
  __m128i v3; // xmm2
  __m128i v4; // xmm1
  __m128i v5; // xmm0

  result = sub_22077B0(0x508u);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 56) = result + 104;
    *(_QWORD *)(result + 112) = result + 160;
    *(_QWORD *)(result + 16) = &unk_501CF64;
    v1 = _mm_loadu_si128(xmmword_3F8F0C0);
    v2 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
    *(_QWORD *)result = off_4A27510;
    *(_DWORD *)(result + 88) = 1065353216;
    *(_DWORD *)(result + 144) = 1065353216;
    *(_DWORD *)(result + 24) = 2;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)(result + 40) = 0;
    *(_QWORD *)(result + 48) = 0;
    *(_QWORD *)(result + 64) = 1;
    *(_QWORD *)(result + 72) = 0;
    *(_QWORD *)(result + 80) = 0;
    *(_QWORD *)(result + 96) = 0;
    *(_QWORD *)(result + 104) = 0;
    *(_QWORD *)(result + 120) = 1;
    *(_QWORD *)(result + 128) = 0;
    *(_QWORD *)(result + 136) = 0;
    *(_QWORD *)(result + 152) = 0;
    *(_QWORD *)(result + 160) = 0;
    *(_BYTE *)(result + 168) = 0;
    *(_QWORD *)(result + 176) = 0;
    *(_QWORD *)(result + 184) = 0;
    *(_QWORD *)(result + 192) = 0;
    *(_QWORD *)(result + 200) = 0;
    *(_QWORD *)(result + 208) = 0;
    *(__m128i *)(result + 216) = v1;
    *(__m128i *)(result + 232) = v2;
    v3 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
    v4 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
    v5 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
    *(_QWORD *)(result + 872) = result + 888;
    *(_QWORD *)(result + 424) = result + 440;
    *(_QWORD *)(result + 600) = result + 616;
    *(_QWORD *)(result + 1056) = result + 1080;
    *(_QWORD *)(result + 432) = 0x1000000000LL;
    *(_QWORD *)(result + 376) = 0;
    *(_QWORD *)(result + 384) = 0;
    *(_QWORD *)(result + 392) = 0;
    *(_QWORD *)(result + 400) = 0;
    *(_QWORD *)(result + 408) = 0;
    *(_QWORD *)(result + 416) = 0;
    *(_QWORD *)(result + 504) = 0;
    *(_QWORD *)(result + 512) = 0;
    *(_QWORD *)(result + 520) = 0;
    *(_QWORD *)(result + 528) = 0;
    *(_QWORD *)(result + 536) = 0;
    *(_QWORD *)(result + 608) = 0x800000000LL;
    *(_QWORD *)(result + 880) = 0x400000000LL;
    *(_QWORD *)(result + 1048) = 0;
    *(_QWORD *)(result + 1064) = 8;
    *(_DWORD *)(result + 1072) = 0;
    *(__m128i *)(result + 248) = v3;
    *(__m128i *)(result + 264) = v4;
    *(__m128i *)(result + 280) = v5;
    *(__m128i *)(result + 296) = v1;
    *(__m128i *)(result + 312) = v2;
    *(__m128i *)(result + 328) = v3;
    *(__m128i *)(result + 344) = v4;
    *(__m128i *)(result + 360) = v5;
    *(_BYTE *)(result + 1076) = 1;
    *(_QWORD *)(result + 1144) = result + 1160;
    *(_QWORD *)(result + 1152) = 0x600000000LL;
    *(_DWORD *)(result + 1208) = 0;
    *(_QWORD *)(result + 1216) = result + 1232;
    *(_QWORD *)(result + 1224) = 0x800000000LL;
    *(_QWORD *)(result + 1264) = 0;
    *(_DWORD *)(result + 1272) = 0;
    *(_QWORD *)(result + 1280) = 0;
  }
  return result;
}
