// Function: sub_1DF3550
// Address: 0x1df3550
//
__int64 sub_1DF3550()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  __m128i v5; // xmm3
  __m128i v6; // xmm2
  __m128i v7; // xmm1
  __m128i v8; // xmm0
  __int64 v9; // rax

  v0 = sub_22077B0(648);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FC59F0;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_DWORD *)(v0 + 24) = 3;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 152) = 0;
    *(_QWORD *)v0 = &unk_49FB790;
    *(_QWORD *)(v0 + 160) = 0;
    *(_QWORD *)(v0 + 168) = 0;
    *(_DWORD *)(v0 + 176) = 8;
    v2 = (_QWORD *)malloc(8u);
    if ( !v2 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v2 = 0;
    }
    *v2 = 0;
    *(_QWORD *)(v1 + 160) = v2;
    *(_QWORD *)(v1 + 168) = 1;
    *(_QWORD *)(v1 + 184) = 0;
    *(_QWORD *)(v1 + 192) = 0;
    *(_DWORD *)(v1 + 200) = 8;
    v3 = (_QWORD *)malloc(8u);
    if ( !v3 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v3 = 0;
    }
    *v3 = 0;
    *(_QWORD *)(v1 + 184) = v3;
    *(_QWORD *)(v1 + 192) = 1;
    *(_QWORD *)(v1 + 208) = 0;
    *(_QWORD *)(v1 + 216) = 0;
    *(_DWORD *)(v1 + 224) = 8;
    v4 = (_QWORD *)malloc(8u);
    if ( !v4 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v4 = 0;
    }
    *(_QWORD *)(v1 + 208) = v4;
    *v4 = 0;
    *(_QWORD *)v1 = off_49FB440;
    *(_QWORD *)(v1 + 216) = 1;
    v5 = _mm_loadu_si128(xmmword_452E800);
    v6 = _mm_loadu_si128(&xmmword_452E800[1]);
    *(_QWORD *)(v1 + 504) = 0;
    v7 = _mm_loadu_si128(&xmmword_452E800[2]);
    v8 = _mm_loadu_si128(&xmmword_452E800[3]);
    *(_QWORD *)(v1 + 512) = 0;
    *(_QWORD *)(v1 + 520) = 0;
    *(_QWORD *)(v1 + 528) = 0;
    *(_QWORD *)(v1 + 424) = unk_452E840;
    *(_QWORD *)(v1 + 496) = unk_452E840;
    *(_QWORD *)(v1 + 552) = v1 + 568;
    *(_QWORD *)(v1 + 536) = 0;
    *(_QWORD *)(v1 + 544) = 0;
    *(_QWORD *)(v1 + 560) = 0x1000000000LL;
    *(__m128i *)(v1 + 360) = v5;
    *(__m128i *)(v1 + 376) = v6;
    *(__m128i *)(v1 + 392) = v7;
    *(__m128i *)(v1 + 408) = v8;
    *(__m128i *)(v1 + 432) = v5;
    *(__m128i *)(v1 + 448) = v6;
    *(__m128i *)(v1 + 464) = v7;
    *(__m128i *)(v1 + 480) = v8;
    v9 = sub_163A1D0();
    sub_1DF3460(v9);
  }
  return v1;
}
