// Function: sub_2F9C060
// Address: 0x2f9c060
//
__int64 sub_2F9C060()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __m128i v2; // xmm4
  __m128i v3; // xmm3
  __m128i v4; // xmm2
  __m128i v5; // xmm1
  __m128i v6; // xmm0
  __int128 *v7; // rax

  v0 = sub_22077B0(0x218u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_50255EC;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A2BE00;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_DWORD *)(v0 + 144) = 1065353216;
    v2 = _mm_loadu_si128(xmmword_3F8F0C0);
    *(_DWORD *)(v0 + 24) = 2;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_QWORD *)(v0 + 64) = 1;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 80) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_QWORD *)(v0 + 104) = 0;
    *(_QWORD *)(v0 + 120) = 1;
    *(_QWORD *)(v0 + 128) = 0;
    *(_QWORD *)(v0 + 136) = 0;
    *(_QWORD *)(v0 + 152) = 0;
    *(_QWORD *)(v0 + 160) = 0;
    *(_BYTE *)(v0 + 168) = 0;
    *(_QWORD *)(v0 + 176) = 0;
    *(_QWORD *)(v0 + 184) = 0;
    *(_QWORD *)(v0 + 192) = 0;
    *(_QWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_QWORD *)(v0 + 224) = 0;
    *(_QWORD *)(v0 + 232) = 0;
    *(__m128i *)(v0 + 240) = v2;
    v3 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
    v4 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
    v5 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
    v6 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
    *(_QWORD *)(v0 + 400) = 0;
    *(_QWORD *)(v0 + 448) = v0 + 464;
    *(_QWORD *)(v0 + 408) = 0;
    *(_QWORD *)(v0 + 416) = 0;
    *(_QWORD *)(v0 + 424) = 0;
    *(_QWORD *)(v0 + 432) = 0;
    *(_QWORD *)(v0 + 440) = 0;
    *(_QWORD *)(v0 + 456) = 0x1000000000LL;
    *(_QWORD *)(v0 + 528) = 0;
    *(__m128i *)(v0 + 256) = v3;
    *(__m128i *)(v0 + 272) = v4;
    *(__m128i *)(v0 + 288) = v5;
    *(__m128i *)(v0 + 304) = v6;
    *(__m128i *)(v0 + 320) = v2;
    *(__m128i *)(v0 + 336) = v3;
    *(__m128i *)(v0 + 352) = v4;
    *(__m128i *)(v0 + 368) = v5;
    *(__m128i *)(v0 + 384) = v6;
    v7 = sub_BC2B00();
    sub_2F9BFE0((__int64)v7);
  }
  return v1;
}
