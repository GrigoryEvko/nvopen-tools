// Function: sub_1E1D8B0
// Address: 0x1e1d8b0
//
__int64 sub_1E1D8B0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __m128i v2; // xmm3
  __m128i v3; // xmm2
  __m128i v4; // xmm1
  __m128i v5; // xmm0
  __int64 v6; // rax

  v0 = sub_22077B0(1856);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FC64C9;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_DWORD *)(v0 + 24) = 3;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)v0 = &unk_49FB790;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 152) = 0;
    sub_1BFC1A0(v0 + 160, 8, 0);
    sub_1BFC1A0(v1 + 184, 8, 0);
    sub_1BFC1A0(v1 + 208, 8, 0);
    *(_QWORD *)(v1 + 472) = 0x1000000000LL;
    v2 = _mm_loadu_si128(xmmword_452E800);
    v3 = _mm_loadu_si128(&xmmword_452E800[1]);
    *(_QWORD *)(v1 + 416) = 0;
    v4 = _mm_loadu_si128(&xmmword_452E800[2]);
    v5 = _mm_loadu_si128(&xmmword_452E800[3]);
    *(_QWORD *)(v1 + 424) = 0;
    *(__m128i *)(v1 + 272) = v2;
    *(__m128i *)(v1 + 288) = v3;
    *(_QWORD *)(v1 + 336) = unk_452E840;
    *(_QWORD *)(v1 + 408) = unk_452E840;
    *(_QWORD *)(v1 + 464) = v1 + 480;
    *(_QWORD *)(v1 + 624) = v1 + 640;
    *(_QWORD *)(v1 + 632) = 0x800000000LL;
    *(__m128i *)(v1 + 304) = v4;
    *(__m128i *)(v1 + 320) = v5;
    *(__m128i *)(v1 + 344) = v2;
    *(__m128i *)(v1 + 360) = v3;
    *(__m128i *)(v1 + 376) = v4;
    *(__m128i *)(v1 + 392) = v5;
    *(_QWORD *)(v1 + 432) = 0;
    *(_QWORD *)(v1 + 440) = 0;
    *(_QWORD *)(v1 + 448) = 0;
    *(_QWORD *)(v1 + 456) = 0;
    *(_BYTE *)(v1 + 552) = 0;
    *(_BYTE *)(v1 + 704) = 0;
    *(_QWORD *)(v1 + 712) = 0;
    *(_QWORD *)(v1 + 720) = 0;
    *(_QWORD *)(v1 + 728) = 0;
    *(_DWORD *)(v1 + 736) = 0;
    *(_QWORD *)(v1 + 744) = v1 + 760;
    *(_QWORD *)(v1 + 912) = v1 + 896;
    *(_QWORD *)(v1 + 920) = v1 + 896;
    *(_QWORD *)(v1 + 944) = 0x800000000LL;
    *(_QWORD *)(v1 + 992) = 0x800000000LL;
    *(_QWORD *)(v1 + 936) = v1 + 952;
    *(_QWORD *)(v1 + 1032) = v1 + 1048;
    *(_QWORD *)(v1 + 752) = 0x2000000000LL;
    *(_DWORD *)(v1 + 896) = 0;
    *(_QWORD *)(v1 + 904) = 0;
    *(_QWORD *)(v1 + 928) = 0;
    *(_QWORD *)(v1 + 984) = v1 + 1000;
    *(_QWORD *)(v1 + 1040) = 0x1000000000LL;
    *(_QWORD *)(v1 + 1816) = 0;
    *(_QWORD *)(v1 + 1824) = 0;
    *(_QWORD *)(v1 + 1832) = 0;
    *(_DWORD *)(v1 + 1840) = 0;
    *(_QWORD *)v1 = off_49FBAB0;
    v6 = sub_163A1D0();
    sub_1E1D7C0(v6);
  }
  return v1;
}
