// Function: sub_352AA80
// Address: 0x352aa80
//
__m128i *sub_352AA80()
{
  __int64 v0; // rax
  __m128i *v1; // r12
  __m128i v2; // xmm4
  __m128i v3; // xmm3
  __m128i v4; // xmm2
  __m128i v5; // xmm1
  __m128i v6; // xmm0
  __int128 *v7; // rax

  v0 = sub_22077B0(0x3C8u);
  v1 = (__m128i *)v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_503D24C;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A38CF0;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_DWORD *)(v0 + 144) = 1065353216;
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
    *(_QWORD *)(v0 + 216) = 0;
    *(_QWORD *)(v0 + 304) = 0;
    *(_QWORD *)(v0 + 312) = 0;
    *(_QWORD *)(v0 + 320) = 0;
    *(_QWORD *)(v0 + 328) = 0;
    *(_QWORD *)(v0 + 336) = 0;
    *(_QWORD *)(v0 + 344) = 0;
    sub_2F5FEE0(v0 + 352);
    v1[52].m128i_i64[0] = 0;
    v1[52].m128i_i64[1] = 0;
    v2 = _mm_loadu_si128(xmmword_3F8F0C0);
    v3 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
    v1[53].m128i_i64[0] = 0;
    v4 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
    v5 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
    v1[53].m128i_i64[1] = 0;
    v6 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
    v1[54].m128i_i64[0] = 0;
    v1[55].m128i_i64[0] = (__int64)v1[56].m128i_i64;
    v1[54].m128i_i64[1] = 0;
    v1[55].m128i_i64[1] = 0x1000000000LL;
    v1[60].m128i_i64[0] = 0;
    v1[42] = v2;
    v1[43] = v3;
    v1[44] = v4;
    v1[45] = v5;
    v1[46] = v6;
    v1[47] = v2;
    v1[48] = v3;
    v1[49] = v4;
    v1[50] = v5;
    v1[51] = v6;
    v7 = sub_BC2B00();
    sub_352AA00((__int64)v7);
  }
  return v1;
}
