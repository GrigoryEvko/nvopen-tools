// Function: sub_34EAFC0
// Address: 0x34eafc0
//
__int64 sub_34EAFC0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __m128i v2; // xmm4
  __m128i v3; // xmm3
  __m128i v4; // xmm2
  __m128i v5; // xmm1
  __m128i v6; // xmm0
  __m128i v7; // xmm0
  void (__fastcall *v8)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v9; // rdx
  __m128i v10; // xmm5
  __int64 v11; // rax
  __int128 *v12; // rax
  __m128i v14; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v15)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-20h]
  __int64 v16; // [rsp+18h] [rbp-18h]

  v15 = 0;
  v0 = sub_22077B0(0x298u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_503B14C;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A384A8;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_DWORD *)(v0 + 144) = 1065353216;
    v2 = _mm_loadu_si128(xmmword_3F8F0C0);
    *(_DWORD *)(v0 + 24) = 2;
    *(__m128i *)(v0 + 224) = v2;
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
    v3 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
    v4 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
    v5 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
    *(__m128i *)(v0 + 304) = v2;
    *(__m128i *)(v0 + 240) = v3;
    v6 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
    *(_QWORD *)(v0 + 432) = v0 + 448;
    *(_QWORD *)(v0 + 440) = 0x1000000000LL;
    *(_QWORD *)(v0 + 568) = v0 + 592;
    *(__m128i *)(v0 + 256) = v4;
    *(__m128i *)(v0 + 272) = v5;
    *(__m128i *)(v0 + 288) = v6;
    *(__m128i *)(v0 + 320) = v3;
    *(__m128i *)(v0 + 336) = v4;
    *(__m128i *)(v0 + 352) = v5;
    *(__m128i *)(v0 + 368) = v6;
    *(_QWORD *)(v0 + 384) = 0;
    *(_QWORD *)(v0 + 392) = 0;
    *(_QWORD *)(v0 + 400) = 0;
    *(_QWORD *)(v0 + 408) = 0;
    *(_QWORD *)(v0 + 416) = 0;
    *(_QWORD *)(v0 + 424) = 0;
    *(_QWORD *)(v0 + 512) = 0;
    *(_QWORD *)(v0 + 520) = 0;
    *(_QWORD *)(v0 + 528) = 0;
    *(_QWORD *)(v0 + 536) = 0;
    *(_QWORD *)(v0 + 544) = 0;
    *(_QWORD *)(v0 + 552) = 0;
    *(_QWORD *)(v0 + 560) = 0;
    *(_QWORD *)(v0 + 576) = 0;
    *(_QWORD *)(v0 + 584) = 8;
    *(_QWORD *)(v0 + 608) = 0;
    *(_DWORD *)(v0 + 616) = 0;
    v7 = _mm_loadu_si128(&v14);
    *(_WORD *)(v0 + 624) = 1;
    v8 = v15;
    v9 = *(_QWORD *)(v1 + 656);
    v15 = 0;
    v10 = _mm_loadu_si128((const __m128i *)(v1 + 632));
    *(_QWORD *)(v1 + 648) = v8;
    *(_DWORD *)(v1 + 628) = -1;
    v11 = v16;
    v14 = v10;
    v16 = v9;
    *(_QWORD *)(v1 + 656) = v11;
    *(__m128i *)(v1 + 632) = v7;
    v12 = sub_BC2B00();
    sub_34EAF40((__int64)v12);
  }
  if ( v15 )
    v15(&v14, &v14, 3);
  return v1;
}
