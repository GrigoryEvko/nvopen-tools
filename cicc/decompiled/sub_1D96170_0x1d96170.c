// Function: sub_1D96170
// Address: 0x1d96170
//
__int64 sub_1D96170()
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
  __m128i v9; // xmm0
  void (__fastcall *v10)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v11; // rdx
  __m128i v12; // xmm4
  __int64 v13; // rax
  __int64 v14; // rax
  __m128i v16; // [rsp+10h] [rbp-30h] BYREF
  void (__fastcall *v17)(_QWORD, _QWORD, _QWORD); // [rsp+20h] [rbp-20h]
  __int64 v18; // [rsp+28h] [rbp-18h]

  v17 = 0;
  v0 = sub_22077B0(688);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FC362C;
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
    *(_QWORD *)v1 = off_49FA898;
    *(_QWORD *)(v1 + 216) = 1;
    v5 = _mm_loadu_si128(xmmword_452E800);
    v6 = _mm_loadu_si128(&xmmword_452E800[1]);
    *(_QWORD *)(v1 + 232) = 0;
    v7 = _mm_loadu_si128(&xmmword_452E800[2]);
    v8 = _mm_loadu_si128(&xmmword_452E800[3]);
    *(_QWORD *)(v1 + 240) = 0;
    *(__m128i *)(v1 + 256) = v5;
    *(__m128i *)(v1 + 272) = v6;
    *(_QWORD *)(v1 + 320) = unk_452E840;
    *(_QWORD *)(v1 + 392) = unk_452E840;
    *(_QWORD *)(v1 + 448) = v1 + 464;
    *(_QWORD *)(v1 + 456) = 0x1000000000LL;
    *(__m128i *)(v1 + 288) = v7;
    *(__m128i *)(v1 + 304) = v8;
    *(__m128i *)(v1 + 328) = v5;
    *(__m128i *)(v1 + 344) = v6;
    *(__m128i *)(v1 + 360) = v7;
    *(__m128i *)(v1 + 376) = v8;
    *(_QWORD *)(v1 + 248) = 0;
    *(_QWORD *)(v1 + 400) = 0;
    *(_QWORD *)(v1 + 408) = 0;
    *(_QWORD *)(v1 + 416) = 0;
    *(_QWORD *)(v1 + 424) = 0;
    *(_QWORD *)(v1 + 432) = 0;
    *(_QWORD *)(v1 + 440) = 0;
    *(_QWORD *)(v1 + 576) = 0;
    *(_QWORD *)(v1 + 584) = v1 + 600;
    v9 = _mm_loadu_si128(&v16);
    *(_QWORD *)(v1 + 592) = 0x800000000LL;
    v10 = v17;
    v11 = *(_QWORD *)(v1 + 680);
    v12 = _mm_loadu_si128((const __m128i *)(v1 + 656));
    *(_QWORD *)(v1 + 632) = 0;
    *(_QWORD *)(v1 + 672) = v10;
    v13 = v18;
    *(_DWORD *)(v1 + 640) = 0;
    *(_DWORD *)(v1 + 652) = -1;
    v17 = 0;
    v18 = v11;
    *(_QWORD *)(v1 + 680) = v13;
    v16 = v12;
    *(__m128i *)(v1 + 656) = v9;
    v14 = sub_163A1D0();
    sub_1D96080(v14);
  }
  if ( v17 )
    v17(&v16, &v16, 3);
  return v1;
}
