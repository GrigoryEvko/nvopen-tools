// Function: sub_27C1C30
// Address: 0x27c1c30
//
const char *__fastcall sub_27C1C30(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4, char a5)
{
  __m128i v6; // xmm1
  __m128i v7; // xmm0
  char (__fastcall *v8)(__int64 *, __int64 *); // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __m128i v13; // xmm4
  __m128i v14; // xmm5
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  void (__fastcall *v17)(__int64, __m128i *, __int64); // rax
  __m128i v19; // [rsp+0h] [rbp-E0h] BYREF
  void (__fastcall *v20)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-D0h]
  char (__fastcall *v21)(__int64 *, __int64 *); // [rsp+18h] [rbp-C8h]
  void *v22; // [rsp+20h] [rbp-C0h] BYREF
  __m128i v23; // [rsp+28h] [rbp-B8h] BYREF
  void (__fastcall *v24)(__int64, __m128i *, __int64); // [rsp+38h] [rbp-A8h]
  char (__fastcall *v25)(__int64 *, __int64 *); // [rsp+40h] [rbp-A0h]
  void *v26; // [rsp+50h] [rbp-90h]
  void *v27; // [rsp+58h] [rbp-88h]
  unsigned __int64 v28; // [rsp+60h] [rbp-80h]
  __m128i v29; // [rsp+68h] [rbp-78h] BYREF
  __m128i v30; // [rsp+78h] [rbp-68h] BYREF
  __m128i v31; // [rsp+88h] [rbp-58h] BYREF
  __m128i v32; // [rsp+98h] [rbp-48h] BYREF
  __int64 v33; // [rsp+A8h] [rbp-38h]

  *(_QWORD *)(a1 + 136) = a1 + 160;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 144) = 16;
  *(_DWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 156) = 1;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_DWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = a1 + 336;
  v6 = _mm_loadu_si128(&v23);
  *(_QWORD *)(a1 + 424) = a1 + 448;
  *(_WORD *)(a1 + 512) = 1;
  v19.m128i_i64[0] = a1;
  v7 = _mm_loadu_si128(&v19);
  *(_QWORD *)(a1 + 328) = 0x200000000LL;
  *(_QWORD *)(a1 + 384) = 0;
  v24 = (void (__fastcall *)(__int64, __m128i *, __int64))sub_27BFDD0;
  v8 = v25;
  *(_QWORD *)(a1 + 392) = 0;
  v21 = v8;
  *(_QWORD *)(a1 + 400) = 0;
  *(_DWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 432) = 2;
  *(_DWORD *)(a1 + 440) = 0;
  *(_BYTE *)(a1 + 444) = 1;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_DWORD *)(a1 + 504) = 0;
  *(_BYTE *)(a1 + 514) = 0;
  v19 = v6;
  v23 = v7;
  v22 = &unk_49DA0D8;
  v20 = 0;
  v25 = sub_27BFD20;
  v9 = *a2;
  v28 = a3;
  v29 = (__m128i)a3;
  LOWORD(v33) = 257;
  v26 = &unk_49E5698;
  v27 = &unk_49D94D0;
  v30 = 0u;
  v31 = 0u;
  v32 = 0u;
  v10 = sub_B2BE50(v9);
  v11 = _mm_loadu_si128(&v29);
  v12 = _mm_loadu_si128(&v30);
  *(_QWORD *)(a1 + 592) = v10;
  v13 = _mm_loadu_si128(&v31);
  *(_QWORD *)(a1 + 600) = a1 + 648;
  v14 = _mm_loadu_si128(&v32);
  *(_QWORD *)(a1 + 608) = a1 + 744;
  v15 = v28;
  *(_QWORD *)(a1 + 520) = a1 + 536;
  *(_QWORD *)(a1 + 664) = v15;
  v16 = v33;
  *(_QWORD *)(a1 + 528) = 0x200000000LL;
  *(_QWORD *)(a1 + 736) = v16;
  v17 = v24;
  *(_QWORD *)(a1 + 616) = 0;
  *(_DWORD *)(a1 + 624) = 0;
  *(_WORD *)(a1 + 628) = 512;
  *(_BYTE *)(a1 + 630) = 7;
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_WORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 648) = &unk_49E5698;
  *(_QWORD *)(a1 + 656) = &unk_49D94D0;
  *(_QWORD *)(a1 + 744) = &unk_49DA0D8;
  *(_QWORD *)(a1 + 768) = 0;
  *(__m128i *)(a1 + 672) = v11;
  *(__m128i *)(a1 + 688) = v12;
  *(__m128i *)(a1 + 704) = v13;
  *(__m128i *)(a1 + 720) = v14;
  if ( v17 )
  {
    v17(a1 + 752, &v23, 2);
    *(_QWORD *)(a1 + 776) = v25;
    *(_QWORD *)(a1 + 768) = v24;
  }
  v26 = &unk_49E5698;
  v27 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  sub_B32BF0(&v22);
  if ( v20 )
    v20(&v19, &v19, 3);
  *(_QWORD *)(a1 + 784) = a1 + 800;
  *(_QWORD *)(a1 + 792) = 0x800000000LL;
  *(_QWORD *)(a1 + 864) = byte_3F871B3;
  return byte_3F871B3;
}
