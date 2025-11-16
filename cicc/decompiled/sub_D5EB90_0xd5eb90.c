// Function: sub_D5EB90
// Address: 0xd5eb90
//
__int64 __fastcall sub_D5EB90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *(__fastcall *v8)(__int64 *, __int64 *, __int64 *, __int64, __int64, __int64); // rdx
  __m128i v9; // xmm1
  __m128i v10; // xmm0
  __m128i v12; // [rsp+10h] [rbp-70h] BYREF
  void (__fastcall *v13)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-60h]
  __int64 *(__fastcall *v14)(__int64 *, __int64 *, __int64 *, __int64, __int64, __int64); // [rsp+28h] [rbp-58h]
  void *v15; // [rsp+30h] [rbp-50h] BYREF
  __m128i v16; // [rsp+38h] [rbp-48h] BYREF
  __int64 (__fastcall *v17)(_QWORD *, _QWORD *, int); // [rsp+48h] [rbp-38h]
  __int64 *(__fastcall *v18)(__int64 *, __int64 *, __int64 *, __int64, __int64, __int64); // [rsp+50h] [rbp-30h]

  *(_QWORD *)(a1 + 8) = a3;
  v8 = v18;
  v9 = _mm_loadu_si128(&v16);
  *(_QWORD *)a1 = a2;
  v14 = v8;
  v18 = sub_D5BC20;
  *(_QWORD *)(a1 + 16) = a4;
  v12.m128i_i64[0] = a1;
  v10 = _mm_loadu_si128(&v12);
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 104) = a1 + 152;
  v17 = sub_D5BA20;
  *(_QWORD *)(a1 + 96) = a4;
  *(_QWORD *)(a1 + 112) = a1 + 168;
  v12 = v9;
  v16 = v10;
  v15 = &unk_49DA0D8;
  *(_QWORD *)(a1 + 32) = 0x200000000LL;
  *(_WORD *)(a1 + 132) = 512;
  v13 = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_DWORD *)(a1 + 128) = 0;
  *(_BYTE *)(a1 + 134) = 7;
  *(_WORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 152) = &unk_49D94D0;
  *(_QWORD *)(a1 + 160) = a2;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 168) = &unk_49DA0D8;
  *(_QWORD *)(a1 + 192) = 0;
  sub_D5BA20((_QWORD *)(a1 + 176), &v16, 2);
  *(_QWORD *)(a1 + 200) = v18;
  *(_QWORD *)(a1 + 192) = v17;
  nullsub_63();
  sub_B32BF0(&v15);
  if ( v13 )
    v13(&v12, &v12, 3);
  *(_QWORD *)(a1 + 352) = a5;
  *(_QWORD *)(a1 + 264) = a1 + 288;
  *(_QWORD *)(a1 + 360) = a6;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 272) = 8;
  *(_DWORD *)(a1 + 280) = 0;
  *(_BYTE *)(a1 + 284) = 1;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = a1 + 400;
  *(_QWORD *)(a1 + 384) = 8;
  *(_DWORD *)(a1 + 392) = 0;
  *(_BYTE *)(a1 + 396) = 1;
  return a1 + 400;
}
