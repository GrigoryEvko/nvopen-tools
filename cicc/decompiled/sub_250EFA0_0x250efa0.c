// Function: sub_250EFA0
// Address: 0x250efa0
//
__int64 __fastcall sub_250EFA0(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  __int16 v9; // ax
  void (__fastcall *v10)(__int64, unsigned __int64 *, __int64); // rax
  void (__fastcall *v11)(__int64, unsigned __int64 *, __int64); // rax
  __m128i v12; // xmm0
  __int64 v13; // rax
  __int64 v14; // rax
  void (__fastcall *v15)(__int64, unsigned __int64 *, __int64); // rax
  __int64 result; // rax
  __int64 *v17; // r12
  __int64 *i; // r14
  __int64 v19; // r13
  __int64 v20; // r8
  __int64 v21; // r9

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  v6 = *(_QWORD *)(a3 + 112);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 128) = v6;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_DWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = a2;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 216) = &unk_4A16C00;
  *(_QWORD *)(a1 + 256) = a1 + 272;
  *(_QWORD *)(a1 + 264) = 0x200000000LL;
  *(_QWORD *)(a1 + 320) = a1 + 336;
  *(_QWORD *)(a1 + 328) = 0x800000000LL;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_DWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_DWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 1;
  *(_QWORD *)(a1 + 400) = a1 + 416;
  *(_QWORD *)(a1 + 208) = a3;
  *(_QWORD *)(a1 + 408) = 0x1000000000LL;
  v7 = (_QWORD *)(a1 + 592);
  do
  {
    if ( v7 )
      *v7 = -4096;
    v7 += 2;
  }
  while ( (_QWORD *)(a1 + 1104) != v7 );
  *(_QWORD *)(a1 + 1632) = 0;
  *(_QWORD *)(a1 + 1640) = 1;
  *(_QWORD *)(a1 + 1104) = a1 + 1120;
  *(_QWORD *)(a1 + 1112) = 0x2000000000LL;
  v8 = (_QWORD *)(a1 + 1648);
  do
  {
    if ( v8 )
      *v8 = -4096;
    v8 += 2;
  }
  while ( (_QWORD *)(a1 + 2160) != v8 );
  *(_QWORD *)(a1 + 2688) = 0;
  *(_QWORD *)(a1 + 2160) = a1 + 2176;
  *(_QWORD *)(a1 + 3152) = a1 + 3168;
  *(_QWORD *)(a1 + 2168) = 0x2000000000LL;
  *(_QWORD *)(a1 + 3568) = a1 + 3592;
  *(_QWORD *)(a1 + 2720) = a1 + 2736;
  *(_QWORD *)(a1 + 3688) = a1 + 3704;
  *(_QWORD *)(a1 + 2728) = 0x1000000000LL;
  *(_QWORD *)(a1 + 3160) = 0x1000000000LL;
  *(_QWORD *)(a1 + 3696) = 0x800000000LL;
  *(_QWORD *)(a1 + 3800) = a1 + 3816;
  *(_QWORD *)(a1 + 3808) = 0x800000000LL;
  *(_QWORD *)(a1 + 2696) = 0;
  *(_QWORD *)(a1 + 2704) = 0;
  *(_DWORD *)(a1 + 2712) = 0;
  *(_QWORD *)(a1 + 3120) = 0;
  *(_QWORD *)(a1 + 3128) = 0;
  *(_QWORD *)(a1 + 3136) = 0;
  *(_DWORD *)(a1 + 3144) = 0;
  *(_QWORD *)(a1 + 3552) = 0;
  *(_QWORD *)(a1 + 3560) = 0;
  *(_QWORD *)(a1 + 3576) = 8;
  *(_DWORD *)(a1 + 3584) = 0;
  *(_BYTE *)(a1 + 3588) = 1;
  *(_QWORD *)(a1 + 3656) = 0;
  *(_QWORD *)(a1 + 3664) = 0;
  *(_QWORD *)(a1 + 3672) = 0;
  *(_DWORD *)(a1 + 3680) = 0;
  *(_QWORD *)(a1 + 3768) = 0;
  *(_QWORD *)(a1 + 3776) = 0;
  *(_QWORD *)(a1 + 3784) = 0;
  *(_DWORD *)(a1 + 3792) = 0;
  *(_QWORD *)(a1 + 3880) = 0;
  *(_QWORD *)(a1 + 3920) = 0x800000000LL;
  *(_QWORD *)(a1 + 3888) = 0;
  *(_QWORD *)(a1 + 3896) = 0;
  *(_DWORD *)(a1 + 3904) = 0;
  *(_QWORD *)(a1 + 3912) = a1 + 3928;
  *(_QWORD *)(a1 + 4120) = 0;
  *(_QWORD *)(a1 + 4128) = 0;
  *(_QWORD *)(a1 + 4136) = 0;
  *(_DWORD *)(a1 + 4144) = 0;
  *(_QWORD *)(a1 + 4152) = a1 + 4168;
  *(_QWORD *)(a1 + 4160) = 0x1000000000LL;
  *(_DWORD *)(a1 + 4296) = a4->m128i_i32[0];
  v9 = a4->m128i_i16[2];
  *(_QWORD *)(a1 + 4320) = 0;
  *(_WORD *)(a1 + 4300) = v9;
  v10 = (void (__fastcall *)(__int64, unsigned __int64 *, __int64))a4[1].m128i_i64[1];
  if ( v10 )
  {
    v10(a1 + 4304, &a4->m128i_u64[1], 2);
    *(_QWORD *)(a1 + 4328) = a4[2].m128i_i64[0];
    *(_QWORD *)(a1 + 4320) = a4[1].m128i_i64[1];
  }
  *(_QWORD *)(a1 + 4352) = 0;
  v11 = (void (__fastcall *)(__int64, unsigned __int64 *, __int64))a4[3].m128i_i64[1];
  if ( v11 )
  {
    v11(a1 + 4336, &a4[2].m128i_u64[1], 2);
    *(_QWORD *)(a1 + 4360) = a4[4].m128i_i64[0];
    *(_QWORD *)(a1 + 4352) = a4[3].m128i_i64[1];
  }
  v12 = _mm_loadu_si128(a4 + 6);
  *(_QWORD *)(a1 + 4368) = a4[4].m128i_i64[1];
  v13 = a4[5].m128i_i64[0];
  *(__m128i *)(a1 + 4392) = v12;
  *(_QWORD *)(a1 + 4376) = v13;
  *(_QWORD *)(a1 + 4384) = a4[5].m128i_i64[1];
  v14 = a4[7].m128i_i64[0];
  *(_QWORD *)(a1 + 4432) = 0;
  *(_QWORD *)(a1 + 4408) = v14;
  v15 = (void (__fastcall *)(__int64, unsigned __int64 *, __int64))a4[8].m128i_i64[1];
  if ( v15 )
  {
    v15(a1 + 4416, &a4[7].m128i_u64[1], 2);
    *(_QWORD *)(a1 + 4440) = a4[9].m128i_i64[0];
    *(_QWORD *)(a1 + 4432) = a4[8].m128i_i64[1];
  }
  result = sub_250EEA0(a1);
  if ( (_BYTE)result )
  {
    v17 = *(__int64 **)(a2 + 32);
    result = *(unsigned int *)(a2 + 40);
    for ( i = &v17[result]; i != v17; ++*(_DWORD *)(a3 + 48) )
    {
      while ( 1 )
      {
        v19 = *v17;
        result = sub_B2DDD0(*v17, 0, 0, 1, 1, 0, 1);
        if ( (_BYTE)result )
          break;
        if ( i == ++v17 )
          return result;
      }
      result = *(unsigned int *)(a3 + 48);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 52) )
      {
        sub_C8D5F0(a3 + 40, (const void *)(a3 + 56), result + 1, 8u, v20, v21);
        result = *(unsigned int *)(a3 + 48);
      }
      ++v17;
      *(_QWORD *)(*(_QWORD *)(a3 + 40) + 8 * result) = v19;
    }
  }
  return result;
}
