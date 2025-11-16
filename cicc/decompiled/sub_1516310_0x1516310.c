// Function: sub_1516310
// Address: 0x1516310
//
__int64 *__fastcall sub_1516310(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __m128i *a5, char *a6)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __m128i v10; // xmm2
  __m128i v11; // xmm0
  __int64 v12; // rax
  char v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __m128i v16; // xmm1
  void (__fastcall *v17)(_QWORD, _QWORD, _QWORD); // rdx
  __int64 v18; // rdx
  _QWORD *v19; // rdx
  void (__fastcall *v21)(__m128i *, __m128i *, __int64); // rax
  __m128i v22; // [rsp+0h] [rbp-50h] BYREF
  void (__fastcall *v23)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v8 = a5[1].m128i_i64[0];
  v9 = v24;
  a5[1].m128i_i64[0] = 0;
  v10 = _mm_loadu_si128(&v22);
  v11 = _mm_loadu_si128(a5);
  v23 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v8;
  v12 = a5[1].m128i_i64[1];
  a5[1].m128i_i64[1] = v9;
  *a5 = v10;
  v13 = *a6;
  v24 = v12;
  v22 = v11;
  v14 = sub_22077B0(1016);
  if ( v14 )
  {
    v15 = *a3;
    v16 = _mm_loadu_si128(&v22);
    *(_QWORD *)(v14 + 248) = a3;
    *(_QWORD *)v14 = v14 + 16;
    *(_QWORD *)(v14 + 216) = v15;
    *(_QWORD *)(v14 + 240) = v15;
    v17 = v23;
    *(_QWORD *)(v14 + 8) = 0x100000000LL;
    *(_QWORD *)(v14 + 272) = v17;
    v18 = v24;
    *(_QWORD *)(v14 + 192) = 0x100000000LL;
    *(_QWORD *)(v14 + 280) = v18;
    *(_QWORD *)(v14 + 24) = 0;
    *(_QWORD *)(v14 + 32) = 1;
    *(_DWORD *)(v14 + 40) = -1;
    *(_QWORD *)(v14 + 56) = 0;
    *(_QWORD *)(v14 + 64) = 1;
    *(_DWORD *)(v14 + 72) = -1;
    *(_QWORD *)(v14 + 88) = 0;
    *(_QWORD *)(v14 + 96) = 1;
    *(_QWORD *)(v14 + 104) = -8;
    *(_QWORD *)(v14 + 120) = 0;
    *(_QWORD *)(v14 + 128) = 1;
    *(_QWORD *)(v14 + 136) = -8;
    *(_QWORD *)(v14 + 152) = 0;
    *(_QWORD *)(v14 + 160) = 1;
    *(_QWORD *)(v14 + 168) = -8;
    *(_QWORD *)(v14 + 184) = v14 + 200;
    *(_QWORD *)(v14 + 224) = a4;
    *(_QWORD *)(v14 + 232) = a2;
    *(_QWORD *)(v14 + 288) = 0;
    *(__m128i *)(v14 + 256) = v16;
    *(_QWORD *)(v14 + 296) = 0;
    *(_QWORD *)(v14 + 304) = 0;
    *(_QWORD *)(v14 + 312) = 0;
    *(_QWORD *)(v14 + 320) = 0x200000000LL;
    *(_QWORD *)(v14 + 328) = 0;
    *(_QWORD *)(v14 + 336) = 0;
    *(_QWORD *)(v14 + 344) = 0;
    *(_QWORD *)(v14 + 360) = 0x800000000LL;
    *(_QWORD *)(v14 + 624) = 0;
    *(_QWORD *)(v14 + 632) = 0;
    *(_QWORD *)(v14 + 640) = 0;
    *(_QWORD *)(v14 + 648) = 0;
    *(_QWORD *)(v14 + 656) = 0;
    *(_QWORD *)(v14 + 664) = 0;
    *(_QWORD *)(v14 + 672) = 0;
    *(_QWORD *)(v14 + 680) = 0;
    *(_QWORD *)(v14 + 688) = 0;
    *(_QWORD *)(v14 + 696) = 0;
    *(_QWORD *)(v14 + 704) = 0;
    *(_QWORD *)(v14 + 712) = 1;
    *(_QWORD *)(v14 + 352) = v14 + 368;
    v19 = (_QWORD *)(v14 + 720);
    do
    {
      if ( v19 )
        *v19 = -8;
      v19 += 2;
    }
    while ( (_QWORD *)(v14 + 976) != v19 );
    *(_QWORD *)(v14 + 976) = 0;
    *(_QWORD *)(v14 + 984) = 0;
    *(_QWORD *)(v14 + 992) = 0;
    *(_DWORD *)(v14 + 1000) = 0;
    *(_DWORD *)(v14 + 1008) = 0;
    *(_BYTE *)(v14 + 1012) = v13;
    *a1 = v14;
  }
  else
  {
    v21 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v23;
    *a1 = 0;
    if ( v21 )
      v21(&v22, &v22, 3);
  }
  return a1;
}
