// Function: sub_2975000
// Address: 0x2975000
//
__int64 __fastcall sub_2975000(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  void (__fastcall *v9)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v10; // rdx
  __m128i v11; // xmm1
  __m128i v12; // xmm0
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  __m128i v16; // xmm2
  void (__fastcall *v17)(_QWORD, _QWORD, _QWORD); // rax
  __m128i v18; // xmm0
  __int64 v19; // rax
  __int128 *v20; // rax
  __m128i v22; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v23)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-30h]
  __int64 v24; // [rsp+18h] [rbp-28h]

  v9 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a1[1].m128i_i64[0];
  v10 = v24;
  a1[1].m128i_i64[0] = 0;
  v11 = _mm_loadu_si128(&v22);
  v12 = _mm_loadu_si128(a1);
  v23 = v9;
  v13 = a1[1].m128i_i64[1];
  a1[1].m128i_i64[1] = v10;
  *a1 = v11;
  v24 = v13;
  v22 = v12;
  v14 = sub_22077B0(0xE8u);
  v15 = v14;
  if ( v14 )
  {
    *(_QWORD *)(v14 + 8) = 0;
    v16 = _mm_loadu_si128((const __m128i *)(v14 + 200));
    *(_QWORD *)(v14 + 16) = &unk_500660C;
    *(_QWORD *)(v14 + 56) = v14 + 104;
    *(_QWORD *)(v14 + 112) = v14 + 160;
    *(_QWORD *)v14 = off_4A22140;
    v17 = v23;
    *(_DWORD *)(v15 + 88) = 1065353216;
    *(_DWORD *)(v15 + 144) = 1065353216;
    v18 = _mm_loadu_si128(&v22);
    *(_DWORD *)(v15 + 24) = 2;
    v22 = v16;
    *(__m128i *)(v15 + 200) = v18;
    *(_QWORD *)(v15 + 32) = 0;
    *(_QWORD *)(v15 + 40) = 0;
    *(_QWORD *)(v15 + 48) = 0;
    *(_QWORD *)(v15 + 64) = 1;
    *(_QWORD *)(v15 + 72) = 0;
    *(_QWORD *)(v15 + 80) = 0;
    *(_QWORD *)(v15 + 96) = 0;
    *(_QWORD *)(v15 + 104) = 0;
    *(_QWORD *)(v15 + 120) = 1;
    *(_QWORD *)(v15 + 128) = 0;
    *(_QWORD *)(v15 + 136) = 0;
    *(_QWORD *)(v15 + 152) = 0;
    *(_QWORD *)(v15 + 160) = 0;
    *(_BYTE *)(v15 + 168) = 0;
    *(_QWORD *)(v15 + 176) = a7;
    *(_QWORD *)(v15 + 184) = a8;
    *(_QWORD *)(v15 + 192) = a9;
    v23 = 0;
    *(_QWORD *)(v15 + 216) = v17;
    v19 = v24;
    v24 = *(_QWORD *)(v15 + 224);
    *(_QWORD *)(v15 + 224) = v19;
    v20 = sub_BC2B00();
    sub_2974DC0((__int64)v20);
    sub_2973D40(v15 + 176);
  }
  if ( v23 )
    v23(&v22, &v22, 3);
  return v15;
}
