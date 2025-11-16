// Function: sub_A03720
// Address: 0xa03720
//
__int64 __fastcall sub_A03720(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, char a5, __m128i *a6)
{
  __int64 (__fastcall *v9)(_QWORD, _QWORD, _QWORD); // rax
  __m128i v10; // xmm0
  __int64 v11; // rdx
  __int64 v12; // rax
  __m128i v13; // xmm4
  bool v14; // zf
  void (__fastcall *v15)(__m128i *, __m128i *, __int64); // rax
  __m128i v16; // xmm3
  __int64 v17; // rdx
  __m128i v18; // xmm0
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rcx
  __m128i v23; // xmm0
  __m128i v24; // xmm1
  __int64 (__fastcall *v25)(_QWORD, _QWORD, _QWORD); // rdx
  __m128i v26; // xmm0
  __int64 v27; // rcx
  __int64 v28; // rdx
  __m128i v29; // xmm2
  __int64 v30; // rcx
  __int64 v31; // rdx
  char v32; // cl
  _QWORD *v33; // rdx
  __int64 result; // rax
  void (__fastcall *v35)(_QWORD, _QWORD, _QWORD); // rax
  __m128i v36; // xmm0
  __m128i v37; // xmm5
  __int64 v38; // rdx
  __int64 v39; // rax
  void (__fastcall *v40)(_QWORD, _QWORD, _QWORD); // rdx
  __m128i v41; // xmm0
  __m128i v42; // xmm6
  __int64 v43; // rsi
  __int64 v44; // rdx
  __m128i v45; // [rsp+0h] [rbp-A0h] BYREF
  __int64 (__fastcall *v46)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-90h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  __m128i v48; // [rsp+20h] [rbp-80h] BYREF
  void (__fastcall *v49)(__m128i *, __m128i *, __int64); // [rsp+30h] [rbp-70h]
  __int64 v50; // [rsp+38h] [rbp-68h]
  __m128i v51; // [rsp+40h] [rbp-60h] BYREF
  void (__fastcall *v52)(_QWORD, _QWORD, _QWORD); // [rsp+50h] [rbp-50h]
  __int64 v53; // [rsp+58h] [rbp-48h]
  char v54; // [rsp+60h] [rbp-40h]

  v9 = (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))a6[1].m128i_i64[0];
  v54 = 0;
  v10 = _mm_loadu_si128(a6);
  v11 = v47;
  a6[1].m128i_i64[0] = 0;
  v46 = v9;
  v12 = a6[1].m128i_i64[1];
  v13 = _mm_loadu_si128(&v48);
  v14 = a6[6].m128i_i8[0] == 0;
  a6[1].m128i_i64[1] = v11;
  v47 = v12;
  v15 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a6[3].m128i_i64[0];
  v16 = _mm_loadu_si128(&v45);
  v17 = v50;
  v45 = v10;
  v49 = v15;
  v18 = _mm_loadu_si128(a6 + 2);
  v19 = a6[3].m128i_i64[1];
  a6[3].m128i_i64[0] = 0;
  a6[3].m128i_i64[1] = v17;
  v50 = v19;
  *a6 = v16;
  a6[2] = v13;
  v48 = v18;
  if ( !v14 )
  {
    v35 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a6[5].m128i_i64[0];
    v36 = _mm_loadu_si128(a6 + 4);
    a6[5].m128i_i64[0] = 0;
    v37 = _mm_loadu_si128(&v51);
    v38 = v53;
    v54 = 1;
    v52 = v35;
    v39 = a6[5].m128i_i64[1];
    a6[4] = v37;
    a6[5].m128i_i64[1] = v38;
    v53 = v39;
    v51 = v36;
  }
  v20 = sub_22077B0(1144);
  if ( v20 )
  {
    v21 = *(_QWORD *)(a2 + 8);
    v22 = *a3;
    *(_QWORD *)v20 = v20 + 16;
    v23 = _mm_loadu_si128(&v45);
    *(_QWORD *)(v20 + 8) = 0x100000000LL;
    v24 = _mm_loadu_si128((const __m128i *)(v20 + 264));
    *(_QWORD *)(v20 + 192) = 0x100000000LL;
    *(_QWORD *)(v20 + 216) = v22;
    if ( v21 > 0xFFFFFFFE )
      LODWORD(v21) = -1;
    *(_QWORD *)(v20 + 248) = v22;
    *(__m128i *)(v20 + 264) = v23;
    *(_DWORD *)(v20 + 224) = v21;
    v25 = v46;
    *(_QWORD *)(v20 + 24) = 0;
    *(_QWORD *)(v20 + 32) = 1;
    *(_DWORD *)(v20 + 40) = -1;
    *(_QWORD *)(v20 + 56) = 0;
    *(_QWORD *)(v20 + 64) = 1;
    *(_DWORD *)(v20 + 72) = -1;
    *(_QWORD *)(v20 + 88) = 0;
    *(_QWORD *)(v20 + 96) = 1;
    *(_QWORD *)(v20 + 104) = -4096;
    *(_QWORD *)(v20 + 120) = 0;
    *(_QWORD *)(v20 + 128) = 1;
    *(_QWORD *)(v20 + 136) = -4096;
    *(_QWORD *)(v20 + 152) = 0;
    *(_QWORD *)(v20 + 160) = 1;
    *(_QWORD *)(v20 + 168) = -4096;
    *(_QWORD *)(v20 + 184) = v20 + 200;
    *(_QWORD *)(v20 + 232) = a4;
    *(_QWORD *)(v20 + 240) = a2;
    *(_QWORD *)(v20 + 256) = a3;
    v45 = v24;
    v46 = 0;
    v26 = _mm_loadu_si128(&v48);
    v27 = *(_QWORD *)(v20 + 288);
    *(_QWORD *)(v20 + 280) = v25;
    v28 = v47;
    v29 = _mm_loadu_si128((const __m128i *)(v20 + 296));
    *(_BYTE *)(v20 + 360) = 0;
    v47 = v27;
    v30 = *(_QWORD *)(v20 + 320);
    *(_QWORD *)(v20 + 288) = v28;
    v48 = v29;
    *(_QWORD *)(v20 + 312) = v49;
    v31 = v50;
    v50 = v30;
    v32 = v54;
    v49 = 0;
    *(_QWORD *)(v20 + 320) = v31;
    *(__m128i *)(v20 + 296) = v26;
    if ( v32 )
    {
      v40 = v52;
      v41 = _mm_loadu_si128(&v51);
      v52 = 0;
      v42 = _mm_loadu_si128((const __m128i *)(v20 + 328));
      v43 = *(_QWORD *)(v20 + 352);
      *(_BYTE *)(v20 + 360) = 1;
      *(_QWORD *)(v20 + 344) = v40;
      v44 = v53;
      v51 = v42;
      v53 = v43;
      *(_QWORD *)(v20 + 352) = v44;
      *(__m128i *)(v20 + 328) = v41;
    }
    *(_QWORD *)(v20 + 368) = 0;
    *(_QWORD *)(v20 + 400) = 0x200000000LL;
    *(_QWORD *)(v20 + 376) = 0;
    *(_QWORD *)(v20 + 384) = 0;
    *(_QWORD *)(v20 + 392) = 0;
    *(_QWORD *)(v20 + 408) = 0;
    *(_QWORD *)(v20 + 416) = 0;
    *(_QWORD *)(v20 + 424) = 0;
    *(_QWORD *)(v20 + 440) = 0x800000000LL;
    *(_QWORD *)(v20 + 704) = 0;
    *(_QWORD *)(v20 + 712) = 0;
    *(_QWORD *)(v20 + 720) = 0;
    *(_QWORD *)(v20 + 728) = 0;
    *(_QWORD *)(v20 + 736) = 0;
    *(_QWORD *)(v20 + 744) = 0;
    *(_QWORD *)(v20 + 752) = 0;
    *(_QWORD *)(v20 + 760) = 0;
    *(_QWORD *)(v20 + 768) = 0;
    *(_QWORD *)(v20 + 776) = 0;
    *(_QWORD *)(v20 + 784) = 0;
    *(_QWORD *)(v20 + 792) = 0;
    *(_QWORD *)(v20 + 800) = 1;
    *(_QWORD *)(v20 + 432) = v20 + 448;
    v33 = (_QWORD *)(v20 + 808);
    do
    {
      if ( v33 )
        *v33 = -4096;
      v33 += 2;
    }
    while ( v33 != (_QWORD *)(v20 + 1064) );
    *(_QWORD *)(v20 + 1064) = 0;
    *(_QWORD *)(v20 + 1072) = 0;
    *(_QWORD *)(v20 + 1080) = 0;
    *(_DWORD *)(v20 + 1088) = 0;
    *(_DWORD *)(v20 + 1096) = 0;
    *(_QWORD *)(v20 + 1104) = 0;
    *(_QWORD *)(v20 + 1112) = 0;
    *(_QWORD *)(v20 + 1120) = 0;
    *(_DWORD *)(v20 + 1128) = 0;
    *(_BYTE *)(v20 + 1136) = a5;
  }
  else
  {
    v32 = v54;
  }
  *a1 = v20;
  if ( v32 )
  {
    v54 = 0;
    if ( v52 )
      v52(&v51, &v51, 3);
  }
  if ( v49 )
    v49(&v48, &v48, 3);
  result = (__int64)v46;
  if ( v46 )
    return v46(&v45, &v45, 3);
  return result;
}
