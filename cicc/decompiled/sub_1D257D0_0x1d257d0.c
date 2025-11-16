// Function: sub_1D257D0
// Address: 0x1d257d0
//
__int64 __fastcall sub_1D257D0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        char a13,
        char a14)
{
  unsigned __int8 v16; // bl
  __int64 v17; // rax
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __int64 v21; // r15
  int v22; // edx
  __int64 **v23; // r15
  __int64 *v24; // rsi
  __int64 v25; // rsi
  int v26; // edx
  char v27; // bl
  unsigned __int16 v28; // ax
  int v29; // eax
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r12
  int v37; // r13d
  __int128 v38; // [rsp-20h] [rbp-1F0h]
  __int128 v39; // [rsp-20h] [rbp-1F0h]
  unsigned __int8 v40; // [rsp+18h] [rbp-1B8h]
  unsigned __int16 v41; // [rsp+18h] [rbp-1B8h]
  unsigned __int16 v42; // [rsp+18h] [rbp-1B8h]
  int v43; // [rsp+20h] [rbp-1B0h]
  unsigned __int8 v45; // [rsp+28h] [rbp-1A8h]
  __int64 v46; // [rsp+30h] [rbp-1A0h]
  __int64 v47; // [rsp+38h] [rbp-198h]
  char v48; // [rsp+44h] [rbp-18Ch]
  unsigned __int8 *v50; // [rsp+58h] [rbp-178h] BYREF
  _QWORD v51[2]; // [rsp+60h] [rbp-170h] BYREF
  __m128i v52; // [rsp+70h] [rbp-160h]
  __m128i v53; // [rsp+80h] [rbp-150h]
  __m128i v54; // [rsp+90h] [rbp-140h]
  __int64 *v55[3]; // [rsp+A0h] [rbp-130h] BYREF
  char v56; // [rsp+BAh] [rbp-116h]
  char v57; // [rsp+BBh] [rbp-115h]
  __int64 v58[5]; // [rsp+E8h] [rbp-E8h] BYREF
  unsigned __int64 v59[2]; // [rsp+110h] [rbp-C0h] BYREF
  _BYTE v60[176]; // [rsp+120h] [rbp-B0h] BYREF

  v16 = a2;
  v40 = a10;
  v46 = a11;
  v48 = a14;
  v17 = sub_1D252B0((__int64)a1, a2, a3, 1, 0);
  v18 = _mm_loadu_si128((const __m128i *)&a7);
  v19 = _mm_loadu_si128((const __m128i *)&a8);
  v51[0] = a5;
  v20 = _mm_loadu_si128((const __m128i *)&a9);
  v21 = v17;
  v47 = v17;
  v59[1] = 0x2000000000LL;
  v43 = v22;
  v51[1] = a6;
  v59[0] = (unsigned __int64)v60;
  v52 = v18;
  v53 = v19;
  v54 = v20;
  sub_16BD430((__int64)v59, 235);
  sub_16BD4C0((__int64)v59, v21);
  v23 = (__int64 **)v51;
  do
  {
    v24 = *v23;
    v23 += 2;
    sub_16BD4C0((__int64)v59, (__int64)v24);
    sub_16BD430((__int64)v59, *((_DWORD *)v23 - 2));
  }
  while ( v55 != v23 );
  v25 = v16;
  if ( !v16 )
    v25 = a3;
  sub_16BD4D0((__int64)v59, v25);
  *((_QWORD *)&v38 + 1) = v46;
  v26 = *(_DWORD *)(a4 + 8);
  *(_QWORD *)&v38 = v40;
  v45 = v40;
  v50 = 0;
  sub_1D189E0((__int64)v55, 235, v26, &v50, v47, v43, v38, a12);
  v27 = a13 & 3;
  v57 = (16 * (v48 & 1)) | (4 * (a13 & 3)) | v57 & 0xE3;
  HIBYTE(v28) = v57;
  LOBYTE(v28) = v56 & 0xFA;
  if ( v58[0] )
  {
    v41 = v28;
    sub_161E7C0((__int64)v58, v58[0]);
    v28 = v41;
  }
  if ( v50 )
  {
    v42 = v28;
    sub_161E7C0((__int64)&v50, (__int64)v50);
    v28 = v42;
  }
  sub_16BD3E0((__int64)v59, v28);
  v29 = sub_1E340A0(a12);
  sub_16BD430((__int64)v59, v29);
  v55[0] = 0;
  v30 = sub_1D17920((__int64)a1, (__int64)v59, a4, (__int64 *)v55);
  v35 = (__int64)v30;
  if ( v30 )
  {
    sub_1E34340(v30[13], a12, v31, v32, v33, v34);
  }
  else
  {
    v35 = a1[26];
    v37 = *(_DWORD *)(a4 + 8);
    if ( v35 )
      a1[26] = *(_QWORD *)v35;
    else
      v35 = sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v39 + 1) = v46;
    *(_QWORD *)&v39 = v45;
    sub_1D189E0(v35, 235, v37, (unsigned __int8 **)a4, v47, v43, v39, a12);
    *(_BYTE *)(v35 + 27) = *(_BYTE *)(v35 + 27) & 0xE3 | (4 * v27) | (16 * (v48 & 1));
    sub_1D23B60((__int64)a1, v35, (__int64)v51, 4);
    sub_16BDA20(a1 + 40, (__int64 *)v35, v55[0]);
    sub_1D172A0((__int64)a1, v35);
  }
  if ( (_BYTE *)v59[0] != v60 )
    _libc_free(v59[0]);
  return v35;
}
