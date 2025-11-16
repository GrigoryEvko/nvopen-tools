// Function: sub_1D2C870
// Address: 0x1d2c870
//
__int64 __fastcall sub_1D2C870(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        char a12,
        char a13)
{
  unsigned __int8 *v16; // rax
  __int64 v17; // rax
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  int v20; // edx
  __int64 **v21; // rbx
  __int64 *v22; // rsi
  __int64 v23; // rsi
  int v24; // edx
  unsigned __int16 v25; // ax
  int v26; // eax
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rbx
  int v34; // r12d
  __int128 v35; // [rsp-20h] [rbp-1F0h]
  __int128 v36; // [rsp-20h] [rbp-1F0h]
  unsigned __int8 v38; // [rsp+18h] [rbp-1B8h]
  int v39; // [rsp+20h] [rbp-1B0h]
  unsigned __int16 v40; // [rsp+20h] [rbp-1B0h]
  unsigned __int16 v41; // [rsp+20h] [rbp-1B0h]
  __int64 v42; // [rsp+28h] [rbp-1A8h]
  int v43; // [rsp+28h] [rbp-1A8h]
  unsigned __int8 v44; // [rsp+37h] [rbp-199h]
  __int64 v45; // [rsp+38h] [rbp-198h]
  char v46; // [rsp+40h] [rbp-190h]
  char v47; // [rsp+44h] [rbp-18Ch]
  __int64 v48; // [rsp+48h] [rbp-188h]
  unsigned __int8 *v49; // [rsp+58h] [rbp-178h] BYREF
  _QWORD v50[2]; // [rsp+60h] [rbp-170h] BYREF
  __m128i v51; // [rsp+70h] [rbp-160h]
  __m128i v52; // [rsp+80h] [rbp-150h]
  __int64 v53; // [rsp+90h] [rbp-140h]
  __int64 v54; // [rsp+98h] [rbp-138h]
  __int64 *v55[3]; // [rsp+A0h] [rbp-130h] BYREF
  char v56; // [rsp+BAh] [rbp-116h]
  char v57; // [rsp+BBh] [rbp-115h]
  __int64 v58[5]; // [rsp+E8h] [rbp-E8h] BYREF
  unsigned __int64 v59[2]; // [rsp+110h] [rbp-C0h] BYREF
  _BYTE v60[176]; // [rsp+120h] [rbp-B0h] BYREF

  v38 = a9;
  v45 = a10;
  v47 = a12;
  v46 = a13;
  v16 = (unsigned __int8 *)(*(_QWORD *)(a5 + 40) + 16LL * (unsigned int)a6);
  v44 = *v16;
  v42 = *((_QWORD *)v16 + 1);
  v17 = sub_1D29190((__int64)a1, 1u, 0, *v16, a5, a6);
  v18 = _mm_loadu_si128((const __m128i *)&a7);
  v50[1] = a3;
  v19 = _mm_loadu_si128((const __m128i *)&a8);
  v39 = v20;
  v48 = v17;
  v50[0] = a2;
  v53 = a5;
  v54 = a6;
  v59[0] = (unsigned __int64)v60;
  v59[1] = 0x2000000000LL;
  v51 = v18;
  v52 = v19;
  sub_16BD430((__int64)v59, 236);
  sub_16BD4C0((__int64)v59, v48);
  v21 = (__int64 **)v50;
  do
  {
    v22 = *v21;
    v21 += 2;
    sub_16BD4C0((__int64)v59, (__int64)v22);
    sub_16BD430((__int64)v59, *((_DWORD *)v21 - 2));
  }
  while ( v55 != v21 );
  v23 = v44;
  if ( !v44 )
    v23 = v42;
  sub_16BD4D0((__int64)v59, v23);
  *((_QWORD *)&v35 + 1) = v45;
  v43 = v39;
  v24 = *(_DWORD *)(a4 + 8);
  v49 = 0;
  *(_QWORD *)&v35 = v38;
  sub_1D189E0((__int64)v55, 236, v24, &v49, v48, v39, v35, a11);
  v57 = (8 * (v46 & 1)) | (4 * (v47 & 1)) | v57 & 0xF3;
  HIBYTE(v25) = v57;
  LOBYTE(v25) = v56 & 0xFA;
  if ( v58[0] )
  {
    v40 = v25;
    sub_161E7C0((__int64)v58, v58[0]);
    v25 = v40;
  }
  if ( v49 )
  {
    v41 = v25;
    sub_161E7C0((__int64)&v49, (__int64)v49);
    v25 = v41;
  }
  sub_16BD3E0((__int64)v59, v25);
  v26 = sub_1E340A0(a11);
  sub_16BD430((__int64)v59, v26);
  v55[0] = 0;
  v27 = sub_1D17920((__int64)a1, (__int64)v59, a4, (__int64 *)v55);
  v32 = (__int64)v27;
  if ( v27 )
  {
    sub_1E34340(v27[13], a11, v28, v29, v30, v31);
  }
  else
  {
    v32 = a1[26];
    v34 = *(_DWORD *)(a4 + 8);
    if ( v32 )
      a1[26] = *(_QWORD *)v32;
    else
      v32 = sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v36 + 1) = v45;
    *(_QWORD *)&v36 = v38;
    sub_1D189E0(v32, 236, v34, (unsigned __int8 **)a4, v48, v43, v36, a11);
    *(_BYTE *)(v32 + 27) = *(_BYTE *)(v32 + 27) & 0xF3 | (4 * (v47 & 1)) | (8 * (v46 & 1));
    sub_1D23B60((__int64)a1, v32, (__int64)v50, 4);
    sub_16BDA20(a1 + 40, (__int64 *)v32, v55[0]);
    sub_1D172A0((__int64)a1, v32);
  }
  if ( (_BYTE *)v59[0] != v60 )
    _libc_free(v59[0]);
  return v32;
}
