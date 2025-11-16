// Function: sub_37B2DF0
// Address: 0x37b2df0
//
void __fastcall sub_37B2DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned __int64 v8; // rsi
  __int64 v9; // rax
  __int16 v10; // dx
  __int64 v11; // rax
  __int16 *v12; // rdx
  __m128i v13; // xmm2
  __int64 v14; // rax
  __int16 v15; // ax
  __int64 v16; // rdx
  unsigned int v17; // eax
  const void *v18; // r9
  unsigned int v19; // r12d
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r8
  __int64 v26; // r9
  __m128i v27; // xmm4
  unsigned __int64 v28; // r12
  __m128i v29; // xmm5
  __m128i v30; // xmm6
  __int64 v31; // rbx
  _BYTE *v32; // rdi
  _BYTE *v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __m128i v38; // xmm0
  __m128i v39; // xmm1
  __m128i v40; // xmm2
  __m128i v41; // xmm3
  __m128i v42; // xmm4
  __int64 v43; // rax
  __int64 v44; // r9
  __m128i v45; // xmm6
  __m128i v46; // xmm7
  __m128i v47; // xmm1
  _BYTE *v48; // rdi
  __int64 n; // [rsp+18h] [rbp-398h]
  __int64 v52; // [rsp+28h] [rbp-388h]
  int v53; // [rsp+80h] [rbp-330h]
  const void *v54; // [rsp+80h] [rbp-330h]
  int v55; // [rsp+90h] [rbp-320h]
  __int64 v56; // [rsp+90h] [rbp-320h]
  unsigned int v57; // [rsp+9Ch] [rbp-314h]
  char v58; // [rsp+DAh] [rbp-2D6h] BYREF
  char v59; // [rsp+DBh] [rbp-2D5h] BYREF
  int v60; // [rsp+DCh] [rbp-2D4h] BYREF
  __int64 v61; // [rsp+E0h] [rbp-2D0h] BYREF
  int v62; // [rsp+E8h] [rbp-2C8h]
  __m128i v63; // [rsp+F0h] [rbp-2C0h] BYREF
  _QWORD v64[2]; // [rsp+100h] [rbp-2B0h] BYREF
  _QWORD v65[2]; // [rsp+110h] [rbp-2A0h] BYREF
  unsigned int v66; // [rsp+120h] [rbp-290h]
  _QWORD v67[2]; // [rsp+130h] [rbp-280h] BYREF
  __m128i v68; // [rsp+140h] [rbp-270h]
  unsigned int v69; // [rsp+150h] [rbp-260h] BYREF
  __int64 v70; // [rsp+158h] [rbp-258h]
  __m128i v71; // [rsp+160h] [rbp-250h]
  __int64 *v72; // [rsp+170h] [rbp-240h]
  __int64 v73[3]; // [rsp+180h] [rbp-230h] BYREF
  unsigned int v74; // [rsp+198h] [rbp-218h]
  __int64 *v75; // [rsp+1A0h] [rbp-210h]
  _QWORD v76[2]; // [rsp+1B0h] [rbp-200h] BYREF
  __m128i v77; // [rsp+1C0h] [rbp-1F0h]
  __int64 *v78; // [rsp+1D0h] [rbp-1E0h]
  __m128i *v79; // [rsp+1D8h] [rbp-1D8h]
  int *v80; // [rsp+1E0h] [rbp-1D0h]
  __m128i v81; // [rsp+1F0h] [rbp-1C0h] BYREF
  __m128i v82; // [rsp+200h] [rbp-1B0h] BYREF
  __m128i v83; // [rsp+210h] [rbp-1A0h] BYREF
  __m128i v84; // [rsp+220h] [rbp-190h] BYREF
  _BYTE *v85; // [rsp+230h] [rbp-180h] BYREF
  __int64 v86; // [rsp+238h] [rbp-178h]
  _BYTE v87[48]; // [rsp+240h] [rbp-170h] BYREF
  __m128i v88; // [rsp+270h] [rbp-140h] BYREF
  __m128i v89; // [rsp+280h] [rbp-130h] BYREF
  __m128i v90; // [rsp+290h] [rbp-120h] BYREF
  __m128i v91; // [rsp+2A0h] [rbp-110h] BYREF
  void *v92; // [rsp+2B0h] [rbp-100h] BYREF
  __int64 v93; // [rsp+2B8h] [rbp-F8h]
  _BYTE s[48]; // [rsp+2C0h] [rbp-F0h] BYREF
  _OWORD v95[4]; // [rsp+2F0h] [rbp-C0h] BYREF
  _QWORD *v96; // [rsp+330h] [rbp-80h] BYREF
  __int64 v97; // [rsp+338h] [rbp-78h]
  __int64 v98; // [rsp+340h] [rbp-70h]
  __m128i v99; // [rsp+348h] [rbp-68h]
  __int64 *v100; // [rsp+358h] [rbp-58h]
  __m128i *v101; // [rsp+360h] [rbp-50h]
  _OWORD *v102; // [rsp+368h] [rbp-48h]
  int *v103; // [rsp+370h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 80);
  v61 = v6;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  if ( v6 )
    sub_B96E90((__int64)&v61, v6, 1);
  v62 = *(_DWORD *)(a2 + 72);
  sub_375E8D0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)&v81, (__int64)&v82);
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(v7 + 40);
  sub_375E8D0(a1, v8, *(_QWORD *)(v7 + 48), (__int64)&v83, (__int64)&v84);
  v9 = *(_QWORD *)(v81.m128i_i64[0] + 48) + 16LL * v81.m128i_u32[2];
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v63.m128i_i16[0] = v10;
  v63.m128i_i64[1] = v11;
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 176) > 0x34u )
    {
LABEL_5:
      v57 = word_4456340[v63.m128i_u16[0] - 1];
      goto LABEL_8;
    }
  }
  else if ( !sub_3007100((__int64)&v63) )
  {
    goto LABEL_7;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v63.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v63.m128i_i16[0] - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_5;
  }
LABEL_7:
  v57 = sub_3007130((__int64)&v63, v8);
LABEL_8:
  v12 = *(__int16 **)(a2 + 48);
  v72 = &v61;
  v13 = _mm_loadu_si128(&v63);
  v69 = v57;
  v14 = *(_QWORD *)(a1 + 8);
  v71 = v13;
  v70 = v14;
  v15 = *v12;
  v16 = *((_QWORD *)v12 + 1);
  LOWORD(v96) = v15;
  v97 = v16;
  if ( v15 )
  {
    if ( (unsigned __int16)(v15 - 176) > 0x34u )
    {
LABEL_10:
      v17 = word_4456340[(unsigned __int16)v96 - 1];
      goto LABEL_13;
    }
  }
  else if ( !sub_3007100((__int64)&v96) )
  {
    goto LABEL_12;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v96 )
  {
    if ( (unsigned __int16)((_WORD)v96 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_10;
  }
LABEL_12:
  v17 = sub_3007130((__int64)&v96, v8);
LABEL_13:
  v18 = *(const void **)(a2 + 96);
  v19 = v17;
  v20 = 4LL * v17;
  v85 = v87;
  v86 = 0xC00000000LL;
  if ( v17 > 0xCuLL )
  {
    v54 = v18;
    v56 = 4LL * v17;
    sub_C8D5F0((__int64)&v85, v87, v17, 4u, v20, (__int64)v18);
    v20 = v56;
    v18 = v54;
    v48 = &v85[4 * (unsigned int)v86];
    goto LABEL_33;
  }
  if ( v20 )
  {
    v48 = v87;
LABEL_33:
    memcpy(v48, v18, v20);
    LODWORD(v20) = v86;
  }
  v73[1] = (__int64)&v63;
  LODWORD(v86) = v20 + v19;
  v73[2] = a1;
  v74 = v57;
  v73[0] = (__int64)&v81;
  v75 = &v61;
  sub_37B1220(v73, (__int64)&v85);
  v65[0] = &v81;
  v65[1] = &v58;
  v66 = v57;
  sub_37B25C0((__int64)v65, (__int64)&v85, v21, v22, v23, v24);
  v27 = _mm_loadu_si128(&v82);
  v28 = 4LL * v57;
  v29 = _mm_loadu_si128(&v83);
  v30 = _mm_loadu_si128(&v84);
  n = 16LL * v57;
  v53 = 0;
  v55 = 2;
  v52 = a1;
  v31 = a3;
  v88 = _mm_loadu_si128(&v81);
  v89 = v27;
  v90 = v29;
  v91 = v30;
  while ( 1 )
  {
    v92 = s;
    v93 = 0xC00000000LL;
    if ( v28 > 0xC )
    {
      sub_C8D5F0((__int64)&v92, s, v28, 4u, v25, v26);
      memset(v92, 255, n);
      v32 = v92;
      LODWORD(v93) = 4 * v57;
      v33 = &v85[4 * v53];
LABEL_21:
      memmove(v32, v33, v28);
      goto LABEL_22;
    }
    if ( v28 )
    {
      if ( n )
        memset(s, 255, n);
      v32 = s;
      LODWORD(v93) = 4 * v57;
      v33 = &v85[4 * v53];
      goto LABEL_21;
    }
    LODWORD(v93) = 0;
LABEL_22:
    sub_37B1220(v73, (__int64)&v92);
    sub_37B25C0((__int64)v65, (__int64)&v92, v34, v35, v36, v37);
    v38 = _mm_loadu_si128(&v63);
    v39 = _mm_loadu_si128(&v81);
    v64[0] = &v60;
    v102 = v95;
    v64[1] = &v59;
    v40 = _mm_loadu_si128(&v82);
    v41 = _mm_loadu_si128(&v83);
    v95[0] = v39;
    v96 = v64;
    v42 = _mm_loadu_si128(&v84);
    v101 = &v81;
    v43 = *(_QWORD *)(v52 + 8);
    v103 = (int *)&v69;
    v98 = v43;
    v95[1] = v40;
    v95[2] = v41;
    v95[3] = v42;
    v99 = v38;
    v60 = -1;
    v59 = 0;
    v97 = v31;
    v100 = &v61;
    v76[0] = v31;
    v76[1] = v43;
    v79 = &v81;
    v67[1] = v43;
    v80 = (int *)&v69;
    v78 = &v61;
    v67[0] = v31;
    v77 = v38;
    v68 = v38;
    sub_9C1970(
      (__int64)v92,
      (unsigned int)v93,
      4u,
      4u,
      1u,
      v44,
      (void (__fastcall *)(__int64))sub_3774B00,
      (__int64)v67,
      (void (__fastcall *)(__int64, _QWORD, _QWORD, __int64, _QWORD, __int64))sub_3775050,
      (__int64)v76,
      (void (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD, __int64))sub_3775190,
      (__int64)&v96);
    v45 = _mm_loadu_si128(&v89);
    v46 = _mm_loadu_si128(&v90);
    v47 = _mm_loadu_si128(&v91);
    v81 = _mm_loadu_si128(&v88);
    v82 = v45;
    v83 = v46;
    v84 = v47;
    if ( v92 != s )
      _libc_free((unsigned __int64)v92);
    v53 += v57;
    if ( v55 == 1 )
      break;
    v55 = 1;
    v31 = a4;
  }
  if ( v85 != v87 )
    _libc_free((unsigned __int64)v85);
  if ( v61 )
    sub_B91220((__int64)&v61, v61);
}
