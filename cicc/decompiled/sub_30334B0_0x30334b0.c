// Function: sub_30334B0
// Address: 0x30334b0
//
void __fastcall sub_30334B0(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 v11; // rsi
  __int16 *v12; // rdx
  __int16 v13; // ax
  __int64 v14; // rdx
  __int32 v15; // r10d
  int *v16; // rdi
  __int64 v17; // r8
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int32 v21; // r12d
  int v22; // r15d
  int *v23; // rax
  __int64 v24; // r15
  __int64 v25; // r8
  int *v26; // rax
  __int64 v27; // rdi
  unsigned __int64 v28; // rsi
  __int64 v29; // r11
  __int64 v30; // rdx
  __int64 v31; // rax
  __m128i v32; // xmm1
  __m128i v33; // xmm2
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 *v37; // r15
  __int64 v38; // r8
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r9
  __int32 v42; // r10d
  __int32 v43; // r11d
  unsigned __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // rax
  __m128i v47; // xmm0
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rsi
  __int64 v51; // r9
  _BYTE *v52; // r14
  __int64 v53; // rax
  __int32 v54; // r10d
  __int64 v55; // r9
  _BYTE *v56; // rsi
  unsigned __int64 v57; // r13
  __int128 v58; // rax
  unsigned int v59; // r12d
  __int32 v60; // r15d
  unsigned __int64 v61; // rcx
  __int64 v62; // r14
  unsigned __int64 v63; // rbx
  __int64 *v64; // rdx
  __int64 v65; // rdx
  __int64 v66; // r8
  __int64 v67; // r11
  __int64 v68; // rax
  __int64 v69; // r12
  __int64 v70; // r9
  unsigned __int32 v71; // r10d
  __int64 *v72; // rax
  __int64 v73; // rbx
  __int64 v74; // rdx
  unsigned __int64 v75; // rax
  __int64 *v76; // rdx
  unsigned __int64 v77; // rdi
  __m128i v78; // xmm0
  __int32 v79; // [rsp+8h] [rbp-1D8h]
  __int32 v80; // [rsp+8h] [rbp-1D8h]
  __int32 v81; // [rsp+8h] [rbp-1D8h]
  __m128i v82; // [rsp+10h] [rbp-1D0h] BYREF
  __m128i v83; // [rsp+20h] [rbp-1C0h] BYREF
  __int64 v84; // [rsp+30h] [rbp-1B0h]
  _OWORD *v85; // [rsp+38h] [rbp-1A8h]
  int *v86; // [rsp+40h] [rbp-1A0h]
  __int64 v87; // [rsp+48h] [rbp-198h]
  __int64 v88; // [rsp+50h] [rbp-190h] BYREF
  int v89; // [rsp+58h] [rbp-188h]
  int v90; // [rsp+60h] [rbp-180h] BYREF
  __int64 v91; // [rsp+68h] [rbp-178h]
  _BYTE *v92; // [rsp+70h] [rbp-170h] BYREF
  __int64 v93; // [rsp+78h] [rbp-168h]
  _BYTE v94[64]; // [rsp+80h] [rbp-160h] BYREF
  int *v95; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v96; // [rsp+C8h] [rbp-118h]
  _BYTE v97[80]; // [rsp+D0h] [rbp-110h] BYREF
  _OWORD *v98; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+128h] [rbp-B8h]
  _OWORD v100[11]; // [rsp+130h] [rbp-B0h] BYREF

  v87 = a2;
  v11 = *(_QWORD *)(a1 + 80);
  v88 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v88, v11, 1);
  v12 = *(__int16 **)(a1 + 48);
  v89 = *(_DWORD *)(a1 + 72);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOWORD(v90) = v13;
  v91 = v14;
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 17) > 0xD3u )
      goto LABEL_5;
    if ( (unsigned __int16)(v13 - 176) > 0x34u )
      goto LABEL_9;
  }
  else
  {
    v86 = &v90;
    if ( !sub_30070B0((__int64)&v90) )
      goto LABEL_5;
    v16 = v86;
    if ( !sub_3007100((__int64)v86) )
      goto LABEL_12;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v90 )
  {
    v16 = &v90;
LABEL_12:
    v15 = sub_3007130((__int64)v16, v11);
    goto LABEL_13;
  }
  if ( (unsigned __int16)(v90 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_9:
  v15 = word_4456340[(unsigned __int16)v90 - 1];
LABEL_13:
  v86 = (int *)v97;
  v95 = (int *)v97;
  v96 = 0x500000000LL;
  if ( v15 )
  {
    v17 = 0;
    LODWORD(v85) = a4;
    v18 = 5;
    v19 = 0;
    v84 = v6;
    v20 = v7;
    v21 = v15;
    v22 = 0;
    while ( 1 )
    {
      LOWORD(v20) = 7;
      if ( v19 + 1 > v18 )
      {
        v83.m128i_i64[0] = (__int64)&v95;
        sub_C8D5F0((__int64)&v95, v86, v19 + 1, 0x10u, v17, a6);
        v19 = (unsigned int)v96;
      }
      v23 = &v95[4 * v19];
      ++v22;
      *(_QWORD *)v23 = v20;
      *((_QWORD *)v23 + 1) = 0;
      v19 = (unsigned int)(v96 + 1);
      LODWORD(v96) = v96 + 1;
      if ( v22 == v21 )
        break;
      v18 = HIDWORD(v96);
    }
    v15 = v21;
    LOBYTE(a4) = (_BYTE)v85;
    v6 = v84;
    v24 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL);
    v25 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 24LL);
    if ( v19 + 1 > (unsigned __int64)HIDWORD(v96) )
    {
      v84 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 24LL);
      LODWORD(v85) = v15;
      sub_C8D5F0((__int64)&v95, v86, v19 + 1, 0x10u, v25, v19 + 1);
      v19 = (unsigned int)v96;
      v25 = v84;
      v15 = (int)v85;
    }
  }
  else
  {
    v25 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 24LL);
    v24 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL);
    v19 = 0;
  }
  v26 = &v95[4 * v19];
  v27 = v87;
  v83.m128i_i32[0] = v15;
  *(_QWORD *)v26 = v24;
  v28 = (unsigned __int64)v95;
  *((_QWORD *)v26 + 1) = v25;
  LODWORD(v96) = v96 + 1;
  v29 = sub_33E5830(v27, v28);
  v84 = v30;
  v98 = v100;
  v85 = v100;
  v99 = 0x800000000LL;
  v31 = *(_QWORD *)(a1 + 40);
  v32 = _mm_loadu_si128((const __m128i *)v31);
  v33 = _mm_loadu_si128((const __m128i *)(v31 + 40));
  LODWORD(v99) = 2;
  v34 = *(_QWORD *)(v31 + 80);
  v100[1] = v33;
  v35 = *(_QWORD *)(v34 + 96);
  v100[0] = v32;
  if ( *(_DWORD *)(v35 + 32) <= 0x40u )
    v36 = *(_QWORD *)(v35 + 24);
  else
    v36 = **(_QWORD **)(v35 + 24);
  v37 = &v88;
  v82.m128i_i64[0] = v29;
  v38 = sub_3400BD0(v87, v36, (unsigned int)&v88, 7, 0, 1, 0);
  v39 = (unsigned int)v99;
  v41 = v40;
  v42 = v83.m128i_i32[0];
  v43 = v82.m128i_i32[0];
  v44 = (unsigned int)v99 + 1LL;
  if ( v44 > HIDWORD(v99) )
  {
    v79 = v82.m128i_i32[0];
    v82.m128i_i64[0] = v38;
    v82.m128i_i64[1] = v41;
    sub_C8D5F0((__int64)&v98, v85, v44, 0x10u, v38, v41);
    v39 = (unsigned int)v99;
    v43 = v79;
    v41 = v82.m128i_i64[1];
    v38 = v82.m128i_i64[0];
    v42 = v83.m128i_i32[0];
  }
  v45 = (__int64 *)&v98[v39];
  *v45 = v38;
  v45[1] = v41;
  v47 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 120LL));
  LODWORD(v99) = v99 + 1;
  v46 = (unsigned int)v99;
  if ( (unsigned __int64)(unsigned int)v99 + 1 > HIDWORD(v99) )
  {
    v80 = v43;
    v83.m128i_i32[0] = v42;
    v82 = v47;
    sub_C8D5F0((__int64)&v98, v85, (unsigned int)v99 + 1LL, 0x10u, v38, v41);
    v46 = (unsigned int)v99;
    v43 = v80;
    v47 = _mm_load_si128(&v82);
    v42 = v83.m128i_i32[0];
  }
  v98[v46] = v47;
  v48 = (unsigned int)(v99 + 1);
  LODWORD(v99) = v99 + 1;
  if ( (_BYTE)a4 )
  {
    v78 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 160LL));
    if ( v48 + 1 > (unsigned __int64)HIDWORD(v99) )
    {
      v81 = v43;
      v82.m128i_i32[0] = v42;
      v83 = v78;
      sub_C8D5F0((__int64)&v98, v85, v48 + 1, 0x10u, v38, v41);
      v48 = (unsigned int)v99;
      v43 = v81;
      v42 = v82.m128i_i32[0];
      v78 = _mm_load_si128(&v83);
    }
    v98[v48] = v78;
    LODWORD(v48) = v99 + 1;
    LODWORD(v99) = v99 + 1;
  }
  v49 = *(_QWORD *)(a1 + 104);
  v50 = *(unsigned __int16 *)(a1 + 96);
  v83.m128i_i32[0] = v42;
  v51 = *(_QWORD *)(a1 + 112);
  v52 = v94;
  v53 = sub_33EA9D0(v87, 47, (unsigned int)&v88, v43, v84, v51, (__int64)v98, (unsigned int)v48, v50, v49);
  v54 = v83.m128i_i32[0];
  v92 = v94;
  v55 = v53;
  v93 = 0x400000000LL;
  if ( v83.m128i_i32[0] )
  {
    v84 = a3;
    v56 = v94;
    v57 = v6;
    v83.m128i_i64[0] = (__int64)&v88;
    *((_QWORD *)&v58 + 1) = 0;
    v59 = 0;
    v60 = v54;
    v61 = 4;
    v62 = v53;
    while ( 1 )
    {
      v63 = v57 & 0xFFFFFFFF00000000LL | v59;
      v57 = v63;
      if ( *((_QWORD *)&v58 + 1) + 1LL > v61 )
      {
        v82.m128i_i64[0] = (__int64)v56;
        sub_C8D5F0((__int64)&v92, v56, *((_QWORD *)&v58 + 1) + 1LL, 0x10u, 0xFFFFFFFF00000000LL, v55);
        *((_QWORD *)&v58 + 1) = (unsigned int)v93;
        v56 = (_BYTE *)v82.m128i_i64[0];
      }
      v64 = (__int64 *)&v92[16 * *((_QWORD *)&v58 + 1)];
      ++v59;
      *v64 = v62;
      v64[1] = v63;
      *((_QWORD *)&v58 + 1) = (unsigned int)(v93 + 1);
      LODWORD(v93) = v93 + 1;
      if ( v59 == v60 )
        break;
      v61 = HIDWORD(v93);
    }
    v54 = v60;
    a3 = v84;
    LODWORD(v37) = v83.m128i_i32[0];
    v55 = v62;
    *(_QWORD *)&v58 = v92;
    v52 = v56;
  }
  else
  {
    *((_QWORD *)&v58 + 1) = 0;
    *(_QWORD *)&v58 = v94;
  }
  v83.m128i_i32[0] = v54;
  v84 = v55;
  v67 = sub_33FC220(v87, 156, (_DWORD)v37, v90, v91, v55, v58);
  v68 = *(unsigned int *)(a3 + 8);
  v69 = v65;
  v70 = v84;
  v71 = v83.m128i_i32[0];
  if ( v68 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v83.m128i_i64[0] = v67;
    v83.m128i_i64[1] = v65;
    LODWORD(v84) = v71;
    v87 = v70;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v68 + 1, 0x10u, v66, v70);
    v68 = *(unsigned int *)(a3 + 8);
    v69 = v83.m128i_i64[1];
    v67 = v83.m128i_i64[0];
    v71 = v84;
    v70 = v87;
  }
  v72 = (__int64 *)(*(_QWORD *)a3 + 16 * v68);
  v73 = v71;
  *v72 = v67;
  v72[1] = v69;
  v74 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  v75 = *(unsigned int *)(a3 + 12);
  *(_DWORD *)(a3 + 8) = v74;
  if ( v74 + 1 > v75 )
  {
    v87 = v70;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v74 + 1, 0x10u, v74 + 1, v70);
    v74 = *(unsigned int *)(a3 + 8);
    v70 = v87;
  }
  v76 = (__int64 *)(*(_QWORD *)a3 + 16 * v74);
  *v76 = v70;
  v77 = (unsigned __int64)v92;
  v76[1] = v73;
  ++*(_DWORD *)(a3 + 8);
  if ( (_BYTE *)v77 != v52 )
    _libc_free(v77);
  if ( v98 != v85 )
    _libc_free((unsigned __int64)v98);
  if ( v95 != v86 )
    _libc_free((unsigned __int64)v95);
LABEL_5:
  if ( v88 )
    sub_B91220((__int64)&v88, v88);
}
