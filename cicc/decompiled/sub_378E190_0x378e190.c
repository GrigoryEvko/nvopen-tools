// Function: sub_378E190
// Address: 0x378e190
//
unsigned __int8 *__fastcall sub_378E190(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 v10; // rcx
  unsigned int v11; // r14d
  unsigned __int64 v12; // rax
  int v13; // ecx
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  __m128i *v16; // rax
  __int64 v17; // r15
  __int64 v18; // r9
  __int64 (__fastcall *v19)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v20; // rax
  unsigned __int16 v21; // si
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rsi
  _QWORD **v26; // rcx
  __int64 v27; // r8
  int v28; // eax
  unsigned int v29; // r13d
  unsigned __int16 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  _WORD *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  unsigned __int16 v36; // ax
  _QWORD *v37; // rdi
  _QWORD *v38; // rax
  __int64 v39; // r8
  __int64 v40; // r9
  int v41; // edx
  int v42; // r15d
  __int64 v43; // r14
  _QWORD *v44; // rdx
  unsigned __int16 *v45; // rax
  unsigned __int16 v46; // r13
  __int64 v47; // rax
  unsigned int v48; // eax
  unsigned int v49; // eax
  unsigned __int8 *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r9
  unsigned __int8 *v53; // r14
  __int64 v55; // r14
  __int128 v56; // rax
  __int64 v57; // r9
  unsigned __int8 *v58; // rax
  int v59; // edx
  int v60; // edi
  unsigned __int8 *v61; // rdx
  unsigned __int64 v62; // rax
  __int64 v63; // r9
  unsigned __int8 *v64; // rcx
  __int64 v65; // r9
  int v66; // edx
  int v67; // edi
  unsigned __int64 v68; // rdx
  __int64 v69; // rax
  unsigned __int64 v70; // r8
  __int64 v71; // r10
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  _QWORD *v74; // rax
  __int64 v75; // rdx
  unsigned __int64 v76; // rdx
  _BYTE *v77; // r14
  __int64 v78; // rdx
  __int64 v79; // rdx
  __int128 v80; // [rsp-20h] [rbp-4D0h]
  __int128 v81; // [rsp-20h] [rbp-4D0h]
  __int128 v82; // [rsp-10h] [rbp-4C0h]
  __int128 v83; // [rsp-10h] [rbp-4C0h]
  __int64 v84; // [rsp+0h] [rbp-4B0h]
  __int64 v86; // [rsp+30h] [rbp-480h]
  __int64 v87; // [rsp+38h] [rbp-478h]
  __int64 v88; // [rsp+40h] [rbp-470h]
  unsigned int v89; // [rsp+4Ch] [rbp-464h]
  __int64 v90; // [rsp+50h] [rbp-460h]
  __int64 v91; // [rsp+58h] [rbp-458h]
  unsigned int v92; // [rsp+58h] [rbp-458h]
  _QWORD *v93; // [rsp+60h] [rbp-450h]
  _QWORD *v94; // [rsp+60h] [rbp-450h]
  unsigned __int64 v95; // [rsp+60h] [rbp-450h]
  int v96; // [rsp+60h] [rbp-450h]
  _QWORD *v97; // [rsp+60h] [rbp-450h]
  unsigned __int64 v98; // [rsp+68h] [rbp-448h]
  unsigned __int64 v99; // [rsp+88h] [rbp-428h]
  __int64 v100; // [rsp+B0h] [rbp-400h] BYREF
  int v101; // [rsp+B8h] [rbp-3F8h]
  __int64 v102; // [rsp+C0h] [rbp-3F0h] BYREF
  __int64 v103; // [rsp+C8h] [rbp-3E8h]
  __int16 v104; // [rsp+D0h] [rbp-3E0h] BYREF
  __int64 v105; // [rsp+D8h] [rbp-3D8h]
  unsigned __int16 v106; // [rsp+E0h] [rbp-3D0h] BYREF
  __int64 v107; // [rsp+E8h] [rbp-3C8h]
  unsigned __int16 v108; // [rsp+F0h] [rbp-3C0h] BYREF
  __int64 v109; // [rsp+F8h] [rbp-3B8h]
  __int128 v110; // [rsp+100h] [rbp-3B0h]
  _BYTE *v111; // [rsp+110h] [rbp-3A0h] BYREF
  __int64 v112; // [rsp+118h] [rbp-398h]
  _BYTE v113[64]; // [rsp+120h] [rbp-390h] BYREF
  _BYTE *v114; // [rsp+160h] [rbp-350h] BYREF
  __int64 v115; // [rsp+168h] [rbp-348h]
  _BYTE v116[256]; // [rsp+170h] [rbp-340h] BYREF
  _QWORD *v117; // [rsp+270h] [rbp-240h] BYREF
  __int64 v118; // [rsp+278h] [rbp-238h]
  _QWORD v119[70]; // [rsp+280h] [rbp-230h] BYREF

  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *(_QWORD *)(v8 + 40);
  v100 = v9;
  v11 = *(_DWORD *)(v8 + 48);
  v90 = v10;
  v99 = _mm_loadu_si128((const __m128i *)(v8 + 40)).m128i_u64[1];
  if ( v9 )
  {
    sub_B96E90((__int64)&v100, v9, 1);
    v8 = *(_QWORD *)(a2 + 40);
  }
  v101 = *(_DWORD *)(a2 + 72);
  v12 = *(unsigned int *)(a2 + 64);
  v112 = 0x400000000LL;
  v13 = 0;
  v14 = 40 * v12;
  v15 = v12;
  v16 = (__m128i *)v113;
  v17 = v8 + v14;
  v111 = v113;
  if ( v14 > 0xA0 )
  {
    v96 = v15;
    sub_C8D5F0((__int64)&v111, v113, v15, 0x10u, a5, a6);
    v13 = v112;
    LODWORD(v15) = v96;
    v16 = (__m128i *)&v111[16 * (unsigned int)v112];
  }
  if ( v17 != v8 )
  {
    do
    {
      if ( v16 )
        *v16 = _mm_loadu_si128((const __m128i *)v8);
      v8 += 40;
      ++v16;
    }
    while ( v17 != v8 );
    v13 = v112;
  }
  v18 = *a1;
  LODWORD(v112) = v13 + v15;
  v19 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v18 + 592LL);
  v20 = *(__int16 **)(a2 + 48);
  v21 = *v20;
  v22 = *((_QWORD *)v20 + 1);
  v23 = a1[1];
  if ( v19 == sub_2D56A50 )
  {
    v24 = v21;
    v25 = v18;
    sub_2FE6CC0((__int64)&v117, v18, *(_QWORD *)(v23 + 64), v24, v22);
    LOWORD(v28) = v118;
    LOWORD(v102) = v118;
    v103 = v119[0];
  }
  else
  {
    v78 = v21;
    v25 = *(_QWORD *)(v23 + 64);
    v28 = v19(v18, v25, v78, v22);
    v26 = &v117;
    LODWORD(v102) = v28;
    v103 = v79;
  }
  if ( (_WORD)v28 )
  {
    if ( (unsigned __int16)(v28 - 176) > 0x34u )
    {
LABEL_14:
      v29 = word_4456340[(unsigned __int16)v102 - 1];
      goto LABEL_17;
    }
  }
  else if ( !sub_3007100((__int64)&v102) )
  {
    goto LABEL_16;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v102 )
  {
    if ( (unsigned __int16)(v102 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_14;
  }
LABEL_16:
  v29 = sub_3007130((__int64)&v102, v25);
LABEL_17:
  v88 = v11;
  v30 = (unsigned __int16 *)(*(_QWORD *)(v90 + 48) + 16LL * v11);
  LODWORD(v31) = *v30;
  v32 = *((_QWORD *)v30 + 1);
  v104 = v31;
  v105 = v32;
  if ( (_WORD)v31 )
  {
    v33 = word_4456580;
    v87 = 0;
    LOWORD(v31) = word_4456580[(int)v31 - 1];
  }
  else
  {
    v31 = sub_3009970((__int64)&v104, v25, v32, (__int64)v26, v27);
    v91 = v31;
    v87 = (__int64)v33;
  }
  v34 = v91;
  LOWORD(v34) = v31;
  v92 = v34;
  v89 = *(_DWORD *)(a2 + 24);
  if ( (_WORD)v102 )
  {
    v35 = 0;
    v36 = word_4456580[(unsigned __int16)v102 - 1];
  }
  else
  {
    v36 = sub_3009970((__int64)&v102, v25, (__int64)v33, v34, v27);
    v35 = v75;
  }
  v37 = (_QWORD *)a1[1];
  v110 = 0;
  v108 = v36;
  LOWORD(v110) = 1;
  v109 = v35;
  v117 = 0;
  LODWORD(v118) = 0;
  v38 = sub_33F17F0(v37, 51, (__int64)&v117, v36, v35);
  v42 = v41;
  if ( v117 )
  {
    v93 = v38;
    sub_B91220((__int64)&v117, (__int64)v117);
    v38 = v93;
  }
  v43 = v29;
  v114 = v116;
  v115 = 0x1000000000LL;
  if ( v29 > 0x10 )
  {
    v97 = v38;
    sub_C8D5F0((__int64)&v114, v116, v29, 0x10u, v39, v40);
    v76 = (unsigned __int64)v114;
    v77 = &v114[16 * v29];
    do
    {
      if ( v76 )
      {
        *(_QWORD *)v76 = v97;
        *(_DWORD *)(v76 + 8) = v42;
      }
      v76 += 16LL;
    }
    while ( (_BYTE *)v76 != v77 );
  }
  else if ( v29 )
  {
    v44 = v116;
    do
    {
      *v44 = v38;
      v44 += 2;
      *((_DWORD *)v44 - 2) = v42;
      --v43;
    }
    while ( v43 );
  }
  LODWORD(v115) = v29;
  v117 = v119;
  v118 = 0x2000000000LL;
  v45 = *(unsigned __int16 **)(a2 + 48);
  v46 = *v45;
  v47 = *((_QWORD *)v45 + 1);
  v106 = v46;
  v107 = v47;
  if ( v46 )
  {
    if ( (unsigned __int16)(v46 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v48 = word_4456340[v46 - 1];
    if ( !word_4456340[v46 - 1] )
    {
LABEL_31:
      v49 = v118;
      goto LABEL_32;
    }
  }
  else
  {
    if ( sub_3007100((__int64)&v106) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v48 = sub_3007130((__int64)&v106, (__int64)v116);
    if ( !v48 )
      goto LABEL_31;
  }
  v55 = 0;
  v86 = v48;
  do
  {
    v94 = (_QWORD *)a1[1];
    *(_QWORD *)&v56 = sub_3400EE0((__int64)v94, v55, (__int64)&v100, 0, (__m128i)0LL);
    v99 = v88 | v99 & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v81 + 1) = v99;
    *(_QWORD *)&v81 = v90;
    v58 = sub_3406EB0(v94, 0x9Eu, (__int64)&v100, v92, v87, v57, v81, v56);
    v60 = v59;
    v61 = v58;
    v62 = (unsigned __int64)v111;
    *((_QWORD *)v111 + 2) = v61;
    *(_DWORD *)(v62 + 24) = v60;
    *((_QWORD *)&v83 + 1) = (unsigned int)v112;
    *(_QWORD *)&v83 = v62;
    v64 = sub_3411BE0((_QWORD *)a1[1], v89, (__int64)&v100, &v108, 2, v63, v83);
    v67 = v66;
    v68 = (unsigned __int64)v114;
    v69 = 16 * v55;
    v70 = v98 & 0xFFFFFFFF00000000LL | 1;
    *(_QWORD *)&v114[v69] = v64;
    v98 = v70;
    *(_DWORD *)(v68 + v69 + 8) = v67;
    v71 = *(_QWORD *)&v114[16 * v55];
    v72 = (unsigned int)v118;
    v73 = (unsigned int)v118 + 1LL;
    if ( v73 > HIDWORD(v118) )
    {
      v84 = *(_QWORD *)&v114[16 * v55];
      v95 = v70;
      sub_C8D5F0((__int64)&v117, v119, v73, 0x10u, v70, v65);
      v72 = (unsigned int)v118;
      v71 = v84;
      v70 = v95;
    }
    v74 = &v117[2 * v72];
    ++v55;
    *v74 = v71;
    v74[1] = v70;
    v49 = v118 + 1;
    LODWORD(v118) = v118 + 1;
  }
  while ( v86 != v55 );
LABEL_32:
  *((_QWORD *)&v82 + 1) = v49;
  *(_QWORD *)&v82 = v117;
  v50 = sub_33FC220((_QWORD *)a1[1], 2, (__int64)&v100, 1, 0, a1[1], v82);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v50, v51);
  *((_QWORD *)&v80 + 1) = (unsigned int)v115;
  *(_QWORD *)&v80 = v114;
  v53 = sub_33FC220((_QWORD *)a1[1], 156, (__int64)&v100, v102, v103, v52, v80);
  if ( v117 != v119 )
    _libc_free((unsigned __int64)v117);
  if ( v114 != v116 )
    _libc_free((unsigned __int64)v114);
  if ( v111 != v113 )
    _libc_free((unsigned __int64)v111);
  if ( v100 )
    sub_B91220((__int64)&v100, v100);
  return v53;
}
