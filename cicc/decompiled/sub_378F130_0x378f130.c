// Function: sub_378F130
// Address: 0x378f130
//
unsigned __int8 *__fastcall sub_378F130(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  unsigned int v3; // ebx
  __int64 v5; // rax
  __int64 v6; // r9
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v10; // rsi
  __int64 v11; // r8
  int v12; // eax
  __int16 v13; // ax
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // kr00_4
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int128 v20; // xmm2
  __int64 v21; // rcx
  unsigned __int16 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  _QWORD *v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  int v30; // edx
  int v31; // r14d
  __int64 v32; // rbx
  _QWORD *v33; // rdx
  _QWORD *v34; // rax
  __int64 v35; // rcx
  _QWORD *v36; // rdx
  _QWORD *v37; // rsi
  _QWORD *i; // rdx
  unsigned int v39; // r13d
  __int64 v40; // r14
  bool v41; // al
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdi
  __int64 v45; // rdx
  unsigned __int64 v46; // rax
  _QWORD *v47; // rbx
  __int128 v48; // rax
  __int64 v49; // r9
  __m128i v50; // rax
  _QWORD *v51; // rbx
  __int128 v52; // rax
  __int64 v53; // r9
  unsigned __int8 *v54; // rax
  _QWORD *v55; // rdi
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 v58; // r9
  unsigned __int8 *v59; // rax
  __int64 v60; // r8
  __int64 v61; // rdx
  __int64 v62; // rbx
  unsigned __int8 *v63; // rdx
  unsigned __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rbx
  _QWORD *v67; // rax
  __int64 v68; // r9
  __m128i v69; // rax
  __int64 v70; // rdi
  __int64 v71; // r9
  unsigned __int8 *v72; // rax
  unsigned __int64 v73; // rdx
  unsigned __int64 v74; // rcx
  unsigned __int8 *v75; // rdx
  _QWORD *v76; // r10
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // rax
  __int16 v80; // si
  __int64 v81; // rax
  unsigned __int8 *v82; // rax
  __int64 v83; // rdx
  __int64 v84; // r9
  unsigned __int8 *v85; // r14
  __int64 v87; // rdx
  unsigned __int64 v88; // rdx
  _BYTE *v89; // rbx
  __int64 v90; // rdx
  __int128 v91; // [rsp-20h] [rbp-300h]
  __int128 v92; // [rsp-10h] [rbp-2F0h]
  __int128 v93; // [rsp-10h] [rbp-2F0h]
  __int64 v94; // [rsp+10h] [rbp-2D0h]
  __int64 v95; // [rsp+18h] [rbp-2C8h]
  unsigned __int8 *v96; // [rsp+20h] [rbp-2C0h]
  __int16 v97; // [rsp+22h] [rbp-2BEh]
  unsigned __int64 v98; // [rsp+28h] [rbp-2B8h]
  __int64 v99; // [rsp+38h] [rbp-2A8h]
  __int64 v100; // [rsp+50h] [rbp-290h]
  __int64 v101; // [rsp+58h] [rbp-288h]
  int v102; // [rsp+64h] [rbp-27Ch]
  __int64 v103; // [rsp+68h] [rbp-278h]
  __int64 v104; // [rsp+70h] [rbp-270h]
  int v105; // [rsp+78h] [rbp-268h]
  __int16 v106; // [rsp+7Eh] [rbp-262h]
  _QWORD *v108; // [rsp+88h] [rbp-258h]
  _QWORD *v109; // [rsp+88h] [rbp-258h]
  __m128i v110; // [rsp+90h] [rbp-250h] BYREF
  __int64 v111; // [rsp+A0h] [rbp-240h]
  __int64 v112; // [rsp+A8h] [rbp-238h]
  __int64 v113; // [rsp+B0h] [rbp-230h]
  __int64 v114; // [rsp+B8h] [rbp-228h]
  __int64 *v115; // [rsp+C0h] [rbp-220h]
  __int64 v116; // [rsp+C8h] [rbp-218h]
  __m128i v117; // [rsp+D0h] [rbp-210h]
  __int64 v118; // [rsp+E0h] [rbp-200h]
  __int64 v119; // [rsp+E8h] [rbp-1F8h]
  unsigned __int8 *v120; // [rsp+F0h] [rbp-1F0h]
  __int64 v121; // [rsp+F8h] [rbp-1E8h]
  __int128 v122; // [rsp+100h] [rbp-1E0h] BYREF
  __int64 v123; // [rsp+110h] [rbp-1D0h] BYREF
  __int64 v124; // [rsp+118h] [rbp-1C8h]
  __int64 v125; // [rsp+120h] [rbp-1C0h] BYREF
  int v126; // [rsp+128h] [rbp-1B8h]
  unsigned __int16 v127; // [rsp+130h] [rbp-1B0h] BYREF
  __int64 v128; // [rsp+138h] [rbp-1A8h]
  __int16 v129; // [rsp+140h] [rbp-1A0h]
  __int64 v130; // [rsp+148h] [rbp-198h]
  __int64 v131; // [rsp+150h] [rbp-190h] BYREF
  __int64 v132; // [rsp+158h] [rbp-188h]
  __m128i v133; // [rsp+160h] [rbp-180h]
  unsigned __int8 *v134; // [rsp+170h] [rbp-170h]
  __int64 v135; // [rsp+178h] [rbp-168h]
  __int64 v136; // [rsp+180h] [rbp-160h]
  int v137; // [rsp+188h] [rbp-158h]
  _BYTE *v138; // [rsp+190h] [rbp-150h] BYREF
  __int64 v139; // [rsp+198h] [rbp-148h]
  _BYTE v140[128]; // [rsp+1A0h] [rbp-140h] BYREF
  _QWORD *v141; // [rsp+220h] [rbp-C0h] BYREF
  __int64 v142; // [rsp+228h] [rbp-B8h]
  _QWORD v143[22]; // [rsp+230h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a2 + 48);
  v6 = *a1;
  v7 = *(_QWORD *)(v5 + 8);
  LOWORD(v122) = *(_WORD *)v5;
  v8 = a1[1];
  *((_QWORD *)&v122 + 1) = v7;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v9 == sub_2D56A50 )
  {
    v10 = v6;
    sub_2FE6CC0((__int64)&v141, v6, *(_QWORD *)(v8 + 64), v122, *((__int64 *)&v122 + 1));
    LOWORD(v12) = v142;
    LOWORD(v123) = v142;
    v124 = v143[0];
  }
  else
  {
    v10 = *(_QWORD *)(v8 + 64);
    v12 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v9)(v6, v10, (unsigned int)v122);
    LODWORD(v123) = v12;
    v124 = v90;
  }
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 176) > 0x34u )
      goto LABEL_5;
  }
  else if ( !sub_3007100((__int64)&v123) )
  {
    goto LABEL_10;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v123 )
  {
    if ( (unsigned __int16)(v123 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_5:
    v110.m128i_i32[0] = word_4456340[(unsigned __int16)v123 - 1];
    v13 = v122;
    if ( !(_WORD)v122 )
      goto LABEL_6;
LABEL_11:
    if ( (unsigned __int16)(v13 - 176) > 0x34u )
      goto LABEL_12;
    goto LABEL_46;
  }
LABEL_10:
  v110.m128i_i32[0] = sub_3007130((__int64)&v123, v10);
  v13 = v122;
  if ( (_WORD)v122 )
    goto LABEL_11;
LABEL_6:
  if ( !sub_3007100((__int64)&v122) )
  {
LABEL_7:
    LODWORD(v111) = sub_3007130((__int64)&v122, v10);
LABEL_8:
    v16 = sub_3009970((__int64)&v122, v10, v14, v15, v11);
    HIWORD(v3) = HIWORD(v16);
    v106 = v16;
    v114 = v17;
    goto LABEL_14;
  }
LABEL_46:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v122 )
    goto LABEL_7;
  if ( (unsigned __int16)(v122 - 176) > 0x34u )
  {
    v18 = (unsigned __int16)v122 - 1;
    LODWORD(v111) = word_4456340[v18];
    goto LABEL_13;
  }
  sub_CA17B0(
    "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::ge"
    "tVectorElementCount() instead");
LABEL_12:
  v14 = (unsigned __int16)v122;
  v18 = (unsigned __int16)v122 - 1;
  v15 = word_4456340[v18];
  LODWORD(v111) = word_4456340[v18];
  if ( !(_WORD)v122 )
    goto LABEL_8;
LABEL_13:
  v114 = 0;
  v106 = word_4456580[v18];
LABEL_14:
  LOWORD(v3) = v106;
  v19 = *(_QWORD *)(a2 + 80);
  v97 = HIWORD(v3);
  v125 = v19;
  if ( v19 )
    sub_B96E90((__int64)&v125, v19, 1);
  v126 = *(_DWORD *)(a2 + 72);
  v23 = *(_QWORD *)(a2 + 40);
  v20 = (__int128)_mm_loadu_si128((const __m128i *)(v23 + 80));
  v100 = *(_QWORD *)v23;
  v101 = *(_QWORD *)(v23 + 120);
  v105 = *(_DWORD *)(v23 + 8);
  v21 = *(_QWORD *)(v23 + 40);
  v102 = *(_DWORD *)(v23 + 128);
  v103 = v21;
  v99 = *(unsigned int *)(v23 + 48);
  v22 = (unsigned __int16 *)(*(_QWORD *)(v21 + 48) + 16 * v99);
  v117 = _mm_loadu_si128((const __m128i *)(v23 + 40));
  LODWORD(v23) = *v22;
  v24 = *((_QWORD *)v22 + 1);
  LOWORD(v141) = v23;
  v142 = v24;
  if ( (_WORD)v23 )
  {
    v112 = 0;
    LOWORD(v23) = word_4456580[(int)v23 - 1];
  }
  else
  {
    v23 = sub_3009970((__int64)&v141, v19, v24, v21, v11);
    v113 = v23;
    v112 = v87;
  }
  v25 = v113;
  v26 = (_QWORD *)a1[1];
  v141 = 0;
  LODWORD(v142) = 0;
  LOWORD(v25) = v23;
  v113 = v25;
  v27 = sub_33F17F0(v26, 51, (__int64)&v141, v3, v114);
  v31 = v30;
  if ( v141 )
  {
    v108 = v27;
    sub_B91220((__int64)&v141, (__int64)v141);
    v27 = v108;
  }
  v32 = v110.m128i_u32[0];
  v138 = v140;
  v139 = 0x800000000LL;
  if ( v110.m128i_i32[0] > 8u )
  {
    v109 = v27;
    sub_C8D5F0((__int64)&v138, v140, v110.m128i_u32[0], 0x10u, v28, v29);
    v88 = (unsigned __int64)v138;
    v89 = &v138[16 * v110.m128i_u32[0]];
    do
    {
      if ( v88 )
      {
        *(_QWORD *)v88 = v109;
        *(_DWORD *)(v88 + 8) = v31;
      }
      v88 += 16LL;
    }
    while ( (_BYTE *)v88 != v89 );
  }
  else if ( v110.m128i_i32[0] )
  {
    v33 = v140;
    do
    {
      *v33 = v27;
      v33 += 2;
      *((_DWORD *)v33 - 2) = v31;
      --v32;
    }
    while ( v32 );
  }
  v142 = 0x800000000LL;
  LODWORD(v139) = v110.m128i_i32[0];
  v34 = v143;
  v35 = (unsigned int)v111;
  v104 = (unsigned int)v111;
  v36 = v143;
  v37 = v143;
  v141 = v143;
  if ( (_DWORD)v111 )
  {
    if ( (unsigned int)v111 > 8uLL )
    {
      sub_C8D5F0((__int64)&v141, v143, (unsigned int)v111, 0x10u, v28, v29);
      v36 = v141;
      v34 = &v141[2 * (unsigned int)v142];
    }
    for ( i = &v36[2 * (unsigned int)v111]; i != v34; v34 += 2 )
    {
      if ( v34 )
      {
        *v34 = 0;
        *((_DWORD *)v34 + 2) = 0;
      }
    }
    HIWORD(v39) = v97;
    v40 = 0;
    LODWORD(v142) = v111;
    do
    {
      v47 = (_QWORD *)a1[1];
      *(_QWORD *)&v48 = sub_3400EE0((__int64)v47, v40, (__int64)&v125, 0, a3);
      v117.m128i_i64[0] = v103;
      v117.m128i_i64[1] = v99 | v117.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v50.m128i_i64[0] = (__int64)sub_3406EB0(
                                    v47,
                                    0x9Eu,
                                    (__int64)&v125,
                                    (unsigned int)v113,
                                    v112,
                                    v49,
                                    __PAIR128__(v117.m128i_u64[1], v103),
                                    v48);
      v51 = (_QWORD *)a1[1];
      v110 = v50;
      *(_QWORD *)&v52 = sub_3400EE0((__int64)v51, v40, (__int64)&v125, 0, a3);
      v54 = sub_3406EB0(v51, 0x9Eu, (__int64)&v125, (unsigned int)v113, v112, v53, v20, v52);
      v55 = (_QWORD *)a1[1];
      v134 = v54;
      v131 = v100;
      v136 = v101;
      a3 = _mm_load_si128(&v110);
      LODWORD(v132) = v105;
      v137 = v102;
      v135 = v56;
      v115 = &v131;
      v127 = 2;
      v116 = 4;
      v57 = *(_DWORD *)(a2 + 24);
      *((_QWORD *)&v92 + 1) = 4;
      *(_QWORD *)&v92 = &v131;
      v129 = 1;
      v133 = a3;
      v128 = 0;
      v130 = 0;
      v59 = sub_3411BE0(v55, v57, (__int64)&v125, &v127, 2, v58, v92);
      LOWORD(v39) = v106;
      v60 = v114;
      v62 = v61;
      v63 = v59;
      v64 = (unsigned __int64)v138;
      v65 = v62;
      v120 = v63;
      v66 = 16 * v40;
      v121 = v65;
      *(_QWORD *)&v138[v66] = v63;
      *(_DWORD *)(v64 + v66 + 8) = v121;
      v67 = &v141[2 * v40];
      *v67 = *(_QWORD *)&v138[16 * v40];
      *((_DWORD *)v67 + 2) = 1;
      v111 = a1[1];
      v69.m128i_i64[0] = (__int64)sub_3401740(v111, 0, (__int64)&v125, v39, v60, v68, v122);
      v70 = a1[1];
      v110 = v69;
      v72 = sub_3401740(v70, 1, (__int64)&v125, v39, v114, v71, v122);
      v74 = v73;
      v75 = v72;
      v76 = (_QWORD *)v111;
      v77 = *(_QWORD *)&v138[16 * v40];
      v78 = *(_QWORD *)&v138[16 * v40 + 8];
      v79 = *(_QWORD *)(*(_QWORD *)&v138[v66] + 48LL) + 16LL * *(unsigned int *)&v138[v66 + 8];
      v80 = *(_WORD *)v79;
      v81 = *(_QWORD *)(v79 + 8);
      LOWORD(v131) = v80;
      v132 = v81;
      if ( v80 )
      {
        v41 = (unsigned __int16)(v80 - 17) <= 0xD3u;
      }
      else
      {
        v94 = v77;
        v95 = v78;
        v96 = v75;
        v98 = v74;
        v41 = sub_30070B0((__int64)&v131);
        v77 = v94;
        v78 = v95;
        v75 = v96;
        v74 = v98;
        v76 = (_QWORD *)v111;
      }
      ++v40;
      v42 = sub_340EC60(
              v76,
              205 - ((unsigned int)!v41 - 1),
              (__int64)&v125,
              v39,
              v114,
              0,
              v77,
              v78,
              __PAIR128__(v74, (unsigned __int64)v75),
              *(_OWORD *)&v110);
      v44 = v43;
      v45 = v42;
      v46 = (unsigned __int64)v138;
      v118 = v45;
      v119 = v44;
      *(_QWORD *)&v138[v66] = v45;
      *(_DWORD *)(v46 + v66 + 8) = v119;
    }
    while ( v104 != v40 );
    v37 = v141;
    v35 = (unsigned int)v142;
  }
  *((_QWORD *)&v93 + 1) = v35;
  *(_QWORD *)&v93 = v37;
  v82 = sub_33FC220((_QWORD *)a1[1], 2, (__int64)&v125, 1, 0, v29, v93);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v82, v83);
  *((_QWORD *)&v91 + 1) = (unsigned int)v139;
  *(_QWORD *)&v91 = v138;
  v85 = sub_33FC220((_QWORD *)a1[1], 156, (__int64)&v125, v123, v124, v84, v91);
  if ( v141 != v143 )
    _libc_free((unsigned __int64)v141);
  if ( v138 != v140 )
    _libc_free((unsigned __int64)v138);
  if ( v125 )
    sub_B91220((__int64)&v125, v125);
  return v85;
}
