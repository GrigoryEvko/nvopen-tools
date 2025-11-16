// Function: sub_37AF6D0
// Address: 0x37af6d0
//
__int64 __fastcall sub_37AF6D0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // ebx
  const __m128i *v4; // rax
  __int64 v5; // rsi
  __m128i v6; // xmm0
  __int64 v7; // rdi
  __int64 v8; // r9
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v10; // rax
  unsigned __int16 v11; // si
  int v12; // eax
  unsigned __int16 *v13; // rdx
  unsigned __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // rcx
  unsigned __int16 *v17; // r13
  int v18; // eax
  __int64 v19; // r8
  unsigned int v20; // r14d
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int16 v23; // r13
  unsigned int v24; // eax
  unsigned __int16 v25; // dx
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  unsigned int v31; // eax
  _QWORD *v32; // rdi
  unsigned int v33; // r13d
  _QWORD *v34; // rax
  __int64 v35; // r9
  unsigned int v36; // edx
  unsigned int v37; // ebx
  __int64 v38; // r8
  __int64 v39; // rbx
  _QWORD *v40; // rdx
  __int16 *v41; // rax
  unsigned __int16 v42; // bx
  __int64 v43; // rax
  unsigned int v44; // eax
  __int64 v45; // rbx
  unsigned __int8 *v46; // rax
  int v47; // edx
  int v48; // edi
  unsigned __int8 *v49; // rdx
  __int64 v50; // rax
  __int128 v51; // rax
  __int128 v52; // rax
  _QWORD *v53; // rdi
  unsigned __int8 *v54; // rax
  int v55; // edx
  int v56; // edi
  unsigned __int8 *v57; // rdx
  __int64 v58; // rax
  unsigned __int8 *v59; // rax
  unsigned __int64 v60; // rdi
  __int64 v61; // r14
  int v63; // eax
  __int64 v64; // rdx
  int v65; // eax
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rax
  unsigned int v69; // edx
  __int64 v70; // rdx
  unsigned __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rdx
  int v74; // r9d
  __int64 v75; // rdx
  int v76; // edx
  _QWORD *v77; // r14
  __int128 v78; // rax
  __int64 v79; // r9
  __int128 v80; // rax
  _QWORD *v81; // rdi
  unsigned __int8 *v82; // rax
  __int64 v83; // r10
  __int64 (__fastcall *v84)(__int64, __int64, unsigned int, __int64); // rax
  char v85; // r9
  __int64 v86; // r13
  __int64 v87; // rax
  unsigned int v88; // edx
  __int64 v89; // rax
  __int64 v90; // rax
  unsigned __int64 v91; // r13
  unsigned __int64 v92; // rax
  bool v93; // cf
  __int64 v94; // rdx
  _QWORD *v95; // rbx
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // r8
  __int64 v99; // r9
  __int64 v100; // rax
  __int64 v101; // r9
  __int128 v102; // rax
  _QWORD *v103; // rdi
  int v104; // eax
  unsigned __int8 *v105; // rax
  int v106; // eax
  __int64 v107; // rdx
  __int64 v108; // rax
  __int64 v109; // rbx
  unsigned int v110; // edx
  __int64 v111; // r9
  __int128 v112; // [rsp-20h] [rbp-260h]
  __int128 v113; // [rsp-10h] [rbp-250h]
  __int128 v114; // [rsp-10h] [rbp-250h]
  int v115; // [rsp+0h] [rbp-240h]
  _QWORD *v116; // [rsp+0h] [rbp-240h]
  __int64 v117; // [rsp+8h] [rbp-238h]
  char v118; // [rsp+8h] [rbp-238h]
  int v119; // [rsp+8h] [rbp-238h]
  __int64 *v120; // [rsp+10h] [rbp-230h]
  _QWORD *v121; // [rsp+10h] [rbp-230h]
  unsigned int v122; // [rsp+18h] [rbp-228h]
  __int64 v123; // [rsp+20h] [rbp-220h]
  int v124; // [rsp+2Ch] [rbp-214h]
  __int64 v125; // [rsp+30h] [rbp-210h]
  int v126; // [rsp+38h] [rbp-208h]
  unsigned int v127; // [rsp+38h] [rbp-208h]
  __int64 v128; // [rsp+38h] [rbp-208h]
  unsigned int v129; // [rsp+40h] [rbp-200h]
  char v130; // [rsp+44h] [rbp-1FCh]
  __int16 v131; // [rsp+44h] [rbp-1FCh]
  __int16 v132; // [rsp+46h] [rbp-1FAh]
  char v133; // [rsp+46h] [rbp-1FAh]
  char v134; // [rsp+46h] [rbp-1FAh]
  __int64 v135; // [rsp+48h] [rbp-1F8h]
  unsigned int v136; // [rsp+50h] [rbp-1F0h]
  _QWORD *v137; // [rsp+50h] [rbp-1F0h]
  __int64 v138; // [rsp+50h] [rbp-1F0h]
  __int128 v140; // [rsp+60h] [rbp-1E0h]
  __int64 v141; // [rsp+90h] [rbp-1B0h]
  unsigned __int64 v142; // [rsp+A8h] [rbp-198h]
  __int64 v143; // [rsp+C0h] [rbp-180h] BYREF
  int v144; // [rsp+C8h] [rbp-178h]
  __int64 v145; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v146; // [rsp+D8h] [rbp-168h]
  __int64 v147; // [rsp+E0h] [rbp-160h] BYREF
  __int64 v148; // [rsp+E8h] [rbp-158h]
  __int64 v149; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v150; // [rsp+F8h] [rbp-148h]
  _QWORD *v151; // [rsp+100h] [rbp-140h] BYREF
  __int64 v152; // [rsp+108h] [rbp-138h]
  _QWORD v153[38]; // [rsp+110h] [rbp-130h] BYREF

  v120 = *(__int64 **)(a1[1] + 64);
  v4 = *(const __m128i **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = _mm_loadu_si128(v4);
  v7 = v4->m128i_i64[0];
  LODWORD(v4) = v4->m128i_i32[2];
  v143 = v5;
  v135 = v7;
  v122 = (unsigned int)v4;
  *((_QWORD *)&v140 + 1) = v6.m128i_i64[1];
  if ( v5 )
    sub_B96E90((__int64)&v143, v5, 1);
  v8 = *a1;
  v144 = *(_DWORD *)(a2 + 72);
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v8 + 592LL);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  if ( v9 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v151, v8, (__int64)v120, v11, *((_QWORD *)v10 + 1));
    LOWORD(v12) = v152;
    LOWORD(v145) = v152;
    v146 = v153[0];
  }
  else
  {
    v12 = v9(v8, (__int64)v120, v11, *((_QWORD *)v10 + 1));
    LODWORD(v145) = v12;
    v146 = v75;
  }
  if ( (_WORD)v12 )
  {
    v13 = word_4456340;
    v130 = (unsigned __int16)(v12 - 176) <= 0x34u;
    LOBYTE(v14) = v130;
    v136 = word_4456340[(unsigned __int16)v12 - 1];
  }
  else
  {
    v142 = sub_3007240((__int64)&v145);
    v14 = HIDWORD(v142);
    v136 = v142;
    v130 = BYTE4(v142);
  }
  v15 = *(unsigned int *)(a2 + 24);
  v16 = *(unsigned int *)(a2 + 28);
  v17 = (unsigned __int16 *)(*(_QWORD *)(v7 + 48) + 16LL * v122);
  v129 = v15;
  v18 = *v17;
  v19 = *((_QWORD *)v17 + 1);
  v124 = *(_DWORD *)(a2 + 28);
  LOWORD(v147) = *v17;
  v148 = v19;
  if ( (_DWORD)v15 == 214 )
  {
    v15 = *a1;
    v133 = v14;
    sub_2FE6CC0((__int64)&v151, *a1, *(_QWORD *)(a1[1] + 64), (unsigned __int16)v18, v19);
    LOBYTE(v14) = v133;
    if ( (_BYTE)v151 != 1 )
      goto LABEL_61;
    v83 = *a1;
    v84 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
    if ( v84 == sub_2D56A50 )
    {
      v15 = *a1;
      sub_2FE6CC0((__int64)&v151, v83, (__int64)v120, v147, v148);
      v85 = v133;
      LOWORD(v151) = v152;
      v152 = v153[0];
    }
    else
    {
      v15 = (__int64)v120;
      v106 = v84(v83, (__int64)v120, v147, v148);
      v85 = v133;
      LODWORD(v151) = v106;
      v152 = v107;
    }
    v134 = v85;
    v86 = sub_32844A0((unsigned __int16 *)&v151, v15);
    v87 = sub_32844A0((unsigned __int16 *)&v145, v15);
    LOBYTE(v14) = v134;
    if ( v86 == v87 )
    {
LABEL_61:
      v18 = (unsigned __int16)v147;
    }
    else
    {
      v15 = (__int64)sub_37AF270((__int64)a1, v6.m128i_u64[0], v6.m128i_i64[1], v6);
      v135 = v15;
      v122 = v88;
      v89 = *(_QWORD *)(v15 + 48) + 16LL * v88;
      *((_QWORD *)&v140 + 1) = v88 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      LOWORD(v88) = *(_WORD *)v89;
      v90 = *(_QWORD *)(v89 + 8);
      LOWORD(v147) = v88;
      v148 = v90;
      v91 = sub_32844A0((unsigned __int16 *)&v145, v15);
      v92 = sub_32844A0((unsigned __int16 *)&v147, v15);
      LOBYTE(v14) = v134;
      v93 = v91 < v92;
      v18 = (unsigned __int16)v147;
      if ( v93 )
        v129 = 216;
      else
        v129 = 214;
    }
  }
  if ( (_WORD)v18 )
  {
    v125 = 0;
    v132 = word_4456580[v18 - 1];
  }
  else
  {
    v118 = v14;
    v65 = sub_3009970((__int64)&v147, v15, (__int64)v13, v16, v19);
    LOBYTE(v14) = v118;
    HIWORD(v2) = HIWORD(v65);
    v132 = v65;
    v125 = v66;
  }
  LOWORD(v2) = v132;
  BYTE4(v151) = v14;
  LODWORD(v151) = v136;
  HIWORD(v20) = HIWORD(v2);
  if ( v130 )
    v22 = (unsigned int)sub_2D43AD0(v132, v136);
  else
    v22 = (unsigned int)sub_2D43050(v132, v136);
  v117 = 0;
  v23 = v22;
  if ( !(_WORD)v22 )
  {
    v126 = sub_3009450(v120, v2, v125, (__int64)v151, v21, v22);
    v23 = v126;
    v117 = v64;
  }
  HIWORD(v24) = HIWORD(v126);
  v25 = v147;
  LOWORD(v24) = v23;
  v127 = v24;
  if ( (_WORD)v147 )
  {
    LODWORD(v26) = word_4456340[(unsigned __int16)v147 - 1];
  }
  else
  {
    v63 = sub_3007240((__int64)&v147);
    v25 = 0;
    LODWORD(v26) = v63;
  }
  v27 = *a1;
  sub_2FE6CC0((__int64)&v151, *a1, *(_QWORD *)(a1[1] + 64), v25, v148);
  if ( (_BYTE)v151 == 7 )
  {
    v68 = *(_QWORD *)(a2 + 40);
    v27 = *(_QWORD *)v68;
    v135 = sub_379AB60((__int64)a1, *(_QWORD *)v68, *(_QWORD *)(v68 + 8));
    *(_QWORD *)&v140 = v135;
    v122 = v69;
    v71 = v69 | *((_QWORD *)&v140 + 1) & 0xFFFFFFFF00000000LL;
    v70 = *(_QWORD *)(v135 + 48) + 16LL * v69;
    *((_QWORD *)&v140 + 1) = v71;
    LOWORD(v71) = *(_WORD *)v70;
    v72 = *(_QWORD *)(v70 + 8);
    LOWORD(v147) = v71;
    v148 = v72;
    if ( (_WORD)v71 )
    {
      v76 = (unsigned __int16)v71;
      LOBYTE(v71) = (unsigned __int16)(v71 - 176) <= 0x34u;
      LODWORD(v26) = word_4456340[v76 - 1];
    }
    else
    {
      v26 = sub_3007240((__int64)&v147);
      v71 = HIDWORD(v26);
    }
    if ( v136 == (_DWORD)v26 && (_BYTE)v71 == v130 )
    {
      v104 = *(_DWORD *)(a2 + 64);
      if ( v104 != 1 )
      {
        if ( v104 == 3 )
        {
          v151 = (_QWORD *)sub_3281590((__int64)&v145);
          v108 = *(_QWORD *)(a2 + 40);
          v109 = *(_QWORD *)(v108 + 48);
          v141 = sub_379AB60((__int64)a1, *(_QWORD *)(v108 + 40), v109);
          *((_QWORD *)&v112 + 1) = v110 | v109 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v112 = v141;
          v61 = sub_340F900(
                  (_QWORD *)a1[1],
                  v129,
                  (__int64)&v143,
                  v145,
                  v146,
                  v111,
                  v140,
                  v112,
                  *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
          goto LABEL_40;
        }
        v82 = sub_3405C90(
                (_QWORD *)a1[1],
                v129,
                (__int64)&v143,
                (unsigned int)v145,
                v146,
                v124,
                v6,
                v140,
                *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
LABEL_66:
        v61 = (__int64)v82;
        goto LABEL_40;
      }
      v105 = sub_33FA050(
               a1[1],
               v129,
               (__int64)&v143,
               (unsigned int)v145,
               v146,
               v124,
               (unsigned __int8 *)v135,
               *((__int64 *)&v140 + 1));
LABEL_86:
      v61 = (__int64)v105;
      goto LABEL_40;
    }
    v151 = (_QWORD *)sub_2D5B750((unsigned __int16 *)&v147);
    v152 = v73;
    v149 = sub_2D5B750((unsigned __int16 *)&v145);
    v150 = v28;
    if ( (_QWORD *)v149 == v151 && (_BYTE)v150 == (_BYTE)v152 )
    {
      switch ( v129 )
      {
        case 0xD7u:
          v61 = (__int64)sub_33FAF80(a1[1], 223, (__int64)&v143, (unsigned int)v145, v146, v74, v6);
          goto LABEL_40;
        case 0xD5u:
          v61 = (__int64)sub_33FAF80(a1[1], 224, (__int64)&v143, (unsigned int)v145, v146, v74, v6);
          goto LABEL_40;
        case 0xD6u:
          v61 = (__int64)sub_33FAF80(a1[1], 225, (__int64)&v143, (unsigned int)v145, v146, v74, v6);
          goto LABEL_40;
      }
    }
  }
  if ( v23 && *(_QWORD *)(*a1 + 8LL * v23 + 112) )
  {
    if ( !(v136 % (unsigned int)v26) )
    {
      v96 = sub_3288990(a1[1], (unsigned int)v147, v148);
      v151 = v153;
      v152 = 0x1000000000LL;
      sub_32982C0((__int64)&v151, v136 / (unsigned int)v26, v96, v97, v98, v99);
      v100 = (__int64)v151;
      *v151 = v135;
      *(_DWORD *)(v100 + 8) = v122;
      *((_QWORD *)&v114 + 1) = (unsigned int)v152;
      *(_QWORD *)&v114 = v151;
      *(_QWORD *)&v102 = sub_33FC220((_QWORD *)a1[1], 159, (__int64)&v143, v127, v117, v101, v114);
      v103 = (_QWORD *)a1[1];
      if ( *(_DWORD *)(a2 + 64) == 1 )
        v61 = (__int64)sub_33FA050(
                         (__int64)v103,
                         v129,
                         (__int64)&v143,
                         (unsigned int)v145,
                         v146,
                         v124,
                         (unsigned __int8 *)v102,
                         *((__int64 *)&v102 + 1));
      else
        v61 = (__int64)sub_3405C90(
                         v103,
                         v129,
                         (__int64)&v143,
                         (unsigned int)v145,
                         v146,
                         v124,
                         v6,
                         v102,
                         *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
      v60 = (unsigned __int64)v151;
      if ( v151 == v153 )
        goto LABEL_40;
      goto LABEL_39;
    }
    v28 = (unsigned int)v26 % v136;
    if ( !((unsigned int)v26 % v136) )
    {
      v77 = (_QWORD *)a1[1];
      *(_QWORD *)&v78 = sub_3400EE0((__int64)v77, 0, (__int64)&v143, 0, v6);
      *(_QWORD *)&v80 = sub_3406EB0(
                          v77,
                          0xA1u,
                          (__int64)&v143,
                          v127,
                          v117,
                          v79,
                          __PAIR128__(v122 | *((_QWORD *)&v140 + 1) & 0xFFFFFFFF00000000LL, v135),
                          v78);
      v81 = (_QWORD *)a1[1];
      if ( *(_DWORD *)(a2 + 64) != 1 )
      {
        v82 = sub_3405C90(
                v81,
                v129,
                (__int64)&v143,
                (unsigned int)v145,
                v146,
                v124,
                v6,
                v80,
                *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
        goto LABEL_66;
      }
      v105 = sub_33FA050(
               (__int64)v81,
               v129,
               (__int64)&v143,
               (unsigned int)v145,
               v146,
               v124,
               (unsigned __int8 *)v80,
               *((__int64 *)&v80 + 1));
      goto LABEL_86;
    }
  }
  if ( (_WORD)v145 )
  {
    v128 = 0;
    v131 = word_4456580[(unsigned __int16)v145 - 1];
  }
  else
  {
    v115 = sub_3009970((__int64)&v145, v27, v28, v29, v30);
    v131 = v115;
    v128 = v67;
  }
  HIWORD(v31) = HIWORD(v115);
  v32 = (_QWORD *)a1[1];
  v151 = 0;
  LOWORD(v31) = v131;
  LODWORD(v152) = 0;
  HIWORD(v33) = HIWORD(v115);
  v34 = sub_33F17F0(v32, 51, (__int64)&v151, v31, v128);
  v37 = v36;
  if ( v151 )
  {
    v121 = v34;
    sub_B91220((__int64)&v151, (__int64)v151);
    v34 = v121;
  }
  v38 = v37;
  v39 = v136;
  v151 = v153;
  v152 = 0x1000000000LL;
  if ( v136 > 0x10 )
  {
    v116 = v34;
    v119 = v38;
    sub_C8D5F0((__int64)&v151, v153, v136, 0x10u, v38, v35);
    v94 = (__int64)v151;
    v95 = &v151[2 * v136];
    do
    {
      if ( v94 )
      {
        *(_QWORD *)v94 = v116;
        *(_DWORD *)(v94 + 8) = v119;
      }
      v94 += 16;
    }
    while ( (_QWORD *)v94 != v95 );
  }
  else if ( v136 )
  {
    v40 = v153;
    do
    {
      *v40 = v34;
      v40 += 2;
      *((_DWORD *)v40 - 2) = v38;
      --v39;
    }
    while ( v39 );
  }
  LODWORD(v152) = v136;
  v41 = *(__int16 **)(a2 + 48);
  v42 = *v41;
  v43 = *((_QWORD *)v41 + 1);
  LOWORD(v149) = v42;
  v150 = v43;
  if ( v42 )
  {
    if ( (unsigned __int16)(v42 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v44 = word_4456340[v42 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v149) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v44 = sub_3007130((__int64)&v149, (__int64)v153);
  }
  v123 = v44;
  v45 = 0;
  if ( v44 )
  {
    do
    {
      while ( 1 )
      {
        v137 = (_QWORD *)a1[1];
        *(_QWORD *)&v51 = sub_3400EE0((__int64)v137, v45, (__int64)&v143, 0, v6);
        LOWORD(v20) = v132;
        *((_QWORD *)&v140 + 1) = v122 | *((_QWORD *)&v140 + 1) & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v52 = sub_3406EB0(
                            v137,
                            0x9Eu,
                            (__int64)&v143,
                            v20,
                            v125,
                            *((__int64 *)&v51 + 1),
                            __PAIR128__(*((unsigned __int64 *)&v140 + 1), v135),
                            v51);
        v53 = (_QWORD *)a1[1];
        v138 = 2 * v45;
        if ( *(_DWORD *)(a2 + 64) == 1 )
          break;
        LOWORD(v33) = v131;
        ++v45;
        v46 = sub_3405C90(v53, v129, (__int64)&v143, v33, v128, v124, v6, v52, *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
        v48 = v47;
        v49 = v46;
        v50 = (__int64)v151;
        v151[v138] = v49;
        *(_DWORD *)(v50 + v138 * 8 + 8) = v48;
        if ( v45 == v123 )
          goto LABEL_38;
      }
      ++v45;
      LOWORD(v33) = v131;
      v54 = sub_33FA050(
              (__int64)v53,
              v129,
              (__int64)&v143,
              v33,
              v128,
              v124,
              (unsigned __int8 *)v52,
              *((__int64 *)&v52 + 1));
      v56 = v55;
      v57 = v54;
      v58 = (__int64)v151;
      v151[v138] = v57;
      *(_DWORD *)(v58 + v138 * 8 + 8) = v56;
    }
    while ( v45 != v123 );
  }
LABEL_38:
  *((_QWORD *)&v113 + 1) = (unsigned int)v152;
  *(_QWORD *)&v113 = v151;
  v59 = sub_33FC220((_QWORD *)a1[1], 156, (__int64)&v143, v145, v146, v35, v113);
  v60 = (unsigned __int64)v151;
  v61 = (__int64)v59;
  if ( v151 != v153 )
LABEL_39:
    _libc_free(v60);
LABEL_40:
  if ( v143 )
    sub_B91220((__int64)&v143, v143);
  return v61;
}
