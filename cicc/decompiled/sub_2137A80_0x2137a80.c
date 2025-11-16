// Function: sub_2137A80
// Address: 0x2137a80
//
void __fastcall sub_2137A80(
        __m128i **a1,
        unsigned __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __m128i a5,
        __m128i a6,
        __m128i a7)
{
  char *v9; // rax
  __int64 v10; // rsi
  char v11; // dl
  const void **v12; // rax
  bool v13; // zf
  __m128i *v14; // rdi
  __int64 v15; // rax
  unsigned int v16; // edx
  char v17; // al
  __m128i *v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __m128i v21; // rax
  __m128i *v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int *v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned int *v29; // r15
  unsigned int *v30; // rbx
  __m128i *v31; // rsi
  __int64 v32; // rax
  __int8 v33; // dl
  __m128i *v34; // rax
  __int64 v35; // rax
  __int32 v36; // edx
  __m128i si128; // xmm2
  __int64 v38; // rax
  __m128i *v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rdx
  signed __int64 v42; // rsi
  __int64 v43; // r15
  __m128i *v44; // rax
  __int64 v45; // rcx
  __int64 v46; // rax
  unsigned __int32 v47; // ebx
  int v48; // edx
  void (***v49)(); // rdi
  void (*v50)(); // rax
  __m128i *v51; // rsi
  __m128i *v52; // rdi
  __int64 v53; // rbx
  __int64 v54; // rax
  __int64 v55; // rdi
  __int16 *v56; // rdx
  __int16 *v57; // r15
  unsigned __int64 v58; // r14
  __int64 v59; // rax
  __int64 v60; // rdx
  const void **v61; // rbx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 *v66; // rax
  __m128i *v67; // rdx
  const __m128i *v68; // r9
  int v69; // ebx
  __int64 v70; // rax
  __int128 v71; // xmm5
  __int64 v72; // r10
  unsigned __int64 v73; // r11
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 *v76; // rax
  __int64 v77; // rcx
  unsigned __int64 v78; // rdx
  __int64 *v79; // r15
  __int128 v80; // rax
  __m128i *v81; // r14
  __int64 (__fastcall *v82)(__m128i *, __int64, __int64, __int64, const void **); // rbx
  __int64 v83; // rax
  unsigned __int8 v84; // al
  __int64 v85; // rdx
  const void **v86; // rbx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rax
  __int64 v91; // rdx
  __int128 v92; // rax
  __int64 *v93; // r14
  __int128 v94; // rax
  __int128 v95; // rax
  __int64 *v96; // r14
  __int64 v97; // rax
  __int64 v98; // rdx
  __int16 *v99; // r15
  const void **v100; // rbx
  __int64 v101; // r8
  __int64 v102; // r9
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 *v105; // r14
  __int64 v106; // rax
  __int64 v107; // rdx
  __int64 v108; // r15
  __int128 v109; // rax
  __int64 v110; // rdx
  const __m128i *v111; // r9
  __int64 v112; // [rsp+8h] [rbp-10C8h]
  __int64 v113; // [rsp+10h] [rbp-10C0h]
  __int64 v114; // [rsp+18h] [rbp-10B8h]
  __int64 *v115; // [rsp+20h] [rbp-10B0h]
  __int128 v116; // [rsp+20h] [rbp-10B0h]
  __int64 v117; // [rsp+38h] [rbp-1098h]
  __int64 v118; // [rsp+38h] [rbp-1098h]
  __int128 v120; // [rsp+40h] [rbp-1090h]
  unsigned __int64 v122; // [rsp+50h] [rbp-1080h]
  __int64 v123; // [rsp+50h] [rbp-1080h]
  unsigned __int64 v124; // [rsp+50h] [rbp-1080h]
  unsigned __int64 v125; // [rsp+58h] [rbp-1078h]
  int v126; // [rsp+60h] [rbp-1070h]
  __int128 v127; // [rsp+60h] [rbp-1070h]
  __int128 v128; // [rsp+60h] [rbp-1070h]
  __int128 v129; // [rsp+70h] [rbp-1060h] BYREF
  const __m128i *v130; // [rsp+80h] [rbp-1050h]
  __int64 *v131; // [rsp+88h] [rbp-1048h]
  __int64 v132; // [rsp+90h] [rbp-1040h]
  __int64 v133; // [rsp+98h] [rbp-1038h]
  __int64 v134; // [rsp+A0h] [rbp-1030h]
  __int64 v135; // [rsp+A8h] [rbp-1028h]
  __m128i v136; // [rsp+B0h] [rbp-1020h]
  __int64 *v137; // [rsp+C0h] [rbp-1010h]
  __int64 v138; // [rsp+C8h] [rbp-1008h]
  __int64 v139; // [rsp+D0h] [rbp-1000h] BYREF
  const void **v140; // [rsp+D8h] [rbp-FF8h]
  __int64 v141; // [rsp+E0h] [rbp-FF0h] BYREF
  int v142; // [rsp+E8h] [rbp-FE8h]
  unsigned int v143; // [rsp+F0h] [rbp-FE0h] BYREF
  const void **v144; // [rsp+F8h] [rbp-FD8h]
  const __m128i *v145; // [rsp+100h] [rbp-FD0h] BYREF
  __m128i *v146; // [rsp+108h] [rbp-FC8h]
  const __m128i *v147; // [rsp+110h] [rbp-FC0h]
  __int64 v148; // [rsp+120h] [rbp-FB0h]
  __int64 v149; // [rsp+128h] [rbp-FA8h]
  __int64 v150; // [rsp+130h] [rbp-FA0h]
  _QWORD v151[4]; // [rsp+140h] [rbp-F90h] BYREF
  __int64 v152[4]; // [rsp+160h] [rbp-F70h] BYREF
  __m128i v153; // [rsp+180h] [rbp-F50h] BYREF
  __m128i v154; // [rsp+190h] [rbp-F40h] BYREF
  __int64 v155; // [rsp+1A0h] [rbp-F30h]
  __m128i v156; // [rsp+1B0h] [rbp-F20h] BYREF
  __int64 v157; // [rsp+1C0h] [rbp-F10h]
  unsigned __int64 v158; // [rsp+1C8h] [rbp-F08h]
  __int64 v159; // [rsp+1D0h] [rbp-F00h]
  __int64 v160; // [rsp+1D8h] [rbp-EF8h]
  __int64 v161; // [rsp+1E0h] [rbp-EF0h]
  const __m128i *v162; // [rsp+1E8h] [rbp-EE8h] BYREF
  __m128i *v163; // [rsp+1F0h] [rbp-EE0h]
  const __m128i *v164; // [rsp+1F8h] [rbp-ED8h]
  __int64 *v165; // [rsp+200h] [rbp-ED0h]
  __int64 v166; // [rsp+208h] [rbp-EC8h] BYREF
  int v167; // [rsp+210h] [rbp-EC0h]
  __int64 v168; // [rsp+218h] [rbp-EB8h]
  const __m128i *v169; // [rsp+220h] [rbp-EB0h]
  __int64 v170; // [rsp+228h] [rbp-EA8h]
  _BYTE v171[1536]; // [rsp+230h] [rbp-EA0h] BYREF
  _BYTE *v172; // [rsp+830h] [rbp-8A0h]
  __int64 v173; // [rsp+838h] [rbp-898h]
  _BYTE v174[512]; // [rsp+840h] [rbp-890h] BYREF
  _BYTE *v175; // [rsp+A40h] [rbp-690h]
  __int64 v176; // [rsp+A48h] [rbp-688h]
  _BYTE v177[1536]; // [rsp+A50h] [rbp-680h] BYREF
  _BYTE *v178; // [rsp+1050h] [rbp-80h]
  __int64 v179; // [rsp+1058h] [rbp-78h]
  _BYTE v180[112]; // [rsp+1060h] [rbp-70h] BYREF

  v9 = *(char **)(a2 + 40);
  v10 = *(_QWORD *)(a2 + 72);
  v11 = *v9;
  v12 = (const void **)*((_QWORD *)v9 + 1);
  v141 = v10;
  v140 = v12;
  LOBYTE(v139) = v11;
  v131 = &v141;
  if ( v10 )
    sub_1623A60((__int64)&v141, v10, 2);
  v13 = *(_WORD *)(a2 + 24) == 75;
  v14 = a1[1];
  v142 = *(_DWORD *)(a2 + 64);
  if ( v13 )
  {
    v70 = *(_QWORD *)(a2 + 32);
    v71 = (__int128)_mm_loadu_si128((const __m128i *)(v70 + 40));
    v72 = *(_QWORD *)v70;
    v73 = *(_QWORD *)(v70 + 8);
    v74 = *(_QWORD *)v70;
    v75 = *(unsigned int *)(v70 + 8);
    v129 = v71;
    *(_QWORD *)&v116 = v72;
    *((_QWORD *)&v116 + 1) = v73;
    v76 = sub_1D332F0(
            v14->m128i_i64,
            54,
            (__int64)v131,
            *(unsigned __int8 *)(*(_QWORD *)(v74 + 40) + 16 * v75),
            *(const void ***)(*(_QWORD *)(v74 + 40) + 16 * v75 + 8),
            0,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128i_i64,
            a7,
            v72,
            v73,
            v71);
    v77 = a3;
    v123 = (__int64)v76;
    v125 = v78;
    sub_200E870((__int64)a1, (__int64)v76, v78, v77, a4, a5, *(double *)a6.m128i_i64, a7);
    v79 = (__int64 *)a1[1];
    *(_QWORD *)&v80 = sub_1D38BB0(
                        (__int64)v79,
                        0,
                        (__int64)v131,
                        (unsigned int)v139,
                        v140,
                        0,
                        a5,
                        *(double *)a6.m128i_i64,
                        a7,
                        0);
    v81 = *a1;
    v127 = v80;
    *(_QWORD *)&v80 = a1[1];
    *((_QWORD *)&v80 + 1) = (*a1)->m128i_i64[0];
    v130 = (const __m128i *)v140;
    v82 = *(__int64 (__fastcall **)(__m128i *, __int64, __int64, __int64, const void **))(*((_QWORD *)&v80 + 1) + 264LL);
    v118 = *(_QWORD *)(v80 + 48);
    v83 = sub_1E0A0C0(*(_QWORD *)(v80 + 32));
    v84 = v82(v81, v83, v118, v139, v140);
    v86 = (const void **)v85;
    LODWORD(v81) = v84;
    v90 = sub_1D28D50(v79, 0x11u, v85, v87, v88, v89);
    *(_QWORD *)&v92 = sub_1D3A900(
                        v79,
                        0x89u,
                        (__int64)v131,
                        (unsigned int)v81,
                        v86,
                        0,
                        (__m128)a5,
                        *(double *)a6.m128i_i64,
                        a7,
                        v129,
                        *((__int16 **)&v129 + 1),
                        v127,
                        v90,
                        v91);
    v93 = (__int64 *)a1[1];
    v128 = v92;
    *(_QWORD *)&v94 = sub_1D38BB0(
                        (__int64)v93,
                        1,
                        (__int64)v131,
                        (unsigned int)v139,
                        v140,
                        0,
                        a5,
                        *(double *)a6.m128i_i64,
                        a7,
                        0);
    *(_QWORD *)&v95 = sub_1F810E0(
                        v93,
                        (__int64)v131,
                        (unsigned int)v139,
                        v140,
                        v128,
                        *((__int16 **)&v128 + 1),
                        (__m128)a5,
                        *(double *)a6.m128i_i64,
                        a7,
                        v94,
                        v129,
                        *((__int64 *)&v129 + 1));
    v96 = sub_1D332F0(
            a1[1]->m128i_i64,
            56,
            (__int64)v131,
            (unsigned int)v139,
            v140,
            0,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128i_i64,
            a7,
            v123,
            v125,
            v95);
    v97 = *(_QWORD *)(a2 + 40);
    v99 = (__int16 *)v98;
    *(_QWORD *)&v129 = a1[1];
    v100 = *(const void ***)(v97 + 24);
    v124 = *(unsigned __int8 *)(v97 + 16);
    v103 = sub_1D28D50((_QWORD *)v129, 0x16u, v98, v124, v101, v102);
    v105 = sub_1D3A900(
             (__int64 *)v129,
             0x89u,
             (__int64)v131,
             v124,
             v100,
             0,
             (__m128)a5,
             *(double *)a6.m128i_i64,
             a7,
             (unsigned __int64)v96,
             v99,
             v116,
             v103,
             v104);
    v106 = *(_QWORD *)(a2 + 40);
    v108 = v107;
    *(_QWORD *)&v129 = a1[1];
    *(_QWORD *)&v109 = sub_1D38BB0(
                         v129,
                         0,
                         (__int64)v131,
                         *(unsigned __int8 *)(v106 + 16),
                         *(const void ***)(v106 + 24),
                         0,
                         a5,
                         *(double *)a6.m128i_i64,
                         a7,
                         0);
    v137 = sub_1F810E0(
             (__int64 *)v129,
             (__int64)v131,
             *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL),
             *(const void ***)(*(_QWORD *)(a2 + 40) + 24LL),
             v128,
             *((__int16 **)&v128 + 1),
             (__m128)a5,
             *(double *)a6.m128i_i64,
             a7,
             v109,
             (__int64)v105,
             v108);
    v138 = v110;
    sub_2013400((__int64)a1, a2, 1, (__int64)v137, (__m128i *)((unsigned int)v110 | v108 & 0xFFFFFFFF00000000LL), v111);
    if ( v141 )
      sub_161E7C0((__int64)v131, v141);
  }
  else
  {
    v117 = sub_1F58E60((__int64)&v139, (_QWORD *)v14[3].m128i_i64[0]);
    v15 = sub_1E0A0C0(a1[1][2].m128i_i64[0]);
    v16 = 8 * sub_15A9520(v15, 0);
    if ( v16 == 32 )
    {
      v17 = 5;
    }
    else if ( v16 > 0x20 )
    {
      v17 = 6;
      if ( v16 != 64 )
      {
        v17 = 0;
        if ( v16 == 128 )
          v17 = 7;
      }
    }
    else
    {
      v17 = 3;
      if ( v16 != 8 )
        v17 = 4 * (v16 == 16);
    }
    LOBYTE(v143) = v17;
    v18 = a1[1];
    v144 = 0;
    v126 = 17;
    v115 = (__int64 *)sub_1F58E60((__int64)&v143, (_QWORD *)v18[3].m128i_i64[0]);
    if ( (_BYTE)v139 != 5 )
    {
      v126 = 18;
      if ( (_BYTE)v139 != 6 )
      {
        v69 = 19;
        if ( (_BYTE)v139 != 7 )
          v69 = 462;
        v126 = v69;
      }
    }
    v21.m128i_i64[0] = (__int64)sub_1D29C20(a1[1], v143, (__int64)v144, 1, v19, v20);
    v22 = a1[1];
    v129 = (__int128)v21;
    v156 = 0u;
    v157 = 0;
    v153 = 0u;
    v154.m128i_i64[0] = 0;
    v23 = sub_1D38BB0((__int64)v22, 0, (__int64)v131, v143, v144, 0, a5, *(double *)a6.m128i_i64, a7, 0);
    v25 = sub_1D2BF40(
            v22,
            (__int64)&a1[1][5].m128i_i64[1],
            0,
            (__int64)v131,
            v23,
            v24,
            v129,
            *((__int64 *)&v129 + 1),
            *(_OWORD *)&v153,
            v154.m128i_i64[0],
            0,
            0,
            (__int64)&v156);
    v26 = *(unsigned int **)(a2 + 32);
    v145 = 0;
    v114 = v25;
    v27 = *(unsigned int *)(a2 + 56);
    v113 = v28;
    v29 = v26;
    v146 = 0;
    v147 = 0;
    v30 = &v26[10 * v27];
    v153 = 0u;
    v154 = 0u;
    LODWORD(v155) = 0;
    v130 = &v153;
    if ( v26 != v30 )
    {
      do
      {
        while ( 1 )
        {
          v32 = *(_QWORD *)(*(_QWORD *)v29 + 40LL) + 16LL * v29[2];
          v33 = *(_BYTE *)v32;
          v156.m128i_i64[1] = *(_QWORD *)(v32 + 8);
          v34 = a1[1];
          v156.m128i_i8[0] = v33;
          v35 = sub_1F58E60((__int64)&v156, (_QWORD *)v34[3].m128i_i64[0]);
          v31 = v146;
          v153.m128i_i64[1] = *(_QWORD *)v29;
          v36 = v29[2];
          v154.m128i_i64[1] = v35;
          v154.m128i_i32[0] = v36;
          LOBYTE(v155) = v155 & 0xFC | 1;
          if ( v146 != v147 )
            break;
          v29 += 10;
          sub_1D27190(&v145, v146, v130);
          if ( v30 == v29 )
            goto LABEL_16;
        }
        if ( v146 )
        {
          a5 = _mm_loadu_si128(&v153);
          *v146 = a5;
          a6 = _mm_loadu_si128(&v154);
          v31[1] = a6;
          v31[2].m128i_i64[0] = v155;
          v31 = v146;
        }
        v29 += 10;
        v146 = (__m128i *)((char *)v31 + 40);
      }
      while ( v30 != v29 );
    }
LABEL_16:
    si128 = _mm_load_si128((const __m128i *)&v129);
    v153.m128i_i64[1] = v129;
    v136 = si128;
    v154.m128i_i32[0] = si128.m128i_i32[2];
    v38 = sub_1647190(v115, 0);
    v39 = v146;
    v154.m128i_i64[1] = v38;
    LOBYTE(v155) = v155 & 0xFC | 1;
    if ( v146 == v147 )
    {
      sub_1D27190(&v145, v146, &v153);
    }
    else
    {
      if ( v146 )
      {
        *v146 = _mm_loadu_si128(&v153);
        v39[1] = _mm_loadu_si128(&v154);
        v39[2].m128i_i64[0] = v155;
        v39 = v146;
      }
      v146 = (__m128i *)((char *)v39 + 40);
    }
    v40 = sub_1D27640((__int64)a1[1], (char *)(*a1)[4631].m128i_i64[v126], v143, (__int64)v144);
    v42 = 0;
    v156 = 0u;
    v43 = v40;
    v44 = a1[1];
    v158 = 0xFFFFFFFF00000020LL;
    v45 = v41;
    v165 = (__int64 *)v44;
    v130 = (const __m128i *)v171;
    v169 = (const __m128i *)v171;
    v170 = 0x2000000000LL;
    v173 = 0x2000000000LL;
    v176 = 0x2000000000LL;
    v178 = v180;
    v179 = 0x400000000LL;
    v46 = v141;
    v172 = v174;
    v157 = 0;
    v159 = 0;
    v160 = 0;
    v161 = 0;
    v162 = 0;
    v163 = 0;
    v164 = 0;
    v167 = 0;
    v168 = 0;
    v175 = v177;
    v166 = v141;
    if ( v141 )
    {
      v112 = v41;
      sub_1623A60((__int64)&v166, v141, 2);
      v46 = (__int64)v162;
      v45 = v112;
      v42 = (char *)v164 - (char *)v162;
    }
    v167 = v142;
    v134 = v114;
    v156.m128i_i64[0] = v114;
    v135 = v113;
    v156.m128i_i32[2] = v113;
    v47 = (*a1)[4978].m128i_u32[v126];
    v133 = v45;
    v157 = v117;
    v132 = v43;
    LODWORD(v161) = v45;
    v160 = v43;
    LODWORD(v159) = v47;
    v48 = -858993459 * (((char *)v146 - (char *)v145) >> 3);
    v162 = v145;
    v163 = v146;
    v145 = 0;
    HIDWORD(v158) = v48;
    v146 = 0;
    v164 = v147;
    v147 = 0;
    if ( v46 )
      j_j___libc_free_0(v46, v42);
    v49 = (void (***)())v165[2];
    v50 = **v49;
    if ( v50 != nullsub_684 )
      ((void (__fastcall *)(void (***)(), __int64, _QWORD, const __m128i **))v50)(v49, v165[4], v47, &v162);
    v51 = *a1;
    LOBYTE(v158) = v158 | 1;
    sub_2056920((__int64)v152, v51, &v156, a5, a6, si128);
    sub_200E870((__int64)a1, v152[0], v152[1], a3, a4, a5, *(double *)a6.m128i_i64, si128);
    v52 = a1[1];
    v53 = (__int64)v131;
    v149 = 0;
    v150 = 0;
    memset(v151, 0, 24);
    v148 = 0;
    v54 = sub_1D2B730(
            v52,
            v143,
            (__int64)v144,
            (__int64)v131,
            v152[2],
            v152[3],
            v129,
            *((__int64 *)&v129 + 1),
            0,
            0,
            0,
            0,
            (__int64)v151,
            0);
    v55 = (__int64)a1[1];
    v57 = v56;
    v58 = v54;
    v131 = (__int64 *)v53;
    *(_QWORD *)&v129 = v55;
    *(_QWORD *)&v120 = sub_1D38BB0(v55, 0, v53, v143, v144, 0, a5, *(double *)a6.m128i_i64, si128, 0);
    v59 = *(_QWORD *)(a2 + 40);
    *((_QWORD *)&v120 + 1) = v60;
    v61 = *(const void ***)(v59 + 24);
    v122 = *(unsigned __int8 *)(v59 + 16);
    v64 = sub_1D28D50((_QWORD *)v129, 0x16u, v60, v122, v62, v63);
    v66 = sub_1D3A900(
            (__int64 *)v129,
            0x89u,
            (__int64)v131,
            v122,
            v61,
            0,
            (__m128)a5,
            *(double *)a6.m128i_i64,
            si128,
            v58,
            v57,
            v120,
            v64,
            v65);
    sub_2013400((__int64)a1, a2, 1, (__int64)v66, v67, v68);
    if ( v178 != v180 )
      _libc_free((unsigned __int64)v178);
    if ( v175 != v177 )
      _libc_free((unsigned __int64)v175);
    if ( v172 != v174 )
      _libc_free((unsigned __int64)v172);
    if ( v169 != v130 )
      _libc_free((unsigned __int64)v169);
    if ( v166 )
      sub_161E7C0((__int64)&v166, v166);
    if ( v162 )
      j_j___libc_free_0(v162, (char *)v164 - (char *)v162);
    if ( v145 )
      j_j___libc_free_0(v145, (char *)v147 - (char *)v145);
    if ( v141 )
      sub_161E7C0((__int64)v131, v141);
  }
}
