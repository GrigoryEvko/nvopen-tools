// Function: sub_3466E90
// Address: 0x3466e90
//
__int64 __fastcall sub_3466E90(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  int v9; // r14d
  __int64 v10; // rsi
  __int64 v11; // r8
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int16 v16; // bx
  __int64 v17; // rax
  bool v18; // al
  unsigned __int16 *v19; // rcx
  int v20; // eax
  __int64 v21; // r15
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // r12
  __int64 v26; // rax
  __int128 v27; // rax
  __int64 v28; // r9
  unsigned __int8 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 (__fastcall *v34)(_DWORD *, __int64, __int64, _QWORD, __int64); // r15
  __int64 v35; // rax
  unsigned int v36; // eax
  __int64 v37; // rdx
  unsigned int *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r9
  unsigned __int8 *v41; // rax
  __int64 v42; // rdx
  __int128 v43; // rax
  unsigned __int8 *v44; // r10
  __int64 v45; // rdx
  __int64 v46; // r11
  unsigned int v47; // ebx
  __int64 v48; // r13
  unsigned __int8 *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r14
  unsigned __int8 *v52; // r13
  __int128 v53; // rax
  __int64 v54; // r9
  __int128 v55; // rax
  __int64 v56; // r9
  __int64 v57; // r14
  unsigned int v58; // edx
  __int64 v59; // rdi
  unsigned int v60; // esi
  __int64 v61; // rdx
  unsigned __int64 v62; // r9
  __int64 v63; // rdi
  __int64 v64; // r13
  __int16 v65; // ax
  __int64 v66; // rdi
  __int128 v67; // kr10_16
  unsigned int v68; // r14d
  unsigned __int64 v69; // rdi
  unsigned int v70; // edx
  __int64 v71; // r8
  char v72; // bl
  bool v73; // al
  int v74; // eax
  __int64 v75; // rdi
  __int64 v76; // r13
  __int64 v77; // rsi
  __int128 v78; // rcx
  __int16 v79; // ax
  __int64 v80; // rsi
  unsigned int v81; // esi
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // r8
  unsigned __int64 v85; // rax
  unsigned __int64 v86; // rdx
  bool v87; // dl
  __int64 v88; // rsi
  __int64 v89; // rax
  unsigned __int64 v90; // rdx
  bool v91; // si
  unsigned __int64 v92; // rdx
  bool v93; // al
  unsigned __int16 v94; // cx
  unsigned int v95; // edx
  __int64 v96; // r8
  unsigned __int16 v97; // cx
  int v98; // eax
  __int128 v99; // rax
  __int64 v100; // r9
  __int128 v101; // rax
  __int64 v102; // r9
  bool v103; // al
  __int128 v104; // rax
  __int64 v105; // r9
  bool v106; // al
  char v107; // bl
  bool v108; // al
  __int64 v109; // rbx
  __int64 v110; // r13
  __int64 v111; // rcx
  __int16 v112; // ax
  __int64 v113; // rcx
  __int128 v114; // kr30_16
  bool v115; // al
  __int64 v116; // r8
  __int128 v117; // rax
  unsigned int v118; // r13d
  __int64 v119; // rdx
  __int64 v120; // rbx
  __int64 v121; // rsi
  __int16 v122; // ax
  __int64 v123; // rsi
  __int128 v124; // kr40_16
  unsigned int v125; // esi
  unsigned __int8 *v126; // rax
  __int64 v127; // rdx
  __int128 v128; // rax
  __int64 v129; // r9
  __int128 v130; // rax
  unsigned __int64 v131; // rdx
  bool v132; // al
  __int128 v133; // [rsp-10h] [rbp-160h]
  __int128 v134; // [rsp+0h] [rbp-150h]
  __int128 v135; // [rsp+10h] [rbp-140h]
  unsigned int v136; // [rsp+10h] [rbp-140h]
  __int64 v137; // [rsp+28h] [rbp-128h]
  __int64 v138; // [rsp+30h] [rbp-120h]
  __int64 v139; // [rsp+30h] [rbp-120h]
  __int64 v140; // [rsp+30h] [rbp-120h]
  __int64 v141; // [rsp+38h] [rbp-118h]
  __int64 v142; // [rsp+38h] [rbp-118h]
  __int64 v143; // [rsp+38h] [rbp-118h]
  unsigned __int8 *v144; // [rsp+38h] [rbp-118h]
  __int64 v145; // [rsp+38h] [rbp-118h]
  unsigned int v146; // [rsp+38h] [rbp-118h]
  __int64 v147; // [rsp+38h] [rbp-118h]
  __int64 v148; // [rsp+40h] [rbp-110h]
  unsigned int v149; // [rsp+40h] [rbp-110h]
  __int64 v150; // [rsp+40h] [rbp-110h]
  unsigned int v151; // [rsp+40h] [rbp-110h]
  __int64 v152; // [rsp+40h] [rbp-110h]
  unsigned int v153; // [rsp+50h] [rbp-100h]
  __int64 v154; // [rsp+50h] [rbp-100h]
  unsigned int v155; // [rsp+50h] [rbp-100h]
  __int128 v156; // [rsp+60h] [rbp-F0h]
  unsigned __int8 *v157; // [rsp+60h] [rbp-F0h]
  unsigned __int8 *v158; // [rsp+60h] [rbp-F0h]
  unsigned int v159; // [rsp+60h] [rbp-F0h]
  __int128 v160; // [rsp+60h] [rbp-F0h]
  __int64 v161; // [rsp+68h] [rbp-E8h]
  __int64 v162; // [rsp+68h] [rbp-E8h]
  unsigned __int8 *v163; // [rsp+70h] [rbp-E0h]
  unsigned __int64 v164; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v165; // [rsp+88h] [rbp-C8h]
  __int64 v166; // [rsp+90h] [rbp-C0h] BYREF
  int v167; // [rsp+98h] [rbp-B8h]
  unsigned __int64 v168; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned int v169; // [rsp+A8h] [rbp-A8h]
  unsigned __int64 v170; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned int v171; // [rsp+B8h] [rbp-98h]
  __int64 v172; // [rsp+C0h] [rbp-90h]
  __int64 v173; // [rsp+C8h] [rbp-88h]
  __int16 v174; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v175; // [rsp+D8h] [rbp-78h]
  unsigned __int64 v176; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v177; // [rsp+E8h] [rbp-68h]
  unsigned __int64 v178; // [rsp+F0h] [rbp-60h]
  unsigned int v179; // [rsp+F8h] [rbp-58h]
  unsigned __int64 v180; // [rsp+100h] [rbp-50h] BYREF
  __int64 v181; // [rsp+108h] [rbp-48h]
  unsigned __int64 v182; // [rsp+110h] [rbp-40h]
  unsigned int v183; // [rsp+118h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_DWORD *)(a2 + 24);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *(_QWORD *)v8;
  v12 = _mm_loadu_si128((const __m128i *)v8);
  v13 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v14 = 16LL * *(unsigned int *)(v8 + 8);
  v15 = v14 + *(_QWORD *)(*(_QWORD *)v8 + 48LL);
  v16 = *(_WORD *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v166 = v10;
  LOWORD(v164) = v16;
  v165 = v17;
  if ( v10 )
  {
    v141 = v11;
    v148 = v14;
    sub_B96E90((__int64)&v166, v10, 1);
    v11 = v141;
    v14 = v148;
  }
  v167 = *(_DWORD *)(a2 + 72);
  if ( v9 == 85 )
  {
    v26 = 1;
    if ( v16 == 1 || v16 && (v26 = v16, *(_QWORD *)&a1[2 * v16 + 28]) )
    {
      if ( !BYTE1(a1[125 * v26 + 1649]) )
      {
        *(_QWORD *)&v27 = sub_3406EB0(
                            (_QWORD *)a3,
                            0xB7u,
                            (__int64)&v166,
                            (unsigned int)v164,
                            v165,
                            a6,
                            *(_OWORD *)&v12,
                            *(_OWORD *)&v13);
        v29 = sub_3406EB0((_QWORD *)a3, 0x39u, (__int64)&v166, (unsigned int)v164, v165, v28, v27, *(_OWORD *)&v13);
        goto LABEL_28;
      }
    }
  }
  else if ( v9 == 83 )
  {
    v22 = 1;
    if ( v16 == 1 || v16 && (v22 = v16, *(_QWORD *)&a1[2 * v16 + 28]) )
    {
      if ( !LOBYTE(a1[125 * v22 + 1649]) )
      {
        *(_QWORD *)&v99 = sub_34074A0((_QWORD *)a3, (__int64)&v166, v13.m128i_i64[0], v13.m128i_i64[1], v164, v165, v12);
        *(_QWORD *)&v101 = sub_3406EB0(
                             (_QWORD *)a3,
                             0xB6u,
                             (__int64)&v166,
                             (unsigned int)v164,
                             v165,
                             v100,
                             *(_OWORD *)&v12,
                             v99);
        v29 = sub_3406EB0((_QWORD *)a3, 0x38u, (__int64)&v166, (unsigned int)v164, v165, v102, v101, *(_OWORD *)&v13);
        goto LABEL_28;
      }
      v149 = 77;
      goto LABEL_17;
    }
  }
  else if ( (unsigned int)(v9 - 82) > 3 )
  {
    goto LABEL_162;
  }
  v149 = v9 - 6;
  if ( !v16 )
  {
    v138 = v11;
    v142 = v14;
    v18 = sub_30070B0((__int64)&v164);
    v14 = v142;
    v11 = v138;
    if ( !v18 )
      goto LABEL_8;
LABEL_21:
    v24 = (__int64)sub_3412A00((_QWORD *)a3, a2, 0, v14, v11, a6, v12);
    goto LABEL_22;
  }
LABEL_17:
  if ( (unsigned __int16)(v16 - 17) <= 0xD3u )
  {
    if ( !*(_QWORD *)&a1[2 * v16 + 28] )
      goto LABEL_21;
    v23 = a1[125 * v16 + 1655];
    if ( v23 )
    {
      if ( v23 != 4 )
        goto LABEL_21;
    }
  }
LABEL_8:
  v19 = (unsigned __int16 *)(*(_QWORD *)(v11 + 48) + v14);
  v20 = *v19;
  v21 = *((_QWORD *)v19 + 1);
  LOWORD(v176) = v20;
  v177 = v21;
  if ( (_WORD)v20 )
  {
    if ( (unsigned __int16)(v20 - 17) > 0xD3u )
    {
      LOWORD(v180) = v20;
      v181 = v21;
LABEL_11:
      if ( (_WORD)v20 != 1 && (unsigned __int16)(v20 - 504) > 7u )
      {
        v139 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v20 - 16];
        goto LABEL_32;
      }
LABEL_162:
      BUG();
    }
    LOWORD(v20) = word_4456580[v20 - 1];
    v83 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v176) )
    {
      v181 = v21;
      LOWORD(v180) = 0;
      goto LABEL_31;
    }
    LOWORD(v20) = sub_3009970((__int64)&v176, v10, v30, v31, v32);
  }
  LOWORD(v180) = v20;
  v181 = v83;
  if ( (_WORD)v20 )
    goto LABEL_11;
LABEL_31:
  v172 = sub_3007260((__int64)&v180);
  v173 = v33;
  LODWORD(v139) = v172;
LABEL_32:
  v34 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
  v143 = *(_QWORD *)(a3 + 64);
  v35 = sub_2E79000(*(__int64 **)(a3 + 40));
  v36 = v34(a1, v35, v143, (unsigned int)v164, v165);
  v38 = (unsigned int *)sub_33E5110((__int64 *)a3, (unsigned int)v164, v165, v36, v37);
  v41 = sub_3411F20((_QWORD *)a3, v149, (__int64)&v166, v38, v39, v40, *(_OWORD *)&v12, *(_OWORD *)&v13);
  v137 = v42;
  v150 = (__int64)v41;
  v144 = v41;
  *(_QWORD *)&v43 = sub_3400BD0(a3, 0, (__int64)&v166, (unsigned int)v164, v165, 0, v12, 0);
  v135 = v43;
  v44 = sub_34015B0(a3, (__int64)&v166, (unsigned int)v164, v165, 0, 0, v12);
  v46 = v45;
  if ( v9 == 83 )
  {
    v70 = v164;
    v71 = v165;
    v180 = v164;
    v181 = v165;
    if ( v16 )
    {
      v94 = v16 - 17;
      if ( (unsigned __int16)(v16 - 10) <= 6u || (unsigned __int16)(v16 - 126) <= 0x31u )
      {
        if ( v94 > 0xD3u )
          goto LABEL_115;
      }
      else if ( v94 > 0xD3u )
      {
        goto LABEL_46;
      }
    }
    else
    {
      v145 = v165;
      v153 = v164;
      v157 = v44;
      v161 = v46;
      v72 = sub_3007030((__int64)&v180);
      v73 = sub_30070B0((__int64)&v180);
      v46 = v161;
      v44 = v157;
      v70 = v153;
      v71 = v145;
      if ( !v73 )
      {
        if ( !v72 )
        {
LABEL_46:
          v74 = a1[15];
LABEL_47:
          if ( v74 != 2 )
          {
            v75 = v150;
            v76 = v150;
            v77 = *(_QWORD *)(v150 + 48);
            v78 = (unsigned __int64)v150;
            v79 = *(_WORD *)(v77 + 16);
            v80 = *(_QWORD *)(v77 + 24);
            LOWORD(v180) = v79;
            v181 = v80;
            if ( v79 )
            {
              v81 = ((unsigned __int16)(v79 - 17) < 0xD4u) + 205;
            }
            else
            {
              v147 = v71;
              v151 = v70;
              v158 = v44;
              v162 = v46;
              v106 = sub_30070B0((__int64)&v180);
              v71 = v147;
              v70 = v151;
              v78 = (unsigned __int64)v75;
              v46 = v162;
              v44 = v158;
              v81 = 205 - (!v106 - 1);
            }
            *((_QWORD *)&v133 + 1) = v46;
            *(_QWORD *)&v133 = v44;
            v82 = sub_340EC60((_QWORD *)a3, v81, (__int64)&v166, v70, v71, 0, v76, 1, v133, v78);
            goto LABEL_51;
          }
          *(_QWORD *)&v104 = sub_33FB160(a3, v150, 1u, (__int64)&v166, (unsigned int)v164, v165, v12);
          v29 = sub_3406EB0(
                  (_QWORD *)a3,
                  0xBBu,
                  (__int64)&v166,
                  (unsigned int)v164,
                  v165,
                  v105,
                  (unsigned __int64)v150,
                  v104);
LABEL_28:
          v24 = (__int64)v29;
          goto LABEL_22;
        }
LABEL_115:
        v74 = a1[16];
        goto LABEL_47;
      }
    }
    v74 = a1[17];
    goto LABEL_47;
  }
  if ( v9 == 85 )
  {
    v95 = v164;
    v96 = v165;
    v180 = v164;
    v181 = v165;
    if ( v16 )
    {
      v97 = v16 - 17;
      if ( (unsigned __int16)(v16 - 10) > 6u && (unsigned __int16)(v16 - 126) > 0x31u )
      {
        if ( v97 <= 0xD3u )
        {
LABEL_107:
          v98 = a1[17];
LABEL_119:
          if ( v98 != 2 )
          {
            v109 = v150;
            v110 = v150;
            v111 = *(_QWORD *)(v150 + 48);
            v112 = *(_WORD *)(v111 + 16);
            v113 = *(_QWORD *)(v111 + 24);
            LOWORD(v180) = v112;
            v181 = v113;
            v114 = (unsigned __int64)v150;
            if ( v112 )
            {
              v115 = (unsigned __int16)(v112 - 17) <= 0xD3u;
            }
            else
            {
              v152 = v96;
              v155 = v95;
              v115 = sub_30070B0((__int64)&v180);
              v96 = v152;
              v95 = v155;
              v114 = (unsigned __int64)v109;
            }
            v82 = sub_340EC60(
                    (_QWORD *)a3,
                    205 - ((unsigned int)!v115 - 1),
                    (__int64)&v166,
                    v95,
                    v96,
                    0,
                    v110,
                    1,
                    v135,
                    v114);
LABEL_51:
            v24 = v82;
            goto LABEL_22;
          }
          v126 = sub_33FB160(a3, v150, 1u, (__int64)&v166, (unsigned int)v164, v165, v12);
          *(_QWORD *)&v128 = sub_34074A0((_QWORD *)a3, (__int64)&v166, (__int64)v126, v127, v164, v165, v12);
          v29 = sub_3406EB0(
                  (_QWORD *)a3,
                  0xBAu,
                  (__int64)&v166,
                  (unsigned int)v164,
                  v165,
                  v129,
                  (unsigned __int64)v150,
                  v128);
          goto LABEL_28;
        }
        goto LABEL_118;
      }
      if ( v97 <= 0xD3u )
        goto LABEL_107;
    }
    else
    {
      v154 = v165;
      v159 = v164;
      v107 = sub_3007030((__int64)&v180);
      v108 = sub_30070B0((__int64)&v180);
      v95 = v159;
      v96 = v154;
      if ( v108 )
        goto LABEL_107;
      if ( !v107 )
      {
LABEL_118:
        v98 = a1[15];
        goto LABEL_119;
      }
    }
    v98 = a1[16];
    goto LABEL_119;
  }
  v47 = v139 - 1;
  v136 = v139;
  v48 = 1LL << ((unsigned __int8)v139 - 1);
  if ( ((v9 - 82) & 0xFFFFFFFD) != 0 )
    goto LABEL_35;
  v169 = v139;
  if ( (unsigned int)v139 > 0x40 )
  {
    sub_C43690((__int64)&v168, 0, 0);
    if ( v169 <= 0x40 )
    {
      v168 |= v48;
      v116 = ~v48;
    }
    else
    {
      v116 = ~v48;
      *(_QWORD *)(v168 + 8LL * (v47 >> 6)) |= v48;
    }
    v171 = v139;
    v140 = v116;
    sub_C43690((__int64)&v170, -1, 1);
    v84 = v140;
    if ( v171 > 0x40 )
    {
      *(_QWORD *)(v170 + 8LL * (v47 >> 6)) &= v140;
      goto LABEL_66;
    }
  }
  else
  {
    v171 = v139;
    v84 = ~v48;
    v85 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v139 - 1) & 0x3F));
    v168 = 1LL << ((unsigned __int8)v139 - 1);
    if ( !(_DWORD)v139 )
      v85 = 0;
    v170 = v85;
  }
  v170 &= v84;
LABEL_66:
  sub_33DD090((__int64)&v176, a3, v12.m128i_i64[0], v12.m128i_i64[1], 0);
  sub_33DD090((__int64)&v180, a3, v13.m128i_i64[0], v13.m128i_i64[1], 0);
  if ( (unsigned int)v177 > 0x40 )
    v86 = *(_QWORD *)(v176 + 8LL * ((unsigned int)(v177 - 1) >> 6));
  else
    v86 = v176;
  v87 = (v86 & (1LL << ((unsigned __int8)v177 - 1))) != 0;
  if ( v9 == 82 )
  {
    v88 = v180;
    v89 = 1LL << ((unsigned __int8)v181 - 1);
    if ( (unsigned int)v181 > 0x40 )
      v88 = *(_QWORD *)(v180 + 8LL * ((unsigned int)(v181 - 1) >> 6));
  }
  else
  {
    v88 = 1LL << ((unsigned __int8)v183 - 1);
    v89 = v182;
    if ( v183 > 0x40 )
      v89 = *(_QWORD *)(v182 + 8LL * ((v183 - 1) >> 6));
  }
  if ( (v89 & v88) != 0 || v87 )
  {
    *(_QWORD *)&v130 = sub_34007B0(a3, (__int64)&v170, (__int64)&v166, v164, v165, 0, v12, 0);
    v24 = sub_3288B20(a3, (int)&v166, v164, v165, v150, 1, v130, (unsigned __int64)v150, 0);
  }
  else
  {
    v90 = v178;
    if ( v179 > 0x40 )
      v90 = *(_QWORD *)(v178 + 8LL * ((v179 - 1) >> 6));
    v91 = (v90 & (1LL << ((unsigned __int8)v179 - 1))) != 0;
    if ( v9 == 82 )
    {
      v131 = v182;
      if ( v183 > 0x40 )
        v131 = *(_QWORD *)(v182 + 8LL * ((v183 - 1) >> 6));
      v93 = (v131 & (1LL << ((unsigned __int8)v183 - 1))) != 0;
    }
    else
    {
      v92 = v180;
      if ( (unsigned int)v181 > 0x40 )
        v92 = *(_QWORD *)(v180 + 8LL * ((unsigned int)(v181 - 1) >> 6));
      v93 = (v92 & (1LL << ((unsigned __int8)v181 - 1))) != 0;
    }
    if ( !v93 && !v91 )
    {
      if ( v183 > 0x40 && v182 )
        j_j___libc_free_0_0(v182);
      if ( (unsigned int)v181 > 0x40 && v180 )
        j_j___libc_free_0_0(v180);
      if ( v179 > 0x40 && v178 )
        j_j___libc_free_0_0(v178);
      if ( (unsigned int)v177 > 0x40 && v176 )
        j_j___libc_free_0_0(v176);
      if ( v171 > 0x40 && v170 )
        j_j___libc_free_0_0(v170);
      if ( v169 > 0x40 && v168 )
        j_j___libc_free_0_0(v168);
LABEL_35:
      LODWORD(v177) = v136;
      if ( v136 > 0x40 )
      {
        sub_C43690((__int64)&v176, 0, 0);
        if ( (unsigned int)v177 > 0x40 )
        {
          *(_QWORD *)(v176 + 8LL * (v47 >> 6)) |= v48;
LABEL_38:
          v49 = sub_34007B0(a3, (__int64)&v176, (__int64)&v166, v164, v165, 0, v12, 0);
          v51 = v50;
          v52 = v49;
          *(_QWORD *)&v53 = sub_3400BD0(a3, v47, (__int64)&v166, (unsigned int)v164, v165, 0, v12, 0);
          *(_QWORD *)&v55 = sub_3406EB0(
                              (_QWORD *)a3,
                              0xBFu,
                              (__int64)&v166,
                              (unsigned int)v164,
                              v165,
                              v54,
                              (unsigned __int64)v150,
                              v53);
          *((_QWORD *)&v134 + 1) = v51;
          *(_QWORD *)&v134 = v52;
          v163 = sub_3406EB0((_QWORD *)a3, 0xBCu, (__int64)&v166, (unsigned int)v164, v165, v56, v55, v134);
          v57 = (__int64)v144;
          v59 = v58;
          *(_QWORD *)&v156 = v163;
          v60 = v164;
          v61 = (__int64)v144;
          v62 = v59 | v137 & 0xFFFFFFFF00000000LL;
          v63 = *((_QWORD *)v144 + 6);
          *((_QWORD *)&v156 + 1) = v62;
          v64 = v165;
          v65 = *(_WORD *)(v63 + 16);
          v66 = *(_QWORD *)(v63 + 24);
          LOWORD(v180) = v65;
          v181 = v66;
          v67 = (unsigned __int64)v144;
          if ( v65 )
          {
            v68 = ((unsigned __int16)(v65 - 17) < 0xD4u) + 205;
          }
          else
          {
            v146 = v164;
            v103 = sub_30070B0((__int64)&v180);
            v60 = v146;
            v61 = v57;
            v67 = (unsigned __int64)v57;
            v68 = 205 - (!v103 - 1);
          }
          v24 = sub_340EC60((_QWORD *)a3, v68, (__int64)&v166, v60, v64, 0, v61, 1, v156, v67);
          if ( (unsigned int)v177 <= 0x40 )
            goto LABEL_22;
          v69 = v176;
          if ( !v176 )
            goto LABEL_22;
          goto LABEL_42;
        }
      }
      else
      {
        v176 = 0;
      }
      v176 |= v48;
      goto LABEL_38;
    }
    *(_QWORD *)&v117 = sub_34007B0(a3, (__int64)&v168, (__int64)&v166, v164, v165, 0, v12, 0);
    v160 = v117;
    v118 = v164;
    v119 = v150;
    v120 = v165;
    v121 = *(_QWORD *)(v150 + 48);
    v122 = *(_WORD *)(v121 + 16);
    v123 = *(_QWORD *)(v121 + 24);
    v174 = v122;
    v175 = v123;
    v124 = (unsigned __int64)v150;
    if ( v122 )
    {
      v125 = ((unsigned __int16)(v122 - 17) < 0xD4u) + 205;
    }
    else
    {
      v132 = sub_30070B0((__int64)&v174);
      v119 = v150;
      v124 = (unsigned __int64)v150;
      v125 = 205 - (!v132 - 1);
    }
    v24 = sub_340EC60((_QWORD *)a3, v125, (__int64)&v166, v118, v120, 0, v119, 1, v160, v124);
  }
  if ( v183 > 0x40 && v182 )
    j_j___libc_free_0_0(v182);
  if ( (unsigned int)v181 > 0x40 && v180 )
    j_j___libc_free_0_0(v180);
  if ( v179 > 0x40 && v178 )
    j_j___libc_free_0_0(v178);
  if ( (unsigned int)v177 > 0x40 && v176 )
    j_j___libc_free_0_0(v176);
  if ( v171 > 0x40 && v170 )
    j_j___libc_free_0_0(v170);
  if ( v169 > 0x40 )
  {
    v69 = v168;
    if ( v168 )
LABEL_42:
      j_j___libc_free_0_0(v69);
  }
LABEL_22:
  if ( v166 )
    sub_B91220((__int64)&v166, v166);
  return v24;
}
