// Function: sub_1F83660
// Address: 0x1f83660
//
__int64 *__fastcall sub_1F83660(
        _QWORD **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v11; // rsi
  _QWORD *v12; // r13
  char *v13; // rax
  char v14; // dl
  const void **v15; // rax
  unsigned __int64 v16; // r14
  __int64 (__fastcall *v17)(_QWORD *, __int64, __int64, unsigned __int64, const void **); // r15
  __int64 v18; // rax
  unsigned int v19; // eax
  char v20; // r13
  const void **v21; // rdx
  const void **v22; // rdx
  int v23; // ecx
  int v24; // r8d
  int v25; // r9d
  unsigned int v26; // ecx
  __int64 v27; // r13
  __int128 v28; // rdi
  __int64 v29; // rdx
  int v30; // ecx
  int v31; // r8d
  int v32; // r9d
  char v33; // r14
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // r13d
  __int64 v37; // r14
  __int64 v38; // r12
  unsigned __int8 *v39; // rax
  __int64 v40; // r13
  unsigned int v41; // r14d
  __int64 v42; // rax
  unsigned int v43; // eax
  const void **v44; // rdx
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // r14
  __int64 v50; // r13
  unsigned int v51; // edx
  unsigned __int64 v52; // r14
  __int128 v53; // rax
  __int64 *v54; // r12
  __int64 *v56; // rax
  __int64 v57; // rdi
  __int64 (*v58)(); // r8
  __int64 v59; // rdx
  int v60; // ecx
  int v61; // r8d
  int v62; // r9d
  __int64 v63; // rax
  __int64 v64; // rax
  unsigned int v65; // r13d
  __int64 v66; // r14
  _QWORD *v67; // rdi
  __int64 v68; // r8
  __int64 v69; // rcx
  __int64 v70; // rax
  __int64 *v71; // r15
  __int64 *v72; // r13
  __int64 v73; // rsi
  _QWORD *v74; // rdi
  __int64 v75; // rcx
  __int64 *v76; // r13
  __int64 *v77; // r14
  __int64 v78; // rsi
  __int64 v79; // rsi
  __int128 v80; // rax
  unsigned __int64 v81; // rdx
  __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // rdx
  __int128 v85; // rax
  __int128 v86; // rax
  __int64 *v87; // r13
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // rax
  __int64 v92; // rdx
  __int128 v93; // rax
  __int64 *v94; // r13
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rax
  __int64 v99; // rdx
  __int128 v100; // rax
  __int64 *v101; // rax
  __int64 *v102; // r13
  unsigned __int64 v103; // rcx
  __int64 *v104; // rsi
  unsigned __int64 v105; // r10
  __int16 *v106; // rdx
  __int16 *v107; // r11
  __int64 v108; // rax
  const void **v109; // r14
  char v110; // dl
  __int64 v111; // rax
  unsigned int v112; // esi
  unsigned int v113; // edx
  __int128 v114; // rax
  __int128 v115; // rax
  __int64 *v116; // r14
  __int64 v117; // rcx
  __int64 v118; // r8
  __int64 v119; // r9
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 *v122; // rax
  __int64 *v123; // r14
  __int64 *v124; // rcx
  unsigned __int64 v125; // r10
  unsigned int v126; // edx
  __int16 *v127; // r13
  __int64 v128; // rax
  unsigned __int64 v129; // r8
  char v130; // dl
  __int64 v131; // rax
  __int16 *v132; // r11
  const void **v133; // rbx
  unsigned int v134; // esi
  bool v135; // al
  bool v136; // al
  __int64 v137; // [rsp-20h] [rbp-160h]
  __int128 v138; // [rsp-10h] [rbp-150h]
  __int128 v139; // [rsp-10h] [rbp-150h]
  const void **v140; // [rsp+10h] [rbp-130h]
  __int128 v141; // [rsp+10h] [rbp-130h]
  unsigned int v142; // [rsp+20h] [rbp-120h]
  __int64 v143; // [rsp+20h] [rbp-120h]
  const void **v144; // [rsp+20h] [rbp-120h]
  __int128 v145; // [rsp+20h] [rbp-120h]
  unsigned __int64 v146; // [rsp+20h] [rbp-120h]
  unsigned __int64 v147; // [rsp+28h] [rbp-118h]
  char v148; // [rsp+30h] [rbp-110h]
  unsigned int v149; // [rsp+30h] [rbp-110h]
  __int128 v150; // [rsp+30h] [rbp-110h]
  __int128 v151; // [rsp+30h] [rbp-110h]
  __int128 v152; // [rsp+30h] [rbp-110h]
  __int16 *v153; // [rsp+38h] [rbp-108h]
  const void **v154; // [rsp+40h] [rbp-100h]
  __int64 v155; // [rsp+48h] [rbp-F8h]
  unsigned int v156; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v157; // [rsp+48h] [rbp-F8h]
  const void **v158; // [rsp+50h] [rbp-F0h]
  unsigned int v159; // [rsp+50h] [rbp-F0h]
  __int64 *v160; // [rsp+50h] [rbp-F0h]
  __int64 *v161; // [rsp+50h] [rbp-F0h]
  __int128 v162; // [rsp+50h] [rbp-F0h]
  __int64 *v163; // [rsp+50h] [rbp-F0h]
  __int128 v164; // [rsp+50h] [rbp-F0h]
  unsigned __int64 v165; // [rsp+58h] [rbp-E8h]
  __int64 v166; // [rsp+58h] [rbp-E8h]
  __int128 v167; // [rsp+60h] [rbp-E0h]
  __int128 v168; // [rsp+60h] [rbp-E0h]
  __int128 v169; // [rsp+70h] [rbp-D0h]
  unsigned __int64 v170; // [rsp+70h] [rbp-D0h]
  __int64 v171; // [rsp+A0h] [rbp-A0h] BYREF
  int v172; // [rsp+A8h] [rbp-98h]
  unsigned __int64 v173; // [rsp+B0h] [rbp-90h] BYREF
  const void **v174; // [rsp+B8h] [rbp-88h]
  __int64 *v175; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v176; // [rsp+C8h] [rbp-78h]
  _QWORD v177[14]; // [rsp+D0h] [rbp-70h] BYREF

  *(_QWORD *)&v167 = a2;
  v11 = *(_QWORD *)(a6 + 72);
  *((_QWORD *)&v167 + 1) = a3;
  *(_QWORD *)&v169 = a4;
  *((_QWORD *)&v169 + 1) = a5;
  v142 = a3;
  v171 = v11;
  if ( v11 )
    sub_1623A60((__int64)&v171, v11, 2);
  v12 = a1[1];
  v172 = *(_DWORD *)(a6 + 64);
  v13 = *(char **)(a6 + 40);
  v14 = *v13;
  v15 = (const void **)*((_QWORD *)v13 + 1);
  LOBYTE(v173) = v14;
  v16 = v173;
  v174 = v15;
  v17 = *(__int64 (__fastcall **)(_QWORD *, __int64, __int64, unsigned __int64, const void **))(*v12 + 264LL);
  v158 = v15;
  v155 = (*a1)[6];
  v18 = sub_1E0A0C0((*a1)[4]);
  v19 = v17(v12, v18, v155, v16, v158);
  v20 = v173;
  v154 = v21;
  v156 = v19;
  if ( (_BYTE)v173 )
  {
    if ( (unsigned __int8)(v173 - 14) <= 0x5Fu )
    {
      switch ( (char)v173 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v20 = 3;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v20 = 4;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v20 = 5;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v20 = 6;
          break;
        case 55:
          v20 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v20 = 8;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v20 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v20 = 10;
          break;
        default:
          v20 = 2;
          break;
      }
      goto LABEL_36;
    }
    goto LABEL_5;
  }
  if ( !sub_1F58D20((__int64)&v173) )
  {
LABEL_5:
    v22 = v174;
    goto LABEL_6;
  }
  v20 = sub_1F596B0((__int64)&v173);
LABEL_6:
  LOBYTE(v175) = v20;
  v176 = (__int64)v22;
  if ( v20 )
  {
LABEL_36:
    v159 = sub_1F6C8D0(v20);
    goto LABEL_8;
  }
  v159 = sub_1F58D40((__int64)&v175);
LABEL_8:
  v27 = sub_1D1ADA0(v169, DWORD2(v169), *((__int64 *)&v169 + 1), v23, v24, v25);
  if ( (*(_BYTE *)(a6 + 80) & 8) == 0 )
  {
    v177[1] = sub_1F6F9E0;
    v177[0] = sub_1F6BBE0;
    if ( v27 )
      v28 = (unsigned __int64)v27;
    else
      v28 = v169;
    v33 = sub_1D169E0((_QWORD *)v28, *((_QWORD **)&v28 + 1), (__int64)&v175, v26);
    if ( v177[0] )
      ((void (__fastcall *)(__int64 **, __int64 **, __int64))v177[0])(&v175, &v175, 3);
    if ( v33 )
    {
      v34 = sub_1D1ADA0(
              *(_QWORD *)(*(_QWORD *)(a6 + 32) + 40LL),
              *(_QWORD *)(*(_QWORD *)(a6 + 32) + 48LL),
              v29,
              v30,
              v31,
              v32);
      if ( !v34 )
        goto LABEL_17;
      v35 = *(_QWORD *)(v34 + 88);
      v36 = *(_DWORD *)(v35 + 32);
      v37 = v35 + 24;
      if ( v36 <= 0x40 )
      {
        if ( !*(_QWORD *)(v35 + 24) )
          goto LABEL_17;
      }
      else if ( v36 == (unsigned int)sub_16A57B0(v35 + 24) )
      {
        goto LABEL_17;
      }
      v74 = a1[1];
      v75 = (__int64)*a1;
      v175 = v177;
      v176 = 0x800000000LL;
      v54 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64 **))(*v74 + 1424LL))(
                         v74,
                         a6,
                         v37,
                         v75,
                         &v175);
      v76 = &v175[(unsigned int)v176];
      if ( v175 != v76 )
      {
        v77 = v175;
        do
        {
          v78 = *v77++;
          sub_1F81BC0((__int64)a1, v78);
        }
        while ( v76 != v77 );
        v76 = v175;
      }
      if ( v76 != v177 )
        _libc_free((unsigned __int64)v76);
      if ( v54 )
        goto LABEL_19;
LABEL_17:
      v38 = (__int64)a1[1];
      v39 = (unsigned __int8 *)(*(_QWORD *)(v167 + 40) + 16LL * v142);
      v40 = *((_QWORD *)v39 + 1);
      v41 = *v39;
      v148 = *((_BYTE *)a1 + 25);
      v42 = sub_1E0A0C0((*a1)[4]);
      v43 = sub_1F40B60(v38, v41, v40, v42, v148);
      v140 = v44;
      v149 = v43;
      v45 = sub_1D38BB0((__int64)*a1, v159, (__int64)&v171, v43, v44, 0, a7, a8, a9, 0);
      v147 = v46;
      v143 = v45;
      v47 = sub_1D309E0(
              *a1,
              128,
              (__int64)&v171,
              (unsigned int)v173,
              v174,
              0,
              *(double *)a7.m128i_i64,
              a8,
              *(double *)a9.m128i_i64,
              v169);
      v49 = v48;
      v50 = sub_1D323C0(*a1, v47, v48, (__int64)&v171, v149, v140, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64);
      v52 = v51 | v49 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v138 + 1) = v52;
      *(_QWORD *)&v138 = v50;
      v137 = v143;
      v144 = v140;
      *(_QWORD *)&v53 = sub_1D332F0(
                          *a1,
                          53,
                          (__int64)&v171,
                          v149,
                          v140,
                          0,
                          *(double *)a7.m128i_i64,
                          a8,
                          a9,
                          v137,
                          v147,
                          v138);
      v141 = v53;
      if ( (unsigned __int8)sub_1F70310(v53, DWORD2(v53), 0) )
      {
        v79 = v159 - 1;
        v160 = *a1;
        *(_QWORD *)&v80 = sub_1D38BB0((__int64)*a1, v79, (__int64)&v171, v149, v144, 0, a7, a8, a9, 0);
        v161 = sub_1D332F0(
                 v160,
                 123,
                 (__int64)&v171,
                 (unsigned int)v173,
                 v174,
                 0,
                 *(double *)a7.m128i_i64,
                 a8,
                 a9,
                 v167,
                 *((unsigned __int64 *)&v167 + 1),
                 v80);
        v165 = v81;
        sub_1F81BC0((__int64)a1, (__int64)v161);
        *(_QWORD *)&v162 = sub_1D332F0(
                             *a1,
                             124,
                             (__int64)&v171,
                             (unsigned int)v173,
                             v174,
                             0,
                             *(double *)a7.m128i_i64,
                             a8,
                             a9,
                             (__int64)v161,
                             v165,
                             v141);
        *((_QWORD *)&v162 + 1) = v82;
        sub_1F81BC0((__int64)a1, v162);
        *(_QWORD *)&v162 = sub_1D332F0(
                             *a1,
                             52,
                             (__int64)&v171,
                             (unsigned int)v173,
                             v174,
                             0,
                             *(double *)a7.m128i_i64,
                             a8,
                             a9,
                             v167,
                             *((unsigned __int64 *)&v167 + 1),
                             v162);
        *((_QWORD *)&v162 + 1) = v83;
        sub_1F81BC0((__int64)a1, v162);
        *((_QWORD *)&v139 + 1) = v52;
        *(_QWORD *)&v139 = v50;
        v163 = sub_1D332F0(
                 *a1,
                 123,
                 (__int64)&v171,
                 (unsigned int)v173,
                 v174,
                 0,
                 *(double *)a7.m128i_i64,
                 a8,
                 a9,
                 v162,
                 *((unsigned __int64 *)&v162 + 1),
                 v139);
        v166 = v84;
        sub_1F81BC0((__int64)a1, (__int64)v163);
        *(_QWORD *)&v85 = sub_1D38BB0((__int64)*a1, 1, (__int64)&v171, (unsigned int)v173, v174, 0, a7, a8, a9, 0);
        v150 = v85;
        *(_QWORD *)&v86 = sub_1D389D0((__int64)*a1, (__int64)&v171, (unsigned int)v173, v174, 0, 0, a7, a8, a9);
        v87 = *a1;
        v145 = v86;
        v91 = sub_1D28D50(*a1, 0x11u, *((__int64 *)&v86 + 1), v88, v89, v90);
        *(_QWORD *)&v93 = sub_1D3A900(
                            v87,
                            0x89u,
                            (__int64)&v171,
                            v156,
                            v154,
                            0,
                            (__m128)a7,
                            a8,
                            a9,
                            v169,
                            *((__int16 **)&v169 + 1),
                            v150,
                            v91,
                            v92);
        v94 = *a1;
        v151 = v93;
        v98 = sub_1D28D50(*a1, 0x11u, *((__int64 *)&v93 + 1), v95, v96, v97);
        *(_QWORD *)&v100 = sub_1D3A900(
                             v94,
                             0x89u,
                             (__int64)&v171,
                             v156,
                             v154,
                             0,
                             (__m128)a7,
                             a8,
                             a9,
                             v169,
                             *((__int16 **)&v169 + 1),
                             v145,
                             v98,
                             v99);
        v101 = sub_1D332F0(
                 *a1,
                 119,
                 (__int64)&v171,
                 v156,
                 v154,
                 0,
                 *(double *)a7.m128i_i64,
                 a8,
                 a9,
                 v151,
                 *((unsigned __int64 *)&v151 + 1),
                 v100);
        v102 = *a1;
        v103 = v173;
        v104 = v101;
        v105 = (unsigned __int64)v101;
        v107 = v106;
        v108 = v101[5] + 16LL * (unsigned int)v106;
        v109 = v174;
        v110 = *(_BYTE *)v108;
        v111 = *(_QWORD *)(v108 + 8);
        LOBYTE(v175) = v110;
        v176 = v111;
        if ( v110 )
        {
          v112 = ((unsigned __int8)(v110 - 14) < 0x60u) + 134;
        }
        else
        {
          v146 = v173;
          v153 = v107;
          v136 = sub_1F58D20((__int64)&v175);
          v103 = v146;
          v105 = (unsigned __int64)v104;
          v107 = v153;
          v112 = 134 - (!v136 - 1);
        }
        *(_QWORD *)&v164 = sub_1D3A900(
                             v102,
                             v112,
                             (__int64)&v171,
                             v103,
                             v109,
                             0,
                             (__m128)a7,
                             a8,
                             a9,
                             v105,
                             v107,
                             v167,
                             (__int64)v163,
                             v166);
        *((_QWORD *)&v164 + 1) = v113 | v166 & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v114 = sub_1D38BB0((__int64)*a1, 0, (__int64)&v171, (unsigned int)v173, v174, 0, a7, a8, a9, 0);
        v152 = v114;
        *(_QWORD *)&v115 = sub_1D332F0(
                             *a1,
                             53,
                             (__int64)&v171,
                             (unsigned int)v173,
                             v174,
                             0,
                             *(double *)a7.m128i_i64,
                             a8,
                             a9,
                             v114,
                             *((unsigned __int64 *)&v114 + 1),
                             v164);
        v116 = *a1;
        v168 = v115;
        v120 = sub_1D28D50(*a1, 0x14u, *((__int64 *)&v115 + 1), v117, v118, v119);
        v122 = sub_1D3A900(
                 v116,
                 0x89u,
                 (__int64)&v171,
                 v156,
                 v154,
                 0,
                 (__m128)a7,
                 a8,
                 a9,
                 v169,
                 *((__int16 **)&v169 + 1),
                 v152,
                 v120,
                 v121);
        v123 = *a1;
        v124 = v122;
        v125 = (unsigned __int64)v122;
        v127 = (__int16 *)v126;
        v128 = v122[5] + 16LL * v126;
        v129 = v173;
        v130 = *(_BYTE *)v128;
        v131 = *(_QWORD *)(v128 + 8);
        v132 = v127;
        v133 = v174;
        LOBYTE(v175) = v130;
        v176 = v131;
        if ( v130 )
        {
          v134 = ((unsigned __int8)(v130 - 14) < 0x60u) + 134;
        }
        else
        {
          v157 = v173;
          v170 = (unsigned __int64)v124;
          v135 = sub_1F58D20((__int64)&v175);
          v129 = v157;
          v125 = v170;
          v132 = v127;
          v134 = 134 - (!v135 - 1);
        }
        v54 = sub_1D3A900(
                v123,
                v134,
                (__int64)&v171,
                v129,
                v133,
                0,
                (__m128)a7,
                a8,
                a9,
                v125,
                v132,
                v168,
                v164,
                *((__int64 *)&v164 + 1));
        goto LABEL_19;
      }
      goto LABEL_18;
    }
  }
  v56 = (__int64 *)(*a1)[4];
  v57 = *v56;
  if ( !v27 )
    goto LABEL_18;
  v58 = *(__int64 (**)())(*a1[1] + 80LL);
  if ( v58 != sub_1F3C990 )
  {
    if ( ((unsigned __int8 (__fastcall *)(_QWORD *, _QWORD, _QWORD, _QWORD))v58)(
           a1[1],
           **(unsigned __int8 **)(a6 + 40),
           *(_QWORD *)(*(_QWORD *)(a6 + 40) + 8LL),
           *(_QWORD *)(*v56 + 112)) )
    {
      goto LABEL_18;
    }
    v57 = *(_QWORD *)(*a1)[4];
  }
  if ( (unsigned __int8)sub_1560180(v57 + 112, 17) )
    goto LABEL_18;
  v63 = sub_1D1ADA0(
          *(_QWORD *)(*(_QWORD *)(a6 + 32) + 40LL),
          *(_QWORD *)(*(_QWORD *)(a6 + 32) + 48LL),
          v59,
          v60,
          v61,
          v62);
  if ( !v63 )
    goto LABEL_18;
  v64 = *(_QWORD *)(v63 + 88);
  v65 = *(_DWORD *)(v64 + 32);
  v66 = v64 + 24;
  if ( v65 > 0x40 )
  {
    if ( v65 != (unsigned int)sub_16A57B0(v64 + 24) )
      goto LABEL_28;
LABEL_18:
    v54 = 0;
    goto LABEL_19;
  }
  if ( !*(_QWORD *)(v64 + 24) )
    goto LABEL_18;
LABEL_28:
  v67 = a1[1];
  v68 = *((unsigned __int8 *)a1 + 24);
  v69 = (__int64)*a1;
  v175 = v177;
  v176 = 0x800000000LL;
  v70 = sub_20B42D0(v67, a6, v66, v69, v68, &v175);
  v71 = v175;
  v54 = (__int64 *)v70;
  v72 = &v175[(unsigned int)v176];
  if ( v175 != v72 )
  {
    do
    {
      v73 = *v71++;
      sub_1F81BC0((__int64)a1, v73);
    }
    while ( v72 != v71 );
    v72 = v175;
  }
  if ( v72 != v177 )
    _libc_free((unsigned __int64)v72);
  if ( !v54 )
    goto LABEL_18;
LABEL_19:
  if ( v171 )
    sub_161E7C0((__int64)&v171, v171);
  return v54;
}
