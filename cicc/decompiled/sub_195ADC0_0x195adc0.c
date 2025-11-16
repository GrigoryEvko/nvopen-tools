// Function: sub_195ADC0
// Address: 0x195adc0
//
__int64 __fastcall sub_195ADC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // r9
  unsigned int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v24; // rdi
  _QWORD *v25; // rbx
  _QWORD *v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // rbx
  _QWORD *v31; // r14
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // r12
  __int64 v38; // r8
  int v39; // r9d
  __int64 v40; // r14
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 *v45; // r15
  __int64 v46; // rdx
  __int64 *v47; // rax
  __int64 v48; // rsi
  unsigned __int64 v49; // rdx
  __int64 v50; // rdx
  unsigned int v51; // eax
  __int64 v52; // rcx
  unsigned __int64 v53; // rsi
  char v54; // al
  __int64 v55; // r8
  __int64 v56; // r9
  _QWORD *v57; // rdx
  __int64 v58; // rax
  void *v59; // rcx
  __int64 v60; // rbx
  int v61; // eax
  __int64 v62; // rax
  int v63; // edx
  __int64 v64; // rdx
  _QWORD *v65; // rax
  __int64 v66; // rcx
  unsigned __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rcx
  double v71; // xmm4_8
  double v72; // xmm5_8
  __int64 *v73; // r12
  __int64 v74; // r13
  __int64 v75; // rax
  __int64 v76; // r14
  unsigned __int64 v77; // rsi
  char v78; // al
  __int64 v79; // r8
  __int64 v80; // r9
  _QWORD *v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rcx
  int v84; // eax
  __int64 v85; // rax
  int v86; // edx
  _QWORD *v87; // rbx
  _QWORD *v88; // r12
  __int64 v89; // rsi
  _QWORD *v90; // rbx
  _QWORD *v91; // r12
  __int64 v92; // rsi
  int v93; // eax
  __int64 v94; // rcx
  unsigned __int64 *v95; // r13
  __int64 *v96; // rcx
  int v97; // eax
  __int64 v98; // rcx
  unsigned __int64 *v99; // rbx
  __int64 *v100; // rcx
  _QWORD *v101; // [rsp+10h] [rbp-1C0h]
  _QWORD *v102; // [rsp+10h] [rbp-1C0h]
  _QWORD *v103; // [rsp+10h] [rbp-1C0h]
  __int64 v104; // [rsp+10h] [rbp-1C0h]
  __int64 v105; // [rsp+18h] [rbp-1B8h]
  __int64 v106; // [rsp+20h] [rbp-1B0h]
  __int64 v107; // [rsp+28h] [rbp-1A8h]
  _QWORD *v108; // [rsp+30h] [rbp-1A0h]
  bool v109; // [rsp+30h] [rbp-1A0h]
  _QWORD *v110; // [rsp+30h] [rbp-1A0h]
  _QWORD *v111; // [rsp+30h] [rbp-1A0h]
  __int64 v112; // [rsp+48h] [rbp-188h]
  __int64 v113; // [rsp+48h] [rbp-188h]
  __int64 v114; // [rsp+50h] [rbp-180h]
  __int64 v115; // [rsp+58h] [rbp-178h]
  __int64 v116; // [rsp+58h] [rbp-178h]
  __int64 v117; // [rsp+58h] [rbp-178h]
  __int64 v118; // [rsp+58h] [rbp-178h]
  bool v119; // [rsp+66h] [rbp-16Ah] BYREF
  char v120; // [rsp+67h] [rbp-169h]
  _QWORD *v121; // [rsp+68h] [rbp-168h] BYREF
  void *v122; // [rsp+70h] [rbp-160h] BYREF
  __int64 v123; // [rsp+78h] [rbp-158h] BYREF
  __int64 v124; // [rsp+80h] [rbp-150h]
  __int64 v125; // [rsp+88h] [rbp-148h]
  __int64 *v126; // [rsp+90h] [rbp-140h]
  __int64 v127; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v128; // [rsp+A8h] [rbp-128h]
  int v129; // [rsp+B0h] [rbp-120h]
  int v130; // [rsp+B4h] [rbp-11Ch]
  unsigned int v131; // [rsp+B8h] [rbp-118h]
  _QWORD *v132; // [rsp+C8h] [rbp-108h]
  unsigned int v133; // [rsp+D8h] [rbp-F8h]
  char v134; // [rsp+E0h] [rbp-F0h]
  char v135; // [rsp+E9h] [rbp-E7h]
  __int64 v136; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v137; // [rsp+F8h] [rbp-D8h]
  int v138; // [rsp+100h] [rbp-D0h]
  int v139; // [rsp+104h] [rbp-CCh]
  unsigned int v140; // [rsp+108h] [rbp-C8h]
  _QWORD *v141; // [rsp+118h] [rbp-B8h]
  unsigned int v142; // [rsp+128h] [rbp-A8h]
  char v143; // [rsp+130h] [rbp-A0h]
  char v144; // [rsp+139h] [rbp-97h]
  __int64 *v145; // [rsp+140h] [rbp-90h] BYREF
  __int64 v146; // [rsp+148h] [rbp-88h] BYREF
  __int64 v147; // [rsp+150h] [rbp-80h] BYREF
  unsigned __int64 v148; // [rsp+158h] [rbp-78h]
  __int64 v149; // [rsp+160h] [rbp-70h]
  unsigned __int64 v150; // [rsp+168h] [rbp-68h]
  __int64 v151; // [rsp+170h] [rbp-60h]
  __int64 v152; // [rsp+178h] [rbp-58h]
  __int64 v153; // [rsp+180h] [rbp-50h]
  unsigned __int64 v154; // [rsp+188h] [rbp-48h]
  __int64 v155; // [rsp+190h] [rbp-40h]
  unsigned __int64 v156; // [rsp+198h] [rbp-38h]

  v14 = *(_QWORD *)(a4 - 72);
  v15 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
  v112 = *(_QWORD *)(a4 - 24);
  v114 = *(_QWORD *)(a4 - 48);
  v16 = sub_157EB90(a2);
  v115 = sub_1632FA0(v16);
  sub_14BCF40(&v119, v14, v15, v115, 1u, 0);
  if ( !v120 || !v119 )
  {
    sub_14BCF40((bool *)&v145, v14, v15, v115, 0, 0);
    v20 = BYTE1(v145);
    if ( !BYTE1(v145) )
      return v20;
    v119 = (char)v145;
    if ( v120 )
    {
      v20 = v119;
      if ( !v119 )
        return v20;
    }
    else
    {
      v20 = v119;
      v120 = 1;
      if ( !v119 )
        return v20;
    }
    v24 = v112;
    v112 = v114;
    v114 = v24;
  }
  v127 = 0;
  v131 = 128;
  v128 = sub_22077B0(0x2000);
  sub_1954940((__int64)&v127);
  v134 = 0;
  v135 = 1;
  v136 = 0;
  v140 = 128;
  v137 = sub_22077B0(0x2000);
  sub_1954940((__int64)&v136);
  v17 = *(_QWORD *)(a3 + 40);
  v18 = *(_QWORD *)(a3 + 32);
  v143 = 0;
  v144 = 1;
  if ( v18 == v17 + 40 || !v18 )
    v19 = 0;
  else
    v19 = v18 - 24;
  v20 = 0;
  v116 = v19;
  if ( *(_DWORD *)(a1 + 256) < (unsigned int)sub_1951E40(a2, v19, *(_DWORD *)(a1 + 256)) )
  {
    if ( !v143 )
      goto LABEL_8;
    goto LABEL_112;
  }
  v107 = sub_1AB5340(a2, v114, v116, &v136, 0);
  v35 = sub_1AB5340(a2, v112, a3, &v127, 0);
  v149 = v107;
  v36 = *(_QWORD *)(a1 + 24);
  v146 = a2 | 4;
  v152 = a2 | 4;
  v105 = v35;
  v154 = v35 & 0xFFFFFFFFFFFFFFFBLL;
  v145 = (__int64 *)v114;
  v147 = v114;
  v150 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v151 = v112;
  v153 = v112;
  v156 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v148 = v107 & 0xFFFFFFFFFFFFFFFBLL;
  v155 = v35;
  sub_15CD9D0(v36, (__int64 *)&v145, 6);
  v37 = *(_QWORD *)(a2 + 48);
  v38 = a2;
  v39 = v116;
  v145 = &v147;
  v146 = 0x400000000LL;
  v40 = v116;
  while ( 1 )
  {
    v42 = v37 - 24;
    if ( !v37 )
      v42 = 0;
    if ( v42 == v40 )
      break;
    if ( *(_BYTE *)(v42 + 16) != 77 )
    {
      v41 = (unsigned int)v146;
      if ( (unsigned int)v146 >= HIDWORD(v146) )
      {
        v118 = v38;
        sub_16CD150((__int64)&v145, &v147, 0, 8, v38, v39);
        v41 = (unsigned int)v146;
        v38 = v118;
      }
      v145[v41] = v42;
      LODWORD(v146) = v146 + 1;
    }
    v37 = *(_QWORD *)(v37 + 8);
  }
  v43 = sub_157EE30(v38);
  v44 = v43 - 24;
  if ( !v43 )
    v44 = 0;
  v106 = v44;
  v113 = (__int64)v145;
  v45 = &v145[(unsigned int)v146];
  if ( v145 != v45 )
  {
    while ( 1 )
    {
      v73 = (__int64 *)*(v45 - 1);
      if ( v73[1] )
        break;
LABEL_97:
      --v45;
      sub_15F20C0(v73);
      if ( (__int64 *)v113 == v45 )
      {
        v45 = v145;
        goto LABEL_158;
      }
    }
    LOWORD(v124) = 257;
    v74 = *v73;
    v75 = sub_1648B60(64);
    v76 = v75;
    if ( v75 )
    {
      v117 = v75;
      sub_15F1EA0(v75, v74, 53, 0, 0, 0);
      *(_DWORD *)(v76 + 56) = 2;
      sub_164B780(v76, (__int64 *)&v122);
      sub_1648880(v76, *(_DWORD *)(v76 + 56), 1);
    }
    else
    {
      v117 = 0;
    }
    v125 = (__int64)v73;
    v123 = 2;
    v124 = 0;
    v109 = v73 + 1 != 0 && v73 + 2 != 0;
    if ( v109 )
      sub_164C220((__int64)&v123);
    v77 = (unsigned __int64)&v122;
    v122 = &unk_49E6B50;
    v126 = &v127;
    v78 = sub_12E4800((__int64)&v127, (__int64)&v122, &v121);
    v81 = v121;
    if ( v78 )
    {
      v82 = v125;
LABEL_105:
      LOBYTE(v77) = v82 != 0;
      v122 = &unk_49EE2B0;
      if ( v82 != -8 && v82 != 0 && v82 != -16 )
      {
        v101 = v81;
        sub_1649B30(&v123);
        v81 = v101;
      }
      v83 = v81[7];
      v84 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
      if ( v84 == *(_DWORD *)(v76 + 56) )
      {
        v104 = v81[7];
        sub_15F55D0(v76, v77, (__int64)v81, v83, v79, v80);
        v83 = v104;
        v84 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
      }
      v85 = (v84 + 1) & 0xFFFFFFF;
      v86 = v85 | *(_DWORD *)(v76 + 20) & 0xF0000000;
      *(_DWORD *)(v76 + 20) = v86;
      if ( (v86 & 0x40000000) != 0 )
        v46 = *(_QWORD *)(v76 - 8);
      else
        v46 = v117 - 24 * v85;
      v47 = (__int64 *)(v46 + 24LL * (unsigned int)(v85 - 1));
      if ( *v47 )
      {
        v48 = v47[1];
        v49 = v47[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v49 = v48;
        if ( v48 )
          *(_QWORD *)(v48 + 16) = *(_QWORD *)(v48 + 16) & 3LL | v49;
      }
      *v47 = v83;
      if ( v83 )
      {
        v50 = *(_QWORD *)(v83 + 8);
        v47[1] = v50;
        if ( v50 )
          *(_QWORD *)(v50 + 16) = (unsigned __int64)(v47 + 1) | *(_QWORD *)(v50 + 16) & 3LL;
        v47[2] = (v83 + 8) | v47[2] & 3;
        *(_QWORD *)(v83 + 8) = v47;
      }
      v51 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v76 + 23) & 0x40) != 0 )
        v52 = *(_QWORD *)(v76 - 8);
      else
        v52 = v117 - 24LL * v51;
      *(_QWORD *)(v52 + 8LL * (v51 - 1) + 24LL * *(unsigned int *)(v76 + 56) + 8) = v105;
      v123 = 2;
      v124 = 0;
      v125 = (__int64)v73;
      if ( v109 )
        sub_164C220((__int64)&v123);
      v53 = (unsigned __int64)&v122;
      v122 = &unk_49E6B50;
      v126 = &v136;
      v54 = sub_12E4800((__int64)&v136, (__int64)&v122, &v121);
      v57 = v121;
      if ( v54 )
      {
        v58 = v125;
        goto LABEL_80;
      }
      v53 = v140;
      ++v136;
      v93 = v138 + 1;
      v55 = 2 * v140;
      if ( 4 * (v138 + 1) >= 3 * v140 )
      {
        LODWORD(v53) = 2 * v140;
      }
      else if ( v140 - v139 - v93 > v140 >> 3 )
      {
        goto LABEL_135;
      }
      sub_12E48B0((__int64)&v136, v53);
      v53 = (unsigned __int64)&v122;
      sub_12E4800((__int64)&v136, (__int64)&v122, &v121);
      v57 = v121;
      v93 = v138 + 1;
LABEL_135:
      v94 = v57[3];
      v138 = v93;
      v95 = v57 + 1;
      v58 = v125;
      if ( v94 == -8 )
      {
        if ( v125 != -8 )
        {
LABEL_140:
          v57[3] = v58;
          if ( v58 == -8 || v58 == 0 || v58 == -16 )
          {
            v58 = v125;
          }
          else
          {
            v111 = v57;
            v53 = v123 & 0xFFFFFFFFFFFFFFF8LL;
            sub_1649AC0(v95, v123 & 0xFFFFFFFFFFFFFFF8LL);
            v58 = v125;
            v57 = v111;
          }
        }
      }
      else
      {
        --v139;
        if ( v94 != v125 )
        {
          if ( v94 && v94 != -16 )
          {
            v110 = v57;
            sub_1649B30(v57 + 1);
            v58 = v125;
            v57 = v110;
          }
          goto LABEL_140;
        }
      }
      v96 = v126;
      v57[5] = 6;
      v57[6] = 0;
      v57[4] = v96;
      v57[7] = 0;
LABEL_80:
      v59 = &unk_49EE2B0;
      LOBYTE(v53) = v58 != 0;
      v122 = &unk_49EE2B0;
      LOBYTE(v59) = v58 != -8;
      if ( ((unsigned __int8)v59 & (v58 != 0)) != 0 && v58 != -16 )
      {
        v108 = v57;
        sub_1649B30(&v123);
        v57 = v108;
      }
      v60 = v57[7];
      v61 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
      if ( v61 == *(_DWORD *)(v76 + 56) )
      {
        sub_15F55D0(v76, v53, (__int64)v57, (__int64)v59, v55, v56);
        v61 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
      }
      v62 = (v61 + 1) & 0xFFFFFFF;
      v63 = v62 | *(_DWORD *)(v76 + 20) & 0xF0000000;
      *(_DWORD *)(v76 + 20) = v63;
      if ( (v63 & 0x40000000) != 0 )
        v64 = *(_QWORD *)(v76 - 8);
      else
        v64 = v117 - 24 * v62;
      v65 = (_QWORD *)(v64 + 24LL * (unsigned int)(v62 - 1));
      if ( *v65 )
      {
        v66 = v65[1];
        v67 = v65[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v67 = v66;
        if ( v66 )
          *(_QWORD *)(v66 + 16) = *(_QWORD *)(v66 + 16) & 3LL | v67;
      }
      *v65 = v60;
      if ( v60 )
      {
        v68 = *(_QWORD *)(v60 + 8);
        v65[1] = v68;
        if ( v68 )
          *(_QWORD *)(v68 + 16) = (unsigned __int64)(v65 + 1) | *(_QWORD *)(v68 + 16) & 3LL;
        v65[2] = (v60 + 8) | v65[2] & 3LL;
        *(_QWORD *)(v60 + 8) = v65;
      }
      v69 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v76 + 23) & 0x40) != 0 )
        v70 = *(_QWORD *)(v76 - 8);
      else
        v70 = v117 - 24 * v69;
      *(_QWORD *)(v70 + 8LL * (unsigned int)(v69 - 1) + 24LL * *(unsigned int *)(v76 + 56) + 8) = v107;
      sub_15F2120(v117, v106);
      sub_164D160((__int64)v73, v76, a5, a6, a7, a8, v71, v72, a11, a12);
      goto LABEL_97;
    }
    v77 = v131;
    ++v127;
    v97 = v129 + 1;
    v79 = 2 * v131;
    if ( 4 * (v129 + 1) >= 3 * v131 )
    {
      LODWORD(v77) = 2 * v131;
    }
    else if ( v131 - v130 - v97 > v131 >> 3 )
    {
      goto LABEL_147;
    }
    sub_12E48B0((__int64)&v127, v77);
    v77 = (unsigned __int64)&v122;
    sub_12E4800((__int64)&v127, (__int64)&v122, &v121);
    v81 = v121;
    v97 = v129 + 1;
LABEL_147:
    v98 = v81[3];
    v129 = v97;
    v99 = v81 + 1;
    v82 = v125;
    if ( v98 == -8 )
    {
      if ( v125 != -8 )
      {
LABEL_152:
        v81[3] = v82;
        if ( v82 == 0 || v82 == -8 || v82 == -16 )
        {
          v82 = v125;
        }
        else
        {
          v103 = v81;
          v77 = v123 & 0xFFFFFFFFFFFFFFF8LL;
          sub_1649AC0(v99, v123 & 0xFFFFFFFFFFFFFFF8LL);
          v82 = v125;
          v81 = v103;
        }
      }
    }
    else
    {
      --v130;
      if ( v98 != v125 )
      {
        if ( v98 && v98 != -16 )
        {
          v102 = v81;
          sub_1649B30(v81 + 1);
          v82 = v125;
          v81 = v102;
        }
        goto LABEL_152;
      }
    }
    v100 = v126;
    v81[5] = 6;
    v81[6] = 0;
    v81[4] = v100;
    v81[7] = 0;
    goto LABEL_105;
  }
LABEL_158:
  if ( v45 != &v147 )
    _libc_free((unsigned __int64)v45);
  v20 = 1;
  if ( !v143 )
  {
LABEL_8:
    v21 = v140;
    if ( !v140 )
      goto LABEL_9;
    goto LABEL_35;
  }
LABEL_112:
  if ( v142 )
  {
    v87 = v141;
    v88 = &v141[2 * v142];
    do
    {
      if ( *v87 != -4 && *v87 != -8 )
      {
        v89 = v87[1];
        if ( v89 )
          sub_161E7C0((__int64)(v87 + 1), v89);
      }
      v87 += 2;
    }
    while ( v88 != v87 );
  }
  j___libc_free_0(v141);
  v21 = v140;
  if ( v140 )
  {
LABEL_35:
    v30 = (_QWORD *)v137;
    v123 = 2;
    v124 = 0;
    v31 = (_QWORD *)(v137 + (v21 << 6));
    v125 = -8;
    v32 = -8;
    v122 = &unk_49E6B50;
    v126 = 0;
    v146 = 2;
    v147 = 0;
    v148 = -16;
    v145 = (__int64 *)&unk_49E6B50;
    v149 = 0;
    while ( 1 )
    {
      v33 = v30[3];
      if ( v33 != v32 )
      {
        v32 = v148;
        if ( v33 != v148 )
        {
          v34 = v30[7];
          if ( v34 != -8 && v34 != 0 && v34 != -16 )
          {
            sub_1649B30(v30 + 5);
            v33 = v30[3];
          }
          v32 = v33;
        }
      }
      *v30 = &unk_49EE2B0;
      if ( v32 != -8 && v32 != 0 && v32 != -16 )
        sub_1649B30(v30 + 1);
      v30 += 8;
      if ( v31 == v30 )
        break;
      v32 = v125;
    }
    v145 = (__int64 *)&unk_49EE2B0;
    if ( v148 != -8 && v148 != 0 && v148 != -16 )
      sub_1649B30(&v146);
    v122 = &unk_49EE2B0;
    if ( v125 != -8 && v125 != 0 && v125 != -16 )
      sub_1649B30(&v123);
  }
LABEL_9:
  j___libc_free_0(v137);
  if ( v134 )
  {
    if ( v133 )
    {
      v90 = v132;
      v91 = &v132[2 * v133];
      do
      {
        if ( *v90 != -4 && *v90 != -8 )
        {
          v92 = v90[1];
          if ( v92 )
            sub_161E7C0((__int64)(v90 + 1), v92);
        }
        v90 += 2;
      }
      while ( v91 != v90 );
    }
    j___libc_free_0(v132);
    v22 = v131;
    if ( !v131 )
      goto LABEL_11;
LABEL_17:
    v25 = (_QWORD *)v128;
    v123 = 2;
    v124 = 0;
    v26 = (_QWORD *)(v128 + (v22 << 6));
    v125 = -8;
    v27 = -8;
    v122 = &unk_49E6B50;
    v126 = 0;
    v146 = 2;
    v147 = 0;
    v148 = -16;
    v145 = (__int64 *)&unk_49E6B50;
    v149 = 0;
    while ( 1 )
    {
      v28 = v25[3];
      if ( v28 != v27 )
      {
        v27 = v148;
        if ( v28 != v148 )
        {
          v29 = v25[7];
          if ( v29 != -8 && v29 != 0 && v29 != -16 )
          {
            sub_1649B30(v25 + 5);
            v28 = v25[3];
          }
          v27 = v28;
        }
      }
      *v25 = &unk_49EE2B0;
      if ( v27 != 0 && v27 != -8 && v27 != -16 )
        sub_1649B30(v25 + 1);
      v25 += 8;
      if ( v26 == v25 )
        break;
      v27 = v125;
    }
    v145 = (__int64 *)&unk_49EE2B0;
    if ( v148 != -8 && v148 != 0 && v148 != -16 )
      sub_1649B30(&v146);
    v122 = &unk_49EE2B0;
    if ( v125 != -8 && v125 != 0 && v125 != -16 )
    {
      sub_1649B30(&v123);
      j___libc_free_0(v128);
      return v20;
    }
    goto LABEL_11;
  }
  v22 = v131;
  if ( v131 )
    goto LABEL_17;
LABEL_11:
  j___libc_free_0(v128);
  return v20;
}
