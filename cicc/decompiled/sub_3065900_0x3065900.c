// Function: sub_3065900
// Address: 0x3065900
//
signed __int64 __fastcall sub_3065900(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        int a6,
        unsigned __int8 *a7)
{
  __int64 v8; // r14
  unsigned int v9; // eax
  __int64 v10; // r9
  unsigned int v11; // ebx
  unsigned __int8 *v12; // rdi
  unsigned __int8 *v13; // rsi
  __int64 v14; // r9
  __int64 v15; // r12
  __int64 v16; // rcx
  unsigned __int16 v17; // bx
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // r12
  __int64 v22; // rcx
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // r15
  unsigned int i; // r13d
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rdx
  __int64 v30; // r15
  unsigned __int16 v31; // bx
  int v32; // r11d
  unsigned int v33; // r9d
  unsigned __int8 v34; // r10
  int v35; // r12d
  __int64 (*v36)(); // rax
  unsigned int v37; // eax
  unsigned __int64 v38; // r12
  bool v39; // cc
  __int64 v40; // rax
  unsigned int v41; // eax
  __int64 v42; // rdx
  char v43; // bl
  unsigned int v44; // eax
  __int64 v45; // rdx
  unsigned int v46; // r10d
  int v47; // edx
  signed __int64 v48; // rcx
  unsigned __int64 v49; // r10
  unsigned __int64 v50; // r12
  signed __int64 result; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rdx
  unsigned int v56; // eax
  __int64 v57; // r9
  unsigned int v58; // ebx
  unsigned __int8 *v59; // rdi
  unsigned __int8 *v60; // rsi
  __int64 v61; // r9
  unsigned int v62; // eax
  unsigned __int64 v63; // r14
  unsigned __int64 v64; // r12
  __int64 v65; // rax
  unsigned __int8 *v66; // rdi
  __int64 v67; // rsi
  __int64 (*v68)(); // rax
  __int64 *v69; // r12
  __int64 (*v70)(); // rbx
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  unsigned int v75; // eax
  char v76; // al
  __int64 v77; // rax
  __int64 (*v78)(); // rax
  unsigned __int16 v79; // bx
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  unsigned __int16 v83; // ax
  __int64 v84; // rdx
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 *v87; // r12
  int v88; // edx
  __int64 (*v89)(); // rbx
  __int64 v90; // rdx
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  unsigned int v94; // eax
  __int64 v95; // rdx
  __int64 v96; // rax
  unsigned int v97; // eax
  char v98; // al
  __int64 *v99; // rdi
  int v100; // r13d
  __int64 *v101; // rdi
  int v102; // eax
  bool v103; // r12
  signed __int64 v104; // rax
  int v105; // edx
  __int64 v106; // rsi
  bool v107; // of
  __int64 v108; // r12
  __int64 (*v109)(); // rcx
  unsigned __int8 *v110; // rsi
  __int64 (*v111)(); // rax
  unsigned int v112; // eax
  unsigned __int8 *v113; // rsi
  char v114; // al
  unsigned int v115; // eax
  unsigned __int64 v116; // rdx
  unsigned __int64 v117; // rax
  __int64 v118; // r10
  unsigned __int64 v119; // r12
  __int64 v120; // rax
  char v121; // al
  char v122; // al
  char v123; // al
  char v124; // al
  char v125; // al
  unsigned int v126; // [rsp+0h] [rbp-D0h]
  unsigned int v127; // [rsp+0h] [rbp-D0h]
  __int64 v128; // [rsp+8h] [rbp-C8h]
  __int64 v129; // [rsp+8h] [rbp-C8h]
  unsigned int v130; // [rsp+8h] [rbp-C8h]
  unsigned int v131; // [rsp+14h] [rbp-BCh]
  unsigned int v132; // [rsp+14h] [rbp-BCh]
  unsigned int v133; // [rsp+14h] [rbp-BCh]
  unsigned int v134; // [rsp+14h] [rbp-BCh]
  unsigned int v135; // [rsp+14h] [rbp-BCh]
  unsigned int v136; // [rsp+14h] [rbp-BCh]
  unsigned int v137; // [rsp+14h] [rbp-BCh]
  int v138; // [rsp+14h] [rbp-BCh]
  unsigned int v139; // [rsp+14h] [rbp-BCh]
  unsigned int v140; // [rsp+14h] [rbp-BCh]
  char v141; // [rsp+1Bh] [rbp-B5h]
  unsigned int v142; // [rsp+1Ch] [rbp-B4h]
  __int64 v145; // [rsp+28h] [rbp-A8h]
  unsigned __int16 v146; // [rsp+30h] [rbp-A0h]
  __int64 v147; // [rsp+30h] [rbp-A0h]
  __int64 v148; // [rsp+38h] [rbp-98h]
  int v149; // [rsp+38h] [rbp-98h]
  int v150; // [rsp+38h] [rbp-98h]
  int v151; // [rsp+38h] [rbp-98h]
  int v152; // [rsp+38h] [rbp-98h]
  int v153; // [rsp+38h] [rbp-98h]
  int v154; // [rsp+38h] [rbp-98h]
  int v155; // [rsp+38h] [rbp-98h]
  unsigned __int8 v156; // [rsp+38h] [rbp-98h]
  int v157; // [rsp+38h] [rbp-98h]
  int v158; // [rsp+38h] [rbp-98h]
  int v159; // [rsp+40h] [rbp-90h]
  char v160; // [rsp+40h] [rbp-90h]
  int v161; // [rsp+44h] [rbp-8Ch]
  __int64 v162; // [rsp+48h] [rbp-88h]
  signed __int64 v163; // [rsp+50h] [rbp-80h]
  signed __int64 v164; // [rsp+58h] [rbp-78h]
  unsigned int v165; // [rsp+58h] [rbp-78h]
  char v166; // [rsp+58h] [rbp-78h]
  __int64 v168; // [rsp+68h] [rbp-68h]
  unsigned int v169; // [rsp+68h] [rbp-68h]
  signed __int64 v170; // [rsp+68h] [rbp-68h]
  unsigned __int64 v171; // [rsp+68h] [rbp-68h]
  unsigned __int64 v172; // [rsp+68h] [rbp-68h]
  __int64 v173; // [rsp+78h] [rbp-58h] BYREF
  __int64 v174; // [rsp+80h] [rbp-50h] BYREF
  __int64 v175; // [rsp+88h] [rbp-48h]
  __int64 v176; // [rsp+90h] [rbp-40h]

  v8 = a3;
  v168 = a4;
  if ( a2 == 48 )
  {
    v56 = sub_BCB060(a4);
    v57 = *(_QWORD *)(a1 + 8);
    v58 = v56;
    v174 = v56;
    v59 = *(unsigned __int8 **)(v57 + 32);
    v60 = &v59[*(_QWORD *)(v57 + 40)];
    if ( v60 != sub_305CF30(v59, (__int64)v60, &v174) && v58 <= (unsigned int)sub_AE43A0(v61, v8) )
      return 0;
  }
  else if ( a2 > 0x30 )
  {
    if ( a2 == 49 && (a3 == a4 || *(_BYTE *)(a3 + 8) == 14 && *(_BYTE *)(a4 + 8) == 14) )
      return 0;
  }
  else if ( a2 == 38 )
  {
    v54 = sub_9208B0(*(_QWORD *)(a1 + 8), a3);
    v175 = v55;
    v174 = v54;
    if ( !(_BYTE)v55 )
    {
      v65 = *(_QWORD *)(a1 + 8);
      v66 = *(unsigned __int8 **)(v65 + 32);
      v67 = *(_QWORD *)(v65 + 40);
      v173 = v174;
      if ( &v66[v67] != sub_305CF30(v66, (__int64)&v66[v67], &v173) )
        return 0;
    }
  }
  else if ( a2 == 47 )
  {
    v9 = sub_BCB060(a3);
    v10 = *(_QWORD *)(a1 + 8);
    v11 = v9;
    v174 = v9;
    v12 = *(unsigned __int8 **)(v10 + 32);
    v13 = &v12[*(_QWORD *)(v10 + 40)];
    if ( v13 != sub_305CF30(v12, (__int64)v13, &v174) && v11 >= (unsigned int)sub_AE43A0(v14, v168) )
      return 0;
  }
  v162 = *(_QWORD *)(a1 + 24);
  v142 = sub_2FEBEF0(v162, a2);
  v15 = *(_QWORD *)v168;
  v163 = 1;
  v16 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)v168, 0);
  v17 = v16;
  v18 = v15;
  v20 = v19;
  while ( 1 )
  {
    LOWORD(v16) = v17;
    sub_2FE6CC0((__int64)&v174, *(_QWORD *)(a1 + 24), v18, v16, v20);
    if ( (_BYTE)v174 == 10 )
      break;
    if ( !(_BYTE)v174 )
    {
      v161 = 0;
      v159 = v17;
      goto LABEL_14;
    }
    if ( (v174 & 0xFB) == 2 )
    {
      v53 = 2 * v163;
      if ( !is_mul_ok(2u, v163) )
      {
        v53 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v163 <= 0 )
          v53 = 0x8000000000000000LL;
      }
      v163 = v53;
    }
    if ( (_WORD)v175 == v17 )
    {
      if ( v17 )
      {
        v159 = v17;
LABEL_13:
        v161 = 0;
        goto LABEL_14;
      }
      if ( v20 == v176 )
      {
        v159 = 0;
        goto LABEL_13;
      }
    }
    v16 = v175;
    v20 = v176;
    v17 = v175;
  }
  if ( v17 )
  {
    v159 = v17;
  }
  else
  {
    v159 = 8;
    v17 = 8;
  }
  v161 = 1;
  v163 = 0;
LABEL_14:
  v164 = 1;
  v21 = *(_QWORD *)v8;
  v146 = v17;
  v22 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)v8, 0);
  v23 = a1;
  v25 = v24;
  for ( i = v22; ; i = (unsigned __int16)v175 )
  {
    LOWORD(v22) = i;
    sub_2FE6CC0((__int64)&v174, *(_QWORD *)(v23 + 24), v21, v22, v25);
    v29 = (unsigned __int16)v175;
    if ( (_BYTE)v174 == 10 )
      break;
    if ( !(_BYTE)v174 )
    {
      v30 = v23;
      v31 = v146;
      v29 = i;
      v32 = 0;
      goto LABEL_21;
    }
    if ( (v174 & 0xFB) == 2 )
    {
      v52 = 2 * v164;
      if ( !is_mul_ok(2u, v164) )
      {
        v52 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v164 <= 0 )
          v52 = 0x8000000000000000LL;
      }
      v164 = v52;
    }
    if ( (_WORD)i == (_WORD)v175 && ((_WORD)v175 || v25 == v176) )
    {
      v30 = v23;
      v31 = v146;
      v32 = 0;
      goto LABEL_21;
    }
    v22 = v175;
    v25 = v176;
  }
  v29 = 8;
  v30 = v23;
  v164 = 0;
  v31 = v146;
  v32 = 1;
  if ( (_WORD)i )
    v29 = i;
LABEL_21:
  if ( v31 <= 1u )
    goto LABEL_239;
  if ( (unsigned __int16)(v31 - 504) <= 7u )
    goto LABEL_239;
  v33 = (unsigned __int16)v29;
  v145 = *(_QWORD *)&byte_444C4A0[16 * v159 - 16];
  v141 = byte_444C4A0[16 * v159 - 8];
  if ( (unsigned __int16)v29 <= 1u || (unsigned __int16)(v29 - 504) <= 7u )
    goto LABEL_239;
  v34 = *(_BYTE *)(v8 + 8);
  v160 = byte_444C4A0[16 * (unsigned __int16)v29 - 8];
  v147 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v29 - 16];
  v35 = *(unsigned __int8 *)(v168 + 8);
  switch ( a2 )
  {
    case '&':
      v36 = *(__int64 (**)())(*(_QWORD *)v162 + 1392LL);
      if ( v36 == sub_2FE3480 )
        goto LABEL_27;
      v130 = (unsigned __int16)v29;
      v138 = v32;
      v156 = *(_BYTE *)(v8 + 8);
      v122 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v36)(
               v162,
               v31,
               0,
               (unsigned __int16)v29,
               0);
      v32 = v138;
      v33 = v130;
      v34 = v156;
      if ( !v122 )
        goto LABEL_27;
      return 0;
    case '\'':
      v27 = (unsigned __int16)v29;
      v68 = *(__int64 (**)())(*(_QWORD *)v162 + 1432LL);
      if ( v68 == sub_2FE34A0 )
        goto LABEL_101;
      v137 = (unsigned __int16)v29;
      v155 = v32;
      v121 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v68)(
               v162,
               v31,
               0,
               (unsigned __int16)v29,
               0);
      v32 = v155;
      v33 = v137;
      if ( !v121 )
        goto LABEL_101;
      return 0;
    case '(':
LABEL_101:
      if ( !a7 )
        goto LABEL_110;
      v69 = *(__int64 **)(v30 + 24);
      v29 = *a7;
      switch ( (_DWORD)v29 )
      {
        case 'E':
          break;
        case 'K':
          v70 = *(__int64 (**)())(*v69 + 1560);
          if ( (a7[7] & 0x40) != 0 )
            v71 = *((_QWORD *)a7 - 1);
          else
            v71 = (__int64)&a7[-32 * (*((_DWORD *)a7 + 1) & 0x7FFFFFF)];
          v131 = v33;
          v149 = v32;
          v126 = sub_30097B0(*(_QWORD *)(*(_QWORD *)v71 + 8LL), 0, v71, v27, v28);
          v128 = v72;
          v75 = sub_30097B0(*((_QWORD *)a7 + 1), 0, v72, v73, v74);
          v32 = v149;
          v33 = v131;
          if ( v70 != sub_2D566A0 )
          {
            v76 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64))v70)(v69, v75, v29, v126, v128);
            v32 = v149;
            v33 = v131;
            if ( v76 )
              return 0;
          }
          break;
        case 'D':
          v77 = *v69;
          v27 = *(_QWORD *)(*v69 + 1424);
          if ( (a7[7] & 0x40) != 0 )
          {
            v113 = (unsigned __int8 *)*((_QWORD *)a7 - 1);
          }
          else
          {
            v29 = 32LL * (*((_DWORD *)a7 + 1) & 0x7FFFFFF);
            v113 = &a7[-v29];
          }
          if ( (__int64 (*)())v27 == sub_2D56670 )
          {
LABEL_109:
            v78 = *(__int64 (**)())(v77 + 1816);
            if ( v78 != sub_2D566C0 )
            {
              v139 = v33;
              v157 = v32;
              v123 = ((__int64 (__fastcall *)(__int64 *, unsigned __int8 *, __int64, __int64))v78)(v69, a7, v29, v27);
              v32 = v157;
              v33 = v139;
              if ( v123 )
                return 0;
            }
LABEL_110:
            if ( a5 != 1 )
              goto LABEL_28;
            v132 = v33;
            v150 = v32;
            v79 = sub_30097B0(v8, 0, v29, v27, v28);
            v83 = sub_30097B0(v168, 0, v80, v81, v82);
            v32 = v150;
            v33 = v132;
            if ( v161 != v150
              || v164 != v163
              || !v83
              || !v79
              || (((int)*(unsigned __int16 *)(v162 + 2 * (v83 + 274LL * v79 + 71704) + 6) >> (4 * ((a2 == 39) + 2)))
                & 0xF) != 0 )
            {
              goto LABEL_28;
            }
            return 0;
          }
          v136 = v33;
          v154 = v32;
          v114 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD))v27)(
                   v69,
                   *(_QWORD *)(*(_QWORD *)v113 + 8LL),
                   *((_QWORD *)a7 + 1));
          v32 = v154;
          v33 = v136;
          if ( v114 )
            return 0;
          break;
        default:
          goto LABEL_239;
      }
      v77 = *v69;
      goto LABEL_109;
    case '.':
      if ( !a7 )
        goto LABEL_29;
      v87 = *(__int64 **)(v30 + 24);
      v88 = *a7;
      if ( v88 == 69 )
        goto LABEL_129;
      if ( v88 == 75 )
      {
        v89 = *(__int64 (**)())(*v87 + 1560);
        if ( (a7[7] & 0x40) != 0 )
          v90 = *((_QWORD *)a7 - 1);
        else
          v90 = (__int64)&a7[-32 * (*((_DWORD *)a7 + 1) & 0x7FFFFFF)];
        v134 = v33;
        v152 = v32;
        v127 = sub_30097B0(*(_QWORD *)(*(_QWORD *)v90 + 8LL), 0, v90, v27, v28);
        v129 = v91;
        v94 = sub_30097B0(*((_QWORD *)a7 + 1), 0, v91, v92, v93);
        v32 = v152;
        v33 = v134;
        if ( v89 != sub_2D566A0 )
        {
          v125 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64))v89)(v87, v94, v95, v127, v129);
          v32 = v152;
          v33 = v134;
          if ( v125 )
            return 0;
        }
LABEL_129:
        v96 = *v87;
        goto LABEL_143;
      }
      if ( v88 != 68 )
        goto LABEL_239;
      v96 = *v87;
      v109 = *(__int64 (**)())(*v87 + 1424);
      if ( (a7[7] & 0x40) != 0 )
        v110 = (unsigned __int8 *)*((_QWORD *)a7 - 1);
      else
        v110 = &a7[-32 * (*((_DWORD *)a7 + 1) & 0x7FFFFFF)];
      if ( v109 != sub_2D56670 )
      {
        v140 = v33;
        v158 = v32;
        v124 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD))v109)(
                 v87,
                 *(_QWORD *)(*(_QWORD *)v110 + 8LL),
                 *((_QWORD *)a7 + 1));
        v32 = v158;
        v33 = v140;
        if ( v124 )
          return 0;
        goto LABEL_129;
      }
LABEL_143:
      v111 = *(__int64 (**)())(v96 + 1816);
      if ( v111 == sub_2D566C0 )
      {
LABEL_28:
        v34 = *(_BYTE *)(v8 + 8);
        v35 = *(unsigned __int8 *)(v168 + 8);
        goto LABEL_29;
      }
      v135 = v33;
      v153 = v32;
      if ( ((unsigned __int8 (__fastcall *)(__int64 *, unsigned __int8 *))v111)(v87, a7) )
        return 0;
      v34 = *(_BYTE *)(v8 + 8);
      v32 = v153;
      v33 = v135;
      v35 = *(unsigned __int8 *)(v168 + 8);
LABEL_29:
      v37 = v35 - 17;
      v38 = 0;
      v39 = v37 <= 1;
      v40 = v168;
      if ( !v39 )
        v40 = 0;
      v148 = v40;
      if ( (unsigned int)v34 - 17 <= 1 )
        v38 = v8;
      if ( v161 == v32
        && v164 == v163
        && *(_QWORD *)(v162 + 8LL * (int)v33 + 112)
        && v142 <= 0x1F3
        && *(_BYTE *)(v142 + v162 + 500LL * v33 + 6414) <= 1u )
      {
        return v164;
      }
      if ( !v40 )
      {
        if ( !v38 )
        {
          if ( !*(_QWORD *)(v162 + 8LL * (int)v33 + 112)
            || v142 <= 0x1F3 && *(_BYTE *)(v142 + v162 + 500LL * v33 + 6414) == 2 )
          {
            return 4;
          }
          else
          {
            return 1;
          }
        }
        if ( a2 == 49 )
        {
          if ( *(_BYTE *)(v38 + 8) != 18 )
          {
            v62 = *(_DWORD *)(v38 + 32);
            LODWORD(v175) = v62;
            if ( v62 > 0x40 )
            {
              sub_C43690((__int64)&v174, -1, 1);
            }
            else
            {
              v63 = 0;
              if ( v62 )
                v63 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v62;
              v174 = v63;
            }
            v64 = sub_3064F80(v30, v38, &v174, 1, 0);
            if ( (unsigned int)v175 > 0x40 && v174 )
              j_j___libc_free_0_0(v174);
            return v64;
          }
          return 0;
        }
LABEL_239:
        BUG();
      }
      if ( !v38 )
      {
        if ( a2 == 49 )
        {
          if ( *(_BYTE *)(v40 + 8) != 18 )
          {
            v112 = *(_DWORD *)(v40 + 32);
            LODWORD(v175) = v112;
            if ( v112 > 0x40 )
            {
              sub_C43690((__int64)&v174, -1, 1);
            }
            else
            {
              if ( v112 )
                v38 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v112;
              v174 = v38;
            }
            result = sub_3064F80(v30, v148, &v174, 0, 1);
            if ( (unsigned int)v175 > 0x40 && v174 )
            {
              v170 = result;
              j_j___libc_free_0_0(v174);
              return v170;
            }
            return result;
          }
          return 0;
        }
        goto LABEL_239;
      }
      if ( v161 == v32 && v147 == v145 && v164 == v163 && v141 == v160 )
      {
        if ( a2 == 39 )
          return v163;
        if ( a2 == 40 )
        {
          result = 2 * v163;
          if ( !is_mul_ok(2u, v163) )
          {
            result = 0x7FFFFFFFFFFFFFFFLL;
            if ( v163 <= 0 )
              return 0x8000000000000000LL;
          }
          return result;
        }
        if ( *(_QWORD *)(v162 + 8LL * (int)v33 + 112)
          && (v142 > 0x1F3 || *(_BYTE *)(v142 + v162 + 500LL * v33 + 6414) != 2) )
        {
          return v163;
        }
      }
      v41 = sub_2D5BAE0(v162, *(_QWORD *)(v30 + 8), (__int64 *)v168, 0);
      sub_2FE6CC0((__int64)&v174, v162, *(_QWORD *)v168, v41, v42);
      v43 = v174;
      v44 = sub_2D5BAE0(v162, *(_QWORD *)(v30 + 8), (__int64 *)v8, 0);
      sub_2FE6CC0((__int64)&v174, v162, *(_QWORD *)v8, v44, v45);
      if ( v43 != 6 && (_BYTE)v174 != 6 )
      {
LABEL_39:
        if ( *(_BYTE *)(v38 + 8) != 18 )
        {
          v46 = *(_DWORD *)(v38 + 32);
          goto LABEL_41;
        }
        return 0;
      }
      v97 = *(_DWORD *)(v148 + 32);
      if ( *(_BYTE *)(v148 + 8) == 18 )
      {
        if ( !v97 )
          goto LABEL_39;
      }
      else if ( v97 <= 1 )
      {
        goto LABEL_39;
      }
      v98 = *(_BYTE *)(v38 + 8);
      v46 = *(_DWORD *)(v38 + 32);
      if ( v98 == 18 )
      {
        if ( !v46 )
          return 0;
      }
      else if ( v46 <= 1 )
      {
LABEL_41:
        if ( (unsigned int)*(unsigned __int8 *)(v168 + 8) - 17 <= 1 )
          v168 = **(_QWORD **)(v168 + 16);
        if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
          v8 = **(_QWORD **)(v8 + 16);
        v165 = v46;
        v48 = sub_3065900(v30, a2, v8, v168, a5, a6, (__int64)a7);
        if ( is_mul_ok(v48, v165) )
        {
          v49 = v48 * v165;
        }
        else if ( !v165 || (v49 = 0x7FFFFFFFFFFFFFFFLL, v48 <= 0) )
        {
          v49 = 0x8000000000000000LL;
        }
        if ( *(_BYTE *)(v38 + 8) == 18 )
        {
          if ( v47 == 1 )
            return v49;
          return v49;
        }
        else
        {
          v115 = *(_DWORD *)(v38 + 32);
          LODWORD(v175) = v115;
          if ( v115 > 0x40 )
          {
            v172 = v49;
            sub_C43690((__int64)&v174, -1, 1);
            v49 = v172;
          }
          else
          {
            v116 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v115;
            if ( !v115 )
              v116 = 0;
            v174 = v116;
          }
          v171 = v49;
          v117 = sub_3064F80(v30, v38, &v174, 1, 1);
          v118 = v171;
          v119 = v117;
          if ( (unsigned int)v175 > 0x40 && v174 )
          {
            j_j___libc_free_0_0(v174);
            v118 = v171;
          }
          v107 = __OFADD__(v118, v119);
          v50 = v118 + v119;
          if ( v107 )
          {
            v120 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v118 <= 0 )
              return 0x8000000000000000LL;
            return v120;
          }
        }
        return v50;
      }
      v99 = *(__int64 **)(v38 + 24);
      v166 = v174;
      LODWORD(v174) = v46 >> 1;
      BYTE4(v174) = v98 == 18;
      v100 = sub_BCE1B0(v99, v174);
      v101 = *(__int64 **)(v148 + 24);
      v169 = *(_DWORD *)(v148 + 32);
      BYTE4(v173) = *(_BYTE *)(v148 + 8) == 18;
      LODWORD(v173) = v169 >> 1;
      v102 = sub_BCE1B0(v101, v173);
      v103 = v166 != 6 || v43 != 6;
      v104 = sub_3065900(v30, a2, v100, v102, a5, a6, (__int64)a7);
      v106 = 2 * v104;
      if ( !is_mul_ok(2u, v104) )
      {
        if ( v104 <= 0 )
          return v103 + 0x8000000000000000LL;
        if ( v105 == 1 )
        {
          result = v103 + 0x7FFFFFFFFFFFFFFFLL;
          if ( result >= 0 && result >= (unsigned __int64)v103 )
            return result;
        }
        else
        {
          result = v103 + 0x7FFFFFFFFFFFFFFFLL;
          if ( result >= 0 && result >= (unsigned __int64)v103 )
            return result;
        }
        return 0x7FFFFFFFFFFFFFFFLL;
      }
      v107 = __OFADD__(v106, v103);
      v108 = v106 + v103;
      if ( !v107 )
        return v108;
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v106 <= 0 )
        return 0x8000000000000000LL;
      return result;
    case '1':
LABEL_27:
      if ( v161 == v32 && v164 == v163 && ((v35 & 0xFD) == 12) == ((v34 & 0xFD) == 12) && v147 == v145 && v141 == v160 )
        return 0;
      goto LABEL_28;
    case '2':
      v84 = v8;
      if ( (unsigned int)v34 - 17 <= 1 )
        v84 = **(_QWORD **)(v8 + 16);
      v85 = *(_DWORD *)(v84 + 8) >> 8;
      v86 = v168;
      if ( (unsigned int)(v35 - 17) <= 1 )
        v86 = **(_QWORD **)(v168 + 16);
      v133 = v33;
      v151 = v32;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v162 + 992LL))(
             v162,
             *(_DWORD *)(v86 + 8) >> 8,
             v85) )
      {
        return 0;
      }
      v34 = *(_BYTE *)(v8 + 8);
      v33 = v133;
      v32 = v151;
      v35 = *(unsigned __int8 *)(v168 + 8);
      goto LABEL_29;
    default:
      goto LABEL_29;
  }
}
