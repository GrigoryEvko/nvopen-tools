// Function: sub_30731F0
// Address: 0x30731f0
//
__int64 __fastcall sub_30731F0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 *a6)
{
  unsigned int v7; // eax
  __int64 v8; // r9
  unsigned int v9; // ebx
  unsigned __int8 *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // r9
  __int64 v13; // rbx
  __int64 v14; // rcx
  unsigned __int16 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // r13
  __int16 v18; // r9
  __int64 v19; // r13
  __int64 v20; // rcx
  unsigned int v21; // r15d
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rdx
  unsigned __int16 v28; // r9
  int v29; // r11d
  unsigned int v30; // r15d
  __int64 v31; // r10
  int v32; // r13d
  int v33; // ebx
  __int64 (*v34)(); // rax
  unsigned int v35; // eax
  __int64 v36; // r9
  __int64 v37; // r13
  unsigned int v38; // eax
  __int64 v39; // rdx
  char v40; // bl
  unsigned int v41; // eax
  __int64 v42; // rdx
  char v43; // r15
  unsigned __int64 v44; // r12
  signed __int64 v45; // rax
  __int64 v46; // r12
  unsigned __int64 v47; // rax
  bool v48; // of
  __int64 result; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned int v54; // eax
  __int64 v55; // r9
  unsigned int v56; // ebx
  unsigned __int8 *v57; // rdi
  __int64 v58; // rsi
  __int64 v59; // r9
  __int64 v60; // rax
  unsigned __int8 *v61; // rdi
  __int64 v62; // rsi
  __int64 (*v63)(); // rax
  __int64 *v64; // r13
  __int64 (*v65)(); // rbx
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  unsigned int v70; // eax
  char v71; // al
  __int64 v72; // rax
  __int64 (*v73)(); // rax
  unsigned __int16 v74; // bx
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  unsigned __int16 v78; // ax
  int v79; // edx
  __int64 *v80; // r13
  __int64 (*v81)(); // rbx
  __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  unsigned int v86; // eax
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rdx
  __int64 v91; // rcx
  unsigned int v92; // eax
  char v93; // al
  __int64 *v94; // rdi
  __int64 v95; // r12
  __int64 *v96; // rdi
  int v97; // eax
  __int64 v98; // rcx
  bool v99; // bl
  signed __int64 v100; // rax
  int v101; // edx
  __int64 v102; // rsi
  __int64 v103; // rbx
  __int64 (*v104)(); // rcx
  unsigned __int8 *v105; // rsi
  __int64 (*v106)(); // rax
  unsigned __int8 *v107; // rsi
  char v108; // al
  char v109; // al
  char v110; // al
  char v111; // al
  char v112; // al
  char v113; // al
  unsigned int v114; // [rsp+0h] [rbp-D0h]
  unsigned int v115; // [rsp+0h] [rbp-D0h]
  __int64 v116; // [rsp+8h] [rbp-C8h]
  __int64 v117; // [rsp+8h] [rbp-C8h]
  int v118; // [rsp+14h] [rbp-BCh]
  int v119; // [rsp+14h] [rbp-BCh]
  int v120; // [rsp+14h] [rbp-BCh]
  int v121; // [rsp+14h] [rbp-BCh]
  int v122; // [rsp+14h] [rbp-BCh]
  int v123; // [rsp+14h] [rbp-BCh]
  int v124; // [rsp+14h] [rbp-BCh]
  int v125; // [rsp+14h] [rbp-BCh]
  int v126; // [rsp+14h] [rbp-BCh]
  int v127; // [rsp+14h] [rbp-BCh]
  __int64 v128; // [rsp+18h] [rbp-B8h]
  __int64 v129; // [rsp+18h] [rbp-B8h]
  __int64 v130; // [rsp+18h] [rbp-B8h]
  __int64 v131; // [rsp+18h] [rbp-B8h]
  __int64 v132; // [rsp+18h] [rbp-B8h]
  __int64 v133; // [rsp+18h] [rbp-B8h]
  __int64 v134; // [rsp+18h] [rbp-B8h]
  __int64 v135; // [rsp+18h] [rbp-B8h]
  __int64 v136; // [rsp+18h] [rbp-B8h]
  __int64 v137; // [rsp+18h] [rbp-B8h]
  char v138; // [rsp+23h] [rbp-ADh]
  unsigned int v139; // [rsp+24h] [rbp-ACh]
  int v141; // [rsp+2Ch] [rbp-A4h]
  char v142; // [rsp+2Ch] [rbp-A4h]
  __int16 v143; // [rsp+30h] [rbp-A0h]
  __int64 v144; // [rsp+30h] [rbp-A0h]
  __int64 v145; // [rsp+38h] [rbp-98h]
  signed __int64 v147; // [rsp+48h] [rbp-88h]
  signed __int64 v148; // [rsp+50h] [rbp-80h]
  __int64 v149; // [rsp+50h] [rbp-80h]
  int v150; // [rsp+58h] [rbp-78h]
  __int64 v152; // [rsp+60h] [rbp-70h]
  __int64 v153; // [rsp+68h] [rbp-68h]
  __int64 v154; // [rsp+78h] [rbp-58h] BYREF
  __int64 v155; // [rsp+80h] [rbp-50h] BYREF
  __int64 v156; // [rsp+88h] [rbp-48h]
  __int64 v157; // [rsp+90h] [rbp-40h]

  v153 = a3;
  v152 = a4;
  if ( a2 == 48 )
  {
    v54 = sub_BCB060(a4);
    v55 = *(_QWORD *)(a1 + 8);
    v56 = v54;
    v57 = *(unsigned __int8 **)(v55 + 32);
    v58 = *(_QWORD *)(v55 + 40);
    v155 = v54;
    if ( &v57[v58] != sub_3071A20(v57, (__int64)&v57[v58], &v155) && v56 <= (unsigned int)sub_AE43A0(v59, v153) )
      return 0;
  }
  else if ( a2 > 0x30 )
  {
    if ( a2 == 49 && (a3 == a4 || *(_BYTE *)(a3 + 8) == 14 && *(_BYTE *)(a4 + 8) == 14) )
      return 0;
  }
  else if ( a2 == 38 )
  {
    v52 = sub_9208B0(*(_QWORD *)(a1 + 8), a3);
    v156 = v53;
    v155 = v52;
    if ( !(_BYTE)v53 )
    {
      v60 = *(_QWORD *)(a1 + 8);
      v61 = *(unsigned __int8 **)(v60 + 32);
      v62 = *(_QWORD *)(v60 + 40);
      v154 = v155;
      if ( &v61[v62] != sub_3071A20(v61, (__int64)&v61[v62], &v154) )
        return 0;
    }
  }
  else if ( a2 == 47 )
  {
    v7 = sub_BCB060(a3);
    v8 = *(_QWORD *)(a1 + 8);
    v9 = v7;
    v10 = *(unsigned __int8 **)(v8 + 32);
    v11 = *(_QWORD *)(v8 + 40);
    v155 = v7;
    if ( &v10[v11] != sub_3071A20(v10, (__int64)&v10[v11], &v155) && v9 >= (unsigned int)sub_AE43A0(v12, v152) )
      return 0;
  }
  v145 = *(_QWORD *)(a1 + 24);
  v139 = sub_2FEBEF0(v145, a2);
  v13 = *(_QWORD *)v152;
  v147 = 1;
  v14 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)v152, 0);
  v15 = v14;
  v17 = v16;
  while ( 1 )
  {
    LOWORD(v14) = v15;
    sub_2FE6CC0((__int64)&v155, *(_QWORD *)(a1 + 24), v13, v14, v17);
    v18 = v156;
    if ( (_BYTE)v155 == 10 )
      break;
    if ( !(_BYTE)v155 )
    {
      v150 = 0;
      v18 = v15;
      v141 = v15;
      goto LABEL_14;
    }
    if ( (v155 & 0xFB) == 2 )
    {
      v51 = 2 * v147;
      if ( !is_mul_ok(2u, v147) )
      {
        v51 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v147 <= 0 )
          v51 = 0x8000000000000000LL;
      }
      v147 = v51;
    }
    if ( v15 == (_WORD)v156 )
    {
      if ( (_WORD)v156 )
      {
        v141 = (unsigned __int16)v156;
LABEL_13:
        v150 = 0;
        goto LABEL_14;
      }
      if ( v17 == v157 )
      {
        v141 = 0;
        goto LABEL_13;
      }
    }
    v14 = v156;
    v17 = v157;
    v15 = v156;
  }
  if ( v15 )
  {
    v18 = v15;
    v141 = v15;
  }
  else
  {
    v141 = 8;
    v18 = 8;
  }
  v150 = 1;
  v147 = 0;
LABEL_14:
  v143 = v18;
  v19 = *(_QWORD *)v153;
  v148 = 1;
  v20 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)v153, 0);
  v21 = v20;
  v22 = v19;
  v24 = v23;
  while ( 1 )
  {
    LOWORD(v20) = v21;
    sub_2FE6CC0((__int64)&v155, *(_QWORD *)(a1 + 24), v22, v20, v24);
    v27 = (unsigned __int16)v156;
    if ( (_BYTE)v155 == 10 )
      break;
    if ( !(_BYTE)v155 )
    {
      v28 = v143;
      v27 = v21;
      v29 = 0;
      goto LABEL_21;
    }
    if ( (v155 & 0xFB) == 2 )
    {
      v50 = 2 * v148;
      if ( !is_mul_ok(2u, v148) )
      {
        v50 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v148 <= 0 )
          v50 = 0x8000000000000000LL;
      }
      v148 = v50;
    }
    if ( (_WORD)v21 == (_WORD)v156 && ((_WORD)v156 || v24 == v157) )
    {
      v28 = v143;
      v29 = 0;
      goto LABEL_21;
    }
    v20 = v156;
    v24 = v157;
    v21 = (unsigned __int16)v156;
  }
  v27 = 8;
  v148 = 0;
  v28 = v143;
  if ( (_WORD)v21 )
    v27 = v21;
  v29 = 1;
LABEL_21:
  if ( v28 <= 1u
    || (unsigned __int16)(v28 - 504) <= 7u
    || (v30 = (unsigned __int16)v27,
        v144 = *(_QWORD *)&byte_444C4A0[16 * v141 - 16],
        v142 = byte_444C4A0[16 * v141 - 8],
        (unsigned __int16)v27 <= 1u)
    || (unsigned __int16)(v27 - 504) <= 7u )
  {
LABEL_213:
    BUG();
  }
  v31 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v27 - 16];
  v138 = byte_444C4A0[16 * (unsigned __int16)v27 - 8];
  v32 = *(unsigned __int8 *)(v152 + 8);
  v33 = *(unsigned __int8 *)(v153 + 8);
  switch ( a2 )
  {
    case '&':
      v34 = *(__int64 (**)())(*(_QWORD *)v145 + 1392LL);
      if ( v34 == sub_2FE3480 )
        goto LABEL_27;
      v125 = v29;
      v135 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v27 - 16];
      v110 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v34)(
               v145,
               v28,
               0,
               (unsigned __int16)v27,
               0);
      v31 = v135;
      v29 = v125;
      if ( !v110 )
        goto LABEL_27;
      return 0;
    case '\'':
      v25 = (unsigned __int16)v27;
      v63 = *(__int64 (**)())(*(_QWORD *)v145 + 1432LL);
      if ( v63 == sub_2FE34A0 )
        goto LABEL_89;
      v124 = v29;
      v134 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v27 - 16];
      v109 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v63)(
               v145,
               v28,
               0,
               (unsigned __int16)v27,
               0);
      v31 = v134;
      v29 = v124;
      if ( !v109 )
        goto LABEL_89;
      return 0;
    case '(':
LABEL_89:
      if ( !a6 )
        goto LABEL_98;
      v64 = *(__int64 **)(a1 + 24);
      v27 = *a6;
      switch ( (_DWORD)v27 )
      {
        case 'E':
          goto LABEL_96;
        case 'K':
          v65 = *(__int64 (**)())(*v64 + 1560);
          if ( (a6[7] & 0x40) != 0 )
            v66 = *((_QWORD *)a6 - 1);
          else
            v66 = (__int64)&a6[-32 * (*((_DWORD *)a6 + 1) & 0x7FFFFFF)];
          v118 = v29;
          v128 = v31;
          v114 = sub_30097B0(*(_QWORD *)(*(_QWORD *)v66 + 8LL), 0, v66, v25, v26);
          v116 = v67;
          v70 = sub_30097B0(*((_QWORD *)a6 + 1), 0, v67, v68, v69);
          v31 = v128;
          v29 = v118;
          if ( v65 != sub_2D566A0 )
          {
            v71 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64))v65)(v64, v70, v27, v114, v116);
            v31 = v128;
            v29 = v118;
            if ( v71 )
              return 0;
          }
          goto LABEL_96;
        case 'D':
          v72 = *v64;
          v25 = *(_QWORD *)(*v64 + 1424);
          if ( (a6[7] & 0x40) != 0 )
          {
            v107 = (unsigned __int8 *)*((_QWORD *)a6 - 1);
          }
          else
          {
            v27 = 32LL * (*((_DWORD *)a6 + 1) & 0x7FFFFFF);
            v107 = &a6[-v27];
          }
          if ( (__int64 (*)())v25 == sub_2D56670 )
          {
LABEL_97:
            v73 = *(__int64 (**)())(v72 + 1816);
            if ( v73 != sub_2D566C0 )
            {
              v126 = v29;
              v136 = v31;
              v111 = ((__int64 (__fastcall *)(__int64 *, unsigned __int8 *, __int64, __int64))v73)(v64, a6, v27, v25);
              v31 = v136;
              v29 = v126;
              if ( v111 )
                return 0;
            }
LABEL_98:
            if ( a5 != 1 )
              goto LABEL_28;
            v119 = v29;
            v129 = v31;
            v74 = sub_30097B0(v153, 0, v27, v25, v26);
            v78 = sub_30097B0(v152, 0, v75, v76, v77);
            v29 = v119;
            v31 = v129;
            if ( v119 != v150
              || v148 != v147
              || !v78
              || !v74
              || (((int)*(unsigned __int16 *)(v145 + 2 * (v78 + 274LL * v74 + 71704) + 6) >> (4 * ((a2 == 39) + 2)))
                & 0xF) != 0 )
            {
              goto LABEL_28;
            }
            return 0;
          }
          v123 = v29;
          v133 = v31;
          v108 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD))v25)(
                   v64,
                   *(_QWORD *)(*(_QWORD *)v107 + 8LL),
                   *((_QWORD *)a6 + 1));
          v31 = v133;
          v29 = v123;
          if ( v108 )
            return 0;
LABEL_96:
          v72 = *v64;
          goto LABEL_97;
      }
      goto LABEL_213;
    case '.':
      if ( !a6 )
        goto LABEL_29;
      v79 = *a6;
      v80 = *(__int64 **)(a1 + 24);
      if ( v79 == 69 )
        goto LABEL_111;
      if ( v79 == 75 )
      {
        v81 = *(__int64 (**)())(*v80 + 1560);
        if ( (a6[7] & 0x40) != 0 )
          v82 = *((_QWORD *)a6 - 1);
        else
          v82 = (__int64)&a6[-32 * (*((_DWORD *)a6 + 1) & 0x7FFFFFF)];
        v120 = v29;
        v130 = *(_QWORD *)&byte_444C4A0[16 * (v30 - 1)];
        v115 = sub_30097B0(*(_QWORD *)(*(_QWORD *)v82 + 8LL), 0, v82, v25, v26);
        v117 = v83;
        v86 = sub_30097B0(*((_QWORD *)a6 + 1), 0, v83, v84, v85);
        v31 = v130;
        v29 = v120;
        if ( v81 != sub_2D566A0 )
        {
          v112 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64))v81)(v80, v86, v87, v115, v117);
          v31 = v130;
          v29 = v120;
          if ( v112 )
            return 0;
        }
LABEL_111:
        v88 = *v80;
        goto LABEL_131;
      }
      if ( v79 != 68 )
        goto LABEL_213;
      v88 = *v80;
      v104 = *(__int64 (**)())(*v80 + 1424);
      if ( (a6[7] & 0x40) != 0 )
        v105 = (unsigned __int8 *)*((_QWORD *)a6 - 1);
      else
        v105 = &a6[-32 * (*((_DWORD *)a6 + 1) & 0x7FFFFFF)];
      if ( v104 != sub_2D56670 )
      {
        v127 = v29;
        v137 = *(_QWORD *)&byte_444C4A0[16 * (v30 - 1)];
        v113 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD))v104)(
                 v80,
                 *(_QWORD *)(*(_QWORD *)v105 + 8LL),
                 *((_QWORD *)a6 + 1));
        v31 = v137;
        v29 = v127;
        if ( v113 )
          return 0;
        goto LABEL_111;
      }
LABEL_131:
      v106 = *(__int64 (**)())(v88 + 1816);
      if ( v106 == sub_2D566C0 )
      {
LABEL_28:
        v32 = *(unsigned __int8 *)(v152 + 8);
        v33 = *(unsigned __int8 *)(v153 + 8);
        goto LABEL_29;
      }
      v122 = v29;
      v132 = v31;
      if ( ((unsigned __int8 (__fastcall *)(__int64 *, unsigned __int8 *))v106)(v80, a6) )
        return 0;
      v31 = v132;
      v29 = v122;
      v32 = *(unsigned __int8 *)(v152 + 8);
      v33 = *(unsigned __int8 *)(v153 + 8);
LABEL_29:
      v35 = v32 - 17;
      v36 = v152;
      v37 = 0;
      if ( v35 > 1 )
        v36 = 0;
      if ( (unsigned int)(v33 - 17) <= 1 )
        v37 = v153;
      if ( v29 == v150
        && v148 == v147
        && *(_QWORD *)(v145 + 8LL * (int)v30 + 112)
        && v139 <= 0x1F3
        && *(_BYTE *)(v139 + v145 + 500LL * v30 + 6414) <= 1u )
      {
        return v148;
      }
      if ( !v36 )
      {
        if ( !v37 )
        {
          if ( !*(_QWORD *)(v145 + 8LL * (int)v30 + 112)
            || v139 <= 0x1F3 && *(_BYTE *)(v139 + v145 + 500LL * v30 + 6414) == 2 )
          {
            return 4;
          }
          else
          {
            return 1;
          }
        }
        if ( a2 == 49 )
          return sub_30727B0(a1, v37, 1, 0);
LABEL_214:
        BUG();
      }
      if ( !v37 )
      {
        if ( a2 == 49 )
          return sub_30727B0(a1, v36, 0, 1);
        goto LABEL_214;
      }
      if ( v29 == v150 && v31 == v144 && v148 == v147 && v142 == v138 )
      {
        if ( a2 == 39 )
          return v147;
        if ( a2 == 40 )
        {
          result = 2 * v147;
          if ( !is_mul_ok(2u, v147) )
          {
            result = 0x7FFFFFFFFFFFFFFFLL;
            if ( v147 <= 0 )
              return 0x8000000000000000LL;
          }
          return result;
        }
        if ( *(_QWORD *)(v145 + 8LL * (int)v30 + 112)
          && (v139 > 0x1F3 || *(_BYTE *)(v139 + v145 + 500LL * v30 + 6414) != 2) )
        {
          return v147;
        }
      }
      v149 = v36;
      v38 = sub_2D5BAE0(v145, *(_QWORD *)(a1 + 8), (__int64 *)v152, 0);
      sub_2FE6CC0((__int64)&v155, v145, *(_QWORD *)v152, v38, v39);
      v40 = v155;
      v41 = sub_2D5BAE0(v145, *(_QWORD *)(a1 + 8), (__int64 *)v153, 0);
      sub_2FE6CC0((__int64)&v155, v145, *(_QWORD *)v153, v41, v42);
      v43 = v155;
      if ( v40 != 6 && (_BYTE)v155 != 6 )
      {
LABEL_39:
        if ( *(_BYTE *)(v37 + 8) != 18 )
        {
          v44 = *(unsigned int *)(v37 + 32);
LABEL_41:
          if ( (unsigned int)*(unsigned __int8 *)(v152 + 8) - 17 <= 1 )
            v152 = **(_QWORD **)(v152 + 16);
          if ( (unsigned int)*(unsigned __int8 *)(v153 + 8) - 17 <= 1 )
            v153 = **(_QWORD **)(v153 + 16);
          v45 = sub_30731F0(a1, a2, v153, v152, a5, a6);
          if ( is_mul_ok(v45, v44) )
          {
            v46 = v45 * v44;
          }
          else if ( !v44 || (v46 = 0x7FFFFFFFFFFFFFFFLL, v45 <= 0) )
          {
            v46 = 0x8000000000000000LL;
          }
          v47 = sub_30727B0(a1, v37, 1, 1);
          v48 = __OFADD__(v46, v47);
          result = v46 + v47;
          if ( v48 )
          {
            result = 0x7FFFFFFFFFFFFFFFLL;
            if ( v46 <= 0 )
              return 0x8000000000000000LL;
          }
          return result;
        }
        return 0;
      }
      v92 = *(_DWORD *)(v149 + 32);
      if ( *(_BYTE *)(v149 + 8) == 18 )
      {
        if ( !v92 )
          goto LABEL_39;
      }
      else if ( v92 <= 1 )
      {
        goto LABEL_39;
      }
      v93 = *(_BYTE *)(v37 + 8);
      v44 = *(unsigned int *)(v37 + 32);
      if ( v93 == 18 )
      {
        if ( !(_DWORD)v44 )
          return 0;
      }
      else if ( (unsigned int)v44 <= 1 )
      {
        goto LABEL_41;
      }
      v94 = *(__int64 **)(v37 + 24);
      LODWORD(v155) = (unsigned int)v44 >> 1;
      BYTE4(v155) = v93 == 18;
      v95 = sub_BCE1B0(v94, v155);
      v96 = *(__int64 **)(v149 + 24);
      v97 = *(_DWORD *)(v149 + 32) >> 1;
      BYTE4(v154) = *(_BYTE *)(v149 + 8) == 18;
      LODWORD(v154) = v97;
      v98 = sub_BCE1B0(v96, v154);
      v99 = v43 != 6 || v40 != 6;
      v100 = sub_30731F0(a1, a2, v95, v98, a5, a6);
      v102 = 2 * v100;
      if ( !is_mul_ok(2u, v100) )
      {
        if ( v100 <= 0 )
          return v99 + 0x8000000000000000LL;
        if ( v101 == 1 )
        {
          result = v99 + 0x7FFFFFFFFFFFFFFFLL;
          if ( result >= 0 && result >= (unsigned __int64)v99 )
            return result;
        }
        else
        {
          result = v99 + 0x7FFFFFFFFFFFFFFFLL;
          if ( result >= 0 && result >= (unsigned __int64)v99 )
            return result;
        }
        return 0x7FFFFFFFFFFFFFFFLL;
      }
      v48 = __OFADD__(v102, v99);
      v103 = v102 + v99;
      if ( !v48 )
        return v103;
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v102 <= 0 )
        return 0x8000000000000000LL;
      return result;
    case '1':
LABEL_27:
      if ( v29 == v150 && v148 == v147 && ((v33 & 0xFD) == 12) == ((v32 & 0xFD) == 12) && v31 == v144 && v142 == v138 )
        return 0;
      goto LABEL_28;
    case '2':
      v89 = v153;
      if ( (unsigned int)(v33 - 17) <= 1 )
        v89 = **(_QWORD **)(v153 + 16);
      v90 = *(_DWORD *)(v89 + 8) >> 8;
      v91 = v152;
      if ( (unsigned int)(v32 - 17) <= 1 )
        v91 = **(_QWORD **)(v152 + 16);
      v121 = v29;
      v131 = *(_QWORD *)&byte_444C4A0[16 * (v30 - 1)];
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v145 + 992LL))(
             v145,
             *(_DWORD *)(v91 + 8) >> 8,
             v90) )
      {
        return 0;
      }
      v29 = v121;
      v31 = v131;
      v32 = *(unsigned __int8 *)(v152 + 8);
      v33 = *(unsigned __int8 *)(v153 + 8);
      goto LABEL_29;
    default:
      goto LABEL_29;
  }
}
