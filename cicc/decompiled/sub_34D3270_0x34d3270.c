// Function: sub_34D3270
// Address: 0x34d3270
//
__int64 __fastcall sub_34D3270(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        int a6,
        unsigned __int8 *a7)
{
  unsigned int v8; // eax
  __int64 v9; // r9
  unsigned int v10; // ebx
  unsigned __int8 *v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // rcx
  unsigned __int16 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // r13
  __int16 v19; // r9
  __int64 v20; // r13
  __int64 v21; // rcx
  unsigned int v22; // r15d
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // r13
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // rdx
  unsigned __int16 v29; // r9
  int v30; // r11d
  unsigned int v31; // r15d
  __int64 v32; // r10
  int v33; // r13d
  int v34; // ebx
  __int64 (*v35)(); // rax
  unsigned int v36; // eax
  __int64 v37; // r9
  __int64 v38; // r13
  unsigned int v39; // eax
  __int64 v40; // rdx
  char v41; // bl
  unsigned int v42; // eax
  __int64 v43; // rdx
  char v44; // r15
  unsigned int v45; // r12d
  signed __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // r12
  unsigned __int64 v49; // kr00_8
  unsigned __int64 v50; // rax
  bool v51; // of
  __int64 result; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  unsigned int v57; // eax
  __int64 v58; // r9
  unsigned int v59; // ebx
  unsigned __int8 *v60; // rdi
  __int64 v61; // rsi
  __int64 v62; // r9
  __int64 v63; // rax
  unsigned __int8 *v64; // rdi
  __int64 v65; // rsi
  __int64 (*v66)(); // rax
  __int64 *v67; // r13
  __int64 (*v68)(); // rbx
  __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  unsigned int v73; // eax
  char v74; // al
  __int64 v75; // rax
  __int64 (*v76)(); // rax
  unsigned __int16 v77; // bx
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  unsigned __int16 v81; // ax
  __int64 *v82; // r13
  int v83; // edx
  __int64 (*v84)(); // rbx
  __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  unsigned int v89; // eax
  __int64 v90; // rdx
  __int64 v91; // rax
  __int64 (*v92)(); // rax
  __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 v95; // rcx
  unsigned int v96; // eax
  char v97; // al
  __int64 *v98; // rdi
  int v99; // r12d
  __int64 *v100; // rdi
  int v101; // eax
  int v102; // eax
  bool v103; // bl
  signed __int64 v104; // rax
  int v105; // edx
  __int64 v106; // rsi
  __int64 v107; // rbx
  __int64 (*v108)(); // rcx
  unsigned __int8 *v109; // rsi
  char v110; // al
  unsigned __int8 *v111; // rsi
  char v112; // al
  char v113; // al
  char v114; // al
  char v115; // al
  char v116; // al
  unsigned int v117; // [rsp+8h] [rbp-C8h]
  unsigned int v118; // [rsp+8h] [rbp-C8h]
  __int64 v119; // [rsp+10h] [rbp-C0h]
  __int64 v120; // [rsp+10h] [rbp-C0h]
  __int64 v121; // [rsp+18h] [rbp-B8h]
  __int64 v122; // [rsp+18h] [rbp-B8h]
  __int64 v123; // [rsp+18h] [rbp-B8h]
  __int64 v124; // [rsp+18h] [rbp-B8h]
  __int64 v125; // [rsp+18h] [rbp-B8h]
  __int64 v126; // [rsp+18h] [rbp-B8h]
  __int64 v127; // [rsp+18h] [rbp-B8h]
  __int64 v128; // [rsp+18h] [rbp-B8h]
  __int64 v129; // [rsp+18h] [rbp-B8h]
  __int64 v130; // [rsp+18h] [rbp-B8h]
  int v131; // [rsp+20h] [rbp-B0h]
  int v132; // [rsp+20h] [rbp-B0h]
  int v133; // [rsp+20h] [rbp-B0h]
  int v134; // [rsp+20h] [rbp-B0h]
  int v135; // [rsp+20h] [rbp-B0h]
  int v136; // [rsp+20h] [rbp-B0h]
  int v137; // [rsp+20h] [rbp-B0h]
  int v138; // [rsp+20h] [rbp-B0h]
  int v139; // [rsp+20h] [rbp-B0h]
  int v140; // [rsp+20h] [rbp-B0h]
  char v141; // [rsp+27h] [rbp-A9h]
  unsigned int v142; // [rsp+28h] [rbp-A8h]
  int v145; // [rsp+34h] [rbp-9Ch]
  char v146; // [rsp+34h] [rbp-9Ch]
  __int16 v147; // [rsp+38h] [rbp-98h]
  __int64 v148; // [rsp+38h] [rbp-98h]
  __int64 v149; // [rsp+40h] [rbp-90h]
  signed __int64 v150; // [rsp+48h] [rbp-88h]
  signed __int64 v151; // [rsp+50h] [rbp-80h]
  __int64 v152; // [rsp+50h] [rbp-80h]
  int v153; // [rsp+58h] [rbp-78h]
  __int64 v155; // [rsp+60h] [rbp-70h]
  __int64 v156; // [rsp+68h] [rbp-68h]
  __int64 v157; // [rsp+78h] [rbp-58h] BYREF
  __int64 v158; // [rsp+80h] [rbp-50h] BYREF
  __int64 v159; // [rsp+88h] [rbp-48h]
  __int64 v160; // [rsp+90h] [rbp-40h]

  v156 = a3;
  v155 = a4;
  if ( a2 == 48 )
  {
    v57 = sub_BCB060(a4);
    v58 = *(_QWORD *)(a1 + 8);
    v59 = v57;
    v60 = *(unsigned __int8 **)(v58 + 32);
    v61 = *(_QWORD *)(v58 + 40);
    v158 = v57;
    if ( &v60[v61] != sub_34CD7F0(v60, (__int64)&v60[v61], &v158) && v59 <= (unsigned int)sub_AE43A0(v62, v156) )
      return 0;
  }
  else if ( a2 > 0x30 )
  {
    if ( a2 == 49 && (a3 == a4 || *(_BYTE *)(a3 + 8) == 14 && *(_BYTE *)(a4 + 8) == 14) )
      return 0;
  }
  else if ( a2 == 38 )
  {
    v55 = sub_9208B0(*(_QWORD *)(a1 + 8), a3);
    v159 = v56;
    v158 = v55;
    if ( !(_BYTE)v56 )
    {
      v63 = *(_QWORD *)(a1 + 8);
      v64 = *(unsigned __int8 **)(v63 + 32);
      v65 = *(_QWORD *)(v63 + 40);
      v157 = v158;
      if ( &v64[v65] != sub_34CD7F0(v64, (__int64)&v64[v65], &v157) )
        return 0;
    }
  }
  else if ( a2 == 47 )
  {
    v8 = sub_BCB060(a3);
    v9 = *(_QWORD *)(a1 + 8);
    v10 = v8;
    v11 = *(unsigned __int8 **)(v9 + 32);
    v12 = *(_QWORD *)(v9 + 40);
    v158 = v8;
    if ( &v11[v12] != sub_34CD7F0(v11, (__int64)&v11[v12], &v158) && v10 >= (unsigned int)sub_AE43A0(v13, v155) )
      return 0;
  }
  v149 = *(_QWORD *)(a1 + 24);
  v142 = sub_2FEBEF0(v149, a2);
  v14 = *(_QWORD *)v155;
  v151 = 1;
  v15 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)v155, 0);
  v16 = v15;
  v18 = v17;
  while ( 1 )
  {
    LOWORD(v15) = v16;
    sub_2FE6CC0((__int64)&v158, *(_QWORD *)(a1 + 24), v14, v15, v18);
    v19 = v159;
    if ( (_BYTE)v158 == 10 )
      break;
    if ( !(_BYTE)v158 )
    {
      v153 = 0;
      v19 = v16;
      v145 = v16;
      goto LABEL_14;
    }
    if ( (v158 & 0xFB) == 2 )
    {
      v54 = 2 * v151;
      if ( !is_mul_ok(2u, v151) )
      {
        v54 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v151 <= 0 )
          v54 = 0x8000000000000000LL;
      }
      v151 = v54;
    }
    if ( v16 == (_WORD)v159 )
    {
      if ( (_WORD)v159 )
      {
        v145 = (unsigned __int16)v159;
LABEL_13:
        v153 = 0;
        goto LABEL_14;
      }
      if ( v18 == v160 )
      {
        v145 = 0;
        goto LABEL_13;
      }
    }
    v15 = v159;
    v18 = v160;
    v16 = v159;
  }
  if ( v16 )
  {
    v19 = v16;
    v145 = v16;
  }
  else
  {
    v145 = 8;
    v19 = 8;
  }
  v153 = 1;
  v151 = 0;
LABEL_14:
  v147 = v19;
  v20 = *(_QWORD *)v156;
  v150 = 1;
  v21 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)v156, 0);
  v22 = v21;
  v23 = v20;
  v25 = v24;
  while ( 1 )
  {
    LOWORD(v21) = v22;
    sub_2FE6CC0((__int64)&v158, *(_QWORD *)(a1 + 24), v23, v21, v25);
    v28 = (unsigned __int16)v159;
    if ( (_BYTE)v158 == 10 )
      break;
    if ( !(_BYTE)v158 )
    {
      v29 = v147;
      v28 = v22;
      v30 = 0;
      goto LABEL_21;
    }
    if ( (v158 & 0xFB) == 2 )
    {
      v53 = 2 * v150;
      if ( !is_mul_ok(2u, v150) )
      {
        v53 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v150 <= 0 )
          v53 = 0x8000000000000000LL;
      }
      v150 = v53;
    }
    if ( (_WORD)v22 == (_WORD)v159 && ((_WORD)v159 || v25 == v160) )
    {
      v29 = v147;
      v30 = 0;
      goto LABEL_21;
    }
    v21 = v159;
    v25 = v160;
    v22 = (unsigned __int16)v159;
  }
  v28 = 8;
  v150 = 0;
  v29 = v147;
  if ( (_WORD)v22 )
    v28 = v22;
  v30 = 1;
LABEL_21:
  if ( v29 <= 1u )
    goto LABEL_207;
  if ( (unsigned __int16)(v29 - 504) <= 7u )
    goto LABEL_207;
  v31 = (unsigned __int16)v28;
  v148 = *(_QWORD *)&byte_444C4A0[16 * v145 - 16];
  v141 = byte_444C4A0[16 * v145 - 8];
  if ( (unsigned __int16)v28 <= 1u || (unsigned __int16)(v28 - 504) <= 7u )
    goto LABEL_207;
  v32 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v28 - 16];
  v146 = byte_444C4A0[16 * (unsigned __int16)v28 - 8];
  v33 = *(unsigned __int8 *)(v155 + 8);
  v34 = *(unsigned __int8 *)(v156 + 8);
  switch ( a2 )
  {
    case '&':
      v35 = *(__int64 (**)())(*(_QWORD *)v149 + 1392LL);
      if ( v35 == sub_2FE3480 )
        goto LABEL_27;
      v129 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v28 - 16];
      v139 = v30;
      v114 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v35)(
               v149,
               v29,
               0,
               (unsigned __int16)v28,
               0);
      v30 = v139;
      v32 = v129;
      if ( !v114 )
        goto LABEL_27;
      return 0;
    case '\'':
      v26 = (unsigned __int16)v28;
      v66 = *(__int64 (**)())(*(_QWORD *)v149 + 1432LL);
      if ( v66 == sub_2FE34A0 )
        goto LABEL_88;
      v128 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v28 - 16];
      v138 = v30;
      v113 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v66)(
               v149,
               v29,
               0,
               (unsigned __int16)v28,
               0);
      v30 = v138;
      v32 = v128;
      if ( !v113 )
        goto LABEL_88;
      return 0;
    case '(':
LABEL_88:
      if ( !a7 )
        goto LABEL_97;
      v67 = *(__int64 **)(a1 + 24);
      v28 = *a7;
      switch ( (_DWORD)v28 )
      {
        case 'E':
          break;
        case 'K':
          v68 = *(__int64 (**)())(*v67 + 1560);
          if ( (a7[7] & 0x40) != 0 )
            v69 = *((_QWORD *)a7 - 1);
          else
            v69 = (__int64)&a7[-32 * (*((_DWORD *)a7 + 1) & 0x7FFFFFF)];
          v121 = v32;
          v131 = v30;
          v117 = sub_30097B0(*(_QWORD *)(*(_QWORD *)v69 + 8LL), 0, v69, v26, v27);
          v119 = v70;
          v73 = sub_30097B0(*((_QWORD *)a7 + 1), 0, v70, v71, v72);
          v30 = v131;
          v32 = v121;
          if ( v68 != sub_2D566A0 )
          {
            v74 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64))v68)(v67, v73, v28, v117, v119);
            v30 = v131;
            v32 = v121;
            if ( v74 )
              return 0;
          }
          break;
        case 'D':
          v75 = *v67;
          v26 = *(_QWORD *)(*v67 + 1424);
          if ( (a7[7] & 0x40) != 0 )
          {
            v111 = (unsigned __int8 *)*((_QWORD *)a7 - 1);
          }
          else
          {
            v28 = 32LL * (*((_DWORD *)a7 + 1) & 0x7FFFFFF);
            v111 = &a7[-v28];
          }
          if ( (__int64 (*)())v26 == sub_2D56670 )
          {
LABEL_96:
            v76 = *(__int64 (**)())(v75 + 1816);
            if ( v76 != sub_2D566C0 )
            {
              v130 = v32;
              v140 = v30;
              v115 = ((__int64 (__fastcall *)(__int64 *, unsigned __int8 *, __int64, __int64))v76)(v67, a7, v28, v26);
              v30 = v140;
              v32 = v130;
              if ( v115 )
                return 0;
            }
LABEL_97:
            if ( a5 != 1 )
              goto LABEL_28;
            v122 = v32;
            v132 = v30;
            v77 = sub_30097B0(v156, 0, v28, v26, v27);
            v81 = sub_30097B0(v155, 0, v78, v79, v80);
            v30 = v132;
            v32 = v122;
            if ( v132 != v153
              || v151 != v150
              || !v81
              || !v77
              || (((int)*(unsigned __int16 *)(v149 + 2 * (v81 + 274LL * v77 + 71704) + 6) >> (4 * ((a2 == 39) + 2)))
                & 0xF) != 0 )
            {
              goto LABEL_28;
            }
            return 0;
          }
          v127 = v32;
          v137 = v30;
          v112 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD))v26)(
                   v67,
                   *(_QWORD *)(*(_QWORD *)v111 + 8LL),
                   *((_QWORD *)a7 + 1));
          v30 = v137;
          v32 = v127;
          if ( v112 )
            return 0;
          break;
        default:
          goto LABEL_207;
      }
      v75 = *v67;
      goto LABEL_96;
    case '.':
      if ( !a7 )
        goto LABEL_29;
      v82 = *(__int64 **)(a1 + 24);
      v83 = *a7;
      if ( v83 == 69 )
        goto LABEL_110;
      if ( v83 == 75 )
      {
        v84 = *(__int64 (**)())(*v82 + 1560);
        if ( (a7[7] & 0x40) != 0 )
          v85 = *((_QWORD *)a7 - 1);
        else
          v85 = (__int64)&a7[-32 * (*((_DWORD *)a7 + 1) & 0x7FFFFFF)];
        v123 = *(_QWORD *)&byte_444C4A0[16 * (v31 - 1)];
        v133 = v30;
        v118 = sub_30097B0(*(_QWORD *)(*(_QWORD *)v85 + 8LL), 0, v85, v26, v27);
        v120 = v86;
        v89 = sub_30097B0(*((_QWORD *)a7 + 1), 0, v86, v87, v88);
        v30 = v133;
        v32 = v123;
        if ( v84 != sub_2D566A0 )
        {
          v116 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64))v84)(v82, v89, v90, v118, v120);
          v30 = v133;
          v32 = v123;
          if ( v116 )
            return 0;
        }
LABEL_110:
        v91 = *v82;
        goto LABEL_111;
      }
      if ( v83 != 68 )
        goto LABEL_207;
      v91 = *v82;
      v108 = *(__int64 (**)())(*v82 + 1424);
      if ( (a7[7] & 0x40) != 0 )
        v109 = (unsigned __int8 *)*((_QWORD *)a7 - 1);
      else
        v109 = &a7[-32 * (*((_DWORD *)a7 + 1) & 0x7FFFFFF)];
      if ( v108 != sub_2D56670 )
      {
        v126 = *(_QWORD *)&byte_444C4A0[16 * (v31 - 1)];
        v136 = v30;
        v110 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD))v108)(
                 v82,
                 *(_QWORD *)(*(_QWORD *)v109 + 8LL),
                 *((_QWORD *)a7 + 1));
        v30 = v136;
        v32 = v126;
        if ( v110 )
          return 0;
        goto LABEL_110;
      }
LABEL_111:
      v92 = *(__int64 (**)())(v91 + 1816);
      if ( v92 == sub_2D566C0 )
      {
LABEL_28:
        v34 = *(unsigned __int8 *)(v156 + 8);
        v33 = *(unsigned __int8 *)(v155 + 8);
        goto LABEL_29;
      }
      v124 = v32;
      v134 = v30;
      if ( ((unsigned __int8 (__fastcall *)(__int64 *, unsigned __int8 *))v92)(v82, a7) )
        return 0;
      v30 = v134;
      v32 = v124;
      v34 = *(unsigned __int8 *)(v156 + 8);
      v33 = *(unsigned __int8 *)(v155 + 8);
LABEL_29:
      v36 = v33 - 17;
      v37 = v155;
      v38 = 0;
      if ( v36 > 1 )
        v37 = 0;
      if ( (unsigned int)(v34 - 17) <= 1 )
        v38 = v156;
      if ( v30 == v153
        && v151 == v150
        && *(_QWORD *)(v149 + 8LL * (int)v31 + 112)
        && v142 <= 0x1F3
        && *(_BYTE *)(v142 + v149 + 500LL * v31 + 6414) <= 1u )
      {
        return v151;
      }
      if ( !v37 )
      {
        if ( !v38 )
        {
          if ( !*(_QWORD *)(v149 + 8LL * (int)v31 + 112)
            || v142 <= 0x1F3 && *(_BYTE *)(v142 + v149 + 500LL * v31 + 6414) == 2 )
          {
            return 4;
          }
          else
          {
            return 1;
          }
        }
        if ( a2 == 49 )
          return sub_34D2080(a1, v38, 1, 0);
LABEL_207:
        BUG();
      }
      if ( !v38 )
      {
        if ( a2 == 49 )
          return sub_34D2080(a1, v37, 0, 1);
        goto LABEL_207;
      }
      if ( v30 == v153 && v32 == v148 && v151 == v150 && v141 == v146 )
      {
        if ( a2 == 39 )
          return v151;
        if ( a2 == 40 )
        {
          result = 2 * v151;
          if ( !is_mul_ok(2u, v151) )
          {
            result = 0x7FFFFFFFFFFFFFFFLL;
            if ( v151 <= 0 )
              return 0x8000000000000000LL;
          }
          return result;
        }
        if ( *(_QWORD *)(v149 + 8LL * (int)v31 + 112)
          && (v142 > 0x1F3 || *(_BYTE *)(v142 + v149 + 500LL * v31 + 6414) != 2) )
        {
          return v151;
        }
      }
      v152 = v37;
      v39 = sub_2D5BAE0(v149, *(_QWORD *)(a1 + 8), (__int64 *)v155, 0);
      sub_2FE6CC0((__int64)&v158, v149, *(_QWORD *)v155, v39, v40);
      v41 = v158;
      v42 = sub_2D5BAE0(v149, *(_QWORD *)(a1 + 8), (__int64 *)v156, 0);
      sub_2FE6CC0((__int64)&v158, v149, *(_QWORD *)v156, v42, v43);
      v44 = v158;
      if ( v41 != 6 && (_BYTE)v158 != 6 )
      {
LABEL_39:
        if ( *(_BYTE *)(v38 + 8) != 18 )
        {
          v45 = *(_DWORD *)(v38 + 32);
LABEL_41:
          if ( (unsigned int)*(unsigned __int8 *)(v155 + 8) - 17 <= 1 )
            v155 = **(_QWORD **)(v155 + 16);
          if ( (unsigned int)*(unsigned __int8 *)(v156 + 8) - 17 <= 1 )
            v156 = **(_QWORD **)(v156 + 16);
          v46 = sub_34D3270(a1, a2, v156, v155, a5, a6, (__int64)a7);
          v47 = v45;
          v49 = v45;
          v48 = v46 * v45;
          if ( !is_mul_ok(v46, v49) )
          {
            if ( !v47 || (v48 = 0x7FFFFFFFFFFFFFFFLL, v46 <= 0) )
              v48 = 0x8000000000000000LL;
          }
          v50 = sub_34D2080(a1, v38, 1, 1);
          v51 = __OFADD__(v48, v50);
          result = v48 + v50;
          if ( v51 )
          {
            result = 0x7FFFFFFFFFFFFFFFLL;
            if ( v48 <= 0 )
              return 0x8000000000000000LL;
          }
          return result;
        }
        return 0;
      }
      v96 = *(_DWORD *)(v152 + 32);
      if ( *(_BYTE *)(v152 + 8) == 18 )
      {
        if ( !v96 )
          goto LABEL_39;
      }
      else if ( v96 <= 1 )
      {
        goto LABEL_39;
      }
      v97 = *(_BYTE *)(v38 + 8);
      v45 = *(_DWORD *)(v38 + 32);
      if ( v97 == 18 )
      {
        if ( !v45 )
          return 0;
      }
      else if ( v45 <= 1 )
      {
        goto LABEL_41;
      }
      v98 = *(__int64 **)(v38 + 24);
      LODWORD(v158) = v45 >> 1;
      BYTE4(v158) = v97 == 18;
      v99 = sub_BCE1B0(v98, v158);
      v100 = *(__int64 **)(v152 + 24);
      v101 = *(_DWORD *)(v152 + 32) >> 1;
      BYTE4(v157) = *(_BYTE *)(v152 + 8) == 18;
      LODWORD(v157) = v101;
      v102 = sub_BCE1B0(v100, v157);
      v103 = v44 != 6 || v41 != 6;
      v104 = sub_34D3270(a1, a2, v99, v102, a5, a6, (__int64)a7);
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
      v51 = __OFADD__(v106, v103);
      v107 = v106 + v103;
      if ( !v51 )
        return v107;
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v106 <= 0 )
        return 0x8000000000000000LL;
      return result;
    case '1':
LABEL_27:
      if ( v30 == v153 && v151 == v150 && ((v33 & 0xFD) == 12) == ((v34 & 0xFD) == 12) && v32 == v148 && v141 == v146 )
        return 0;
      goto LABEL_28;
    case '2':
      v93 = v156;
      if ( (unsigned int)(v34 - 17) <= 1 )
        v93 = **(_QWORD **)(v156 + 16);
      v94 = *(_DWORD *)(v93 + 8) >> 8;
      v95 = v155;
      if ( (unsigned int)(v33 - 17) <= 1 )
        v95 = **(_QWORD **)(v155 + 16);
      v125 = *(_QWORD *)&byte_444C4A0[16 * (v31 - 1)];
      v135 = v30;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v149 + 992LL))(
             v149,
             *(_DWORD *)(v95 + 8) >> 8,
             v94) )
      {
        return 0;
      }
      v32 = v125;
      v30 = v135;
      v34 = *(unsigned __int8 *)(v156 + 8);
      v33 = *(unsigned __int8 *)(v155 + 8);
      goto LABEL_29;
    default:
      goto LABEL_29;
  }
}
