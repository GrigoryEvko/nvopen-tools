// Function: sub_1588490
// Address: 0x1588490
//
__int64 __fastcall sub_1588490(__int64 a1, __int64 a2, unsigned __int8 a3, int *a4, __int64 **a5, unsigned __int64 a6)
{
  __int64 result; // rax
  __int64 **v7; // r15
  __int64 v8; // rax
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 **v12; // rcx
  __int64 **v13; // rax
  __int64 v14; // rdx
  _BYTE *v15; // r12
  int v16; // ebx
  int v17; // r13d
  __int64 *v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // r14
  __int64 v22; // rdi
  __int64 **v23; // r8
  __int64 v24; // rax
  char v25; // al
  __int64 v26; // r14
  unsigned int v27; // r13d
  __int64 v28; // r15
  char v29; // dl
  char v30; // al
  unsigned __int64 v31; // r12
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  _BYTE *v36; // rdx
  __int64 v37; // r9
  __int64 *v38; // rbx
  __int64 v39; // rdi
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // r9
  size_t v43; // rdi
  __int64 v44; // rax
  __int64 v45; // r15
  _QWORD *v46; // r12
  _QWORD *v47; // r12
  unsigned int v48; // r15d
  unsigned int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdi
  _QWORD *v53; // r15
  __int64 v54; // rax
  int v55; // r12d
  unsigned int v56; // r13d
  __int64 v57; // rax
  __int64 v58; // r14
  unsigned __int64 v59; // rbx
  unsigned int v60; // eax
  int v61; // ecx
  _BYTE *v62; // r10
  __int64 v63; // rax
  bool v64; // zf
  char v65; // al
  __int64 v66; // rax
  __int64 v67; // rax
  _QWORD *v68; // r15
  __int64 v69; // r12
  unsigned int v70; // eax
  __int64 v71; // rax
  __int64 v72; // rsi
  int v73; // eax
  unsigned __int64 v74; // rdi
  __int64 *v75; // rdi
  unsigned int v76; // ebx
  unsigned int v77; // ebx
  _QWORD *v78; // rax
  _QWORD *v79; // rcx
  __int64 *v80; // rsi
  __int64 v81; // rcx
  __int64 v82; // rax
  __int64 v83; // rdi
  __int64 v84; // rdx
  __int64 v85; // rdx
  _QWORD *v86; // r14
  __int64 v87; // rdi
  __int64 v88; // rax
  __int64 v89; // r12
  _QWORD *v90; // r13
  _QWORD *v91; // r14
  __int64 v92; // rdi
  __int64 v93; // rax
  __int64 v94; // rbx
  unsigned __int64 v95; // rdi
  __int64 v96; // rax
  __int64 v97; // rbx
  char v98; // cl
  unsigned __int64 v99; // r13
  __int64 v100; // rax
  unsigned __int64 v101; // rcx
  __int64 v102; // rdx
  _QWORD *v103; // r8
  __int64 v104; // rax
  _QWORD *v105; // rbx
  unsigned __int64 v106; // r14
  _QWORD *v107; // rax
  __int64 *v108; // r14
  __int64 v109; // rdx
  unsigned int v110; // ecx
  unsigned int v111; // eax
  __int64 v112; // rsi
  __int64 v113; // rbx
  __int64 v114; // rax
  __int64 v115; // rax
  __int64 **v116; // r8
  signed __int64 v117; // r14
  __int64 v118; // rax
  __int64 **v119; // r15
  __int64 v120; // rbx
  int v121; // edx
  int v122; // ebx
  int v123; // ebx
  __int64 v124; // rax
  _BOOL4 v125; // ebx
  int v126; // r12d
  int v127; // r13d
  __int64 v128; // r14
  int v129; // eax
  __int64 v130; // rax
  __int64 v131; // [rsp+8h] [rbp-158h]
  unsigned int v132; // [rsp+14h] [rbp-14Ch]
  __int64 v133; // [rsp+28h] [rbp-138h]
  bool v136; // [rsp+47h] [rbp-119h]
  bool v137; // [rsp+47h] [rbp-119h]
  __int64 v138; // [rsp+48h] [rbp-118h]
  __int64 v139; // [rsp+48h] [rbp-118h]
  char v141; // [rsp+58h] [rbp-108h]
  int v142; // [rsp+58h] [rbp-108h]
  __int64 **v143; // [rsp+60h] [rbp-100h]
  bool v144; // [rsp+68h] [rbp-F8h]
  __int64 v145; // [rsp+68h] [rbp-F8h]
  __int64 v146; // [rsp+70h] [rbp-F0h]
  __int64 v147; // [rsp+70h] [rbp-F0h]
  __int64 v148; // [rsp+70h] [rbp-F0h]
  __int64 v149; // [rsp+70h] [rbp-F0h]
  __int64 v150; // [rsp+70h] [rbp-F0h]
  __int64 v151; // [rsp+70h] [rbp-F0h]
  __int64 *v152; // [rsp+78h] [rbp-E8h]
  __int64 v153; // [rsp+78h] [rbp-E8h]
  __int64 v154; // [rsp+78h] [rbp-E8h]
  __int64 **v156; // [rsp+80h] [rbp-E0h]
  __int64 v157; // [rsp+88h] [rbp-D8h]
  int v158; // [rsp+98h] [rbp-C8h] BYREF
  char v159; // [rsp+9Ch] [rbp-C4h]
  _BYTE *v160; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v161; // [rsp+A8h] [rbp-B8h]
  _BYTE v162[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v157 = a2;
  if ( !a6 )
    return v157;
  v7 = a5;
  v8 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    v8 = **(_QWORD **)(v8 + 16);
  v9 = *(_DWORD *)(v8 + 8) >> 8;
  v10 = sub_15F9F50(*(_QWORD *)(v8 + 24), a5, a6);
  v11 = sub_1646BA0(v10, v9);
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
  {
    v11 = sub_16463B0(v11, *(_QWORD *)(*(_QWORD *)a2 + 32LL));
  }
  else
  {
    v12 = &v7[a6];
    if ( v7 != v12 )
    {
      v13 = v7;
      while ( 1 )
      {
        v14 = **v13;
        if ( *(_BYTE *)(v14 + 8) == 16 )
          break;
        if ( v12 == ++v13 )
          goto LABEL_12;
      }
      v11 = sub_16463B0(v11, *(_QWORD *)(v14 + 32));
    }
  }
LABEL_12:
  if ( *(_BYTE *)(a2 + 16) == 9 )
    return sub_1599EF0(v11);
  v15 = *v7;
  if ( a6 != 1 )
  {
    if ( !(unsigned __int8)sub_1593BB0(a2) )
      goto LABEL_29;
    v16 = a6;
    if ( !(_DWORD)a6 )
    {
      v17 = 0;
LABEL_20:
      v19 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
        v19 = **(_QWORD **)(v19 + 16);
      v20 = sub_15F9F50(a1, v7, a6);
      v21 = sub_1646BA0(v20, *(_DWORD *)(v19 + 8) >> 8);
      v22 = sub_1646BA0(v20, *(_DWORD *)(v19 + 8) >> 8);
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
        v22 = sub_16463B0(v21, *(_QWORD *)(*(_QWORD *)a2 + 32LL));
      if ( v17 )
      {
        v23 = v7;
        while ( 1 )
        {
          v24 = **v23;
          if ( *(_BYTE *)(v24 + 8) == 16 )
            break;
          if ( &v7[(unsigned int)(v17 - 1) + 1] == ++v23 )
            return sub_15A06D0(v22);
        }
        v22 = sub_16463B0(v21, *(_QWORD *)(v24 + 32));
      }
      return sub_15A06D0(v22);
    }
    goto LABEL_16;
  }
  if ( (unsigned __int8)sub_1593BB0(v15) || v15[16] == 9 )
  {
    if ( *(_BYTE *)(v11 + 8) == 16 && *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
      return sub_15A0390(*(_QWORD *)(v11 + 32));
    return v157;
  }
  if ( (unsigned __int8)sub_1593BB0(a2) )
  {
    v16 = 1;
LABEL_16:
    v17 = 0;
    while ( 1 )
    {
      v18 = v7[v17];
      if ( *((_BYTE *)v18 + 16) != 9 && !(unsigned __int8)sub_1593BB0(v18) )
        break;
      if ( v16 == ++v17 )
        goto LABEL_20;
    }
  }
LABEL_29:
  if ( *(_BYTE *)(a2 + 16) != 5 )
    goto LABEL_33;
  if ( *(_WORD *)(a2 + 18) == 32 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v85 = *(_QWORD *)(a2 - 8);
    else
      v85 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v86 = (_QWORD *)(v85 + 24);
    v87 = sub_16348C0(a2) | 4;
    v88 = a2;
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v88 = *(_QWORD *)(a2 - 8) + 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    if ( v86 == (_QWORD *)v88 )
    {
      if ( !(unsigned __int8)sub_1593BB0(v15) )
        goto LABEL_31;
    }
    else
    {
      v153 = (__int64)v15;
      v89 = -1;
      v90 = v86;
      v91 = (_QWORD *)v88;
      while ( 1 )
      {
        v94 = v87;
        v95 = v87 & 0xFFFFFFFFFFFFFFF8LL;
        v96 = v95;
        v97 = (v94 >> 2) & 1;
        if ( !(_DWORD)v97 || !v95 )
          v96 = sub_1643D30(v95, *v90);
        v98 = *(_BYTE *)(v96 + 8);
        if ( ((v98 - 14) & 0xFD) != 0 )
        {
          v87 = 0;
          if ( v98 == 13 )
            v87 = v96;
          v93 = v89;
        }
        else
        {
          v92 = *(_QWORD *)(v96 + 24);
          v93 = *(_QWORD *)(v96 + 32);
          v87 = v92 | 4;
        }
        v90 += 3;
        if ( v91 == v90 )
          break;
        v89 = v93;
      }
      v99 = v89;
      v15 = (_BYTE *)v153;
      if ( !(unsigned __int8)sub_1593BB0(v153) )
      {
        if ( !(_BYTE)v97 || *(_BYTE *)(v153 + 16) != 13 || v99 != -1 && !(unsigned __int8)sub_1581070(v99, v153) )
          goto LABEL_31;
        v100 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        if ( *(_BYTE *)(**(_QWORD **)(a2 + 24 * ((unsigned int)(v100 - 1) - v100)) + 8LL) == 16 )
          goto LABEL_31;
LABEL_183:
        v101 = 16;
        v102 = 0;
        v160 = v162;
        v161 = 0x1000000000LL;
        if ( a6 + v100 > 0x10 )
        {
          sub_16CD150(&v160, v162, a6 + v100, 8);
          v102 = (unsigned int)v161;
          v101 = HIDWORD(v161) - (unsigned __int64)(unsigned int)v161;
          v100 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        }
        v103 = (_QWORD *)(a2 - 24);
        v104 = 24 * (1 - v100);
        v105 = (_QWORD *)(v104 + a2);
        v106 = 0xAAAAAAAAAAAAAAABLL * ((-24 - v104) >> 3);
        if ( v106 > v101 )
        {
          sub_16CD150(&v160, v162, v106 + v102, 8);
          v102 = (unsigned int)v161;
          v103 = (_QWORD *)(a2 - 24);
        }
        v107 = &v160[8 * v102];
        if ( v103 != v105 )
        {
          do
          {
            if ( v107 )
              *v107 = *v105;
            v105 += 3;
            ++v107;
          }
          while ( v103 != v105 );
          LODWORD(v102) = v161;
        }
        LODWORD(v161) = v106 + v102;
        v108 = *(__int64 **)(a2
                           + 24
                           * ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu)
                            - 1
                            - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        if ( !(unsigned __int8)sub_1593BB0(v15) )
        {
          v109 = *v108;
          if ( *v108 == *(_QWORD *)v15 )
          {
            v108 = (__int64 *)sub_15A2A30(11, v15, v108, 0, 0);
          }
          else
          {
            v110 = *(_DWORD *)(*(_QWORD *)v15 + 8LL);
            v111 = v110 >> 8;
            if ( v110 <= 0x3FFF )
              v111 = 64;
            v112 = *(_DWORD *)(v109 + 8) >> 8;
            if ( v111 >= (unsigned int)v112 )
              v112 = v111;
            v154 = sub_1644C60(*(_QWORD *)v109, v112);
            v113 = sub_15A4620(v15, v154);
            v114 = sub_15A4620(v108, v154);
            v108 = (__int64 *)sub_15A2A30(11, v113, v114, 0, 0);
          }
        }
        v115 = (unsigned int)v161;
        if ( (unsigned int)v161 >= HIDWORD(v161) )
        {
          sub_16CD150(&v160, v162, 0, 8);
          v115 = (unsigned int)v161;
        }
        *(_QWORD *)&v160[8 * v115] = v108;
        v116 = &v7[a6];
        v117 = 8 * a6 - 8;
        v119 = v7 + 1;
        LODWORD(v161) = v161 + 1;
        v118 = (unsigned int)v161;
        v120 = v117 >> 3;
        if ( v117 >> 3 > HIDWORD(v161) - (unsigned __int64)(unsigned int)v161 )
        {
          v156 = v116;
          sub_16CD150(&v160, v162, v120 + (unsigned int)v161, 8);
          v118 = (unsigned int)v161;
          v116 = v156;
        }
        if ( v119 != v116 )
        {
          memcpy(&v160[8 * v118], v119, v117);
          LODWORD(v118) = v161;
        }
        LODWORD(v161) = v120 + v118;
        v121 = v120 + v118;
        v122 = *(_BYTE *)(v157 + 17) >> 1 >> 1;
        if ( v122 )
        {
          v123 = v122 - 1;
          v124 = *(_DWORD *)(v157 + 20) & 0xFFFFFFF;
          if ( (_DWORD)v124 - 2 != v123 )
          {
LABEL_207:
            v159 = 1;
            v158 = v123;
            goto LABEL_208;
          }
          if ( (unsigned __int8)sub_1593BB0(v15) )
          {
            v121 = v161;
            v124 = *(_DWORD *)(v157 + 20) & 0xFFFFFFF;
            goto LABEL_207;
          }
          v121 = v161;
        }
        v159 = 0;
        v124 = *(_DWORD *)(v157 + 20) & 0xFFFFFFF;
LABEL_208:
        v125 = 0;
        if ( a3 )
          v125 = (*(_BYTE *)(v157 + 17) & 2) != 0;
        v126 = v121;
        v127 = (int)v160;
        v128 = *(_QWORD *)(v157 - 24 * v124);
        v129 = sub_16348C0(v157);
        result = sub_15A2E80(v129, v128, v127, v126, v125, (unsigned int)&v158, 0);
        v74 = (unsigned __int64)v160;
        if ( v160 != v162 )
          goto LABEL_119;
        return result;
      }
    }
    v100 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    goto LABEL_183;
  }
LABEL_31:
  v25 = sub_1594510(a2);
  if ( a6 != 1 && v25 && (unsigned __int8)sub_1593BB0(v15) )
  {
    v80 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v81 = *(_QWORD *)v157;
    v82 = *v80;
    v146 = *(_QWORD *)v157;
    if ( *(_BYTE *)(*v80 + 8) == 15 && *(_BYTE *)(v81 + 8) == 15 )
    {
      v83 = *(_QWORD *)(v82 + 24);
      if ( *(_BYTE *)(v83 + 8) == 14 )
      {
        v84 = *(_QWORD *)(v81 + 24);
        if ( *(_BYTE *)(v84 + 8) == 14
          && *(_QWORD *)(v84 + 24) == *(_QWORD *)(v83 + 24)
          && *(_DWORD *)(v146 + 8) >> 8 == *(_DWORD *)(v82 + 8) >> 8 )
        {
          BYTE4(v160) = *((_BYTE *)a4 + 4);
          if ( BYTE4(v160) )
            LODWORD(v160) = *a4;
          return sub_15A2E80(v83, (_DWORD)v80, (_DWORD)v7, a6, a3, (unsigned int)&v160, 0);
        }
      }
    }
    goto LABEL_34;
  }
LABEL_33:
  v146 = *(_QWORD *)a2;
LABEL_34:
  v160 = v162;
  v161 = 0x800000000LL;
  v136 = (unsigned __int8)(*((_BYTE *)*v7 + 16) - 12) > 1u;
  if ( (_DWORD)a6 == 1 )
    goto LABEL_120;
  v143 = v7;
  v26 = a1;
  v27 = 1;
  v131 = 8 * a6;
  while ( 1 )
  {
    v138 = v27;
    v28 = (__int64)v143[v138];
    v152 = (__int64 *)&v143[v138];
    v29 = *(_BYTE *)(v28 + 16);
    if ( (unsigned __int8)(v29 - 12) > 1u )
    {
LABEL_77:
      v136 = 1;
      goto LABEL_75;
    }
    v133 = v27 - 1;
    if ( (unsigned __int8)(*((_BYTE *)v143[v133] + 16) - 12) <= 1u && (!*((_BYTE *)a4 + 4) || *a4 + 1 != v27) )
    {
      v30 = *(_BYTE *)(v26 + 8);
      if ( v30 != 13 )
      {
        if ( v30 == 16 )
          goto LABEL_77;
        if ( v29 == 13 )
        {
          v31 = *(_QWORD *)(v26 + 32);
          if ( (unsigned __int8)sub_1581070(v31, v28) )
            goto LABEL_75;
          v32 = *(_DWORD *)(v28 + 32);
          v33 = *(__int64 **)(v28 + 24);
          if ( v32 > 0x40 )
            v34 = *v33;
          else
            v34 = (__int64)((_QWORD)v33 << (64 - (unsigned __int8)v32)) >> (64 - (unsigned __int8)v32);
          if ( v34 < 0 )
            goto LABEL_77;
          v144 = v136;
LABEL_48:
          if ( *(_BYTE *)(v146 + 8) == 13 )
          {
            v136 = 1;
            v28 = *v152;
            goto LABEL_75;
          }
          v35 = (unsigned int)v161;
          if ( a6 >= (unsigned int)v161 )
          {
            if ( a6 <= (unsigned int)v161 )
            {
              v36 = v160;
            }
            else
            {
              if ( a6 > HIDWORD(v161) )
              {
                sub_16CD150(&v160, v162, a6, 8);
                v35 = (unsigned int)v161;
              }
              v36 = v160;
              v78 = &v160[8 * v35];
              v79 = &v160[v131];
              if ( v78 != (_QWORD *)&v160[v131] )
              {
                do
                {
                  if ( v78 )
                    *v78 = 0;
                  ++v78;
                }
                while ( v79 != v78 );
                v36 = v160;
              }
              LODWORD(v161) = a6;
              v31 = *(_QWORD *)(v26 + 32);
            }
          }
          else
          {
            v36 = v160;
            LODWORD(v161) = a6;
            v31 = *(_QWORD *)(v26 + 32);
          }
          v37 = *v152;
          v38 = *(__int64 **)&v36[8 * v133];
          if ( !v38 )
            v38 = v143[v133];
          v39 = *(_QWORD *)v37;
          v40 = *(_BYTE *)(*(_QWORD *)v37 + 8LL);
          v141 = *(_BYTE *)(*v38 + 8);
          v137 = v40 == 16 || v141 == 16;
          if ( v40 == 16 || v141 != 16 )
          {
            if ( v141 == 16 || v40 != 16 )
            {
              if ( v40 == 16 )
                goto LABEL_59;
              goto LABEL_60;
            }
            v150 = *v152;
            v66 = sub_15A1590(*(_QWORD *)(v39 + 32));
            v37 = v150;
            v38 = (__int64 *)v66;
            v39 = *(_QWORD *)v150;
            if ( *(_BYTE *)(*(_QWORD *)v150 + 8LL) == 16 )
              goto LABEL_59;
            sub_15A0680(v39, v31, 0);
            v42 = v150;
          }
          else
          {
            v41 = sub_15A1590(*(_QWORD *)(*v38 + 32));
            v39 = *(_QWORD *)v41;
            v37 = v41;
            if ( *(_BYTE *)(*(_QWORD *)v41 + 8LL) != 16 )
            {
              v147 = v41;
              sub_15A0680(v39, v31, 0);
              v42 = v147;
              goto LABEL_57;
            }
LABEL_59:
            v39 = **(_QWORD **)(v39 + 16);
LABEL_60:
            v148 = v37;
            v44 = sub_15A0680(v39, v31, 0);
            v42 = v148;
            v45 = v44;
            if ( !v137 )
            {
              v46 = &v160[v138 * 8];
              *v46 = sub_15A2CD0(v148, v44);
              v47 = (_QWORD *)sub_15A2C90(v148, v45, 0);
              v48 = sub_16431D0(*v47);
              v49 = sub_16431D0(*v38);
              if ( v48 < v49 )
                v48 = v49;
              if ( v48 < 0x40 )
                v48 = 64;
              v50 = sub_16498A0(v47);
              v149 = sub_1644C60(v50, v48);
              goto LABEL_66;
            }
          }
          if ( v141 == 16 )
LABEL_57:
            v43 = *(unsigned int *)(*v38 + 32);
          else
            v43 = *(unsigned int *)(*(_QWORD *)v42 + 32LL);
          v151 = v42;
          v67 = sub_15A1590(v43);
          v68 = &v160[v138 * 8];
          v69 = v67;
          *v68 = sub_15A2CD0(v151, v67);
          v139 = v151;
          v47 = (_QWORD *)sub_15A2C90(v151, v69, 0);
          v48 = sub_16431D0(*v47);
          v70 = sub_16431D0(*v38);
          if ( v48 < v70 )
            v48 = v70;
          if ( v48 < 0x40 )
            v48 = 64;
          v71 = sub_16498A0(v47);
          v149 = sub_1644C60(v71, v48);
          if ( v137 )
          {
            if ( v141 == 16 )
              v72 = *(unsigned int *)(*v38 + 32);
            else
              v72 = *(unsigned int *)(*(_QWORD *)v139 + 32LL);
            v149 = sub_16463B0(v149, v72);
          }
LABEL_66:
          v51 = *v38;
          if ( *(_BYTE *)(*v38 + 8) == 16 )
            v51 = **(_QWORD **)(v51 + 16);
          if ( !(unsigned __int8)sub_1642F90(v51, v48) )
            v38 = (__int64 *)sub_15A4460(v38, v149, 0);
          v52 = *v47;
          if ( *(_BYTE *)(*v47 + 8LL) == 16 )
            v52 = **(_QWORD **)(v52 + 16);
          if ( !(unsigned __int8)sub_1642F90(v52, v48) )
            v47 = (_QWORD *)sub_15A4460(v47, v149, 0);
          v53 = &v160[8 * v133];
          *v53 = sub_15A2B30(v38, v47, 0, 0);
          v28 = *v152;
          v136 = v144;
          goto LABEL_75;
        }
        v142 = sub_15958F0(v28);
        if ( v142 )
        {
          v132 = v27;
          v55 = 1;
          v56 = 0;
          v145 = v26;
          do
          {
            v58 = sub_15A0940(v28, v56);
            v59 = *(_QWORD *)(v145 + 32);
            v55 &= sub_1581070(v59, v58);
            v60 = *(_DWORD *)(v58 + 32);
            if ( v60 <= 0x40 )
              v57 = (__int64)(*(_QWORD *)(v58 + 24) << (64 - (unsigned __int8)v60)) >> (64 - (unsigned __int8)v60);
            else
              v57 = **(_QWORD **)(v58 + 24);
            if ( v57 < 0 )
            {
              v136 = 1;
              v27 = v132;
              v26 = v145;
              v28 = *v152;
              goto LABEL_75;
            }
            ++v56;
          }
          while ( v142 != v56 );
          v65 = v136 | v55;
          v64 = (v136 | (unsigned __int8)v55) == 0;
          v31 = v59;
          v26 = v145;
          v27 = v132;
          v144 = v65;
          if ( v64 )
            goto LABEL_48;
        }
        v28 = *v152;
      }
    }
LABEL_75:
    ++v27;
    v54 = sub_1643D30(v26, v28);
    v146 = v26;
    if ( (_DWORD)a6 == v27 )
      break;
    v26 = v54;
  }
  v61 = v161;
  v7 = v143;
  if ( (_DWORD)v161 )
  {
    v62 = v160;
    if ( (_DWORD)a6 )
    {
      v63 = 0;
      do
      {
        if ( !*(_QWORD *)&v62[v63 * 8] )
        {
          *(_QWORD *)&v62[v63 * 8] = v143[v63];
          v62 = v160;
        }
        ++v63;
      }
      while ( v63 != (unsigned int)a6 );
      v61 = v161;
    }
    if ( *((_BYTE *)a4 + 4) )
    {
      v73 = *a4;
      v159 = 1;
      v158 = v73;
    }
    else
    {
      v159 = 0;
    }
    result = sub_15A2E80(a1, v157, (_DWORD)v62, v61, a3, (unsigned int)&v158, 0);
    goto LABEL_118;
  }
LABEL_120:
  if ( a3 || v136 || *(_BYTE *)(v157 + 16) != 3 || (*(_BYTE *)(v157 + 32) & 0xF) == 9 )
    goto LABEL_132;
  if ( !(unsigned __int8)sub_1593BB0(*v7) )
  {
    v75 = *v7;
    if ( *((_BYTE *)*v7 + 16) == 13 )
    {
      v76 = *((_DWORD *)v75 + 8);
      if ( v76 > 0x40 )
      {
        if ( (unsigned int)sub_16A57B0(v75 + 3) == v76 - 1 )
          goto LABEL_128;
LABEL_132:
        result = 0;
        goto LABEL_118;
      }
      if ( v75[3] != 1 )
      {
        result = 0;
        goto LABEL_118;
      }
    }
    else
    {
      v130 = sub_15A0FF0();
      if ( !v130 || *(_BYTE *)(v130 + 16) != 13 )
        goto LABEL_132;
      if ( !sub_1455000(v130 + 24) )
      {
        result = 0;
        goto LABEL_118;
      }
    }
LABEL_128:
    if ( (_DWORD)a6 != 1 )
    {
      v77 = 1;
      while ( (unsigned __int8)sub_1593BB0(v7[v77]) )
      {
        if ( (_DWORD)a6 == ++v77 )
          goto LABEL_219;
      }
      goto LABEL_132;
    }
  }
LABEL_219:
  v159 = *((_BYTE *)a4 + 4);
  if ( v159 )
    v158 = *a4;
  result = sub_15A2E80(a1, v157, (_DWORD)v7, a6, 1, (unsigned int)&v158, 0);
LABEL_118:
  v74 = (unsigned __int64)v160;
  if ( v160 != v162 )
  {
LABEL_119:
    v157 = result;
    _libc_free(v74);
    return v157;
  }
  return result;
}
