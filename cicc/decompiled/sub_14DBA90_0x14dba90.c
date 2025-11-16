// Function: sub_14DBA90
// Address: 0x14dba90
//
__int64 __fastcall sub_14DBA90(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r14
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r12
  unsigned int v21; // r13d
  __int64 v22; // rdx
  unsigned int v23; // eax
  __int64 *v24; // rcx
  __int64 *v25; // r9
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 **v28; // r15
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 *v32; // rsi
  __int64 v33; // r14
  __int64 *v34; // r15
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 *v37; // rbx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rsi
  __int64 v42; // rsi
  int v43; // r13d
  __int64 v44; // r14
  unsigned int v45; // r14d
  __int64 v46; // rsi
  __int64 i; // r12
  unsigned __int8 v48; // al
  __int64 v49; // r15
  unsigned int v50; // ebx
  _QWORD *v51; // rax
  _DWORD *v52; // rdx
  int v53; // ecx
  __int64 v54; // rax
  _DWORD *v55; // rdi
  __int64 v56; // rsi
  __int64 v57; // rax
  _DWORD *v58; // rax
  char v59; // al
  __int64 v60; // r13
  __int64 v61; // rsi
  unsigned __int64 v62; // r12
  __int64 v63; // rax
  unsigned __int64 v64; // r13
  _QWORD *v65; // rax
  __int64 v66; // rbx
  __int64 v67; // rax
  int v68; // r12d
  unsigned int v69; // ebx
  unsigned __int64 v70; // rdx
  unsigned int v71; // ebx
  unsigned __int64 v72; // rax
  __int64 v73; // rax
  unsigned __int64 v74; // rbx
  __int64 v75; // r12
  unsigned __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // r13
  _QWORD *v80; // r9
  __int64 v81; // rax
  _QWORD *v82; // r13
  unsigned __int64 v83; // r10
  _BYTE *v84; // rdi
  int v85; // edx
  _QWORD *v86; // rax
  _BYTE *v87; // rcx
  _BYTE *v88; // rax
  __int64 **v89; // rax
  __int64 *v90; // r13
  __int64 v91; // rax
  __int64 v92; // r13
  __int64 v93; // rbx
  __int64 v94; // rax
  unsigned int v95; // r13d
  unsigned __int64 v96; // rax
  unsigned __int64 *v97; // r8
  __int64 v98; // rax
  unsigned __int64 v99; // rsi
  int v100; // ebx
  __int64 v101; // rax
  int v102; // edi
  int v103; // ecx
  int v104; // edx
  int v105; // r8d
  __int64 v106; // rdx
  __int64 v107; // rbx
  int v108; // eax
  unsigned int v109; // ebx
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // r9
  unsigned __int64 *v113; // r8
  __int64 v114; // r10
  __int64 v115; // rax
  unsigned __int64 v116; // rsi
  unsigned int v117; // eax
  __int64 v118; // rax
  __int64 v119; // r12
  __int64 v120; // rax
  _QWORD *v121; // r8
  __int64 v122; // rsi
  __int64 v123; // r9
  int v124; // [rsp+0h] [rbp-210h]
  char v125; // [rsp+7h] [rbp-209h]
  __int64 v126; // [rsp+8h] [rbp-208h]
  unsigned int v128; // [rsp+18h] [rbp-1F8h]
  __int64 v129; // [rsp+20h] [rbp-1F0h]
  char v130; // [rsp+28h] [rbp-1E8h]
  __int64 v131; // [rsp+28h] [rbp-1E8h]
  _QWORD *v132; // [rsp+28h] [rbp-1E8h]
  unsigned __int64 *v133; // [rsp+28h] [rbp-1E8h]
  unsigned __int64 *v134; // [rsp+28h] [rbp-1E8h]
  int v136; // [rsp+30h] [rbp-1E0h]
  __int64 v137; // [rsp+38h] [rbp-1D8h]
  unsigned __int64 v138; // [rsp+38h] [rbp-1D8h]
  __int64 v140; // [rsp+50h] [rbp-1C0h]
  unsigned __int64 v141; // [rsp+50h] [rbp-1C0h]
  __int64 v142; // [rsp+58h] [rbp-1B8h]
  unsigned __int8 v143; // [rsp+58h] [rbp-1B8h]
  char v144; // [rsp+60h] [rbp-1B0h]
  __int64 *v145; // [rsp+60h] [rbp-1B0h]
  __int64 v146; // [rsp+60h] [rbp-1B0h]
  __int64 v147; // [rsp+68h] [rbp-1A8h]
  __int64 v148; // [rsp+68h] [rbp-1A8h]
  unsigned __int64 *v149; // [rsp+68h] [rbp-1A8h]
  __int64 v150; // [rsp+68h] [rbp-1A8h]
  char v151; // [rsp+77h] [rbp-199h] BYREF
  __int64 v152; // [rsp+78h] [rbp-198h] BYREF
  unsigned __int64 v153; // [rsp+80h] [rbp-190h] BYREF
  unsigned int v154; // [rsp+88h] [rbp-188h]
  _QWORD *v155; // [rsp+90h] [rbp-180h] BYREF
  unsigned int v156; // [rsp+98h] [rbp-178h]
  _QWORD *v157; // [rsp+A0h] [rbp-170h] BYREF
  unsigned int v158; // [rsp+A8h] [rbp-168h]
  __int64 v159; // [rsp+B0h] [rbp-160h] BYREF
  unsigned int v160; // [rsp+B8h] [rbp-158h]
  unsigned __int64 v161; // [rsp+C0h] [rbp-150h] BYREF
  unsigned int v162; // [rsp+C8h] [rbp-148h]
  _BYTE *v163; // [rsp+D0h] [rbp-140h] BYREF
  __int64 v164; // [rsp+D8h] [rbp-138h]
  _BYTE v165[304]; // [rsp+E0h] [rbp-130h] BYREF

  v137 = a1;
  v125 = *(_BYTE *)(a1 + 17) >> 1;
  v152 = sub_16348C0(a1);
  v126 = sub_16348E0(a1);
  v9 = v152;
  v129 = *(_QWORD *)a1;
  v10 = *(unsigned __int8 *)(v152 + 8);
  if ( (unsigned __int8)v10 > 0xFu || (v11 = 35454, !_bittest64(&v11, v10)) )
  {
    if ( (unsigned int)(v10 - 13) > 1 && (_DWORD)v10 != 16 || !(unsigned __int8)sub_16435F0(v152, 0) )
      return 0;
    v9 = v152;
  }
  v130 = 0;
  v14 = *(_BYTE *)(a1 + 17) >> 1 >> 1;
  if ( v14 )
  {
    v130 = 1;
    v124 = v14 - 1;
  }
  v15 = sub_15A9650(a4, v129, v11, v6, v7, v8);
  v140 = v15;
  v20 = v15;
  if ( *(_BYTE *)(v15 + 8) == 16 )
    v140 = **(_QWORD **)(v15 + 16);
  v163 = v165;
  v164 = 0x2000000000LL;
  if ( (_DWORD)a3 != 1 )
  {
    v144 = 0;
    v21 = 1;
    while ( 1 )
    {
      while ( 1 )
      {
        v28 = &a2[v21];
        if ( v21 != 1 && *(_BYTE *)(sub_15FA030(v9, a2 + 1, v21 - 1) + 8) == 13 )
          goto LABEL_24;
        v25 = *v28;
        v29 = **v28;
        if ( *(_BYTE *)(v29 + 8) != 16 )
          break;
        v22 = v20;
        if ( **(_QWORD **)(v29 + 16) == v140 )
          goto LABEL_24;
LABEL_17:
        v142 = v22;
        v145 = *v28;
        v23 = sub_15FBEB0(*v28, 1, v22, 1);
        v26 = sub_15A46C0(v23, v145, v142, 0);
        v27 = (unsigned int)v164;
        if ( (unsigned int)v164 >= HIDWORD(v164) )
        {
          sub_16CD150(&v163, v165, 0, 8);
          v27 = (unsigned int)v164;
        }
        v144 = 1;
        ++v21;
        *(_QWORD *)&v163[8 * v27] = v26;
        LODWORD(v164) = v164 + 1;
        if ( (_DWORD)a3 == v21 )
        {
LABEL_27:
          v31 = (unsigned __int64)v163;
          if ( v144 )
          {
            v32 = *a2;
            if ( v130 )
            {
              BYTE4(v161) = 1;
              LODWORD(v161) = v124;
            }
            else
            {
              BYTE4(v161) = 0;
            }
            v33 = sub_15A2E80(v9, (_DWORD)v32, (_DWORD)v163, v164, 0, (unsigned int)&v161, 0);
            v12 = sub_14DBA30(v33, a4, a5);
            if ( v12 )
            {
              if ( v163 == v165 )
                return v12;
LABEL_32:
              _libc_free((unsigned __int64)v163);
            }
            else
            {
              v12 = v33;
              if ( v163 != v165 )
                goto LABEL_32;
            }
            if ( v12 )
              return v12;
LABEL_34:
            v34 = *a2;
            v35 = **a2;
            if ( *(_BYTE *)(v35 + 8) == 15 )
              goto LABEL_35;
          }
          else
          {
            if ( v163 == v165 )
              goto LABEL_34;
            _libc_free((unsigned __int64)v163);
            v34 = *a2;
            v35 = **a2;
            if ( *(_BYTE *)(v35 + 8) == 15 )
            {
LABEL_35:
              v146 = sub_15A9650(a4, v35, v31, v24, v18, v25);
              v36 = 1;
              while ( *((_BYTE *)a2[v36] + 16) == 13 )
              {
                if ( ++v36 == (_DWORD)a3 )
                  goto LABEL_52;
              }
              if ( a3 == 2 )
              {
                if ( (unsigned __int8)sub_1642F90(v126, 8) )
                {
                  v37 = a2[1];
                  if ( *((_BYTE *)v37 + 16) == 5
                    && *((_WORD *)v37 + 9) == 13
                    && (unsigned __int8)sub_1593BB0(v37[-3 * (*((_DWORD *)v37 + 5) & 0xFFFFFFF)]) )
                  {
                    v38 = sub_15A4180(v34, *v37, 0);
                    v39 = sub_15A2B60(v38, v37[3 * (1LL - (*((_DWORD *)v37 + 5) & 0xFFFFFFF))], 0, 0);
                    v40 = sub_15A3BA0(v39, v129, 0);
                    v12 = sub_14DBA30(v40, a4, a5);
                    if ( !v12 )
                      return v40;
                    return v12;
                  }
                }
              }
            }
          }
          return 0;
        }
      }
      if ( v29 != v140 )
      {
        v22 = v20;
        if ( *(_BYTE *)(v20 + 8) == 16 )
          v22 = **(_QWORD **)(v20 + 16);
        goto LABEL_17;
      }
LABEL_24:
      v30 = (unsigned int)v164;
      if ( (unsigned int)v164 >= HIDWORD(v164) )
      {
        sub_16CD150(&v163, v165, 0, 8);
        v30 = (unsigned int)v164;
      }
      v24 = *v28;
      ++v21;
      *(_QWORD *)&v163[8 * v30] = *v28;
      LODWORD(v164) = v164 + 1;
      if ( (_DWORD)a3 == v21 )
        goto LABEL_27;
    }
  }
  v34 = *a2;
  v41 = **a2;
  if ( *(_BYTE *)(v41 + 8) != 15 )
    return 0;
  v146 = sub_15A9650(a4, v41, v16, v17, v18, v19);
LABEL_52:
  v42 = v146;
  v43 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v42 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v77 = *(_QWORD *)(v42 + 32);
        v42 = *(_QWORD *)(v42 + 24);
        v43 *= (_DWORD)v77;
        continue;
      case 1:
        LODWORD(v44) = 16;
        break;
      case 2:
        LODWORD(v44) = 32;
        break;
      case 3:
      case 9:
        LODWORD(v44) = 64;
        break;
      case 4:
        LODWORD(v44) = 80;
        break;
      case 5:
      case 6:
        LODWORD(v44) = 128;
        break;
      case 7:
        LODWORD(v44) = 8 * sub_15A9520(a4, 0);
        break;
      case 0xB:
        LODWORD(v44) = *(_DWORD *)(v42 + 8) >> 8;
        break;
      case 0xD:
        v44 = 8LL * *(_QWORD *)sub_15A9930(a4, v42);
        break;
      case 0xE:
        v75 = *(_QWORD *)(v42 + 32);
        v148 = *(_QWORD *)(v42 + 24);
        v76 = (unsigned int)sub_15A9FE0(a4, v148);
        v44 = 8 * v75 * v76 * ((v76 + ((unsigned __int64)(sub_127FA20(a4, v148) + 7) >> 3) - 1) / v76);
        break;
      case 0xF:
        LODWORD(v44) = 8 * sub_15A9520(a4, *(_DWORD *)(v42 + 8) >> 8);
        break;
    }
    break;
  }
  v45 = v43 * v44;
  v154 = v45;
  v46 = sub_15A9FF0(a4, v152, a2 + 1, a3 - 1);
  v141 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v45;
  if ( v45 > 0x40 )
    sub_16A4EF0(&v153, v46, 0);
  else
    v153 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v45) & v46;
  v143 = v125 & 1;
  for ( i = sub_14D1290(v34, &v152); ; i = v92 )
  {
    v48 = *(_BYTE *)(i + 16);
    if ( v48 <= 0x17u )
    {
      if ( v48 != 5 || *(_WORD *)(i + 18) != 32 )
        goto LABEL_63;
    }
    else if ( v48 != 56 )
    {
      goto LABEL_63;
    }
    v143 &= *(_BYTE *)(i + 17) >> 1;
    v78 = 24LL * (*(_DWORD *)(i + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(i + 23) & 0x40) != 0 )
    {
      v79 = *(_QWORD *)(i - 8);
      v80 = (_QWORD *)(v79 + v78);
    }
    else
    {
      v80 = (_QWORD *)i;
      v79 = i - v78;
    }
    v81 = v78 - 24;
    v82 = (_QWORD *)(v79 + 24);
    v164 = 0x400000000LL;
    v163 = v165;
    v83 = 0xAAAAAAAAAAAAAAABLL * (v81 >> 3);
    if ( (unsigned __int64)v81 > 0x60 )
    {
      v132 = v80;
      v138 = 0xAAAAAAAAAAAAAAABLL * (v81 >> 3);
      sub_16CD150(&v163, v165, v138, 8);
      v84 = v163;
      v85 = v164;
      LODWORD(v83) = v138;
      v80 = v132;
      v86 = &v163[8 * (unsigned int)v164];
    }
    else
    {
      v84 = v165;
      v85 = 0;
      v86 = v165;
    }
    if ( v80 != v82 )
    {
      do
      {
        if ( v86 )
          *v86 = *v82;
        v82 += 3;
        ++v86;
      }
      while ( v80 != v82 );
      v84 = v163;
      v85 = v164;
    }
    LODWORD(v164) = v85 + v83;
    v87 = &v84[8 * (unsigned int)(v85 + v83)];
    if ( v87 != v84 )
      break;
LABEL_157:
    if ( (*(_BYTE *)(i + 23) & 0x40) != 0 )
      v89 = *(__int64 ***)(i - 8);
    else
      v89 = (__int64 **)(i - 24LL * (*(_DWORD *)(i + 20) & 0xFFFFFFF));
    v90 = *v89;
    v152 = sub_16348C0(i);
    v91 = sub_15A9FF0(a4, v152, v163, (unsigned int)v164);
    v162 = v45;
    if ( v45 > 0x40 )
      sub_16A4EF0(&v161, v91, 0);
    else
      v161 = v141 & v91;
    sub_16A7200(&v153, &v161);
    if ( v162 > 0x40 && v161 )
      j_j___libc_free_0_0(v161);
    v92 = sub_14D1290(v90, &v152);
    if ( v163 != v165 )
      _libc_free((unsigned __int64)v163);
    v137 = i;
  }
  v88 = v84;
  while ( *(_BYTE *)(*(_QWORD *)v88 + 16LL) == 13 )
  {
    v88 += 8;
    if ( v87 == v88 )
      goto LABEL_157;
  }
  if ( v84 != v165 )
    _libc_free((unsigned __int64)v84);
  v137 = i;
LABEL_63:
  v156 = v45;
  if ( v45 > 0x40 )
    sub_16A4EF0(&v155, 0, 0);
  else
    v155 = 0;
  if ( *(_BYTE *)(i + 16) == 5 && *(_WORD *)(i + 18) == 46 )
  {
    v120 = *(_QWORD *)(i - 24LL * (*(_DWORD *)(i + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(v120 + 16) == 13 )
    {
      sub_16A5D10(&v163, v120 + 24, v45);
      if ( v156 > 0x40 && v155 )
        j_j___libc_free_0_0(v155);
      v155 = v163;
      v156 = v164;
    }
  }
  v49 = *(_QWORD *)i;
  if ( (unsigned __int8)sub_1593BB0(i) )
    goto LABEL_71;
  v50 = v156;
  if ( v156 > 0x40 )
  {
    if ( v50 - (unsigned int)sub_16A57B0(&v155) > 0x40 )
      goto LABEL_71;
    v51 = (_QWORD *)*v155;
  }
  else
  {
    v51 = v155;
  }
  if ( v51 )
  {
LABEL_71:
    v52 = *(_DWORD **)(a4 + 408);
    v53 = *(_DWORD *)(v49 + 8) >> 8;
    v54 = 4LL * *(unsigned int *)(a4 + 416);
    v55 = &v52[(unsigned __int64)v54 / 4];
    v56 = v54 >> 2;
    v57 = v54 >> 4;
    if ( !v57 )
      goto LABEL_216;
    v58 = &v52[4 * v57];
    do
    {
      if ( v53 == *v52 )
        goto LABEL_78;
      if ( v53 == v52[1] )
      {
        ++v52;
        goto LABEL_78;
      }
      if ( v53 == v52[2] )
      {
        v52 += 2;
        goto LABEL_78;
      }
      if ( v53 == v52[3] )
      {
        v52 += 3;
        goto LABEL_78;
      }
      v52 += 4;
    }
    while ( v58 != v52 );
    v56 = v55 - v52;
LABEL_216:
    if ( v56 != 2 )
    {
      if ( v56 != 3 )
      {
        if ( v56 == 1 )
          goto LABEL_219;
        goto LABEL_220;
      }
      if ( v53 == *v52 )
        goto LABEL_78;
      ++v52;
    }
    if ( v53 != *v52 )
    {
      ++v52;
LABEL_219:
      if ( v53 != *v52 )
      {
LABEL_220:
        v162 = v154;
        if ( v154 > 0x40 )
          sub_16A4FD0(&v161, &v153);
        else
          v161 = v153;
        sub_16A7200(&v161, &v155);
        v117 = v162;
        v162 = 0;
        LODWORD(v164) = v117;
        v163 = (_BYTE *)v161;
        v118 = sub_16498A0(i);
        v119 = sub_159C0E0(v118, &v163);
        if ( (unsigned int)v164 > 0x40 && v163 )
          j_j___libc_free_0_0(v163);
        if ( v162 > 0x40 && v161 )
          j_j___libc_free_0_0(v161);
        v12 = sub_15A3BA0(v119, v129, 0);
        goto LABEL_114;
      }
    }
LABEL_78:
    if ( v55 == v52 )
      goto LABEL_220;
  }
  v136 = i;
  v163 = v165;
  v164 = 0x2000000000LL;
  while ( 1 )
  {
    v59 = *(_BYTE *)(v49 + 8);
    if ( v59 == 13 )
      break;
    if ( v59 == 15 )
    {
      if ( (_DWORD)v164 )
      {
LABEL_105:
        v68 = v136;
        goto LABEL_106;
      }
      v49 = v152;
      v96 = *(unsigned __int8 *)(v152 + 8);
      if ( (unsigned __int8)v96 > 0xFu || (v106 = 35454, !_bittest64(&v106, v96)) )
      {
        if ( (unsigned int)(v96 - 13) > 1 && (_DWORD)v96 != 16 || !(unsigned __int8)sub_16435F0(v152, 0) )
          goto LABEL_111;
      }
    }
    else
    {
      if ( ((v59 - 14) & 0xFD) != 0 )
        goto LABEL_105;
      v49 = *(_QWORD *)(v49 + 24);
    }
    v60 = 1;
    v61 = v49;
    v62 = (unsigned int)sub_15A9FE0(a4, v49);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v61 + 8) )
      {
        case 1:
          v63 = 16;
          goto LABEL_88;
        case 2:
          v63 = 32;
          goto LABEL_88;
        case 3:
        case 9:
          v63 = 64;
          goto LABEL_88;
        case 4:
          v63 = 80;
          goto LABEL_88;
        case 5:
        case 6:
          v63 = 128;
          goto LABEL_88;
        case 7:
          v63 = 8 * (unsigned int)sub_15A9520(a4, 0);
          goto LABEL_88;
        case 0xB:
          v63 = *(_DWORD *)(v61 + 8) >> 8;
          goto LABEL_88;
        case 0xD:
          v63 = 8LL * *(_QWORD *)sub_15A9930(a4, v61);
          goto LABEL_88;
        case 0xE:
          v147 = *(_QWORD *)(v61 + 32);
          v131 = *(_QWORD *)(v61 + 24);
          v74 = (unsigned int)sub_15A9FE0(a4, v131);
          v63 = 8 * v147 * v74 * ((v74 + ((unsigned __int64)(sub_127FA20(a4, v131) + 7) >> 3) - 1) / v74);
          goto LABEL_88;
        case 0xF:
          v63 = 8 * (unsigned int)sub_15A9520(a4, *(_DWORD *)(v61 + 8) >> 8);
LABEL_88:
          v158 = v45;
          v64 = (unsigned __int64)(v63 * v60 + 7) >> 3;
          if ( v45 <= 0x40 )
          {
            v157 = (_QWORD *)(v141 & (v62 * ((v62 + v64 - 1) / v62)));
LABEL_90:
            v65 = v157;
            goto LABEL_91;
          }
          sub_16A4EF0(&v157, v62 * ((v62 + v64 - 1) / v62), 0);
          v95 = v158;
          if ( v158 <= 0x40 )
            goto LABEL_90;
          if ( v95 - (unsigned int)sub_16A57B0(&v157) > 0x40 )
            goto LABEL_92;
          v65 = (_QWORD *)*v157;
LABEL_91:
          if ( !v65 )
          {
            v93 = sub_15A0680(v146, 0, 0);
            v94 = (unsigned int)v164;
            if ( (unsigned int)v164 >= HIDWORD(v164) )
            {
              sub_16CD150(&v163, v165, 0, 8);
              v94 = (unsigned int)v164;
            }
            *(_QWORD *)&v163[8 * v94] = v93;
            LODWORD(v164) = v164 + 1;
            goto LABEL_101;
          }
LABEL_92:
          sub_16AA350(&v159, &v153, &v157, &v151);
          if ( v151 )
          {
            v68 = v136;
            if ( v160 > 0x40 && v159 )
              j_j___libc_free_0_0(v159);
            if ( v158 > 0x40 && v157 )
              j_j___libc_free_0_0(v157);
LABEL_106:
            v69 = v154;
            v70 = v153;
            if ( v154 > 0x40 )
            {
              v71 = v69 - sub_16A57B0(&v153);
              goto LABEL_108;
            }
LABEL_187:
            v72 = v70;
            goto LABEL_110;
          }
          sub_16A7B50(&v161, &v159, &v157);
          sub_16A7590(&v153, &v161);
          if ( v162 > 0x40 && v161 )
            j_j___libc_free_0_0(v161);
          v66 = sub_15A1070(v146, &v159);
          v67 = (unsigned int)v164;
          if ( (unsigned int)v164 >= HIDWORD(v164) )
          {
            sub_16CD150(&v163, v165, 0, 8);
            v67 = (unsigned int)v164;
          }
          *(_QWORD *)&v163[8 * v67] = v66;
          LODWORD(v164) = v164 + 1;
          if ( v160 > 0x40 && v159 )
            j_j___libc_free_0_0(v159);
LABEL_101:
          if ( v158 > 0x40 && v157 )
            j_j___libc_free_0_0(v157);
LABEL_104:
          if ( v49 == v126 )
            goto LABEL_105;
          break;
        case 0x10:
          v73 = *(_QWORD *)(v61 + 32);
          v61 = *(_QWORD *)(v61 + 24);
          v60 *= v73;
          continue;
        default:
          JUMPOUT(0x419798);
      }
      break;
    }
  }
  v97 = (unsigned __int64 *)sub_15A9930(a4, v49);
  v98 = 1LL << ((unsigned __int8)v154 - 1);
  if ( v154 <= 0x40 )
  {
    v99 = v153;
    v70 = v153;
    if ( (v153 & v98) != 0 || *v97 <= v153 )
    {
      v68 = v136;
      goto LABEL_187;
    }
LABEL_205:
    v134 = v97;
    v109 = sub_15A8020(v97, v99);
    v110 = sub_1643350(*(_QWORD *)v49);
    v111 = sub_159C470(v110, v109, 0);
    v112 = v109;
    v113 = v134;
    v114 = v111;
    v115 = (unsigned int)v164;
    if ( (unsigned int)v164 >= HIDWORD(v164) )
    {
      v150 = v114;
      sub_16CD150(&v163, v165, 0, 8);
      v115 = (unsigned int)v164;
      v112 = v109;
      v113 = v134;
      v114 = v150;
    }
    *(_QWORD *)&v163[8 * v115] = v114;
    LODWORD(v164) = v164 + 1;
    v116 = v113[v112 + 2];
    v162 = v45;
    if ( v45 > 0x40 )
      sub_16A4EF0(&v161, v116, 0);
    else
      v161 = v141 & v116;
    sub_16A7590(&v153, &v161);
    if ( v162 > 0x40 && v161 )
      j_j___libc_free_0_0(v161);
    v49 = sub_1643D80(v49, v109);
    goto LABEL_104;
  }
  v128 = v154;
  v133 = v97;
  v149 = (unsigned __int64 *)v153;
  v107 = *(_QWORD *)(v153 + 8LL * ((v154 - 1) >> 6)) & v98;
  v108 = sub_16A57B0(&v153);
  v97 = v133;
  if ( v107 )
  {
    v68 = v136;
    v71 = v128 - v108;
LABEL_108:
    if ( v71 <= 0x40 )
      goto LABEL_109;
    goto LABEL_111;
  }
  if ( v128 - v108 > 0x40 )
    goto LABEL_111;
  v99 = *v149;
  if ( *v133 > *v149 )
    goto LABEL_205;
  v68 = v136;
LABEL_109:
  v72 = *(_QWORD *)v153;
LABEL_110:
  if ( v72 )
  {
LABEL_111:
    v12 = 0;
    goto LABEL_112;
  }
  v100 = *(_BYTE *)(v137 + 17) >> 1 >> 1;
  if ( v100 )
  {
    v101 = sub_16348C0(v137);
    v102 = v152;
    if ( v152 != v101 )
      goto LABEL_190;
    v103 = v164;
    v104 = (int)v163;
    if ( v100 - 1 < (unsigned int)v164 )
    {
      v121 = v163;
      v122 = 24;
      while ( 1 )
      {
        v123 = (*(_BYTE *)(v137 + 23) & 0x40) != 0
             ? *(_QWORD *)(v137 - 8)
             : v137 - 24LL * (*(_DWORD *)(v137 + 20) & 0xFFFFFFF);
        if ( *v121 != *(_QWORD *)(v123 + v122) )
          break;
        v122 += 24;
        ++v121;
        if ( 24LL * (unsigned int)(v100 - 1) + 48 == v122 )
        {
          BYTE4(v161) = 1;
          v105 = v143;
          LODWORD(v161) = v100 - 1;
          goto LABEL_192;
        }
      }
    }
  }
  else
  {
    v102 = v152;
LABEL_190:
    v103 = v164;
    v104 = (int)v163;
  }
  BYTE4(v161) = 0;
  v105 = v143;
LABEL_192:
  v12 = sub_15A2E80(v102, v68, v104, v103, v105, (unsigned int)&v161, 0);
  if ( v126 != v49 )
  {
    if ( !(unsigned __int8)sub_1593BB0(v12) || *(_BYTE *)(v129 + 8) == 9 )
      v12 = sub_14D44C0(v12, v129, (_BYTE *)a4);
    else
      v12 = sub_15A06D0(v129);
  }
LABEL_112:
  if ( v163 != v165 )
    _libc_free((unsigned __int64)v163);
LABEL_114:
  if ( v156 > 0x40 && v155 )
    j_j___libc_free_0_0(v155);
  if ( v154 > 0x40 && v153 )
    j_j___libc_free_0_0(v153);
  return v12;
}
