// Function: sub_97B6F0
// Address: 0x97b6f0
//
__int64 __fastcall sub_97B6F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  int v8; // edx
  char v9; // al
  __int64 v10; // r13
  __int64 *v12; // rsi
  unsigned int v13; // r13d
  __int64 v14; // r14
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // r14
  __int64 v19; // r15
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  int v24; // ecx
  char *v25; // r13
  __int64 **v26; // r9
  int v27; // edx
  int v28; // r8d
  _BYTE *v29; // r13
  char *v30; // r15
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rdx
  unsigned int v37; // r13d
  __int64 v38; // rax
  unsigned __int64 v39; // rax
  unsigned __int64 *v40; // rsi
  char *v41; // r8
  unsigned __int64 v42; // rax
  char v43; // al
  char *v44; // r15
  __int64 v45; // rsi
  __int64 v46; // rbx
  unsigned __int64 v47; // rdx
  unsigned int v48; // r12d
  _QWORD *v49; // rax
  int v50; // eax
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // rbx
  int v54; // eax
  int v55; // r12d
  __int64 v56; // rax
  char *v57; // r12
  char *v58; // r13
  __int64 v59; // rax
  __int64 *v60; // r12
  __int64 v61; // r9
  __int64 *v62; // rax
  int v63; // edx
  unsigned __int64 *v64; // rdi
  unsigned __int64 v65; // rdx
  unsigned __int64 *v66; // rcx
  unsigned __int64 *v67; // rax
  char *v68; // r13
  __int64 v69; // rax
  __int64 v70; // rsi
  char v71; // al
  unsigned int v72; // eax
  unsigned int v73; // eax
  unsigned int v74; // eax
  unsigned int v75; // eax
  unsigned int v76; // eax
  unsigned int v77; // eax
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rax
  _BYTE *v81; // rax
  __int64 v82; // r14
  unsigned int v83; // r12d
  __int64 *v84; // rbx
  __int64 v85; // rax
  unsigned int v86; // eax
  __int64 *v88; // [rsp+18h] [rbp-238h]
  char v90; // [rsp+38h] [rbp-218h]
  __int64 v91; // [rsp+38h] [rbp-218h]
  char *v92; // [rsp+38h] [rbp-218h]
  __int64 v93; // [rsp+40h] [rbp-210h]
  __int64 v94; // [rsp+40h] [rbp-210h]
  char *v95; // [rsp+40h] [rbp-210h]
  char *v96; // [rsp+40h] [rbp-210h]
  char *v97; // [rsp+40h] [rbp-210h]
  char *v98; // [rsp+40h] [rbp-210h]
  char *v99; // [rsp+40h] [rbp-210h]
  char *v100; // [rsp+40h] [rbp-210h]
  char *v101; // [rsp+40h] [rbp-210h]
  int v102; // [rsp+40h] [rbp-210h]
  int v103; // [rsp+40h] [rbp-210h]
  char *v104; // [rsp+40h] [rbp-210h]
  char *v105; // [rsp+40h] [rbp-210h]
  __int64 v106; // [rsp+48h] [rbp-208h]
  unsigned __int64 v107; // [rsp+48h] [rbp-208h]
  int v108; // [rsp+48h] [rbp-208h]
  int v109; // [rsp+48h] [rbp-208h]
  unsigned __int8 v110; // [rsp+53h] [rbp-1FDh]
  unsigned int v111; // [rsp+54h] [rbp-1FCh]
  __int64 *v112; // [rsp+58h] [rbp-1F8h]
  int v113; // [rsp+58h] [rbp-1F8h]
  int v114; // [rsp+58h] [rbp-1F8h]
  char v115; // [rsp+67h] [rbp-1E9h] BYREF
  __int64 v116; // [rsp+68h] [rbp-1E8h] BYREF
  unsigned __int64 v117; // [rsp+70h] [rbp-1E0h] BYREF
  unsigned int v118; // [rsp+78h] [rbp-1D8h]
  __int64 *v119; // [rsp+80h] [rbp-1D0h] BYREF
  unsigned int v120; // [rsp+88h] [rbp-1C8h]
  __int64 v121; // [rsp+90h] [rbp-1C0h] BYREF
  unsigned int v122; // [rsp+98h] [rbp-1B8h]
  char v123; // [rsp+A0h] [rbp-1B0h]
  __int64 *v124; // [rsp+B0h] [rbp-1A0h] BYREF
  unsigned int v125; // [rsp+B8h] [rbp-198h]
  __int64 v126; // [rsp+C0h] [rbp-190h] BYREF
  unsigned int v127; // [rsp+C8h] [rbp-188h]
  char v128; // [rsp+D0h] [rbp-180h]
  __int64 *v129; // [rsp+E0h] [rbp-170h] BYREF
  unsigned int v130; // [rsp+E8h] [rbp-168h]
  __int64 v131; // [rsp+F0h] [rbp-160h] BYREF
  unsigned int v132; // [rsp+F8h] [rbp-158h]
  char v133; // [rsp+100h] [rbp-150h]
  __int64 *v134; // [rsp+110h] [rbp-140h] BYREF
  __int64 v135; // [rsp+118h] [rbp-138h]
  __int64 v136; // [rsp+120h] [rbp-130h] BYREF
  unsigned int v137; // [rsp+128h] [rbp-128h]
  char v138; // [rsp+130h] [rbp-120h]

  v7 = sub_BB5290(a1, a2, a3);
  v8 = *(unsigned __int8 *)(v7 + 8);
  v88 = *(__int64 **)(a1 + 8);
  if ( (_BYTE)v8 != 12 )
  {
    v9 = *(_BYTE *)(v7 + 8);
    if ( (unsigned __int8)v8 <= 3u )
      goto LABEL_3;
    if ( (_BYTE)v8 != 5 )
    {
      if ( (v8 & 0xFB) != 0xA && (v8 & 0xFD) != 4 )
      {
        if ( (unsigned __int8)(v8 - 15) > 3u && v8 != 20 || !(unsigned __int8)sub_BCEBA0(v7, 0) )
          return 0;
        v9 = *(_BYTE *)(v7 + 8);
      }
LABEL_3:
      if ( v9 == 18 )
        return 0;
    }
  }
  sub_BB52D0(&v119, a1);
  v12 = v88;
  v110 = *(_BYTE *)(a1 + 1) >> 1;
  v106 = sub_AE4570(a4, v88);
  if ( (unsigned int)*(unsigned __int8 *)(v106 + 8) - 17 > 1 )
    v112 = (__int64 *)v106;
  else
    v112 = **(__int64 ***)(v106 + 16);
  v134 = &v136;
  v135 = 0x2000000000LL;
  if ( (_DWORD)a3 == 1 )
  {
    if ( !v123 )
    {
      v30 = (char *)*a2;
      v35 = *(_QWORD *)(*a2 + 8);
      if ( *(_BYTE *)(v35 + 8) != 14 )
        return 0;
      v32 = sub_AE4570(a4, v35);
      goto LABEL_85;
    }
    v10 = 0;
    goto LABEL_55;
  }
  v90 = 0;
  v13 = 1;
  v93 = a4;
  do
  {
    while ( 1 )
    {
      v18 = &a2[v13];
      if ( v13 != 1 )
      {
        v12 = a2 + 1;
        if ( *(_BYTE *)(sub_B4DCA0(v7, a2 + 1, v13 - 1) + 8) == 15 )
        {
          v19 = *v18;
          goto LABEL_32;
        }
      }
      v19 = *v18;
      v20 = *(_QWORD *)(*v18 + 8);
      v21 = *(unsigned __int8 *)(v20 + 8);
      if ( (unsigned int)(v21 - 17) <= 1 )
        break;
      if ( (__int64 *)v20 == v112 )
        goto LABEL_32;
LABEL_27:
      v14 = (__int64)v112;
      if ( v21 == 17 )
        v14 = v106;
LABEL_19:
      v15 = sub_B50D10(v19, 1, v14, 1);
      v12 = (__int64 *)v19;
      v16 = sub_96F480(v15, v19, v14, v93);
      if ( !v16 )
      {
        a4 = v93;
        goto LABEL_53;
      }
      v17 = (unsigned int)v135;
      if ( (unsigned __int64)(unsigned int)v135 + 1 > HIDWORD(v135) )
      {
        v12 = &v136;
        v91 = v16;
        sub_C8D5F0(&v134, &v136, (unsigned int)v135 + 1LL, 8);
        v17 = (unsigned int)v135;
        v16 = v91;
      }
      v90 = 1;
      ++v13;
      v134[v17] = v16;
      LODWORD(v135) = v135 + 1;
      if ( (_DWORD)a3 == v13 )
        goto LABEL_35;
    }
    v12 = v112;
    if ( **(__int64 ***)(v20 + 16) != v112 )
    {
      v14 = v106;
      if ( v21 == 18 )
        goto LABEL_19;
      goto LABEL_27;
    }
LABEL_32:
    v22 = (unsigned int)v135;
    v23 = (unsigned int)v135 + 1LL;
    if ( v23 > HIDWORD(v135) )
    {
      v12 = &v136;
      sub_C8D5F0(&v134, &v136, v23, 8);
      v22 = (unsigned int)v135;
    }
    ++v13;
    v134[v22] = v19;
    LODWORD(v135) = v135 + 1;
  }
  while ( (_DWORD)a3 != v13 );
LABEL_35:
  a4 = v93;
  if ( !v90 )
  {
LABEL_53:
    v10 = 0;
    goto LABEL_41;
  }
  v128 = 0;
  if ( v123 )
  {
    v125 = v120;
    if ( v120 > 0x40 )
      sub_C43780(&v124, &v119);
    else
      v124 = v119;
    v127 = v122;
    if ( v122 > 0x40 )
      sub_C43780(&v126, &v121);
    else
      v126 = v121;
    v128 = 1;
    v24 = v135;
    v25 = (char *)*a2;
    v133 = 0;
    v130 = v125;
    v27 = (int)v134;
    v28 = v110;
    if ( v125 > 0x40 )
    {
      v103 = (int)v134;
      v109 = v135;
      sub_C43780(&v129, &v124);
      v28 = v110;
      v27 = v103;
      v24 = v109;
      v26 = &v129;
    }
    else
    {
      v26 = &v129;
      v129 = v124;
    }
    v132 = v127;
    if ( v127 > 0x40 )
    {
      v102 = v28;
      v108 = v27;
      v114 = v24;
      sub_C43780(&v131, &v126);
      LODWORD(v26) = (unsigned int)&v129;
      v28 = v102;
      v27 = v108;
      v24 = v114;
    }
    else
    {
      v131 = v126;
    }
    v133 = 1;
  }
  else
  {
    v24 = v135;
    v25 = (char *)*a2;
    v133 = 0;
    v26 = &v129;
    v27 = (int)v134;
    v28 = v110;
  }
  v29 = (_BYTE *)sub_AD9FD0(v7, (_DWORD)v25, v27, v24, v28, (_DWORD)v26, 0);
  if ( v133 )
  {
    v133 = 0;
    if ( v132 > 0x40 && v131 )
      j_j___libc_free_0_0(v131);
    if ( v130 > 0x40 && v129 )
      j_j___libc_free_0_0(v129);
  }
  if ( v128 )
  {
    v128 = 0;
    if ( v127 > 0x40 && v126 )
      j_j___libc_free_0_0(v126);
    if ( v125 > 0x40 && v124 )
      j_j___libc_free_0_0(v124);
  }
  v12 = (__int64 *)a4;
  v10 = sub_97B670(v29, a4, a5);
LABEL_41:
  if ( v134 != &v136 )
    _libc_free(v134, v12);
  if ( v123 )
  {
LABEL_55:
    v123 = 0;
    if ( v122 > 0x40 && v121 )
      j_j___libc_free_0_0(v121);
    if ( v120 > 0x40 && v119 )
      j_j___libc_free_0_0(v119);
  }
  if ( v10 )
    return v10;
  v30 = (char *)*a2;
  v31 = *(_QWORD *)(*a2 + 8);
  if ( *(_BYTE *)(v31 + 8) != 14 )
    return 0;
  v32 = sub_AE4570(a4, v31);
  if ( (_DWORD)a3 != 1 )
  {
    v33 = 1;
    do
    {
      v34 = a2[v33];
      if ( *(_BYTE *)v34 != 17 || *(_BYTE *)(*(_QWORD *)(v34 + 8) + 8LL) != 12 )
        return 0;
    }
    while ( ++v33 != (_DWORD)a3 );
  }
LABEL_85:
  v134 = (__int64 *)sub_9208B0(a4, v32);
  v135 = v36;
  v37 = sub_CA1930(&v134);
  v111 = v37;
  v38 = sub_AE54E0(a4, v7, a2 + 1, a3 - 1);
  v118 = v37;
  if ( v37 > 0x40 )
  {
    sub_C43690(&v117, v38, 1);
  }
  else
  {
    v39 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v37) & v38;
    if ( !v37 )
      v39 = 0;
    v117 = v39;
  }
  v40 = (unsigned __int64 *)a1;
  sub_BB52D0(&v124, a1);
  if ( v128 )
  {
    v40 = (unsigned __int64 *)&v124;
    sub_AB4E00(&v134, &v124, v37);
    if ( v128 )
    {
      if ( v125 > 0x40 && v124 )
        j_j___libc_free_0_0(v124);
      v124 = v134;
      v86 = v135;
      LODWORD(v135) = 0;
      v125 = v86;
      if ( v127 > 0x40 && v126 )
      {
        j_j___libc_free_0_0(v126);
        v126 = v136;
        v127 = v137;
        if ( (unsigned int)v135 > 0x40 && v134 )
          j_j___libc_free_0_0(v134);
      }
      else
      {
        v126 = v136;
        v127 = v137;
      }
    }
    else
    {
      v128 = 1;
      v125 = v135;
      v124 = v134;
      v127 = v137;
      v126 = v136;
    }
  }
  v115 = 0;
  v41 = v30;
  v113 = *(_BYTE *)(a1 + 1) >> 1;
  v42 = 0;
  if ( v37 )
    v42 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v37;
  v107 = v42;
  while ( 2 )
  {
    v43 = *v41;
    if ( (unsigned __int8)*v41 <= 0x1Cu )
    {
      if ( v43 != 5 || *((_WORD *)v41 + 1) != 34 )
      {
LABEL_95:
        v44 = v41;
        goto LABEL_96;
      }
    }
    else if ( v43 != 63 )
    {
      goto LABEL_95;
    }
    v113 = ((unsigned __int8)v41[1] >> 1) & (unsigned __int8)v113;
    v56 = 32LL * (*((_DWORD *)v41 + 1) & 0x7FFFFFF);
    if ( (v41[7] & 0x40) != 0 )
    {
      v57 = (char *)*((_QWORD *)v41 - 1);
      v58 = &v57[v56];
    }
    else
    {
      v58 = v41;
      v57 = &v41[-v56];
    }
    v59 = v56 - 32;
    v134 = &v136;
    v60 = (__int64 *)(v57 + 32);
    v135 = 0x400000000LL;
    v61 = v59 >> 5;
    if ( (unsigned __int64)v59 > 0x80 )
    {
      v40 = (unsigned __int64 *)&v136;
      v92 = v41;
      v94 = v59 >> 5;
      sub_C8D5F0(&v134, &v136, v59 >> 5, 8);
      v64 = (unsigned __int64 *)v134;
      v63 = v135;
      LODWORD(v61) = v94;
      v41 = v92;
      v62 = &v134[(unsigned int)v135];
    }
    else
    {
      v62 = &v136;
      v63 = 0;
      v64 = (unsigned __int64 *)&v136;
    }
    if ( v58 != (char *)v60 )
    {
      do
      {
        if ( v62 )
          *v62 = *v60;
        v60 += 4;
        ++v62;
      }
      while ( v58 != (char *)v60 );
      v64 = (unsigned __int64 *)v134;
      v63 = v135;
    }
    LODWORD(v135) = v61 + v63;
    v65 = (unsigned int)(v61 + v63);
    v66 = &v64[v65];
    if ( v66 == v64 )
    {
LABEL_148:
      if ( v128 )
      {
LABEL_149:
        v68 = *(char **)&v41[-32 * (*((_DWORD *)v41 + 1) & 0x7FFFFFF)];
        v69 = sub_BB5290(v41, v40, v65);
        v70 = sub_AE54E0(a4, v69, v134, (unsigned int)v135);
        v120 = v111;
        if ( v111 > 0x40 )
          sub_C43690(&v119, v70, 1);
        else
          v119 = (__int64 *)(v107 & v70);
        v40 = &v117;
        sub_C45F70(&v129, &v117, &v119, &v115);
        if ( v118 > 0x40 && v117 )
          j_j___libc_free_0_0(v117);
        v117 = (unsigned __int64)v129;
        v118 = v130;
        if ( v120 > 0x40 && v119 )
          j_j___libc_free_0_0(v119);
        if ( v134 != &v136 )
          _libc_free(v134, &v117);
        v41 = v68;
        continue;
      }
      v40 = (unsigned __int64 *)v41;
      v95 = v41;
      sub_BB52D0(&v129, v41);
      v41 = v95;
      if ( v128 )
      {
        if ( v133 )
        {
          if ( v125 > 0x40 && v124 )
          {
            j_j___libc_free_0_0(v124);
            v41 = v95;
          }
          v124 = v129;
          v74 = v130;
          v130 = 0;
          v125 = v74;
          if ( v127 > 0x40 && v126 )
          {
            v104 = v41;
            j_j___libc_free_0_0(v126);
            v41 = v104;
          }
          v126 = v131;
          v75 = v132;
          v132 = 0;
          v127 = v75;
          v71 = v133;
        }
        else
        {
          v128 = 0;
          if ( v127 > 0x40 && v126 )
          {
            j_j___libc_free_0_0(v126);
            v41 = v95;
          }
          if ( v125 > 0x40 && v124 )
          {
            v96 = v41;
            j_j___libc_free_0_0(v124);
            v41 = v96;
          }
          v71 = v133;
        }
        if ( !v71 )
          goto LABEL_173;
      }
      else
      {
        if ( !v133 )
          goto LABEL_149;
        v72 = v130;
        v128 = 1;
        v130 = 0;
        v125 = v72;
        v124 = v129;
        v73 = v132;
        v132 = 0;
        v127 = v73;
        v126 = v131;
      }
      v133 = 0;
      if ( v132 > 0x40 && v131 )
      {
        v100 = v41;
        j_j___libc_free_0_0(v131);
        v41 = v100;
      }
      if ( v130 > 0x40 && v129 )
      {
        v101 = v41;
        j_j___libc_free_0_0(v129);
        v41 = v101;
      }
LABEL_173:
      if ( v128 )
      {
        v97 = v41;
        sub_AB4E00(&v119, &v124, v111);
        v40 = (unsigned __int64 *)&v119;
        sub_AB1F90(&v129, &v119, &v117);
        v41 = v97;
        if ( v128 )
        {
          if ( v125 > 0x40 && v124 )
          {
            j_j___libc_free_0_0(v124);
            v41 = v97;
          }
          v124 = v129;
          v76 = v130;
          v130 = 0;
          v125 = v76;
          if ( v127 > 0x40 && v126 )
          {
            v105 = v41;
            j_j___libc_free_0_0(v126);
            v41 = v105;
            v126 = v131;
            v127 = v132;
            if ( v130 > 0x40 && v129 )
            {
              j_j___libc_free_0_0(v129);
              v41 = v105;
            }
          }
          else
          {
            v126 = v131;
            v127 = v132;
          }
        }
        else
        {
          v128 = 1;
          v125 = v130;
          v124 = v129;
          v127 = v132;
          v126 = v131;
        }
        if ( v122 > 0x40 && v121 )
        {
          v98 = v41;
          j_j___libc_free_0_0(v121);
          v41 = v98;
        }
        if ( v120 > 0x40 && v119 )
        {
          v99 = v41;
          j_j___libc_free_0_0(v119);
          v41 = v99;
        }
      }
      goto LABEL_149;
    }
    break;
  }
  v67 = v64;
  while ( 1 )
  {
    v65 = *v67;
    if ( *(_BYTE *)*v67 != 17 )
      break;
    if ( v66 == ++v67 )
      goto LABEL_148;
  }
  v44 = v41;
  if ( v64 != (unsigned __int64 *)&v136 )
    _libc_free(v64, v40);
LABEL_96:
  if ( (v113 & 2) != 0 && (v113 & 1) == 0 && v115 )
    v113 &= 0xFFFFFFFC;
  v45 = *((_QWORD *)v44 + 1);
  v120 = sub_AE43A0(a4, v45);
  if ( v120 > 0x40 )
  {
    v45 = 0;
    sub_C43690(&v119, 0, 0);
  }
  else
  {
    v119 = 0;
  }
  if ( *v44 == 5 && *((_WORD *)v44 + 1) == 48 )
  {
    v81 = *(_BYTE **)&v44[-32 * (*((_DWORD *)v44 + 1) & 0x7FFFFFF)];
    if ( *v81 == 17 )
    {
      v45 = (__int64)(v81 + 24);
      sub_C44AB0(&v134, v81 + 24, v120);
      if ( v120 > 0x40 && v119 )
        j_j___libc_free_0_0(v119);
      v119 = v134;
      v120 = v135;
    }
  }
  v46 = *((_QWORD *)v44 + 1);
  if ( !(unsigned __int8)sub_AC30F0(v44) )
  {
    v48 = v120;
    if ( v120 <= 0x40 )
    {
      v49 = v119;
      goto LABEL_107;
    }
    if ( v48 - (unsigned int)sub_C444A0(&v119) <= 0x40 )
    {
      v49 = (_QWORD *)*v119;
LABEL_107:
      if ( !v49 )
        goto LABEL_256;
    }
  }
  v45 = *(_DWORD *)(v46 + 8) >> 8;
  if ( !*(_BYTE *)(sub_AE2980(a4, v45) + 16) )
  {
    sub_C44740(&v129, &v119);
    sub_C45EE0(&v129, &v117);
    v77 = v130;
    v130 = 0;
    LODWORD(v135) = v77;
    v134 = v129;
    sub_C43D80(&v119, &v134, 0);
    if ( (unsigned int)v135 > 0x40 && v134 )
      j_j___libc_free_0_0(v134);
    if ( v130 > 0x40 && v129 )
      j_j___libc_free_0_0(v129);
    v79 = sub_BD5C60(v44, &v134, v78);
    v80 = sub_ACCFD0(v79, &v119);
    v10 = sub_AD4C70(v80, v88, 0);
  }
  else
  {
LABEL_256:
    if ( (v113 & 1) == 0 )
    {
      v47 = v117;
      v45 = 1LL << ((unsigned __int8)v118 - 1);
      if ( v118 > 0x40 )
        v47 = *(_QWORD *)(v117 + 8LL * ((v118 - 1) >> 6));
      if ( (v47 & v45) == 0 )
      {
        v45 = a4;
        v82 = sub_BD4FF0(v44, a4, &v129, &v134);
        if ( v82 )
        {
          if ( !(_BYTE)v129 )
          {
            v83 = v118;
            v84 = (__int64 *)v117;
            if ( v118 > 0x40 )
            {
              if ( v83 + 1 - (unsigned int)sub_969260((__int64)&v117) <= 0x40 )
              {
                v85 = *v84;
                goto LABEL_260;
              }
              v47 = (v83 - 1) >> 6;
              if ( (v84[v47] & (1LL << ((unsigned __int8)v83 - 1))) != 0 )
LABEL_261:
                v113 |= 3u;
            }
            else
            {
              if ( v118 )
                v85 = (__int64)(v117 << (64 - (unsigned __int8)v118)) >> (64 - (unsigned __int8)v118);
              else
                v85 = 0;
LABEL_260:
              if ( v82 >= v85 )
                goto LABEL_261;
            }
          }
        }
      }
    }
    if ( (v113 & 2) != 0 )
    {
      v47 = v117;
      v45 = 1LL << ((unsigned __int8)v118 - 1);
      if ( v118 > 0x40 )
        v47 = *(_QWORD *)(v117 + 8LL * ((v118 - 1) >> 6));
      v50 = v113 | 4;
      if ( (v47 & v45) != 0 )
        v50 = v113;
      v113 = v50;
    }
    v51 = sub_BD5C60(v44, v45, v47);
    v133 = 0;
    v52 = v51;
    if ( v128 )
    {
      v130 = v125;
      if ( v125 > 0x40 )
        sub_C43780(&v129, &v124);
      else
        v129 = v124;
      v132 = v127;
      if ( v127 > 0x40 )
        sub_C43780(&v131, &v126);
      else
        v131 = v126;
      v133 = 1;
    }
    v53 = sub_ACCFD0(v52, &v117);
    v54 = sub_BCB2B0(v52);
    v138 = 0;
    v55 = v54;
    if ( v133 )
    {
      LODWORD(v135) = v130;
      if ( v130 > 0x40 )
        sub_C43780(&v134, &v129);
      else
        v134 = v129;
      v137 = v132;
      if ( v132 > 0x40 )
        sub_C43780(&v136, &v131);
      else
        v136 = v131;
      v138 = 1;
    }
    v116 = v53;
    v10 = sub_AD9FD0(v55, (_DWORD)v44, (unsigned int)&v116, 1, v113, (unsigned int)&v134, 0);
    if ( v138 )
    {
      v138 = 0;
      if ( v137 > 0x40 && v136 )
        j_j___libc_free_0_0(v136);
      if ( (unsigned int)v135 > 0x40 && v134 )
        j_j___libc_free_0_0(v134);
    }
    if ( v133 )
    {
      v133 = 0;
      if ( v132 > 0x40 && v131 )
        j_j___libc_free_0_0(v131);
      if ( v130 > 0x40 && v129 )
        j_j___libc_free_0_0(v129);
    }
  }
  if ( v120 > 0x40 && v119 )
    j_j___libc_free_0_0(v119);
  if ( v128 )
  {
    v128 = 0;
    if ( v127 > 0x40 && v126 )
      j_j___libc_free_0_0(v126);
    if ( v125 > 0x40 && v124 )
      j_j___libc_free_0_0(v124);
  }
  if ( v118 > 0x40 && v117 )
    j_j___libc_free_0_0(v117);
  return v10;
}
