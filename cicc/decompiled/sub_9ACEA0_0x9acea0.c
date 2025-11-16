// Function: sub_9ACEA0
// Address: 0x9acea0
//
__int64 *__fastcall sub_9ACEA0(
        __int64 *a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __m128i *a7)
{
  __int64 v8; // r12
  unsigned __int8 *v9; // rbx
  unsigned int v10; // eax
  unsigned int v11; // edx
  int v12; // eax
  bool v13; // al
  char v14; // r13
  unsigned int v15; // r13d
  int v16; // eax
  bool v17; // r13
  int v18; // eax
  int v19; // eax
  __int64 v20; // rax
  bool v21; // cc
  int v22; // ecx
  unsigned int v23; // eax
  __int64 v24; // rsi
  unsigned __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rax
  int v29; // ecx
  unsigned int v30; // eax
  __int64 v31; // rsi
  unsigned __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // rax
  _QWORD *v36; // rax
  __int64 v38; // rax
  int v39; // eax
  unsigned int v40; // r12d
  __int64 v41; // rdx
  unsigned __int64 v42; // r13
  __int64 v43; // rdi
  __int64 v44; // rdi
  unsigned __int8 *v45; // rax
  unsigned __int8 *v46; // rdx
  unsigned __int8 *v47; // r12
  unsigned __int8 *v48; // rax
  unsigned int v49; // ebx
  _BYTE *v52; // rcx
  _BYTE *v53; // rdx
  __int64 v54; // rdx
  bool v55; // al
  unsigned __int8 **v56; // rcx
  bool v57; // r8
  unsigned int v58; // eax
  unsigned int v59; // edx
  unsigned int v60; // eax
  __int64 v61; // rsi
  bool v62; // zf
  __int64 v63; // rcx
  _BYTE *v64; // rdx
  unsigned __int8 **v65; // rsi
  unsigned __int8 **v66; // rax
  unsigned __int8 **v67; // rdx
  unsigned __int8 *v68; // rcx
  __int64 v69; // r8
  __int64 v70; // rdi
  int v71; // eax
  bool v72; // al
  __int64 v73; // rax
  unsigned int v75; // ecx
  __int64 v79; // rsi
  unsigned __int8 *v80; // rcx
  __int64 v81; // rax
  unsigned __int8 *v82; // rcx
  _BYTE *v83; // rax
  __int64 v84; // rdx
  bool v85; // al
  unsigned __int8 *v86; // rcx
  bool v87; // cl
  unsigned int v88; // r8d
  __int64 v89; // rax
  unsigned int v90; // r8d
  int v91; // eax
  int v92; // eax
  __int64 v93; // rbx
  unsigned int i; // r12d
  _BYTE *v95; // rax
  bool v96; // [rsp+Ch] [rbp-A4h]
  unsigned int v97; // [rsp+Ch] [rbp-A4h]
  bool v98; // [rsp+Ch] [rbp-A4h]
  unsigned __int64 v99; // [rsp+10h] [rbp-A0h]
  unsigned int v100; // [rsp+10h] [rbp-A0h]
  int v101; // [rsp+10h] [rbp-A0h]
  __int64 v102; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v103; // [rsp+18h] [rbp-98h]
  unsigned int v104; // [rsp+18h] [rbp-98h]
  _BYTE *v105; // [rsp+18h] [rbp-98h]
  __int64 v106; // [rsp+18h] [rbp-98h]
  unsigned __int8 **v107; // [rsp+18h] [rbp-98h]
  unsigned int v108; // [rsp+20h] [rbp-90h]
  __int64 v109; // [rsp+20h] [rbp-90h]
  unsigned __int64 v110; // [rsp+20h] [rbp-90h]
  _BYTE *v111; // [rsp+20h] [rbp-90h]
  __int64 v112; // [rsp+20h] [rbp-90h]
  unsigned __int8 **v113; // [rsp+20h] [rbp-90h]
  int v114; // [rsp+20h] [rbp-90h]
  unsigned int v116; // [rsp+30h] [rbp-80h]
  unsigned int v117; // [rsp+30h] [rbp-80h]
  __int64 v118; // [rsp+30h] [rbp-80h]
  unsigned __int64 v119; // [rsp+30h] [rbp-80h]
  int v120; // [rsp+30h] [rbp-80h]
  unsigned int v121; // [rsp+30h] [rbp-80h]
  unsigned __int8 **v122; // [rsp+30h] [rbp-80h]
  int v123; // [rsp+30h] [rbp-80h]
  __int64 v124; // [rsp+30h] [rbp-80h]
  __int64 v125; // [rsp+30h] [rbp-80h]
  int v126; // [rsp+30h] [rbp-80h]
  unsigned __int8 *v127; // [rsp+30h] [rbp-80h]
  unsigned int v129; // [rsp+3Ch] [rbp-74h]
  __int64 v130; // [rsp+40h] [rbp-70h]
  int v131; // [rsp+40h] [rbp-70h]
  unsigned int v132; // [rsp+40h] [rbp-70h]
  __int64 v133; // [rsp+40h] [rbp-70h]
  unsigned int v134; // [rsp+40h] [rbp-70h]
  int v136; // [rsp+48h] [rbp-68h]
  unsigned __int8 *v137; // [rsp+58h] [rbp-58h] BYREF
  unsigned __int8 **v138; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int8 **v139; // [rsp+68h] [rbp-48h]
  unsigned __int64 v140; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v141; // [rsp+78h] [rbp-38h]

  v8 = a4;
  v9 = a2;
  v10 = *(_DWORD *)(a4 + 8);
  v129 = v10;
  *((_DWORD *)a1 + 2) = v10;
  if ( v10 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    *((_DWORD *)a1 + 6) = v129;
    sub_C43690(a1 + 2, 0, 0);
  }
  else
  {
    *a1 = 0;
    *((_DWORD *)a1 + 6) = v10;
    a1[2] = 0;
  }
  v11 = *(_DWORD *)(v8 + 24);
  v130 = v8 + 16;
  if ( v11 <= 0x40 )
  {
    v13 = *(_QWORD *)(v8 + 16) == 0;
  }
  else
  {
    v116 = *(_DWORD *)(v8 + 24);
    v12 = sub_C444A0(v130);
    v11 = v116;
    v13 = v116 == v12;
  }
  v14 = 1;
  if ( v13 )
  {
    v15 = *(_DWORD *)(a5 + 24);
    if ( v15 <= 0x40 )
    {
      v17 = *(_QWORD *)(a5 + 16) == 0;
    }
    else
    {
      v117 = v11;
      v16 = sub_C444A0(a5 + 16);
      v11 = v117;
      v17 = v15 == v16;
    }
    v14 = !v17;
  }
  v18 = *a2;
  v137 = 0;
  if ( (unsigned __int8)v18 <= 0x1Cu )
    v19 = *((unsigned __int16 *)a2 + 1);
  else
    v19 = v18 - 29;
  if ( v19 == 29 )
  {
    LODWORD(v139) = *(_DWORD *)(v8 + 8);
    if ( (unsigned int)v139 > 0x40 )
    {
      sub_C43780(&v138, v8);
      v141 = *(_DWORD *)(v8 + 24);
      if ( v141 <= 0x40 )
        goto LABEL_56;
    }
    else
    {
      v38 = *(_QWORD *)v8;
      v141 = v11;
      v138 = (unsigned __int8 **)v38;
      if ( v11 <= 0x40 )
      {
LABEL_56:
        v140 = *(_QWORD *)(v8 + 16);
LABEL_57:
        sub_C7BD50(&v138, a5);
        v21 = *((_DWORD *)a1 + 2) <= 0x40u;
        v39 = (int)v139;
        LODWORD(v139) = 0;
        v40 = v141;
        v41 = (__int64)v138;
        v141 = 0;
        v42 = v140;
        if ( !v21 )
        {
          v43 = *a1;
          if ( *a1 )
          {
            v133 = (__int64)v138;
            v136 = v39;
            j_j___libc_free_0_0(v43);
            v41 = v133;
            v39 = v136;
          }
        }
        v21 = *((_DWORD *)a1 + 6) <= 0x40u;
        *a1 = v41;
        *((_DWORD *)a1 + 2) = v39;
        if ( !v21 )
        {
          v44 = a1[2];
          if ( v44 )
            j_j___libc_free_0_0(v44);
        }
        v21 = v141 <= 0x40;
        a1[2] = v42;
        *((_DWORD *)a1 + 6) = v40;
        if ( v21 )
          goto LABEL_66;
LABEL_64:
        if ( v140 )
          j_j___libc_free_0_0(v140);
LABEL_66:
        if ( (unsigned int)v139 > 0x40 && v138 )
          j_j___libc_free_0_0(v138);
        goto LABEL_29;
      }
    }
    sub_C43780(&v140, v130);
    goto LABEL_57;
  }
  if ( v19 == 30 )
  {
    LODWORD(v139) = *(_DWORD *)(v8 + 8);
    if ( (unsigned int)v139 > 0x40 )
    {
      sub_C43780(&v138, v8);
      v141 = *(_DWORD *)(v8 + 24);
      if ( v141 <= 0x40 )
        goto LABEL_15;
    }
    else
    {
      v20 = *(_QWORD *)v8;
      v141 = v11;
      v138 = (unsigned __int8 **)v20;
      if ( v11 <= 0x40 )
      {
LABEL_15:
        v140 = *(_QWORD *)(v8 + 16);
LABEL_16:
        sub_C7BDB0(&v138, a5);
        v21 = *((_DWORD *)a1 + 2) <= 0x40u;
        v22 = (int)v139;
        LODWORD(v139) = 0;
        v23 = v141;
        v24 = (__int64)v138;
        v141 = 0;
        v25 = v140;
        if ( !v21 )
        {
          v26 = *a1;
          if ( *a1 )
          {
            v103 = v140;
            v108 = v23;
            v118 = (__int64)v138;
            v131 = v22;
            j_j___libc_free_0_0(v26);
            v25 = v103;
            v23 = v108;
            v24 = v118;
            v22 = v131;
          }
        }
        v21 = *((_DWORD *)a1 + 6) <= 0x40u;
        *a1 = v24;
        *((_DWORD *)a1 + 2) = v22;
        if ( !v21 )
        {
          v27 = a1[2];
          if ( v27 )
          {
            v119 = v25;
            v132 = v23;
            j_j___libc_free_0_0(v27);
            v25 = v119;
            v23 = v132;
          }
        }
        v21 = v141 <= 0x40;
        a1[2] = v25;
        *((_DWORD *)a1 + 6) = v23;
        if ( !v21 && v140 )
          j_j___libc_free_0_0(v140);
        if ( (unsigned int)v139 > 0x40 && v138 )
          j_j___libc_free_0_0(v138);
        if ( !v14 )
          goto LABEL_29;
        v62 = *v9 == 59;
        v140 = 0;
        v138 = &v137;
        v139 = &v137;
        if ( !v62 )
          goto LABEL_29;
        v63 = *((_QWORD *)v9 - 8);
        v64 = (_BYTE *)*((_QWORD *)v9 - 4);
        if ( v63 )
        {
          v137 = (unsigned __int8 *)*((_QWORD *)v9 - 8);
          v65 = &v137;
          if ( *v64 != 42 )
            goto LABEL_115;
          v79 = *((_QWORD *)v64 - 8);
          if ( v63 != v79 || !v79 )
          {
            v65 = &v137;
            goto LABEL_115;
          }
          if ( (unsigned __int8)sub_995B10((_QWORD **)&v140, *((_QWORD *)v64 - 4)) )
          {
LABEL_118:
            if ( (v9[7] & 0x40) != 0 )
              v67 = (unsigned __int8 **)*((_QWORD *)v9 - 1);
            else
              v67 = (unsigned __int8 **)&v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
            if ( v137 != *v67 )
              v8 = a5;
            sub_C7C2B0(&v138, v8);
            sub_984AC0(a1, (__int64 *)&v138);
            if ( v141 <= 0x40 )
              goto LABEL_66;
            goto LABEL_64;
          }
          v64 = (_BYTE *)*((_QWORD *)v9 - 4);
        }
        if ( !v64 )
          goto LABEL_29;
        v65 = v138;
LABEL_115:
        *v65 = v64;
        v66 = (unsigned __int8 **)*((_QWORD *)v9 - 8);
        if ( *(_BYTE *)v66 == 42
          && *(v66 - 8) == *v139
          && (unsigned __int8)sub_995B10((_QWORD **)&v140, (__int64)*(v66 - 4)) )
        {
          goto LABEL_118;
        }
LABEL_29:
        v14 = 0;
        goto LABEL_46;
      }
    }
    sub_C43780(&v140, v130);
    goto LABEL_16;
  }
  if ( v19 != 28 )
    BUG();
  LODWORD(v139) = *(_DWORD *)(v8 + 8);
  if ( (unsigned int)v139 > 0x40 )
  {
    sub_C43780(&v138, v8);
    v141 = *(_DWORD *)(v8 + 24);
    if ( v141 <= 0x40 )
      goto LABEL_33;
LABEL_77:
    sub_C43780(&v140, v130);
    goto LABEL_34;
  }
  v28 = *(_QWORD *)v8;
  v141 = v11;
  v138 = (unsigned __int8 **)v28;
  if ( v11 > 0x40 )
    goto LABEL_77;
LABEL_33:
  v140 = *(_QWORD *)(v8 + 16);
LABEL_34:
  sub_C7BCF0(&v138, a5);
  v21 = *((_DWORD *)a1 + 2) <= 0x40u;
  v29 = (int)v139;
  LODWORD(v139) = 0;
  v30 = v141;
  v31 = (__int64)v138;
  v141 = 0;
  v32 = v140;
  if ( !v21 )
  {
    v33 = *a1;
    if ( *a1 )
    {
      v99 = v140;
      v104 = v30;
      v109 = (__int64)v138;
      v120 = v29;
      j_j___libc_free_0_0(v33);
      v32 = v99;
      v30 = v104;
      v31 = v109;
      v29 = v120;
    }
  }
  v21 = *((_DWORD *)a1 + 6) <= 0x40u;
  *a1 = v31;
  *((_DWORD *)a1 + 2) = v29;
  if ( !v21 )
  {
    v34 = a1[2];
    if ( v34 )
    {
      v110 = v32;
      v121 = v30;
      j_j___libc_free_0_0(v34);
      v32 = v110;
      v30 = v121;
    }
  }
  v21 = v141 <= 0x40;
  a1[2] = v32;
  *((_DWORD *)a1 + 6) = v30;
  if ( !v21 && v140 )
    j_j___libc_free_0_0(v140);
  if ( (unsigned int)v139 > 0x40 && v138 )
    j_j___libc_free_0_0(v138);
  if ( !v14 || *v9 != 57 )
    goto LABEL_45;
  v52 = (_BYTE *)*((_QWORD *)v9 - 8);
  v53 = (_BYTE *)*((_QWORD *)v9 - 4);
  if ( !v52 )
    goto LABEL_136;
  v137 = (unsigned __int8 *)*((_QWORD *)v9 - 8);
  if ( *v53 != 44 )
    goto LABEL_98;
  v69 = *((_QWORD *)v53 - 8);
  if ( *(_BYTE *)v69 != 17 )
  {
    v124 = *(_QWORD *)(v69 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v124 + 8) - 17 > 1 || *(_BYTE *)v69 > 0x15u )
      goto LABEL_98;
    v105 = v53;
    v112 = *((_QWORD *)v53 - 8);
    v81 = sub_AD7630(v69, 0);
    v53 = v105;
    if ( !v81 || *(_BYTE *)v81 != 17 )
    {
      if ( *(_BYTE *)(v124 + 8) == 17 )
      {
        v126 = *(_DWORD *)(v124 + 32);
        if ( v126 )
        {
          v87 = 0;
          v88 = 0;
          while ( 1 )
          {
            v96 = v87;
            v100 = v88;
            v89 = sub_AD69F0(v112, v88);
            v90 = v100;
            v87 = v96;
            if ( !v89 )
              break;
            if ( *(_BYTE *)v89 != 13 )
            {
              if ( *(_BYTE *)v89 != 17 )
                break;
              if ( *(_DWORD *)(v89 + 32) <= 0x40u )
              {
                v87 = *(_QWORD *)(v89 + 24) == 0;
              }
              else
              {
                v97 = v100;
                v101 = *(_DWORD *)(v89 + 32);
                v91 = sub_C444A0(v89 + 24);
                v90 = v97;
                v87 = v101 == v91;
              }
              if ( !v87 )
                break;
            }
            v88 = v90 + 1;
            if ( v126 == v88 )
            {
              v53 = v105;
              if ( v87 )
                goto LABEL_134;
              goto LABEL_135;
            }
          }
        }
      }
      goto LABEL_135;
    }
    if ( *(_DWORD *)(v81 + 32) <= 0x40u )
    {
      if ( *(_QWORD *)(v81 + 24) )
        goto LABEL_135;
      goto LABEL_134;
    }
    v111 = v105;
    v70 = v81 + 24;
    v123 = *(_DWORD *)(v81 + 32);
    goto LABEL_132;
  }
  if ( *(_DWORD *)(v69 + 32) > 0x40u )
  {
    v111 = v53;
    v70 = v69 + 24;
    v123 = *(_DWORD *)(v69 + 32);
LABEL_132:
    v71 = sub_C444A0(v70);
    v53 = v111;
    v72 = v123 == v71;
    goto LABEL_133;
  }
  v72 = *(_QWORD *)(v69 + 24) == 0;
LABEL_133:
  if ( !v72 )
  {
LABEL_135:
    v53 = (_BYTE *)*((_QWORD *)v9 - 4);
LABEL_136:
    if ( !v53 )
      goto LABEL_45;
    v52 = (_BYTE *)*((_QWORD *)v9 - 8);
LABEL_98:
    v137 = v53;
    if ( *v52 == 44 )
    {
      v54 = *((_QWORD *)v52 - 8);
      if ( *(_BYTE *)v54 == 17 )
      {
        v122 = (unsigned __int8 **)v52;
        v55 = sub_9867B0(v54 + 24);
        v56 = v122;
        v57 = v55;
      }
      else
      {
        v113 = (unsigned __int8 **)v52;
        v125 = *(_QWORD *)(v54 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 > 1 || *(_BYTE *)v54 > 0x15u )
          goto LABEL_45;
        v106 = *((_QWORD *)v52 - 8);
        v83 = (_BYTE *)sub_AD7630(v54, 0);
        v84 = v106;
        if ( v83 && *v83 == 17 )
        {
          v85 = sub_9867B0((__int64)(v83 + 24));
          v56 = v113;
          v57 = v85;
        }
        else
        {
          if ( *(_BYTE *)(v125 + 8) != 17 )
            goto LABEL_45;
          v92 = *(_DWORD *)(v125 + 32);
          v107 = v113;
          v57 = 0;
          v127 = v9;
          v93 = v84;
          v114 = v92;
          v102 = v8;
          for ( i = 0; v114 != i; ++i )
          {
            v98 = v57;
            v95 = (_BYTE *)sub_AD69F0(v93, i);
            if ( v95 )
            {
              v57 = v98;
              if ( *v95 == 13 )
                continue;
              if ( *v95 == 17 )
              {
                v57 = sub_9867B0((__int64)(v95 + 24));
                if ( v57 )
                  continue;
              }
            }
            v9 = v127;
            goto LABEL_45;
          }
          v56 = v107;
          v9 = v127;
          v8 = v102;
        }
      }
      if ( v57 && *(v56 - 4) == v137 )
        goto LABEL_103;
    }
LABEL_45:
    v14 = 1;
    goto LABEL_46;
  }
LABEL_134:
  if ( *((unsigned __int8 **)v53 - 4) != v137 )
    goto LABEL_135;
LABEL_103:
  v58 = *(_DWORD *)(v8 + 24);
  if ( v58 <= 0x40 )
  {
    _RCX = *(_QWORD *)(v8 + 16);
    v59 = 64;
    __asm { tzcnt   rsi, rcx }
    if ( _RCX )
      v59 = _RSI;
    if ( v58 <= v59 )
      v59 = *(_DWORD *)(v8 + 24);
  }
  else
  {
    v59 = sub_C44590(v130);
  }
  v60 = *(_DWORD *)(a5 + 24);
  if ( v60 <= 0x40 )
  {
    _RSI = *(_QWORD *)(a5 + 16);
    v75 = 64;
    __asm { tzcnt   rdi, rsi }
    if ( _RSI )
      v75 = _RDI;
    if ( v60 > v75 )
      v60 = v75;
  }
  else
  {
    v134 = v59;
    v60 = sub_C44590(a5 + 16);
    v59 = v134;
  }
  v61 = v8;
  if ( v60 < v59 )
    v61 = a5;
  sub_C7C0F0(&v138, v61);
  sub_984AC0(a1, (__int64 *)&v138);
  sub_969240((__int64 *)&v140);
  sub_969240((__int64 *)&v138);
LABEL_46:
  v35 = *a1;
  if ( *((_DWORD *)a1 + 2) > 0x40u )
    v35 = *(_QWORD *)v35;
  if ( (v35 & 1) != 0 )
    return a1;
  v36 = (_QWORD *)a1[2];
  if ( *((_DWORD *)a1 + 6) > 0x40u )
    v36 = (_QWORD *)*v36;
  if ( ((unsigned __int8)v36 & 1) != 0 || (unsigned __int8)(*v9 - 42) > 0x11u )
    return a1;
  v45 = (unsigned __int8 *)*((_QWORD *)v9 - 8);
  v46 = (unsigned __int8 *)*((_QWORD *)v9 - 4);
  if ( !v45 )
  {
    if ( v46 )
    {
      v137 = (unsigned __int8 *)*((_QWORD *)v9 - 4);
      BUG();
    }
    return a1;
  }
  v137 = (unsigned __int8 *)*((_QWORD *)v9 - 8);
  if ( *v46 == 42 )
  {
    v68 = (unsigned __int8 *)*((_QWORD *)v46 - 8);
    v47 = (unsigned __int8 *)*((_QWORD *)v46 - 4);
    if ( v45 == v68 && v68 )
    {
      if ( v47 )
        goto LABEL_89;
    }
    else if ( v47 != 0 && v45 == v47 && v68 )
    {
      goto LABEL_128;
    }
  }
  v137 = v46;
  if ( *v45 != 42 )
    goto LABEL_82;
  v68 = (unsigned __int8 *)*((_QWORD *)v45 - 8);
  v47 = (unsigned __int8 *)*((_QWORD *)v45 - 4);
  if ( v68 != v46 || !v68 )
  {
    if ( v47 == 0 || v47 != v46 || !v68 )
      goto LABEL_82;
LABEL_128:
    v47 = v68;
    goto LABEL_89;
  }
  if ( v47 )
    goto LABEL_89;
LABEL_82:
  v137 = v45;
  if ( *v46 == 44 )
  {
    v80 = (unsigned __int8 *)*((_QWORD *)v46 - 8);
    if ( v45 == v80 )
    {
      if ( v80 )
      {
        v47 = (unsigned __int8 *)*((_QWORD *)v46 - 4);
        if ( v47 )
          goto LABEL_89;
      }
    }
  }
  v137 = v46;
  if ( *v45 == 44 )
  {
    v82 = (unsigned __int8 *)*((_QWORD *)v45 - 8);
    if ( v82 )
    {
      if ( v82 == v46 )
      {
        v47 = (unsigned __int8 *)*((_QWORD *)v45 - 4);
        if ( v47 )
          goto LABEL_89;
      }
    }
  }
  if ( (v137 = v45, *v46 == 44)
    && (v47 = (unsigned __int8 *)*((_QWORD *)v46 - 8)) != 0
    && (v86 = (unsigned __int8 *)*((_QWORD *)v46 - 4), v45 == v86)
    && v86
    || (v137 = v46, *v45 == 44)
    && (v47 = (unsigned __int8 *)*((_QWORD *)v45 - 8)) != 0
    && (v48 = (unsigned __int8 *)*((_QWORD *)v45 - 4), v48 == v46)
    && v48 )
  {
LABEL_89:
    sub_9878D0((__int64)&v138, v129);
    sub_9AB8E0(v47, a3, (unsigned __int64 *)&v138, a6 + 1, a7);
    v49 = v141;
    if ( v141 > 0x40 )
    {
      if ( !(unsigned int)sub_C445E0(&v140) )
        goto LABEL_143;
    }
    else
    {
      _RAX = ~v140;
      if ( v140 != -1 )
      {
        __asm { tzcnt   rax, rax }
        if ( !(_DWORD)_RAX )
        {
LABEL_92:
          if ( (unsigned int)v139 > 0x40 && v138 )
            j_j___libc_free_0_0(v138);
          return a1;
        }
      }
    }
    if ( v14 )
    {
      v73 = *a1;
      if ( *((_DWORD *)a1 + 2) <= 0x40u )
      {
        *a1 = v73 | 1;
        goto LABEL_142;
      }
    }
    else
    {
      v73 = a1[2];
      if ( *((_DWORD *)a1 + 6) <= 0x40u )
      {
        a1[2] = v73 | 1;
LABEL_142:
        if ( v49 <= 0x40 )
          goto LABEL_92;
LABEL_143:
        if ( v140 )
          j_j___libc_free_0_0(v140);
        goto LABEL_92;
      }
    }
    *(_QWORD *)v73 |= 1uLL;
    v49 = v141;
    goto LABEL_142;
  }
  return a1;
}
