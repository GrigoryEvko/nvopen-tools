// Function: sub_1742F80
// Address: 0x1742f80
//
__int64 __fastcall sub_1742F80(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // r13
  double v11; // xmm4_8
  double v12; // xmm5_8
  __int64 result; // rax
  unsigned __int64 v14; // rax
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 v17; // rdx
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // rbx
  int v21; // r12d
  __int64 v22; // r15
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // r12
  __int64 v27; // rax
  __int64 v28; // r12
  _QWORD *v29; // rdi
  __int64 v30; // rax
  char v31; // al
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r15
  __int64 v37; // rax
  unsigned __int64 v38; // rbx
  __int64 *v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // r14
  __int64 v42; // rdi
  __int64 *v43; // rax
  __int64 v44; // r14
  _QWORD *v45; // rax
  __int64 **v46; // rax
  __int64 v47; // r12
  unsigned __int64 v48; // r15
  _QWORD *v49; // rdi
  __int64 v50; // r15
  __int64 v51; // rbx
  _QWORD *v52; // rax
  double v53; // xmm4_8
  double v54; // xmm5_8
  __int64 v55; // r14
  __int64 *v56; // rax
  unsigned __int64 v57; // r14
  __int64 v58; // r15
  _QWORD *v59; // rax
  __int64 **v60; // rax
  __int64 v61; // rsi
  __int64 v62; // rbx
  __int64 v63; // rdx
  __int64 v64; // rcx
  _QWORD *v65; // rdi
  __int64 v66; // rax
  __int64 v67; // r14
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rdi
  _QWORD *v71; // rax
  _QWORD *v72; // rbx
  _QWORD *v73; // r15
  unsigned __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r8
  __int64 v77; // r15
  unsigned __int64 v78; // rbx
  _BYTE **v79; // r13
  __int64 v80; // rbx
  __int64 v81; // r14
  bool v82; // r12
  __int64 v83; // rdi
  unsigned __int8 v84; // al
  __int64 v85; // rdi
  unsigned __int8 v86; // al
  __int64 v87; // rdi
  unsigned __int8 v88; // al
  __int64 v89; // rax
  __int64 v90; // r8
  __int64 v91; // r14
  unsigned __int64 v92; // rax
  bool v93; // al
  _QWORD *v94; // rax
  bool v95; // zf
  __int64 v96; // rcx
  unsigned __int64 v97; // rsi
  __int64 v98; // rcx
  __int64 v99; // rdx
  unsigned __int64 v100; // r14
  __int64 v101; // rax
  __int64 v102; // r15
  __int64 v103; // rbx
  _QWORD *v104; // rax
  double v105; // xmm4_8
  double v106; // xmm5_8
  unsigned __int64 v107; // rbx
  _QWORD *v108; // rdi
  __int64 v109; // rax
  unsigned __int64 v110; // rbx
  __int64 *v111; // rax
  __int64 v112; // rbx
  __int64 v113; // r13
  _QWORD *v114; // r12
  _QWORD *v115; // rax
  __int64 v116; // rdx
  int v117; // ecx
  bool v118; // al
  unsigned __int64 v119; // r14
  __int64 v120; // rax
  __int64 v121; // rdx
  _BYTE *v122; // rcx
  unsigned __int64 v123; // rsi
  __int64 v124; // rdx
  _BYTE *v125; // rdx
  unsigned __int64 v126; // rcx
  __int64 v127; // rdx
  int v128; // edx
  __int64 v129; // rax
  __int64 v130; // rax
  __int64 v131; // rax
  _QWORD *v132; // [rsp+8h] [rbp-A8h]
  __int64 v133; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v134; // [rsp+18h] [rbp-98h]
  __int64 v135; // [rsp+18h] [rbp-98h]
  __int64 v136; // [rsp+18h] [rbp-98h]
  __int64 v137; // [rsp+18h] [rbp-98h]
  __int64 v138; // [rsp+20h] [rbp-90h]
  unsigned __int64 v139; // [rsp+28h] [rbp-88h]
  __int64 *v140; // [rsp+28h] [rbp-88h]
  __int64 v141; // [rsp+30h] [rbp-80h]
  unsigned __int64 v142; // [rsp+30h] [rbp-80h]
  __int64 *v143; // [rsp+30h] [rbp-80h]
  char v144; // [rsp+40h] [rbp-70h]
  __int64 v145; // [rsp+40h] [rbp-70h]
  __int64 ***v146; // [rsp+40h] [rbp-70h]
  __int64 ***v147; // [rsp+40h] [rbp-70h]
  __int64 v148[2]; // [rsp+48h] [rbp-68h] BYREF
  __int64 v149; // [rsp+58h] [rbp-58h] BYREF
  int *v150; // [rsp+60h] [rbp-50h] BYREF
  __int64 v151; // [rsp+68h] [rbp-48h]
  _BYTE v152[64]; // [rsp+70h] [rbp-40h] BYREF

  v10 = (__int64 *)a1;
  v148[0] = a2;
  v144 = sub_140B1C0(a2 & 0xFFFFFFFFFFFFFFF8LL, *(_QWORD **)(a1 + 2648), 0);
  if ( v144 )
    return sub_170E170((__int64 *)a1, v148[0] & 0xFFFFFFFFFFFFFFF8LL, a3, a4, a5, a6, v11, v12, a9, a10);
  v150 = (int *)v152;
  v151 = 0x400000000LL;
  v14 = sub_1389B50(v148);
  v17 = v148[0];
  v18 = v14;
  v19 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
  v20 = (v148[0] & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v148[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  if ( v14 != v20 )
  {
    v21 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v22 = *(_QWORD *)v20;
        if ( *(_BYTE *)(**(_QWORD **)v20 + 8LL) == 15 )
          break;
        v20 += 24LL;
        ++v21;
        if ( v18 == v20 )
          goto LABEL_13;
      }
      v139 = v17 & 0xFFFFFFFFFFFFFFF8LL;
      v23 = (_QWORD *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 56);
      if ( (v17 & 4) != 0 )
      {
        if ( (unsigned __int8)sub_1560290(v23, v21, 32) )
          goto LABEL_12;
        v24 = *(_QWORD *)(v139 - 24);
        if ( *(_BYTE *)(v24 + 16) )
          goto LABEL_26;
      }
      else
      {
        if ( (unsigned __int8)sub_1560290(v23, v21, 32) )
          goto LABEL_12;
        v24 = *(_QWORD *)(v139 - 72);
        if ( *(_BYTE *)(v24 + 16) )
        {
LABEL_26:
          if ( (unsigned __int8)sub_14BFF20(v22, v10[333], 0, v10[330], v148[0] & 0xFFFFFFFFFFFFFFF8LL, v10[332]) )
          {
            v34 = (unsigned int)v151;
            if ( (unsigned int)v151 >= HIDWORD(v151) )
            {
              sub_16CD150((__int64)&v150, v152, 0, 4, v32, v33);
              v34 = (unsigned int)v151;
            }
            v150[v34] = v21;
            LODWORD(v151) = v151 + 1;
          }
          goto LABEL_12;
        }
      }
      v149 = *(_QWORD *)(v24 + 112);
      if ( !(unsigned __int8)sub_1560290(&v149, v21, 32) )
        goto LABEL_26;
LABEL_12:
      v20 += 24LL;
      v17 = v148[0];
      ++v21;
      if ( v18 == v20 )
      {
LABEL_13:
        v19 = v17 & 0xFFFFFFFFFFFFFFF8LL;
        v25 = (v17 >> 2) & 1;
        goto LABEL_14;
      }
    }
  }
  v25 = (v148[0] >> 2) & 1;
LABEL_14:
  if ( (_DWORD)v151 )
  {
    v149 = *(_QWORD *)(v19 + 56);
    v26 = (__int64 *)sub_16498A0(v19);
    v27 = sub_155CEC0(v26, 32, 0);
    v149 = sub_1563E10(&v149, v26, v150, (unsigned int)v151, v27);
    v19 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
    LOBYTE(v25) = v148[0] >> 2;
    *(_QWORD *)((v148[0] & 0xFFFFFFFFFFFFFFF8LL) + 56) = v149;
    LOBYTE(v25) = v25 & 1;
    v144 = 1;
  }
  if ( (_BYTE)v25 )
  {
    v28 = *(_QWORD *)(v19 - 24);
    v29 = (_QWORD *)(v19 + 56);
    if ( !*(_BYTE *)(v28 + 16) )
    {
LABEL_18:
      if ( (unsigned __int8)sub_1560260(v29, -1, 8) )
        goto LABEL_46;
      v30 = *(_QWORD *)(v19 - 24);
      if ( *(_BYTE *)(v30 + 16) )
        goto LABEL_21;
LABEL_20:
      v149 = *(_QWORD *)(v30 + 112);
      if ( !(unsigned __int8)sub_1560260(&v149, -1, 8) )
        goto LABEL_21;
LABEL_46:
      if ( !(unsigned __int8)sub_1560180(v28 + 112, 8) && (*(_BYTE *)(v28 + 33) & 0x20) == 0 )
      {
        v38 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
        v149 = *(_QWORD *)((v148[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
        v39 = (__int64 *)sub_16498A0(v148[0] & 0xFFFFFFFFFFFFFFF8LL);
        v149 = sub_1563C10(&v149, v39, -1, 8);
        *(_QWORD *)(v38 + 56) = v149;
LABEL_49:
        result = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
        goto LABEL_43;
      }
      goto LABEL_21;
    }
  }
  else
  {
    v28 = *(_QWORD *)(v19 - 72);
    v29 = (_QWORD *)(v19 + 56);
    if ( !*(_BYTE *)(v28 + 16) )
      goto LABEL_31;
  }
  if ( (unsigned __int8)sub_1740B60(v10, v148[0], a3, a4, a5, a6, v15, v16, a9, a10) )
  {
LABEL_42:
    result = 0;
    goto LABEL_43;
  }
  v31 = *(_BYTE *)(v28 + 16);
  if ( !v31 )
  {
    v19 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
    v29 = (_QWORD *)((v148[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
    if ( (v148[0] & 4) != 0 )
      goto LABEL_18;
LABEL_31:
    if ( (unsigned __int8)sub_1560260(v29, -1, 8) )
      goto LABEL_46;
    v30 = *(_QWORD *)(v19 - 72);
    if ( *(_BYTE *)(v30 + 16) )
    {
LABEL_21:
      if ( ((*(_WORD *)(v28 + 18) >> 4) & 0x3FF) != ((*(unsigned __int16 *)((v148[0] & 0xFFFFFFFFFFFFFFF8LL) + 18) >> 2)
                                                   & 0x3FFFDFFF)
        && !sub_15E4F60(v28) )
      {
        v55 = v148[0];
        v56 = (__int64 *)sub_16498A0(v28);
        v57 = v55 & 0xFFFFFFFFFFFFFFF8LL;
        v58 = sub_159C4F0(v56);
        v59 = (_QWORD *)sub_16498A0(v28);
        v60 = (__int64 **)sub_16471A0(v59, 0);
        v61 = 2;
        v62 = sub_1599EF0(v60);
        v65 = sub_1648A60(64, 2u);
        if ( v65 )
        {
          v61 = v58;
          sub_15F9660((__int64)v65, v58, v62, v57);
        }
        if ( *(_BYTE *)(*(_QWORD *)v57 + 8LL) )
        {
          v101 = sub_1599EF0(*(__int64 ***)v57);
          v102 = *(_QWORD *)(v57 + 8);
          v147 = (__int64 ***)v101;
          if ( v102 )
          {
            v103 = *v10;
            do
            {
              v104 = sub_1648700(v102);
              sub_170B990(v103, (__int64)v104);
              v102 = *(_QWORD *)(v102 + 8);
            }
            while ( v102 );
            if ( v147 == (__int64 ***)v57 )
              v147 = (__int64 ***)sub_1599EF0(*v147);
            v61 = (__int64)v147;
            sub_164D160(v57, (__int64)v147, a3, a4, a5, a6, v105, v106, a9, a10);
          }
        }
        if ( *(_BYTE *)(v57 + 16) == 78 )
        {
          result = sub_170BC50((__int64)v10, v57);
        }
        else
        {
          v94 = (_QWORD *)sub_15A06D0(*(__int64 ***)v28, v61, v63, v64);
          v95 = *(_QWORD *)(v57 - 72) == 0;
          *(_QWORD *)(v57 + 64) = *(_QWORD *)(*v94 + 24LL);
          if ( !v95 )
          {
            v96 = *(_QWORD *)(v57 - 64);
            v97 = *(_QWORD *)(v57 - 56) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v97 = v96;
            if ( v96 )
              *(_QWORD *)(v96 + 16) = v97 | *(_QWORD *)(v96 + 16) & 3LL;
          }
          *(_QWORD *)(v57 - 72) = v94;
          v98 = v94[1];
          *(_QWORD *)(v57 - 64) = v98;
          if ( v98 )
            *(_QWORD *)(v98 + 16) = (v57 - 64) | *(_QWORD *)(v98 + 16) & 3LL;
          v99 = *(_QWORD *)(v57 - 56);
          v100 = v57 - 72;
          *(_QWORD *)(v100 + 16) = v99 & 3 | (unsigned __int64)(v94 + 1);
          v94[1] = v100;
          result = 0;
        }
        goto LABEL_43;
      }
      v31 = *(_BYTE *)(v28 + 16);
      goto LABEL_36;
    }
    goto LABEL_20;
  }
LABEL_36:
  if ( v31 == 15 )
  {
    v40 = sub_15F2060(v148[0] & 0xFFFFFFFFFFFFFFF8LL);
    if ( !sub_15E4690(v40, 0) )
      goto LABEL_51;
    v31 = *(_BYTE *)(v28 + 16);
  }
  if ( v31 == 9 )
  {
LABEL_51:
    v41 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
    v42 = *(_QWORD *)(v148[0] & 0xFFFFFFFFFFFFFFF8LL);
    if ( *(_BYTE *)(v42 + 8) )
    {
      v146 = (__int64 ***)sub_1599EF0((__int64 **)v42);
      v41 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
      v50 = *(_QWORD *)((v148[0] & 0xFFFFFFFFFFFFFFF8LL) + 8);
      if ( v50 )
      {
        v51 = *v10;
        do
        {
          v52 = sub_1648700(v50);
          sub_170B990(v51, (__int64)v52);
          v50 = *(_QWORD *)(v50 + 8);
        }
        while ( v50 );
        if ( v146 == (__int64 ***)v41 )
          v146 = (__int64 ***)sub_1599EF0(*v146);
        sub_164D160(v41, (__int64)v146, a3, a4, a5, a6, v53, v54, a9, a10);
        v41 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
      }
    }
    if ( *(_BYTE *)(v41 + 16) != 29 )
    {
      v43 = (__int64 *)sub_16498A0(v28);
      v44 = sub_159C4F0(v43);
      v45 = (_QWORD *)sub_16498A0(v28);
      v46 = (__int64 **)sub_16471A0(v45, 0);
      v47 = sub_1599EF0(v46);
      v48 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
      v49 = sub_1648A60(64, 2u);
      if ( v49 )
        sub_15F9660((__int64)v49, v44, v47, v48);
      result = sub_170BC50((__int64)v10, v148[0] & 0xFFFFFFFFFFFFFFF8LL);
      goto LABEL_43;
    }
    goto LABEL_42;
  }
  v35 = sub_1649C60(v28);
  v36 = v35;
  if ( *(_BYTE *)(v35 + 16) == 78 )
  {
    v66 = *(_QWORD *)(v35 - 24);
    if ( !*(_BYTE *)(v66 + 16) && (*(_BYTE *)(v66 + 33) & 0x20) != 0 && *(_DWORD *)(v66 + 36) == 2 )
    {
      v67 = *(_QWORD *)(v36 - 24LL * (*(_DWORD *)(v36 + 20) & 0xFFFFFFF));
      v68 = sub_1649C60(v67);
      v69 = v68;
      if ( v67 != v68 )
      {
        v70 = *(_QWORD *)(v68 + 8);
        v141 = v68;
        if ( !v70 )
          goto LABEL_76;
        if ( *(_QWORD *)(v70 + 8) )
          goto LABEL_76;
        v71 = sub_1648700(v70);
        v69 = v141;
        if ( (_QWORD *)v67 != v71 )
          goto LABEL_76;
      }
      if ( *(_BYTE *)(v69 + 16) != 53 )
        goto LABEL_76;
      v112 = *(_QWORD *)(v67 + 8);
      if ( !v112 )
        goto LABEL_76;
      v143 = v10;
      v113 = v28;
      v114 = 0;
      do
      {
        v115 = sub_1648700(v112);
        if ( *((_BYTE *)v115 + 16) != 78
          || (v127 = *(v115 - 3), *(_BYTE *)(v127 + 16))
          || (*(_BYTE *)(v127 + 33) & 0x20) == 0 )
        {
LABEL_136:
          v28 = v113;
          v10 = v143;
          goto LABEL_76;
        }
        v128 = *(_DWORD *)(v127 + 36);
        if ( v128 == 109 )
        {
          if ( v114 )
            goto LABEL_136;
          v114 = v115;
        }
        else if ( v128 != 2 )
        {
          goto LABEL_136;
        }
        v112 = *(_QWORD *)(v112 + 8);
      }
      while ( v112 );
      v76 = (__int64)v114;
      v28 = v113;
      v10 = v143;
      if ( !v76 || (v129 = *(_QWORD *)(v76 - 24LL * (*(_DWORD *)(v76 + 20) & 0xFFFFFFF))) == 0 || v67 != v129 )
      {
LABEL_76:
        v72 = (_QWORD *)(v36 + 24);
        v73 = *(_QWORD **)(*(_QWORD *)(v36 + 40) + 48LL);
        while ( 1 )
        {
          if ( v72 == v73 )
            goto LABEL_39;
          v74 = *v72 & 0xFFFFFFFFFFFFFFF8LL;
          v72 = (_QWORD *)v74;
          if ( !v74 )
            BUG();
          if ( *(_BYTE *)(v74 - 8) == 78 )
          {
            v75 = *(_QWORD *)(v74 - 48);
            if ( !*(_BYTE *)(v75 + 16)
              && (*(_BYTE *)(v75 + 33) & 0x20) != 0
              && *(_DWORD *)(v75 + 36) == 109
              && v67 == *(_QWORD *)(v74 - 24LL * (*(_DWORD *)(v74 - 4) & 0xFFFFFFF) - 24) )
            {
              break;
            }
          }
          if ( (unsigned __int8)sub_15F3040(v74 - 24) )
            goto LABEL_39;
        }
        v76 = v74 - 24;
      }
      result = sub_1741D80((__int64)v10, v148[0], v76);
      goto LABEL_43;
    }
  }
LABEL_39:
  v37 = *(_QWORD *)(*(_QWORD *)v28 + 24LL);
  if ( *(_DWORD *)(v37 + 8) >> 8 )
  {
    v77 = (unsigned int)(*(_DWORD *)(v37 + 12) - 1);
    v78 = (v148[0] & 0xFFFFFFFFFFFFFFF8LL)
        + 24 * (v77 - (*(_DWORD *)((v148[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
    v142 = sub_1389B50(v148);
    if ( v78 != v142 )
    {
      v133 = v28;
      v140 = v10;
      v79 = (_BYTE **)v78;
      while ( 1 )
      {
        v80 = (__int64)*v79;
        if ( (unsigned __int8)((*v79)[16] - 60) > 0xCu )
          goto LABEL_114;
        v81 = v148[0];
        v138 = v140[333];
        v82 = sub_15FB860(*v79);
        if ( !v82 )
          goto LABEL_114;
        v83 = 0;
        v134 = v81 & 0xFFFFFFFFFFFFFFF8LL;
        v84 = *(_BYTE *)((v81 & 0xFFFFFFFFFFFFFFF8LL) + 16);
        if ( v84 > 0x17u )
        {
          if ( v84 == 78 )
          {
            v83 = v134 | 4;
          }
          else
          {
            v83 = 0;
            if ( v84 == 29 )
              v83 = v81 & 0xFFFFFFFFFFFFFFF8LL;
          }
        }
        if ( sub_1642CF0(v83) )
          goto LABEL_114;
        v85 = 0;
        v86 = *(_BYTE *)(v134 + 16);
        if ( v86 > 0x17u )
        {
          if ( v86 == 78 )
          {
            v85 = v134 | 4;
          }
          else
          {
            v85 = 0;
            if ( v86 == 29 )
              v85 = v81 & 0xFFFFFFFFFFFFFFF8LL;
          }
        }
        if ( sub_1642D80(v85) )
          goto LABEL_114;
        v87 = 0;
        v88 = *(_BYTE *)(v134 + 16);
        if ( v88 > 0x17u )
        {
          if ( v88 == 78 )
          {
            v87 = v134 | 4;
          }
          else
          {
            v87 = 0;
            if ( v88 == 29 )
              v87 = v81 & 0xFFFFFFFFFFFFFFF8LL;
          }
        }
        if ( sub_1642DF0(v87) )
          goto LABEL_114;
        v132 = (_QWORD *)(v134 + 56);
        if ( (v81 & 4) != 0 )
        {
          if ( !(unsigned __int8)sub_1560290((_QWORD *)(v134 + 56), v77, 6) )
          {
            v89 = *(_QWORD *)(v134 - 24);
            if ( *(_BYTE *)(v89 + 16) || (v149 = *(_QWORD *)(v89 + 112), !(unsigned __int8)sub_1560290(&v149, v77, 6)) )
            {
              if ( !(unsigned __int8)sub_1560290(v132, v77, 11) )
              {
                v131 = *(_QWORD *)(v134 - 24);
                if ( *(_BYTE *)(v131 + 16) )
                  goto LABEL_148;
                goto LABEL_175;
              }
            }
          }
        }
        else if ( !(unsigned __int8)sub_1560290(v132, v77, 6) )
        {
          v130 = *(_QWORD *)(v134 - 72);
          if ( *(_BYTE *)(v130 + 16) || (v149 = *(_QWORD *)(v130 + 112), !(unsigned __int8)sub_1560290(&v149, v77, 6)) )
          {
            if ( !(unsigned __int8)sub_1560290(v132, v77, 11) )
            {
              v131 = *(_QWORD *)(v134 - 72);
              if ( *(_BYTE *)(v131 + 16) )
                goto LABEL_148;
LABEL_175:
              v149 = *(_QWORD *)(v131 + 112);
              if ( !(unsigned __int8)sub_1560290(&v149, v77, 11) )
                goto LABEL_148;
            }
          }
        }
        v90 = *(_QWORD *)(*(_QWORD *)v80 + 24LL);
        v91 = *(_QWORD *)(**(_QWORD **)(v80 - 24) + 24LL);
        v92 = *(unsigned __int8 *)(v91 + 8);
        if ( (unsigned __int8)v92 > 0xFu || (v116 = 35454, !_bittest64(&v116, v92)) )
        {
          if ( (unsigned int)(v92 - 13) > 1 && (_DWORD)v92 != 16 )
            goto LABEL_114;
          v135 = *(_QWORD *)(*(_QWORD *)v80 + 24LL);
          v93 = sub_16435F0(*(_QWORD *)(**(_QWORD **)(v80 - 24) + 24LL), 0);
          v90 = v135;
          if ( !v93 )
            goto LABEL_114;
        }
        v117 = *(unsigned __int8 *)(v90 + 8);
        if ( (unsigned __int8)v117 > 0xFu || ((0x8A7EuLL >> v117) & 1) == 0 )
        {
          if ( (unsigned int)(v117 - 13) > 1 && v117 != 16 )
            goto LABEL_114;
          v136 = v90;
          v118 = sub_16435F0(v90, 0);
          v90 = v136;
          if ( !v118 )
            goto LABEL_114;
        }
        v137 = v90;
        v119 = sub_12BE0A0(v138, v91);
        if ( v119 != sub_12BE0A0(v138, v137) )
          goto LABEL_114;
LABEL_148:
        v120 = *(_QWORD *)(v80 - 24);
        v121 = (__int64)*v79;
        if ( v120 )
        {
          if ( v121 )
          {
            v122 = v79[1];
            v123 = (unsigned __int64)v79[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v123 = v122;
            if ( v122 )
              *((_QWORD *)v122 + 2) = v123 | *((_QWORD *)v122 + 2) & 3LL;
          }
          *v79 = (_BYTE *)v120;
          v124 = *(_QWORD *)(v120 + 8);
          v79[1] = (_BYTE *)v124;
          if ( v124 )
            *(_QWORD *)(v124 + 16) = (unsigned __int64)(v79 + 1) | *(_QWORD *)(v124 + 16) & 3LL;
          v79[2] = (_BYTE *)((v120 + 8) | (unsigned __int64)v79[2] & 3);
          *(_QWORD *)(v120 + 8) = v79;
        }
        else if ( v121 )
        {
          v125 = v79[1];
          v126 = (unsigned __int64)v79[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v126 = v125;
          if ( !v125 )
          {
            *v79 = 0;
            v144 = v82;
            goto LABEL_114;
          }
          *((_QWORD *)v125 + 2) = v126 | *((_QWORD *)v125 + 2) & 3LL;
          *v79 = 0;
        }
        v144 = v82;
LABEL_114:
        v79 += 3;
        LODWORD(v77) = v77 + 1;
        if ( (_BYTE **)v142 == v79 )
        {
          v28 = v133;
          break;
        }
      }
    }
  }
  if ( *(_BYTE *)(v28 + 16) != 20 )
  {
LABEL_41:
    if ( !v144 )
      goto LABEL_42;
    goto LABEL_49;
  }
  v107 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
  v108 = (_QWORD *)((v148[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( (v148[0] & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v108, -1, 30) )
      goto LABEL_41;
    v109 = *(_QWORD *)(v107 - 24);
    if ( *(_BYTE *)(v109 + 16) )
      goto LABEL_132;
    goto LABEL_131;
  }
  if ( (unsigned __int8)sub_1560260(v108, -1, 30) )
    goto LABEL_41;
  v109 = *(_QWORD *)(v107 - 72);
  if ( !*(_BYTE *)(v109 + 16) )
  {
LABEL_131:
    v149 = *(_QWORD *)(v109 + 112);
    if ( (unsigned __int8)sub_1560260(&v149, -1, 30) )
      goto LABEL_41;
  }
LABEL_132:
  v110 = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
  v149 = *(_QWORD *)((v148[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
  v111 = (__int64 *)sub_16498A0(v148[0] & 0xFFFFFFFFFFFFFFF8LL);
  v149 = sub_1563AB0(&v149, v111, -1, 30);
  *(_QWORD *)(v110 + 56) = v149;
  result = v148[0] & 0xFFFFFFFFFFFFFFF8LL;
LABEL_43:
  if ( v150 != (int *)v152 )
  {
    v145 = result;
    _libc_free((unsigned __int64)v150);
    return v145;
  }
  return result;
}
