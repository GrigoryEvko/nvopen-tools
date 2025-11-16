// Function: sub_17512B0
// Address: 0x17512b0
//
__int64 __fastcall sub_17512B0(
        __int64 a1,
        __int64 ***a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v12; // r15
  __int64 v14; // r14
  __int64 **v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 ***v21; // rax
  __int64 v22; // r13
  __int64 v23; // r14
  __int64 v24; // r12
  _QWORD *v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  int v28; // edx
  char v29; // al
  __int64 v30; // rcx
  int v31; // edx
  int v32; // edx
  __int64 **v33; // rcx
  __int64 v34; // r8
  __int64 **v35; // rax
  unsigned int v36; // eax
  unsigned int v37; // ecx
  unsigned int v38; // r9d
  _QWORD *v39; // rcx
  __int64 v40; // rax
  unsigned int v41; // edx
  __int64 v42; // r12
  __int64 v43; // rax
  _QWORD *v44; // r12
  __int64 v45; // rax
  __int64 v46; // rdi
  unsigned __int8 *v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 **v52; // rax
  __int64 *v53; // r14
  __int64 *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rdx
  int v57; // eax
  int v58; // eax
  int v59; // eax
  __int64 **v60; // rdx
  __int64 v61; // r9
  __int64 v62; // r11
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 **v66; // r14
  unsigned __int8 v67; // dl
  int v68; // eax
  __int64 **v69; // rax
  __int64 v70; // rcx
  unsigned int v71; // eax
  __int64 v72; // rdx
  _QWORD *v73; // rsi
  __int64 v74; // r12
  __int64 v75; // rax
  _QWORD *v76; // r12
  __int64 v77; // rax
  __int64 v78; // r13
  __int64 v79; // r12
  _QWORD *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rcx
  int v83; // edx
  int v84; // edx
  __int64 **v85; // rcx
  unsigned int v86; // edx
  __int64 v87; // rax
  __int64 v88; // rdx
  int v89; // ecx
  int v90; // ecx
  __int64 **v91; // rdx
  __int64 *v92; // r9
  __int64 v93; // rbx
  unsigned int v94; // r13d
  unsigned int v95; // eax
  __int64 v96; // rcx
  unsigned int v97; // r10d
  unsigned __int8 *v98; // r9
  _QWORD *v99; // rax
  unsigned int v100; // ebx
  unsigned int v101; // eax
  int v102; // r13d
  __int64 **v103; // rax
  __int64 v104; // rdi
  unsigned __int8 *v105; // rax
  unsigned int v106; // ebx
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // r12
  _QWORD *v110; // rax
  __int64 v111; // rax
  __int64 **v112; // rcx
  _QWORD *v113; // rcx
  _QWORD *v114; // rax
  char v115; // al
  __int64 v116; // rdx
  __int64 v117; // r9
  __int64 v118; // rdx
  int v119; // eax
  unsigned int v120; // eax
  __int64 v121; // rdx
  __int64 v122; // r9
  unsigned int v123; // ecx
  __int64 *v124; // r14
  unsigned __int64 v125; // rax
  __int64 v126; // r15
  __int64 v127; // rdx
  __int64 *v128; // r15
  __int64 v129; // rax
  __int64 **v130; // rsi
  __int64 *v131; // r9
  unsigned int v132; // eax
  __int64 v133; // rsi
  __int64 v134; // rdx
  int v135; // ecx
  int v136; // ecx
  __int64 **v137; // rdx
  int v138; // eax
  __int64 v139; // rcx
  __int64 *v140; // [rsp+8h] [rbp-B8h]
  __int64 v141; // [rsp+10h] [rbp-B0h]
  __int64 *v142; // [rsp+18h] [rbp-A8h]
  __int64 v143; // [rsp+18h] [rbp-A8h]
  unsigned int v144; // [rsp+18h] [rbp-A8h]
  unsigned int v145; // [rsp+20h] [rbp-A0h]
  __int64 v146; // [rsp+20h] [rbp-A0h]
  __int64 **v147; // [rsp+28h] [rbp-98h]
  __int64 v148; // [rsp+28h] [rbp-98h]
  unsigned __int8 *v149; // [rsp+28h] [rbp-98h]
  __int64 v150; // [rsp+28h] [rbp-98h]
  int v151; // [rsp+30h] [rbp-90h]
  __int64 *v152; // [rsp+30h] [rbp-90h]
  _QWORD *v153; // [rsp+30h] [rbp-90h]
  __int64 *v154; // [rsp+30h] [rbp-90h]
  unsigned int v155; // [rsp+30h] [rbp-90h]
  _QWORD *v156; // [rsp+30h] [rbp-90h]
  unsigned int v157; // [rsp+30h] [rbp-90h]
  __int64 *v158; // [rsp+30h] [rbp-90h]
  __int64 v159; // [rsp+38h] [rbp-88h]
  __int64 v160; // [rsp+38h] [rbp-88h]
  __int64 v161; // [rsp+38h] [rbp-88h]
  __int64 v162; // [rsp+38h] [rbp-88h]
  unsigned __int8 *v163; // [rsp+38h] [rbp-88h]
  _QWORD *v164; // [rsp+38h] [rbp-88h]
  __int64 v165; // [rsp+38h] [rbp-88h]
  unsigned __int64 v166; // [rsp+38h] [rbp-88h]
  __int64 v167; // [rsp+38h] [rbp-88h]
  __int64 v168; // [rsp+38h] [rbp-88h]
  __int64 v169; // [rsp+40h] [rbp-80h] BYREF
  __int64 v170; // [rsp+48h] [rbp-78h] BYREF
  unsigned __int64 v171; // [rsp+54h] [rbp-6Ch]
  int v172; // [rsp+5Ch] [rbp-64h]
  __int64 v173[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v174[2]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v175; // [rsp+80h] [rbp-40h]

  v12 = sub_174B490((__int64 *)a1, (__int64)a2, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( v12 )
    return v12;
  v14 = (__int64)*(a2 - 3);
  v15 = *a2;
  v159 = *(_QWORD *)v14;
  if ( (*((_BYTE *)*a2 + 8) == 16 || (unsigned __int8)sub_1705440(a1, v159, (__int64)v15))
    && (unsigned int)sub_16431D0(v159) != 32
    && (unsigned __int8)sub_174A4F0(v14, (__int64)v15, (__int64 *)a1, (__int64)a2) )
  {
    v21 = sub_174BF40((__int64 *)a1, v14, v15, 0);
    v22 = (__int64)a2[1];
    v23 = (__int64)v21;
    if ( !v22 )
      return v12;
    v24 = *(_QWORD *)a1;
    do
    {
      v25 = sub_1648700(v22);
      sub_170B990(v24, (__int64)v25);
      v22 = *(_QWORD *)(v22 + 8);
    }
    while ( v22 );
    goto LABEL_17;
  }
  v16 = (__int64)*(a2 - 3);
  if ( *(_BYTE *)(v16 + 16) == 79 )
  {
    v171 = sub_14B2890(v16, &v169, &v170, 0, 0);
    v172 = v28;
    if ( (_DWORD)v171 )
      return v12;
  }
  if ( (unsigned __int8)sub_17AD890(a1, a2) )
    return (__int64)a2;
  if ( (unsigned int)sub_16431D0((__int64)v15) == 1 )
  {
    v45 = sub_15A0680(v159, 1, 0);
    v46 = *(_QWORD *)(a1 + 8);
    v175 = 257;
    v47 = sub_1729500(v46, (unsigned __int8 *)v14, v45, v174, *(double *)a3.m128_u64, a4, a5);
    v50 = sub_15A06D0(*(__int64 ***)v47, v14, v48, v49);
    v175 = 257;
    v51 = v50;
    v12 = (__int64)sub_1648A60(56, 2u);
    if ( v12 )
    {
      v52 = *(__int64 ***)v47;
      if ( *(_BYTE *)(*(_QWORD *)v47 + 8LL) == 16 )
      {
        v53 = v52[4];
        v54 = (__int64 *)sub_1643320(*v52);
        v55 = (__int64)sub_16463B0(v54, (unsigned int)v53);
      }
      else
      {
        v55 = sub_1643320(*v52);
      }
      sub_15FEC10(v12, v55, 51, 33, (__int64)v47, v51, (__int64)v174, 0);
    }
    return v12;
  }
  v17 = *(_QWORD *)(v14 + 8);
  if ( v17 && !*(_QWORD *)(v17 + 8) )
  {
    v29 = *(_BYTE *)(v14 + 16);
    if ( v29 != 48 )
    {
      if ( v29 != 5 )
        goto LABEL_10;
      if ( *(_WORD *)(v14 + 18) != 24 )
        goto LABEL_26;
      v82 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
      v83 = *(unsigned __int8 *)(v82 + 16);
      if ( (unsigned __int8)v83 > 0x17u )
      {
        v84 = v83 - 24;
      }
      else
      {
        if ( (_BYTE)v83 != 5 )
        {
LABEL_26:
          if ( *(_WORD *)(v14 + 18) != 24 )
            goto LABEL_10;
          v30 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
          v31 = *(unsigned __int8 *)(v30 + 16);
          if ( (unsigned __int8)v31 > 0x17u )
          {
            v32 = v31 - 24;
          }
          else
          {
            if ( (_BYTE)v31 != 5 )
              goto LABEL_10;
            v32 = *(unsigned __int16 *)(v30 + 18);
          }
          if ( v32 != 38 )
            goto LABEL_10;
          v33 = (*(_BYTE *)(v30 + 23) & 0x40) != 0
              ? *(__int64 ***)(v30 - 8)
              : (__int64 **)(v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF));
          v140 = *v33;
          if ( !*v33 )
            goto LABEL_10;
          v34 = *(_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
          if ( *(_BYTE *)(v34 + 16) != 13 )
            goto LABEL_10;
          goto LABEL_35;
        }
        v84 = *(unsigned __int16 *)(v82 + 18);
      }
      if ( v84 != 37 )
        goto LABEL_26;
      v85 = (*(_BYTE *)(v82 + 23) & 0x40) != 0
          ? *(__int64 ***)(v82 - 8)
          : (__int64 **)(v82 - 24LL * (*(_DWORD *)(v82 + 20) & 0xFFFFFFF));
      v152 = *v85;
      if ( !*v85 )
        goto LABEL_26;
      v70 = *(_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v70 + 16) != 13 )
        goto LABEL_26;
LABEL_81:
      v160 = v70;
      v71 = sub_1643030(*v152);
      v73 = *(_QWORD **)(v160 + 24);
      if ( *(_DWORD *)(v160 + 32) <= 0x40u )
      {
        if ( (unsigned __int64)v73 < v71 )
        {
          v74 = *(_QWORD *)(a1 + 8);
          v175 = 257;
          goto LABEL_84;
        }
      }
      else if ( *v73 < (unsigned __int64)v71 )
      {
        v74 = *(_QWORD *)(a1 + 8);
        v175 = 257;
        v73 = (_QWORD *)*v73;
LABEL_84:
        v75 = sub_15A0680(*v152, (__int64)v73, 0);
        v76 = sub_172C310(v74, (__int64)v152, v75, v174, 0, *(double *)a3.m128_u64, a4, a5);
        sub_164B7C0((__int64)v76, v14);
        v175 = 257;
        return sub_15FE0A0(v76, (__int64)v15, 0, (__int64)v174, 0);
      }
      v77 = sub_15A06D0(v15, (__int64)v73, v72, v160);
      v78 = (__int64)a2[1];
      v23 = v77;
      if ( !v78 )
        return v12;
      v79 = *(_QWORD *)a1;
      do
      {
        v80 = sub_1648700(v78);
        sub_170B990(v79, (__int64)v80);
        v78 = *(_QWORD *)(v78 + 8);
      }
      while ( v78 );
LABEL_17:
      if ( a2 == (__int64 ***)v23 )
        v23 = sub_1599EF0(*a2);
      v12 = (__int64)a2;
      sub_164D160((__int64)a2, v23, a3, a4, a5, a6, v26, v27, a9, a10);
      return v12;
    }
    v56 = *(_QWORD *)(v14 - 48);
    v57 = *(unsigned __int8 *)(v56 + 16);
    if ( (unsigned __int8)v57 > 0x17u )
    {
      v68 = v57 - 24;
    }
    else
    {
      if ( (_BYTE)v57 != 5 )
        goto LABEL_55;
      v68 = *(unsigned __int16 *)(v56 + 18);
    }
    if ( v68 == 37 )
    {
      v69 = (*(_BYTE *)(v56 + 23) & 0x40) != 0
          ? *(__int64 ***)(v56 - 8)
          : (__int64 **)(v56 - 24LL * (*(_DWORD *)(v56 + 20) & 0xFFFFFFF));
      v152 = *v69;
      if ( *v69 )
      {
        v70 = *(_QWORD *)(v14 - 24);
        if ( *(_BYTE *)(v70 + 16) == 13 )
          goto LABEL_81;
      }
    }
LABEL_55:
    v58 = *(unsigned __int8 *)(v56 + 16);
    if ( (unsigned __int8)v58 <= 0x17u )
    {
      if ( (_BYTE)v58 != 5 )
        goto LABEL_10;
      v59 = *(unsigned __int16 *)(v56 + 18);
    }
    else
    {
      v59 = v58 - 24;
    }
    if ( v59 != 38 )
      goto LABEL_10;
    v60 = (*(_BYTE *)(v56 + 23) & 0x40) != 0
        ? *(__int64 ***)(v56 - 8)
        : (__int64 **)(v56 - 24LL * (*(_DWORD *)(v56 + 20) & 0xFFFFFFF));
    v140 = *v60;
    if ( !*v60 )
      goto LABEL_10;
    v34 = *(_QWORD *)(v14 - 24);
    if ( *(_BYTE *)(v34 + 16) != 13 )
      goto LABEL_10;
LABEL_35:
    if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
      v35 = *(__int64 ***)(v14 - 8);
    else
      v35 = (__int64 **)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
    v141 = v34;
    v142 = *v35;
    v151 = sub_1643030(**v35);
    v145 = sub_1643030(*v140);
    v147 = *a2;
    v36 = sub_1643030((__int64)*a2);
    v37 = v36;
    if ( v145 >= v36 )
      v37 = v145;
    v38 = v151 - v37;
    v39 = *(_QWORD **)(v141 + 24);
    if ( *(_DWORD *)(v141 + 32) > 0x40u )
      v39 = (_QWORD *)*v39;
    if ( (unsigned int)v39 <= v38 )
    {
      if ( v145 == v36 )
      {
        v86 = v145 - 1;
        v175 = 257;
        if ( v145 - 1 > (unsigned int)v39 )
          v86 = (unsigned int)v39;
        v87 = sub_15A0680((__int64)v147, v86, 0);
        return sub_15FB440(25, v140, v87, (__int64)v174, 0);
      }
      v40 = v142[1];
      if ( v40 && !*(_QWORD *)(v40 + 8) )
      {
        v41 = v145 - 1;
        v42 = *(_QWORD *)(a1 + 8);
        v175 = 257;
        if ( v145 - 1 > (unsigned int)v39 )
          v41 = (unsigned int)v39;
        v43 = sub_15A0680(*v140, v41, 0);
        v44 = sub_173DE00(v42, (__int64)v140, v43, v174, 0, *(double *)a3.m128_u64, a4, a5);
        sub_164B7C0((__int64)v44, v14);
        v175 = 257;
        return sub_15FE0A0(v44, (__int64)*a2, 1, (__int64)v174, 0);
      }
    }
  }
LABEL_10:
  v18 = sub_1750FB0((_QWORD *)a1, (__int64 *)a2, *(double *)a3.m128_u64, a4, a5);
  if ( v18 )
    return v18;
  v61 = (__int64)*(a2 - 3);
  v62 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v61 + 16) == 85 )
  {
    v81 = *(_QWORD *)(v61 + 8);
    if ( v81 )
    {
      if ( !*(_QWORD *)(v81 + 8) && *(_BYTE *)(*(_QWORD *)(v61 - 48) + 16LL) == 9 )
      {
        v148 = *(_QWORD *)(a1 + 8);
        v153 = *(a2 - 3);
        if ( sub_15A1020(*(_BYTE **)(v61 - 24), (__int64)a2, v19, v20) )
        {
          if ( *v153 == *(_QWORD *)*(v153 - 9) )
          {
            v111 = sub_1599EF0(*a2);
            v112 = *a2;
            v175 = 257;
            v143 = v111;
            v149 = sub_1708970(v148, 36, *(v153 - 9), v112, v174);
            v113 = (_QWORD *)*(v153 - 3);
            v175 = 257;
            v156 = v113;
            v114 = sub_1648A60(56, 3u);
            if ( v114 )
            {
              v164 = v114;
              sub_15FA660((__int64)v114, v149, v143, v156, (__int64)v174, 0);
              return (__int64)v164;
            }
          }
        }
        v62 = *(_QWORD *)(a1 + 8);
      }
    }
  }
  v18 = (__int64)sub_174B990(a2, v62);
  if ( v18 )
    return v18;
  v63 = *(_QWORD *)(v14 + 8);
  if ( v63
    && !*(_QWORD *)(v63 + 8)
    && *(_BYTE *)(v159 + 8) == 11
    && (unsigned __int8)sub_1705440(a1, v159, (__int64)v15) )
  {
    v115 = *(_BYTE *)(v14 + 16);
    if ( v115 == 47 )
    {
      v117 = *(_QWORD *)(v14 - 48);
      if ( !v117 )
        goto LABEL_68;
      v118 = *(_QWORD *)(v14 - 24);
      if ( *(_BYTE *)(v118 + 16) != 13 )
        goto LABEL_68;
    }
    else
    {
      if ( v115 != 5 )
        goto LABEL_68;
      if ( *(_WORD *)(v14 + 18) != 23 )
        goto LABEL_68;
      v116 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
      v117 = *(_QWORD *)(v14 - 24 * v116);
      if ( !v117 )
        goto LABEL_68;
      v118 = *(_QWORD *)(v14 + 24 * (1 - v116));
      if ( *(_BYTE *)(v118 + 16) != 13 )
        goto LABEL_68;
    }
    v119 = *(unsigned __int8 *)(v117 + 16);
    if ( (unsigned __int8)v119 <= 0x17u )
    {
      if ( (_BYTE)v119 == 5 && (unsigned int)*(unsigned __int16 *)(v117 + 18) - 24 <= 1 )
        goto LABEL_68;
    }
    else if ( (unsigned int)(v119 - 48) <= 1 )
    {
      v139 = (*(_BYTE *)(v117 + 23) & 0x40) != 0
           ? *(_QWORD *)(v117 - 8)
           : v117 - 24LL * (*(_DWORD *)(v117 + 20) & 0xFFFFFFF);
      if ( *(_BYTE *)(*(_QWORD *)(v139 + 24) + 16LL) <= 0x10u )
        goto LABEL_68;
    }
    v165 = v118;
    v150 = v117;
    v120 = sub_16431D0((__int64)v15);
    v121 = v165;
    v122 = v150;
    v157 = v120;
    v123 = *(_DWORD *)(v165 + 32);
    v166 = v120;
    v124 = (__int64 *)(v121 + 24);
    v144 = v123;
    if ( v123 > 0x40 )
    {
      v146 = v121;
      v138 = sub_16A57B0(v121 + 24);
      v122 = v150;
      if ( v144 - v138 > 0x40 )
        goto LABEL_68;
      v125 = **(_QWORD **)(v146 + 24);
    }
    else
    {
      v125 = *(_QWORD *)(v121 + 24);
    }
    if ( v166 > v125 )
    {
      v126 = *(_QWORD *)(a1 + 8);
      v167 = v122;
      v173[0] = (__int64)sub_1649960(v122);
      v173[1] = v127;
      v175 = 773;
      v174[0] = (__int64)v173;
      v174[1] = (__int64)".tr";
      v128 = (__int64 *)sub_1708970(v126, 36, v167, v15, v174);
      v175 = 257;
      sub_16A5A50((__int64)v173, v124, v157);
      v129 = sub_15A1070((__int64)v15, (__int64)v173);
      v12 = sub_15FB440(23, v128, v129, (__int64)v174, 0);
      sub_135E100(v173);
      return v12;
    }
  }
LABEL_68:
  v64 = (__int64)*(a2 - 3);
  v65 = *(_QWORD *)(v64 + 8);
  if ( !v65 )
    return v12;
  if ( *(_QWORD *)(v65 + 8) )
    return v12;
  v66 = *a2;
  if ( *((_BYTE *)*a2 + 8) != 11 )
    return v12;
  v67 = *(_BYTE *)(v64 + 16);
  if ( v67 <= 0x17u )
  {
    if ( v67 != 5 )
      return 0;
    if ( *(_WORD *)(v64 + 18) != 47 )
    {
LABEL_160:
      if ( *(_WORD *)(v64 + 18) != 24 )
        return 0;
      v133 = *(_DWORD *)(v64 + 20) & 0xFFFFFFF;
      v134 = *(_QWORD *)(v64 - 24 * v133);
      v135 = *(unsigned __int8 *)(v134 + 16);
      if ( (unsigned __int8)v135 > 0x17u )
      {
        v136 = v135 - 24;
      }
      else
      {
        if ( (_BYTE)v135 != 5 )
          return 0;
        v136 = *(unsigned __int16 *)(v134 + 18);
      }
      if ( v136 != 47 )
        return 0;
      v137 = (*(_BYTE *)(v134 + 23) & 0x40) != 0
           ? *(__int64 ***)(v134 - 8)
           : (__int64 **)(v134 - 24LL * (*(_DWORD *)(v134 + 20) & 0xFFFFFFF));
      v92 = *v137;
      if ( !*v137 )
        return 0;
      v93 = *(_QWORD *)(v64 + 24 * (1 - v133));
      if ( *(_BYTE *)(v93 + 16) != 13 )
        return 0;
      goto LABEL_126;
    }
LABEL_153:
    if ( (*(_BYTE *)(v64 + 23) & 0x40) != 0 )
      v130 = *(__int64 ***)(v64 - 8);
    else
      v130 = (__int64 **)(v64 - 24LL * (*(_DWORD *)(v64 + 20) & 0xFFFFFFF));
    v131 = *v130;
    if ( *v130 )
    {
      if ( *(_BYTE *)(*v131 + 8) != 16 )
        return 0;
      v158 = *v130;
      v168 = *v131;
      v94 = sub_1643030(*v131);
      v132 = sub_1643030((__int64)v66);
      v96 = v168;
      v98 = (unsigned __int8 *)v158;
      v97 = v132;
      v100 = v94 % v132;
      if ( v94 % v132 )
        return v12;
LABEL_131:
      v101 = v94 / v97;
      v102 = v94 / v97;
      if ( v66 != *(__int64 ***)(v96 + 24) )
      {
        v155 = v97;
        v162 = (__int64)v98;
        v103 = (__int64 **)sub_16463B0((__int64 *)v66, v101);
        v104 = *(_QWORD *)(a1 + 8);
        v174[0] = (__int64)"bc";
        v175 = 259;
        v105 = sub_1708970(v104, 47, v162, v103, v174);
        v97 = v155;
        v98 = v105;
      }
      v106 = v100 / v97;
      if ( **(_BYTE **)(a1 + 2664) )
        v106 = v102 - 1 - v106;
      v107 = *(_QWORD *)(a1 + 8);
      v175 = 257;
      v163 = v98;
      v108 = sub_1643350(*(_QWORD **)(v107 + 24));
      v109 = sub_159C470(v108, v106, 0);
      v110 = sub_1648A60(56, 2u);
      v12 = (__int64)v110;
      if ( v110 )
        sub_15FA320((__int64)v110, v163, v109, (__int64)v174, 0);
      return v12;
    }
    if ( v67 != 5 )
      return 0;
    goto LABEL_160;
  }
  if ( v67 == 71 )
    goto LABEL_153;
  if ( v67 != 48 )
    return 0;
  v88 = *(_QWORD *)(v64 - 48);
  v89 = *(unsigned __int8 *)(v88 + 16);
  if ( (unsigned __int8)v89 > 0x17u )
  {
    v90 = v89 - 24;
  }
  else
  {
    if ( (_BYTE)v89 != 5 )
      return 0;
    v90 = *(unsigned __int16 *)(v88 + 18);
  }
  if ( v90 != 47 )
    return 0;
  v91 = (*(_BYTE *)(v88 + 23) & 0x40) != 0
      ? *(__int64 ***)(v88 - 8)
      : (__int64 **)(v88 - 24LL * (*(_DWORD *)(v88 + 20) & 0xFFFFFFF));
  v92 = *v91;
  if ( !*v91 )
    return 0;
  v93 = *(_QWORD *)(v64 - 24);
  if ( *(_BYTE *)(v93 + 16) != 13 )
    return 0;
LABEL_126:
  if ( *(_BYTE *)(*v92 + 8) != 16 )
    return 0;
  v154 = v92;
  v161 = *v92;
  v94 = sub_1643030(*v92);
  v95 = sub_1643030((__int64)v66);
  v96 = v161;
  v97 = v95;
  v98 = (unsigned __int8 *)v154;
  v99 = *(_QWORD **)(v93 + 24);
  if ( *(_DWORD *)(v93 + 32) > 0x40u )
    v99 = (_QWORD *)*v99;
  v100 = (unsigned int)v99;
  if ( !(v94 % v97) && !((unsigned int)v99 % v97) )
    goto LABEL_131;
  return v12;
}
