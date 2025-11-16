// Function: sub_29FED30
// Address: 0x29fed30
//
__int64 __fastcall sub_29FED30(__int64 a1, __int64 *a2, __int64 a3, char a4, _QWORD *a5)
{
  __int64 v9; // r14
  unsigned __int64 v10; // rax
  int v11; // eax
  unsigned int v12; // r14d
  __int64 v13; // r15
  __int64 v14; // rbx
  int v15; // r12d
  __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // r12
  __int64 v20; // r14
  _QWORD *v21; // rbx
  __int64 v22; // r15
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rcx
  int v30; // r10d
  __int64 *v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rax
  __int64 v34; // r8
  int v35; // r10d
  unsigned int v36; // edi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 *v40; // rax
  __int64 *v41; // rax
  __int64 v42; // r11
  int v43; // r10d
  __int64 v44; // rsi
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  __int64 v47; // rax
  _QWORD *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  unsigned int v56; // edx
  int v57; // eax
  bool v58; // al
  int v59; // r10d
  __int64 v60; // rax
  char v61; // al
  bool v62; // al
  __int64 v63; // rax
  _QWORD *v64; // rax
  unsigned int v65; // eax
  __int64 v66; // r8
  const char *v67; // rdx
  int v68; // r10d
  _QWORD *v69; // rax
  _QWORD *v70; // rax
  unsigned int v71; // r10d
  __int64 v72; // r8
  char v73; // al
  const char *v74; // rdi
  __int64 v75; // rax
  _QWORD *v76; // rax
  __int64 v77; // r12
  unsigned __int64 v78; // rax
  unsigned __int64 v79; // rdx
  unsigned __int64 v80; // rax
  __int64 v81; // r15
  __int64 v82; // rax
  unsigned __int8 *v83; // rax
  unsigned __int8 *v84; // r15
  __int64 v85; // rax
  unsigned int v86; // eax
  __int64 v87; // rax
  _QWORD *v88; // rax
  unsigned int v89; // eax
  const char *v90; // rdx
  bool v91; // al
  __int64 v92; // rax
  _QWORD *v93; // rax
  _QWORD *v94; // rax
  __int64 v95; // rax
  _QWORD *v96; // rax
  _QWORD *v97; // rax
  __int64 *v98; // rax
  int v99; // eax
  bool v100; // al
  bool v101; // al
  __int64 v102; // rax
  __int64 v103; // rax
  unsigned int v104; // eax
  __int64 v105; // r8
  _QWORD *v106; // rax
  __int64 v107; // rax
  _QWORD *v108; // rax
  _QWORD *v109; // rax
  char v110; // al
  unsigned __int64 v111; // rdi
  __int64 v112; // rax
  _QWORD *v113; // rax
  int v114; // edx
  bool v115; // cl
  __int64 v116; // rax
  _QWORD *v117; // rax
  _QWORD *v118; // rax
  __int64 v119; // rax
  _QWORD *v120; // rax
  _QWORD *v121; // rax
  int v122; // [rsp+8h] [rbp-488h]
  __int64 v123; // [rsp+8h] [rbp-488h]
  bool v124; // [rsp+8h] [rbp-488h]
  __int64 v125; // [rsp+8h] [rbp-488h]
  __int64 v126; // [rsp+10h] [rbp-480h]
  _QWORD *v127; // [rsp+18h] [rbp-478h]
  unsigned int v128; // [rsp+18h] [rbp-478h]
  unsigned int v129; // [rsp+18h] [rbp-478h]
  __int64 v130; // [rsp+20h] [rbp-470h]
  int v131; // [rsp+28h] [rbp-468h]
  int v132; // [rsp+28h] [rbp-468h]
  __int64 v133; // [rsp+28h] [rbp-468h]
  int v134; // [rsp+28h] [rbp-468h]
  __int64 v135; // [rsp+28h] [rbp-468h]
  __int64 v136; // [rsp+30h] [rbp-460h]
  unsigned __int64 v137; // [rsp+38h] [rbp-458h]
  __int64 v138; // [rsp+40h] [rbp-450h]
  unsigned __int64 v139; // [rsp+48h] [rbp-448h]
  _QWORD *v140; // [rsp+50h] [rbp-440h]
  int v141; // [rsp+50h] [rbp-440h]
  _QWORD *v142; // [rsp+58h] [rbp-438h]
  __int64 v143; // [rsp+60h] [rbp-430h]
  __int64 v144; // [rsp+68h] [rbp-428h]
  bool v145; // [rsp+68h] [rbp-428h]
  unsigned int v146; // [rsp+70h] [rbp-420h]
  int v147; // [rsp+70h] [rbp-420h]
  __int64 v148; // [rsp+70h] [rbp-420h]
  _QWORD *v149; // [rsp+70h] [rbp-420h]
  __int64 v150; // [rsp+70h] [rbp-420h]
  _QWORD *v151; // [rsp+70h] [rbp-420h]
  unsigned int v152; // [rsp+70h] [rbp-420h]
  __int64 v153; // [rsp+78h] [rbp-418h]
  __int64 v154; // [rsp+78h] [rbp-418h]
  __int64 v155; // [rsp+78h] [rbp-418h]
  _QWORD *v156; // [rsp+78h] [rbp-418h]
  _QWORD *v157; // [rsp+78h] [rbp-418h]
  __int64 v158; // [rsp+80h] [rbp-410h]
  _QWORD *v159; // [rsp+80h] [rbp-410h]
  _QWORD *v160; // [rsp+80h] [rbp-410h]
  char v161; // [rsp+8Fh] [rbp-401h]
  __int64 v162; // [rsp+90h] [rbp-400h]
  int v163; // [rsp+90h] [rbp-400h]
  char v164; // [rsp+90h] [rbp-400h]
  int v165; // [rsp+90h] [rbp-400h]
  __int64 v166; // [rsp+98h] [rbp-3F8h]
  __int64 *v167; // [rsp+98h] [rbp-3F8h]
  int v168; // [rsp+98h] [rbp-3F8h]
  int v169; // [rsp+98h] [rbp-3F8h]
  __int64 v170; // [rsp+98h] [rbp-3F8h]
  char v171; // [rsp+98h] [rbp-3F8h]
  __int64 v172; // [rsp+98h] [rbp-3F8h]
  char v173; // [rsp+98h] [rbp-3F8h]
  __int64 v174; // [rsp+A0h] [rbp-3F0h]
  _QWORD *v175; // [rsp+A0h] [rbp-3F0h]
  __int64 v176; // [rsp+A0h] [rbp-3F0h]
  __int64 v177; // [rsp+A8h] [rbp-3E8h]
  __int64 v178; // [rsp+A8h] [rbp-3E8h]
  int v180; // [rsp+B0h] [rbp-3E0h]
  _QWORD *v181; // [rsp+B0h] [rbp-3E0h]
  int v182; // [rsp+B0h] [rbp-3E0h]
  unsigned __int64 v184; // [rsp+C0h] [rbp-3D0h] BYREF
  unsigned int v185; // [rsp+C8h] [rbp-3C8h]
  char v186; // [rsp+E0h] [rbp-3B0h]
  char v187; // [rsp+E1h] [rbp-3AFh]
  __int64 *v188; // [rsp+F0h] [rbp-3A0h] BYREF
  __int64 v189; // [rsp+F8h] [rbp-398h]
  __int64 v190; // [rsp+100h] [rbp-390h] BYREF
  _QWORD *v191; // [rsp+108h] [rbp-388h]

  if ( !(unsigned __int8)sub_D4B3D0(a3) )
  {
    *a5 = "loop not in LoopSimplify form";
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
  v9 = sub_D47930(a3);
  v177 = v9 + 48;
  v10 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 + 48 == v10 )
    goto LABEL_197;
  if ( !v10 )
    goto LABEL_196;
  if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 > 0xA )
LABEL_197:
    BUG();
  if ( *(_QWORD *)(v10 + 24) || (*(_BYTE *)(v10 - 17) & 0x20) != 0 )
  {
    if ( sub_B91F50(v10 - 24, "loop_constrainer.loop.clone", 0x1Bu) )
    {
      *a5 = "loop has already been cloned";
      *(_BYTE *)(a1 + 96) = 0;
      return a1;
    }
    v24 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    v10 = v24;
    if ( v24 == v177 )
    {
LABEL_21:
      *a5 = "no loop latch";
      *(_BYTE *)(a1 + 96) = 0;
      return a1;
    }
    if ( !v24 )
      BUG();
  }
  if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 > 0xA )
    goto LABEL_21;
  v174 = v10 - 24;
  v11 = sub_B46E30(v10 - 24);
  if ( !v11 )
    goto LABEL_21;
  v166 = v9;
  v162 = a3 + 56;
  v12 = 0;
  v13 = v174;
  v175 = a5;
  v14 = a3;
  v15 = v11;
  while ( 1 )
  {
    v16 = sub_B46EC0(v13, v12);
    if ( *(_BYTE *)(v14 + 84) )
    {
      v17 = *(_QWORD **)(v14 + 64);
      v18 = &v17[*(unsigned int *)(v14 + 76)];
      if ( v17 == v18 )
        break;
      while ( v16 != *v17 )
      {
        if ( v18 == ++v17 )
          goto LABEL_23;
      }
      goto LABEL_19;
    }
    if ( !sub_C8CA60(v162, v16) )
      break;
LABEL_19:
    if ( v15 == ++v12 )
    {
      a5 = v175;
      goto LABEL_21;
    }
  }
LABEL_23:
  v19 = v14;
  v20 = v166;
  v21 = v175;
  v176 = **(_QWORD **)(v19 + 32);
  v22 = sub_D4B130(v19);
  if ( !v22 )
  {
    *v21 = "no preheader";
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
  v23 = *(_QWORD *)(v166 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v139 = v23;
  if ( v177 == v23 || !v23 || (v137 = v23 - 24, (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 > 0xA) )
LABEL_196:
    BUG();
  if ( *(_BYTE *)(v23 - 24) != 31 || (*(_DWORD *)(v23 - 20) & 0x7FFFFFF) == 1 )
  {
    *v21 = "latch terminator not conditional branch";
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
  v153 = *(_QWORD *)(v23 - 120);
  if ( *(_BYTE *)v153 != 82 || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v153 - 64) + 8LL) + 8LL) != 12 )
  {
    *v21 = "latch terminator branch not conditional on integral icmp";
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
  v158 = *(_QWORD *)(v23 - 56);
  v25 = sub_D47930(v19);
  v136 = sub_DBA6E0((__int64)a2, v19, v25, 2);
  if ( sub_D96A50(v136) )
    v136 = sub_DCF3A0(a2, (char *)v19, 2);
  v161 = sub_D96A50(v136);
  if ( v161 )
  {
    *v21 = "could not compute latch count";
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
  v138 = *(_QWORD *)(v153 - 64);
  v146 = *(_WORD *)(v153 + 2) & 0x3F;
  v167 = sub_DD8400((__int64)a2, v138);
  v126 = *(_QWORD *)(v138 + 8);
  v142 = *(_QWORD **)(v153 - 32);
  v26 = sub_DD8400((__int64)a2, (__int64)v142);
  v29 = (__int64)v167;
  v30 = v146;
  v178 = (__int64)v26;
  if ( *((_WORD *)v167 + 12) != 8 )
  {
    if ( *((_WORD *)v26 + 12) != 8 )
    {
      *v21 = "no add recurrences in the icmp";
      *(_BYTE *)(a1 + 96) = 0;
      return a1;
    }
    v30 = sub_B52F50(v146);
    v31 = v167;
    v167 = (__int64 *)v178;
    v29 = (__int64)v142;
    v178 = (__int64)v31;
    v32 = (_QWORD *)v138;
    v138 = (__int64)v142;
    v142 = v32;
  }
  if ( v19 != v167[6] )
  {
    *v21 = "LHS in cmp is not an AddRec for this loop";
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
  if ( v167[5] != 2
    || (v147 = v30, v33 = sub_D33D80(v167, (__int64)a2, v27, v29, v28), v35 = v147, *(_WORD *)(v33 + 24)) )
  {
    *v21 = "LHS in icmp not induction variable";
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
  v143 = *(_QWORD *)(v33 + 32);
  if ( (*(_WORD *)(v153 + 2) & 0x3Fu) - 32 <= 1 && (*((_BYTE *)v167 + 28) & 4) == 0 )
  {
    v47 = sub_D95540(*(_QWORD *)v167[4]);
    v155 = sub_BCCE00(*(_QWORD **)v47, 2 * (*(_DWORD *)(v47 + 8) >> 8));
    v48 = sub_DC5000((__int64)a2, (__int64)v167, v155, 0);
    v35 = v147;
    if ( *((_WORD *)v48 + 12) != 8 )
      goto LABEL_63;
    v144 = (__int64)v48;
    v141 = v147;
    v149 = sub_DC5000((__int64)a2, *(_QWORD *)v167[4], v155, 0);
    v52 = sub_D33D80(v167, (__int64)a2, v49, v50, v51);
    v53 = sub_DC5000((__int64)a2, v52, v155, 0);
    v34 = v144;
    v156 = v53;
    v35 = v141;
    if ( v149 != **(_QWORD ***)(v144 + 32)
      || (v60 = sub_D33D80((_QWORD *)v144, (__int64)a2, v54, v55, v144), v35 = v141, v156 != (_QWORD *)v60) )
    {
LABEL_63:
      if ( (*((_BYTE *)v167 + 28) & 4) == 0 )
      {
        *v21 = "LHS in icmp needs nsw for equality predicates";
        *(_BYTE *)(a1 + 96) = 0;
        return a1;
      }
    }
  }
  v36 = *(_DWORD *)(v143 + 32);
  v37 = *(_QWORD *)(v143 + 24);
  v38 = v36 - 1;
  if ( v36 > 0x40 )
    v37 = *(_QWORD *)(v37 + 8LL * ((unsigned int)v38 >> 6));
  v131 = v35;
  v130 = v37 & (1LL << ((unsigned __int8)v36 - 1));
  v154 = *(_QWORD *)v167[4];
  v39 = sub_D33D80(v167, (__int64)a2, v37, v38, v34);
  v40 = sub_DCAF50(a2, v39, 0);
  v140 = sub_DC7ED0(a2, v154, (__int64)v40, 0, 0);
  v41 = sub_DD8400((__int64)a2, v143);
  v42 = v143 + 24;
  v43 = v131;
  v148 = (__int64)v41;
  if ( *(_BYTE *)v142 <= 0x1Cu )
  {
LABEL_140:
    v157 = 0;
  }
  else
  {
    v44 = v142[5];
    if ( *(_BYTE *)(v19 + 84) )
    {
      v45 = *(_QWORD **)(v19 + 64);
      v46 = &v45[*(unsigned int *)(v19 + 76)];
      while ( v46 != v45 )
      {
        if ( v44 == *v45 )
        {
          v157 = (_QWORD *)v178;
          goto LABEL_66;
        }
        ++v45;
      }
      goto LABEL_140;
    }
    v98 = sub_C8CA60(v162, v44);
    v42 = v143 + 24;
    v43 = v131;
    v157 = (_QWORD *)v178;
    if ( !v98 )
      goto LABEL_140;
  }
LABEL_66:
  v145 = v158 == v176;
  v56 = *(_DWORD *)(v143 + 32);
  if ( v130 )
  {
    if ( v56 )
    {
      if ( v56 <= 0x40 )
      {
        v100 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v56) == *(_QWORD *)(v143 + 24);
      }
      else
      {
        v134 = v43;
        v165 = *(_DWORD *)(v143 + 32);
        v99 = sub_C445E0(v42);
        v43 = v134;
        v100 = v165 == v99;
      }
      if ( !v100 )
        goto LABEL_145;
    }
    v164 = v145 && v43 == 33;
    if ( v164 )
    {
      v43 = 38;
      goto LABEL_148;
    }
    if ( v43 != 32 || v158 == v176 )
    {
LABEL_145:
      v101 = v43 == 40;
      v164 = v43 == 38;
      if ( (v43 & 0xFFFFFFFB) == 34 && v145 )
      {
        v164 |= v101;
        if ( !a4 && !v164 )
          goto LABEL_75;
        goto LABEL_148;
      }
      v114 = v43 - 36;
      v161 = 0;
      v115 = ((v43 - 36) & 0xFFFFFFFB) == 0;
    }
    else
    {
      if ( (*((_BYTE *)v167 + 28) & 2) != 0 && (v161 = sub_F70730(v178, v19, a2, 0)) != 0 )
      {
        v119 = sub_D95540(v178);
        v120 = sub_DA2C50((__int64)a2, v119, 1, 0);
        v121 = sub_DC7ED0(a2, v178, (__int64)v120, 0, 0);
        v43 = 36;
        v178 = (__int64)v121;
      }
      else
      {
        v161 = sub_F70730(v178, v19, a2, 1);
        if ( v161 )
        {
          v116 = sub_D95540(v178);
          v117 = sub_DA2C50((__int64)a2, v116, 1, 0);
          v118 = sub_DC7ED0(a2, v178, (__int64)v117, 0, 0);
          v43 = 40;
          v178 = (__int64)v118;
        }
        else
        {
          v43 = 32;
        }
      }
      v114 = v43 - 36;
      v101 = v43 == 40;
      v115 = ((v43 - 36) & 0xFFFFFFFB) == 0;
    }
    if ( v158 == v176 || !v115 )
    {
      *v21 = "expected icmp sgt semantically, found something else";
      *(_BYTE *)(a1 + 96) = 0;
      return a1;
    }
    v164 |= v101;
    if ( !v164 && !a4 )
      goto LABEL_75;
    if ( (v114 & 0xFFFFFFFB) != 0 && (v43 & 0xFFFFFFFB) != 0x22 )
      goto LABEL_164;
LABEL_148:
    v182 = v43;
    if ( !sub_DAEB70((__int64)a2, v178, v19) )
      goto LABEL_164;
    v124 = sub_B532B0(v182);
    v129 = !v124 ? 34 : 38;
    v135 = sub_DE4F70(a2, (__int64)v140, v19);
    v102 = sub_DE4F70(a2, v178, v19);
    v172 = v102;
    if ( v158 == v176 )
    {
      if ( (unsigned __int8)sub_DDD5B0(a2, v19, v129, v135, v102) )
        goto LABEL_103;
      goto LABEL_164;
    }
    v103 = sub_D95540(v148);
    v191 = sub_DA2C50((__int64)a2, v103, 1, 0);
    v190 = v148;
    v188 = &v190;
    v189 = 0x200000002LL;
    v160 = sub_DC7EB0(a2, (__int64)&v188, 0, 0);
    if ( v188 != &v190 )
      _libc_free((unsigned __int64)v188);
    v104 = *(_DWORD *)(sub_D95540(v178) + 8) >> 8;
    LODWORD(v189) = v104;
    if ( v124 )
    {
      v105 = 1LL << ((unsigned __int8)v104 - 1);
      if ( v104 <= 0x40 )
      {
        v188 = 0;
LABEL_155:
        v188 = (__int64 *)(v105 | (unsigned __int64)v188);
        goto LABEL_156;
      }
      v125 = 1LL << ((unsigned __int8)v104 - 1);
      v152 = v104 - 1;
      sub_C43690((__int64)&v188, 0, 0);
      v105 = v125;
      if ( (unsigned int)v189 <= 0x40 )
        goto LABEL_155;
      v188[v152 >> 6] |= v125;
    }
    else if ( v104 > 0x40 )
    {
      sub_C43690((__int64)&v188, 0, 0);
    }
    else
    {
      v188 = 0;
    }
LABEL_156:
    v106 = sub_DA26C0(a2, (__int64)&v188);
    v151 = sub_DCC810(a2, (__int64)v106, (__int64)v160, 0, 0);
    v107 = sub_D95540(v172);
    v108 = sub_DA2C50((__int64)a2, v107, 1, 0);
    v109 = sub_DCC810(a2, v172, (__int64)v108, 0, 0);
    v110 = sub_DDD5B0(a2, v19, v129, v135, (__int64)v109);
    if ( v110 )
    {
      v110 = sub_DDD5B0(a2, v19, v129, v172, (__int64)v151);
      if ( (unsigned int)v189 <= 0x40 || (v111 = (unsigned __int64)v188) == 0 )
      {
LABEL_160:
        if ( v110 )
        {
          if ( !v161 )
          {
            v112 = sub_D95540(v178);
            v113 = sub_DA2C50((__int64)a2, v112, 1, 0);
            v157 = sub_DCC810(a2, v178, (__int64)v113, 0, 0);
          }
          goto LABEL_103;
        }
LABEL_164:
        *v21 = "Unsafe bounds";
        *(_BYTE *)(a1 + 96) = 0;
        return a1;
      }
    }
    else
    {
      if ( (unsigned int)v189 <= 0x40 )
        goto LABEL_164;
      v111 = (unsigned __int64)v188;
      if ( !v188 )
        goto LABEL_164;
    }
    v173 = v110;
    j_j___libc_free_0_0(v111);
    v110 = v173;
    goto LABEL_160;
  }
  if ( v56 <= 0x40 )
  {
    if ( *(_QWORD *)(v143 + 24) == 1 )
      goto LABEL_69;
    goto LABEL_136;
  }
  v132 = v43;
  v163 = *(_DWORD *)(v143 + 32);
  v57 = sub_C444A0(v42);
  v43 = v132;
  if ( v57 != v163 - 1 )
    goto LABEL_136;
LABEL_69:
  if ( v43 == 33 && v158 == v176 )
  {
    if ( !(unsigned __int8)sub_F705A0((__int64)v140, v19, a2) || (v61 = sub_F705A0(v178, v19, a2), v43 = 36, !v61) )
      v43 = 40;
    goto LABEL_73;
  }
  if ( v43 != 32 || v158 == v176 )
  {
LABEL_136:
    if ( ((v43 - 36) & 0xFFFFFFFB) != 0 || !v145 )
    {
      v161 = 0;
      v91 = (v43 & 0xFFFFFFFB) == 34;
      goto LABEL_131;
    }
    v169 = v43;
    v62 = sub_B532B0(v43);
    v59 = v169;
    v164 = v62;
    if ( !a4 && !v62 )
    {
LABEL_75:
      *v21 = "unsigned latch conditions are explicitly prohibited";
      *(_BYTE *)(a1 + 96) = 0;
      return a1;
    }
    goto LABEL_85;
  }
  if ( (*((_BYTE *)v167 + 28) & 2) != 0 && (v161 = sub_F70610(v178, v19, a2, 0)) != 0 )
  {
    v95 = sub_D95540(v178);
    v96 = sub_DA2C50((__int64)a2, v95, 1, 0);
    v97 = sub_DCC810(a2, v178, (__int64)v96, 0, 0);
    v43 = 34;
    v178 = (__int64)v97;
  }
  else
  {
    v161 = sub_F70610(v178, v19, a2, 1);
    if ( v161 )
    {
      v92 = sub_D95540(v178);
      v93 = sub_DA2C50((__int64)a2, v92, 1, 0);
      v94 = sub_DCC810(a2, v178, (__int64)v93, 0, 0);
      v43 = 38;
      v178 = (__int64)v94;
    }
    else
    {
      v43 = 32;
    }
  }
  v91 = (v43 & 0xFFFFFFFB) == 34;
LABEL_131:
  if ( v158 == v176 || !v91 )
  {
    *v21 = "expected icmp slt semantically, found something else";
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
LABEL_73:
  v168 = v43;
  v58 = sub_B532B0(v43);
  v59 = v168;
  v164 = v58;
  if ( !a4 && !v58 )
    goto LABEL_75;
  if ( ((v168 - 36) & 0xFFFFFFFB) != 0 && (v168 & 0xFFFFFFFB) != 0x22 )
  {
LABEL_80:
    *v21 = "Unsafe loop bounds";
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
LABEL_85:
  v180 = v59;
  if ( !sub_DAEB70((__int64)a2, v178, v19) )
    goto LABEL_80;
  if ( !sub_B532B0(v180) )
  {
    v133 = sub_DE4F70(a2, (__int64)v140, v19);
    v170 = sub_DE4F70(a2, v178, v19);
    if ( v158 != v176 )
    {
      v87 = sub_D95540(v148);
      v88 = sub_DA2C50((__int64)a2, v87, 1, 0);
      v181 = sub_DCC810(a2, v148, (__int64)v88, 0, 0);
      v89 = *(_DWORD *)(sub_D95540(v178) + 8) >> 8;
      v185 = v89;
      if ( v89 > 0x40 )
      {
        sub_C43690((__int64)&v184, -1, 1);
        v68 = 36;
      }
      else
      {
        v68 = 36;
        v90 = (const char *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v89);
        if ( !v89 )
          v90 = 0;
        v184 = (unsigned __int64)v90;
      }
      goto LABEL_93;
    }
    v86 = 36;
    goto LABEL_117;
  }
  v133 = sub_DE4F70(a2, (__int64)v140, v19);
  v170 = sub_DE4F70(a2, v178, v19);
  if ( v158 == v176 )
  {
    v86 = 40;
LABEL_117:
    if ( !(unsigned __int8)sub_DDD5B0(a2, v19, v86, v133, v170) )
      goto LABEL_80;
    goto LABEL_103;
  }
  v63 = sub_D95540(v148);
  v64 = sub_DA2C50((__int64)a2, v63, 1, 0);
  v181 = sub_DCC810(a2, v148, (__int64)v64, 0, 0);
  v65 = *(_DWORD *)(sub_D95540(v178) + 8) >> 8;
  v185 = v65;
  v66 = ~(1LL << ((unsigned __int8)v65 - 1));
  if ( v65 > 0x40 )
  {
    v123 = ~(1LL << ((unsigned __int8)v65 - 1));
    v128 = v65 - 1;
    sub_C43690((__int64)&v184, -1, 1);
    v66 = v123;
    if ( v185 <= 0x40 )
      goto LABEL_92;
    v68 = 40;
    *(_QWORD *)(v184 + 8LL * (v128 >> 6)) &= v123;
  }
  else
  {
    v67 = (const char *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v65);
    if ( !v65 )
      v67 = 0;
    v184 = (unsigned __int64)v67;
LABEL_92:
    v184 &= v66;
    v68 = 40;
  }
LABEL_93:
  v122 = v68;
  v69 = sub_DA26C0(a2, (__int64)&v184);
  v159 = sub_DCC810(a2, (__int64)v69, (__int64)v181, 0, 0);
  v188 = &v190;
  v190 = v170;
  v191 = (_QWORD *)v148;
  v189 = 0x200000002LL;
  v70 = sub_DC7EB0(a2, (__int64)&v188, 0, 0);
  v71 = v122;
  v72 = (__int64)v70;
  if ( v188 != &v190 )
  {
    v127 = v70;
    _libc_free((unsigned __int64)v188);
    v72 = (__int64)v127;
    v71 = v122;
  }
  v150 = v71;
  v73 = sub_DDD5B0(a2, v19, v71, v133, v72);
  if ( v73 )
  {
    v73 = sub_DDD5B0(a2, v19, v150 & 0xFFFFFF00FFFFFFFFLL, v170, (__int64)v159);
    if ( v185 > 0x40 )
    {
      v74 = (const char *)v184;
      if ( v184 )
        goto LABEL_98;
    }
  }
  else
  {
    if ( v185 <= 0x40 )
      goto LABEL_80;
    v74 = (const char *)v184;
    if ( !v184 )
      goto LABEL_80;
LABEL_98:
    v171 = v73;
    j_j___libc_free_0_0((unsigned __int64)v74);
    v73 = v171;
  }
  if ( !v73 )
    goto LABEL_80;
  if ( !v161 && !v145 )
  {
    v75 = sub_D95540(v178);
    v76 = sub_DA2C50((__int64)a2, v75, 1, 0);
    v157 = sub_DC7ED0(a2, v178, (__int64)v76, 0, 0);
  }
LABEL_103:
  v77 = *(_QWORD *)(v139 - 32LL * v145 - 56);
  v78 = sub_AA4E30(v22);
  sub_27C1C30((__int64)&v188, a2, v78, (__int64)"loop-constrainer", 1);
  v79 = *(_QWORD *)(v22 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v79 == v22 + 48 )
  {
    v80 = 0;
    goto LABEL_107;
  }
  if ( !v79 )
    goto LABEL_196;
  v80 = v79 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v79 - 24) - 30 >= 0xB )
    v80 = 0;
LABEL_107:
  v81 = v80 + 24;
  if ( v157 )
  {
    v82 = sub_D95540((__int64)v157);
    v142 = sub_F8DB90((__int64)&v188, (__int64)v157, v82, v81, 0);
  }
  v83 = (unsigned __int8 *)sub_F8DB90((__int64)&v188, (__int64)v140, v126, v81, 0);
  v187 = 1;
  v84 = v83;
  v186 = 3;
  v184 = (unsigned __int64)"indvar.start";
  sub_BD6B50(v83, (const char **)&v184);
  v85 = sub_D95540(v136);
  *v21 = 0;
  *(_QWORD *)a1 = "main";
  *(_QWORD *)(a1 + 16) = v20;
  *(_QWORD *)(a1 + 8) = v176;
  *(_QWORD *)(a1 + 32) = v77;
  *(_QWORD *)(a1 + 24) = v137;
  *(_DWORD *)(a1 + 40) = v145;
  *(_QWORD *)(a1 + 48) = v138;
  *(_QWORD *)(a1 + 56) = v84;
  *(_QWORD *)(a1 + 64) = v143;
  *(_QWORD *)(a1 + 88) = v85;
  *(_QWORD *)(a1 + 72) = v142;
  *(_BYTE *)(a1 + 96) = 1;
  *(_BYTE *)(a1 + 81) = v164;
  *(_BYTE *)(a1 + 80) = v130 == 0;
  sub_27C20B0((__int64)&v188);
  return a1;
}
