// Function: sub_1FC15C0
// Address: 0x1fc15c0
//
_QWORD *__fastcall sub_1FC15C0(
        _QWORD **a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r14
  unsigned __int8 *v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rdx
  const void **v16; // rcx
  char v17; // di
  unsigned __int16 v18; // dx
  int v19; // r15d
  __int64 v20; // rsi
  int v21; // eax
  unsigned __int8 v22; // r15
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned int v27; // eax
  unsigned int v28; // esi
  unsigned __int64 v29; // rdx
  unsigned int v30; // ebx
  __int16 v31; // ax
  int v32; // edx
  __int64 v33; // rcx
  _QWORD *v34; // rax
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 v37; // rbx
  _QWORD *v38; // r9
  int v39; // ecx
  char *v40; // rax
  __int64 (__fastcall *v41)(__int64, __int64); // r15
  __int64 v42; // rax
  int v43; // eax
  int v44; // r8d
  unsigned int v45; // edx
  unsigned __int8 v46; // al
  __int64 v47; // rsi
  __int64 *v48; // r13
  unsigned int v49; // r15d
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // r14
  __int64 v53; // rdx
  __int64 v54; // r15
  _QWORD *result; // rax
  __int64 *v56; // rdi
  _QWORD *v57; // r12
  _QWORD *v58; // rax
  __int64 v59; // rbx
  __int64 v60; // rax
  __int64 v61; // rsi
  __int64 *v62; // r13
  __int64 v63; // rsi
  int v64; // eax
  __int64 v65; // rdx
  int v66; // edx
  unsigned int v67; // eax
  unsigned int v68; // r14d
  unsigned __int64 v69; // r12
  unsigned __int64 v70; // rcx
  __int64 v71; // r13
  unsigned int v72; // eax
  __int64 v73; // rdx
  __int64 v74; // rax
  _QWORD *v75; // rdi
  __int64 (*v76)(); // rax
  __int64 v77; // rdx
  _QWORD *v78; // rax
  unsigned int *v79; // rdx
  __int64 v80; // rcx
  char v81; // al
  __int64 v82; // rdx
  __int64 v83; // rbx
  char v84; // r15
  bool v85; // cl
  unsigned int v86; // eax
  _QWORD *v87; // rdi
  __int64 (*v88)(); // rax
  __int64 v89; // rax
  __int64 v90; // r8
  __int16 v91; // ax
  int v92; // r15d
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // r9
  __int64 v96; // rax
  _QWORD *v97; // rdx
  __int64 v98; // r14
  __int64 v99; // rbx
  bool v100; // al
  __int64 v101; // rsi
  __int64 *v102; // r13
  __int64 v103; // rax
  char v104; // dl
  __int64 v105; // rax
  char v106; // r8
  __int64 v107; // rax
  int v108; // r15d
  __int64 v109; // r8
  char v110; // al
  __int64 v111; // rdx
  __int64 *v112; // rdx
  __int64 v113; // r12
  __int64 v114; // r15
  __int64 v115; // rdx
  __int64 v116; // rsi
  __int64 *v117; // r13
  unsigned int *v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rax
  unsigned int v121; // r8d
  int v122; // ecx
  __int64 v123; // rax
  int v124; // edx
  __int16 v125; // ax
  __int64 *v126; // r11
  unsigned __int8 *v127; // rax
  __int64 v128; // rsi
  const void **v129; // r8
  __int64 v130; // rcx
  unsigned int v131; // edx
  __int64 v132; // rdx
  __int64 v133; // rcx
  __int64 v134; // r8
  int v135; // r9d
  __int64 v137; // rax
  char v138; // dl
  __int64 v139; // rax
  __int64 v141; // rsi
  __int64 *v142; // r13
  char v143; // al
  __int64 v144; // rdx
  bool v145; // cl
  char v146; // r8
  unsigned int v147; // r15d
  unsigned int v148; // eax
  __int64 v149; // rax
  int v150; // eax
  unsigned int v151; // eax
  int v152; // eax
  bool v153; // cl
  int v154; // eax
  __int128 v155; // [rsp-10h] [rbp-100h]
  __int64 v156; // [rsp+0h] [rbp-F0h]
  _QWORD **v157; // [rsp+8h] [rbp-E8h]
  __int64 v158; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v159; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v160; // [rsp+20h] [rbp-D0h]
  char v161; // [rsp+2Bh] [rbp-C5h]
  unsigned int v162; // [rsp+2Ch] [rbp-C4h]
  __int64 v163; // [rsp+30h] [rbp-C0h]
  __int64 v164; // [rsp+38h] [rbp-B8h]
  __int64 v165; // [rsp+38h] [rbp-B8h]
  unsigned int v166; // [rsp+38h] [rbp-B8h]
  __int64 v167; // [rsp+40h] [rbp-B0h]
  __int64 v168; // [rsp+48h] [rbp-A8h]
  const void **v169; // [rsp+50h] [rbp-A0h]
  unsigned int v170; // [rsp+50h] [rbp-A0h]
  __int64 v171; // [rsp+50h] [rbp-A0h]
  __int64 v172; // [rsp+50h] [rbp-A0h]
  __int64 v173; // [rsp+58h] [rbp-98h]
  unsigned int v174; // [rsp+60h] [rbp-90h]
  int v175; // [rsp+60h] [rbp-90h]
  _QWORD *v176; // [rsp+60h] [rbp-90h]
  const void **v177; // [rsp+60h] [rbp-90h]
  bool v178; // [rsp+60h] [rbp-90h]
  char v179; // [rsp+60h] [rbp-90h]
  bool v180; // [rsp+60h] [rbp-90h]
  char v181; // [rsp+68h] [rbp-88h]
  _QWORD *v182; // [rsp+68h] [rbp-88h]
  int v183; // [rsp+68h] [rbp-88h]
  _QWORD *v184; // [rsp+68h] [rbp-88h]
  _QWORD *v185; // [rsp+68h] [rbp-88h]
  unsigned int v186; // [rsp+68h] [rbp-88h]
  bool v187; // [rsp+68h] [rbp-88h]
  __int64 *v188; // [rsp+68h] [rbp-88h]
  bool v189; // [rsp+68h] [rbp-88h]
  bool v190; // [rsp+68h] [rbp-88h]
  bool v191; // [rsp+68h] [rbp-88h]
  bool v192; // [rsp+68h] [rbp-88h]
  int v193; // [rsp+68h] [rbp-88h]
  unsigned int v194; // [rsp+70h] [rbp-80h] BYREF
  __int64 v195; // [rsp+78h] [rbp-78h]
  __int64 v196; // [rsp+80h] [rbp-70h] BYREF
  const void **v197; // [rsp+88h] [rbp-68h]
  __int64 v198; // [rsp+90h] [rbp-60h] BYREF
  __int64 v199; // [rsp+98h] [rbp-58h]
  __int64 v200; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v201; // [rsp+A8h] [rbp-48h]
  __int64 v202; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v203; // [rsp+B8h] [rbp-38h]

  v10 = a2;
  v11 = *(_QWORD *)(a2 + 32);
  v12 = *(_QWORD *)v11;
  v174 = *(_DWORD *)(v11 + 8);
  v13 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v11 + 40LL) + 16LL * v174);
  v14 = *v13;
  v195 = *((_QWORD *)v13 + 1);
  v15 = *(_QWORD *)(a2 + 40);
  LOBYTE(v194) = v14;
  v16 = *(const void ***)(v15 + 8);
  v17 = *(_BYTE *)v15;
  v18 = *(_WORD *)(v12 + 24);
  v181 = v17;
  v169 = v16;
  LOBYTE(v196) = v17;
  v197 = v16;
  if ( v18 == 48 )
    goto LABEL_53;
  v19 = v18;
  if ( v18 == 111 )
  {
    v58 = *(_QWORD **)(v12 + 32);
    v57 = (_QWORD *)*v58;
    v59 = v58[1];
    v60 = *(_QWORD *)(*v58 + 40LL) + 16LL * *((unsigned int *)v58 + 2);
    if ( *(_BYTE *)v60 != v17 || *(const void ***)(v60 + 8) != v16 && !*(_BYTE *)v60 )
    {
      v61 = *(_QWORD *)(v12 + 72);
      v62 = *a1;
      v202 = v61;
      if ( v61 )
        sub_1623A60((__int64)&v202, v61, 2);
      LODWORD(v203) = *(_DWORD *)(v12 + 64);
      result = (_QWORD *)sub_1D322C0(
                           v62,
                           (__int64)v57,
                           v59,
                           (__int64)&v202,
                           (unsigned int)v196,
                           v197,
                           *(double *)a3.m128i_i64,
                           a4,
                           *(double *)a5.m128i_i64);
      v63 = v202;
      if ( !v202 )
        return result;
      goto LABEL_61;
    }
    return v57;
  }
  v20 = *(_QWORD *)(v11 + 40);
  v160 = *(_QWORD *)(v11 + 48);
  v162 = *(_DWORD *)(v11 + 48);
  v21 = *(unsigned __int16 *)(v20 + 24);
  v173 = v20;
  if ( v21 != 32 && v21 != 10 )
  {
    if ( !*(_BYTE *)sub_1E0A0C0((*a1)[4]) )
    {
      v22 = v194;
      goto LABEL_7;
    }
    v164 = 0;
    goto LABEL_69;
  }
  v26 = *(_QWORD *)(v20 + 88);
  if ( (_BYTE)v14 )
  {
    v28 = word_42FA680[(unsigned __int8)(v14 - 14)];
    v166 = *(_DWORD *)(v26 + 32);
    if ( v166 <= 0x40 )
      goto LABEL_16;
  }
  else
  {
    v165 = *(_QWORD *)(v20 + 88);
    v27 = sub_1F58D30((__int64)&v194);
    v26 = v165;
    v28 = v27;
    v166 = *(_DWORD *)(v165 + 32);
    if ( v166 <= 0x40 )
    {
LABEL_16:
      v29 = *(_QWORD *)(v26 + 24);
      goto LABEL_17;
    }
  }
  v163 = v26;
  if ( v166 - (unsigned int)sub_16A57B0(v26 + 24) > 0x40 )
  {
LABEL_53:
    v56 = *a1;
    v202 = 0;
    LODWORD(v203) = 0;
    v57 = sub_1D2B300(v56, 0x30u, (__int64)&v202, v196, (__int64)v197, a9);
    if ( v202 )
      sub_161E7C0((__int64)&v202, v202);
    return v57;
  }
  v29 = **(_QWORD **)(v163 + 24);
LABEL_17:
  if ( v28 <= v29 )
    goto LABEL_53;
  if ( v19 == 104 )
  {
    if ( (_BYTE)v14 )
    {
      if ( a1[1][v14 + 15] )
      {
        if ( sub_1D18C00(v12, 1, v174)
          || (v75 = a1[1], v76 = *(__int64 (**)())(*v75 + 952LL), v76 != sub_1F3CC20)
          && ((unsigned __int8 (__fastcall *)(_QWORD *, _QWORD, __int64))v76)(v75, v194, v195) )
        {
          v77 = *(_QWORD *)(v173 + 88);
          v78 = *(_QWORD **)(v77 + 24);
          if ( *(_DWORD *)(v77 + 32) > 0x40u )
            v78 = (_QWORD *)*v78;
          v79 = (unsigned int *)(*(_QWORD *)(v12 + 32) + 40LL * (unsigned int)v78);
          result = *(_QWORD **)v79;
          v80 = *(_QWORD *)(*(_QWORD *)v79 + 40LL) + 16LL * v79[2];
          if ( *(_BYTE *)v80 == v181 && (*(const void ***)(v80 + 8) == v169 || v181) )
            return result;
        }
      }
    }
  }
  v164 = v173;
  if ( !*(_BYTE *)sub_1E0A0C0((*a1)[4]) )
  {
    v30 = 0;
    goto LABEL_21;
  }
LABEL_69:
  v22 = v194;
  if ( (_BYTE)v194 )
    v64 = word_42FA680[(unsigned __int8)(v194 - 14)];
  else
    v64 = sub_1F58D30((__int64)&v194);
  v30 = v64 - 1;
  if ( v164 )
  {
LABEL_21:
    v31 = *(_WORD *)(v12 + 24);
    if ( v31 == 158 )
    {
      if ( sub_1D18C00(v12, 1, v174) )
      {
        v96 = *(_QWORD *)(v164 + 88);
        v97 = *(_QWORD **)(v96 + 24);
        if ( *(_DWORD *)(v96 + 32) > 0x40u )
          v97 = (_QWORD *)*v97;
        v22 = v194;
        if ( (_QWORD *)v30 == v97
          && ((_BYTE)v194
            ? (unsigned __int8)(v194 - 14) <= 0x47u || (unsigned __int8)(v194 - 2) <= 5u
            : sub_1F58CF0((__int64)&v194)) )
        {
          a3 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v12 + 32));
          v137 = *(_QWORD *)(**(_QWORD **)(v12 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v12 + 32) + 8LL);
          v138 = *(_BYTE *)v137;
          v139 = *(_QWORD *)(v137 + 8);
          LOBYTE(v202) = v138;
          v203 = v139;
          if ( v138 ? (unsigned __int8)(v138 - 2) <= 5u : sub_1F58D10((__int64)&v202) )
          {
            v141 = *(_QWORD *)(v10 + 72);
            v142 = *a1;
            v202 = v141;
            if ( v141 )
              sub_1623A60((__int64)&v202, v141, 2);
            LODWORD(v203) = *(_DWORD *)(v10 + 64);
            result = (_QWORD *)sub_1D309E0(
                                 v142,
                                 145,
                                 (__int64)&v202,
                                 (unsigned int)v196,
                                 v197,
                                 0,
                                 *(double *)a3.m128i_i64,
                                 a4,
                                 *(double *)a5.m128i_i64,
                                 *(_OWORD *)&a3);
            v63 = v202;
            if ( !v202 )
              return result;
            goto LABEL_61;
          }
        }
        v31 = *(_WORD *)(v12 + 24);
        goto LABEL_23;
      }
      v31 = *(_WORD *)(v12 + 24);
    }
    v22 = v194;
LABEL_23:
    if ( v31 == 105 )
      goto LABEL_99;
    if ( v31 != 110 )
    {
      v161 = 1;
      goto LABEL_9;
    }
    if ( v22 )
      v32 = word_42FA680[(unsigned __int8)(v22 - 14)];
    else
      v32 = sub_1F58D30((__int64)&v194);
    v33 = *(_QWORD *)(v164 + 88);
    v34 = *(_QWORD **)(v33 + 24);
    if ( *(_DWORD *)(v33 + 32) > 0x40u )
      v34 = (_QWORD *)*v34;
    v35 = *(unsigned int *)(*(_QWORD *)(v12 + 88) + 4LL * (unsigned int)v34);
    if ( (_DWORD)v35 != -1 )
    {
      v36 = *(_QWORD *)(v12 + 32);
      if ( v32 > (int)v35 )
      {
        v37 = *(_QWORD *)v36;
        v159 = *(unsigned int *)(v36 + 8);
      }
      else
      {
        v35 = (unsigned int)(v35 - v32);
        v37 = *(_QWORD *)(v36 + 40);
        v159 = *(unsigned int *)(v36 + 48);
      }
      if ( *(_WORD *)(v37 + 24) == 104 )
      {
        v112 = (__int64 *)(*(_QWORD *)(v37 + 32) + 40 * v35);
        v113 = *v112;
        v114 = v112[1];
        result = (_QWORD *)*v112;
        v115 = *(_QWORD *)(*v112 + 40) + 16LL * *((unsigned int *)v112 + 2);
        if ( *(_BYTE *)v115 == v181 && (*(const void ***)(v115 + 8) == v169 || v181) )
          return result;
        v116 = *(_QWORD *)(v37 + 72);
        v117 = *a1;
        v202 = v116;
        if ( v116 )
          sub_1623A60((__int64)&v202, v116, 2);
        LODWORD(v203) = *(_DWORD *)(v37 + 64);
        result = (_QWORD *)sub_1D322C0(
                             v117,
                             v113,
                             v114,
                             (__int64)&v202,
                             (unsigned int)v196,
                             v197,
                             *(double *)a3.m128i_i64,
                             a4,
                             *(double *)a5.m128i_i64);
        if ( !v202 )
          return result;
      }
      else
      {
        v38 = a1[1];
        v161 = *((_BYTE *)a1 + 24);
        if ( v161 )
        {
          v39 = 1;
          if ( v22 == 1 || v22 && (v39 = v22, v38[v22 + 15]) )
          {
            v40 = (char *)v38 + 259 * (unsigned int)v39;
            if ( v40[2528] )
            {
              if ( v38[v39 + 15] && v40[2532] != 2 )
                goto LABEL_9;
            }
          }
        }
        v175 = v35;
        v182 = a1[1];
        v41 = *(__int64 (__fastcall **)(__int64, __int64))(*v38 + 48LL);
        v42 = sub_1E0A0C0((*a1)[4]);
        if ( v41 == sub_1D13A20 )
        {
          v43 = sub_15A9520(v42, 0);
          v44 = v175;
          v45 = 8 * v43;
          if ( 8 * v43 == 32 )
          {
            v46 = 5;
          }
          else if ( v45 > 0x20 )
          {
            v46 = 6;
            if ( v45 != 64 )
            {
              v46 = 0;
              if ( v45 == 128 )
                v46 = 7;
            }
          }
          else
          {
            v46 = 3;
            if ( v45 != 8 )
              v46 = 4 * (v45 == 16);
          }
        }
        else
        {
          v46 = v41((__int64)v182, v42);
          v44 = v175;
        }
        v47 = *(_QWORD *)(v12 + 72);
        v48 = *a1;
        v49 = v46;
        v202 = v47;
        if ( v47 )
        {
          v183 = v44;
          sub_1623A60((__int64)&v202, v47, 2);
          v44 = v183;
        }
        LODWORD(v203) = *(_DWORD *)(v12 + 64);
        v50 = sub_1D38BB0((__int64)v48, v44, (__int64)&v202, v49, 0, 0, a3, a4, a5, 0);
        v51 = *(_QWORD *)(v10 + 72);
        v52 = v50;
        v54 = v53;
        v200 = v51;
        if ( v51 )
          sub_1623A60((__int64)&v200, v51, 2);
        *((_QWORD *)&v155 + 1) = v54;
        *(_QWORD *)&v155 = v52;
        LODWORD(v201) = *(_DWORD *)(v10 + 64);
        result = sub_1D332F0(
                   v48,
                   106,
                   (__int64)&v200,
                   (unsigned int)v196,
                   v197,
                   0,
                   *(double *)a3.m128i_i64,
                   a4,
                   a5,
                   v37,
                   v159,
                   v155);
        if ( v200 )
        {
          v176 = result;
          sub_161E7C0((__int64)&v200, v200);
          result = v176;
        }
        if ( !v202 )
          return result;
      }
      v184 = result;
      sub_161E7C0((__int64)&v202, v202);
      return v184;
    }
    goto LABEL_53;
  }
LABEL_7:
  v164 = 0;
  if ( *(_WORD *)(v12 + 24) != 105 )
  {
    v161 = 0;
LABEL_9:
    v23 = *(_QWORD *)(v12 + 48);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v12 + 48);
      do
      {
        v25 = *(_QWORD *)(v24 + 16);
        if ( *(_WORD *)(v25 + 24) != 106 )
          goto LABEL_12;
        v65 = *(_QWORD *)(v25 + 32);
        if ( *(_QWORD *)v65 != v12 )
          goto LABEL_12;
        if ( *(_DWORD *)(v65 + 8) != v174 )
          goto LABEL_12;
        v66 = *(unsigned __int16 *)(*(_QWORD *)(v65 + 40) + 24LL);
        if ( v66 != 32 && v66 != 10 )
          goto LABEL_12;
        v24 = *(_QWORD *)(v24 + 32);
      }
      while ( v24 );
    }
    if ( v22 )
      v67 = word_42FA680[(unsigned __int8)(v22 - 14)];
    else
      v67 = sub_1F58D30((__int64)&v194);
    LODWORD(v203) = v67;
    if ( v67 > 0x40 )
    {
      sub_16A4EF0((__int64)&v202, 0, 0);
      v23 = *(_QWORD *)(v12 + 48);
    }
    else
    {
      v202 = 0;
    }
    if ( !v23 )
    {
LABEL_94:
      if ( (unsigned __int8)sub_1FC14D0((__int64)a1, v12, v174, (int)&v202, 1u) )
      {
        result = (_QWORD *)v10;
        if ( (unsigned int)v203 > 0x40 && v202 )
        {
          j_j___libc_free_0_0(v202);
          return (_QWORD *)v10;
        }
        return result;
      }
      if ( (unsigned int)v203 > 0x40 && v202 )
        j_j___libc_free_0_0(v202);
      v22 = v194;
LABEL_12:
      if ( v22 )
      {
        switch ( v22 )
        {
          case 0xEu:
          case 0xFu:
          case 0x10u:
          case 0x11u:
          case 0x12u:
          case 0x13u:
          case 0x14u:
          case 0x15u:
          case 0x16u:
          case 0x17u:
          case 0x38u:
          case 0x39u:
          case 0x3Au:
          case 0x3Bu:
          case 0x3Cu:
          case 0x3Du:
            v84 = 2;
            break;
          case 0x18u:
          case 0x19u:
          case 0x1Au:
          case 0x1Bu:
          case 0x1Cu:
          case 0x1Du:
          case 0x1Eu:
          case 0x1Fu:
          case 0x20u:
          case 0x3Eu:
          case 0x3Fu:
          case 0x40u:
          case 0x41u:
          case 0x42u:
          case 0x43u:
            v84 = 3;
            break;
          case 0x21u:
          case 0x22u:
          case 0x23u:
          case 0x24u:
          case 0x25u:
          case 0x26u:
          case 0x27u:
          case 0x28u:
          case 0x44u:
          case 0x45u:
          case 0x46u:
          case 0x47u:
          case 0x48u:
          case 0x49u:
            v84 = 4;
            break;
          case 0x29u:
          case 0x2Au:
          case 0x2Bu:
          case 0x2Cu:
          case 0x2Du:
          case 0x2Eu:
          case 0x2Fu:
          case 0x30u:
          case 0x4Au:
          case 0x4Bu:
          case 0x4Cu:
          case 0x4Du:
          case 0x4Eu:
          case 0x4Fu:
            v84 = 5;
            break;
          case 0x31u:
          case 0x32u:
          case 0x33u:
          case 0x34u:
          case 0x35u:
          case 0x36u:
          case 0x50u:
          case 0x51u:
          case 0x52u:
          case 0x53u:
          case 0x54u:
          case 0x55u:
            v84 = 6;
            break;
          case 0x37u:
            v84 = 7;
            break;
          case 0x56u:
          case 0x57u:
          case 0x58u:
          case 0x62u:
          case 0x63u:
          case 0x64u:
            v84 = 8;
            break;
          case 0x59u:
          case 0x5Au:
          case 0x5Bu:
          case 0x5Cu:
          case 0x5Du:
          case 0x65u:
          case 0x66u:
          case 0x67u:
          case 0x68u:
          case 0x69u:
            v84 = 9;
            break;
          case 0x5Eu:
          case 0x5Fu:
          case 0x60u:
          case 0x61u:
          case 0x6Au:
          case 0x6Bu:
          case 0x6Cu:
          case 0x6Du:
            v84 = 10;
            break;
        }
        LOBYTE(v198) = v84;
        v83 = v198;
        v199 = 0;
        v168 = 0;
        v202 = v198;
        v203 = 0;
        v167 = 0;
        if ( v181 == v84 )
          goto LABEL_118;
      }
      else
      {
        v81 = sub_1F596B0((__int64)&v194);
        LOBYTE(v198) = v81;
        v83 = v198;
        v84 = v81;
        v167 = v82;
        v199 = v82;
        v168 = v82;
        v202 = v198;
        v203 = v82;
        if ( v181 == v81 )
        {
          if ( v81 || v169 == (const void **)v82 )
            goto LABEL_118;
          goto LABEL_115;
        }
      }
      if ( v181 )
      {
        v186 = sub_1F6C8D0(v181);
LABEL_125:
        if ( v84 )
          v86 = sub_1F6C8D0(v84);
        else
          v86 = sub_1F58D40((__int64)&v202);
        if ( v86 > v186 )
        {
          v87 = a1[1];
          v88 = *(__int64 (**)())(*v87 + 800LL);
          if ( v88 == sub_1D12DF0
            || !((unsigned __int8 (__fastcall *)(_QWORD *, __int64, __int64, _QWORD, const void **))v88)(
                  v87,
                  v83,
                  v168,
                  (unsigned int)v196,
                  v197) )
          {
            return 0;
          }
        }
LABEL_118:
        if ( *(_WORD *)(v12 + 24) != 158 )
        {
          v85 = 0;
          goto LABEL_120;
        }
        v85 = sub_1D18C00(v12, 1, v174);
        if ( !v85 )
          return 0;
        v103 = *(_QWORD *)(**(_QWORD **)(v12 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v12 + 32) + 8LL);
        v104 = *(_BYTE *)v103;
        v105 = *(_QWORD *)(v103 + 8);
        LOBYTE(v200) = v104;
        v201 = v105;
        if ( v104 )
        {
          if ( (unsigned __int8)(v104 - 14) > 0x5Fu )
            return 0;
          switch ( v104 )
          {
            case 24:
            case 25:
            case 26:
            case 27:
            case 28:
            case 29:
            case 30:
            case 31:
            case 32:
            case 62:
            case 63:
            case 64:
            case 65:
            case 66:
            case 67:
              v106 = 3;
              break;
            case 33:
            case 34:
            case 35:
            case 36:
            case 37:
            case 38:
            case 39:
            case 40:
            case 68:
            case 69:
            case 70:
            case 71:
            case 72:
            case 73:
              v106 = 4;
              break;
            case 41:
            case 42:
            case 43:
            case 44:
            case 45:
            case 46:
            case 47:
            case 48:
            case 74:
            case 75:
            case 76:
            case 77:
            case 78:
            case 79:
              v106 = 5;
              break;
            case 49:
            case 50:
            case 51:
            case 52:
            case 53:
            case 54:
            case 80:
            case 81:
            case 82:
            case 83:
            case 84:
            case 85:
              v106 = 6;
              break;
            case 55:
              v106 = 7;
              break;
            case 86:
            case 87:
            case 88:
            case 98:
            case 99:
            case 100:
              v106 = 8;
              break;
            case 89:
            case 90:
            case 91:
            case 92:
            case 93:
            case 101:
            case 102:
            case 103:
            case 104:
            case 105:
              v106 = 9;
              break;
            case 94:
            case 95:
            case 96:
            case 97:
            case 106:
            case 107:
            case 108:
            case 109:
              v106 = 10;
              break;
            default:
              v106 = 2;
              break;
          }
          v107 = 0;
        }
        else
        {
          v178 = v85;
          if ( !sub_1F58D20((__int64)&v200) )
            return 0;
          v143 = sub_1F596B0((__int64)&v200);
          v85 = v178;
          v106 = v143;
          v107 = v144;
        }
        LOBYTE(v202) = v106;
        v203 = v107;
        if ( v84 == v106 )
        {
          if ( v106 || v107 == v167 )
          {
LABEL_186:
            if ( (_BYTE)v194 )
            {
              v108 = word_42FA680[(unsigned __int8)(v194 - 14)];
            }
            else
            {
              v192 = v85;
              v154 = sub_1F58D30((__int64)&v194);
              v85 = v192;
              v108 = v154;
            }
            v109 = *(_QWORD *)(v12 + 32);
            if ( (_BYTE)v200 )
            {
              v12 = *(_QWORD *)v109;
              if ( word_42FA680[(unsigned __int8)(v200 - 14)] == v108 )
                v85 = 0;
              v174 = *(_DWORD *)(v109 + 8);
              switch ( (char)v200 )
              {
                case 14:
                case 15:
                case 16:
                case 17:
                case 18:
                case 19:
                case 20:
                case 21:
                case 22:
                case 23:
                case 56:
                case 57:
                case 58:
                case 59:
                case 60:
                case 61:
                  v110 = 2;
                  break;
                case 24:
                case 25:
                case 26:
                case 27:
                case 28:
                case 29:
                case 30:
                case 31:
                case 32:
                case 62:
                case 63:
                case 64:
                case 65:
                case 66:
                case 67:
                  v110 = 3;
                  break;
                case 33:
                case 34:
                case 35:
                case 36:
                case 37:
                case 38:
                case 39:
                case 40:
                case 68:
                case 69:
                case 70:
                case 71:
                case 72:
                case 73:
                  v110 = 4;
                  break;
                case 41:
                case 42:
                case 43:
                case 44:
                case 45:
                case 46:
                case 47:
                case 48:
                case 74:
                case 75:
                case 76:
                case 77:
                case 78:
                case 79:
                  v110 = 5;
                  break;
                case 49:
                case 50:
                case 51:
                case 52:
                case 53:
                case 54:
                case 80:
                case 81:
                case 82:
                case 83:
                case 84:
                case 85:
                  v110 = 6;
                  break;
                case 55:
                  v110 = 7;
                  break;
                case 86:
                case 87:
                case 88:
                case 98:
                case 99:
                case 100:
                  v110 = 8;
                  break;
                case 89:
                case 90:
                case 91:
                case 92:
                case 93:
                case 101:
                case 102:
                case 103:
                case 104:
                case 105:
                  v110 = 9;
                  break;
                case 94:
                case 95:
                case 96:
                case 97:
                case 106:
                case 107:
                case 108:
                case 109:
                  v110 = 10;
                  break;
                default:
                  ++*(_DWORD *)(v10 + 576);
                  BUG();
              }
              v111 = 0;
            }
            else
            {
              v172 = *(_QWORD *)(v12 + 32);
              v180 = v85;
              v152 = sub_1F58D30((__int64)&v200);
              v153 = v180;
              v12 = *(_QWORD *)v172;
              if ( v152 == v108 )
                v153 = 0;
              v174 = *(_DWORD *)(v172 + 8);
              v191 = v153;
              v110 = sub_1F596B0((__int64)&v200);
              v85 = v191;
            }
            LOBYTE(v198) = v110;
            v199 = v111;
LABEL_120:
            if ( !*((_BYTE *)a1 + 24) )
            {
              if ( !v164
                && sub_1D18C00(v12, 1, v174)
                && *(_WORD *)(v12 + 24) == 185
                && (*(_BYTE *)(v12 + 27) & 0xC) == 0
                && (*(_WORD *)(v12 + 26) & 0x380) == 0
                && !(unsigned __int8)sub_1D19270(*(_QWORD *)(*(_QWORD *)(v10 + 32) + 40LL), v12, v132, v133, v134, v135)
                && *(_WORD *)(v12 + 24) == 185
                && (*(_BYTE *)(v12 + 26) & 8) == 0 )
              {
                return (_QWORD *)sub_1F988B0(
                                   a1,
                                   v10,
                                   v194,
                                   v195,
                                   *(_QWORD *)(*(_QWORD *)(v10 + 32) + 40LL),
                                   *(_QWORD *)(*(_QWORD *)(v10 + 32) + 48LL),
                                   *(double *)a3.m128i_i64,
                                   a4,
                                   a5,
                                   v12);
              }
              return 0;
            }
            if ( !v161 )
              return 0;
            v89 = *(_QWORD *)(v173 + 88);
            v90 = *(_QWORD *)(v89 + 24);
            if ( *(_DWORD *)(v89 + 32) > 0x40u )
              v90 = **(_QWORD **)(v89 + 24);
            v91 = *(_WORD *)(v12 + 24);
            v92 = v90;
            if ( v91 == 185 )
            {
              if ( (*(_BYTE *)(v12 + 27) & 0xC) != 0 || (*(_WORD *)(v12 + 26) & 0x380) != 0 )
                return 0;
              goto LABEL_146;
            }
            if ( v91 == 111 )
            {
              v118 = *(unsigned int **)(v12 + 32);
              v119 = *(_QWORD *)v118;
              v120 = *(_QWORD *)(*(_QWORD *)v118 + 40LL) + 16LL * v118[2];
              if ( (_BYTE)v198 != *(_BYTE *)v120 || v199 != *(_QWORD *)(v120 + 8) && !*(_BYTE *)v120 )
                return 0;
              if ( *(_WORD *)(v119 + 24) != 185 )
                return 0;
              if ( (*(_BYTE *)(v119 + 27) & 0xC) != 0 )
                return 0;
              if ( (*(_WORD *)(v119 + 26) & 0x380) != 0 )
                return 0;
              if ( !sub_1D18C00(v12, 1, v174) )
                return 0;
              v12 = **(_QWORD **)(v12 + 32);
              if ( !v12 )
                return 0;
LABEL_146:
              if ( !sub_1D18C00(v12, 1, 0) || (*(_BYTE *)(v12 + 26) & 8) != 0 )
                return 0;
              if ( v92 == -1 )
                return sub_1D2B530(*a1, v83, v168, v93, v94, v95);
              else
                return (_QWORD *)sub_1F988B0(
                                   a1,
                                   v10,
                                   v194,
                                   v195,
                                   v173,
                                   v160 & 0xFFFFFFFF00000000LL | v162,
                                   *(double *)a3.m128i_i64,
                                   a4,
                                   a5,
                                   v12);
            }
            if ( v91 != 110 )
              return 0;
            v170 = v90;
            v187 = v85;
            if ( !sub_1D18C00(v12, 1, v174) || v187 )
              return 0;
            v121 = v170;
            if ( (_BYTE)v194 )
            {
              v122 = word_42FA680[(unsigned __int8)(v194 - 14)];
              if ( v92 > v122 )
              {
                v92 = -1;
                goto LABEL_231;
              }
            }
            else
            {
              v150 = sub_1F58D30((__int64)&v194);
              v121 = v170;
              v122 = v150;
              if ( v150 < v92 )
              {
                v92 = -1;
LABEL_275:
                if ( v92 >= v122 )
                {
                  v149 = *(_QWORD *)(v12 + 32);
                  v12 = *(_QWORD *)(v149 + 40);
                  v124 = *(_DWORD *)(v149 + 48);
                  goto LABEL_232;
                }
LABEL_231:
                v123 = *(_QWORD *)(v12 + 32);
                v12 = *(_QWORD *)v123;
                v124 = *(_DWORD *)(v123 + 8);
LABEL_232:
                v125 = *(_WORD *)(v12 + 24);
                if ( v125 == 158 )
                {
                  v193 = v122;
                  if ( !sub_1D18C00(v12, 1, v124) )
                    return 0;
                  v122 = v193;
                  v12 = **(_QWORD **)(v12 + 32);
                  v125 = *(_WORD *)(v12 + 24);
                }
                if ( v125 != 185 || (*(_BYTE *)(v12 + 27) & 0xC) != 0 || (*(_WORD *)(v12 + 26) & 0x380) != 0 )
                  return 0;
                v126 = *a1;
                if ( v122 <= v92 )
                  v92 -= v122;
                v127 = (unsigned __int8 *)(*(_QWORD *)(v173 + 40) + 16LL * v162);
                v128 = *(_QWORD *)(v173 + 72);
                v129 = (const void **)*((_QWORD *)v127 + 1);
                v130 = *v127;
                v202 = v128;
                v160 = v162 | v160 & 0xFFFFFFFF00000000LL;
                if ( v128 )
                {
                  v171 = v130;
                  v177 = v129;
                  v188 = v126;
                  sub_1623A60((__int64)&v202, v128, 2);
                  v130 = v171;
                  v129 = v177;
                  v126 = v188;
                }
                LODWORD(v203) = *(_DWORD *)(v173 + 64);
                v173 = sub_1D38BB0((__int64)v126, v92, (__int64)&v202, v130, v129, 0, a3, a4, a5, 0);
                v162 = v131;
                if ( v202 )
                  sub_161E7C0((__int64)&v202, v202);
                goto LABEL_146;
              }
            }
            v92 = *(_DWORD *)(*(_QWORD *)(v12 + 88) + 4LL * v121);
            goto LABEL_275;
          }
        }
        else if ( v84 )
        {
          v147 = sub_1F6C8D0(v84);
          goto LABEL_269;
        }
        v179 = v106;
        v190 = v85;
        v151 = sub_1F58D40((__int64)&v198);
        v146 = v179;
        v145 = v190;
        v147 = v151;
LABEL_269:
        if ( v146 )
        {
          v148 = sub_1F6C8D0(v146);
        }
        else
        {
          v189 = v145;
          v148 = sub_1F58D40((__int64)&v202);
          v85 = v189;
        }
        if ( v148 < v147 )
          return 0;
        goto LABEL_186;
      }
LABEL_115:
      v186 = sub_1F58D40((__int64)&v196);
      goto LABEL_125;
    }
    v158 = v12;
    v157 = a1;
    v156 = v10;
    while ( 1 )
    {
      v71 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v23 + 16) + 32LL) + 40LL) + 88LL);
      if ( (_BYTE)v194 )
      {
        v68 = *(_DWORD *)(v71 + 32);
        v69 = word_42FA680[(unsigned __int8)(v194 - 14)];
        if ( v68 <= 0x40 )
          goto LABEL_85;
LABEL_89:
        if ( v68 - (unsigned int)sub_16A57B0(v71 + 24) > 0x40 )
          goto LABEL_86;
        v70 = **(_QWORD **)(v71 + 24);
        if ( v69 <= v70 )
          goto LABEL_86;
LABEL_91:
        v73 = 1LL << v70;
        if ( (unsigned int)v203 > 0x40 )
        {
          *(_QWORD *)(v202 + 8LL * ((unsigned int)v70 >> 6)) |= v73;
          goto LABEL_86;
        }
        v202 |= v73;
        v23 = *(_QWORD *)(v23 + 32);
        if ( !v23 )
        {
LABEL_93:
          v12 = v158;
          a1 = v157;
          v10 = v156;
          goto LABEL_94;
        }
      }
      else
      {
        v72 = sub_1F58D30((__int64)&v194);
        v68 = *(_DWORD *)(v71 + 32);
        v69 = v72;
        if ( v68 > 0x40 )
          goto LABEL_89;
LABEL_85:
        v70 = *(_QWORD *)(v71 + 24);
        if ( v69 > v70 )
          goto LABEL_91;
LABEL_86:
        v23 = *(_QWORD *)(v23 + 32);
        if ( !v23 )
          goto LABEL_93;
      }
    }
  }
LABEL_99:
  v74 = *(_QWORD *)(v12 + 32);
  if ( v173 != *(_QWORD *)(v74 + 80) || v162 != *(_DWORD *)(v74 + 88) )
  {
    v161 = v164 != 0;
    goto LABEL_9;
  }
  v98 = *(_QWORD *)(v74 + 40);
  v99 = *(_QWORD *)(v74 + 48);
  if ( v22 )
    v100 = (unsigned __int8)(v22 - 14) <= 0x47u || (unsigned __int8)(v22 - 2) <= 5u;
  else
    v100 = sub_1F58CF0((__int64)&v194);
  if ( !v100 )
    return (_QWORD *)v98;
  v101 = *(_QWORD *)(v10 + 72);
  v102 = *a1;
  v202 = v101;
  if ( v101 )
    sub_1623A60((__int64)&v202, v101, 2);
  LODWORD(v203) = *(_DWORD *)(v10 + 64);
  result = (_QWORD *)sub_1D321C0(
                       v102,
                       v98,
                       v99,
                       (__int64)&v202,
                       (unsigned int)v196,
                       v197,
                       *(double *)a3.m128i_i64,
                       a4,
                       *(double *)a5.m128i_i64);
  v63 = v202;
  if ( v202 )
  {
LABEL_61:
    v185 = result;
    sub_161E7C0((__int64)&v202, v63);
    return v185;
  }
  return result;
}
