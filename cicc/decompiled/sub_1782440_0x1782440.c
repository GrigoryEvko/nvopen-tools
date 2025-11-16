// Function: sub_1782440
// Address: 0x1782440
//
__int64 __fastcall sub_1782440(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  _BYTE *v13; // r14
  __int64 v14; // r13
  __int64 v15; // r12
  char v16; // bl
  __int64 v17; // rax
  unsigned __int8 *v18; // rax
  __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v23; // rcx
  double v24; // xmm4_8
  double v25; // xmm5_8
  unsigned __int8 v26; // al
  __int64 v27; // r11
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  void *v31; // rax
  int v32; // edx
  _BYTE *v33; // rdx
  _BYTE *v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // r8
  char v37; // cl
  bool v38; // al
  bool v39; // al
  __int64 v40; // rax
  unsigned __int8 v41; // al
  __int64 v42; // rax
  bool v43; // al
  __int64 v44; // r13
  __int64 v45; // r12
  __int64 v46; // rax
  _QWORD *v47; // rax
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rdx
  void *v50; // rcx
  int v51; // eax
  _QWORD *v52; // rdx
  char v53; // al
  unsigned __int8 *v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // r10
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 *v59; // rsi
  __int64 v60; // rdx
  char v61; // cl
  __int64 v62; // rcx
  __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rbx
  __int64 v66; // rax
  unsigned __int8 *v67; // r14
  __int64 v68; // r13
  __int64 v69; // rax
  __int64 v70; // r14
  __int64 v71; // rax
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  void *v74; // rcx
  _BYTE *v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rcx
  int v78; // eax
  _BYTE *v79; // rax
  _BYTE *v80; // rdi
  unsigned __int8 v81; // al
  __int64 v82; // r9
  unsigned int v83; // edx
  __int64 v84; // rcx
  int v85; // eax
  _BYTE *v86; // rax
  _BYTE *v87; // rdi
  unsigned __int8 v88; // al
  unsigned int v89; // eax
  int v90; // eax
  int v91; // r10d
  unsigned int v92; // ecx
  __int64 v93; // rax
  __int64 v94; // rax
  bool v95; // al
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // rax
  __int64 v99; // rcx
  int v100; // eax
  __int64 v101; // rcx
  __int64 v102; // rax
  bool v103; // al
  __int64 v104; // rax
  bool v105; // al
  __int64 v106; // rax
  _BOOL4 v107; // esi
  __int64 v108; // rax
  __int64 v109; // rax
  unsigned int v110; // edx
  __int64 v111; // rax
  char v112; // cl
  unsigned int v113; // edx
  int v114; // eax
  bool v115; // al
  __int64 v116; // rax
  _BOOL4 v117; // esi
  _BYTE *v118; // rax
  __int64 v119; // r15
  __int64 v120; // rax
  __int64 v121; // rcx
  unsigned __int64 v122; // rdx
  __int64 v123; // rdx
  __int64 v124; // rdx
  unsigned __int64 v125; // rax
  __int64 v126; // rax
  __int64 v127; // rcx
  int v128; // eax
  _QWORD *v129; // rax
  __int64 v130; // r15
  __int64 *v131; // rax
  __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // rdx
  int v135; // edi
  __int64 v136; // rax
  _BYTE *v137; // rdx
  __int64 v138; // rdx
  unsigned __int16 v139; // cx
  __int64 v140; // rdi
  __int64 v141; // rax
  __int64 v142; // r15
  __int64 *v143; // rax
  __int64 v144; // rdi
  __int64 v145; // rdx
  __int64 v146; // rax
  __int64 *v147; // [rsp+0h] [rbp-D0h]
  __int64 *v148; // [rsp+10h] [rbp-C0h]
  _QWORD **v149; // [rsp+10h] [rbp-C0h]
  char v150; // [rsp+18h] [rbp-B8h]
  __int64 v151; // [rsp+18h] [rbp-B8h]
  __int64 v152; // [rsp+18h] [rbp-B8h]
  __int64 v153; // [rsp+18h] [rbp-B8h]
  __int64 v154; // [rsp+20h] [rbp-B0h]
  __int64 v155; // [rsp+20h] [rbp-B0h]
  __int64 v156; // [rsp+20h] [rbp-B0h]
  unsigned int v157; // [rsp+20h] [rbp-B0h]
  unsigned int v158; // [rsp+20h] [rbp-B0h]
  __int64 v159; // [rsp+20h] [rbp-B0h]
  __int64 v160; // [rsp+20h] [rbp-B0h]
  unsigned int v161; // [rsp+20h] [rbp-B0h]
  __int64 v162; // [rsp+28h] [rbp-A8h]
  __int64 *v163; // [rsp+28h] [rbp-A8h]
  __int64 v164; // [rsp+28h] [rbp-A8h]
  __int64 v165; // [rsp+28h] [rbp-A8h]
  unsigned int v166; // [rsp+28h] [rbp-A8h]
  int v167; // [rsp+28h] [rbp-A8h]
  __int64 v168; // [rsp+30h] [rbp-A0h]
  int v169; // [rsp+30h] [rbp-A0h]
  int v170; // [rsp+30h] [rbp-A0h]
  __int64 v171; // [rsp+30h] [rbp-A0h]
  char v172; // [rsp+30h] [rbp-A0h]
  __int64 v173; // [rsp+30h] [rbp-A0h]
  __int64 v174; // [rsp+30h] [rbp-A0h]
  __int64 v175; // [rsp+30h] [rbp-A0h]
  _QWORD **v176; // [rsp+30h] [rbp-A0h]
  __int64 v177; // [rsp+30h] [rbp-A0h]
  unsigned int v178; // [rsp+30h] [rbp-A0h]
  int v179; // [rsp+30h] [rbp-A0h]
  __int64 v180; // [rsp+30h] [rbp-A0h]
  __int64 v181; // [rsp+30h] [rbp-A0h]
  __int64 v182; // [rsp+30h] [rbp-A0h]
  __int64 v183; // [rsp+30h] [rbp-A0h]
  char v184; // [rsp+3Fh] [rbp-91h]
  __int64 v185; // [rsp+40h] [rbp-90h]
  _BYTE *v186; // [rsp+48h] [rbp-88h]
  __int64 v187; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v188; // [rsp+58h] [rbp-78h]
  __int64 v189; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v190; // [rsp+68h] [rbp-68h]
  __int16 v191; // [rsp+70h] [rbp-60h]
  __int64 v192; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v193; // [rsp+88h] [rbp-48h]
  __int16 v194; // [rsp+90h] [rbp-40h]

  v13 = (_BYTE *)a2;
  v14 = a2;
  v15 = *(_QWORD *)(a2 - 24);
  v16 = *(_BYTE *)(a2 + 16);
  v186 = *(_BYTE **)(a2 - 48);
  v185 = *(_QWORD *)a2;
  v17 = *(_QWORD *)(v15 + 8);
  if ( v17 )
  {
    if ( !*(_QWORD *)(v17 + 8) )
    {
      v18 = sub_1780EA0(v15, (__int64)a1, a2, a4, *(double *)a5.m128_u64, a6, a7);
      if ( v18 )
      {
        if ( *(_QWORD *)(a2 - 24) )
        {
          v19 = *(_QWORD *)(a2 - 16);
          v20 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v20 = v19;
          if ( v19 )
            *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
        }
        *(_QWORD *)(a2 - 24) = v18;
        v21 = *((_QWORD *)v18 + 1);
        *(_QWORD *)(a2 - 16) = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = (a2 - 16) | *(_QWORD *)(v21 + 16) & 3LL;
        *(_QWORD *)(a2 - 8) = *(_QWORD *)(a2 - 8) & 3LL | (unsigned __int64)(v18 + 8);
        *((_QWORD *)v18 + 1) = a2 - 24;
        return v14;
      }
    }
  }
  if ( (unsigned __int8)sub_1781E40(a1, a2, a3, a4) )
    return v14;
  v26 = *(_BYTE *)(v15 + 16);
  v27 = v15 + 24;
  v184 = v16 == 42;
  if ( v26 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) != 16 )
      goto LABEL_43;
    if ( v26 > 0x10u )
      goto LABEL_43;
    v64 = sub_15A1020((_BYTE *)v15, a2, *(_QWORD *)v15, v23);
    if ( !v64 || *(_BYTE *)(v64 + 16) != 13 )
      goto LABEL_43;
    v27 = v64 + 24;
  }
  v28 = (unsigned __int8)v186[16];
  if ( v16 == 42 )
  {
    if ( (_BYTE)v28 == 42 )
    {
      v163 = (__int64 *)*((_QWORD *)v186 - 6);
      if ( !v163 )
        goto LABEL_141;
      v54 = (unsigned __int8 *)*((_QWORD *)v186 - 3);
      v96 = v54[16];
      if ( (_BYTE)v96 == 13 )
        goto LABEL_79;
      v97 = *(_QWORD *)v54;
      if ( *(_BYTE *)(*(_QWORD *)v54 + 8LL) != 16 || (unsigned __int8)v96 > 0x10u )
      {
LABEL_141:
        v77 = 0x80A800000000LL;
        if ( !_bittest64(&v77, v28) )
          goto LABEL_37;
        v78 = (unsigned __int8)v28 - 24;
        goto LABEL_143;
      }
    }
    else
    {
      if ( (_BYTE)v28 != 5 )
        goto LABEL_188;
      if ( *((_WORD *)v186 + 9) != 18 )
        goto LABEL_129;
      v132 = *((_DWORD *)v186 + 5) & 0xFFFFFFF;
      v97 = 4 * v132;
      v163 = *(__int64 **)&v186[-24 * v132];
      if ( !v163 )
        goto LABEL_129;
      v96 = 1 - v132;
      v54 = *(unsigned __int8 **)&v186[24 * (1 - v132)];
      if ( v54[16] == 13 )
        goto LABEL_79;
      if ( *(_BYTE *)(*(_QWORD *)v54 + 8LL) != 16 )
      {
LABEL_129:
        v73 = *((unsigned __int16 *)v186 + 9);
        if ( (unsigned __int16)v73 > 0x17u )
          goto LABEL_37;
        v74 = &loc_80A800;
        if ( !_bittest64((const __int64 *)&v74, v73) )
          goto LABEL_216;
        LODWORD(v73) = (unsigned __int16)v73;
        v28 = 5;
        goto LABEL_132;
      }
    }
    v175 = v27;
    v98 = sub_15A1020(v54, a2, v96, v97);
    v27 = v175;
    if ( !v98 || (v56 = v98 + 24, *(_BYTE *)(v98 + 16) != 13) )
    {
      v28 = (unsigned __int8)v186[16];
      goto LABEL_188;
    }
    goto LABEL_80;
  }
  if ( (_BYTE)v28 == 41 )
  {
    v163 = (__int64 *)*((_QWORD *)v186 - 6);
    if ( v163 )
    {
      v54 = (unsigned __int8 *)*((_QWORD *)v186 - 3);
      v55 = v54[16];
      if ( (_BYTE)v55 == 13 )
      {
LABEL_79:
        v56 = (__int64)(v54 + 24);
        goto LABEL_80;
      }
      v30 = *(_QWORD *)v54;
      if ( *(_BYTE *)(*(_QWORD *)v54 + 8LL) == 16 && (unsigned __int8)v55 <= 0x10u )
        goto LABEL_224;
    }
LABEL_110:
    v63 = 0x80A800000000LL;
    if ( !_bittest64(&v63, v28) )
      goto LABEL_34;
    v32 = (unsigned __int8)v28 - 24;
LABEL_20:
    if ( v32 == 15 && (v186[17] & 2) != 0 )
    {
      if ( (v186[23] & 0x40) != 0 )
      {
        v33 = (_BYTE *)*((_QWORD *)v186 - 1);
        v148 = *(__int64 **)v33;
        if ( !*(_QWORD *)v33 )
          goto LABEL_34;
      }
      else
      {
        v33 = &v186[-24 * (*((_DWORD *)v186 + 5) & 0xFFFFFFF)];
        v148 = *(__int64 **)v33;
        if ( !*(_QWORD *)v33 )
          goto LABEL_34;
      }
      v34 = (_BYTE *)*((_QWORD *)v33 + 3);
      v35 = (unsigned __int8)v34[16];
      if ( (_BYTE)v35 == 13 )
      {
LABEL_25:
        v36 = (__int64)(v34 + 24);
        goto LABEL_26;
      }
      if ( *(_BYTE *)(*(_QWORD *)v34 + 8LL) == 16 && (unsigned __int8)v35 <= 0x10u )
      {
        v180 = v27;
        v133 = sub_15A1020(v34, a2, v35, *(_QWORD *)v34);
        v27 = v180;
        if ( !v133 || (v36 = v133 + 24, *(_BYTE *)(v133 + 16) != 13) )
        {
LABEL_33:
          v28 = (unsigned __int8)v186[16];
          goto LABEL_34;
        }
        goto LABEL_26;
      }
    }
LABEL_34:
    if ( v16 != 42 )
    {
      if ( (unsigned __int8)v28 > 0x17u )
      {
        if ( (unsigned __int8)v28 > 0x2Fu )
          goto LABEL_37;
        v84 = 0x80A800000000LL;
        if ( !_bittest64(&v84, v28) )
          goto LABEL_37;
        v85 = (unsigned __int8)v28 - 24;
LABEL_160:
        if ( v85 != 23 || (v186[17] & 2) == 0 )
          goto LABEL_37;
        if ( (v186[23] & 0x40) != 0 )
        {
          v86 = (_BYTE *)*((_QWORD *)v186 - 1);
          v147 = *(__int64 **)v86;
          if ( !*(_QWORD *)v86 )
            goto LABEL_37;
        }
        else
        {
          v86 = &v186[-24 * (*((_DWORD *)v186 + 5) & 0xFFFFFFF)];
          v147 = *(__int64 **)v86;
          if ( !*(_QWORD *)v86 )
            goto LABEL_37;
        }
        v87 = (_BYTE *)*((_QWORD *)v86 + 3);
        v88 = v87[16];
        if ( v88 == 13 )
        {
          v89 = *((_DWORD *)v87 + 8);
          v82 = (__int64)(v87 + 24);
        }
        else
        {
          if ( *(_BYTE *)(*(_QWORD *)v87 + 8LL) != 16 )
            goto LABEL_37;
          if ( v88 > 0x10u )
            goto LABEL_37;
          v181 = v27;
          v136 = sub_15A1020(v87, a2, *(_QWORD *)v87, v84);
          v27 = v181;
          if ( !v136 || *(_BYTE *)(v136 + 16) != 13 )
            goto LABEL_37;
          v82 = v136 + 24;
          v89 = *(_DWORD *)(v136 + 32);
        }
        v188 = v89;
        v172 = v16 == 42;
        if ( v89 <= 0x40 )
          goto LABEL_167;
        goto LABEL_199;
      }
      if ( (_BYTE)v28 != 5 )
        goto LABEL_37;
      v29 = *((unsigned __int16 *)v186 + 9);
LABEL_158:
      if ( (unsigned __int16)v29 > 0x17u )
        goto LABEL_37;
      v84 = (__int64)&loc_80A800;
      v85 = (unsigned __int16)v29;
      if ( !_bittest64(&v84, v29) )
        goto LABEL_37;
      goto LABEL_160;
    }
    goto LABEL_139;
  }
  if ( (_BYTE)v28 != 5 )
    goto LABEL_107;
  if ( *((_WORD *)v186 + 9) != 17
    || (v109 = *((_DWORD *)v186 + 5) & 0xFFFFFFF, (v163 = *(__int64 **)&v186[-24 * v109]) == 0) )
  {
LABEL_17:
    v29 = *((unsigned __int16 *)v186 + 9);
    v30 = v29;
    if ( (unsigned __int16)v29 > 0x17u )
      goto LABEL_158;
    goto LABEL_18;
  }
  v55 = 1 - v109;
  v54 = *(unsigned __int8 **)&v186[24 * (1 - v109)];
  if ( v54[16] == 13 )
    goto LABEL_79;
  v30 = 17;
  if ( *(_BYTE *)(*(_QWORD *)v54 + 8LL) != 16 )
  {
LABEL_18:
    v31 = &loc_80A800;
    v32 = (unsigned __int16)v30;
    if ( !_bittest64((const __int64 *)&v31, v30) )
    {
      v29 = *((unsigned __int16 *)v186 + 9);
      goto LABEL_158;
    }
    v28 = 5;
    goto LABEL_20;
  }
LABEL_224:
  v177 = v27;
  v108 = sub_15A1020(v54, a2, v55, v30);
  v27 = v177;
  if ( v108 )
  {
    v56 = v108 + 24;
    if ( *(_BYTE *)(v108 + 16) == 13 )
    {
LABEL_80:
      v190 = *(_DWORD *)(v56 + 8);
      if ( v190 > 0x40 )
      {
        v155 = v56;
        v173 = v27;
        sub_16A4EF0((__int64)&v189, 0, v184);
        v56 = v155;
        v27 = v173;
      }
      else
      {
        v189 = 0;
      }
      a2 = v56;
      v154 = v27;
      if ( v16 == 42 )
        sub_16AA420((__int64)&v192, v56, v27, (bool *)&v187);
      else
        sub_16AA580((__int64)&v192, v56, v27, (bool *)&v187);
      v27 = v154;
      if ( v190 > 0x40 && v189 )
      {
        j_j___libc_free_0_0(v189);
        v27 = v154;
      }
      v189 = v192;
      v190 = v193;
      if ( !(_BYTE)v187 )
      {
        v194 = 257;
        v57 = sub_15A1070(v185, (__int64)&v189);
        v14 = sub_15FB440((unsigned int)(unsigned __int8)v13[16] - 24, v163, v57, (__int64)&v192, 0);
        if ( v190 <= 0x40 )
          return v14;
        goto LABEL_89;
      }
      if ( v193 > 0x40 && v192 )
      {
        v171 = v27;
        j_j___libc_free_0_0(v192);
        v27 = v171;
      }
    }
  }
  v28 = (unsigned __int8)v186[16];
LABEL_107:
  if ( v16 != 42 )
  {
    if ( (unsigned __int8)v28 > 0x17u )
    {
      if ( (unsigned __int8)v28 > 0x2Fu )
        goto LABEL_34;
      goto LABEL_110;
    }
    if ( (_BYTE)v28 != 5 )
      goto LABEL_34;
    goto LABEL_17;
  }
LABEL_188:
  if ( (unsigned __int8)v28 <= 0x17u )
  {
    if ( (_BYTE)v28 != 5 )
      goto LABEL_139;
    goto LABEL_129;
  }
  if ( (unsigned __int8)v28 > 0x2Fu )
    goto LABEL_139;
  v99 = 0x80A800000000LL;
  if ( !_bittest64(&v99, v28) )
    goto LABEL_139;
  LODWORD(v73) = (unsigned __int8)v28 - 24;
LABEL_132:
  if ( (_DWORD)v73 != 15 || (v186[17] & 4) == 0 )
    goto LABEL_139;
  if ( (v186[23] & 0x40) != 0 )
  {
    v75 = (_BYTE *)*((_QWORD *)v186 - 1);
    v148 = *(__int64 **)v75;
    if ( !*(_QWORD *)v75 )
      goto LABEL_139;
  }
  else
  {
    v75 = &v186[-24 * (*((_DWORD *)v186 + 5) & 0xFFFFFFF)];
    v148 = *(__int64 **)v75;
    if ( !*(_QWORD *)v75 )
      goto LABEL_139;
  }
  v34 = (_BYTE *)*((_QWORD *)v75 + 3);
  v76 = (unsigned __int8)v34[16];
  if ( (_BYTE)v76 == 13 )
    goto LABEL_25;
  if ( *(_BYTE *)(*(_QWORD *)v34 + 8LL) == 16 && (unsigned __int8)v76 <= 0x10u )
  {
    v183 = v27;
    v146 = sub_15A1020(v34, a2, v76, *(_QWORD *)v34);
    v27 = v183;
    if ( v146 )
    {
      v36 = v146 + 24;
      if ( *(_BYTE *)(v146 + 16) == 13 )
      {
LABEL_26:
        v37 = v16 == 42;
        v190 = *(_DWORD *)(v36 + 8);
        if ( v190 > 0x40 )
        {
          v151 = v27;
          v156 = v36;
          sub_16A4EF0((__int64)&v189, 0, v184);
          v27 = v151;
          v36 = v156;
          v37 = v16 == 42;
        }
        else
        {
          v189 = 0;
        }
        v150 = v37;
        v162 = v36;
        v168 = v27;
        if ( sub_177F430(v27, v36, (unsigned __int64 *)&v189, v37) )
        {
          v194 = 257;
          v102 = sub_15A1070(v185, (__int64)&v189);
          v14 = sub_15FB440((unsigned int)(unsigned __int8)v13[16] - 24, v148, v102, (__int64)&v192, 0);
          v103 = sub_15F23D0((__int64)v13);
          sub_15F2350(v14, v103);
        }
        else
        {
          a2 = v168;
          v38 = sub_177F430(v162, v168, (unsigned __int64 *)&v189, v150);
          v27 = v168;
          if ( !v38 )
          {
            if ( v190 > 0x40 && v189 )
            {
              j_j___libc_free_0_0(v189);
              v27 = v168;
            }
            goto LABEL_33;
          }
          v194 = 257;
          v106 = sub_15A1070(v185, (__int64)&v189);
          v107 = 0;
          v14 = sub_15FB440(15, v148, v106, (__int64)&v192, 0);
          if ( v16 != 42 )
            v107 = (v186[17] & 2) != 0;
          sub_15F2310(v14, v107);
          sub_15F2330(v14, (v186[17] & 4) != 0);
        }
        if ( v190 <= 0x40 )
          return v14;
LABEL_89:
        v58 = v189;
        if ( !v189 )
          return v14;
        goto LABEL_90;
      }
    }
    v28 = (unsigned __int8)v186[16];
  }
LABEL_139:
  if ( (unsigned __int8)v28 > 0x17u )
  {
    if ( (unsigned __int8)v28 > 0x2Fu )
      goto LABEL_37;
    goto LABEL_141;
  }
  if ( (_BYTE)v28 != 5 )
    goto LABEL_37;
  v73 = *((unsigned __int16 *)v186 + 9);
LABEL_216:
  if ( (unsigned __int16)v73 > 0x17u )
    goto LABEL_37;
  v77 = (__int64)&loc_80A800;
  v78 = (unsigned __int16)v73;
  if ( !_bittest64(&v77, v73) )
    goto LABEL_37;
LABEL_143:
  if ( v78 != 23 || (v186[17] & 4) == 0 )
    goto LABEL_37;
  if ( (v186[23] & 0x40) != 0 )
  {
    v79 = (_BYTE *)*((_QWORD *)v186 - 1);
    v147 = *(__int64 **)v79;
    if ( !*(_QWORD *)v79 )
      goto LABEL_37;
  }
  else
  {
    v79 = &v186[-24 * (*((_DWORD *)v186 + 5) & 0xFFFFFFF)];
    v147 = *(__int64 **)v79;
    if ( !*(_QWORD *)v79 )
      goto LABEL_37;
  }
  v80 = (_BYTE *)*((_QWORD *)v79 + 3);
  v81 = v80[16];
  v82 = (__int64)(v80 + 24);
  if ( v81 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v80 + 8LL) != 16 )
      goto LABEL_37;
    if ( v81 > 0x10u )
      goto LABEL_37;
    v182 = v27;
    v141 = sub_15A1020(v80, a2, *(_QWORD *)v80, v77);
    v27 = v182;
    if ( !v141 || *(_BYTE *)(v141 + 16) != 13 )
      goto LABEL_37;
    v82 = v141 + 24;
  }
  v83 = *(_DWORD *)(v82 + 8);
  if ( v83 <= 0x40 )
  {
    if ( *(_QWORD *)v82 == v83 - 1 )
      goto LABEL_37;
    v188 = *(_DWORD *)(v82 + 8);
    v172 = v16 == 42;
LABEL_167:
    v187 = 0;
    goto LABEL_168;
  }
  v159 = v27;
  v166 = *(_DWORD *)(v82 + 8);
  v176 = (_QWORD **)v82;
  v100 = sub_16A57B0(v82);
  v82 = (__int64)v176;
  v27 = v159;
  if ( v166 - v100 <= 0x40 && **v176 == v166 - 1 )
    goto LABEL_37;
  v188 = v166;
  v172 = v16 == 42;
LABEL_199:
  v153 = v27;
  v160 = v82;
  sub_16A4EF0((__int64)&v187, 0, v172);
  v27 = v153;
  v82 = v160;
LABEL_168:
  if ( *(_DWORD *)(v82 + 8) <= 0x40u )
  {
    v101 = *(_QWORD *)v82;
    v190 = *(_DWORD *)(v82 + 8);
    v189 = 0;
    v94 = 1LL << v101;
  }
  else
  {
    v157 = *(_DWORD *)(v82 + 8);
    v164 = v27;
    v149 = (_QWORD **)v82;
    v90 = sub_16A57B0(v82);
    v91 = -1;
    v92 = v157 - v90;
    v93 = 0x8000000000000000LL;
    if ( v92 <= 0x40 )
    {
      v91 = **v149;
      v93 = 1LL << v91;
    }
    v190 = v157;
    v152 = v93;
    v158 = v91;
    sub_16A4EF0((__int64)&v189, 0, 0);
    v94 = v152;
    v27 = v164;
    if ( v190 > 0x40 )
    {
      *(_QWORD *)(v189 + 8LL * (v158 >> 6)) |= v152;
      goto LABEL_173;
    }
  }
  v189 |= v94;
LABEL_173:
  v165 = v27;
  if ( sub_177F430(v27, (__int64)&v189, (unsigned __int64 *)&v187, v172) )
  {
    v194 = 257;
    v104 = sub_15A1070(v185, (__int64)&v187);
    v14 = sub_15FB440((unsigned int)(unsigned __int8)v13[16] - 24, v147, v104, (__int64)&v192, 0);
    v105 = sub_15F23D0((__int64)v13);
    sub_15F2350(v14, v105);
    goto LABEL_208;
  }
  a2 = v165;
  v95 = sub_177F430((__int64)&v189, v165, (unsigned __int64 *)&v187, v172);
  v27 = v165;
  if ( v95 )
  {
    v194 = 257;
    v116 = sub_15A1070(v185, (__int64)&v187);
    v117 = 0;
    v14 = sub_15FB440(15, v147, v116, (__int64)&v192, 0);
    if ( v16 != 42 )
      v117 = (v186[17] & 2) != 0;
    sub_15F2310(v14, v117);
    sub_15F2330(v14, (v186[17] & 4) != 0);
LABEL_208:
    if ( v190 > 0x40 && v189 )
      j_j___libc_free_0_0(v189);
    if ( v188 <= 0x40 )
      return v14;
    v58 = v187;
    if ( !v187 )
      return v14;
LABEL_90:
    j_j___libc_free_0_0(v58);
    return v14;
  }
  if ( v190 > 0x40 && v189 )
  {
    j_j___libc_free_0_0(v189);
    v27 = v165;
  }
  if ( v188 > 0x40 && v187 )
  {
    v174 = v27;
    j_j___libc_free_0_0(v187);
    v27 = v174;
  }
LABEL_37:
  if ( *(_DWORD *)(v27 + 8) <= 0x40u )
  {
    v39 = *(_QWORD *)v27 == 0;
  }
  else
  {
    v169 = *(_DWORD *)(v27 + 8);
    v39 = v169 == (unsigned int)sub_16A57B0(v27);
  }
  if ( !v39 )
  {
    a2 = (__int64)v13;
    v40 = sub_1713A90(a1, v13, a5, a6, a7, a8, v24, v25, a11, a12);
    if ( v40 )
      return v40;
  }
LABEL_43:
  v41 = v186[16];
  if ( v41 == 13 )
  {
    if ( *((_DWORD *)v186 + 8) > 0x40u )
    {
      v170 = *((_DWORD *)v186 + 8);
      v42 = (__int64)v186;
LABEL_46:
      v43 = v170 - 1 == (unsigned int)sub_16A57B0(v42 + 24);
      goto LABEL_47;
    }
    v43 = *((_QWORD *)v186 + 3) == 1;
    goto LABEL_47;
  }
  if ( *(_BYTE *)(*(_QWORD *)v186 + 8LL) != 16 || v41 > 0x10u )
  {
LABEL_56:
    if ( (unsigned __int8)sub_17AD890(a1, v13) )
      return v14;
    v48 = (unsigned __int8)v186[16];
    if ( (_BYTE)v48 == 37 )
    {
      v59 = (__int64 *)*((_QWORD *)v186 - 6);
      if ( !v59 || (v60 = *((_QWORD *)v186 - 3)) == 0 )
      {
        if ( v16 != 42 )
          goto LABEL_101;
LABEL_279:
        v127 = 0x80A800000000LL;
        if ( !_bittest64(&v127, v48) )
          goto LABEL_68;
        v128 = (unsigned __int8)v48 - 24;
LABEL_281:
        if ( v128 == 23 && (v186[17] & 4) != 0 )
        {
          v129 = (_QWORD *)sub_13CF970((__int64)v186);
          if ( v15 == *v129 )
          {
            v130 = v129[3];
            if ( v130 )
            {
              v194 = 257;
              v131 = (__int64 *)sub_15A0680(v185, 1, 0);
              v14 = sub_15FB440(23, v131, v130, (__int64)&v192, 0);
              sub_15F2330(v14, 1);
              return v14;
            }
          }
        }
LABEL_68:
        v53 = *(_BYTE *)(v15 + 16);
        if ( v53 == 39 )
        {
          v118 = *(_BYTE **)(v15 - 48);
          v119 = *(_QWORD *)(v15 - 24);
          if ( v186 != v118 || !v118 )
          {
            if ( !v119 || v186 != (_BYTE *)v119 )
              return 0;
            v119 = *(_QWORD *)(v15 - 48);
          }
        }
        else
        {
          if ( v53 != 5 || *(_WORD *)(v15 + 18) != 15 )
            return 0;
          v137 = *(_BYTE **)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
          v119 = *(_QWORD *)(v15 + 24 * (1LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
          if ( !v137 || v186 != v137 )
          {
            if ( v186 != (_BYTE *)v119 || !v119 || !v137 )
              return 0;
            v119 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
LABEL_255:
            if ( (*(_BYTE *)(v15 + 17) & 4) != 0 && v16 == 42 || v16 != 42 && (*(_BYTE *)(v15 + 17) & 2) != 0 )
            {
              v120 = sub_15A0680(v185, 1, 0);
              if ( *((_QWORD *)v13 - 6) )
              {
                v121 = *((_QWORD *)v13 - 5);
                v122 = *((_QWORD *)v13 - 4) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v122 = v121;
                if ( v121 )
                  *(_QWORD *)(v121 + 16) = *(_QWORD *)(v121 + 16) & 3LL | v122;
              }
              *((_QWORD *)v13 - 6) = v120;
              if ( v120 )
              {
                v123 = *(_QWORD *)(v120 + 8);
                *((_QWORD *)v13 - 5) = v123;
                if ( v123 )
                  *(_QWORD *)(v123 + 16) = (unsigned __int64)(v13 - 40) | *(_QWORD *)(v123 + 16) & 3LL;
                *((_QWORD *)v13 - 4) = (v120 + 8) | *((_QWORD *)v13 - 4) & 3LL;
                *(_QWORD *)(v120 + 8) = v13 - 48;
              }
              if ( *((_QWORD *)v13 - 3) )
              {
                v124 = *((_QWORD *)v13 - 2);
                v125 = *((_QWORD *)v13 - 1) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v125 = v124;
                if ( v124 )
                  *(_QWORD *)(v124 + 16) = *(_QWORD *)(v124 + 16) & 3LL | v125;
              }
              *((_QWORD *)v13 - 3) = v119;
              v126 = *(_QWORD *)(v119 + 8);
              *((_QWORD *)v13 - 2) = v126;
              if ( v126 )
                *(_QWORD *)(v126 + 16) = (unsigned __int64)(v13 - 16) | *(_QWORD *)(v126 + 16) & 3LL;
              *((_QWORD *)v13 - 1) = *((_QWORD *)v13 - 1) & 3LL | (v119 + 8);
              *(_QWORD *)(v119 + 8) = v13 - 24;
              return v14;
            }
            return 0;
          }
        }
        if ( !v119 )
          return 0;
        goto LABEL_255;
      }
    }
    else
    {
      if ( (_BYTE)v48 != 5 )
        goto LABEL_98;
      if ( *((_WORD *)v186 + 9) != 13
        || (v59 = *(__int64 **)&v186[-24 * (*((_DWORD *)v186 + 5) & 0xFFFFFFF)]) == 0
        || (v60 = *(_QWORD *)&v186[24 * (1LL - (*((_DWORD *)v186 + 5) & 0xFFFFFFF))]) == 0 )
      {
        if ( v16 != 42 )
          goto LABEL_61;
        goto LABEL_328;
      }
    }
    v61 = *(_BYTE *)(v60 + 16);
    if ( v16 != 42 )
    {
      if ( v61 == 44 )
      {
        if ( *(__int64 **)(v60 - 48) != v59 || (v134 = *(_QWORD *)(v60 - 24), v15 != v134) )
        {
LABEL_98:
          if ( v16 != 42 )
          {
            if ( (unsigned __int8)v48 <= 0x17u )
            {
              if ( (_BYTE)v48 != 5 )
                goto LABEL_68;
LABEL_61:
              v49 = *((unsigned __int16 *)v186 + 9);
              if ( (unsigned __int16)v49 > 0x17u )
                goto LABEL_68;
              v50 = &loc_80A800;
              v51 = (unsigned __int16)v49;
              if ( !_bittest64((const __int64 *)&v50, v49) )
                goto LABEL_68;
              goto LABEL_63;
            }
            if ( (unsigned __int8)v48 > 0x2Fu )
              goto LABEL_68;
LABEL_101:
            v62 = 0x80A800000000LL;
            if ( !_bittest64(&v62, v48) )
              goto LABEL_68;
            v51 = (unsigned __int8)v48 - 24;
LABEL_63:
            if ( v51 == 23 && (v186[17] & 2) != 0 )
            {
              v52 = (v186[23] & 0x40) != 0
                  ? (_QWORD *)*((_QWORD *)v186 - 1)
                  : &v186[-24 * (*((_DWORD *)v186 + 5) & 0xFFFFFFF)];
              if ( v15 == *v52 )
              {
                v142 = v52[3];
                if ( v142 )
                {
                  v194 = 257;
                  v143 = (__int64 *)sub_15A0680(v185, 1, 0);
                  v14 = sub_15FB440(23, v143, v142, (__int64)&v192, 0);
                  sub_15F2310(v14, 1);
                  return v14;
                }
              }
            }
            goto LABEL_68;
          }
          goto LABEL_277;
        }
      }
      else
      {
        if ( v61 != 5 )
          goto LABEL_98;
        if ( *(_WORD *)(v60 + 18) != 20 )
          goto LABEL_98;
        v140 = *(_DWORD *)(v60 + 20) & 0xFFFFFFF;
        if ( *(__int64 **)(v60 - 24 * v140) != v59 )
          goto LABEL_98;
        v134 = *(_QWORD *)(v60 + 24 * (1 - v140));
        if ( v15 != v134 )
          goto LABEL_98;
      }
      if ( !v134 )
        goto LABEL_98;
LABEL_305:
      v135 = (unsigned __int8)v13[16];
      v194 = 257;
      return sub_15FB440(v135 - 24, v59, v15, (__int64)&v192, 0);
    }
    if ( v61 == 45 )
    {
      if ( *(__int64 **)(v60 - 48) == v59 )
      {
        v138 = *(_QWORD *)(v60 - 24);
        if ( v15 == v138 )
        {
          if ( v138 )
            goto LABEL_305;
        }
      }
    }
    else if ( v61 == 5 && *(_WORD *)(v60 + 18) == 21 )
    {
      v144 = *(_DWORD *)(v60 + 20) & 0xFFFFFFF;
      if ( *(__int64 **)(v60 - 24 * v144) == v59 )
      {
        v145 = *(_QWORD *)(v60 + 24 * (1 - v144));
        if ( v145 )
        {
          if ( v15 == v145 )
            goto LABEL_305;
        }
      }
    }
LABEL_277:
    if ( (unsigned __int8)v48 <= 0x17u )
    {
      if ( (_BYTE)v48 != 5 )
        goto LABEL_68;
LABEL_328:
      v139 = *((_WORD *)v186 + 9);
      if ( v139 > 0x17u )
        goto LABEL_68;
      v128 = v139;
      if ( (((unsigned __int64)&loc_80A800 >> v139) & 1) == 0 )
        goto LABEL_68;
      goto LABEL_281;
    }
    if ( (unsigned __int8)v48 > 0x2Fu )
      goto LABEL_68;
    goto LABEL_279;
  }
  v42 = sub_15A1020(v186, a2, *(_QWORD *)v186, (__int64)v186);
  if ( !v42 || *(_BYTE *)(v42 + 16) != 13 )
  {
    v110 = 0;
    v167 = *(_QWORD *)(*(_QWORD *)v186 + 32LL);
    if ( !v167 )
      goto LABEL_48;
    while ( 1 )
    {
      v178 = v110;
      v111 = sub_15A0A60((__int64)v186, v110);
      if ( !v111 )
        goto LABEL_56;
      v112 = *(_BYTE *)(v111 + 16);
      v113 = v178;
      if ( v112 != 9 )
      {
        if ( v112 != 13 )
          goto LABEL_56;
        if ( *(_DWORD *)(v111 + 32) <= 0x40u )
        {
          v115 = *(_QWORD *)(v111 + 24) == 1;
        }
        else
        {
          v161 = v178;
          v179 = *(_DWORD *)(v111 + 32);
          v114 = sub_16A57B0(v111 + 24);
          v113 = v161;
          v115 = v179 - 1 == v114;
        }
        if ( !v115 )
          goto LABEL_56;
      }
      v110 = v113 + 1;
      if ( v167 == v110 )
        goto LABEL_48;
    }
  }
  if ( *(_DWORD *)(v42 + 32) > 0x40u )
  {
    v170 = *(_DWORD *)(v42 + 32);
    goto LABEL_46;
  }
  v43 = *(_QWORD *)(v42 + 24) == 1;
LABEL_47:
  if ( !v43 )
    goto LABEL_56;
LABEL_48:
  v44 = a1[1];
  if ( v16 == 42 )
  {
    v194 = 257;
    if ( *(_BYTE *)(v15 + 16) > 0x10u || v186[16] > 0x10u )
    {
      v67 = sub_170A2B0(v44, 11, (__int64 *)v15, (__int64)v186, &v192, 0, 0);
    }
    else
    {
      v65 = sub_15A2B30((__int64 *)v15, (__int64)v186, 0, 0, *(double *)a5.m128_u64, a6, a7);
      v66 = sub_14DBA30(v65, *(_QWORD *)(v44 + 96), 0);
      if ( v66 )
        v65 = v66;
      v67 = (unsigned __int8 *)v65;
    }
    v68 = a1[1];
    v194 = 257;
    v69 = sub_15A0680(v185, 3, 0);
    if ( v67[16] > 0x10u || *(_BYTE *)(v69 + 16) > 0x10u )
    {
      v70 = (__int64)sub_177F2B0(v68, 36, (__int64)v67, v69, &v192);
    }
    else
    {
      v70 = sub_15A37B0(0x24u, v67, (_QWORD *)v69, 0);
      v71 = sub_14DBA30(v70, *(_QWORD *)(v68 + 96), 0);
      if ( v71 )
        v70 = v71;
    }
    v194 = 257;
    v72 = sub_15A0680(v185, 0, 0);
    return sub_14EDD70(v70, (_QWORD *)v15, v72, (__int64)&v192, 0, 0);
  }
  v191 = 257;
  if ( *(_BYTE *)(v15 + 16) > 0x10u || v186[16] > 0x10u )
  {
    v45 = (__int64)sub_177F2B0(v44, 32, v15, (__int64)v186, &v189);
  }
  else
  {
    v45 = sub_15A37B0(0x20u, (_QWORD *)v15, v186, 0);
    v46 = sub_14DBA30(v45, *(_QWORD *)(v44 + 96), 0);
    if ( v46 )
      v45 = v46;
  }
  v194 = 257;
  v47 = sub_1648A60(56, 1u);
  v14 = (__int64)v47;
  if ( v47 )
    sub_15FC690((__int64)v47, v45, v185, (__int64)&v192, 0);
  return v14;
}
