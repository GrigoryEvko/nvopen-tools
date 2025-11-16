// Function: sub_175B5C0
// Address: 0x175b5c0
//
_QWORD *__fastcall sub_175B5C0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  _QWORD *v5; // r15
  int v6; // r10d
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v12; // rcx
  int v13; // esi
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 ***v16; // rax
  __int64 **v17; // rdi
  __int64 v18; // r12
  _QWORD *v19; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  bool v29; // zf
  __int64 v30; // rsi
  __int64 v31; // rbx
  int v32; // edx
  __int64 v33; // r9
  __int16 v34; // r10
  __int64 *v35; // rax
  __int64 v36; // r13
  unsigned __int8 *v37; // r13
  __int64 v38; // rcx
  int v39; // eax
  int v40; // eax
  unsigned __int8 **v41; // rax
  unsigned __int8 *v42; // rax
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rsi
  char v46; // al
  __int64 v47; // r15
  __int64 v48; // rax
  char v49; // al
  __int64 v50; // rax
  unsigned __int64 v51; // rcx
  unsigned int v52; // r8d
  unsigned __int64 v53; // rsi
  unsigned int v54; // r15d
  bool v55; // al
  __int16 v56; // bx
  const char *v57; // rax
  __int64 *v58; // rdx
  unsigned __int8 *v59; // rax
  __int64 v60; // r9
  __int64 v61; // r12
  __int64 v62; // rax
  _QWORD *v63; // rax
  char v64; // al
  __int64 v65; // rax
  __int64 v66; // rax
  int v67; // eax
  int v68; // eax
  unsigned __int8 **v69; // rax
  unsigned __int8 *v70; // rax
  int v71; // r15d
  bool v72; // al
  int v73; // r10d
  __int64 v74; // r9
  __int64 v75; // rsi
  char v76; // cl
  __int64 v77; // r13
  __int64 v78; // rax
  char v79; // al
  __int64 v80; // rax
  unsigned int v81; // eax
  unsigned int v82; // r12d
  unsigned int v83; // eax
  __int64 v84; // r12
  __int64 v85; // rax
  unsigned __int8 *v86; // r12
  __int64 v87; // rax
  __int64 v88; // rbx
  _QWORD *v89; // rax
  char v90; // al
  __int64 v91; // rax
  char v92; // al
  __int64 v93; // rax
  int v94; // eax
  unsigned __int64 v95; // rax
  __int64 v96; // rax
  char v97; // dl
  unsigned __int8 **v98; // rbx
  __int64 v99; // rax
  __int64 v100; // rax
  _QWORD *v101; // rax
  char v102; // al
  unsigned __int8 *v103; // r13
  unsigned __int8 *v104; // r8
  unsigned __int8 *v105; // rsi
  __int64 v106; // rdi
  unsigned __int8 *v107; // rax
  unsigned __int8 *v108; // rsi
  __int64 v109; // rdx
  __int64 v110; // rcx
  __int64 v111; // rax
  char v112; // al
  __int64 v113; // rax
  unsigned __int64 v114; // r15
  unsigned int v115; // r8d
  unsigned __int64 v116; // rcx
  int v117; // r15d
  bool v118; // al
  __int64 v119; // rbx
  const char *v120; // rax
  __int64 *v121; // rdx
  unsigned __int8 *v122; // rax
  char v123; // r8
  __int64 v124; // r9
  unsigned __int8 *v125; // rbx
  __int16 v126; // r10
  unsigned int v127; // eax
  unsigned __int64 v128; // rax
  __int64 v129; // r15
  __int64 v130; // rax
  __int64 *v131; // rdx
  __int64 v132; // rax
  unsigned __int8 *v133; // r12
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // rax
  __int64 v137; // r14
  __int64 v138; // rcx
  unsigned __int8 *v139; // rdx
  int v140; // eax
  unsigned __int64 v141; // rax
  __int64 v142; // rax
  __int64 v143; // rcx
  int v144; // eax
  __int64 v145; // rdi
  __int64 **v146; // rcx
  unsigned __int8 *v147; // rax
  unsigned __int8 *v148; // r12
  char v149; // [rsp+8h] [rbp-D8h]
  __int64 v150; // [rsp+8h] [rbp-D8h]
  unsigned int v151; // [rsp+18h] [rbp-C8h]
  __int64 v152; // [rsp+18h] [rbp-C8h]
  __int64 v153; // [rsp+18h] [rbp-C8h]
  unsigned int v154; // [rsp+18h] [rbp-C8h]
  __int64 v155; // [rsp+18h] [rbp-C8h]
  __int16 v156; // [rsp+18h] [rbp-C8h]
  int v157; // [rsp+18h] [rbp-C8h]
  __int64 v158; // [rsp+18h] [rbp-C8h]
  __int16 v159; // [rsp+20h] [rbp-C0h]
  __int64 v160; // [rsp+20h] [rbp-C0h]
  __int64 v161; // [rsp+20h] [rbp-C0h]
  int v162; // [rsp+20h] [rbp-C0h]
  __int64 v163; // [rsp+20h] [rbp-C0h]
  __int64 v164; // [rsp+20h] [rbp-C0h]
  unsigned int v165; // [rsp+20h] [rbp-C0h]
  __int16 v166; // [rsp+20h] [rbp-C0h]
  __int16 v167; // [rsp+20h] [rbp-C0h]
  unsigned int v168; // [rsp+20h] [rbp-C0h]
  __int64 ***v169; // [rsp+20h] [rbp-C0h]
  __int16 v170; // [rsp+20h] [rbp-C0h]
  __int64 v171; // [rsp+20h] [rbp-C0h]
  __int16 v172; // [rsp+28h] [rbp-B8h]
  __int64 v173; // [rsp+28h] [rbp-B8h]
  __int16 v174; // [rsp+28h] [rbp-B8h]
  __int64 v175; // [rsp+28h] [rbp-B8h]
  __int64 v176; // [rsp+28h] [rbp-B8h]
  int v177; // [rsp+28h] [rbp-B8h]
  __int16 v178; // [rsp+28h] [rbp-B8h]
  int v179; // [rsp+28h] [rbp-B8h]
  __int16 v180; // [rsp+28h] [rbp-B8h]
  __int64 v181; // [rsp+28h] [rbp-B8h]
  __int64 v182; // [rsp+28h] [rbp-B8h]
  __int64 v183; // [rsp+28h] [rbp-B8h]
  __int16 v184; // [rsp+28h] [rbp-B8h]
  __int16 v185; // [rsp+28h] [rbp-B8h]
  __int16 v186; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v187; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int8 *v188; // [rsp+38h] [rbp-A8h] BYREF
  __int64 ***v189; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v190; // [rsp+48h] [rbp-98h] BYREF
  __int64 v191; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v192; // [rsp+58h] [rbp-88h] BYREF
  unsigned __int64 v193; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v194; // [rsp+68h] [rbp-78h]
  unsigned __int64 v195; // [rsp+70h] [rbp-70h] BYREF
  __int64 *v196; // [rsp+78h] [rbp-68h]
  __int16 v197; // [rsp+80h] [rbp-60h]
  __int64 *v198; // [rsp+90h] [rbp-50h] BYREF
  char *v199; // [rsp+98h] [rbp-48h]
  __int16 v200; // [rsp+A0h] [rbp-40h]

  v5 = 0;
  v6 = *(_WORD *)(a2 + 18) & 0x7FFF;
  v7 = (unsigned int)(v6 - 32);
  if ( (unsigned int)v7 > 1 )
    return v5;
  v8 = *(_QWORD *)(a2 - 48);
  v9 = *(_QWORD *)(a2 - 24);
  v10 = a1;
  v11 = a2;
  v12 = *(unsigned __int8 *)(v8 + 16);
  if ( (_BYTE)v12 == 52 )
  {
    v16 = *(__int64 ****)(v8 - 48);
    if ( !v16 )
      goto LABEL_5;
    v187 = *(unsigned __int8 **)(v8 - 48);
    v15 = *(_QWORD *)(v8 - 24);
    if ( !v15 )
      goto LABEL_5;
LABEL_15:
    v188 = (unsigned __int8 *)v15;
    if ( (__int64 ***)v9 == v16 )
    {
LABEL_18:
      v172 = v6;
      v17 = *v16;
      goto LABEL_19;
    }
    if ( v9 == v15 )
    {
      v15 = (__int64)v16;
      goto LABEL_18;
    }
    v7 = *(unsigned __int8 *)(v9 + 16);
    if ( (_BYTE)v7 == 52 )
    {
      a2 = *(_QWORD *)(v9 - 48);
      if ( !a2 )
        goto LABEL_32;
      v189 = *(__int64 ****)(v9 - 48);
      v21 = *(_QWORD *)(v9 - 24);
      if ( !v21 )
        goto LABEL_29;
    }
    else
    {
      if ( (_BYTE)v7 != 5 || *(_WORD *)(v9 + 18) != 28 )
        goto LABEL_32;
      v13 = *(_DWORD *)(v9 + 20);
      if ( !*(_QWORD *)(v9 - 24LL * (v13 & 0xFFFFFFF))
        || (v189 = *(__int64 ****)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)),
            (v21 = *(_QWORD *)(v9 + 24 * (1LL - (v13 & 0xFFFFFFF)))) == 0) )
      {
LABEL_10:
        v14 = v13 & 0xFFFFFFF;
        a2 = *(_QWORD *)(v9 - 24 * v14);
        if ( !a2 )
          goto LABEL_32;
        v187 = *(unsigned __int8 **)(v9 - 24 * v14);
        v7 = 1 - v14;
        v15 = *(_QWORD *)(v9 + 24 * (1 - v14));
        if ( !v15 )
          goto LABEL_32;
        goto LABEL_30;
      }
    }
    v29 = *(_BYTE *)(v15 + 16) == 13;
    v190 = v21;
    if ( v29 && *(_BYTE *)(v190 + 16) == 13 )
    {
      v30 = *(_QWORD *)(v9 + 8);
      if ( v30 )
      {
        if ( !*(_QWORD *)(v30 + 8) )
        {
          v31 = *(_QWORD *)(v10 + 8);
          v159 = v6;
          v173 = v10;
          sub_13A38D0((__int64)&v195, v15 + 24);
          v32 = (int)v196;
          v33 = v173;
          v34 = v159;
          if ( (unsigned int)v196 > 0x40 )
          {
            v171 = v173;
            v186 = v34;
            sub_16A8F00((__int64 *)&v195, (__int64 *)(v190 + 24));
            v32 = (int)v196;
            v35 = (__int64 *)v195;
            v33 = v171;
            v34 = v186;
          }
          else
          {
            v35 = (__int64 *)(*(_QWORD *)(v190 + 24) ^ v195);
            v195 = (unsigned __int64)v35;
          }
          LODWORD(v199) = v32;
          v198 = v35;
          LODWORD(v196) = 0;
          v174 = v34;
          v160 = v33;
          v36 = sub_159C0E0(*(__int64 **)(v31 + 24), (__int64)&v198);
          sub_135E100((__int64 *)&v198);
          sub_135E100((__int64 *)&v195);
          v200 = 257;
          v37 = sub_172B670(*(_QWORD *)(v160 + 8), (__int64)v189, v36, (__int64 *)&v198, a3, a4, a5);
          v200 = 257;
          v5 = sub_1648A60(56, 2u);
          if ( !v5 )
            return v5;
          v38 = (__int64)v37;
          goto LABEL_76;
        }
      }
    }
    if ( v189 == v16 )
    {
      v185 = v6;
      v200 = 257;
      v5 = sub_1648A60(56, 2u);
      if ( !v5 )
        return v5;
      v143 = v190;
    }
    else
    {
      if ( (__int64 ***)v190 != v16 )
      {
        if ( v189 == (__int64 ***)v15 )
        {
          v174 = v6;
          v200 = 257;
          v5 = sub_1648A60(56, 2u);
          if ( !v5 )
            return v5;
          v38 = v190;
        }
        else
        {
          if ( v190 != v15 )
            goto LABEL_6;
          v174 = v6;
          v200 = 257;
          v5 = sub_1648A60(56, 2u);
          if ( !v5 )
            return v5;
          v38 = (__int64)v189;
        }
LABEL_76:
        sub_17582E0((__int64)v5, v174, (__int64)v187, v38, (__int64)&v198);
        return v5;
      }
      v185 = v6;
      v200 = 257;
      v5 = sub_1648A60(56, 2u);
      if ( !v5 )
        return v5;
      v143 = (__int64)v189;
    }
    sub_17582E0((__int64)v5, v185, (__int64)v188, v143, (__int64)&v198);
    return v5;
  }
  if ( (_BYTE)v12 == 5 && *(_WORD *)(v8 + 18) == 28 )
  {
    v27 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
    v16 = *(__int64 ****)(v8 - 24 * v27);
    if ( v16 )
    {
      v187 = *(unsigned __int8 **)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
      a2 = 1 - v27;
      v7 = 3 * (1 - v27);
      v15 = *(_QWORD *)(v8 + 8 * v7);
      if ( v15 )
        goto LABEL_15;
    }
  }
LABEL_5:
  v7 = *(unsigned __int8 *)(v9 + 16);
LABEL_6:
  if ( (_BYTE)v7 != 52 )
  {
    if ( (_BYTE)v7 == 5 && *(_WORD *)(v9 + 18) == 28 )
    {
      v13 = *(_DWORD *)(v9 + 20);
      goto LABEL_10;
    }
LABEL_32:
    v195 = (unsigned __int64)&v187;
    v196 = (__int64 *)&v188;
    v22 = *(_QWORD *)(v8 + 8);
    if ( !v22 )
      goto LABEL_36;
    if ( *(_QWORD *)(v22 + 8) )
    {
LABEL_34:
      if ( v22 && !*(_QWORD *)(v22 + 8) )
      {
        v39 = *(unsigned __int8 *)(v8 + 16);
        if ( (unsigned __int8)v39 > 0x17u )
        {
          v40 = v39 - 24;
        }
        else
        {
          if ( (_BYTE)v39 != 5 )
            goto LABEL_36;
          v40 = *(unsigned __int16 *)(v8 + 18);
        }
        if ( v40 == 37 )
        {
          v41 = (*(_BYTE *)(v8 + 23) & 0x40) != 0
              ? *(unsigned __int8 ***)(v8 - 8)
              : (unsigned __int8 **)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
          v42 = *v41;
          if ( v42 )
          {
            v187 = v42;
            v43 = *(_BYTE *)(v9 + 16);
            if ( v43 == 50 )
            {
              if ( *(_QWORD *)(v9 - 48) )
              {
                v188 = *(unsigned __int8 **)(v9 - 48);
                v45 = *(_QWORD *)(v9 - 24);
                if ( *(_BYTE *)(v45 + 16) == 13 )
                  goto LABEL_89;
              }
            }
            else if ( v43 == 5 && *(_WORD *)(v9 + 18) == 26 )
            {
              v44 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
              if ( *(_QWORD *)(v9 - 24 * v44) )
              {
                v188 = *(unsigned __int8 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
                v45 = *(_QWORD *)(v9 + 24 * (1 - v44));
                if ( *(_BYTE *)(v45 + 16) == 13 )
                {
LABEL_89:
                  v191 = v45;
                  goto LABEL_125;
                }
              }
            }
          }
        }
      }
LABEL_36:
      v23 = *(_QWORD *)(v9 + 8);
      if ( !v23 || *(_QWORD *)(v23 + 8) )
        goto LABEL_38;
      v64 = *(_BYTE *)(v8 + 16);
      if ( v64 == 50 )
      {
        if ( !*(_QWORD *)(v8 - 48) )
          goto LABEL_38;
        v188 = *(unsigned __int8 **)(v8 - 48);
        v66 = *(_QWORD *)(v8 - 24);
        if ( *(_BYTE *)(v66 + 16) != 13 )
          goto LABEL_38;
      }
      else
      {
        if ( v64 != 5 )
          goto LABEL_38;
        if ( *(_WORD *)(v8 + 18) != 26 )
          goto LABEL_38;
        v65 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
        if ( !*(_QWORD *)(v8 - 24 * v65) )
          goto LABEL_38;
        v188 = *(unsigned __int8 **)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
        v66 = *(_QWORD *)(v8 + 24 * (1 - v65));
        if ( *(_BYTE *)(v66 + 16) != 13 )
          goto LABEL_38;
      }
      v191 = v66;
      v67 = *(unsigned __int8 *)(v9 + 16);
      if ( (unsigned __int8)v67 > 0x17u )
      {
        v68 = v67 - 24;
      }
      else
      {
        if ( (_BYTE)v67 != 5 )
          goto LABEL_38;
        v68 = *(unsigned __int16 *)(v9 + 18);
      }
      if ( v68 != 37
        || ((*(_BYTE *)(v9 + 23) & 0x40) == 0
          ? (v69 = (unsigned __int8 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)))
          : (v69 = *(unsigned __int8 ***)(v9 - 8)),
            (v70 = *v69) == 0) )
      {
LABEL_38:
        v195 = (unsigned __int64)&v187;
        v196 = &v191;
        v24 = *(_QWORD *)(v8 + 8);
        if ( !v24 || *(_QWORD *)(v24 + 8) )
          goto LABEL_40;
        v46 = *(_BYTE *)(v8 + 16);
        if ( v46 == 48 )
        {
          if ( !*(_QWORD *)(v8 - 48) )
            goto LABEL_40;
          v187 = *(unsigned __int8 **)(v8 - 48);
          v47 = *(_QWORD *)(v8 - 24);
          if ( *(_BYTE *)(v47 + 16) != 13 )
            goto LABEL_40;
          v191 = *(_QWORD *)(v8 - 24);
        }
        else
        {
          if ( v46 != 5 || *(_WORD *)(v8 + 18) != 24 || !(unsigned __int8)sub_1757140((_QWORD **)&v195, v8) )
            goto LABEL_40;
          v47 = v191;
        }
        v48 = *(_QWORD *)(v9 + 8);
        if ( v48 && !*(_QWORD *)(v48 + 8) )
        {
          v49 = *(_BYTE *)(v9 + 16);
          if ( v49 == 48 )
          {
            if ( *(_QWORD *)(v9 - 48) )
            {
              v188 = *(unsigned __int8 **)(v9 - 48);
              if ( *(_QWORD *)(v9 - 24) == v47 )
                goto LABEL_102;
            }
          }
          else if ( v49 == 5 && *(_WORD *)(v9 + 18) == 24 )
          {
            v50 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
            if ( *(_QWORD *)(v9 - 24 * v50) )
            {
              v188 = *(unsigned __int8 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
              if ( *(_QWORD *)(v9 + 24 * (1 - v50)) == v47 )
                goto LABEL_102;
            }
          }
        }
LABEL_40:
        v198 = (__int64 *)&v187;
        v199 = (char *)&v191;
        v25 = *(_QWORD *)(v8 + 8);
        if ( !v25 || *(_QWORD *)(v25 + 8) )
        {
LABEL_42:
          v198 = (__int64 *)&v187;
          v199 = (char *)&v191;
          v26 = *(_QWORD *)(v8 + 8);
          if ( !v26 || *(_QWORD *)(v26 + 8) )
            goto LABEL_44;
          v76 = *(_BYTE *)(v8 + 16);
          if ( v76 == 47 )
          {
            if ( !*(_QWORD *)(v8 - 48) )
              goto LABEL_44;
            v187 = *(unsigned __int8 **)(v8 - 48);
            v77 = *(_QWORD *)(v8 - 24);
            if ( *(_BYTE *)(v77 + 16) != 13 )
              goto LABEL_44;
            v191 = *(_QWORD *)(v8 - 24);
          }
          else
          {
            if ( v76 != 5 || *(_WORD *)(v8 + 18) != 23 )
              goto LABEL_44;
            if ( !(unsigned __int8)sub_1757140(&v198, v8) )
              goto LABEL_135;
            v77 = v191;
          }
          v78 = *(_QWORD *)(v9 + 8);
          if ( !v78 || *(_QWORD *)(v78 + 8) )
            goto LABEL_135;
          v112 = *(_BYTE *)(v9 + 16);
          if ( v112 == 47 )
          {
            if ( !*(_QWORD *)(v9 - 48) )
              goto LABEL_135;
            v188 = *(unsigned __int8 **)(v9 - 48);
            if ( v77 != *(_QWORD *)(v9 - 24) )
              goto LABEL_135;
          }
          else
          {
            if ( v112 != 5 )
              goto LABEL_135;
            if ( *(_WORD *)(v9 + 18) != 23 )
              goto LABEL_135;
            v113 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
            if ( !*(_QWORD *)(v9 - 24 * v113) )
              goto LABEL_135;
            v188 = *(unsigned __int8 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
            if ( v77 != *(_QWORD *)(v9 + 24 * (1 - v113)) )
              goto LABEL_135;
          }
          v114 = *(unsigned int *)(v77 + 32);
          v115 = v114;
          if ( (unsigned int)v114 > 0x40 )
          {
            v155 = v10;
            v184 = v6;
            v140 = sub_16A57B0(v77 + 24);
            v115 = v114;
            LOWORD(v6) = v184;
            v10 = v155;
            if ( (unsigned int)(v114 - v140) > 0x40 )
              goto LABEL_135;
            v141 = **(_QWORD **)(v77 + 24);
            if ( v114 < v141 )
              goto LABEL_135;
            v117 = **(_QWORD **)(v77 + 24);
            v118 = (_DWORD)v141 != 0 && v115 > (unsigned int)v141;
          }
          else
          {
            v116 = *(_QWORD *)(v77 + 24);
            if ( v114 < v116 )
              goto LABEL_135;
            v117 = *(_QWORD *)(v77 + 24);
            v118 = v115 > (unsigned int)v116 && (_DWORD)v116 != 0;
          }
          if ( v118 )
          {
            v119 = *(_QWORD *)(v10 + 8);
            v166 = v6;
            v182 = v10;
            v154 = v115;
            v120 = sub_1649960(v11);
            v198 = (__int64 *)&v195;
            v196 = v121;
            v195 = (unsigned __int64)v120;
            v200 = 773;
            v199 = ".unshifted";
            v122 = sub_172B670(v119, (__int64)v187, (__int64)v188, (__int64 *)&v198, a3, a4, a5);
            v123 = v154;
            v124 = v182;
            v125 = v122;
            v126 = v166;
            v194 = v154;
            v127 = v154 - v117;
            if ( v154 > 0x40 )
            {
              v156 = v166;
              v168 = v127;
              v149 = v123;
              sub_16A4EF0((__int64)&v193, 0, 0);
              v127 = v168;
              v126 = v156;
              v124 = v182;
              if ( v168 <= 0x40 )
              {
                v128 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v117 - v149 + 64);
                if ( v194 > 0x40 )
                {
                  *(_QWORD *)v193 |= v128;
                  goto LABEL_216;
                }
LABEL_215:
                v193 |= v128;
LABEL_216:
                v167 = v126;
                v129 = *(_QWORD *)(v124 + 8);
                v183 = v124;
                v195 = (unsigned __int64)sub_1649960(v11);
                v199 = ".mask";
                v130 = *(_QWORD *)(v183 + 8);
                v196 = v131;
                v200 = 773;
                v198 = (__int64 *)&v195;
                v132 = sub_159C0E0(*(__int64 **)(v130 + 24), (__int64)&v193);
                v133 = sub_1729500(v129, v125, v132, (__int64 *)&v198, a3, a4, a5);
                v136 = sub_15A06D0(*(__int64 ***)v191, (__int64)v125, v134, v135);
                v200 = 257;
                v137 = v136;
                v5 = sub_1648A60(56, 2u);
                if ( !v5 )
                {
LABEL_219:
                  sub_135E100((__int64 *)&v193);
                  return v5;
                }
                v138 = v137;
                v139 = v133;
LABEL_218:
                sub_17582E0((__int64)v5, v167, (__int64)v139, v138, (__int64)&v198);
                goto LABEL_219;
              }
            }
            else
            {
              v193 = 0;
              if ( v127 <= 0x40 )
              {
                v128 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v117 - (unsigned __int8)v154 + 64);
                goto LABEL_215;
              }
            }
            v158 = v124;
            v170 = v126;
            sub_16A5260(&v193, 0, v127);
            v124 = v158;
            v126 = v170;
            goto LABEL_216;
          }
LABEL_135:
          v26 = *(_QWORD *)(v8 + 8);
LABEL_44:
          v192 = 0;
          if ( v26 )
          {
            if ( !*(_QWORD *)(v26 + 8) )
            {
              v178 = v6;
              v164 = v10;
              v198 = (__int64 *)&v187;
              v199 = (char *)&v192;
              v79 = sub_175B460((__int64)&v198, v8);
              LOWORD(v6) = v178;
              if ( v79 )
              {
                if ( *(_BYTE *)(v9 + 16) == 13 )
                {
                  v191 = v9;
                  v80 = *((_QWORD *)v187 + 1);
                  if ( !v80 || *(_QWORD *)(v80 + 8) )
                  {
                    v81 = sub_1643030(*(_QWORD *)v187);
                    LOWORD(v6) = v178;
                    v82 = v81;
                    if ( v81 > v192 )
                    {
                      v83 = sub_1643030(*(_QWORD *)v8);
                      sub_13D0120((__int64)&v193, v82, v83);
                      sub_1757210((__int64 *)&v193, v192);
                      sub_16A5C50((__int64)&v195, (const void **)(v191 + 24), v82);
                      sub_1757210((__int64 *)&v195, v192);
                      v200 = 257;
                      v84 = *(_QWORD *)(v164 + 8);
                      v85 = sub_159C0E0(*(__int64 **)(v84 + 24), (__int64)&v193);
                      v86 = sub_1729500(v84, v187, v85, (__int64 *)&v198, a3, a4, a5);
                      v87 = sub_159C0E0(*(__int64 **)(*(_QWORD *)(v164 + 8) + 24LL), (__int64)&v195);
                      v200 = 257;
                      v88 = v87;
                      v89 = sub_1648A60(56, 2u);
                      v5 = v89;
                      if ( v89 )
                        sub_17582E0((__int64)v89, v178, (__int64)v86, v88, (__int64)&v198);
                      sub_135E100((__int64 *)&v195);
                      sub_135E100((__int64 *)&v193);
                      return v5;
                    }
                  }
                }
              }
            }
          }
          if ( *(_BYTE *)(v8 + 16) != 78 )
            return 0;
          v96 = *(_QWORD *)(v8 - 24);
          if ( *(_BYTE *)(v96 + 16) || *(_DWORD *)(v96 + 36) != 6 )
          {
            v96 = *(_QWORD *)(v8 - 24);
            v97 = *(_BYTE *)(v96 + 16);
          }
          else
          {
            if ( !*(_QWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL)
                            - 24LL * (*(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)) )
            {
LABEL_181:
              if ( *(_DWORD *)(v96 + 36) != 5 )
                return 0;
              v98 = (unsigned __int8 **)((v8 & 0xFFFFFFFFFFFFFFF8LL)
                                       - 24LL * (*(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
              if ( !*v98 )
                return 0;
              v29 = *(_BYTE *)(v9 + 16) == 78;
              v187 = *v98;
              if ( !v29 )
                return 0;
              v99 = *(_QWORD *)(v9 - 24);
              if ( *(_BYTE *)(v99 + 16) )
                return 0;
              if ( *(_DWORD *)(v99 + 36) != 5 )
                return 0;
              v100 = *(_QWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL)
                               - 24LL * (*(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
              if ( !v100 )
                return 0;
              goto LABEL_187;
            }
            v29 = *(_BYTE *)(v9 + 16) == 78;
            v187 = *(unsigned __int8 **)((v8 & 0xFFFFFFFFFFFFFFF8LL)
                                       - 24LL * (*(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
            if ( v29 )
            {
              v142 = *(_QWORD *)(v9 - 24);
              if ( !*(_BYTE *)(v142 + 16) && *(_DWORD *)(v142 + 36) == 6 )
              {
                v100 = *(_QWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL)
                                 - 24LL * (*(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
                if ( v100 )
                {
LABEL_187:
                  v188 = (unsigned __int8 *)v100;
                  v180 = v6;
                  v200 = 257;
                  v101 = sub_1648A60(56, 2u);
                  v5 = v101;
                  if ( v101 )
                    sub_17582E0((__int64)v101, v180, (__int64)v187, (__int64)v188, (__int64)&v198);
                  return v5;
                }
              }
            }
            v96 = *(_QWORD *)(v8 - 24);
            v97 = *(_BYTE *)(v96 + 16);
          }
          if ( v97 )
            return 0;
          goto LABEL_181;
        }
        v90 = *(_BYTE *)(v8 + 16);
        if ( v90 == 49 )
        {
          if ( !*(_QWORD *)(v8 - 48) )
            goto LABEL_42;
          v187 = *(unsigned __int8 **)(v8 - 48);
          v47 = *(_QWORD *)(v8 - 24);
          if ( *(_BYTE *)(v47 + 16) != 13 )
            goto LABEL_42;
          v191 = *(_QWORD *)(v8 - 24);
        }
        else
        {
          if ( v90 != 5 || *(_WORD *)(v8 + 18) != 25 || !(unsigned __int8)sub_1757140(&v198, v8) )
            goto LABEL_42;
          v47 = v191;
        }
        v91 = *(_QWORD *)(v9 + 8);
        if ( !v91 || *(_QWORD *)(v91 + 8) )
          goto LABEL_42;
        v92 = *(_BYTE *)(v9 + 16);
        if ( v92 == 49 )
        {
          if ( !*(_QWORD *)(v9 - 48) )
            goto LABEL_42;
          v188 = *(unsigned __int8 **)(v9 - 48);
          if ( *(_QWORD *)(v9 - 24) != v47 )
            goto LABEL_42;
        }
        else
        {
          if ( v92 != 5 )
            goto LABEL_42;
          if ( *(_WORD *)(v9 + 18) != 25 )
            goto LABEL_42;
          v93 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
          if ( !*(_QWORD *)(v9 - 24 * v93) )
            goto LABEL_42;
          v188 = *(unsigned __int8 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
          if ( *(_QWORD *)(v9 + 24 * (1 - v93)) != v47 )
            goto LABEL_42;
        }
LABEL_102:
        v51 = *(unsigned int *)(v47 + 32);
        v52 = v51;
        if ( (unsigned int)v51 > 0x40 )
        {
          v153 = v10;
          v179 = v6;
          v165 = *(_DWORD *)(v47 + 32);
          v94 = sub_16A57B0(v47 + 24);
          v52 = v165;
          v6 = v179;
          v10 = v153;
          if ( v165 - v94 > 0x40 )
            goto LABEL_42;
          v95 = **(_QWORD **)(v47 + 24);
          if ( v165 < v95 )
            goto LABEL_42;
          v54 = **(_QWORD **)(v47 + 24);
          v55 = v165 > (unsigned int)v95 && (_DWORD)v95 != 0;
        }
        else
        {
          v53 = *(_QWORD *)(v47 + 24);
          if ( v51 < v53 )
            goto LABEL_42;
          v54 = *(_QWORD *)(v47 + 24);
          v55 = (_DWORD)v53 != 0 && (unsigned int)v51 > (unsigned int)v53;
        }
        v151 = v52;
        if ( !v55 )
          goto LABEL_42;
        v161 = v10;
        v175 = *(_QWORD *)(v10 + 8);
        v56 = (v6 != 33) + 35;
        v57 = sub_1649960(v11);
        v198 = (__int64 *)&v195;
        v200 = 773;
        v196 = v58;
        v195 = (unsigned __int64)v57;
        v199 = ".unshifted";
        v59 = sub_172B670(v175, (__int64)v187, (__int64)v188, (__int64 *)&v198, a3, a4, a5);
        v60 = v161;
        v61 = (__int64)v59;
        v62 = 1LL << v54;
        LODWORD(v196) = v151;
        if ( v151 > 0x40 )
        {
          sub_16A4EF0((__int64)&v195, 0, 0);
          v62 = 1LL << v54;
          v60 = v161;
          if ( (unsigned int)v196 > 0x40 )
          {
            *(_QWORD *)(v195 + 8LL * (v54 >> 6)) |= 1LL << v54;
            goto LABEL_109;
          }
        }
        else
        {
          v195 = 0;
        }
        v195 |= v62;
LABEL_109:
        v176 = sub_159C0E0(*(__int64 **)(*(_QWORD *)(v60 + 8) + 24LL), (__int64)&v195);
        v200 = 257;
        v63 = sub_1648A60(56, 2u);
        v5 = v63;
        if ( v63 )
          sub_17582E0((__int64)v63, v56, v61, v176, (__int64)&v198);
        sub_135E100((__int64 *)&v195);
        return v5;
      }
      v187 = v70;
      v45 = v191;
LABEL_125:
      v152 = v10;
      v162 = v6;
      sub_13A38D0((__int64)&v198, v45 + 24);
      sub_16A7490((__int64)&v198, 1);
      v71 = (int)v199;
      v194 = (unsigned int)v199;
      v193 = (unsigned __int64)v198;
      v72 = sub_14A9C60((__int64)&v193);
      v73 = v162;
      v74 = v152;
      if ( v72 )
      {
        v75 = *(_QWORD *)v187;
        if ( *(_BYTE *)(*(_QWORD *)v187 + 8LL) == 11 )
        {
          v150 = v152;
          v157 = v162;
          v169 = (__int64 ***)v187;
          v144 = sub_1455840((__int64)&v193);
          v73 = v157;
          v74 = v150;
          if ( *(_DWORD *)(v75 + 8) >> 8 == v71 - 1 - v144 )
          {
            v145 = *(_QWORD *)(v150 + 8);
            v197 = 257;
            v146 = *v169;
            v167 = v157;
            v147 = sub_1708970(v145, 36, (__int64)v188, v146, (__int64 *)&v195);
            v200 = 257;
            v148 = v147;
            v5 = sub_1648A60(56, 2u);
            if ( !v5 )
              goto LABEL_219;
            v139 = v187;
            v138 = (__int64)v148;
            goto LABEL_218;
          }
        }
      }
      v163 = v74;
      v177 = v73;
      sub_135E100((__int64 *)&v193);
      v6 = v177;
      v10 = v163;
      goto LABEL_38;
    }
    if ( (_BYTE)v12 == 50 )
    {
      if ( !*(_QWORD *)(v8 - 48) )
        goto LABEL_34;
      v187 = *(unsigned __int8 **)(v8 - 48);
      if ( !*(_QWORD *)(v8 - 24) )
        goto LABEL_34;
      v188 = *(unsigned __int8 **)(v8 - 24);
    }
    else
    {
      if ( (_BYTE)v12 != 5 || *(_WORD *)(v8 + 18) != 26 )
        goto LABEL_34;
      if ( !(unsigned __int8)sub_17570E0((_QWORD **)&v195, v8) )
        goto LABEL_59;
    }
    v198 = (__int64 *)&v189;
    v199 = (char *)&v190;
    v28 = *(_QWORD *)(v9 + 8);
    if ( !v28 || *(_QWORD *)(v28 + 8) )
    {
LABEL_59:
      v22 = *(_QWORD *)(v8 + 8);
      goto LABEL_34;
    }
    v102 = *(_BYTE *)(v9 + 16);
    if ( v102 == 50 )
    {
      v103 = *(unsigned __int8 **)(v9 - 48);
      if ( !v103 )
        goto LABEL_59;
      v189 = *(__int64 ****)(v9 - 48);
      v104 = *(unsigned __int8 **)(v9 - 24);
      if ( !v104 )
        goto LABEL_59;
      v190 = *(_QWORD *)(v9 - 24);
    }
    else
    {
      if ( v102 != 5 || *(_WORD *)(v9 + 18) != 26 || !(unsigned __int8)sub_17570E0(&v198, v9) )
        goto LABEL_59;
      v103 = (unsigned __int8 *)v189;
      v104 = (unsigned __int8 *)v190;
    }
    v105 = v188;
    if ( v187 != v103 )
    {
      if ( v187 == v104 )
      {
        v104 = v103;
        v103 = v187;
      }
      else if ( v103 == v188 )
      {
        v105 = v187;
      }
      else
      {
        if ( v188 != v104 )
          goto LABEL_59;
        v104 = v103;
        v103 = v188;
        v105 = v187;
      }
    }
    if ( v105 )
    {
      v106 = *(_QWORD *)(v10 + 8);
      v181 = v10;
      v5 = (_QWORD *)v11;
      v200 = 257;
      v107 = sub_172B670(v106, (__int64)v105, (__int64)v104, (__int64 *)&v198, a3, a4, a5);
      v200 = 257;
      v108 = sub_1729500(*(_QWORD *)(v181 + 8), v107, (__int64)v103, (__int64 *)&v198, a3, a4, a5);
      sub_1593B40((_QWORD *)(v11 - 48), (__int64)v108);
      v111 = sub_15A06D0(*(__int64 ***)v108, (__int64)v108, v109, v110);
      sub_1593B40((_QWORD *)(v11 - 24), v111);
      return v5;
    }
    goto LABEL_59;
  }
  a2 = *(_QWORD *)(v9 - 48);
  if ( !a2 )
    goto LABEL_32;
LABEL_29:
  v187 = (unsigned __int8 *)a2;
  v15 = *(_QWORD *)(v9 - 24);
  if ( !v15 )
    goto LABEL_32;
LABEL_30:
  v188 = (unsigned __int8 *)v15;
  if ( v8 != a2 )
  {
    if ( v8 != v15 )
      goto LABEL_32;
    v15 = a2;
  }
  v172 = v6;
  v17 = *(__int64 ***)a2;
LABEL_19:
  v200 = 257;
  v18 = sub_15A06D0(v17, a2, v7, v12);
  v19 = sub_1648A60(56, 2u);
  v5 = v19;
  if ( v19 )
    sub_17582E0((__int64)v19, v172, v15, v18, (__int64)&v198);
  return v5;
}
