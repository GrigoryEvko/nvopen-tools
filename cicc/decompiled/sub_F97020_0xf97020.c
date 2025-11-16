// Function: sub_F97020
// Address: 0xf97020
//
__int64 __fastcall sub_F97020(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // r8
  bool v8; // r13
  __int64 v9; // rcx
  _BYTE *v10; // rdi
  const char *v11; // rdx
  __int64 v12; // rax
  void **p_base; // r14
  __int64 v14; // rdi
  unsigned int v15; // eax
  bool v16; // zf
  char *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  char v21; // al
  __int64 v22; // rbx
  const char *v23; // rdi
  __int64 v24; // r12
  char *v25; // rdi
  _BYTE *v27; // rdi
  __int64 v28; // r12
  _QWORD *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rbx
  _BYTE *v35; // rax
  char v36; // dl
  __int64 v37; // rdx
  bool v38; // bl
  __int64 v39; // rdi
  unsigned __int8 *v40; // r13
  __int64 v41; // rdx
  __int64 v42; // r8
  __int64 v43; // rcx
  __int64 v44; // rsi
  char *v45; // rbx
  __int64 v46; // rcx
  char *v47; // rax
  unsigned int v48; // eax
  bool v49; // dl
  __int64 v50; // r15
  __int64 v51; // rbx
  __int64 v52; // rcx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rsi
  int v57; // r14d
  __int64 v58; // rax
  __int64 v59; // r13
  unsigned int *v60; // r14
  __int64 v61; // r12
  __int64 v62; // rdx
  __int64 v63; // r12
  __int64 v64; // r14
  __int64 j; // r12
  __int64 v66; // rcx
  __int64 v67; // rdi
  __int64 v68; // rdx
  int v69; // eax
  __int64 v70; // rdx
  __int64 v71; // r13
  int v72; // r15d
  int v73; // r14d
  __int64 v74; // rdx
  int v75; // eax
  unsigned int v76; // esi
  __int64 v77; // rax
  __int64 v78; // rsi
  __int64 v79; // rsi
  _QWORD *v80; // rdx
  char *v81; // rdx
  __int64 v82; // rdi
  __int64 *v83; // rax
  _QWORD *v84; // rax
  _QWORD *i; // rdx
  char v86; // al
  char v87; // dl
  _BYTE *v88; // rbx
  unsigned int v89; // eax
  __int64 *v90; // rax
  __int64 v91; // rax
  __int64 v92; // r9
  __int64 v93; // rdx
  unsigned __int64 v94; // r8
  __int64 v95; // rax
  unsigned __int64 v96; // rdx
  _QWORD *v97; // rax
  __int64 v98; // r14
  __int64 v99; // rbx
  __int64 v100; // r13
  __int64 v101; // rdx
  unsigned int v102; // esi
  __int64 v103; // rax
  _BYTE *v104; // rdi
  _BYTE *v105; // rdi
  _QWORD *v106; // rax
  __int64 v107; // r14
  __int64 v108; // rax
  __int64 v109; // r13
  __int64 v110; // r13
  __int64 v111; // rbx
  __int64 v112; // rdx
  unsigned int v113; // esi
  unsigned __int8 *v114; // rdi
  __int64 v115; // rdx
  unsigned int v116; // edx
  unsigned __int64 v117; // rax
  unsigned __int64 v118; // rax
  unsigned __int64 v119; // rcx
  unsigned __int64 v120; // rax
  unsigned int v121; // edx
  unsigned __int64 v122; // rcx
  __int64 v123; // rax
  unsigned __int64 v124; // rcx
  const void *v125; // rax
  unsigned __int8 *v126; // rdi
  __int64 v127; // rdx
  __int64 v128; // rdx
  unsigned int v129; // eax
  unsigned int v130; // eax
  unsigned int v131; // eax
  unsigned int v132; // eax
  unsigned __int8 *v133; // rdi
  unsigned __int64 v134; // rax
  _BYTE *v135; // rax
  __int64 v136; // rdx
  _BYTE *v137; // rax
  _BYTE *v138; // rax
  unsigned int v139; // edx
  __int64 v140; // rdx
  __int64 v141; // rcx
  __int64 v142; // r8
  __int64 v143; // r9
  const void *v144; // rax
  unsigned int v145; // edx
  __int64 v146; // rdx
  __int64 v147; // rcx
  __int64 v148; // r8
  __int64 v149; // r9
  __int64 v150; // rdx
  __int64 v151; // rcx
  __int64 v152; // r8
  unsigned int v153; // edx
  const void *v154; // rax
  __int64 *v155; // rax
  __int64 v156; // rdx
  __int64 v157; // rcx
  __int64 v158; // r8
  __int64 v159; // r9
  __int64 *v160; // rdi
  unsigned int v161; // edx
  const void *v162; // rax
  __int64 *v163; // rax
  __int64 v164; // rdx
  __int64 v165; // rcx
  __int64 v166; // r8
  __int64 v167; // r9
  __int64 v168; // [rsp+0h] [rbp-230h]
  __int64 v169; // [rsp+8h] [rbp-228h]
  __int64 *v170; // [rsp+8h] [rbp-228h]
  __int64 *v171; // [rsp+8h] [rbp-228h]
  _QWORD *v172; // [rsp+10h] [rbp-220h]
  __int64 v173; // [rsp+10h] [rbp-220h]
  __int64 v174; // [rsp+18h] [rbp-218h]
  _BYTE *v175; // [rsp+18h] [rbp-218h]
  __int64 v176; // [rsp+18h] [rbp-218h]
  __int64 v177; // [rsp+28h] [rbp-208h]
  bool v180; // [rsp+50h] [rbp-1E0h]
  __int64 v181; // [rsp+50h] [rbp-1E0h]
  __int64 v182; // [rsp+50h] [rbp-1E0h]
  __int64 v183; // [rsp+60h] [rbp-1D0h]
  __int64 v185; // [rsp+68h] [rbp-1C8h]
  unsigned __int64 v186; // [rsp+70h] [rbp-1C0h] BYREF
  unsigned int v187; // [rsp+78h] [rbp-1B8h]
  __int64 v188; // [rsp+80h] [rbp-1B0h] BYREF
  unsigned int v189; // [rsp+88h] [rbp-1A8h]
  unsigned __int64 v190; // [rsp+90h] [rbp-1A0h] BYREF
  unsigned int v191; // [rsp+98h] [rbp-198h]
  const void *v192; // [rsp+A0h] [rbp-190h] BYREF
  unsigned int v193; // [rsp+A8h] [rbp-188h]
  const void *v194; // [rsp+B0h] [rbp-180h] BYREF
  unsigned int v195; // [rsp+B8h] [rbp-178h]
  const void *v196; // [rsp+C0h] [rbp-170h] BYREF
  unsigned int v197; // [rsp+C8h] [rbp-168h]
  __int16 v198; // [rsp+D0h] [rbp-160h]
  const char *v199; // [rsp+E0h] [rbp-150h] BYREF
  __int64 v200; // [rsp+E8h] [rbp-148h]
  _QWORD v201[2]; // [rsp+F0h] [rbp-140h] BYREF
  __int16 v202; // [rsp+100h] [rbp-130h]
  __int64 v203; // [rsp+130h] [rbp-100h] BYREF
  __int64 v204; // [rsp+138h] [rbp-F8h]
  __int64 v205; // [rsp+140h] [rbp-F0h] BYREF
  int v206; // [rsp+148h] [rbp-E8h]
  char v207; // [rsp+14Ch] [rbp-E4h]
  __int64 v208; // [rsp+150h] [rbp-E0h] BYREF
  __int64 v209; // [rsp+190h] [rbp-A0h]
  _BYTE *v210; // [rsp+198h] [rbp-98h]
  unsigned __int8 *v211; // [rsp+1A0h] [rbp-90h]
  void *base; // [rsp+1A8h] [rbp-88h] BYREF
  __int64 v213; // [rsp+1B0h] [rbp-80h]
  _BYTE v214[64]; // [rsp+1B8h] [rbp-78h] BYREF
  unsigned int v215; // [rsp+1F8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 - 96);
  v183 = a2;
  if ( *(_BYTE *)v4 <= 0x1Cu )
  {
    LODWORD(p_base) = 0;
    return (unsigned int)p_base;
  }
  v209 = a4;
  base = v214;
  v210 = 0;
  v211 = 0;
  v213 = 0x800000000LL;
  v215 = 0;
  if ( *(_BYTE *)v4 <= 0x1Cu )
    goto LABEL_11;
  v5 = *(_QWORD *)(v4 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  a2 = 1;
  v8 = sub_BCAC40(v5, 1);
  if ( !v8 )
    goto LABEL_11;
  if ( *(_BYTE *)v4 != 58 )
  {
    if ( *(_BYTE *)v4 == 86 )
    {
      v9 = *(_QWORD *)(v4 + 8);
      if ( *(_QWORD *)(*(_QWORD *)(v4 - 96) + 8LL) == v9 )
      {
        v10 = *(_BYTE **)(v4 - 64);
        if ( *v10 <= 0x15u )
        {
          v8 = sub_AD7A80(v10, 1, v6, v9, v7);
          goto LABEL_12;
        }
      }
    }
LABEL_11:
    v8 = 0;
  }
LABEL_12:
  v206 = 0;
  v204 = (__int64)&v208;
  v11 = (const char *)v201;
  v205 = 0x100000008LL;
  v200 = 0x800000001LL;
  v199 = (const char *)v201;
  v207 = 1;
  v208 = v4;
  v203 = 1;
  v201[0] = v4;
  LODWORD(v12) = 1;
  while ( 1 )
  {
    p_base = *(void ***)&v11[8 * (unsigned int)v12 - 8];
    LODWORD(v200) = v12 - 1;
    if ( *(_BYTE *)p_base <= 0x1Cu )
      goto LABEL_21;
    v14 = (__int64)p_base[1];
    v15 = *(unsigned __int8 *)(v14 + 8) - 17;
    if ( v8 )
    {
      if ( v15 <= 1 )
        v14 = **(_QWORD **)(v14 + 16);
      a2 = 1;
      v16 = !sub_BCAC40(v14, 1);
      v21 = *(_BYTE *)p_base;
      if ( v16 )
        goto LABEL_41;
      if ( v21 == 58 )
        goto LABEL_130;
      if ( v21 != 86 )
        goto LABEL_41;
      v22 = (__int64)*(p_base - 12);
      if ( *(void **)(v22 + 8) != p_base[1] )
        goto LABEL_21;
      v104 = *(p_base - 8);
      if ( *v104 > 0x15u )
        goto LABEL_21;
      v28 = (__int64)*(p_base - 4);
      if ( !sub_AD7A80(v104, 1, (__int64)v17, v18, v19) )
        break;
      goto LABEL_39;
    }
    if ( v15 <= 1 )
      v14 = **(_QWORD **)(v14 + 16);
    a2 = 1;
    v16 = !sub_BCAC40(v14, 1);
    v21 = *(_BYTE *)p_base;
    if ( v16 )
      goto LABEL_41;
    if ( v21 == 57 )
    {
LABEL_130:
      if ( (*((_BYTE *)p_base + 7) & 0x40) != 0 )
      {
        v83 = (__int64 *)*(p_base - 1);
      }
      else
      {
        v17 = (char *)(32LL * (*((_DWORD *)p_base + 1) & 0x7FFFFFF));
        v83 = (__int64 *)((char *)p_base - v17);
      }
      v22 = *v83;
      if ( !*v83 )
        goto LABEL_21;
      v28 = v83[4];
      if ( !v28 )
        goto LABEL_21;
      goto LABEL_134;
    }
    if ( v21 != 86 )
      goto LABEL_41;
    v22 = (__int64)*(p_base - 12);
    if ( *(void **)(v22 + 8) != p_base[1] )
      goto LABEL_21;
    v27 = *(p_base - 4);
    if ( *v27 > 0x15u )
      goto LABEL_21;
    v28 = (__int64)*(p_base - 8);
    if ( !sub_AC30F0((__int64)v27) )
      break;
LABEL_39:
    if ( !v28 )
      break;
LABEL_134:
    if ( !v207 )
      goto LABEL_144;
    v84 = (_QWORD *)v204;
    a2 = HIDWORD(v205);
    v18 = v204 + 8LL * HIDWORD(v205);
    v17 = (char *)v204;
    if ( v204 == v18 )
    {
LABEL_177:
      if ( HIDWORD(v205) >= (unsigned int)v205 )
      {
LABEL_144:
        sub_C8CC70((__int64)&v203, v28, (__int64)v17, v18, v19, v20);
        v86 = v207;
        if ( (_BYTE)i )
          goto LABEL_179;
      }
      else
      {
        ++HIDWORD(v205);
        *(_QWORD *)v18 = v28;
        ++v203;
LABEL_179:
        v95 = (unsigned int)v200;
        v18 = HIDWORD(v200);
        v96 = (unsigned int)v200 + 1LL;
        if ( v96 > HIDWORD(v200) )
        {
          sub_C8D5F0((__int64)&v199, v201, v96, 8u, v19, v20);
          v95 = (unsigned int)v200;
        }
        i = v199;
        *(_QWORD *)&v199[8 * v95] = v28;
        v86 = v207;
        LODWORD(v200) = v200 + 1;
      }
      if ( !v86 )
        goto LABEL_146;
      v84 = (_QWORD *)v204;
      a2 = HIDWORD(v205);
      goto LABEL_139;
    }
    while ( v28 != *(_QWORD *)v17 )
    {
      v17 += 8;
      if ( (char *)v18 == v17 )
        goto LABEL_177;
    }
LABEL_139:
    for ( i = &v84[(unsigned int)a2]; i != v84; ++v84 )
    {
      if ( v22 == *v84 )
        goto LABEL_52;
    }
    if ( (unsigned int)a2 >= (unsigned int)v205 )
    {
LABEL_146:
      a2 = v22;
      sub_C8CC70((__int64)&v203, v22, (__int64)i, v18, v19, v20);
      v12 = (unsigned int)v200;
      if ( !v87 )
        goto LABEL_53;
      goto LABEL_147;
    }
    a2 = (unsigned int)(a2 + 1);
    HIDWORD(v205) = a2;
    *i = v22;
    v12 = (unsigned int)v200;
    ++v203;
LABEL_147:
    if ( v12 + 1 > (unsigned __int64)HIDWORD(v200) )
    {
      a2 = (__int64)v201;
      sub_C8D5F0((__int64)&v199, v201, v12 + 1, 8u, v19, v20);
      v12 = (unsigned int)v200;
    }
    *(_QWORD *)&v199[8 * v12] = v22;
    LODWORD(v12) = v200 + 1;
    LODWORD(v200) = v200 + 1;
LABEL_53:
    if ( !(_DWORD)v12 )
      goto LABEL_23;
    v11 = v199;
  }
  v21 = *(_BYTE *)p_base;
LABEL_41:
  if ( v21 != 82 )
    goto LABEL_21;
  a2 = v209;
  v29 = (*((_BYTE *)p_base + 7) & 0x40) != 0 ? *(p_base - 1) : &p_base[-4 * (*((_DWORD *)p_base + 1) & 0x7FFFFFF)];
  v30 = sub_F8E510(v29[4], v209);
  v34 = v30;
  if ( !v30 )
    goto LABEL_21;
  a2 = *((_WORD *)p_base + 1) & 0x3F;
  if ( (_DWORD)a2 == !v8 + 32 )
  {
    v35 = *(p_base - 8);
    v36 = *v35;
    if ( *v35 != 57 )
    {
LABEL_47:
      if ( v36 == 58 )
      {
        v31 = *((_QWORD *)v35 - 8);
        v173 = v31;
        if ( v31 )
        {
          v133 = (unsigned __int8 *)*((_QWORD *)v35 - 4);
          v31 = *v133;
          a2 = (__int64)(v133 + 24);
          if ( (_BYTE)v31 == 17 )
            goto LABEL_241;
          v136 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v133 + 1) + 8LL) - 17;
          if ( (unsigned int)v136 <= 1 && (unsigned __int8)v31 <= 0x15u )
          {
            a2 = 0;
            v137 = sub_AD7630((__int64)v133, 0, v136);
            if ( !v137 || (a2 = (__int64)(v137 + 24), *v137 != 17) )
            {
LABEL_246:
              v35 = *(p_base - 8);
              goto LABEL_247;
            }
LABEL_241:
            v187 = *(_DWORD *)(a2 + 8);
            if ( v187 > 0x40 )
            {
              sub_C43780((__int64)&v186, (const void **)a2);
              if ( v187 <= 0x40 )
              {
                v134 = v186;
                goto LABEL_243;
              }
              if ( (unsigned int)sub_C44630((__int64)&v186) != 1 )
              {
LABEL_245:
                sub_969240((__int64 *)&v186);
                goto LABEL_246;
              }
            }
            else
            {
              v134 = *(_QWORD *)a2;
              v186 = *(_QWORD *)a2;
LABEL_243:
              if ( !v134 || (v134 & (v134 - 1)) != 0 )
                goto LABEL_245;
            }
            a2 = v34 + 24;
            v171 = (__int64 *)(v34 + 24);
            sub_9865C0((__int64)&v190, v34 + 24);
            if ( v191 <= 0x40 )
            {
              v195 = v191;
              v144 = (const void *)(v186 | v190);
              v191 = 0;
              v190 = (unsigned __int64)v144;
              v194 = v144;
              goto LABEL_279;
            }
            a2 = (__int64)&v186;
            sub_C43BD0(&v190, (__int64 *)&v186);
            v145 = v191;
            v144 = (const void *)v190;
            v191 = 0;
            v195 = v145;
            v194 = (const void *)v190;
            if ( v145 <= 0x40 )
            {
LABEL_279:
              if ( *(const void **)(v34 + 24) == v144 )
                goto LABEL_283;
            }
            else
            {
              a2 = v34 + 24;
              if ( sub_C43C50((__int64)&v194, (const void **)v171) )
              {
LABEL_283:
                sub_969240((__int64 *)&v194);
                sub_969240((__int64 *)&v190);
                if ( v210 )
                {
                  if ( v210 == (_BYTE *)v173 )
                    goto LABEL_285;
LABEL_270:
                  sub_969240((__int64 *)&v186);
                  goto LABEL_21;
                }
                v210 = (_BYTE *)v173;
LABEL_285:
                p_base = &base;
                sub_F969F0((__int64)&base, v34, v146, v147, v148, v149);
                sub_9865C0((__int64)&v188, (__int64)&v186);
                sub_987160((__int64)&v188, (__int64)&v186, v150, v151, v152);
                v153 = v189;
                v189 = 0;
                v191 = v153;
                v190 = v188;
                if ( v153 > 0x40 )
                {
                  sub_C43B90(&v190, v171);
                  v153 = v191;
                  v154 = (const void *)v190;
                }
                else
                {
                  v154 = (const void *)(*(_QWORD *)(v34 + 24) & v188);
                  v190 = (unsigned __int64)v154;
                }
                v195 = v153;
                v194 = v154;
                v191 = 0;
                v155 = (__int64 *)sub_BD5C60(v34);
                a2 = sub_ACCFD0(v155, (__int64)&v194);
                sub_F969F0((__int64)&base, a2, v156, v157, v158, v159);
                sub_969240((__int64 *)&v194);
                sub_969240((__int64 *)&v190);
                v160 = &v188;
LABEL_288:
                sub_969240(v160);
                ++v215;
                sub_969240((__int64 *)&v186);
                goto LABEL_52;
              }
            }
            sub_969240((__int64 *)&v194);
            sub_969240((__int64 *)&v190);
            goto LABEL_245;
          }
LABEL_247:
          if ( !v35 )
            goto LABEL_21;
        }
      }
LABEL_48:
      v37 = (__int64)v210;
      if ( v210 )
      {
        if ( v35 != v210 )
          goto LABEL_21;
      }
      else
      {
        v210 = v35;
      }
      a2 = v34;
      ++v215;
      sub_F969F0((__int64)&base, v34, v37, v31, v32, v33);
      v38 = *(p_base - 8) != 0;
      goto LABEL_51;
    }
    v31 = *((_QWORD *)v35 - 8);
    v168 = v31;
    if ( !v31 )
      goto LABEL_48;
    v114 = (unsigned __int8 *)*((_QWORD *)v35 - 4);
    v115 = *v114;
    a2 = (__int64)(v114 + 24);
    if ( (_BYTE)v115 != 17 )
    {
      v31 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v114 + 1) + 8LL) - 17;
      if ( (unsigned int)v31 > 1 || (unsigned __int8)v115 > 0x15u )
        goto LABEL_48;
      a2 = 0;
      v135 = sub_AD7630((__int64)v114, 0, v115);
      if ( !v135 || (a2 = (__int64)(v135 + 24), *v135 != 17) )
      {
LABEL_221:
        v35 = *(p_base - 8);
        v36 = *v35;
        goto LABEL_47;
      }
    }
    v116 = *(_DWORD *)(a2 + 8);
    v195 = v116;
    if ( v116 <= 0x40 )
    {
      v117 = *(_QWORD *)a2;
      goto LABEL_208;
    }
    sub_C43780((__int64)&v194, (const void **)a2);
    v116 = v195;
    if ( v195 <= 0x40 )
    {
      v117 = (unsigned __int64)v194;
LABEL_208:
      v118 = ~v117;
      v187 = v116;
      v119 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v116;
      a2 = 0;
      if ( !v116 )
        v119 = 0;
      v120 = v119 & v118;
      v186 = v120;
    }
    else
    {
      sub_C43D10((__int64)&v194);
      v120 = (unsigned __int64)v194;
      v187 = v195;
      v186 = (unsigned __int64)v194;
      if ( v195 > 0x40 )
      {
        if ( (unsigned int)sub_C44630((__int64)&v186) != 1 )
          goto LABEL_220;
LABEL_213:
        a2 = (__int64)&v186;
        v170 = (__int64 *)(v34 + 24);
        sub_9865C0((__int64)&v188, (__int64)&v186);
        v121 = v189;
        if ( v189 > 0x40 )
        {
          sub_C43D10((__int64)&v188);
          v121 = v189;
          v123 = v188;
          v189 = 0;
          v191 = v121;
          v190 = v188;
          if ( v121 <= 0x40 )
            goto LABEL_217;
          a2 = v34 + 24;
          sub_C43B90(&v190, v170);
          v139 = v191;
          v125 = (const void *)v190;
          v191 = 0;
          v195 = v139;
          v194 = (const void *)v190;
          if ( v139 <= 0x40 )
          {
            v124 = *(_QWORD *)(v34 + 24);
            goto LABEL_218;
          }
          a2 = v34 + 24;
          if ( !sub_C43C50((__int64)&v194, (const void **)v170) )
          {
LABEL_219:
            sub_969240((__int64 *)&v194);
            sub_969240((__int64 *)&v190);
            sub_969240(&v188);
LABEL_220:
            sub_969240((__int64 *)&v186);
            goto LABEL_221;
          }
        }
        else
        {
          v189 = 0;
          v122 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v121;
          a2 = 0;
          if ( !v121 )
            v122 = 0;
          v123 = v122 & ~v188;
          v188 = v123;
LABEL_217:
          v124 = *(_QWORD *)(v34 + 24);
          v195 = v121;
          v191 = 0;
          v125 = (const void *)(v124 & v123);
          v190 = (unsigned __int64)v125;
          v194 = v125;
LABEL_218:
          if ( (const void *)v124 != v125 )
            goto LABEL_219;
        }
        sub_969240((__int64 *)&v194);
        sub_969240((__int64 *)&v190);
        sub_969240(&v188);
        if ( v210 )
        {
          if ( v210 != (_BYTE *)v168 )
            goto LABEL_270;
        }
        else
        {
          v210 = (_BYTE *)v168;
        }
        p_base = &base;
        sub_F969F0((__int64)&base, v34, v140, v141, v142, v143);
        sub_9865C0((__int64)&v190, (__int64)v170);
        v161 = v191;
        if ( v191 > 0x40 )
        {
          sub_C43BD0(&v190, (__int64 *)&v186);
          v161 = v191;
          v162 = (const void *)v190;
        }
        else
        {
          v162 = (const void *)(v186 | v190);
          v190 |= v186;
        }
        v195 = v161;
        v194 = v162;
        v191 = 0;
        v163 = (__int64 *)sub_BD5C60(v34);
        a2 = sub_ACCFD0(v163, (__int64)&v194);
        sub_F969F0((__int64)&base, a2, v164, v165, v166, v167);
        sub_969240((__int64 *)&v194);
        v160 = (__int64 *)&v190;
        goto LABEL_288;
      }
    }
    if ( !v120 || (v120 & (v120 - 1)) != 0 )
      goto LABEL_220;
    goto LABEL_213;
  }
  sub_AB1A50((__int64)&v190, a2, v30 + 24);
  if ( (*((_BYTE *)p_base + 7) & 0x40) != 0 )
    v88 = *(_BYTE **)*(p_base - 1);
  else
    v88 = p_base[-4 * (*((_DWORD *)p_base + 1) & 0x7FFFFFF)];
  if ( *v88 == 42 )
  {
    v175 = (_BYTE *)*((_QWORD *)v88 - 8);
    if ( v175 )
    {
      v126 = (unsigned __int8 *)*((_QWORD *)v88 - 4);
      v127 = *v126;
      if ( (_BYTE)v127 == 17 )
      {
        v128 = (__int64)(v126 + 24);
LABEL_225:
        sub_AB1F90((__int64)&v194, (__int64 *)&v190, v128);
        if ( v191 > 0x40 && v190 )
          j_j___libc_free_0_0(v190);
        v190 = (unsigned __int64)v194;
        v129 = v195;
        v195 = 0;
        v191 = v129;
        if ( v193 > 0x40 && v192 )
          j_j___libc_free_0_0(v192);
        v192 = v196;
        v130 = v197;
        v197 = 0;
        v193 = v130;
        sub_969240((__int64 *)&v196);
        sub_969240((__int64 *)&v194);
        v88 = v175;
      }
      else if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v126 + 1) + 8LL) - 17 <= 1
             && (unsigned __int8)v127 <= 0x15u )
      {
        v138 = sub_AD7630((__int64)v126, 0, v127);
        if ( v138 )
        {
          if ( *v138 == 17 )
          {
            v128 = (__int64)(v138 + 24);
            goto LABEL_225;
          }
        }
      }
    }
  }
  if ( !v8 )
  {
    sub_ABB300((__int64)&v194, (__int64)&v190);
    if ( v191 > 0x40 && v190 )
      j_j___libc_free_0_0(v190);
    v190 = (unsigned __int64)v194;
    v131 = v195;
    v195 = 0;
    v191 = v131;
    if ( v193 > 0x40 && v192 )
      j_j___libc_free_0_0(v192);
    v192 = v196;
    v132 = v197;
    v197 = 0;
    v193 = v132;
    sub_969240((__int64 *)&v196);
    sub_969240((__int64 *)&v194);
  }
  a2 = 8;
  if ( (unsigned __int8)sub_AB0550((__int64)&v190, 8u) || sub_AAF7D0((__int64)&v190) )
    goto LABEL_167;
  if ( !v210 )
  {
    v210 = v88;
    goto LABEL_158;
  }
  if ( v210 != v88 )
  {
LABEL_167:
    v38 = 0;
    goto LABEL_168;
  }
LABEL_158:
  v89 = v191;
  v195 = v191;
  if ( v191 > 0x40 )
  {
    a2 = (__int64)&v190;
    sub_C43780((__int64)&v194, (const void **)&v190);
    v89 = v195;
  }
  else
  {
    v194 = (const void *)v190;
  }
  while ( 2 )
  {
    if ( v89 > 0x40 )
    {
      a2 = (__int64)&v192;
      if ( sub_C43C50((__int64)&v194, &v192) )
        break;
      goto LABEL_161;
    }
    if ( v194 != v192 )
    {
LABEL_161:
      v90 = (__int64 *)sub_BD5C60((__int64)p_base);
      a2 = (__int64)&v194;
      v91 = sub_ACCFD0(v90, (__int64)&v194);
      v93 = (unsigned int)v213;
      v94 = (unsigned int)v213 + 1LL;
      if ( v94 > HIDWORD(v213) )
      {
        a2 = (__int64)v214;
        v176 = v91;
        sub_C8D5F0((__int64)&base, v214, (unsigned int)v213 + 1LL, 8u, v94, v92);
        v93 = (unsigned int)v213;
        v91 = v176;
      }
      *((_QWORD *)base + v93) = v91;
      LODWORD(v213) = v213 + 1;
      sub_C46250((__int64)&v194);
      v89 = v195;
      continue;
    }
    break;
  }
  v38 = 1;
  sub_969240((__int64 *)&v194);
  ++v215;
LABEL_168:
  if ( v193 > 0x40 && v192 )
    j_j___libc_free_0_0(v192);
  if ( v191 > 0x40 && v190 )
    j_j___libc_free_0_0(v190);
LABEL_51:
  if ( v38 )
  {
LABEL_52:
    LODWORD(v12) = v200;
    goto LABEL_53;
  }
LABEL_21:
  if ( !v211 )
  {
    v211 = (unsigned __int8 *)p_base;
    goto LABEL_52;
  }
  v210 = 0;
LABEL_23:
  if ( v207 )
  {
    v23 = v199;
    if ( v199 != (const char *)v201 )
      goto LABEL_25;
  }
  else
  {
    _libc_free(v204, a2);
    v23 = v199;
    if ( v199 != (const char *)v201 )
LABEL_25:
      _libc_free(v23, a2);
  }
  v24 = (__int64)v210;
  LOBYTE(p_base) = v210 == 0 || v215 <= 1;
  if ( (_BYTE)p_base )
  {
    v25 = (char *)base;
    LODWORD(p_base) = 0;
    goto LABEL_28;
  }
  v39 = *(_QWORD *)(v4 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v39 + 8) - 17 <= 1 )
    v39 = **(_QWORD **)(v39 + 16);
  v40 = v211;
  v180 = sub_BCAC40(v39, 1);
  if ( v180 && *(_BYTE *)v4 != 58 )
  {
    v180 = 0;
    if ( *(_BYTE *)v4 == 86 )
    {
      v43 = *(_QWORD *)(v4 + 8);
      if ( *(_QWORD *)(*(_QWORD *)(v4 - 96) + 8LL) == v43 )
      {
        v105 = *(_BYTE **)(v4 - 64);
        if ( *v105 <= 0x15u )
          v180 = sub_AD7A80(v105, 1, v41, v43, v42);
      }
    }
  }
  v25 = (char *)base;
  v44 = 8LL * (unsigned int)v213;
  if ( (unsigned int)v213 > 1uLL )
  {
    qsort(base, v44 >> 3, 8u, (__compar_fn_t)sub_F8E3F0);
    v25 = (char *)base;
    v44 = 8LL * (unsigned int)v213;
  }
  a2 = (__int64)&v25[v44];
  if ( (char *)a2 == v25 )
  {
    v49 = 1;
    v48 = 0;
    goto LABEL_68;
  }
  v45 = v25;
  do
  {
    v47 = v45;
    v45 += 8;
    if ( (char *)a2 == v45 )
    {
      a2 = (a2 - (__int64)v25) >> 3;
      v48 = a2;
      v49 = (unsigned int)a2 <= 1;
      goto LABEL_68;
    }
    v46 = *((_QWORD *)v45 - 1);
  }
  while ( v46 != *(_QWORD *)v45 );
  if ( (char *)a2 == v47 )
  {
    v48 = (a2 - (__int64)v25) >> 3;
    v49 = v48 <= 1;
    goto LABEL_68;
  }
  v80 = v47 + 16;
  if ( (char *)a2 == v47 + 16 )
    goto LABEL_298;
  while ( 1 )
  {
    if ( *v80 != v46 )
    {
      *((_QWORD *)v47 + 1) = *v80;
      v47 += 8;
    }
    if ( (_QWORD *)a2 == ++v80 )
      break;
    v46 = *(_QWORD *)v47;
  }
  v25 = (char *)base;
  v81 = (char *)base + 8 * (unsigned int)v213 - a2;
  v45 = &v81[(_QWORD)(v47 + 8)];
  if ( (void *)a2 == (char *)base + 8 * (unsigned int)v213 )
  {
LABEL_298:
    v48 = (v45 - v25) >> 3;
    v49 = v48 <= 1;
LABEL_68:
    LODWORD(v213) = v48;
    if ( !v40 )
      goto LABEL_70;
LABEL_69:
    if ( v49 )
      goto LABEL_28;
    goto LABEL_70;
  }
  memmove(v47 + 8, (const void *)a2, (size_t)v81);
  v25 = (char *)base;
  LODWORD(v213) = (v45 - (_BYTE *)base) >> 3;
  v49 = (unsigned int)v213 <= 1;
  if ( v40 )
    goto LABEL_69;
LABEL_70:
  v50 = *(_QWORD *)(v183 - 32);
  v177 = *(_QWORD *)(v183 - 64);
  if ( !v180 )
  {
    v177 = *(_QWORD *)(v183 - 32);
    v50 = *(_QWORD *)(v183 - 64);
  }
  v203 = (__int64)&v205;
  v51 = *(_QWORD *)(v183 + 40);
  v204 = 0x200000000LL;
  if ( v40 )
  {
    v199 = "switch.early.test";
    v52 = *(_QWORD *)(a1 + 8);
    v202 = 259;
    v174 = sub_F36990(v51, (__int64 *)(v183 + 24), 0, v52, 0, 0, (void **)&v199, 0);
    v172 = (_QWORD *)sub_986580(v51);
    sub_D5F1F0(a3, (__int64)v172);
    if ( !sub_98ED60(v40, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL), v183, 0, 0) )
    {
      v198 = 257;
      v202 = 257;
      v106 = sub_BD2C40(72, unk_3F10A14);
      v107 = (__int64)v106;
      if ( v106 )
        sub_B549F0((__int64)v106, (__int64)v40, (__int64)&v199, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, const void **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v107,
        &v194,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      v108 = *(_QWORD *)a3;
      v109 = 16LL * *(unsigned int *)(a3 + 8);
      if ( v108 != v108 + v109 )
      {
        v169 = v51;
        v110 = v108 + v109;
        v111 = *(_QWORD *)a3;
        do
        {
          v112 = *(_QWORD *)(v111 + 8);
          v113 = *(_DWORD *)v111;
          v111 += 16;
          sub_B99FD0(v107, v113, v112);
        }
        while ( v110 != v111 );
        v51 = v169;
      }
      v40 = (unsigned __int8 *)v107;
    }
    if ( v180 )
    {
      sub_F94450((__int64 *)a3, (__int64)v40, v50, v174, 0, 0);
    }
    else
    {
      v202 = 257;
      v97 = sub_BD2C40(72, 3u);
      v98 = (__int64)v97;
      if ( v97 )
        sub_B4C9A0((__int64)v97, v174, v50, (__int64)v40, 3u, 0, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v98,
        &v199,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      if ( *(_QWORD *)a3 != *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8) )
      {
        v182 = v51;
        v99 = *(_QWORD *)a3;
        v100 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        do
        {
          v101 = *(_QWORD *)(v99 + 8);
          v102 = *(_DWORD *)v99;
          v99 += 16;
          sub_B99FD0(v98, v102, v101);
        }
        while ( v100 != v99 );
        v51 = v182;
      }
    }
    sub_B43D60(v172);
    if ( *(_QWORD *)(a1 + 8) )
      sub_F35FA0((__int64)&v203, v51, v50 & 0xFFFFFFFFFFFFFFFBLL, v53, v54, v55);
    v56 = v51;
    v51 = v174;
    sub_F91F00(v50, v56, v174, 0);
  }
  sub_D5F1F0(a3, v183);
  if ( *(_BYTE *)(*(_QWORD *)(v24 + 8) + 8LL) == 14 )
  {
    v199 = "magicptr";
    v202 = 259;
    v103 = sub_AE4450(a4, *(_QWORD *)(v24 + 8));
    v24 = sub_F94180((__int64 *)a3, 47, v24, v103, (__int64)&v199, 0, (int)v194, 0);
  }
  v57 = v213;
  v202 = 257;
  v58 = sub_BD2DA0(80);
  v59 = v58;
  if ( v58 )
    sub_B53A60(v58, v24, v177, v57, 0, 0);
  a2 = v59;
  (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v59,
    &v199,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v60 = *(unsigned int **)a3;
  v61 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v61 )
  {
    do
    {
      v62 = *((_QWORD *)v60 + 1);
      a2 = *v60;
      v60 += 4;
      sub_B99FD0(v59, a2, v62);
    }
    while ( (unsigned int *)v61 != v60 );
  }
  if ( (_DWORD)v213 )
  {
    v63 = 8LL * (unsigned int)v213;
    v64 = 0;
    do
    {
      a2 = *(_QWORD *)((char *)base + v64);
      v64 += 8;
      sub_B53E30(v59, a2, v50);
    }
    while ( v64 != v63 );
  }
  for ( j = *(_QWORD *)(v50 + 56); ; j = *(_QWORD *)(j + 8) )
  {
    if ( !j )
      BUG();
    v66 = j - 24;
    if ( *(_BYTE *)(j - 24) != 84 )
      break;
    v67 = *(_QWORD *)(j - 32);
    v68 = 0x1FFFFFFFE0LL;
    v69 = *(_DWORD *)(j - 20) & 0x7FFFFFF;
    if ( v69 )
    {
      v70 = 0;
      a2 = v67 + 32LL * *(unsigned int *)(j + 48);
      do
      {
        if ( v51 == *(_QWORD *)(a2 + 8 * v70) )
        {
          v68 = 32 * v70;
          goto LABEL_96;
        }
        ++v70;
      }
      while ( v69 != (_DWORD)v70 );
      v68 = 0x1FFFFFFFE0LL;
    }
LABEL_96:
    v71 = *(_QWORD *)(v67 + v68);
    v72 = v213 - 1;
    if ( (_DWORD)v213 != 1 )
    {
      v73 = 0;
      v74 = v71 + 16;
      while ( 1 )
      {
        if ( v69 == *(_DWORD *)(j + 48) )
        {
          v181 = v74;
          v185 = v66;
          sub_B48D90(v66);
          v74 = v181;
          v66 = v185;
          v69 = *(_DWORD *)(j - 20) & 0x7FFFFFF;
        }
        v75 = (v69 + 1) & 0x7FFFFFF;
        v76 = v75 | *(_DWORD *)(j - 20) & 0xF8000000;
        v77 = *(_QWORD *)(j - 32) + 32LL * (unsigned int)(v75 - 1);
        *(_DWORD *)(j - 20) = v76;
        if ( *(_QWORD *)v77 )
        {
          v78 = *(_QWORD *)(v77 + 8);
          **(_QWORD **)(v77 + 16) = v78;
          if ( v78 )
            *(_QWORD *)(v78 + 16) = *(_QWORD *)(v77 + 16);
        }
        *(_QWORD *)v77 = v71;
        if ( v71 )
        {
          v79 = *(_QWORD *)(v71 + 16);
          *(_QWORD *)(v77 + 8) = v79;
          if ( v79 )
            *(_QWORD *)(v79 + 16) = v77 + 8;
          *(_QWORD *)(v77 + 16) = v74;
          *(_QWORD *)(v71 + 16) = v77;
        }
        ++v73;
        a2 = (*(_DWORD *)(j - 20) & 0x7FFFFFFu) - 1;
        *(_QWORD *)(*(_QWORD *)(j - 32) + 32LL * *(unsigned int *)(j + 48) + 8 * a2) = v51;
        if ( v72 == v73 )
          break;
        v69 = *(_DWORD *)(j - 20) & 0x7FFFFFF;
      }
    }
  }
  sub_F91380((char *)v183);
  v82 = *(_QWORD *)(a1 + 8);
  if ( v82 )
  {
    a2 = v203;
    sub_FFB3D0(v82, v203, (unsigned int)v204);
  }
  if ( (__int64 *)v203 != &v205 )
    _libc_free(v203, a2);
  v25 = (char *)base;
  LODWORD(p_base) = 1;
LABEL_28:
  if ( v25 != v214 )
    _libc_free(v25, a2);
  return (unsigned int)p_base;
}
