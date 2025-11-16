// Function: sub_147EE30
// Address: 0x147ee30
//
__int64 __fastcall sub_147EE30(_QWORD *a1, __int64 **a2, unsigned int a3, unsigned int a4, __m128i a5, __m128i a6)
{
  __int64 **v6; // rbx
  __int64 *v9; // r13
  unsigned int v10; // ecx
  __int64 v11; // r14
  __int16 v12; // ax
  __int64 v13; // r12
  unsigned int v14; // r14d
  __int64 v15; // rdi
  int v16; // eax
  bool v17; // al
  bool v18; // r9
  bool v19; // al
  unsigned int v20; // r8d
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 i; // rax
  __int64 v24; // r14
  __int64 v25; // rdi
  unsigned __int16 v26; // dx
  __int64 v27; // rdx
  char v28; // al
  unsigned int v29; // r12d
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  const void *v34; // r10
  signed __int64 v35; // r8
  __int64 v36; // r13
  unsigned int v37; // eax
  char v39; // r12
  _BYTE *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 *v44; // r14
  const void *v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // eax
  bool v49; // zf
  __int64 v50; // rsi
  unsigned int v51; // edi
  __int64 v52; // rdx
  __int64 v53; // rax
  unsigned int v54; // r14d
  __int64 *v55; // rax
  unsigned int v56; // r12d
  __int64 v57; // rbx
  __int64 v58; // rdi
  __int64 v59; // rdx
  int v60; // eax
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // r10
  __int64 v65; // r9
  bool v66; // r15
  unsigned __int64 v67; // r11
  unsigned __int64 v68; // rcx
  __int64 v69; // r8
  unsigned __int64 v70; // rsi
  unsigned __int64 v71; // rdi
  unsigned __int64 v72; // rax
  int v73; // edx
  int v74; // eax
  unsigned __int64 v75; // rbx
  __int64 v76; // r13
  unsigned __int64 v77; // r9
  unsigned __int64 v78; // r10
  unsigned __int64 v79; // rsi
  __int64 v80; // r9
  unsigned __int64 v81; // rdi
  unsigned __int64 v82; // rcx
  unsigned __int64 v83; // rax
  __int64 v84; // r9
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rcx
  __int64 *v89; // rax
  __int64 v90; // r14
  __int16 v91; // dx
  __int64 *v92; // r13
  unsigned int v93; // r14d
  __int64 *v94; // rbx
  __int64 v95; // rdx
  __int64 v96; // rsi
  char v97; // dl
  __int64 v98; // rdx
  __int64 *v99; // rcx
  unsigned int v100; // eax
  __int64 v101; // r13
  _QWORD *v102; // r9
  _QWORD *v103; // r13
  _QWORD *v104; // r14
  char v105; // dl
  __int64 v106; // r15
  _QWORD *v107; // rax
  _QWORD *v108; // rsi
  _QWORD *v109; // rcx
  __int64 v110; // rax
  __int64 *v111; // rax
  __int64 *v112; // rbx
  __int64 v113; // rdx
  __int64 *v114; // r14
  unsigned int v115; // r8d
  __int64 v116; // rdx
  __int64 v117; // rax
  __int64 *v118; // rdi
  __int64 *v119; // rax
  __int64 v120; // r12
  __int64 v121; // rax
  unsigned __int64 v122; // rdx
  unsigned int v123; // r14d
  __int64 v124; // r9
  __int64 v125; // rax
  unsigned int v126; // r8d
  __int64 v127; // r12
  __int64 v128; // r14
  __int64 v129; // rdx
  __int64 v130; // rbx
  __int64 v131; // rdx
  __int64 *v132; // rax
  __int64 v133; // [rsp+8h] [rbp-248h]
  __int64 v134; // [rsp+10h] [rbp-240h]
  __int64 v135; // [rsp+18h] [rbp-238h]
  __int64 v136; // [rsp+20h] [rbp-230h]
  __int64 v137; // [rsp+28h] [rbp-228h]
  __int16 v138; // [rsp+40h] [rbp-210h]
  unsigned int v139; // [rsp+44h] [rbp-20Ch]
  unsigned int v141; // [rsp+58h] [rbp-1F8h]
  int v142; // [rsp+5Ch] [rbp-1F4h]
  int v143; // [rsp+60h] [rbp-1F0h]
  unsigned __int64 v144; // [rsp+68h] [rbp-1E8h]
  unsigned __int64 v145; // [rsp+70h] [rbp-1E0h]
  int v147; // [rsp+7Ch] [rbp-1D4h]
  __int64 v148; // [rsp+88h] [rbp-1C8h]
  __int64 v149; // [rsp+90h] [rbp-1C0h]
  __int64 v150; // [rsp+98h] [rbp-1B8h]
  unsigned __int64 v151; // [rsp+A0h] [rbp-1B0h]
  char v152; // [rsp+ADh] [rbp-1A3h]
  __int64 v153; // [rsp+B0h] [rbp-1A0h]
  __int64 v154; // [rsp+B8h] [rbp-198h]
  bool v155; // [rsp+C0h] [rbp-190h]
  unsigned __int64 v156; // [rsp+C0h] [rbp-190h]
  __int64 v157; // [rsp+C0h] [rbp-190h]
  __int64 v158; // [rsp+C8h] [rbp-188h]
  signed __int64 v159; // [rsp+C8h] [rbp-188h]
  __int64 **v160; // [rsp+C8h] [rbp-188h]
  __int64 v161; // [rsp+C8h] [rbp-188h]
  const void *v162; // [rsp+D0h] [rbp-180h]
  bool v163; // [rsp+D0h] [rbp-180h]
  unsigned int v164; // [rsp+D0h] [rbp-180h]
  unsigned int v165; // [rsp+D0h] [rbp-180h]
  unsigned int v166; // [rsp+D8h] [rbp-178h]
  __int64 *v167; // [rsp+D8h] [rbp-178h]
  __int64 v168; // [rsp+D8h] [rbp-178h]
  __int64 v169; // [rsp+D8h] [rbp-178h]
  __int64 *v170; // [rsp+E0h] [rbp-170h] BYREF
  __int64 v171; // [rsp+E8h] [rbp-168h]
  __int64 v172; // [rsp+F0h] [rbp-160h] BYREF
  __int64 v173; // [rsp+F8h] [rbp-158h]
  __int64 v174; // [rsp+100h] [rbp-150h]
  __int64 *v175; // [rsp+110h] [rbp-140h] BYREF
  __int64 v176; // [rsp+118h] [rbp-138h]
  _BYTE v177[64]; // [rsp+120h] [rbp-130h] BYREF
  __int64 *v178; // [rsp+160h] [rbp-F0h] BYREF
  __int64 v179; // [rsp+168h] [rbp-E8h] BYREF
  __int64 v180; // [rsp+170h] [rbp-E0h] BYREF
  _BYTE v181[64]; // [rsp+178h] [rbp-D8h] BYREF
  __int64 v182; // [rsp+1B8h] [rbp-98h] BYREF
  _BYTE *v183; // [rsp+1C0h] [rbp-90h]
  _BYTE *v184; // [rsp+1C8h] [rbp-88h]
  __int64 v185; // [rsp+1D0h] [rbp-80h]
  int v186; // [rsp+1D8h] [rbp-78h]
  _BYTE v187[112]; // [rsp+1E0h] [rbp-70h] BYREF

  v6 = a2;
  if ( *((_DWORD *)a2 + 2) == 1 )
  {
    v22 = (__int64)*a2;
    return *(_QWORD *)v22;
  }
  while ( 1 )
  {
    sub_14637D0((__int64 *)v6, a1[8], a1[7], a5, a6);
    v138 = sub_1478130((__int64)a1, 5, v6, a3);
    if ( dword_4F9AE00 < a4 )
      return sub_146E250((__int64)a1, (__int64)a2, v138);
    v9 = *v6;
    v10 = *((_DWORD *)v6 + 2);
    v154 = **v6;
    if ( *(_WORD *)(v154 + 24) )
    {
      v20 = 0;
      v21 = 0;
      if ( !v10 )
        return sub_146E250((__int64)a1, (__int64)a2, v138);
      goto LABEL_17;
    }
    v11 = v9[1];
    v12 = *(_WORD *)(v11 + 24);
    if ( v10 != 2 || v12 != 4 )
      goto LABEL_42;
    if ( *(_QWORD *)(v11 + 40) == 2 )
    {
      v96 = v9[1];
      LOBYTE(v170) = 0;
      v178 = (__int64 *)&v170;
      v179 = (__int64)v181;
      v180 = 0x800000000LL;
      v182 = 0;
      v183 = v187;
      v184 = v187;
      v185 = 8;
      v186 = 0;
      v175 = (__int64 *)v11;
      sub_1412190((__int64)&v182, v96);
      if ( v97 )
      {
        v111 = v175;
        *(_BYTE *)v178 |= *((_WORD *)v175 + 12) == 0;
        if ( (unsigned __int16)(*((_WORD *)v111 + 12) - 4) <= 1u )
          sub_1458920((__int64)&v179, &v175);
      }
      v98 = v179;
      v161 = v11;
      v157 = (__int64)a1;
      v99 = v178;
      v100 = v180;
      v40 = (_BYTE *)v179;
      while ( 2 )
      {
        if ( !v100 || *(_BYTE *)v99 )
        {
          a1 = (_QWORD *)v157;
          v39 = (char)v170;
          if ( v184 != v183 )
          {
            _libc_free((unsigned __int64)v184);
            v40 = (_BYTE *)v179;
          }
          if ( v40 != v181 )
            _libc_free((unsigned __int64)v40);
          if ( v39 )
          {
            v120 = sub_13A5B60(v157, v154, *(_QWORD *)(*(_QWORD *)(v161 + 32) + 8LL), 0, a4 + 1);
            v121 = sub_13A5B60(v157, v154, **(_QWORD **)(v161 + 32), 0, a4 + 1);
            return sub_13A5B00(v157, v121, v120, 0, a4 + 1);
          }
          v9 = *v6;
          v11 = (*v6)[1];
          v12 = *(_WORD *)(v11 + 24);
LABEL_42:
          if ( v12 )
          {
            v154 = *v9;
          }
          else
          {
            v41 = v154;
            do
            {
              sub_16A7B50(&v178, *(_QWORD *)(v41 + 32) + 24LL, *(_QWORD *)(v11 + 32) + 24LL);
              v42 = sub_15E0530(a1[3]);
              v43 = sub_159C0E0(v42, &v178);
              if ( (unsigned int)v179 > 0x40 && v178 )
                j_j___libc_free_0_0(v178);
              v44 = *v6;
              *v44 = sub_145CE20((__int64)a1, v43);
              v9 = *v6;
              v45 = *v6 + 2;
              v46 = *((unsigned int *)v6 + 2);
              v47 = (__int64)&(*v6)[v46];
              if ( (const void *)v47 != v45 )
              {
                memmove(v9 + 1, v45, v47 - (_QWORD)v45);
                v9 = *v6;
                LODWORD(v46) = *((_DWORD *)v6 + 2);
              }
              v48 = v46 - 1;
              v49 = v48 == 1;
              *((_DWORD *)v6 + 2) = v48;
              v41 = *v9;
              if ( v49 )
                return v41;
              v11 = v9[1];
            }
            while ( !*(_WORD *)(v11 + 24) );
            v154 = *v9;
          }
          goto LABEL_7;
        }
        v101 = *(_QWORD *)(v98 + 8LL * v100-- - 8);
        LODWORD(v180) = v100;
        switch ( (__int16)v100 )
        {
          case 0:
          case 1:
            v102 = *(_QWORD **)(v101 + 32);
            v103 = &v102[*(_QWORD *)(v101 + 40)];
            if ( v102 == v103 )
              continue;
            v104 = v102;
            break;
          default:
LABEL_166:
            v99 = v178;
            v98 = v179;
            v100 = v180;
            v40 = (_BYTE *)v179;
            continue;
        }
        break;
      }
      while ( 1 )
      {
        v106 = *v104;
        v107 = v183;
        if ( v184 == v183 )
        {
          v108 = &v183[8 * HIDWORD(v185)];
          if ( v183 != (_BYTE *)v108 )
          {
            v109 = 0;
            do
            {
              if ( v106 == *v107 )
                goto LABEL_156;
              if ( *v107 == -2 )
                v109 = v107;
              ++v107;
            }
            while ( v108 != v107 );
            if ( v109 )
            {
              *v109 = v106;
              --v186;
              ++v182;
              goto LABEL_155;
            }
          }
          if ( HIDWORD(v185) < (unsigned int)v185 )
          {
            ++HIDWORD(v185);
            *v108 = v106;
            ++v182;
LABEL_155:
            *(_BYTE *)v178 |= *(_WORD *)(v106 + 24) == 0;
            if ( (unsigned __int16)(*(_WORD *)(v106 + 24) - 4) <= 1u )
            {
              v110 = (unsigned int)v180;
              if ( (unsigned int)v180 >= HIDWORD(v180) )
              {
                sub_16CD150(&v179, v181, 0, 8);
                v110 = (unsigned int)v180;
              }
              *(_QWORD *)(v179 + 8 * v110) = v106;
              LODWORD(v180) = v180 + 1;
            }
            goto LABEL_156;
          }
        }
        sub_16CCBA0(&v182, *v104);
        if ( v105 )
          goto LABEL_155;
LABEL_156:
        if ( v103 == ++v104 )
          goto LABEL_166;
      }
    }
LABEL_7:
    v13 = *(_QWORD *)(v154 + 32);
    v14 = *(_DWORD *)(v13 + 32);
    v15 = v13 + 24;
    if ( v14 <= 0x40 )
    {
      v17 = *(_QWORD *)(v13 + 24) == 1;
    }
    else
    {
      v16 = sub_16A57B0(v15);
      v15 = v13 + 24;
      v17 = v14 - 1 == v16;
    }
    if ( v17 )
    {
      v88 = *((unsigned int *)v6 + 2);
      if ( v9 + 1 != &v9[v88] )
      {
        memmove(v9, v9 + 1, 8 * v88 - 8);
        LODWORD(v88) = *((_DWORD *)v6 + 2);
      }
      v10 = v88 - 1;
      v20 = 0;
      *((_DWORD *)v6 + 2) = v10;
      goto LABEL_15;
    }
    if ( v14 <= 0x40 )
      v18 = *(_QWORD *)(v13 + 24) == 0;
    else
      v18 = v14 == (unsigned int)sub_16A57B0(v15);
    if ( v18 )
      return v154;
    v19 = sub_1456170(v154);
    v10 = *((_DWORD *)v6 + 2);
    v20 = 1;
    v155 = v19;
    if ( v19 && v10 == 2 )
    {
      v89 = *v6;
      v90 = (*v6)[1];
      v22 = (__int64)*v6;
      v91 = *(_WORD *)(v90 + 24);
      if ( v91 != 4 )
      {
        if ( v91 == 7 )
        {
          v178 = &v180;
          v179 = 0x400000000LL;
          v112 = *(__int64 **)(v90 + 32);
          v113 = *(_QWORD *)(v90 + 40);
          if ( v112 != &v112[v113] )
          {
            v168 = v90;
            v114 = &v112[v113];
            v115 = a4 + 1;
            while ( 1 )
            {
              v116 = *v112;
              v164 = v115;
              ++v112;
              v175 = (__int64 *)sub_13A5B60((__int64)a1, *v89, v116, 0, v115);
              sub_1458920((__int64)&v178, &v175);
              if ( v114 == v112 )
                break;
              v115 = v164;
              v89 = *a2;
            }
            v90 = v168;
          }
          v117 = sub_14785F0((__int64)a1, &v178, *(_QWORD *)(v90 + 48), *(_WORD *)(v90 + 26) & 1);
          v118 = v178;
          v154 = v117;
          if ( v178 != &v180 )
            goto LABEL_214;
          return v154;
        }
        v21 = 1;
        goto LABEL_18;
      }
      v178 = &v180;
      v179 = 0x400000000LL;
      v92 = *(__int64 **)(v90 + 32);
      v167 = &v92[*(_QWORD *)(v90 + 40)];
      v93 = a4 + 1;
      if ( v92 != v167 )
      {
        v160 = v6;
        v94 = v92;
        v163 = 0;
        while ( 1 )
        {
          v95 = *v94++;
          v175 = (__int64 *)sub_13A5B60((__int64)a1, *v89, v95, 0, v93);
          if ( *((_WORD *)v175 + 12) == 5 )
          {
            sub_1458920((__int64)&v178, &v175);
            if ( v167 == v94 )
            {
              v6 = v160;
              if ( v163 )
              {
LABEL_218:
                v119 = sub_147DD40((__int64)a1, (__int64 *)&v178, 0, v93, a5, a6);
                v118 = v178;
                v154 = (__int64)v119;
                if ( v178 != &v180 )
                  goto LABEL_214;
                return v154;
              }
              if ( v178 != &v180 )
                _libc_free((unsigned __int64)v178);
              v10 = *((_DWORD *)v160 + 2);
              v20 = 1;
              goto LABEL_15;
            }
          }
          else
          {
            sub_1458920((__int64)&v178, &v175);
            if ( v167 == v94 )
              goto LABEL_218;
            v163 = v155;
          }
          v89 = *v160;
        }
      }
    }
    else
    {
LABEL_15:
      if ( v10 == 1 )
        return **a2;
    }
    v21 = v20;
    if ( v20 >= v10 )
      return sub_146E250((__int64)a1, (__int64)a2, v138);
LABEL_17:
    v22 = (__int64)*v6;
LABEL_18:
    for ( i = v20 + 1; ; ++i )
    {
      v24 = 8 * v21;
      v25 = v22 + v24;
      v26 = *(_WORD *)(*(_QWORD *)(v22 + v24) + 24LL);
      if ( v26 > 4u )
        break;
      ++v20;
      v21 = i;
      if ( v20 == v10 )
        return sub_146E250((__int64)a1, (__int64)a2, v138);
    }
    if ( v26 != 5 )
      break;
    v27 = v22;
    v28 = 0;
    v29 = v20;
    v30 = *(_QWORD *)(v22 + v24);
    while ( dword_4F9B340 >= v10 )
    {
      v31 = v27 + 8LL * v10;
      if ( v31 != v25 + 8 )
      {
        memmove((void *)v25, (const void *)(v25 + 8), v31 - (v25 + 8));
        v10 = *((_DWORD *)v6 + 2);
      }
      v32 = v10 - 1;
      v33 = *((unsigned int *)v6 + 3);
      *((_DWORD *)v6 + 2) = v32;
      v34 = *(const void **)(v30 + 32);
      v35 = 8LL * *(_QWORD *)(v30 + 40);
      v36 = v35 >> 3;
      if ( v35 >> 3 > (unsigned __int64)(v33 - v32) )
      {
        v159 = v35;
        v162 = v34;
        sub_16CD150(v6, v6 + 2, v36 + v32, 8);
        v32 = *((unsigned int *)v6 + 2);
        v35 = v159;
        v34 = v162;
      }
      v27 = (__int64)*v6;
      if ( v35 )
      {
        memcpy((void *)(v27 + 8 * v32), v34, v35);
        v27 = (__int64)*v6;
        LODWORD(v32) = *((_DWORD *)v6 + 2);
      }
      v37 = v36 + v32;
      v25 = v27 + v24;
      *((_DWORD *)v6 + 2) = v37;
      v30 = *(_QWORD *)(v27 + v24);
      v10 = v37;
      v28 = 1;
      if ( *(_WORD *)(v30 + 24) != 5 )
      {
        v22 = v27;
        goto LABEL_33;
      }
    }
    v20 = v29;
    v22 = v27;
    if ( !v28 )
      break;
LABEL_33:
    ++a4;
    a3 = 0;
    if ( v10 == 1 )
      return *(_QWORD *)v22;
  }
  v50 = v20;
  v51 = v20;
  if ( v10 <= v20 )
    return sub_146E250((__int64)a1, (__int64)a2, v138);
  v52 = v22;
  v53 = v20 + 1;
  while ( *(_WORD *)(*(_QWORD *)(v22 + 8 * v50) + 24LL) <= 6u )
  {
    ++v51;
    v50 = v53;
    if ( v51 == v10 )
      return sub_146E250((__int64)a1, (__int64)a2, v138);
    ++v53;
  }
  v141 = v51;
  v136 = v51;
  if ( v10 <= v51 )
    return sub_146E250((__int64)a1, (__int64)a2, v138);
  v54 = v10;
LABEL_60:
  v133 = v136;
  v55 = (__int64 *)(v22 + 8 * v136);
  if ( *(_WORD *)(*v55 + 24) != 7 )
    return sub_146E250((__int64)a1, (__int64)a2, v138);
  v178 = &v180;
  v179 = 0x800000000LL;
  v154 = *v55;
  v137 = *(_QWORD *)(*v55 + 48);
  if ( v54 )
  {
    v56 = 0;
    while ( 1 )
    {
      v57 = v56;
      if ( sub_146D950((__int64)a1, *(_QWORD *)(v52 + v57 * 8), v137) )
      {
        sub_1458920((__int64)&v178, &(*a2)[v57]);
        v58 = (__int64)&(*a2)[v57];
        v59 = (__int64)&(*a2)[*((unsigned int *)a2 + 2)];
        v60 = *((_DWORD *)a2 + 2);
        if ( v59 != v58 + 8 )
        {
          memmove((void *)v58, (const void *)(v58 + 8), v59 - (v58 + 8));
          v60 = *((_DWORD *)a2 + 2);
        }
        --v54;
        *((_DWORD *)a2 + 2) = v60 - 1;
        if ( v54 == v56 )
        {
LABEL_69:
          if ( (_DWORD)v179 )
          {
            v175 = (__int64 *)v177;
            v176 = 0x400000000LL;
            v122 = *(_QWORD *)(v154 + 40);
            if ( v122 > 4 )
              sub_16CD150(&v175, v177, v122, 8);
            v123 = a4 + 1;
            v124 = sub_147EE30(a1, &v178, 0, a4 + 1);
            v125 = *(_QWORD *)(v154 + 40);
            if ( (_DWORD)v125 )
            {
              v126 = a4 + 1;
              v127 = 0;
              v128 = v124;
              v169 = 8LL * (unsigned int)v125;
              do
              {
                v165 = v126;
                v129 = *(_QWORD *)(*(_QWORD *)(v154 + 32) + v127);
                v127 += 8;
                v170 = (__int64 *)sub_13A5B60((__int64)a1, v128, v129, 0, v126);
                sub_1458920((__int64)&v175, &v170);
                v126 = v165;
              }
              while ( v127 != v169 );
              v123 = v165;
            }
            v130 = sub_14785F0((__int64)a1, &v175, v137, (unsigned __int16)(v138 & *(_WORD *)(v154 + 26)) & 0xFFFE);
            if ( *((_DWORD *)a2 + 2) != 1 )
            {
              LODWORD(v131) = 0;
              v132 = *a2;
              if ( v154 != **a2 )
              {
                do
                  v131 = (unsigned int)(v131 + 1);
                while ( v154 != v132[v131] );
                v132 += v131;
              }
              *v132 = v130;
              v130 = sub_147EE30(a1, a2, 0, v123);
            }
            if ( v175 != (__int64 *)v177 )
              _libc_free((unsigned __int64)v175);
            v154 = v130;
            goto LABEL_213;
          }
          v136 = ++v141;
          if ( v141 != *((_DWORD *)a2 + 2) )
          {
            v22 = (__int64)*a2;
            goto LABEL_72;
          }
LABEL_206:
          if ( v178 != &v180 )
            _libc_free((unsigned __int64)v178);
LABEL_208:
          v54 = *((_DWORD *)a2 + 2);
          if ( v141 >= v54 )
            return sub_146E250((__int64)a1, (__int64)a2, v138);
          v22 = (__int64)*a2;
          v52 = (__int64)*a2;
          goto LABEL_60;
        }
      }
      else if ( v54 == ++v56 )
      {
        goto LABEL_69;
      }
      v52 = (__int64)*a2;
    }
  }
  v136 = ++v141;
  if ( v141 == *((_DWORD *)a2 + 2) )
    goto LABEL_208;
LABEL_72:
  v152 = 0;
  v61 = (__int64)a1;
  v139 = v141;
  v62 = v136;
  while ( 1 )
  {
    v135 = v62;
    v153 = *(_QWORD *)(v22 + 8 * v62);
    if ( *(_WORD *)(v153 + 24) != 7 )
    {
LABEL_205:
      a1 = (_QWORD *)v61;
      if ( v152 )
        goto LABEL_212;
      goto LABEL_206;
    }
    if ( v137 == *(_QWORD *)(v153 + 48)
      && *(_QWORD *)(v154 + 40) + *(_QWORD *)(v153 + 40) - 1LL <= (unsigned __int64)(unsigned int)dword_4F9AB60 )
    {
      break;
    }
    ++v139;
LABEL_74:
    v62 = v139;
    if ( *((_DWORD *)a2 + 2) == v139 )
      goto LABEL_205;
    v22 = (__int64)*a2;
  }
  v148 = sub_1456040(**(_QWORD **)(v154 + 32));
  v145 = sub_1456C90(v61, v148);
  v175 = (__int64 *)v177;
  v176 = 0x700000000LL;
  v63 = *(_QWORD *)(v154 + 40);
  if ( (_DWORD)v63 + (unsigned int)*(_QWORD *)(v153 + 40) == 1 )
    goto LABEL_202;
  v144 = 0;
  v134 = (unsigned int)v63 + (unsigned int)*(_QWORD *)(v153 + 40) - 2;
  while ( 2 )
  {
    v158 = sub_145CF80(v61, v148, 0, 0);
    v143 = v144 + 1;
    v142 = 2 * v144;
    if ( 2 * (_DWORD)v144 + 1 == (_DWORD)v144 )
    {
LABEL_200:
      v66 = 0;
      goto LABEL_119;
    }
    v156 = v144;
    v147 = v144;
    v64 = *(_QWORD *)(v154 + 40);
    v65 = *(_QWORD *)(v153 + 40);
    while ( 2 )
    {
      while ( 2 )
      {
        v66 = v144 == 0 || v156 == v144;
        if ( v66 )
          goto LABEL_189;
        if ( v156 > v144 )
        {
          v149 = 0;
          goto LABEL_190;
        }
        if ( v156 > v144 >> 1 )
        {
          v67 = v144 - v156;
          goto LABEL_87;
        }
        if ( !v156 )
        {
LABEL_189:
          v149 = 1;
LABEL_190:
          v74 = v147 - v64 + 1;
          if ( v74 < v147 - (int)v144 )
            v74 = v147 - v144;
          v73 = v144 + 1;
          if ( v143 >= (int)v65 )
            v73 = v65;
          if ( v74 < v73 )
            goto LABEL_98;
          if ( v147 == v142 )
            goto LABEL_200;
          ++v147;
          --v156;
          continue;
        }
        break;
      }
      v67 = v156;
LABEL_87:
      v68 = v144;
      v69 = 1;
      v70 = 1;
      do
      {
        v71 = v69 * v68;
        if ( v68 > 1 && v71 / v68 != v69 )
          v66 = 1;
        --v68;
        v72 = v71 / v70++;
        v69 = v72;
      }
      while ( v70 <= v67 );
      v149 = v72;
      v73 = v65;
      v74 = v147 - v64 + 1;
      if ( v74 < v147 - (int)v144 )
        v74 = v147 - v144;
      if ( v143 < (int)v65 )
        v73 = v144 + 1;
      if ( v73 <= v74 )
      {
LABEL_186:
        if ( v147 == v142 )
          goto LABEL_119;
        ++v147;
        --v156;
        if ( v66 )
          goto LABEL_224;
        v64 = *(_QWORD *)(v154 + 40);
        v65 = *(_QWORD *)(v153 + 40);
        continue;
      }
      break;
    }
    if ( v66 )
      goto LABEL_119;
LABEL_98:
    v75 = v144 - v74;
    v166 = a4 + 1;
    v76 = 8LL * v74;
    v151 = v75 - (unsigned int)(v73 - 1 - v74);
    while ( 2 )
    {
      v66 = v156 == 0 || v75 == v156;
      if ( v66 )
      {
        v77 = v149;
        v66 = 0;
      }
      else
      {
        v77 = 0;
        if ( v75 <= v156 )
        {
          if ( v75 <= v156 >> 1 )
          {
            v77 = v149;
            if ( v75 )
            {
              v78 = v75;
              goto LABEL_103;
            }
          }
          else
          {
            v78 = v156 - v75;
LABEL_103:
            v79 = v156;
            v80 = 1;
            v81 = 1;
            do
            {
              v82 = v80 * v79;
              if ( v79 > 1 && v82 / v79 != v80 )
                v66 = 1;
              --v79;
              v83 = v82 / v81++;
              v80 = v83;
            }
            while ( v81 <= v78 );
            v77 = v83 * v149;
            if ( v145 > 0x40 && v83 > 1 && v77 / v83 != v149 )
              v66 = 1;
          }
        }
      }
      v84 = sub_145CF80(v61, v148, v77, 0);
      v85 = *(_QWORD *)(*(_QWORD *)(v153 + 32) + v76);
      v173 = *(_QWORD *)(*(_QWORD *)(v154 + 32) + 8LL * (unsigned int)(v75 + v147 - v144));
      v170 = &v172;
      v174 = v85;
      v172 = v84;
      v171 = 0x300000003LL;
      v86 = sub_147EE30(v61, &v170, 0, v166);
      if ( v170 != &v172 )
      {
        v150 = v86;
        _libc_free((unsigned __int64)v170);
        v86 = v150;
      }
      v173 = v86;
      v170 = &v172;
      v172 = v158;
      v171 = 0x200000002LL;
      v158 = (__int64)sub_147DD40(v61, (__int64 *)&v170, 0, v166, a5, a6);
      if ( v170 != &v172 )
        _libc_free((unsigned __int64)v170);
      if ( v75 == v151 )
        goto LABEL_186;
      --v75;
      v76 += 8;
      if ( !v66 )
        continue;
      break;
    }
    if ( v147 == v142 )
      goto LABEL_119;
LABEL_224:
    v66 = 1;
LABEL_119:
    v87 = (unsigned int)v176;
    if ( (unsigned int)v176 >= HIDWORD(v176) )
    {
      sub_16CD150(&v175, v177, 0, 8);
      v87 = (unsigned int)v176;
    }
    v175[v87] = v158;
    LODWORD(v176) = v176 + 1;
    if ( v134 != v144 )
    {
      ++v144;
      if ( v66 )
        goto LABEL_123;
      continue;
    }
    break;
  }
  if ( v66 )
  {
LABEL_123:
    ++v139;
LABEL_124:
    if ( v175 != (__int64 *)v177 )
      _libc_free((unsigned __int64)v175);
    goto LABEL_74;
  }
LABEL_202:
  v154 = sub_14785F0(v61, &v175, *(_QWORD *)(v154 + 48), 0);
  if ( *((_DWORD *)a2 + 2) == 2 )
  {
    if ( v175 != (__int64 *)v177 )
      _libc_free((unsigned __int64)v175);
  }
  else
  {
    (*a2)[v133] = v154;
    sub_1453AF0((__int64)a2, (char *)&(*a2)[v135]);
    if ( *(_WORD *)(v154 + 24) == 7 )
    {
      v152 = 1;
      goto LABEL_124;
    }
    a1 = (_QWORD *)v61;
    if ( v175 != (__int64 *)v177 )
      _libc_free((unsigned __int64)v175);
LABEL_212:
    v154 = sub_147EE30(a1, a2, 0, a4 + 1);
  }
LABEL_213:
  v118 = v178;
  if ( v178 != &v180 )
LABEL_214:
    _libc_free((unsigned __int64)v118);
  return v154;
}
