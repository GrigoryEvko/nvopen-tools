// Function: sub_1CA2920
// Address: 0x1ca2920
//
__int64 __fastcall sub_1CA2920(
        _QWORD *a1,
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
  double v12; // xmm4_8
  double v13; // xmm5_8
  const char **v14; // rbx
  const char **v15; // rax
  const char *v16; // r14
  double v17; // xmm4_8
  double v18; // xmm5_8
  int v19; // esi
  int *v20; // rax
  int *v21; // r8
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  int *v25; // rax
  __int64 v26; // rdx
  _BOOL8 v27; // rdi
  __int64 *v28; // r12
  __int64 v29; // rbx
  char v30; // al
  unsigned __int64 v31; // r14
  unsigned __int8 v32; // al
  __int64 *v33; // rsi
  __int64 *v34; // rbx
  __int64 v35; // rdx
  __int64 v36; // rsi
  int v37; // r9d
  _BYTE *v38; // rax
  __int64 v39; // rcx
  unsigned __int64 v40; // rdx
  __int64 v41; // rdx
  int v42; // r9d
  _BYTE *v43; // rax
  __int64 v44; // rcx
  unsigned __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rdx
  __int64 v48; // r12
  __int64 v49; // r12
  char v50; // al
  __int64 v51; // r11
  __int64 v52; // rsi
  char v53; // al
  _BYTE *v54; // r11
  int v55; // r9d
  _BYTE *v56; // rax
  _QWORD *v57; // r12
  __int64 v58; // rcx
  unsigned __int64 v59; // rdx
  __int64 v60; // rdx
  unsigned __int64 v61; // rsi
  __int64 *v62; // rbx
  __int64 v63; // r14
  unsigned __int64 v64; // rdx
  unsigned int v65; // eax
  _BYTE *v66; // rax
  _QWORD *v67; // rdx
  __int64 v68; // rsi
  unsigned __int64 v69; // rcx
  __int64 v70; // rcx
  __int64 v71; // rax
  int v72; // esi
  __int64 *v73; // rdi
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  bool v78; // zf
  __int64 v79; // rdx
  __int64 v80; // rcx
  unsigned __int64 v81; // rax
  __int64 v82; // rcx
  int *v83; // rax
  int *v84; // rsi
  __int64 v85; // rcx
  __int64 v86; // rdx
  int *v87; // rax
  int *v88; // rsi
  __int64 v89; // rcx
  __int64 v90; // rdx
  __int64 v91; // rax
  __m128i si128; // xmm0
  __m128i v93; // xmm0
  __int64 v94; // r11
  __int64 v95; // rsi
  char v96; // al
  _BYTE *v97; // r11
  int v98; // r9d
  _BYTE *v99; // rax
  _QWORD *v100; // r12
  __int64 v101; // rcx
  unsigned __int64 v102; // rdx
  __int64 v103; // rdx
  __int64 v104; // rdx
  _QWORD *v105; // r12
  int *v107; // rax
  int *v108; // rsi
  __int64 v109; // rcx
  __int64 v110; // rdx
  __int64 v111; // rax
  int *v112; // rax
  int *v113; // rsi
  __int64 v114; // rcx
  __int64 v115; // rdx
  __int64 v116; // rax
  __int64 v117; // r11
  __int64 v118; // rsi
  char v119; // al
  __int64 v120; // rdx
  __int64 v121; // rax
  __int64 v122; // r11
  __int64 v123; // rsi
  char v124; // al
  _BYTE *v125; // r11
  int v126; // r9d
  const char *v127; // rsi
  int *v128; // rax
  int *v129; // r8
  __int64 v130; // rcx
  __int64 v131; // rdx
  __int64 v132; // rax
  __int64 v133; // rax
  unsigned int v134; // eax
  __int64 v135; // rdx
  int v136; // r14d
  _BYTE *v137; // rax
  unsigned __int64 v138; // rax
  __int64 v139; // rax
  __int64 *v140; // rsi
  int *v141; // rax
  int *v142; // r8
  __int64 v143; // rax
  __int64 v144; // rcx
  unsigned __int64 v145; // rsi
  __int64 v146; // rcx
  int *v147; // rax
  int *v148; // rsi
  int *v149; // rdi
  __int64 v150; // rax
  int *v151; // [rsp+0h] [rbp-170h]
  int *v152; // [rsp+8h] [rbp-168h]
  unsigned __int8 v153; // [rsp+8h] [rbp-168h]
  int *v154; // [rsp+8h] [rbp-168h]
  _BYTE *v155; // [rsp+18h] [rbp-158h]
  const char **v156; // [rsp+20h] [rbp-150h]
  __int64 *v157; // [rsp+20h] [rbp-150h]
  __int64 *v158; // [rsp+20h] [rbp-150h]
  _BYTE *v159; // [rsp+28h] [rbp-148h]
  int v160; // [rsp+30h] [rbp-140h]
  _BYTE *v161; // [rsp+30h] [rbp-140h]
  _BYTE *v162; // [rsp+30h] [rbp-140h]
  _BYTE *v163; // [rsp+30h] [rbp-140h]
  _BYTE *v164; // [rsp+30h] [rbp-140h]
  unsigned int v165; // [rsp+30h] [rbp-140h]
  __int64 *v166; // [rsp+38h] [rbp-138h]
  __int64 *v167; // [rsp+38h] [rbp-138h]
  unsigned int v168; // [rsp+4Ch] [rbp-124h] BYREF
  unsigned int v169; // [rsp+50h] [rbp-120h] BYREF
  unsigned int v170; // [rsp+54h] [rbp-11Ch] BYREF
  __int64 v171; // [rsp+58h] [rbp-118h] BYREF
  __int64 *v172; // [rsp+60h] [rbp-110h] BYREF
  __int64 *v173; // [rsp+68h] [rbp-108h]
  __int64 *v174; // [rsp+70h] [rbp-100h]
  unsigned __int64 v175; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v176; // [rsp+88h] [rbp-E8h]
  __int64 v177[4]; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v178; // [rsp+B0h] [rbp-C0h] BYREF
  int v179; // [rsp+B8h] [rbp-B8h] BYREF
  int *v180; // [rsp+C0h] [rbp-B0h]
  int *v181; // [rsp+C8h] [rbp-A8h]
  int *v182; // [rsp+D0h] [rbp-A0h]
  __int64 v183; // [rsp+D8h] [rbp-98h]
  const char *v184; // [rsp+E0h] [rbp-90h] BYREF
  unsigned __int64 v185; // [rsp+E8h] [rbp-88h] BYREF
  __int64 v186[4]; // [rsp+F0h] [rbp-80h] BYREF
  unsigned __int64 *v187; // [rsp+110h] [rbp-60h] BYREF
  __int64 v188; // [rsp+118h] [rbp-58h] BYREF
  __int64 v189; // [rsp+120h] [rbp-50h] BYREF
  __int64 *v190; // [rsp+128h] [rbp-48h]
  __int64 *v191; // [rsp+130h] [rbp-40h]
  __int64 v192; // [rsp+138h] [rbp-38h]

  sub_1C995B0(a1, a2);
  v14 = (const char **)a1[9];
  v179 = 0;
  v181 = &v179;
  v182 = &v179;
  v15 = (const char **)a1[10];
  v180 = 0;
  v183 = 0;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v156 = v15;
  while ( v156 != v14 )
  {
    v16 = *v14;
    LODWORD(v188) = 0;
    v189 = 0;
    v190 = &v188;
    v191 = &v188;
    v192 = 0;
    LODWORD(v171) = *(_DWORD *)(**((_QWORD **)v16 - 6) + 8LL) >> 8;
    LODWORD(v175) = v171;
    if ( (unsigned int)sub_1C9F820(
                         (__int64)a1,
                         a2,
                         *((_QWORD *)v16 - 6),
                         &v178,
                         &v187,
                         (int *)&v171,
                         a3,
                         a4,
                         a5,
                         a6,
                         v12,
                         v13,
                         a9,
                         a10) == 1
      && (unsigned int)sub_1C9F820(
                         (__int64)a1,
                         a2,
                         *((_QWORD *)v16 - 3),
                         &v178,
                         &v187,
                         (int *)&v175,
                         a3,
                         a4,
                         a5,
                         a6,
                         v17,
                         v18,
                         a9,
                         a10) == 1 )
    {
      v19 = v171;
      if ( (_DWORD)v171 == (_DWORD)v175 )
      {
        v20 = v180;
        v21 = &v179;
        if ( !v180 )
          goto LABEL_14;
        do
        {
          while ( 1 )
          {
            v22 = *((_QWORD *)v20 + 2);
            v23 = *((_QWORD *)v20 + 3);
            if ( *((_QWORD *)v20 + 4) >= (unsigned __int64)v16 )
              break;
            v20 = (int *)*((_QWORD *)v20 + 3);
            if ( !v23 )
              goto LABEL_12;
          }
          v21 = v20;
          v20 = (int *)*((_QWORD *)v20 + 2);
        }
        while ( v22 );
LABEL_12:
        if ( v21 == &v179 || *((_QWORD *)v21 + 4) > (unsigned __int64)v16 )
        {
LABEL_14:
          v151 = v21;
          v24 = sub_22077B0(48);
          *(_QWORD *)(v24 + 32) = v16;
          *(_DWORD *)(v24 + 40) = 0;
          v152 = (int *)v24;
          v25 = (int *)sub_1C70330(&v178, v151, (unsigned __int64 *)(v24 + 32));
          if ( v26 )
          {
            v27 = &v179 == (int *)v26 || v25 || (unsigned __int64)v16 < *(_QWORD *)(v26 + 32);
            sub_220F040(v27, v152, v26, &v179);
            ++v183;
            v21 = v152;
          }
          else
          {
            v149 = v152;
            v154 = v25;
            j_j___libc_free_0(v149, 48);
            v21 = v154;
          }
          v19 = v171;
        }
        v21[10] = v19;
        v140 = v173;
        v184 = v16;
        if ( v173 == v174 )
        {
          sub_17C2330((__int64)&v172, v173, &v184);
        }
        else
        {
          if ( v173 )
          {
            *v173 = (__int64)v16;
            v140 = v173;
          }
          v173 = v140 + 1;
        }
      }
    }
    ++v14;
    sub_1C97470(v189);
  }
  v28 = (__int64 *)a1[3];
  v166 = (__int64 *)a1[4];
  if ( v28 != v166 )
  {
    while ( 1 )
    {
      v29 = *v28;
      v168 = 0;
      v171 = v29;
      v30 = *(_BYTE *)(v29 + 16);
      switch ( v30 )
      {
        case '6':
          v31 = *(_QWORD *)(v29 - 24);
          v29 = 0;
          break;
        case '7':
          v31 = *(_QWORD *)(v29 - 24);
          break;
        case 'N':
          v121 = *(_QWORD *)(v29 - 24);
          if ( *(_BYTE *)(v121 + 16)
            || (*(_BYTE *)(v121 + 33) & 0x20) == 0
            || !(unsigned __int8)sub_1C98880((__int64)a1, *(_DWORD *)(v121 + 36), &v168) )
          {
            goto LABEL_33;
          }
          v31 = *(_QWORD *)(v29 + 24 * (v168 - (unsigned __int64)(*(_DWORD *)(v29 + 20) & 0xFFFFFFF)));
          v29 = 0;
          break;
        case ':':
          v31 = *(_QWORD *)(v29 - 72);
          v29 = 0;
          break;
        case ';':
          v31 = *(_QWORD *)(v29 - 48);
          v29 = 0;
          break;
        default:
          goto LABEL_33;
      }
      v169 = *(_DWORD *)(*(_QWORD *)v31 + 8LL) >> 8;
      if ( !v169 )
        break;
LABEL_33:
      if ( v166 == ++v28 )
        goto LABEL_34;
    }
    LODWORD(v188) = 0;
    v189 = 0;
    v190 = &v188;
    v191 = &v188;
    v192 = 0;
    v160 = sub_1C9F820((__int64)a1, a2, v31, &v178, &v187, (int *)&v169, a3, a4, a5, a6, v12, v13, a9, a10);
    v170 = 0;
    if ( (unsigned __int8)sub_1C98370(a1, v171, v31, &v170) )
    {
      v169 = v170;
    }
    else if ( v160 != 1 )
    {
      if ( unk_4FBE1ED && *(_BYTE *)a1 )
      {
        v176 = 0;
        v175 = (unsigned __int64)v177;
        LOBYTE(v177[0]) = 0;
        v158 = (__int64 *)(v171 + 48);
        sub_15E0530(a2);
        sub_1C315E0((__int64)&v184, v158);
        sub_2241490(&v175, v184, v185);
        if ( v184 != (const char *)v186 )
          j_j___libc_free_0(v184, v186[0] + 1);
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v176) <= 0x4A )
          sub_4262D8((__int64)"basic_string::append");
        sub_2241490(&v175, ": Warning: Cannot tell what pointer points to, assuming global memory space", 75);
        sub_1C3F040((__int64)&v175);
        if ( byte_4FBE000 && byte_4FBDF20 )
          sub_1CCC570(v31);
        if ( (__int64 *)v175 != v177 )
          j_j___libc_free_0(v175, v177[0] + 1);
      }
      goto LABEL_32;
    }
    v32 = *(_BYTE *)(v171 + 16);
    if ( v32 > 0x17u )
    {
      if ( v32 == 78 )
      {
        v120 = *(_QWORD *)(v171 - 24);
        if ( !*(_BYTE *)(v120 + 16) && (*(_BYTE *)(v120 + 33) & 0x20) != 0 )
        {
          if ( !sub_1C98920((__int64)a1, *(_DWORD *)(v120 + 36)) && !v29 )
          {
            v33 = v173;
            if ( v173 != v174 )
              goto LABEL_29;
            goto LABEL_174;
          }
LABEL_27:
          if ( v169 != 4 )
            goto LABEL_28;
LABEL_126:
          v175 = 71;
          v184 = (const char *)v186;
          v91 = sub_22409D0(&v184, &v175, 0);
          v184 = (const char *)v91;
          v186[0] = v175;
          *(__m128i *)v91 = _mm_load_si128((const __m128i *)&xmmword_42DFCC0);
          si128 = _mm_load_si128((const __m128i *)&xmmword_42DFCD0);
          *(_DWORD *)(v91 + 64) = 1886593145;
          *(__m128i *)(v91 + 16) = si128;
          v93 = _mm_load_si128((const __m128i *)&xmmword_42DFCE0);
          *(_WORD *)(v91 + 68) = 25441;
          *(__m128i *)(v91 + 32) = v93;
          a3 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42DFCF0);
          *(_BYTE *)(v91 + 70) = 101;
          *(__m128 *)(v91 + 48) = a3;
          v185 = v175;
          v184[v175] = 0;
          sub_1C979E0(v171, (__int64)&v184);
          if ( v184 != (const char *)v186 )
            j_j___libc_free_0(v184, v186[0] + 1);
          goto LABEL_32;
        }
      }
      else if ( v32 == 58 )
      {
        if ( v169 == 4 )
          goto LABEL_126;
        goto LABEL_28;
      }
    }
    if ( v29 || v32 == 59 )
      goto LABEL_27;
LABEL_28:
    v33 = v173;
    if ( v173 != v174 )
    {
LABEL_29:
      if ( v33 )
      {
        *v33 = v171;
        v33 = v173;
      }
      v173 = v33 + 1;
      goto LABEL_32;
    }
LABEL_174:
    sub_170B610((__int64)&v172, v33, &v171);
LABEL_32:
    sub_1C97470(v189);
    goto LABEL_33;
  }
LABEL_34:
  v34 = v172;
  LODWORD(v185) = 0;
  v186[1] = (__int64)&v185;
  v186[2] = (__int64)&v185;
  v186[0] = 0;
  v186[3] = 0;
  v167 = v173;
  if ( v172 == v173 )
  {
    v153 = 0;
    goto LABEL_69;
  }
  do
  {
    while ( 1 )
    {
      v49 = *v34;
      v50 = *(_BYTE *)(*v34 + 16);
      switch ( v50 )
      {
        case 'K':
          v35 = *(_QWORD *)(v49 - 48);
          v36 = *v34;
          LODWORD(v171) = 0;
          v175 = v35;
          if ( (unsigned __int8)sub_1C98370(a1, v36, v35, (unsigned int *)&v171) )
          {
            v37 = v171;
LABEL_38:
            v38 = sub_1CA1B70(a1, a2, *(_BYTE **)(v49 - 48), v49, &v184, v37, 0);
            if ( *(_QWORD *)(v49 - 48) )
            {
              v39 = *(_QWORD *)(v49 - 40);
              v40 = *(_QWORD *)(v49 - 32) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v40 = v39;
              if ( v39 )
                *(_QWORD *)(v39 + 16) = *(_QWORD *)(v39 + 16) & 3LL | v40;
            }
            *(_QWORD *)(v49 - 48) = v38;
            if ( v38 )
            {
              v41 = *((_QWORD *)v38 + 1);
              *(_QWORD *)(v49 - 40) = v41;
              if ( v41 )
                *(_QWORD *)(v41 + 16) = (v49 - 40) | *(_QWORD *)(v41 + 16) & 3LL;
              *(_QWORD *)(v49 - 32) = (unsigned __int64)(v38 + 8) | *(_QWORD *)(v49 - 32) & 3LL;
              *((_QWORD *)v38 + 1) = v49 - 48;
            }
            v175 = *(_QWORD *)(v49 - 24);
            LODWORD(v171) = 0;
            if ( (unsigned __int8)sub_1C98370(a1, v49, v175, (unsigned int *)&v171) )
            {
              v42 = v171;
              goto LABEL_47;
            }
            v83 = v180;
            if ( v180 )
            {
              v84 = &v179;
              do
              {
                while ( 1 )
                {
                  v85 = *((_QWORD *)v83 + 2);
                  v86 = *((_QWORD *)v83 + 3);
                  if ( *((_QWORD *)v83 + 4) >= v175 )
                    break;
                  v83 = (int *)*((_QWORD *)v83 + 3);
                  if ( !v86 )
                    goto LABEL_110;
                }
                v84 = v83;
                v83 = (int *)*((_QWORD *)v83 + 2);
              }
              while ( v85 );
LABEL_110:
              if ( v84 != &v179 && *((_QWORD *)v84 + 4) <= v175 )
                goto LABEL_113;
            }
            else
            {
              v84 = &v179;
            }
            v187 = &v175;
            v84 = (int *)sub_1C9E550(&v178, v84, &v187);
LABEL_113:
            v42 = v84[10];
LABEL_47:
            v43 = sub_1CA1B70(a1, a2, *(_BYTE **)(v49 - 24), v49, &v184, v42, 0);
            if ( *(_QWORD *)(v49 - 24) )
            {
              v44 = *(_QWORD *)(v49 - 16);
              v45 = *(_QWORD *)(v49 - 8) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v45 = v44;
              if ( v44 )
                *(_QWORD *)(v44 + 16) = *(_QWORD *)(v44 + 16) & 3LL | v45;
            }
            *(_QWORD *)(v49 - 24) = v43;
            if ( v43 )
            {
              v46 = *((_QWORD *)v43 + 1);
              *(_QWORD *)(v49 - 16) = v46;
              if ( v46 )
                *(_QWORD *)(v46 + 16) = (v49 - 16) | *(_QWORD *)(v46 + 16) & 3LL;
              v47 = *(_QWORD *)(v49 - 8);
              v48 = v49 - 24;
              *(_QWORD *)(v48 + 16) = (unsigned __int64)(v43 + 8) | v47 & 3;
              *((_QWORD *)v43 + 1) = v48;
            }
            goto LABEL_54;
          }
          v87 = v180;
          if ( v180 )
          {
            v88 = &v179;
            do
            {
              while ( 1 )
              {
                v89 = *((_QWORD *)v87 + 2);
                v90 = *((_QWORD *)v87 + 3);
                if ( *((_QWORD *)v87 + 4) >= v175 )
                  break;
                v87 = (int *)*((_QWORD *)v87 + 3);
                if ( !v90 )
                  goto LABEL_119;
              }
              v88 = v87;
              v87 = (int *)*((_QWORD *)v87 + 2);
            }
            while ( v89 );
LABEL_119:
            if ( v88 != &v179 && *((_QWORD *)v88 + 4) <= v175 )
              goto LABEL_122;
          }
          else
          {
            v88 = &v179;
          }
          v187 = &v175;
          v88 = (int *)sub_1C9E550(&v178, v88, &v187);
LABEL_122:
          v37 = v88[10];
          goto LABEL_38;
        case '6':
          v51 = *(_QWORD *)(v49 - 24);
          v52 = *v34;
          LODWORD(v171) = 0;
          v175 = v51;
          v161 = (_BYTE *)v51;
          v53 = sub_1C98370(a1, v52, v51, (unsigned int *)&v171);
          v54 = v161;
          if ( v53 )
          {
            v55 = v171;
            goto LABEL_59;
          }
          v107 = v180;
          if ( v180 )
          {
            v108 = &v179;
            do
            {
              while ( 1 )
              {
                v109 = *((_QWORD *)v107 + 2);
                v110 = *((_QWORD *)v107 + 3);
                if ( *((_QWORD *)v107 + 4) >= v175 )
                  break;
                v107 = (int *)*((_QWORD *)v107 + 3);
                if ( !v110 )
                  goto LABEL_149;
              }
              v108 = v107;
              v107 = (int *)*((_QWORD *)v107 + 2);
            }
            while ( v109 );
LABEL_149:
            if ( v108 != &v179 && *((_QWORD *)v108 + 4) <= v175 )
              goto LABEL_152;
          }
          else
          {
            v108 = &v179;
          }
          v187 = &v175;
          v111 = sub_1C9E550(&v178, v108, &v187);
          v54 = v161;
          v108 = (int *)v111;
LABEL_152:
          v55 = v108[10];
          goto LABEL_59;
        case '7':
          v94 = *(_QWORD *)(v49 - 24);
          v95 = *v34;
          LODWORD(v171) = 0;
          v175 = v94;
          v162 = (_BYTE *)v94;
          v96 = sub_1C98370(a1, v95, v94, (unsigned int *)&v171);
          v97 = v162;
          if ( v96 )
          {
            v98 = v171;
LABEL_131:
            v99 = sub_1CA1B70(a1, a2, v97, v49, &v184, v98, 0);
            if ( (*(_BYTE *)(v49 + 23) & 0x40) != 0 )
              v100 = *(_QWORD **)(v49 - 8);
            else
              v100 = (_QWORD *)(v49 - 24LL * (*(_DWORD *)(v49 + 20) & 0xFFFFFFF));
            if ( v100[3] )
            {
              v101 = v100[4];
              v102 = v100[5] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v102 = v101;
              if ( v101 )
                *(_QWORD *)(v101 + 16) = *(_QWORD *)(v101 + 16) & 3LL | v102;
            }
            v100[3] = v99;
            if ( v99 )
            {
              v103 = *((_QWORD *)v99 + 1);
              v100[4] = v103;
              if ( v103 )
                *(_QWORD *)(v103 + 16) = (unsigned __int64)(v100 + 4) | *(_QWORD *)(v103 + 16) & 3LL;
              v104 = v100[5];
              v105 = v100 + 3;
              v105[2] = (unsigned __int64)(v99 + 8) | v104 & 3;
              *((_QWORD *)v99 + 1) = v105;
            }
            goto LABEL_54;
          }
          v112 = v180;
          if ( v180 )
          {
            v113 = &v179;
            do
            {
              while ( 1 )
              {
                v114 = *((_QWORD *)v112 + 2);
                v115 = *((_QWORD *)v112 + 3);
                if ( *((_QWORD *)v112 + 4) >= v175 )
                  break;
                v112 = (int *)*((_QWORD *)v112 + 3);
                if ( !v115 )
                  goto LABEL_158;
              }
              v113 = v112;
              v112 = (int *)*((_QWORD *)v112 + 2);
            }
            while ( v114 );
LABEL_158:
            if ( v113 == &v179 || *((_QWORD *)v113 + 4) > v175 )
            {
LABEL_160:
              v187 = &v175;
              v116 = sub_1C9E550(&v178, v113, &v187);
              v97 = v162;
              v113 = (int *)v116;
            }
            v98 = v113[10];
            goto LABEL_131;
          }
          v113 = &v179;
          goto LABEL_160;
      }
      if ( v50 != 58 )
        break;
      v117 = *(_QWORD *)(v49 - 72);
      v118 = *v34;
      LODWORD(v171) = 0;
      v175 = v117;
      v163 = (_BYTE *)v117;
      v119 = sub_1C98370(a1, v118, v117, (unsigned int *)&v171);
      v54 = v163;
      if ( v119 )
      {
        v55 = v171;
        goto LABEL_165;
      }
      v128 = v180;
      if ( v180 )
      {
        v129 = &v179;
        do
        {
          while ( 1 )
          {
            v130 = *((_QWORD *)v128 + 2);
            v131 = *((_QWORD *)v128 + 3);
            if ( *((_QWORD *)v128 + 4) >= v175 )
              break;
            v128 = (int *)*((_QWORD *)v128 + 3);
            if ( !v131 )
              goto LABEL_192;
          }
          v129 = v128;
          v128 = (int *)*((_QWORD *)v128 + 2);
        }
        while ( v130 );
LABEL_192:
        if ( v129 != &v179 && *((_QWORD *)v129 + 4) <= v175 )
          goto LABEL_195;
      }
      else
      {
        v129 = &v179;
      }
      v187 = &v175;
      v132 = sub_1C9E550(&v178, v129, &v187);
      v54 = v163;
      v129 = (int *)v132;
LABEL_195:
      v55 = v129[10];
LABEL_165:
      if ( (unsigned int)(v55 - 4) <= 1 )
      {
        if ( v55 != 5 )
        {
          sub_1C95A60((__int64 *)&v187, ": Warning: Cannot do atomic on constant memory");
          sub_1C979E0(v49, (__int64)&v187);
          if ( v187 != (unsigned __int64 *)&v189 )
            j_j___libc_free_0(v187, v189 + 1);
          goto LABEL_54;
        }
        goto LABEL_212;
      }
LABEL_59:
      v56 = sub_1CA1B70(a1, a2, v54, v49, &v184, v55, 0);
      if ( (*(_BYTE *)(v49 + 23) & 0x40) != 0 )
        v57 = *(_QWORD **)(v49 - 8);
      else
        v57 = (_QWORD *)(v49 - 24LL * (*(_DWORD *)(v49 + 20) & 0xFFFFFFF));
      if ( *v57 )
      {
        v58 = v57[1];
        v59 = v57[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v59 = v58;
        if ( v58 )
          *(_QWORD *)(v58 + 16) = *(_QWORD *)(v58 + 16) & 3LL | v59;
      }
      *v57 = v56;
      if ( v56 )
      {
        v60 = *((_QWORD *)v56 + 1);
        v61 = (unsigned __int64)(v56 + 8);
        v57[1] = v60;
        if ( v60 )
          *(_QWORD *)(v60 + 16) = (unsigned __int64)(v57 + 1) | *(_QWORD *)(v60 + 16) & 3LL;
        goto LABEL_67;
      }
LABEL_54:
      if ( v167 == ++v34 )
        goto LABEL_68;
    }
    if ( v50 != 59 )
    {
      v133 = *(_QWORD *)(v49 - 24);
      if ( *(_BYTE *)(v133 + 16) )
LABEL_272:
        BUG();
      v134 = *(_DWORD *)(v133 + 36);
      v170 = 0;
      v165 = v134;
      sub_1C98880((__int64)a1, v134, &v170);
      v135 = *(_QWORD *)(v49 + 24 * (v170 - (unsigned __int64)(*(_DWORD *)(v49 + 20) & 0xFFFFFFF)));
      LODWORD(v171) = 0;
      v155 = (_BYTE *)v135;
      v175 = v135;
      if ( (unsigned __int8)sub_1C98370(a1, v49, v135, (unsigned int *)&v171) )
      {
        v136 = v171;
LABEL_201:
        if ( sub_1C302D0(v165) && (unsigned int)(v136 - 3) <= 2 )
        {
          v127 = ": Warning: Cannot do vector atomic on local memory";
          if ( v136 != 5 )
          {
            v127 = ": Warning: Cannot do vector atomic on constant memory";
            if ( v136 != 4 )
              v127 = ": Warning: Cannot to vector atomic on shared memory";
          }
          goto LABEL_186;
        }
        if ( !(unsigned __int8)sub_1C30260(v165) )
        {
          if ( !sub_1C98810((__int64)a1, v165) || (unsigned int)(v136 - 4) > 1 )
            goto LABEL_206;
          v127 = ": Warning: cannot perform wmma load or store on local memory";
          if ( v136 != 5 )
            v127 = ": Warning: cannot perform wmma load or store on constant memory";
          goto LABEL_186;
        }
        if ( (unsigned int)(v136 - 4) <= 1 )
        {
          if ( v136 != 5 )
            goto LABEL_185;
          goto LABEL_212;
        }
        sub_1C98810((__int64)a1, v165);
LABEL_206:
        v137 = sub_1CA1B70(a1, a2, v155, v49, &v184, v136, 0);
        sub_1593B40(
          (_QWORD *)(v49 + 24 * (v170 - (unsigned __int64)(*(_DWORD *)(v49 + 20) & 0xFFFFFFF))),
          (__int64)v137);
        v187 = (unsigned __int64 *)&v189;
        v188 = 0x300000000LL;
        if ( (unsigned __int8)sub_1C30260(v165) )
        {
          v138 = 3 * (v170 - (unsigned __int64)(*(_DWORD *)(v49 + 20) & 0xFFFFFFF));
LABEL_208:
          v175 = **(_QWORD **)(v49 + 8 * v138);
          sub_12AA070((__int64)&v187, &v175);
          goto LABEL_209;
        }
        v150 = *(_DWORD *)(v49 + 20) & 0xFFFFFFF;
        if ( v165 == 137 )
        {
          v175 = **(_QWORD **)(v49 - 24 * v150);
        }
        else
        {
          if ( (v165 & 0xFFFFFFFD) != 0x85 )
          {
            v138 = 3 * (v170 - v150);
            goto LABEL_208;
          }
          v175 = **(_QWORD **)(v49 - 24 * v150);
          sub_12AA070((__int64)&v187, &v175);
          v175 = **(_QWORD **)(v49 + 24 * (1LL - (*(_DWORD *)(v49 + 20) & 0xFFFFFFF)));
        }
        sub_12AA070((__int64)&v187, &v175);
        v175 = **(_QWORD **)(v49 + 24 * (2LL - (*(_DWORD *)(v49 + 20) & 0xFFFFFFF)));
        sub_12AA070((__int64)&v187, &v175);
LABEL_209:
        v139 = sub_15E26F0(*(__int64 **)(a2 + 40), v165, (__int64 *)v187, (unsigned int)v188);
        *(_QWORD *)(v49 + 64) = *(_QWORD *)(*(_QWORD *)v139 + 24LL);
        sub_1593B40((_QWORD *)(v49 - 24), v139);
        if ( v187 != (unsigned __int64 *)&v189 )
          _libc_free((unsigned __int64)v187);
        goto LABEL_54;
      }
      v147 = v180;
      if ( v180 )
      {
        v148 = &v179;
        do
        {
          if ( *((_QWORD *)v147 + 4) < v175 )
          {
            v147 = (int *)*((_QWORD *)v147 + 3);
          }
          else
          {
            v148 = v147;
            v147 = (int *)*((_QWORD *)v147 + 2);
          }
        }
        while ( v147 );
        if ( v148 != &v179 && *((_QWORD *)v148 + 4) <= v175 )
          goto LABEL_251;
      }
      else
      {
        v148 = &v179;
      }
      v187 = &v175;
      v148 = (int *)sub_1C9E550(&v178, v148, &v187);
LABEL_251:
      v136 = v148[10];
      goto LABEL_201;
    }
    v122 = *(_QWORD *)(v49 - 48);
    v123 = *v34;
    LODWORD(v171) = 0;
    v175 = v122;
    v164 = (_BYTE *)v122;
    v124 = sub_1C98370(a1, v123, v122, (unsigned int *)&v171);
    v125 = v164;
    if ( v124 )
    {
      v126 = v171;
      goto LABEL_183;
    }
    v141 = v180;
    if ( v180 )
    {
      v142 = &v179;
      do
      {
        if ( *((_QWORD *)v141 + 4) < v175 )
        {
          v141 = (int *)*((_QWORD *)v141 + 3);
        }
        else
        {
          v142 = v141;
          v141 = (int *)*((_QWORD *)v141 + 2);
        }
      }
      while ( v141 );
      if ( v142 != &v179 && *((_QWORD *)v142 + 4) <= v175 )
        goto LABEL_232;
    }
    else
    {
      v142 = &v179;
    }
    v187 = &v175;
    v143 = sub_1C9E550(&v178, v142, &v187);
    v125 = v164;
    v142 = (int *)v143;
LABEL_232:
    v126 = v142[10];
LABEL_183:
    if ( (unsigned int)(v126 - 4) <= 1 )
    {
      if ( v126 != 5 )
      {
LABEL_185:
        v127 = ": Warning: Cannot do atomic on constant memory";
LABEL_186:
        sub_1C95A60((__int64 *)&v187, v127);
        sub_1C979E0(v49, (__int64)&v187);
        sub_2240A30(&v187);
        goto LABEL_54;
      }
LABEL_212:
      v127 = ": Warning: Cannot do atomic on local memory";
      goto LABEL_186;
    }
    v56 = sub_1CA1B70(a1, a2, v125, v49, &v184, v126, 0);
    if ( (*(_BYTE *)(v49 + 23) & 0x40) != 0 )
      v57 = *(_QWORD **)(v49 - 8);
    else
      v57 = (_QWORD *)(v49 - 24LL * (*(_DWORD *)(v49 + 20) & 0xFFFFFFF));
    if ( *v57 )
    {
      v144 = v57[1];
      v145 = v57[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v145 = v144;
      if ( v144 )
        *(_QWORD *)(v144 + 16) = v145 | *(_QWORD *)(v144 + 16) & 3LL;
    }
    *v57 = v56;
    if ( !v56 )
      goto LABEL_54;
    v146 = *((_QWORD *)v56 + 1);
    v61 = (unsigned __int64)(v56 + 8);
    v57[1] = v146;
    if ( v146 )
      *(_QWORD *)(v146 + 16) = (unsigned __int64)(v57 + 1) | *(_QWORD *)(v146 + 16) & 3LL;
LABEL_67:
    ++v34;
    v57[2] = v61 | v57[2] & 3LL;
    *((_QWORD *)v56 + 1) = v57;
  }
  while ( v167 != v34 );
LABEL_68:
  v153 = 1;
LABEL_69:
  v62 = (__int64 *)a1[6];
  v157 = (__int64 *)a1[7];
  while ( v157 != v62 )
  {
    v63 = *v62;
    v64 = *(_QWORD *)(*v62 + 24 * (1LL - (*(_DWORD *)(*v62 + 20) & 0xFFFFFFF)));
    v159 = (_BYTE *)v64;
    v65 = *(_DWORD *)(*(_QWORD *)v64 + 8LL);
    LODWORD(v188) = 0;
    v189 = 0;
    v192 = 0;
    LODWORD(v171) = v65 >> 8;
    v190 = &v188;
    v191 = &v188;
    if ( (unsigned int)sub_1C9F820((__int64)a1, a2, v64, &v178, &v187, (int *)&v171, a3, a4, a5, a6, v12, v13, a9, a10) == 1 )
    {
      v66 = sub_1CA1B70(a1, a2, v159, v63, &v184, v171, 0);
      v67 = (_QWORD *)(v63 + 24 * (1LL - (*(_DWORD *)(v63 + 20) & 0xFFFFFFF)));
      if ( *v67 )
      {
        v68 = v67[1];
        v69 = v67[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v69 = v68;
        if ( v68 )
          *(_QWORD *)(v68 + 16) = *(_QWORD *)(v68 + 16) & 3LL | v69;
      }
      *v67 = v66;
      if ( v66 )
      {
        v70 = *((_QWORD *)v66 + 1);
        v67[1] = v70;
        if ( v70 )
          *(_QWORD *)(v70 + 16) = (unsigned __int64)(v67 + 1) | *(_QWORD *)(v70 + 16) & 3LL;
        v67[2] = (unsigned __int64)(v66 + 8) | v67[2] & 3LL;
        *((_QWORD *)v66 + 1) = v67;
      }
      v175 = (unsigned __int64)v177;
      v176 = 0x300000000LL;
      v71 = *(_QWORD *)(v63 - 24);
      if ( *(_BYTE *)(v71 + 16) )
        goto LABEL_272;
      v72 = *(_DWORD *)(v71 + 36);
      v73 = *(__int64 **)(a2 + 40);
      v74 = **(_QWORD **)(v63 - 24LL * (*(_DWORD *)(v63 + 20) & 0xFFFFFFF));
      LODWORD(v176) = 1;
      v177[0] = v74;
      v75 = **(_QWORD **)(v63 + 24 * (1LL - (*(_DWORD *)(v63 + 20) & 0xFFFFFFF)));
      LODWORD(v176) = 2;
      v177[1] = v75;
      v76 = **(_QWORD **)(v63 + 24 * (2LL - (*(_DWORD *)(v63 + 20) & 0xFFFFFFF)));
      LODWORD(v176) = 3;
      v177[2] = v76;
      v77 = sub_15E26F0(v73, v72, v177, 3);
      v78 = *(_QWORD *)(v63 - 24) == 0;
      v79 = v77;
      *(_QWORD *)(v63 + 64) = *(_QWORD *)(*(_QWORD *)v77 + 24LL);
      if ( !v78 )
      {
        v80 = *(_QWORD *)(v63 - 16);
        v81 = *(_QWORD *)(v63 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v81 = v80;
        if ( v80 )
          *(_QWORD *)(v80 + 16) = *(_QWORD *)(v80 + 16) & 3LL | v81;
      }
      *(_QWORD *)(v63 - 24) = v79;
      v82 = *(_QWORD *)(v79 + 8);
      *(_QWORD *)(v63 - 16) = v82;
      if ( v82 )
        *(_QWORD *)(v82 + 16) = (v63 - 16) | *(_QWORD *)(v82 + 16) & 3LL;
      *(_QWORD *)(v63 - 8) = (v79 + 8) | *(_QWORD *)(v63 - 8) & 3LL;
      *(_QWORD *)(v79 + 8) = v63 - 24;
      if ( (__int64 *)v175 != v177 )
        _libc_free(v175);
      v153 = 1;
    }
    ++v62;
    sub_1C97470(v189);
  }
  sub_1C96910(v186[0]);
  j___libc_free_0(0);
  if ( v172 )
    j_j___libc_free_0(v172, (char *)v174 - (char *)v172);
  sub_1C96570((__int64)v180);
  return v153;
}
