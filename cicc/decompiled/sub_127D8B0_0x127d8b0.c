// Function: sub_127D8B0
// Address: 0x127d8b0
//
__int64 __fastcall sub_127D8B0(__int64 a1, const __m128i *a2, __int64 a3)
{
  __int64 v3; // rdi
  const __m128i *v4; // r15
  _BYTE *v6; // r12
  unsigned __int64 v8; // rbx
  size_t v9; // r12
  size_t v10; // rbx
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  char *v13; // r14
  size_t v14; // r12
  size_t v15; // r13
  __int64 v16; // rdi
  __int64 j; // rax
  unsigned __int8 v18; // bl
  __int64 v19; // rdx
  __int64 v20; // r13
  __int8 v21; // al
  __int64 v22; // r13
  __int64 v23; // r13
  __int64 v24; // rax
  char v25; // dl
  char v26; // cl
  unsigned __int64 v27; // r14
  char v28; // al
  __int64 v29; // rax
  char v30; // dl
  int v31; // eax
  unsigned __int64 v32; // rbx
  int v33; // r13d
  bool i; // r14
  __int64 v35; // rax
  __int64 v36; // rax
  char v37; // r14
  __int64 v38; // r15
  _BYTE *v39; // rdi
  size_t v40; // rbx
  __int64 v41; // rax
  unsigned __int64 v42; // r13
  __int64 v43; // rax
  const char *v44; // rdx
  __int64 v45; // r13
  _BYTE *v46; // r13
  _BYTE *v47; // rbx
  __int64 v48; // rax
  char *v49; // rax
  size_t v50; // rcx
  char *v51; // r14
  char v52; // bl
  int v53; // r13d
  unsigned __int64 v54; // r14
  __int64 v55; // rdi
  _BYTE *v56; // rbx
  __int64 v57; // rsi
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdi
  __int64 v63; // r13
  unsigned int v64; // r14d
  __int64 v65; // rax
  __int64 v66; // rax
  int v67; // r12d
  int v68; // eax
  __int64 v69; // rax
  __int64 v70; // rsi
  __int64 v71; // r12
  __int64 v72; // rax
  __m128i *v73; // rsi
  __int64 v74; // rax
  __m128i *v75; // rsi
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r12
  unsigned __int64 v79; // r13
  char v80; // r14
  __int64 v81; // rbx
  __int64 v82; // r15
  _QWORD *v83; // rax
  __int64 v84; // r9
  unsigned __int64 v85; // r13
  _BOOL4 v86; // eax
  __int64 v87; // rax
  __int64 v88; // rsi
  __int64 v89; // rbx
  __int64 v90; // rax
  __int64 v91; // r13
  unsigned int v92; // ebx
  __int64 v93; // r12
  __int64 v94; // r12
  unsigned __int64 v95; // rbx
  __int64 v96; // rdi
  __int64 *v97; // rax
  _BYTE *v98; // rsi
  unsigned int v99; // eax
  __int64 v100; // rax
  __int64 v101; // rax
  size_t v102; // rbx
  __int64 v103; // rax
  unsigned __int64 v104; // r13
  __int64 v105; // r14
  __int64 v106; // rax
  size_t v107; // rbx
  _BYTE *v108; // r13
  __int64 v109; // rax
  unsigned int v110; // r15d
  __int64 v111; // rcx
  unsigned __int8 v112; // al
  __int64 v113; // r12
  unsigned __int64 v114; // rbx
  unsigned __int64 v115; // r14
  char *v116; // r8
  unsigned __int64 v117; // rcx
  _BYTE *v118; // rdi
  _BYTE *v119; // rsi
  __int64 v120; // rdx
  unsigned int v121; // ebx
  __int64 v122; // rax
  __int64 k; // rax
  __int64 v124; // rax
  __int64 v125; // r12
  __int64 v126; // rdi
  __int64 v127; // rax
  _BYTE *v128; // rsi
  unsigned __int8 v129; // al
  __int64 v130; // rax
  __int64 v131; // rbx
  __int64 v132; // rsi
  __int64 m; // rax
  unsigned __int64 v134; // r14
  __int64 n; // rax
  unsigned int ii; // r12d
  __int64 v137; // rsi
  __int64 v138; // r8
  __int64 v139; // rdx
  __int64 v140; // rax
  unsigned __int64 jj; // rbx
  __int64 v142; // rsi
  __int64 v143; // rax
  __int64 v144; // r13
  __int64 v145; // rbx
  __int64 v146; // rax
  unsigned int v147; // ecx
  int v148; // ebx
  unsigned int v149; // r13d
  int v150; // r12d
  unsigned int v151; // r14d
  signed int v152; // eax
  unsigned int v153; // ecx
  unsigned int v154; // edx
  unsigned __int64 v155; // rax
  _QWORD *v156; // r14
  unsigned int v157; // eax
  __int64 v158; // rax
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 v161; // rax
  __int64 v162; // rax
  unsigned __int64 v163; // [rsp+10h] [rbp-F0h]
  __int64 v164; // [rsp+10h] [rbp-F0h]
  __int64 v165; // [rsp+18h] [rbp-E8h]
  __int64 v166; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v167; // [rsp+18h] [rbp-E8h]
  int v168; // [rsp+20h] [rbp-E0h]
  __int64 v169; // [rsp+20h] [rbp-E0h]
  __int64 v170; // [rsp+28h] [rbp-D8h]
  __int64 v171; // [rsp+28h] [rbp-D8h]
  __int64 v172; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v173; // [rsp+30h] [rbp-D0h]
  unsigned int v174; // [rsp+30h] [rbp-D0h]
  const __m128i *v175; // [rsp+30h] [rbp-D0h]
  __int64 v176; // [rsp+38h] [rbp-C8h]
  unsigned int v177; // [rsp+38h] [rbp-C8h]
  __int64 v178; // [rsp+38h] [rbp-C8h]
  __int64 v179; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v180; // [rsp+48h] [rbp-B8h]
  __int64 v181; // [rsp+48h] [rbp-B8h]
  char v182; // [rsp+48h] [rbp-B8h]
  unsigned int v183; // [rsp+48h] [rbp-B8h]
  float v184; // [rsp+50h] [rbp-B0h]
  double v185; // [rsp+50h] [rbp-B0h]
  __int64 v186; // [rsp+50h] [rbp-B0h]
  bool v187; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v189; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v190; // [rsp+68h] [rbp-98h]
  unsigned __int64 v191; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v192; // [rsp+78h] [rbp-88h]
  __int64 v193; // [rsp+80h] [rbp-80h] BYREF
  __int64 v194; // [rsp+88h] [rbp-78h]
  unsigned __int64 v195; // [rsp+90h] [rbp-70h] BYREF
  char *v196; // [rsp+98h] [rbp-68h]
  __int64 v197; // [rsp+A0h] [rbp-60h]
  __int64 v198; // [rsp+B0h] [rbp-50h] BYREF
  _BYTE *v199; // [rsp+B8h] [rbp-48h] BYREF
  _BYTE *v200; // [rsp+C0h] [rbp-40h]

  if ( !a2 )
    return 0;
  v3 = a2[9].m128i_i64[0];
  v4 = a2;
  if ( v3 && sub_6E9180(v3) )
    sub_127B550("constant expressions are not supported!", (const __m128i *)a2[4].m128i_i32, 1);
  switch ( a2[10].m128i_i8[13] )
  {
    case 1:
      v31 = sub_620E90((__int64)a2);
      v32 = a2[8].m128i_u64[0];
      v33 = v31;
      for ( i = v31 == 0; *(_BYTE *)(v32 + 140) == 12; v32 = *(_QWORD *)(v32 + 160) )
        ;
      if ( *(_QWORD *)(v32 + 128) != 16 )
      {
        LODWORD(v196) = 64;
        v195 = 0;
        BYTE4(v196) = v31 == 0;
LABEL_51:
        if ( v33 )
          v35 = sub_620FA0((__int64)a2, &v198);
        else
          v35 = sub_620FD0((__int64)a2, &v198);
        if ( (unsigned int)v196 > 0x40 )
        {
          *(_QWORD *)v195 = v35;
          memset((void *)(v195 + 8), 0, 8 * (unsigned int)(((unsigned __int64)(unsigned int)v196 + 63) >> 6) - 8);
        }
        else
        {
          v195 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v196) & v35;
        }
        goto LABEL_55;
      }
      LODWORD(v196) = 128;
      sub_16A4EF0(&v195, 0, 0);
      v48 = *(_QWORD *)(v32 + 128);
      BYTE4(v196) = i;
      if ( v48 != 16 )
        goto LABEL_51;
      v49 = sub_622850(a2);
      v50 = 0;
      v51 = v49;
      if ( v49 )
        v50 = strlen(v49);
      sub_16A9890(&v198, 128, v51, v50, 10);
      if ( (unsigned int)v196 > 0x40 && v195 )
        j_j___libc_free_0_0(v195);
      v195 = v198;
      LODWORD(v196) = (_DWORD)v199;
LABEL_55:
      v36 = sub_127A030(*(_QWORD *)a1 + 8LL, v32, 0);
      v37 = *(_BYTE *)(v36 + 8);
      v38 = v36;
      if ( v37 == 15 || 8 * (unsigned int)*(_QWORD *)(v32 + 128) > 0x3F )
      {
        v6 = (_BYTE *)sub_159C0E0(*(_QWORD *)(a1 + 24), &v195);
        if ( v37 == 15 )
        {
          v99 = sub_127B390();
          v100 = sub_1644900(*(_QWORD *)(a1 + 24), v99);
          v101 = sub_15A4750(v6, v100, 0);
          v6 = (_BYTE *)sub_15A3BA0(v101, v38, 0);
        }
      }
      else
      {
        v52 = BYTE4(v196);
        sub_16A5A50(&v198, &v195);
        v53 = (int)v199;
        v54 = v198;
        if ( (unsigned int)v196 > 0x40 && v195 )
          j_j___libc_free_0_0(v195);
        v195 = v54;
        LODWORD(v196) = v53;
        v55 = *(_QWORD *)(a1 + 24);
        BYTE4(v196) = v52;
        v6 = (_BYTE *)sub_159C0E0(v55, &v195);
      }
      if ( (unsigned int)v196 > 0x40 )
      {
        v39 = (_BYTE *)v195;
        if ( v195 )
          goto LABEL_61;
      }
      return (__int64)v6;
    case 2:
      v8 = qword_4F06B40[a2[10].m128i_i8[8] & 7];
      if ( a3 )
        v9 = v8 * sub_8D4490(a3);
      else
        v9 = a2[11].m128i_u64[0];
      if ( v8 != 4 )
      {
        if ( v8 <= 4 )
        {
          if ( v8 != 1 )
          {
            if ( v8 != 2 )
              return 0;
            v10 = 0;
            v11 = sub_2207820(v9);
            v12 = a2[11].m128i_u64[0];
            v13 = (char *)v11;
            if ( v12 )
            {
              do
              {
                *(_WORD *)&v13[v10] = sub_722AB0((unsigned __int8 *)(v10 + a2[11].m128i_i64[1]), 2);
                v10 += 2LL;
              }
              while ( v12 > v10 );
              if ( v9 <= v10 )
                goto LABEL_19;
              goto LABEL_18;
            }
            if ( v9 )
            {
              do
              {
LABEL_18:
                *(_WORD *)&v13[v10] = 0;
                v10 += 2LL;
              }
              while ( v9 > v10 );
LABEL_19:
              v14 = v9 >> 1;
              v15 = 2 * v14;
              v16 = sub_1644C60(*(_QWORD *)(a1 + 24), 16);
LABEL_20:
              sub_1645D80(v16, v14);
              v6 = (_BYTE *)sub_15991C0(v13, v15);
LABEL_21:
              j_j___libc_free_0_0(v13);
              return (__int64)v6;
            }
            v57 = 16;
            v58 = *(_QWORD *)(a1 + 24);
            goto LABEL_105;
          }
          v105 = 0;
          v106 = sub_2207820(v9);
          v107 = a2[11].m128i_u64[0];
          v108 = (_BYTE *)v106;
          if ( v107 )
          {
            do
            {
              v108[v105] = sub_722AB0((unsigned __int8 *)(v105 + a2[11].m128i_i64[1]), 1);
              ++v105;
            }
            while ( v107 != v105 );
            if ( v9 <= v107 )
            {
              v162 = sub_1644C60(*(_QWORD *)(a1 + 24), 8);
              sub_1645D80(v162, v9);
              v6 = (_BYTE *)sub_15991C0(v108, v9);
LABEL_184:
              v39 = v108;
LABEL_61:
              j_j___libc_free_0_0(v39);
              return (__int64)v6;
            }
          }
          else if ( !v9 )
          {
LABEL_183:
            v109 = sub_1644C60(*(_QWORD *)(a1 + 24), 8);
            sub_1645D80(v109, v9);
            v6 = (_BYTE *)sub_15991C0(v108, v9);
            if ( v108 )
              goto LABEL_184;
            return (__int64)v6;
          }
          memset(&v108[v107], 0, v9 - v107);
          goto LABEL_183;
        }
        if ( v8 == 8 )
        {
          v40 = 0;
          v41 = sub_2207820(v9);
          v42 = a2[11].m128i_u64[0];
          v13 = (char *)v41;
          if ( v42 )
          {
            do
            {
              *(_QWORD *)&v13[v40] = sub_722AB0((unsigned __int8 *)(v40 + a2[11].m128i_i64[1]), 8);
              v40 += 8LL;
            }
            while ( v42 > v40 );
            if ( v9 <= v40 )
              goto LABEL_70;
            goto LABEL_69;
          }
          if ( v9 )
          {
            do
            {
LABEL_69:
              *(_QWORD *)&v13[v40] = 0;
              v40 += 8LL;
            }
            while ( v9 > v40 );
LABEL_70:
            v14 = v9 >> 3;
            v15 = 8 * v14;
            v16 = sub_1644C60(*(_QWORD *)(a1 + 24), 64);
            goto LABEL_20;
          }
          v57 = 64;
          v58 = *(_QWORD *)(a1 + 24);
LABEL_105:
          v59 = sub_1644C60(v58, v57);
          sub_1645D80(v59, 0);
          v6 = (_BYTE *)sub_15991C0(v13, 0);
          if ( v13 )
            goto LABEL_21;
          return (__int64)v6;
        }
        return 0;
      }
      v102 = 0;
      v103 = sub_2207820(v9);
      v104 = a2[11].m128i_u64[0];
      v13 = (char *)v103;
      if ( v104 )
      {
        do
        {
          *(_DWORD *)&v13[v102] = sub_722AB0((unsigned __int8 *)(v102 + a2[11].m128i_i64[1]), 4);
          v102 += 4LL;
        }
        while ( v104 > v102 );
        if ( v9 <= v102 )
          goto LABEL_178;
      }
      else if ( !v9 )
      {
        v57 = 32;
        v58 = *(_QWORD *)(a1 + 24);
        goto LABEL_105;
      }
      do
      {
        *(_DWORD *)&v13[v102] = 0;
        v102 += 4LL;
      }
      while ( v9 > v102 );
LABEL_178:
      v14 = v9 >> 2;
      v15 = 4 * v14;
      v16 = sub_1644C60(*(_QWORD *)(a1 + 24), 32);
      goto LABEL_20;
    case 3:
      for ( j = a2[8].m128i_i64[0]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      v18 = *(_BYTE *)(j + 160);
      v193 = sub_709B30(v18, a2 + 11);
      v194 = v19;
      if ( v18 == 4 )
        goto LABEL_79;
      if ( v18 <= 4u )
      {
        if ( v18 == 2 )
        {
          LODWORD(v184) = sub_12F9960(&v193);
          v20 = sub_1698270();
          sub_169D3B0(&v195, v184);
          sub_169E320(&v199, &v195, v20);
          sub_1698460(&v195);
          v6 = (_BYTE *)sub_159CCF0(*(_QWORD *)(a1 + 24), &v198);
          if ( v199 != (_BYTE *)sub_16982C0() )
          {
LABEL_28:
            sub_1698460(&v199);
            return (__int64)v6;
          }
          v46 = v200;
          if ( !v200 )
            return (__int64)v6;
          v56 = &v200[32 * *((_QWORD *)v200 - 1)];
          if ( v200 != v56 )
          {
            do
            {
              v56 -= 32;
              sub_127D120((_QWORD *)v56 + 1);
            }
            while ( v46 != v56 );
          }
LABEL_83:
          j_j_j___libc_free_0_0(v46 - 8);
          return (__int64)v6;
        }
LABEL_64:
        sub_127B550("unsupported float variant!", (const __m128i *)a2[4].m128i_i32, 1);
      }
      if ( v18 > 8u )
      {
        if ( v18 != 13 )
          goto LABEL_64;
      }
      else if ( v18 == 5 )
      {
        goto LABEL_64;
      }
      if ( qword_4F04C50 )
      {
        v43 = *(_QWORD *)(qword_4F04C50 + 32LL);
        if ( v43 )
        {
          if ( (*(_BYTE *)(v43 + 198) & 0x10) != 0 )
          {
            v44 = "long double";
            if ( v18 != 6 )
            {
              v44 = "__float80";
              if ( v18 != 7 )
                v44 = "__float128";
            }
            sub_684B10(0xE51u, (const __m128i *)a2[4].m128i_i32, (__int64)v44);
          }
        }
      }
LABEL_79:
      v185 = COERCE_DOUBLE(sub_12F99A0(&v193));
      v45 = sub_1698280();
      sub_169D3F0(&v195, v185);
      sub_169E320(&v199, &v195, v45);
      sub_1698460(&v195);
      v6 = (_BYTE *)sub_159CCF0(*(_QWORD *)(a1 + 24), &v198);
      if ( v199 != (_BYTE *)sub_16982C0() )
        goto LABEL_28;
      v46 = v200;
      if ( !v200 )
        return (__int64)v6;
      v47 = &v200[32 * *((_QWORD *)v200 - 1)];
      if ( v200 != v47 )
      {
        do
        {
          v47 -= 32;
          sub_127D120((_QWORD *)v47 + 1);
        }
        while ( v46 != v47 );
      }
      goto LABEL_83;
    case 6:
      v21 = a2[11].m128i_i8[0];
      if ( v21 == 1 )
      {
        v70 = a2[11].m128i_i64[1];
        if ( (*(_BYTE *)(v70 + 89) & 1) != 0 )
        {
          v6 = (_BYTE *)sub_1280350(*(_QWORD *)(a1 + 8));
          if ( v6[16] > 0x10u )
            sub_127B550("failed to lookup function static variable", (const __m128i *)v4[4].m128i_i32, 1);
        }
        else
        {
          v6 = (_BYTE *)sub_1277140(*(__int64 **)a1, v70, 0);
        }
      }
      else if ( v21 == 2 )
      {
        v78 = a2[11].m128i_i64[1];
        if ( *(_BYTE *)(v78 + 173) != 2 )
          sub_127B550("taking address of non-string constant is not supported!", (_DWORD *)(v78 + 64), 1);
        v6 = (_BYTE *)sub_126A1B0(*(__int64 **)a1, a2[11].m128i_i64[1], 0);
      }
      else
      {
        if ( v21 )
          sub_127B550("unsupported constant variant!", (const __m128i *)a2[4].m128i_i32, 1);
        v71 = a2[11].m128i_i64[1];
        if ( (*(_BYTE *)(v71 + 197) & 0x60) != 0 && *(_QWORD *)(v71 + 128) )
          v71 = *(_QWORD *)(v71 + 128);
        if ( dword_4D046EC )
        {
          v72 = a2[9].m128i_i64[0];
          v73 = (__m128i *)&a2[4];
          if ( v72 )
            v73 = (__m128i *)(v72 + 36);
          sub_127C010(v71, v73);
        }
        v74 = v4[9].m128i_i64[0];
        v75 = (__m128i *)&v4[4];
        if ( v74 )
          v75 = (__m128i *)(v74 + 36);
        sub_127C5E0(v71, v75);
        v6 = (_BYTE *)sub_1276020(*(_QWORD *)a1, v71, 0, v76, v77);
      }
      v22 = v4[12].m128i_i64[0];
      if ( v22 )
      {
        sub_622920((unsigned __int8)byte_4F06A60[0], &v193, &v191);
        v60 = sub_1644900(*(_QWORD *)(a1 + 24), (unsigned int)(8 * v193));
        v61 = sub_15A0680(v60, v22, 0);
        v62 = *(_QWORD *)(a1 + 24);
        v195 = v61;
        v63 = *(_QWORD *)v6;
        v64 = *(_DWORD *)(*(_QWORD *)v6 + 8LL);
        v65 = sub_1643330(v62);
        v66 = sub_1646BA0(v65, v64 >> 8);
        v67 = sub_15A4510(v6, v66, 0);
        v68 = sub_1643330(*(_QWORD *)(a1 + 24));
        BYTE4(v198) = 0;
        v69 = sub_15A2E80(v68, v67, (unsigned int)&v195, 1, 0, (unsigned int)&v198, 0);
        v6 = (_BYTE *)sub_15A4510(v69, v63, 0);
      }
      v23 = sub_127A030(*(_QWORD *)a1 + 8LL, v4[8].m128i_u64[0], 0);
      v24 = *(_QWORD *)v6;
      if ( v23 == *(_QWORD *)v6 )
        return (__int64)v6;
      v25 = *(_BYTE *)(v23 + 8);
      v26 = *(_BYTE *)(v24 + 8);
      if ( v25 == 15 )
      {
        if ( v26 == 15 )
        {
          if ( *(_DWORD *)(v23 + 8) >> 8 == *(_DWORD *)(v24 + 8) >> 8 )
            return sub_15A4510(v6, v23, 0);
          else
            return sub_15A4A70(v6, v23);
        }
LABEL_98:
        sub_127B550("unsupported cast from address constant!", (const __m128i *)v4[4].m128i_i32, 1);
      }
      if ( v26 != 15 || v25 != 11 )
        goto LABEL_98;
      if ( *(_DWORD *)(v24 + 8) >> 8 )
      {
        v159 = sub_1643330(*(_QWORD *)(a1 + 24));
        v160 = sub_1646BA0(v159, 0);
        v6 = (_BYTE *)sub_15A4A70(v6, v160);
      }
      return sub_15A4180(v6, v23, 0);
    case 0xA:
      v27 = a2[8].m128i_u64[0];
      v28 = *(_BYTE *)(v27 + 140);
      switch ( v28 )
      {
        case 12:
          v29 = a2[8].m128i_i64[0];
          do
          {
            v29 = *(_QWORD *)(v29 + 160);
            v30 = *(_BYTE *)(v29 + 140);
          }
          while ( v30 == 12 );
          if ( v30 != 10 )
          {
            if ( v30 != 11 )
            {
              if ( v30 != 8 )
                goto LABEL_46;
              v195 = 0;
              v196 = 0;
              v197 = 0;
              do
                v27 = *(_QWORD *)(v27 + 160);
              while ( *(_BYTE *)(v27 + 140) == 12 );
              goto LABEL_132;
            }
            v198 = 0;
            v199 = 0;
            v200 = 0;
            do
              v27 = *(_QWORD *)(v27 + 160);
            while ( *(_BYTE *)(v27 + 140) == 12 );
LABEL_142:
            v88 = a2[11].m128i_i64[0];
            if ( v88 )
            {
              if ( *(_BYTE *)(v88 + 173) != 13 )
              {
                v89 = *(_QWORD *)(v27 + 160);
                if ( v89 )
                {
                  while ( 1 )
                  {
                    if ( (*(_BYTE *)(v89 + 146) & 8) == 0 )
                    {
                      if ( (*(_BYTE *)(v89 + 144) & 4) == 0 )
                        goto LABEL_218;
                      if ( *(_QWORD *)(v89 + 8) )
                        break;
                    }
                    v89 = *(_QWORD *)(v89 + 112);
                    if ( !v89 )
                      goto LABEL_149;
                  }
LABEL_251:
                  sub_127B550(
                    "initialization of bit-field in union not supported!",
                    (const __m128i *)v4[4].m128i_i32,
                    1);
                }
LABEL_149:
                sub_127B550("cannot find initialized union member!", (const __m128i *)v4[4].m128i_i32, 1);
              }
              v89 = *(_QWORD *)(v88 + 184);
              if ( !v89 )
                goto LABEL_149;
              if ( (*(_BYTE *)(v89 + 144) & 4) != 0 )
                goto LABEL_251;
              v88 = *(_QWORD *)(v88 + 120);
LABEL_218:
              v195 = sub_127D8B0(a1, v88, *(_QWORD *)(v89 + 120));
              sub_127D5D0((__int64)&v198, &v195);
              for ( k = *(_QWORD *)(v89 + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                ;
              v124 = *(_QWORD *)(k + 128);
            }
            else
            {
              v124 = 0;
            }
            v125 = *(_QWORD *)(v27 + 128) - v124;
            if ( v125 )
            {
              v126 = sub_1643330(*(_QWORD *)(a1 + 24));
              if ( v125 != 1 )
                v126 = sub_1645D80(v126, v125);
              v127 = sub_15A06D0(v126);
              v128 = v199;
              v195 = v127;
              if ( v199 == v200 )
              {
                sub_127D720((__int64)&v198, v199, &v195);
              }
              else
              {
                if ( v199 )
                {
                  *(_QWORD *)v199 = v127;
                  v128 = v199;
                }
                v199 = v128 + 8;
              }
            }
            v129 = sub_127B460(v27);
            v113 = sub_15943F0(v198, (__int64)&v199[-v198] >> 3, v129);
            v130 = sub_127A030(*(_QWORD *)a1 + 8LL, v27, 0);
            v131 = v130;
            if ( *(_BYTE *)(v130 + 8) == 13 && (unsigned __int8)sub_1643C60(v130, v113) )
              v113 = v131;
            goto LABEL_196;
          }
          v198 = 0;
          v199 = 0;
          v200 = 0;
          do
            v27 = *(_QWORD *)(v27 + 160);
          while ( *(_BYTE *)(v27 + 140) == 12 );
          break;
        case 10:
          v198 = 0;
          v199 = 0;
          v200 = 0;
          break;
        case 11:
          v198 = 0;
          v199 = 0;
          v200 = 0;
          goto LABEL_142;
        case 8:
          v195 = 0;
          v196 = 0;
          v197 = 0;
LABEL_132:
          v79 = sub_8D4050(v27);
          v170 = sub_127A030(*(_QWORD *)a1 + 8LL, v27, 0);
          v186 = sub_127A030(*(_QWORD *)a1 + 8LL, v79, 0);
          v173 = *(_QWORD *)(v27 + 176);
          if ( !v173 )
            goto LABEL_185;
          if ( a2[11].m128i_i64[0] )
          {
            v80 = 0;
            v180 = 0;
            v81 = a2[11].m128i_i64[0];
            v82 = v79;
            do
            {
              v84 = v81;
              v85 = 1;
              if ( *(_BYTE *)(v81 + 173) != 11
                || (v85 = *(_QWORD *)(v81 + 184),
                    v176 = *(_QWORD *)(v81 + 176),
                    v86 = sub_8D44E0(v176, v82),
                    v84 = v176,
                    v86) )
              {
                v83 = (_QWORD *)sub_127D8B0(a1, v84, v82);
                v191 = (unsigned __int64)v83;
              }
              else
              {
                v85 = 1;
                v87 = sub_127D8B0(a1, v176, *(_QWORD *)(v176 + 128));
                v83 = (_QWORD *)sub_127D000(v87, v82);
                v191 = (unsigned __int64)v83;
              }
              v80 |= *v83 != v186;
              sub_127D2E0((__int64)&v195, v196, v85, &v191);
              v81 = *(_QWORD *)(v81 + 120);
              v180 += v85;
            }
            while ( v81 );
            v114 = v180;
            if ( v173 <= v180 )
            {
LABEL_203:
              if ( v80 )
              {
                v116 = v196;
                v117 = v195;
                v198 = 0;
                v199 = 0;
                v200 = 0;
                if ( v196 == (char *)v195 )
                {
                  v139 = 0;
                  v138 = 0;
                }
                else
                {
                  v118 = 0;
                  v119 = 0;
                  v120 = 0;
                  v121 = 0;
                  while ( 1 )
                  {
                    v122 = **(_QWORD **)(v117 + 8 * v120);
                    v193 = v122;
                    if ( v119 == v118 )
                    {
                      sub_1278040((__int64)&v198, v119, &v193);
                      v119 = v199;
                      v116 = v196;
                      v117 = v195;
                    }
                    else
                    {
                      if ( v119 )
                      {
                        *(_QWORD *)v119 = v122;
                        v119 = v199;
                        v116 = v196;
                        v117 = v195;
                      }
                      v119 += 8;
                      v199 = v119;
                    }
                    v120 = ++v121;
                    if ( v121 >= (unsigned __int64)((__int64)&v116[-v117] >> 3) )
                      break;
                    v118 = v200;
                  }
                  v138 = v198;
                  v139 = (__int64)&v119[-v198] >> 3;
                }
                v140 = sub_1645600(*(_QWORD *)(a1 + 24), v138, v139, 1);
                v6 = (_BYTE *)sub_159F090(v140, v195, (__int64)&v196[-v195] >> 3);
                if ( v198 )
                  j_j___libc_free_0(v198, &v200[-v198]);
                goto LABEL_186;
              }
LABEL_185:
              v6 = (_BYTE *)sub_159DFD0(v170, v195, (__int64)&v196[-v195] >> 3);
LABEL_186:
              if ( v195 )
                j_j___libc_free_0(v195, v197 - v195);
              return (__int64)v6;
            }
          }
          else
          {
            v80 = 0;
            v114 = 0;
          }
          v182 = v80;
          v115 = v114;
          do
          {
            ++v115;
            v198 = sub_15A06D0(v186);
            sub_127D5D0((__int64)&v195, &v198);
          }
          while ( v173 > v115 );
          v80 = v182;
          goto LABEL_203;
        default:
LABEL_46:
          sub_127B550("unsupported aggregate constant!", (const __m128i *)a2[4].m128i_i32, 1);
      }
      v90 = sub_127A030(*(_QWORD *)a1 + 8LL, v27, 0);
      v179 = v90;
      v177 = *(_DWORD *)(v90 + 12);
      v187 = v177 != 0 && a2[11].m128i_i64[0] == 0;
      if ( v187 )
      {
        v6 = (_BYTE *)sub_1598F00(v90);
        goto LABEL_197;
      }
      v91 = *(_QWORD *)(v27 + 160);
      if ( !v91 )
      {
        v92 = 0;
        if ( v177 )
          goto LABEL_189;
        goto LABEL_193;
      }
      v163 = v27;
      v92 = 0;
      v93 = a2[11].m128i_i64[0];
      v171 = *(_QWORD *)a1 + 8LL;
      v181 = a2[11].m128i_i64[1];
      break;
    case 0xE:
      return 0;
    default:
      sub_127B550("unsupported constant variant!", (const __m128i *)a2[4].m128i_i32, 1);
  }
  while ( 1 )
  {
    if ( (*(_BYTE *)(v91 + 146) & 8) == 0 )
    {
      if ( (*(_BYTE *)(v91 + 144) & 4) != 0 )
      {
        v187 = 1;
      }
      else
      {
        v174 = sub_1277B60(v171, v91);
        if ( v92 >= v174 )
        {
          v174 = v92;
        }
        else
        {
          v165 = v93;
          v94 = 8LL * v92;
          v95 = 8 * (v92 + (unsigned __int64)(~v92 + v174) + 1);
          do
          {
            v96 = *(_QWORD *)(*(_QWORD *)(v179 + 16) + v94);
            v94 += 8;
            v195 = sub_15A06D0(v96);
            sub_127D5D0((__int64)&v198, &v195);
          }
          while ( v95 != v94 );
          v93 = v165;
        }
        v97 = (__int64 *)sub_127D8B0(a1, v93, *(_QWORD *)(v91 + 120));
        v193 = (__int64)v97;
        if ( (*(_BYTE *)(v91 + 145) & 0x10) != 0 )
        {
          v132 = *v97;
          for ( m = *(_QWORD *)(v91 + 120); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
            ;
          v134 = *(_QWORD *)(m + 128);
          for ( n = *(_QWORD *)(v93 + 128); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
            ;
          if ( v134 < *(_QWORD *)(n + 128) )
          {
            v161 = sub_15A9930(*(_QWORD *)(*(_QWORD *)a1 + 368LL), v132);
            v168 = sub_15A8020(v161, v134);
          }
          else
          {
            v168 = *(_DWORD *)(v132 + 12);
          }
          if ( v168 )
          {
            v166 = v93;
            for ( ii = 0; ii != v168; ++ii )
            {
              v137 = ii;
              v195 = sub_15A0A60(v193, v137);
              sub_127D5D0((__int64)&v198, &v195);
            }
            v93 = v166;
            v92 = v168 + v174;
          }
          else
          {
            v92 = v174;
          }
        }
        else
        {
          v98 = v199;
          if ( v199 == v200 )
          {
            sub_127D720((__int64)&v198, v199, &v193);
          }
          else
          {
            if ( v199 )
            {
              *(_QWORD *)v199 = v97;
              v98 = v199;
            }
            v199 = v98 + 8;
          }
          v92 = v174 + 1;
        }
        if ( (*(_BYTE *)(v91 + 144) & 4) == 0 )
        {
LABEL_157:
          if ( v93 == v181 )
            goto LABEL_188;
          v93 = *(_QWORD *)(v93 + 120);
          goto LABEL_159;
        }
      }
      if ( *(_QWORD *)(v91 + 8) )
        goto LABEL_157;
    }
LABEL_159:
    v91 = *(_QWORD *)(v91 + 112);
    if ( !v91 )
    {
LABEL_188:
      v27 = v163;
      if ( v177 > v92 )
      {
LABEL_189:
        v175 = v4;
        v110 = v92;
        do
        {
          v111 = v110++;
          v195 = sub_15A06D0(*(_QWORD *)(*(_QWORD *)(v179 + 16) + 8 * v111));
          sub_127D5D0((__int64)&v198, &v195);
        }
        while ( v177 > v110 );
        v4 = v175;
      }
      if ( v187 )
      {
        for ( jj = v4[8].m128i_u64[0]; *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
          ;
        v142 = sub_127A030(*(_QWORD *)a1 + 8LL, jj, 0);
        if ( *(_BYTE *)(v142 + 8) != 13 )
          v142 = 0;
        v143 = sub_15A9930(*(_QWORD *)(*(_QWORD *)a1 + 368LL), v142);
        v144 = *(_QWORD *)(jj + 160);
        v164 = v143;
        v169 = v4[11].m128i_i64[1];
        v178 = v4[11].m128i_i64[0];
        if ( v144 )
        {
          v167 = v27;
          while ( 1 )
          {
            if ( (*(_BYTE *)(v144 + 146) & 8) == 0 )
            {
              if ( (*(_BYTE *)(v144 + 144) & 4) != 0 )
              {
                if ( *(_QWORD *)(v144 + 8) )
                {
                  v183 = sub_15A8020(v164, *(_QWORD *)(v144 + 128));
                  v145 = **(_QWORD **)(v198 + 8LL * v183);
                  if ( v145 != sub_1643330(*(_QWORD *)(a1 + 24)) )
                    sub_127B550("unexpected error while initializing bitfield!", (const __m128i *)v4[4].m128i_i32, 1);
                  v146 = sub_127D8B0(a1, v178, 0);
                  if ( !v146 )
LABEL_359:
                    sub_127B550("bit-field constant must have a known value at compile time!", (_DWORD *)(v178 + 64), 1);
                  v147 = *(_DWORD *)(v146 + 32);
                  v190 = v147;
                  if ( v147 > 0x40 )
                  {
                    sub_16A4FD0(&v189, v146 + 24);
                    v147 = v190;
                  }
                  else
                  {
                    v189 = *(_QWORD *)(v146 + 24);
                  }
                  v148 = *(unsigned __int8 *)(v144 + 137);
                  if ( v147 > v148 )
                  {
                    sub_16A5A50(&v195, &v189);
                    if ( v190 > 0x40 && v189 )
                      j_j___libc_free_0_0(v189);
                    v148 = *(unsigned __int8 *)(v144 + 137);
                    v189 = v195;
                    v190 = (unsigned int)v196;
                  }
                  if ( v148 )
                  {
                    v172 = v144;
                    v149 = v183;
                    while ( 1 )
                    {
                      v158 = *(_QWORD *)(v198 + 8LL * v149);
                      if ( !v158 )
                        goto LABEL_359;
                      v192 = *(_DWORD *)(v158 + 32);
                      if ( v192 > 0x40 )
                        sub_16A4FD0(&v191, v158 + 24);
                      else
                        v191 = *(_QWORD *)(v158 + 24);
                      if ( v149 == v183 )
                      {
                        v151 = *(unsigned __int8 *)(v172 + 136);
                        v150 = 8 - v151;
                        if ( (int)(8 - v151) > v148 )
                          v150 = v148;
                      }
                      else
                      {
                        v150 = 8;
                        if ( v148 <= 8 )
                          v150 = v148;
                        v151 = 0;
                      }
                      v152 = v190;
                      LODWORD(v194) = v190;
                      if ( v190 > 0x40 )
                      {
                        sub_16A4FD0(&v193, &v189);
                        v152 = v190;
                      }
                      else
                      {
                        v193 = v189;
                      }
                      if ( v152 > v150 )
                      {
                        sub_16A5A50(&v195, &v193);
                        if ( (unsigned int)v194 > 0x40 && v193 )
                          j_j___libc_free_0_0(v193);
                        v153 = (unsigned int)v196;
                        v193 = v195;
                        LODWORD(v194) = (_DWORD)v196;
                      }
                      else
                      {
                        v153 = v194;
                      }
                      if ( v153 <= 7 )
                      {
                        sub_16A5C50(&v195, &v193, 8);
                        if ( (unsigned int)v194 > 0x40 && v193 )
                          j_j___libc_free_0_0(v193);
                        v153 = (unsigned int)v196;
                        v193 = v195;
                        LODWORD(v194) = (_DWORD)v196;
                      }
                      LODWORD(v196) = v153;
                      if ( v153 > 0x40 )
                      {
                        sub_16A4FD0(&v195, &v193);
                        v153 = (unsigned int)v196;
                        if ( (unsigned int)v196 > 0x40 )
                        {
                          sub_16A7DC0(&v195, v151);
                          v154 = v194;
                          goto LABEL_285;
                        }
                        v154 = v194;
                      }
                      else
                      {
                        v154 = v153;
                        v195 = v193;
                      }
                      v155 = 0;
                      if ( v151 != v153 )
                        v155 = (v195 << v151) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v153);
                      v195 = v155;
LABEL_285:
                      if ( v154 > 0x40 && v193 )
                        j_j___libc_free_0_0(v193);
                      v193 = v195;
                      LODWORD(v194) = (_DWORD)v196;
                      if ( v192 > 0x40 )
                        sub_16A89F0(&v191, &v193);
                      else
                        v191 |= v195;
                      v156 = (_QWORD *)(v198 + 8LL * v149);
                      *v156 = sub_159C0E0(*(_QWORD *)(a1 + 24), &v191);
                      v157 = v190;
                      LODWORD(v196) = v190;
                      if ( v190 <= 0x40 )
                      {
                        v195 = v189;
LABEL_292:
                        if ( v157 == v150 )
                          v195 = 0;
                        else
                          v195 >>= v150;
                        goto LABEL_294;
                      }
                      sub_16A4FD0(&v195, &v189);
                      v157 = (unsigned int)v196;
                      if ( (unsigned int)v196 <= 0x40 )
                        goto LABEL_292;
                      sub_16A8110(&v195, (unsigned int)v150);
LABEL_294:
                      if ( v190 > 0x40 && v189 )
                        j_j___libc_free_0_0(v189);
                      v148 -= v150;
                      v189 = v195;
                      v190 = (unsigned int)v196;
                      if ( (unsigned int)v194 > 0x40 && v193 )
                        j_j___libc_free_0_0(v193);
                      if ( v192 > 0x40 && v191 )
                        j_j___libc_free_0_0(v191);
                      ++v149;
                      if ( !v148 )
                      {
                        v144 = v172;
                        break;
                      }
                    }
                  }
                  if ( v169 == v178 )
                  {
                    v27 = v167;
                    if ( v190 > 0x40 && v189 )
                      j_j___libc_free_0_0(v189);
                    break;
                  }
                  v178 = *(_QWORD *)(v178 + 120);
                  if ( v190 > 0x40 && v189 )
                    j_j___libc_free_0_0(v189);
                }
              }
              else
              {
                if ( v169 == v178 )
                  goto LABEL_327;
                v178 = *(_QWORD *)(v178 + 120);
              }
            }
            v144 = *(_QWORD *)(v144 + 112);
            if ( !v144 )
            {
LABEL_327:
              v27 = v167;
              break;
            }
          }
        }
      }
LABEL_193:
      if ( v199 != (_BYTE *)v198 )
      {
        v112 = sub_127B460(v27);
        v113 = sub_15943F0(v198, (__int64)&v199[-v198] >> 3, v112);
        if ( (unsigned __int8)sub_1643C60(v179, v113) )
          v113 = v179;
LABEL_196:
        v6 = (_BYTE *)sub_159F090(v113, v198, (__int64)&v199[-v198] >> 3);
        goto LABEL_197;
      }
      v6 = (_BYTE *)sub_159F090(v179, v198, 0);
LABEL_197:
      if ( v198 )
        j_j___libc_free_0(v198, &v200[-v198]);
      return (__int64)v6;
    }
  }
}
