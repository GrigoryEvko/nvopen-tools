// Function: sub_116E930
// Address: 0x116e930
//
__int64 __fastcall sub_116E930(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rdx
  unsigned int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 *v13; // rsi
  __int64 *v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 *v17; // rbx
  __int64 *v18; // r12
  __int64 *v19; // rax
  _BYTE *v20; // rcx
  __int64 v21; // r13
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // r12
  char *v26; // rbx
  char v27; // al
  __int64 *v28; // rax
  __int64 v29; // rax
  _BYTE *v30; // r14
  char *v31; // rdx
  __int64 v32; // r15
  unsigned __int64 v33; // rbx
  unsigned __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // r12
  __int64 **v43; // r13
  __int64 v44; // r12
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r14
  __int64 v49; // r12
  __int64 v50; // r12
  __int64 *v51; // r13
  __int64 *i; // rbx
  unsigned __int8 *v53; // r14
  __int64 v54; // r8
  char v55; // al
  __int64 v56; // rdx
  __m128i *v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rbx
  __int64 *v60; // rcx
  __int64 v61; // rax
  __int64 v62; // rdx
  int v63; // esi
  __int64 v64; // r12
  __int64 v65; // r14
  __int64 *v66; // rdi
  int v67; // r8d
  unsigned int v68; // ecx
  __int64 *v69; // rax
  __int64 v70; // r9
  __int64 v71; // rcx
  int v72; // eax
  int v73; // eax
  unsigned int v74; // esi
  __int64 v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rsi
  __int64 v78; // rax
  _QWORD *v79; // rdx
  _QWORD *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  unsigned int v83; // ecx
  unsigned int v84; // edx
  int v85; // r14d
  unsigned __int64 v86; // rdx
  int v87; // ecx
  __int64 *v88; // rax
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r12
  __int64 v92; // r14
  int v93; // eax
  int v94; // eax
  unsigned int v95; // edx
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rdx
  unsigned __int64 v99; // rax
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // r10
  __int64 v103; // r11
  __int64 v104; // r12
  __int64 v105; // rcx
  int v106; // eax
  int v107; // eax
  unsigned int v108; // edx
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rdx
  __int64 *v112; // rax
  __int64 *v113; // rdx
  __int64 v114; // rsi
  char *v115; // rax
  char *v116; // rdx
  __int64 v117; // rsi
  __int64 v118; // rcx
  __int64 v119; // rbx
  __int64 v120; // r13
  __int64 v121; // rdx
  unsigned int v122; // esi
  __int64 *v123; // r12
  __int64 v124; // rax
  signed __int64 v125; // rsi
  __int64 v126; // r14
  __int64 v127; // rax
  unsigned __int64 v128; // rdx
  unsigned __int64 v129; // r14
  unsigned __int64 *v130; // rax
  __int64 *v131; // rsi
  __int64 v132; // rcx
  _DWORD *v133; // rdi
  _QWORD *v134; // [rsp+8h] [rbp-368h]
  __int64 **v135; // [rsp+8h] [rbp-368h]
  __int64 v136; // [rsp+10h] [rbp-360h]
  _QWORD *v137; // [rsp+10h] [rbp-360h]
  __int64 v138; // [rsp+10h] [rbp-360h]
  __int64 *v139; // [rsp+30h] [rbp-340h]
  __int64 v140; // [rsp+38h] [rbp-338h]
  __int64 *v141; // [rsp+40h] [rbp-330h]
  __int64 v142; // [rsp+48h] [rbp-328h]
  __int64 *v143; // [rsp+50h] [rbp-320h]
  __int64 v144; // [rsp+50h] [rbp-320h]
  __int64 v145; // [rsp+50h] [rbp-320h]
  __int64 *v146; // [rsp+58h] [rbp-318h]
  int v147; // [rsp+60h] [rbp-310h]
  __int64 *v148; // [rsp+60h] [rbp-310h]
  int v149; // [rsp+78h] [rbp-2F8h]
  unsigned __int32 v150; // [rsp+7Ch] [rbp-2F4h]
  __int64 v151; // [rsp+80h] [rbp-2F0h]
  int v152; // [rsp+88h] [rbp-2E8h]
  unsigned int v153; // [rsp+88h] [rbp-2E8h]
  __int64 v154; // [rsp+88h] [rbp-2E8h]
  __int64 v157; // [rsp+B0h] [rbp-2C0h] BYREF
  _QWORD *v158; // [rsp+B8h] [rbp-2B8h]
  __int64 v159; // [rsp+C0h] [rbp-2B0h]
  unsigned int v160; // [rsp+C8h] [rbp-2A8h]
  __int64 v161; // [rsp+D0h] [rbp-2A0h] BYREF
  __int64 v162; // [rsp+D8h] [rbp-298h]
  __int64 v163; // [rsp+E0h] [rbp-290h]
  unsigned int v164; // [rsp+E8h] [rbp-288h]
  __int64 v165[4]; // [rsp+F0h] [rbp-280h] BYREF
  __int16 v166; // [rsp+110h] [rbp-260h]
  __m128i v167[2]; // [rsp+120h] [rbp-250h] BYREF
  __int16 v168; // [rsp+140h] [rbp-230h]
  __int64 *v169; // [rsp+150h] [rbp-220h] BYREF
  __int64 v170; // [rsp+158h] [rbp-218h]
  __m128i *v171; // [rsp+160h] [rbp-210h]
  __int64 v172; // [rsp+168h] [rbp-208h]
  __int16 v173; // [rsp+170h] [rbp-200h]
  __int64 *v174; // [rsp+180h] [rbp-1F0h] BYREF
  __int64 v175; // [rsp+188h] [rbp-1E8h]
  _QWORD v176[8]; // [rsp+190h] [rbp-1E0h] BYREF
  __int64 v177; // [rsp+1D0h] [rbp-1A0h] BYREF
  __int64 *v178; // [rsp+1D8h] [rbp-198h]
  __int64 v179; // [rsp+1E0h] [rbp-190h]
  int v180; // [rsp+1E8h] [rbp-188h]
  char v181; // [rsp+1ECh] [rbp-184h]
  __int64 v182; // [rsp+1F0h] [rbp-180h] BYREF
  void *base; // [rsp+230h] [rbp-140h] BYREF
  __int64 v184; // [rsp+238h] [rbp-138h]
  _BYTE v185[304]; // [rsp+240h] [rbp-130h] BYREF

  v6 = v176;
  v7 = 0;
  base = v185;
  v184 = 0x1000000000LL;
  v178 = &v182;
  v174 = v176;
  v176[0] = a2;
  v180 = 0;
  v181 = 1;
  v182 = a2;
  v177 = 1;
  v175 = 0x800000001LL;
  v179 = 0x100000008LL;
  v8 = 0;
  while ( 1 )
  {
    v9 = v6[v8];
    v10 = *(_QWORD *)(v9 - 8);
    v11 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
    v12 = 32 * v11;
    if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
    {
      v13 = (__int64 *)(v10 + v12);
      v14 = *(__int64 **)(v9 - 8);
    }
    else
    {
      v13 = (__int64 *)v9;
      v14 = (__int64 *)(v9 - v12);
    }
    v15 = 32LL * *(unsigned int *)(v9 + 72);
    v16 = v15 + 8 * v11;
    v17 = (__int64 *)(v10 + v15);
    v18 = (__int64 *)(v10 + v16);
    if ( v17 != v18 )
    {
      v19 = v17;
      if ( v14 != v13 )
      {
        v20 = (_BYTE *)*v14;
        if ( *(_BYTE *)*v14 == 34 )
          goto LABEL_10;
        while ( 1 )
        {
          ++v19;
          v14 += 4;
          if ( v18 == v19 || v14 == v13 )
            break;
          v20 = (_BYTE *)*v14;
          if ( *(_BYTE *)*v14 == 34 )
          {
LABEL_10:
            if ( *((_QWORD *)v20 + 5) == *v19 )
              goto LABEL_11;
          }
        }
        while ( 1 )
        {
LABEL_20:
          while ( 1 )
          {
            v23 = *v17;
            v24 = sub_AA5190(*v17);
            if ( v24 )
              break;
            if ( v18 == ++v17 )
              goto LABEL_23;
          }
          if ( v24 == v23 + 48 )
            goto LABEL_11;
          if ( v18 == ++v17 )
            goto LABEL_23;
        }
      }
      if ( v17 != v18 )
        goto LABEL_20;
    }
LABEL_23:
    v25 = *(_QWORD *)(v9 + 16);
    if ( v25 )
    {
      while ( 1 )
      {
        v26 = *(char **)(v25 + 24);
        v27 = *v26;
        if ( *v26 != 84 )
          break;
        if ( v181 )
        {
          v28 = v178;
          v11 = HIDWORD(v179);
          v14 = &v178[HIDWORD(v179)];
          if ( v178 != v14 )
          {
            while ( v26 != (char *)*v28 )
            {
              if ( v14 == ++v28 )
                goto LABEL_49;
            }
            goto LABEL_30;
          }
LABEL_49:
          if ( HIDWORD(v179) < (unsigned int)v179 )
          {
            ++HIDWORD(v179);
            *v14 = (__int64)v26;
            ++v177;
            goto LABEL_51;
          }
        }
        v13 = *(__int64 **)(v25 + 24);
        sub_C8CC70((__int64)&v177, (__int64)v13, (__int64)v14, v11, a5, a6);
        if ( (_BYTE)v14 )
        {
LABEL_51:
          v38 = (unsigned int)v175;
          v11 = HIDWORD(v175);
          v39 = (unsigned int)v175 + 1LL;
          if ( v39 > HIDWORD(v175) )
          {
            v13 = v176;
            sub_C8D5F0((__int64)&v174, v176, v39, 8u, a5, a6);
            v38 = (unsigned int)v175;
          }
          v14 = v174;
          v174[v38] = (__int64)v26;
          LODWORD(v175) = v175 + 1;
          v25 = *(_QWORD *)(v25 + 8);
          if ( !v25 )
            goto LABEL_31;
        }
        else
        {
LABEL_30:
          v25 = *(_QWORD *)(v25 + 8);
          if ( !v25 )
            goto LABEL_31;
        }
      }
      if ( v27 == 67 )
      {
        v40 = (unsigned int)v184;
        v11 = HIDWORD(v184);
        v14 = (__int64 *)((unsigned int)v184 + 1LL);
        if ( (unsigned __int64)v14 > HIDWORD(v184) )
        {
          v13 = (__int64 *)v185;
          sub_C8D5F0((__int64)&base, v185, (unsigned __int64)v14, 0x10u, a5, a6);
          v40 = (unsigned int)v184;
        }
        v41 = (char *)base + 16 * v40;
        *v41 = v7;
        v41[1] = v26;
        LODWORD(v184) = v184 + 1;
      }
      else
      {
        if ( v27 != 55 )
          goto LABEL_11;
        v29 = *((_QWORD *)v26 + 2);
        if ( !v29 )
          goto LABEL_11;
        if ( *(_QWORD *)(v29 + 8) )
          goto LABEL_11;
        v30 = *(_BYTE **)(v29 + 24);
        if ( *v30 != 67 )
          goto LABEL_11;
        v31 = (v26[7] & 0x40) != 0 ? (char *)*((_QWORD *)v26 - 1) : &v26[-32 * (*((_DWORD *)v26 + 1) & 0x7FFFFFF)];
        v32 = *((_QWORD *)v31 + 4);
        if ( *(_BYTE *)v32 != 17 )
          goto LABEL_11;
        v33 = (unsigned int)sub_BCB060(*((_QWORD *)v26 + 1));
        if ( *(_DWORD *)(v32 + 32) > 0x40u )
        {
          v152 = *(_DWORD *)(v32 + 32);
          if ( v152 - (unsigned int)sub_C444A0(v32 + 24) > 0x40 )
            goto LABEL_11;
          v34 = **(_QWORD **)(v32 + 24);
          if ( v33 <= v34 )
            goto LABEL_11;
        }
        else
        {
          v34 = *(_QWORD *)(v32 + 24);
          if ( v33 <= v34 )
            goto LABEL_11;
        }
        v11 = HIDWORD(v184);
        v35 = (v34 << 32) | v7;
        v36 = (unsigned int)v184;
        v14 = (__int64 *)((unsigned int)v184 + 1LL);
        if ( (unsigned __int64)v14 > HIDWORD(v184) )
        {
          v13 = (__int64 *)v185;
          sub_C8D5F0((__int64)&base, v185, (unsigned __int64)v14, 0x10u, a5, a6);
          v36 = (unsigned int)v184;
        }
        v37 = (__int64 *)((char *)base + 16 * v36);
        *v37 = v35;
        v37[1] = (__int64)v30;
        LODWORD(v184) = v184 + 1;
      }
      goto LABEL_30;
    }
LABEL_31:
    v8 = v7 + 1;
    v7 = v8;
    if ( (_DWORD)v175 == (_DWORD)v8 )
      break;
    v6 = v174;
  }
  v149 = v184;
  if ( (_DWORD)v184 )
  {
    if ( (unsigned int)v184 == 1 )
    {
      v157 = 0;
      v158 = 0;
      v159 = 0;
      v160 = 0;
      v161 = 0;
      v162 = 0;
      v163 = 0;
      v164 = 0;
    }
    else
    {
      qsort(base, (16LL * (unsigned int)v184) >> 4, 0x10u, (__compar_fn_t)sub_116D090);
      v157 = 0;
      v158 = 0;
      v149 = v184;
      v159 = 0;
      v160 = 0;
      v161 = 0;
      v162 = 0;
      v163 = 0;
      v164 = 0;
      if ( !(_DWORD)v184 )
      {
LABEL_83:
        v50 = sub_ACADE0(*(__int64 ***)(a2 + 8));
        v51 = &v174[(unsigned int)v175];
        for ( i = v174 + 1; v51 != i; ++i )
        {
          v53 = (unsigned __int8 *)*i;
          if ( *(_QWORD *)(*i + 16) )
          {
            sub_10A5FE0(*(_QWORD *)(a1 + 40), *i);
            v54 = v50;
            if ( v53 == (unsigned __int8 *)v50 )
              v54 = sub_ACADE0(*(__int64 ***)(v50 + 8));
            if ( !*(_QWORD *)(v54 + 16)
              && *(_BYTE *)v54 > 0x1Cu
              && (*(_BYTE *)(v54 + 7) & 0x10) == 0
              && (v53[7] & 0x10) != 0 )
            {
              v154 = v54;
              sub_BD6B90((unsigned __int8 *)v54, v53);
              v54 = v154;
            }
            sub_BD84D0((__int64)v53, v54);
          }
        }
        v21 = a2;
        if ( *(_QWORD *)(a2 + 16) )
        {
          sub_10A5FE0(*(_QWORD *)(a1 + 40), a2);
          if ( a2 == v50 )
            v50 = sub_ACADE0(*(__int64 ***)(a2 + 8));
          if ( !*(_QWORD *)(v50 + 16)
            && *(_BYTE *)v50 > 0x1Cu
            && (*(_BYTE *)(v50 + 7) & 0x10) == 0
            && (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
          {
            sub_BD6B90((unsigned __int8 *)v50, (unsigned __int8 *)a2);
          }
          sub_BD84D0(a2, v50);
        }
        else
        {
          v21 = 0;
        }
        sub_C7D6A0(v162, 24LL * v164, 8);
        v13 = (__int64 *)(16LL * v160);
        sub_C7D6A0((__int64)v158, (__int64)v13, 8);
        goto LABEL_12;
      }
    }
    v43 = &v169;
    v153 = 0;
    while ( 1 )
    {
      v151 = 16LL * v153;
      v44 = v174[*(unsigned int *)((char *)base + v151)];
      v150 = *(_DWORD *)((char *)base + v151 + 4);
      v45 = *(_QWORD *)(*(_QWORD *)((char *)base + v151 + 8) + 8LL);
      v167[0].m128i_i32[2] = v150;
      v167[0].m128i_i64[0] = v44;
      v46 = sub_BCAE30(v45);
      v170 = v47;
      v169 = (__int64 *)v46;
      v167[0].m128i_i32[3] = sub_CA1930(v43);
      v48 = sub_116D540((__int64)&v161, v167)->m128i_i64[0];
      if ( !v48 )
        break;
LABEL_75:
      v49 = *(_QWORD *)((char *)base + v151 + 8);
      if ( *(_QWORD *)(v49 + 16) )
      {
        sub_10A5FE0(*(_QWORD *)(a1 + 40), *(_QWORD *)((char *)base + v151 + 8));
        if ( v48 == v49 )
          v48 = sub_ACADE0(*(__int64 ***)(v48 + 8));
        if ( !*(_QWORD *)(v48 + 16)
          && *(_BYTE *)v48 > 0x1Cu
          && (*(_BYTE *)(v48 + 7) & 0x10) == 0
          && (*(_BYTE *)(v49 + 7) & 0x10) != 0 )
        {
          sub_BD6B90((unsigned __int8 *)v48, (unsigned __int8 *)v49);
        }
        sub_BD84D0(v49, v48);
      }
      if ( v149 == ++v153 )
        goto LABEL_83;
    }
    v141 = (__int64 *)v44;
    v168 = 265;
    v167[0].m128i_i32[0] = v150;
    v165[0] = (__int64)sub_BD5D20(v44);
    v165[2] = (__int64)".off";
    v55 = v168;
    v166 = 773;
    v165[1] = v56;
    if ( (_BYTE)v168 )
    {
      if ( (_BYTE)v168 == 1 )
      {
        v131 = v165;
        v132 = 10;
        v133 = v43;
        while ( v132 )
        {
          *v133 = *(_DWORD *)v131;
          v131 = (__int64 *)((char *)v131 + 4);
          ++v133;
          --v132;
        }
      }
      else
      {
        if ( HIBYTE(v168) == 1 )
        {
          v140 = v167[0].m128i_i64[1];
          v57 = (__m128i *)v167[0].m128i_i64[0];
        }
        else
        {
          v57 = v167;
          v55 = 2;
        }
        v171 = v57;
        v169 = v165;
        LOBYTE(v173) = 2;
        v172 = v140;
        HIBYTE(v173) = v55;
      }
    }
    else
    {
      v173 = 256;
    }
    v147 = *(_DWORD *)(v44 + 4) & 0x7FFFFFF;
    v58 = sub_BD2DA0(80);
    v59 = v58;
    if ( v58 )
    {
      sub_B44260(v58, v45, 55, 0x8000000u, v44 + 24, 0);
      *(_DWORD *)(v59 + 72) = v147;
      sub_BD6B50((unsigned __int8 *)v59, (const char **)v43);
      sub_BD2A10(v59, *(_DWORD *)(v59 + 72), 1);
    }
    if ( (*(_BYTE *)(v44 + 7) & 0x40) != 0 )
    {
      v60 = *(__int64 **)(v44 - 8);
      v61 = *(_DWORD *)(v44 + 4) & 0x7FFFFFF;
      v148 = v60;
      v141 = &v60[4 * v61];
    }
    else
    {
      v61 = *(_DWORD *)(v44 + 4) & 0x7FFFFFF;
      v148 = (__int64 *)(v44 - 32 * v61);
      v60 = *(__int64 **)(v44 - 8);
    }
    v62 = 4LL * *(unsigned int *)(v44 + 72);
    v139 = &v60[v61 + v62];
    v146 = &v60[v62];
    if ( v139 != &v60[v62] && v148 != v141 )
    {
      v142 = v44;
      while ( 1 )
      {
        v63 = v160;
        v64 = *v146;
        v165[0] = *v146;
        v65 = *v148;
        if ( !v160 )
          break;
        v66 = 0;
        v67 = 1;
        v68 = (v160 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
        v69 = &v158[2 * v68];
        v70 = *v69;
        if ( v64 != *v69 )
        {
          while ( v70 != -4096 )
          {
            if ( v70 == -8192 && !v66 )
              v66 = v69;
            v68 = (v160 - 1) & (v67 + v68);
            v69 = &v158[2 * v68];
            v70 = *v69;
            if ( v64 == *v69 )
              goto LABEL_115;
            ++v67;
          }
          if ( v66 )
            v69 = v66;
          ++v157;
          v87 = v159 + 1;
          v169 = v69;
          if ( 4 * ((int)v159 + 1) < 3 * v160 )
          {
            if ( v160 - HIDWORD(v159) - v87 > v160 >> 3 )
            {
LABEL_156:
              LODWORD(v159) = v87;
              if ( *v69 != -4096 )
                --HIDWORD(v159);
              *v69 = v64;
              v88 = v69 + 1;
              *v88 = 0;
              v143 = v88;
              goto LABEL_159;
            }
LABEL_240:
            sub_116E750((__int64)&v157, v63);
            sub_116E230((__int64)&v157, v165, v43);
            v64 = v165[0];
            v87 = v159 + 1;
            v69 = v169;
            goto LABEL_156;
          }
LABEL_239:
          v63 = 2 * v160;
          goto LABEL_240;
        }
LABEL_115:
        v143 = v69 + 1;
        v71 = v69[1];
        if ( v71 )
        {
          v72 = *(_DWORD *)(v59 + 4) & 0x7FFFFFF;
          if ( v72 == *(_DWORD *)(v59 + 72) )
          {
            v145 = v71;
            sub_B48D90(v59);
            v71 = v145;
            v72 = *(_DWORD *)(v59 + 4) & 0x7FFFFFF;
          }
          v73 = (v72 + 1) & 0x7FFFFFF;
          v74 = v73 | *(_DWORD *)(v59 + 4) & 0xF8000000;
          v75 = *(_QWORD *)(v59 - 8) + 32LL * (unsigned int)(v73 - 1);
          *(_DWORD *)(v59 + 4) = v74;
          if ( *(_QWORD *)v75 )
          {
            v76 = *(_QWORD *)(v75 + 8);
            **(_QWORD **)(v75 + 16) = v76;
            if ( v76 )
              *(_QWORD *)(v76 + 16) = *(_QWORD *)(v75 + 16);
          }
          *(_QWORD *)v75 = v71;
          v77 = *(_QWORD *)(v71 + 16);
          *(_QWORD *)(v75 + 8) = v77;
          if ( v77 )
            *(_QWORD *)(v77 + 16) = v75 + 8;
          *(_QWORD *)(v75 + 16) = v71 + 16;
          *(_QWORD *)(v71 + 16) = v75;
          *(_QWORD *)(*(_QWORD *)(v59 - 8)
                    + 32LL * *(unsigned int *)(v59 + 72)
                    + 8LL * ((*(_DWORD *)(v59 + 4) & 0x7FFFFFFu) - 1)) = v64;
          goto LABEL_124;
        }
LABEL_159:
        if ( v142 == v65 )
        {
          *v143 = v59;
          sub_F0A850(v59, v59, v165[0]);
        }
        else
        {
          v167[0].m128i_i64[0] = v142;
          v167[0].m128i_i32[2] = v150;
          v89 = sub_BCAE30(v45);
          v170 = v90;
          v169 = (__int64 *)v89;
          v167[0].m128i_i32[3] = sub_CA1930(v43);
          v91 = sub_116D540((__int64)&v161, v167)->m128i_i64[0];
          if ( v91 )
          {
            *v143 = v91;
            v92 = v165[0];
            v93 = *(_DWORD *)(v59 + 4) & 0x7FFFFFF;
            if ( v93 == *(_DWORD *)(v59 + 72) )
            {
              sub_B48D90(v59);
              v93 = *(_DWORD *)(v59 + 4) & 0x7FFFFFF;
            }
            v94 = (v93 + 1) & 0x7FFFFFF;
            v95 = v94 | *(_DWORD *)(v59 + 4) & 0xF8000000;
            v96 = *(_QWORD *)(v59 - 8) + 32LL * (unsigned int)(v94 - 1);
            *(_DWORD *)(v59 + 4) = v95;
            if ( *(_QWORD *)v96 )
            {
              v97 = *(_QWORD *)(v96 + 8);
              **(_QWORD **)(v96 + 16) = v97;
              if ( v97 )
                *(_QWORD *)(v97 + 16) = *(_QWORD *)(v96 + 16);
            }
            *(_QWORD *)v96 = v91;
            v98 = *(_QWORD *)(v91 + 16);
            *(_QWORD *)(v96 + 8) = v98;
            if ( v98 )
              *(_QWORD *)(v98 + 16) = v96 + 8;
            *(_QWORD *)(v96 + 16) = v91 + 16;
            *(_QWORD *)(v91 + 16) = v96;
            *(_QWORD *)(*(_QWORD *)(v59 - 8)
                      + 32LL * *(unsigned int *)(v59 + 72)
                      + 8LL * ((*(_DWORD *)(v59 + 4) & 0x7FFFFFFu) - 1)) = v92;
          }
          else
          {
            v99 = *(_QWORD *)(v165[0] + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v99 != v165[0] + 48 )
            {
              if ( !v99 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v99 - 24) - 30 <= 0xA )
                v91 = v99 - 24;
            }
            sub_D5F1F0(*(_QWORD *)(a1 + 32), v91);
            v102 = v65;
            if ( v150 )
            {
              v173 = 259;
              v123 = *(__int64 **)(a1 + 32);
              v169 = (__int64 *)"extract";
              v124 = sub_AD64C0(*(_QWORD *)(v65 + 8), v150, 0);
              v102 = sub_F94560(v123, v65, v124, (__int64)v43, 0);
            }
            v168 = 259;
            v103 = *(_QWORD *)(a1 + 32);
            v167[0].m128i_i64[0] = (__int64)"extract.t";
            if ( v45 == *(_QWORD *)(v102 + 8) )
            {
              v104 = v102;
            }
            else
            {
              v134 = (_QWORD *)v103;
              v136 = v102;
              v104 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v103 + 80) + 120LL))(
                       *(_QWORD *)(v103 + 80),
                       38,
                       v102,
                       v45);
              if ( !v104 )
              {
                v117 = v136;
                v137 = v134;
                v173 = 257;
                v104 = sub_B51D30(38, v117, v45, (__int64)v43, 0, 0);
                (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(*(_QWORD *)v137[11] + 16LL))(
                  v137[11],
                  v104,
                  v167,
                  v137[7],
                  v137[8]);
                v118 = *v137 + 16LL * *((unsigned int *)v137 + 2);
                if ( *v134 != v118 )
                {
                  v138 = v59;
                  v119 = *v134;
                  v135 = v43;
                  v120 = v118;
                  do
                  {
                    v121 = *(_QWORD *)(v119 + 8);
                    v122 = *(_DWORD *)v119;
                    v119 += 16;
                    sub_B99FD0(v104, v122, v121);
                  }
                  while ( v120 != v119 );
                  v59 = v138;
                  v43 = v135;
                }
              }
            }
            *v143 = v104;
            v105 = v165[0];
            v106 = *(_DWORD *)(v59 + 4) & 0x7FFFFFF;
            if ( v106 == *(_DWORD *)(v59 + 72) )
            {
              v144 = v165[0];
              sub_B48D90(v59);
              v105 = v144;
              v106 = *(_DWORD *)(v59 + 4) & 0x7FFFFFF;
            }
            v107 = (v106 + 1) & 0x7FFFFFF;
            v108 = v107 | *(_DWORD *)(v59 + 4) & 0xF8000000;
            v109 = *(_QWORD *)(v59 - 8) + 32LL * (unsigned int)(v107 - 1);
            *(_DWORD *)(v59 + 4) = v108;
            if ( *(_QWORD *)v109 )
            {
              v110 = *(_QWORD *)(v109 + 8);
              **(_QWORD **)(v109 + 16) = v110;
              if ( v110 )
                *(_QWORD *)(v110 + 16) = *(_QWORD *)(v109 + 16);
            }
            *(_QWORD *)v109 = v104;
            if ( v104 )
            {
              v111 = *(_QWORD *)(v104 + 16);
              *(_QWORD *)(v109 + 8) = v111;
              if ( v111 )
                *(_QWORD *)(v111 + 16) = v109 + 8;
              *(_QWORD *)(v109 + 16) = v104 + 16;
              *(_QWORD *)(v104 + 16) = v109;
            }
            *(_QWORD *)(*(_QWORD *)(v59 - 8)
                      + 32LL * *(unsigned int *)(v59 + 72)
                      + 8LL * ((*(_DWORD *)(v59 + 4) & 0x7FFFFFFu) - 1)) = v105;
            if ( *(_BYTE *)v65 != 84 )
              goto LABEL_124;
            if ( v181 )
            {
              v112 = v178;
              v113 = &v178[HIDWORD(v179)];
              if ( v178 != v113 )
              {
                while ( v65 != *v112 )
                {
                  if ( v113 == ++v112 )
                    goto LABEL_124;
                }
LABEL_201:
                v114 = (unsigned int)v175;
                if ( (unsigned __int64)(unsigned int)v175 >> 2 )
                {
                  v115 = (char *)v174;
                  v116 = (char *)&v174[4 * ((unsigned __int64)(unsigned int)v175 >> 2)];
                  while ( v65 != *(_QWORD *)v115 )
                  {
                    if ( v65 == *((_QWORD *)v115 + 1) )
                    {
                      v115 += 8;
                      goto LABEL_221;
                    }
                    if ( v65 == *((_QWORD *)v115 + 2) )
                    {
                      v115 += 16;
                      goto LABEL_221;
                    }
                    if ( v65 == *((_QWORD *)v115 + 3) )
                    {
                      v115 += 24;
                      goto LABEL_221;
                    }
                    v115 += 32;
                    if ( v115 == v116 )
                      goto LABEL_218;
                  }
                  goto LABEL_221;
                }
                v116 = (char *)v174;
LABEL_218:
                v115 = (char *)&v174[v114];
                v125 = (char *)&v174[v114] - v116;
                if ( v125 == 16 )
                  goto LABEL_226;
                if ( v125 != 24 )
                {
                  if ( v125 != 8 )
                  {
LABEL_221:
                    v126 = v115 - (char *)v174;
                    v127 = (unsigned int)v184;
                    v128 = (unsigned int)v184 + 1LL;
                    v129 = ((unsigned __int64)v150 << 32) | (unsigned int)(v126 >> 3);
                    if ( v128 > HIDWORD(v184) )
                    {
                      sub_C8D5F0((__int64)&base, v185, v128, 0x10u, v100, v101);
                      v127 = (unsigned int)v184;
                    }
                    v130 = (unsigned __int64 *)((char *)base + 16 * v127);
                    ++v149;
                    *v130 = v129;
                    v130[1] = v104;
                    LODWORD(v184) = v184 + 1;
                    goto LABEL_124;
                  }
LABEL_228:
                  if ( v65 == *(_QWORD *)v116 )
                    v115 = v116;
                  goto LABEL_221;
                }
                if ( v65 != *(_QWORD *)v116 )
                {
                  v116 += 8;
LABEL_226:
                  if ( v65 != *(_QWORD *)v116 )
                  {
                    v116 += 8;
                    goto LABEL_228;
                  }
                }
                v115 = v116;
                goto LABEL_221;
              }
            }
            else if ( sub_C8CA60((__int64)&v177, v65) )
            {
              goto LABEL_201;
            }
          }
        }
LABEL_124:
        ++v146;
        v148 += 4;
        if ( v148 == v141 || v139 == v146 )
        {
          v44 = v142;
          goto LABEL_127;
        }
      }
      ++v157;
      v169 = 0;
      goto LABEL_239;
    }
LABEL_127:
    ++v157;
    if ( !(_DWORD)v159 )
    {
      if ( HIDWORD(v159) )
      {
        v78 = v160;
        if ( v160 <= 0x40 )
        {
LABEL_130:
          v79 = v158;
          v80 = &v158[2 * v78];
          if ( v158 != v80 )
          {
            do
            {
              *v79 = -4096;
              v79 += 2;
            }
            while ( v80 != v79 );
          }
          v159 = 0;
          goto LABEL_133;
        }
        sub_C7D6A0((__int64)v158, 16LL * v160, 8);
        v158 = 0;
        v159 = 0;
        v160 = 0;
      }
LABEL_133:
      v167[0].m128i_i64[0] = v44;
      v48 = v59;
      v167[0].m128i_i32[2] = v150;
      v81 = sub_BCAE30(v45);
      v170 = v82;
      v169 = (__int64 *)v81;
      v167[0].m128i_i32[3] = sub_CA1930(v43);
      sub_116D540((__int64)&v161, v167)->m128i_i64[0] = v59;
      goto LABEL_75;
    }
    v83 = 4 * v159;
    v78 = v160;
    if ( (unsigned int)(4 * v159) < 0x40 )
      v83 = 64;
    if ( v160 <= v83 )
      goto LABEL_130;
    if ( (_DWORD)v159 == 1 )
    {
      v85 = 64;
    }
    else
    {
      _BitScanReverse(&v84, v159 - 1);
      v85 = 1 << (33 - (v84 ^ 0x1F));
      if ( v85 < 64 )
        v85 = 64;
      if ( v85 == v160 )
        goto LABEL_144;
    }
    sub_C7D6A0((__int64)v158, 16LL * v160, 8);
    v86 = ((((((((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
             | (4 * v85 / 3u + 1)
             | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 4)
           | (((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
           | (4 * v85 / 3u + 1)
           | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
           | (4 * v85 / 3u + 1)
           | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 4)
         | (((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
         | (4 * v85 / 3u + 1)
         | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 16;
    v160 = (v86
          | (((((((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
              | (4 * v85 / 3u + 1)
              | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 4)
            | (((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
            | (4 * v85 / 3u + 1)
            | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
            | (4 * v85 / 3u + 1)
            | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 4)
          | (((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
          | (4 * v85 / 3u + 1)
          | ((4 * v85 / 3u + 1) >> 1))
         + 1;
    v158 = (_QWORD *)sub_C7D670(
                       16
                     * ((v86
                       | (((((((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
                           | (4 * v85 / 3u + 1)
                           | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 4)
                         | (((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
                         | (4 * v85 / 3u + 1)
                         | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 8)
                       | (((((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
                         | (4 * v85 / 3u + 1)
                         | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 4)
                       | (((4 * v85 / 3u + 1) | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1)) >> 2)
                       | (4 * v85 / 3u + 1)
                       | ((unsigned __int64)(4 * v85 / 3u + 1) >> 1))
                      + 1),
                       8);
LABEL_144:
    sub_116E2F0((__int64)&v157);
    goto LABEL_133;
  }
  v21 = a2;
  v42 = sub_ACADE0(*(__int64 ***)(a2 + 8));
  if ( !*(_QWORD *)(a2 + 16) )
  {
LABEL_11:
    v21 = 0;
    goto LABEL_12;
  }
  sub_10A5FE0(*(_QWORD *)(a1 + 40), a2);
  if ( v42 == a2 )
    v42 = sub_ACADE0(*(__int64 ***)(a2 + 8));
  if ( !*(_QWORD *)(v42 + 16)
    && *(_BYTE *)v42 > 0x1Cu
    && (*(_BYTE *)(v42 + 7) & 0x10) == 0
    && (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    sub_BD6B90((unsigned __int8 *)v42, (unsigned __int8 *)a2);
  }
  v13 = (__int64 *)v42;
  sub_BD84D0(a2, v42);
LABEL_12:
  if ( !v181 )
    _libc_free(v178, v13);
  if ( v174 != v176 )
    _libc_free(v174, v13);
  if ( base != v185 )
    _libc_free(base, v13);
  return v21;
}
