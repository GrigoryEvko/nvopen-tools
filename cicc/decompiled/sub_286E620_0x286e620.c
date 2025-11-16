// Function: sub_286E620
// Address: 0x286e620
//
void __fastcall sub_286E620(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 *v10; // r13
  __int64 v11; // rax
  __int64 *v12; // r12
  __int64 v13; // r15
  __int64 *v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rbx
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  __int64 *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  float v24; // xmm0_4
  unsigned int v25; // esi
  __int64 v26; // r11
  __int64 v27; // rax
  __int64 v28; // rcx
  int v29; // edx
  __int64 v30; // rdi
  int v31; // edx
  __int64 v32; // r15
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 *v37; // r15
  __int64 v38; // r12
  __int64 *v39; // rax
  __int64 v40; // r12
  __int64 *v41; // rax
  __int64 *v42; // rax
  __int64 v43; // r13
  __int64 v44; // r14
  unsigned int v45; // r9d
  __int64 v46; // rbx
  float v47; // xmm2_4
  __int64 v48; // rsi
  __int64 *v49; // rdi
  unsigned int v50; // r11d
  int v51; // r10d
  __int64 *v52; // rdx
  int v53; // esi
  int v54; // ecx
  unsigned int v55; // eax
  _QWORD *v56; // rdx
  __int64 v57; // r11
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  _QWORD *v60; // rax
  __int64 v61; // rbx
  __int64 v62; // rdi
  _QWORD *i; // rdx
  __int64 j; // rax
  __int64 v65; // r13
  int v66; // r11d
  _QWORD *v67; // r10
  unsigned int v68; // ecx
  _QWORD *v69; // rdx
  __int64 v70; // r8
  unsigned int v71; // edi
  unsigned int v72; // edi
  unsigned int v73; // ecx
  __int64 v74; // r8
  int v75; // r10d
  __int64 *v76; // r9
  unsigned int v77; // r14d
  unsigned int v78; // ecx
  unsigned int v79; // r15d
  __int64 v80; // rbx
  float v81; // xmm2_4
  int v82; // r10d
  unsigned int v83; // eax
  __int64 v84; // rcx
  int v85; // eax
  __int64 v86; // rdx
  __int64 v87; // rax
  char *v88; // rdi
  __int64 *v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r12
  char v93; // di
  __int64 *v94; // r13
  __int64 *v95; // r14
  __int64 v96; // rsi
  __int64 *v97; // rax
  __int64 v98; // rsi
  __int64 *v99; // rax
  unsigned int v100; // eax
  __int64 v101; // r8
  int v102; // edi
  int v103; // edi
  _QWORD *v104; // rsi
  unsigned int v105; // eax
  __int64 v106; // r9
  __int64 *v107; // r13
  __int64 *v108; // r12
  __m128i v109; // xmm2
  __int64 v110; // r8
  char v111; // al
  __int64 v112; // rax
  __int64 v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // r8
  __int64 v116; // r9
  _QWORD *v117; // rcx
  _QWORD *v118; // rdx
  float v119; // xmm0_4
  unsigned int v120; // ecx
  int v121; // edi
  __int64 v122; // rsi
  unsigned int v123; // edx
  int v124; // eax
  __int64 *v125; // r13
  __int64 v126; // rdi
  unsigned int v127; // edx
  int v128; // eax
  __int64 v129; // rdi
  int v130; // esi
  __int64 v131; // rcx
  __int64 v132; // rsi
  unsigned int v133; // r11d
  int v134; // r10d
  __int64 *v135; // r9
  __int64 *v136; // rdx
  unsigned int v137; // r8d
  int v138; // ecx
  __int64 v139; // rsi
  __int64 v140; // rbx
  int v141; // ecx
  __int64 v142; // rsi
  int v143; // edi
  unsigned int v144; // r8d
  __int64 v145; // rax
  int v146; // r9d
  __int64 v147; // rsi
  __int64 v148; // rdx
  unsigned int v149; // r8d
  int v150; // ecx
  __int64 v151; // rsi
  int v152; // esi
  __int64 *v153; // rcx
  float v154; // xmm0_4
  __int64 v155; // [rsp+10h] [rbp-190h]
  __int64 v156; // [rsp+18h] [rbp-188h]
  unsigned int v157; // [rsp+20h] [rbp-180h]
  float *v158; // [rsp+20h] [rbp-180h]
  __int64 v159; // [rsp+20h] [rbp-180h]
  __int64 v160; // [rsp+28h] [rbp-178h]
  __int64 v161; // [rsp+30h] [rbp-170h]
  __int64 v163; // [rsp+40h] [rbp-160h]
  __int64 v164; // [rsp+48h] [rbp-158h]
  float v165; // [rsp+54h] [rbp-14Ch]
  unsigned __int64 v166; // [rsp+58h] [rbp-148h]
  __int64 v167; // [rsp+60h] [rbp-140h]
  unsigned int v168; // [rsp+60h] [rbp-140h]
  float v169; // [rsp+70h] [rbp-130h]
  float v170; // [rsp+74h] [rbp-12Ch]
  __int64 v171; // [rsp+78h] [rbp-128h]
  __int64 v172; // [rsp+80h] [rbp-120h]
  __int64 v173; // [rsp+88h] [rbp-118h]
  __int64 *v174; // [rsp+90h] [rbp-110h]
  int v175; // [rsp+90h] [rbp-110h]
  float v176; // [rsp+98h] [rbp-108h]
  float v177; // [rsp+98h] [rbp-108h]
  __int64 v178; // [rsp+A0h] [rbp-100h] BYREF
  _QWORD *v179; // [rsp+A8h] [rbp-F8h]
  __int64 v180; // [rsp+B0h] [rbp-F0h]
  unsigned int v181; // [rsp+B8h] [rbp-E8h]
  __int64 v182; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 *v183; // [rsp+C8h] [rbp-D8h]
  __int64 v184; // [rsp+D0h] [rbp-D0h]
  int v185; // [rsp+D8h] [rbp-C8h]
  char v186; // [rsp+DCh] [rbp-C4h]
  char v187; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v188; // [rsp+100h] [rbp-A0h]
  __m128i v189; // [rsp+108h] [rbp-98h]
  char v190; // [rsp+118h] [rbp-88h]
  __int64 v191; // [rsp+120h] [rbp-80h]
  char *v192[2]; // [rsp+128h] [rbp-78h] BYREF
  _BYTE v193[32]; // [rsp+138h] [rbp-68h] BYREF
  __int64 v194; // [rsp+158h] [rbp-48h]
  __m128i v195; // [rsp+160h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 1320);
  v160 = *(unsigned int *)(a1 + 1328);
  v7 = v6 + 2184 * v160;
  if ( v6 == v7 )
  {
    if ( (unsigned int)qword_5001308 > 1 )
      return;
  }
  else
  {
    a6 = (unsigned int)qword_5001308;
    v8 = 1;
    while ( 1 )
    {
      v9 = *(unsigned int *)(v6 + 768);
      if ( (unsigned int)v9 >= (unsigned int)qword_5001308 )
        break;
      v8 *= v9;
      if ( (unsigned int)qword_5001308 <= v8 )
        break;
      v6 += 2184;
      if ( v7 == v6 )
        return;
    }
  }
  v182 = 0;
  v183 = (__int64 *)&v187;
  v184 = 4;
  v10 = *(__int64 **)(a1 + 36312);
  v11 = *(unsigned int *)(a1 + 36320);
  v185 = 0;
  v186 = 1;
  v178 = 0;
  v12 = &v10[v11];
  v179 = 0;
  v180 = 0;
  v181 = 0;
  if ( v10 == v12 )
    goto LABEL_37;
  v13 = *v10;
LABEL_8:
  v14 = v183;
  v15 = &v183[HIDWORD(v184)];
  if ( v183 != v15 )
  {
    do
    {
      if ( v13 == *v14 )
        goto LABEL_12;
      ++v14;
    }
    while ( v15 != v14 );
  }
LABEL_15:
  v16 = *(_QWORD *)(a1 + 1320);
  v17 = v16 + 2184LL * *(unsigned int *)(a1 + 1328);
  if ( v16 != v17 )
  {
    v176 = 1.0;
    while ( 1 )
    {
      if ( *(_BYTE *)(v16 + 2148) )
      {
        v18 = *(_QWORD **)(v16 + 2128);
        v19 = &v18[*(unsigned int *)(v16 + 2140)];
        if ( v18 == v19 )
          goto LABEL_24;
        while ( v13 != *v18 )
        {
          if ( v19 == ++v18 )
            goto LABEL_24;
        }
      }
      else if ( !sub_C8CA60(v16 + 2120, v13) )
      {
        goto LABEL_24;
      }
      v24 = sub_2850D10(v16, v13);
      if ( v24 == 0.0 )
      {
        if ( !v186 )
          goto LABEL_73;
        v42 = v183;
        v20 = &v183[HIDWORD(v184)];
        if ( v183 != v20 )
        {
          while ( v13 != *v42 )
          {
            if ( v20 == ++v42 )
              goto LABEL_71;
          }
          goto LABEL_24;
        }
LABEL_71:
        if ( HIDWORD(v184) < (unsigned int)v184 )
        {
          ++HIDWORD(v184);
          *v20 = v13;
          ++v182;
        }
        else
        {
LABEL_73:
          sub_C8CC70((__int64)&v182, v13, (__int64)v20, v21, v22, v23);
        }
      }
      else
      {
        v176 = v24 * v176;
      }
LABEL_24:
      v16 += 2184;
      if ( v16 == v17 )
      {
        v25 = v181;
        if ( v181 )
          goto LABEL_26;
LABEL_204:
        ++v178;
LABEL_205:
        sub_286E430((__int64)&v178, 2 * v25);
        if ( v181 )
        {
          v31 = v180 + 1;
          v120 = (v181 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v27 = (__int64)&v179[2 * v120];
          a6 = *(_QWORD *)v27;
          if ( v13 != *(_QWORD *)v27 )
          {
            v121 = 1;
            v122 = 0;
            while ( a6 != -4096 )
            {
              if ( !v122 && a6 == -8192 )
                v122 = v27;
              v120 = (v181 - 1) & (v121 + v120);
              v27 = (__int64)&v179[2 * v120];
              a6 = *(_QWORD *)v27;
              if ( v13 == *(_QWORD *)v27 )
                goto LABEL_33;
              ++v121;
            }
            if ( v122 )
              v27 = v122;
          }
          goto LABEL_33;
        }
LABEL_338:
        LODWORD(v180) = v180 + 1;
        BUG();
      }
    }
  }
  v25 = v181;
  v176 = 1.0;
  if ( !v181 )
    goto LABEL_204;
LABEL_26:
  a6 = v25 - 1;
  LODWORD(v26) = a6 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v27 = (__int64)&v179[2 * (unsigned int)v26];
  v28 = *(_QWORD *)v27;
  if ( v13 == *(_QWORD *)v27 )
    goto LABEL_12;
  v29 = 1;
  v30 = 0;
  while ( 1 )
  {
    if ( v28 == -4096 )
    {
      if ( v30 )
        v27 = v30;
      ++v178;
      v31 = v180 + 1;
      if ( 4 * ((int)v180 + 1) >= 3 * v25 )
        goto LABEL_205;
      if ( v25 - HIDWORD(v180) - v31 <= v25 >> 3 )
      {
        sub_286E430((__int64)&v178, v25);
        if ( !v181 )
          goto LABEL_338;
        a6 = 0;
        LODWORD(v140) = (v181 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v31 = v180 + 1;
        v141 = 1;
        v27 = (__int64)&v179[2 * (unsigned int)v140];
        v142 = *(_QWORD *)v27;
        if ( v13 != *(_QWORD *)v27 )
        {
          while ( v142 != -4096 )
          {
            if ( v142 == -8192 && !a6 )
              a6 = v27;
            v140 = (v181 - 1) & ((_DWORD)v140 + v141);
            v27 = (__int64)&v179[2 * v140];
            v142 = *(_QWORD *)v27;
            if ( v13 == *(_QWORD *)v27 )
              goto LABEL_33;
            ++v141;
          }
          if ( a6 )
            v27 = a6;
        }
      }
LABEL_33:
      LODWORD(v180) = v31;
      if ( *(_QWORD *)v27 != -4096 )
        --HIDWORD(v180);
      ++v10;
      *(_QWORD *)v27 = v13;
      *(float *)(v27 + 8) = v176;
      if ( v12 != v10 )
        goto LABEL_13;
      goto LABEL_36;
    }
    if ( !v30 && v28 == -8192 )
      v30 = v27;
    v26 = (unsigned int)a6 & ((_DWORD)v26 + v29);
    v27 = (__int64)&v179[2 * v26];
    v28 = *(_QWORD *)v27;
    if ( v13 == *(_QWORD *)v27 )
      break;
    ++v29;
  }
LABEL_12:
  while ( v12 != ++v10 )
  {
LABEL_13:
    v13 = *v10;
    if ( v186 )
      goto LABEL_8;
    if ( !sub_C8CA60((__int64)&v182, v13) )
      goto LABEL_15;
  }
LABEL_36:
  v160 = *(unsigned int *)(a1 + 1328);
LABEL_37:
  v161 = 0;
  v163 = 0;
  if ( v160 )
  {
LABEL_38:
    v32 = *(_QWORD *)(a1 + 1320) + v161;
    v166 = *(unsigned int *)(v32 + 768);
    if ( v166 <= 1 )
      goto LABEL_62;
    v33 = *(_QWORD *)(v32 + 760);
    v172 = 0;
    v171 = 0;
    v34 = *(unsigned int *)(v33 + 48);
    v164 = 0;
    v173 = *(_QWORD *)(a1 + 1320) + v161;
    v169 = (float)(int)(v34 - ((*(_QWORD *)(v33 + 88) == 0) - 1));
    v165 = v169;
LABEL_40:
    v35 = v172 + v33;
    v36 = *(_QWORD *)(v35 + 40);
    v167 = v35;
    v174 = (__int64 *)(v36 + 8 * v34);
    if ( v174 == (__int64 *)v36 )
    {
      v170 = 0.0;
      v177 = 0.0;
      goto LABEL_48;
    }
    v170 = 0.0;
    v37 = *(__int64 **)(v35 + 40);
    v177 = 0.0;
    while ( 1 )
    {
LABEL_42:
      v38 = *v37;
      if ( v186 )
      {
        v39 = v183;
        v36 = (__int64)&v183[HIDWORD(v184)];
        if ( v183 != (__int64 *)v36 )
        {
          while ( v38 != *v39 )
          {
            if ( (__int64 *)v36 == ++v39 )
              goto LABEL_75;
          }
          goto LABEL_47;
        }
      }
      else if ( sub_C8CA60((__int64)&v182, v38) )
      {
        goto LABEL_47;
      }
LABEL_75:
      v43 = v181;
      v44 = (__int64)v179;
      if ( !v181 )
      {
        ++v178;
        goto LABEL_99;
      }
      v45 = v181 - 1;
      v46 = (__int64)&v179[2 * ((v181 - 1) & (((unsigned int)v38 >> 4) ^ ((unsigned int)v38 >> 9)))];
      if ( v38 == *(_QWORD *)v46 )
        break;
      v48 = *(_QWORD *)v46;
      v49 = &v179[2 * (v45 & (((unsigned int)v38 >> 4) ^ ((unsigned int)v38 >> 9)))];
      v50 = (v181 - 1) & (((unsigned int)v38 >> 4) ^ ((unsigned int)v38 >> 9));
      v51 = 1;
      v52 = 0;
      while ( 1 )
      {
        if ( v48 == -4096 )
        {
          if ( !v52 )
            v52 = v49;
          ++v178;
          v53 = v180 + 1;
          if ( 4 * ((int)v180 + 1) < 3 * v181 )
          {
            if ( v181 - HIDWORD(v180) - v53 <= v181 >> 3 )
            {
              sub_286E430((__int64)&v178, v181);
              if ( !v181 )
              {
LABEL_341:
                LODWORD(v180) = v180 + 1;
                BUG();
              }
              v76 = 0;
              v82 = 1;
              v83 = (v181 - 1) & (((unsigned int)v38 >> 4) ^ ((unsigned int)v38 >> 9));
              v53 = v180 + 1;
              v52 = &v179[2 * v83];
              v84 = *v52;
              if ( v38 != *v52 )
              {
                while ( v84 != -4096 )
                {
                  if ( !v76 && v84 == -8192 )
                    v76 = v52;
                  v83 = (v181 - 1) & (v82 + v83);
                  v52 = &v179[2 * v83];
                  v84 = *v52;
                  if ( v38 == *v52 )
                    goto LABEL_88;
                  ++v82;
                }
                goto LABEL_118;
              }
            }
            goto LABEL_88;
          }
LABEL_99:
          v58 = (((((((2 * v181 - 1) | ((unsigned __int64)(2 * v181 - 1) >> 1)) >> 2)
                  | (2 * v181 - 1)
                  | ((unsigned __int64)(2 * v181 - 1) >> 1)) >> 4)
                | (((2 * v181 - 1) | ((unsigned __int64)(2 * v181 - 1) >> 1)) >> 2)
                | (2 * v181 - 1)
                | ((unsigned __int64)(2 * v181 - 1) >> 1)) >> 8)
              | (((((2 * v181 - 1) | ((unsigned __int64)(2 * v181 - 1) >> 1)) >> 2)
                | (2 * v181 - 1)
                | ((unsigned __int64)(2 * v181 - 1) >> 1)) >> 4)
              | (((2 * v181 - 1) | ((unsigned __int64)(2 * v181 - 1) >> 1)) >> 2)
              | (2 * v181 - 1)
              | ((unsigned __int64)(2 * v181 - 1) >> 1);
          v59 = ((v58 >> 16) | v58) + 1;
          if ( (unsigned int)v59 < 0x40 )
            LODWORD(v59) = 64;
          v181 = v59;
          v60 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v59, 8);
          v179 = v60;
          if ( v44 )
          {
            v180 = 0;
            v61 = 16 * v43;
            v62 = v44 + 16 * v43;
            for ( i = &v60[2 * v181]; i != v60; v60 += 2 )
            {
              if ( v60 )
                *v60 = -4096;
            }
            for ( j = v44; v62 != j; j += 16 )
            {
              v65 = *(_QWORD *)j;
              if ( *(_QWORD *)j != -4096 && v65 != -8192 )
              {
                if ( !v181 )
                {
                  MEMORY[0] = *(_QWORD *)j;
                  BUG();
                }
                v66 = 1;
                v67 = 0;
                v68 = (v181 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
                v69 = &v179[2 * v68];
                v70 = *v69;
                if ( v65 != *v69 )
                {
                  while ( v70 != -4096 )
                  {
                    if ( v70 == -8192 && !v67 )
                      v67 = v69;
                    v68 = (v181 - 1) & (v66 + v68);
                    v69 = &v179[2 * v68];
                    v70 = *v69;
                    if ( v65 == *v69 )
                      goto LABEL_111;
                    ++v66;
                  }
                  if ( v67 )
                    v69 = v67;
                }
LABEL_111:
                *v69 = v65;
                *((_DWORD *)v69 + 2) = *(_DWORD *)(j + 8);
                LODWORD(v180) = v180 + 1;
              }
            }
            sub_C7D6A0(v44, v61, 8);
            v60 = v179;
            v71 = v181;
            v53 = v180 + 1;
          }
          else
          {
            v180 = 0;
            v71 = v181;
            v117 = &v60[2 * v181];
            if ( v60 != v117 )
            {
              v118 = v60;
              do
              {
                if ( v118 )
                  *v118 = -4096;
                v118 += 2;
              }
              while ( v117 != v118 );
            }
            v53 = 1;
          }
          if ( !v71 )
            goto LABEL_341;
          v72 = v71 - 1;
          v73 = v72 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
          v52 = &v60[2 * v73];
          v74 = *v52;
          if ( v38 != *v52 )
          {
            v75 = 1;
            v76 = 0;
            while ( v74 != -4096 )
            {
              if ( !v76 && v74 == -8192 )
                v76 = v52;
              v73 = v72 & (v75 + v73);
              v52 = &v60[2 * v73];
              v74 = *v52;
              if ( v38 == *v52 )
                goto LABEL_88;
              ++v75;
            }
LABEL_118:
            if ( v76 )
              v52 = v76;
          }
LABEL_88:
          LODWORD(v180) = v53;
          if ( *v52 != -4096 )
            --HIDWORD(v180);
          *v52 = v38;
          *((_DWORD *)v52 + 2) = 0;
          v177 = (float)(0.0 / sub_2850D10(v173, v38)) + v177;
          if ( *(_WORD *)(v38 + 24) == 8 )
          {
            LODWORD(v43) = v181;
            if ( !v181 )
            {
              ++v178;
              goto LABEL_93;
            }
            LODWORD(a6) = v181 - 1;
            v44 = (__int64)v179;
            v100 = ((unsigned int)v38 >> 4) ^ ((unsigned int)v38 >> 9);
            LODWORD(v35) = (v181 - 1) & v100;
            v46 = (__int64)&v179[2 * (unsigned int)v35];
            v101 = *(_QWORD *)v46;
            if ( v38 == *(_QWORD *)v46 )
            {
LABEL_78:
              v47 = *(float *)(v46 + 8);
            }
            else
            {
LABEL_160:
              v102 = 1;
              v56 = 0;
              while ( v101 != -4096 )
              {
                if ( v101 == -8192 && !v56 )
                  v56 = (_QWORD *)v46;
                LODWORD(v35) = a6 & (v102 + v35);
                v46 = v44 + 16LL * (unsigned int)v35;
                v101 = *(_QWORD *)v46;
                if ( v38 == *(_QWORD *)v46 )
                  goto LABEL_78;
                ++v102;
              }
              if ( !v56 )
                v56 = (_QWORD *)v46;
              ++v178;
              v54 = v180 + 1;
              if ( 4 * ((int)v180 + 1) >= (unsigned int)(3 * v43) )
              {
LABEL_93:
                sub_286E430((__int64)&v178, 2 * v43);
                if ( !v181 )
                  goto LABEL_340;
                v54 = v180 + 1;
                v55 = (v181 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                v56 = &v179[2 * v55];
                v57 = *v56;
                if ( v38 != *v56 )
                {
                  v143 = 1;
                  v104 = 0;
                  while ( v57 != -4096 )
                  {
                    if ( !v104 && v57 == -8192 )
                      v104 = v56;
                    v55 = (v181 - 1) & (v143 + v55);
                    v56 = &v179[2 * v55];
                    v57 = *v56;
                    if ( v38 == *v56 )
                      goto LABEL_95;
                    ++v143;
                  }
                  goto LABEL_253;
                }
              }
              else if ( (int)v43 - (v54 + HIDWORD(v180)) <= (unsigned int)v43 >> 3 )
              {
                v157 = v100;
                sub_286E430((__int64)&v178, v43);
                if ( !v181 )
                {
LABEL_340:
                  LODWORD(v180) = v180 + 1;
                  BUG();
                }
                v103 = 1;
                v104 = 0;
                v105 = (v181 - 1) & v157;
                v54 = v180 + 1;
                v56 = &v179[2 * v105];
                v106 = *v56;
                if ( v38 != *v56 )
                {
                  while ( v106 != -4096 )
                  {
                    if ( v106 == -8192 && !v104 )
                      v104 = v56;
                    v105 = (v181 - 1) & (v103 + v105);
                    v56 = &v179[2 * v105];
                    v106 = *v56;
                    if ( v38 == *v56 )
                      goto LABEL_95;
                    ++v103;
                  }
LABEL_253:
                  if ( v104 )
                    v56 = v104;
                }
              }
LABEL_95:
              LODWORD(v180) = v54;
              if ( *v56 != -4096 )
                --HIDWORD(v180);
              *v56 = v38;
              v47 = 0.0;
              *((_DWORD *)v56 + 2) = 0;
            }
            ++v37;
            v170 = v170 + (float)(v47 / sub_2850D10(v173, v38));
            if ( v174 == v37 )
              goto LABEL_48;
            goto LABEL_42;
          }
LABEL_47:
          if ( v174 == ++v37 )
            goto LABEL_48;
          goto LABEL_42;
        }
        if ( v52 || v48 != -8192 )
          v49 = v52;
        v50 = v45 & (v51 + v50);
        v158 = (float *)&v179[2 * v50];
        v48 = *(_QWORD *)v158;
        if ( v38 == *(_QWORD *)v158 )
          break;
        ++v51;
        v52 = v49;
        v49 = &v179[2 * v50];
      }
      v155 = *(_QWORD *)v46;
      v119 = sub_2850D10(v173, v38);
      v35 = ((_DWORD)v43 - 1) & (((unsigned int)v38 >> 4) ^ ((unsigned int)v38 >> 9));
      a6 = (unsigned int)(v43 - 1);
      v101 = v155;
      v100 = ((unsigned int)v38 >> 4) ^ ((unsigned int)v38 >> 9);
      v177 = (float)(v158[2] / v119) + v177;
      if ( *(_WORD *)(v38 + 24) == 8 )
        goto LABEL_160;
      if ( v174 == ++v37 )
      {
LABEL_48:
        v40 = *(_QWORD *)(v167 + 88);
        if ( !v40 )
          goto LABEL_54;
        if ( v186 )
        {
          v41 = v183;
          v36 = (__int64)&v183[HIDWORD(v184)];
          if ( v183 != (__int64 *)v36 )
          {
            do
            {
              if ( v40 == *v41 )
                goto LABEL_54;
              ++v41;
            }
            while ( (__int64 *)v36 != v41 );
          }
        }
        else if ( sub_C8CA60((__int64)&v182, v40) )
        {
          goto LABEL_54;
        }
        v77 = v181;
        if ( !v181 )
        {
          ++v178;
          goto LABEL_213;
        }
        v78 = v181 - 1;
        v79 = ((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4);
        v80 = (__int64)&v179[2 * ((v181 - 1) & v79)];
        if ( v40 != *(_QWORD *)v80 )
        {
          v132 = *(_QWORD *)v80;
          v125 = &v179[2 * (v78 & v79)];
          v133 = (v181 - 1) & v79;
          v134 = 1;
          v135 = 0;
          while ( v132 != -4096 )
          {
            if ( v132 == -8192 && !v135 )
              v135 = v125;
            v133 = v78 & (v134 + v133);
            v125 = &v179[2 * v133];
            v132 = *v125;
            if ( v40 == *v125 )
            {
              v156 = (__int64)v179;
              v159 = *(_QWORD *)v80;
              v168 = v181 - 1;
              v175 = v78 & v79;
              v154 = sub_2850D10(v173, v40);
              v144 = v175;
              v35 = v168;
              v145 = v159;
              v36 = v156;
              v177 = (float)(*((float *)v125 + 2) / v154) + v177;
              if ( *(_WORD *)(v40 + 24) != 8 )
                goto LABEL_54;
LABEL_273:
              if ( v40 == v145 )
              {
LABEL_125:
                v81 = *(float *)(v80 + 8);
                goto LABEL_126;
              }
              v146 = 1;
              v147 = 0;
              while ( v145 != -4096 )
              {
                if ( !v147 && v145 == -8192 )
                  v147 = v80;
                v144 = v35 & (v146 + v144);
                v80 = v36 + 16LL * v144;
                v145 = *(_QWORD *)v80;
                if ( v40 == *(_QWORD *)v80 )
                  goto LABEL_125;
                ++v146;
              }
              if ( v147 )
                v80 = v147;
              ++v178;
              v128 = v180 + 1;
              if ( 4 * ((int)v180 + 1) < 3 * v77 )
              {
                if ( v77 - (v128 + HIDWORD(v180)) > v77 >> 3 )
                  goto LABEL_280;
                sub_286E430((__int64)&v178, v77);
                if ( v181 )
                {
                  v148 = 0;
                  v149 = (v181 - 1) & v79;
                  v150 = 1;
                  v128 = v180 + 1;
                  v80 = (__int64)&v179[2 * v149];
                  v151 = *(_QWORD *)v80;
                  if ( v40 != *(_QWORD *)v80 )
                  {
                    while ( v151 != -4096 )
                    {
                      if ( v151 == -8192 && !v148 )
                        v148 = v80;
                      v149 = (v181 - 1) & (v150 + v149);
                      v80 = (__int64)&v179[2 * v149];
                      v151 = *(_QWORD *)v80;
                      if ( v40 == *(_QWORD *)v80 )
                        goto LABEL_280;
                      ++v150;
                    }
                    if ( v148 )
                      v80 = v148;
                  }
LABEL_280:
                  LODWORD(v180) = v128;
                  if ( *(_QWORD *)v80 != -4096 )
                    --HIDWORD(v180);
                  *(_QWORD *)v80 = v40;
                  v81 = 0.0;
                  *(_DWORD *)(v80 + 8) = 0;
LABEL_126:
                  v170 = (float)(v81 / sub_2850D10(v173, v40)) + v170;
                  goto LABEL_54;
                }
LABEL_336:
                LODWORD(v180) = v180 + 1;
                BUG();
              }
LABEL_220:
              sub_286E430((__int64)&v178, 2 * v77);
              if ( v181 )
              {
                v127 = (v181 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
                v128 = v180 + 1;
                v80 = (__int64)&v179[2 * v127];
                v129 = *(_QWORD *)v80;
                if ( v40 != *(_QWORD *)v80 )
                {
                  v130 = 1;
                  v131 = 0;
                  while ( v129 != -4096 )
                  {
                    if ( !v131 && v129 == -8192 )
                      v131 = v80;
                    v127 = (v181 - 1) & (v130 + v127);
                    v80 = (__int64)&v179[2 * v127];
                    v129 = *(_QWORD *)v80;
                    if ( v40 == *(_QWORD *)v80 )
                      goto LABEL_280;
                    ++v130;
                  }
                  if ( v131 )
                    v80 = v131;
                }
                goto LABEL_280;
              }
              goto LABEL_336;
            }
            ++v134;
          }
          if ( v135 )
            v125 = v135;
          ++v178;
          v124 = v180 + 1;
          if ( 4 * ((int)v180 + 1) >= 3 * v181 )
          {
LABEL_213:
            sub_286E430((__int64)&v178, 2 * v181);
            if ( v181 )
            {
              v123 = (v181 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
              v124 = v180 + 1;
              v125 = &v179[2 * v123];
              v126 = *v125;
              if ( v40 != *v125 )
              {
                v152 = 1;
                v153 = 0;
                while ( v126 != -4096 )
                {
                  if ( !v153 && v126 == -8192 )
                    v153 = v125;
                  v123 = (v181 - 1) & (v152 + v123);
                  v125 = &v179[2 * v123];
                  v126 = *v125;
                  if ( v40 == *v125 )
                    goto LABEL_215;
                  ++v152;
                }
                if ( v153 )
                  v125 = v153;
              }
              goto LABEL_215;
            }
          }
          else
          {
            if ( v181 - HIDWORD(v180) - v124 > v181 >> 3 )
            {
LABEL_215:
              LODWORD(v180) = v124;
              if ( *v125 != -4096 )
                --HIDWORD(v180);
              *v125 = v40;
              *((_DWORD *)v125 + 2) = 0;
              v177 = (float)(0.0 / sub_2850D10(v173, v40)) + v177;
              if ( *(_WORD *)(v40 + 24) != 8 )
                goto LABEL_54;
              v77 = v181;
              if ( v181 )
              {
                LODWORD(v35) = v181 - 1;
                v36 = (__int64)v179;
                v79 = ((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4);
                v144 = (v181 - 1) & v79;
                v80 = (__int64)&v179[2 * v144];
                v145 = *(_QWORD *)v80;
                goto LABEL_273;
              }
              ++v178;
              goto LABEL_220;
            }
            sub_286E430((__int64)&v178, v181);
            if ( v181 )
            {
              v136 = 0;
              v137 = (v181 - 1) & v79;
              v138 = 1;
              v124 = v180 + 1;
              v125 = &v179[2 * v137];
              v139 = *v125;
              if ( v40 != *v125 )
              {
                while ( v139 != -4096 )
                {
                  if ( !v136 && v139 == -8192 )
                    v136 = v125;
                  v137 = (v181 - 1) & (v138 + v137);
                  v125 = &v179[2 * v137];
                  v139 = *v125;
                  if ( v40 == *v125 )
                    goto LABEL_215;
                  ++v138;
                }
                if ( v136 )
                  v125 = v136;
              }
              goto LABEL_215;
            }
          }
          LODWORD(v180) = v180 + 1;
          BUG();
        }
        v177 = (float)(*(float *)(v80 + 8) / sub_2850D10(v173, v40)) + v177;
        if ( *(_WORD *)(v40 + 24) == 8 )
          goto LABEL_125;
LABEL_54:
        if ( v169 > v177 )
        {
          v164 = v171;
          v165 = v170;
          v169 = v177;
        }
        else if ( v177 == v169 && v165 > v170 )
        {
          v165 = v170;
          v169 = v177;
          v164 = v171;
        }
        ++v171;
        v172 += 112;
        if ( v166 != v171 )
        {
          v33 = *(_QWORD *)(v173 + 760);
          v34 = *(unsigned int *)(v33 + v172 + 48);
          goto LABEL_40;
        }
        if ( !v164 )
          goto LABEL_136;
        v107 = *(__int64 **)(v173 + 760);
        v108 = &v107[14 * v164];
        v109 = _mm_loadu_si128((const __m128i *)(v108 + 1));
        v110 = (__int64)(v108 + 5);
        v188 = *v108;
        v111 = *((_BYTE *)v108 + 24);
        v189 = v109;
        v190 = v111;
        v191 = v108[4];
        v192[0] = v193;
        v192[1] = (char *)0x400000000LL;
        if ( *((_DWORD *)v108 + 12) )
        {
          sub_28502F0((__int64)v192, (char **)v108 + 5, v36, v35, v110, a6);
          v110 = (__int64)(v108 + 5);
        }
        v112 = v108[11];
        v195 = _mm_loadu_si128((const __m128i *)v108 + 6);
        v194 = v112;
        *v108 = *v107;
        v108[1] = v107[1];
        *((_BYTE *)v108 + 16) = *((_BYTE *)v107 + 16);
        *((_BYTE *)v108 + 24) = *((_BYTE *)v107 + 24);
        v108[4] = v107[4];
        sub_28502F0(v110, (char **)v107 + 5, v36, v35, v110, a6);
        v108[11] = v107[11];
        v108[12] = v107[12];
        *((_BYTE *)v108 + 104) = *((_BYTE *)v107 + 104);
        *v107 = v188;
        v107[1] = v189.m128i_i64[0];
        *((_BYTE *)v107 + 16) = v189.m128i_i8[8];
        *((_BYTE *)v107 + 24) = v190;
        v107[4] = v191;
        sub_28502F0((__int64)(v107 + 5), v192, v113, v114, v115, v116);
        v88 = v192[0];
        v107[11] = v194;
        v107[12] = v195.m128i_i64[0];
        *((_BYTE *)v107 + 104) = v195.m128i_i8[8];
        if ( v88 != v193 )
        {
LABEL_138:
          _libc_free((unsigned __int64)v88);
          v85 = *(_DWORD *)(v173 + 768);
          if ( v85 != 1 )
            goto LABEL_137;
        }
        else
        {
LABEL_136:
          while ( 2 )
          {
            v85 = *(_DWORD *)(v173 + 768);
            if ( v85 != 1 )
            {
LABEL_137:
              v86 = (unsigned int)(v85 - 1);
              *(_DWORD *)(v173 + 768) = v86;
              v87 = *(_QWORD *)(v173 + 760) + 112 * v86;
              v88 = *(char **)(v87 + 40);
              if ( v88 == (char *)(v87 + 56) )
                continue;
              goto LABEL_138;
            }
            break;
          }
        }
        sub_2855860(v173, v163, a1 + 36280);
        v92 = *(_QWORD *)(v173 + 760);
        v93 = v186;
        v94 = *(__int64 **)(v92 + 40);
        v95 = &v94[*(unsigned int *)(v92 + 48)];
        if ( v94 != v95 )
        {
          while ( 2 )
          {
            while ( 1 )
            {
              v96 = *v94;
              if ( !v93 )
                break;
              v97 = v183;
              v90 = HIDWORD(v184);
              v89 = &v183[HIDWORD(v184)];
              if ( v183 == v89 )
              {
LABEL_155:
                if ( HIDWORD(v184) >= (unsigned int)v184 )
                  break;
                v90 = (unsigned int)(HIDWORD(v184) + 1);
                ++v94;
                ++HIDWORD(v184);
                *v89 = v96;
                v93 = v186;
                ++v182;
                if ( v95 == v94 )
                  goto LABEL_146;
              }
              else
              {
                while ( v96 != *v97 )
                {
                  if ( v89 == ++v97 )
                    goto LABEL_155;
                }
                if ( v95 == ++v94 )
                  goto LABEL_146;
              }
            }
            ++v94;
            sub_C8CC70((__int64)&v182, v96, (__int64)v89, v90, v91, a6);
            v93 = v186;
            if ( v95 == v94 )
              break;
            continue;
          }
        }
LABEL_146:
        v98 = *(_QWORD *)(v92 + 88);
        if ( v98 )
        {
          if ( v186 )
          {
            v99 = v183;
            v90 = HIDWORD(v184);
            v89 = &v183[HIDWORD(v184)];
            if ( v183 == v89 )
            {
LABEL_60:
              if ( HIDWORD(v184) >= (unsigned int)v184 )
                goto LABEL_61;
              ++HIDWORD(v184);
              *v89 = v98;
              ++v182;
            }
            else
            {
              while ( v98 != *v99 )
              {
                if ( v89 == ++v99 )
                  goto LABEL_60;
              }
            }
          }
          else
          {
LABEL_61:
            sub_C8CC70((__int64)&v182, v98, (__int64)v89, v90, v91, a6);
          }
        }
LABEL_62:
        ++v163;
        v161 += 2184;
        if ( v163 == v160 )
          goto LABEL_63;
        goto LABEL_38;
      }
    }
    v177 = (float)(*(float *)(v46 + 8) / sub_2850D10(v173, v38)) + v177;
    if ( *(_WORD *)(v38 + 24) == 8 )
      goto LABEL_78;
    goto LABEL_47;
  }
LABEL_63:
  sub_C7D6A0((__int64)v179, 16LL * v181, 8);
  if ( !v186 )
    _libc_free((unsigned __int64)v183);
}
