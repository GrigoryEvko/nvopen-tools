// Function: sub_2BD7F70
// Address: 0x2bd7f70
//
__int64 __fastcall sub_2BD7F70(
        __int64 a1,
        __int64 a2,
        unsigned __int64 i,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  __int64 *v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r12
  _QWORD *v11; // rax
  __int64 v12; // r9
  __int64 *v13; // r14
  int v14; // r11d
  __int64 v15; // rcx
  unsigned __int64 *v16; // rdx
  unsigned __int64 *v17; // rbx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r13
  unsigned int v20; // eax
  int v21; // edx
  unsigned __int64 v22; // r10
  __int64 v23; // rcx
  unsigned int v24; // eax
  unsigned __int64 *v25; // rdx
  __int64 v26; // r13
  unsigned __int64 *v27; // rax
  char v28; // al
  char v29; // bl
  __int64 *v30; // r14
  __int64 *v31; // r12
  char v32; // di
  _QWORD *v33; // rax
  char v34; // dl
  __int64 v35; // rax
  __int64 *v36; // r13
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rcx
  int v39; // eax
  __int64 *v40; // rdx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  int v43; // eax
  unsigned int v44; // eax
  int v45; // edi
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  unsigned __int64 *v48; // r10
  unsigned int v49; // r12d
  unsigned __int64 v50; // rcx
  int v51; // eax
  unsigned __int64 *v52; // r13
  int v53; // r14d
  unsigned __int64 *v54; // r12
  unsigned int v55; // eax
  __int64 v56; // r15
  unsigned int v57; // eax
  __int64 v58; // r12
  __int64 v59; // r14
  __int64 v60; // rsi
  int v61; // eax
  _QWORD *v62; // rax
  int v63; // eax
  unsigned __int8 v64; // al
  __int64 v65; // rax
  unsigned __int64 *v66; // rbx
  unsigned __int64 *v67; // r12
  unsigned __int64 v68; // rdi
  unsigned __int64 *v70; // rax
  unsigned int v71; // eax
  unsigned __int64 v72; // rdi
  int v73; // ecx
  int v74; // edi
  char v75; // al
  __int64 v76; // r13
  __int64 *v77; // rax
  __int64 v78; // r14
  __int64 v79; // r13
  int v80; // r10d
  __int64 v81; // rdi
  int v82; // r10d
  __int64 *v83; // rax
  __int64 v84; // rsi
  __int64 v85; // rdi
  __int64 v86; // rax
  __int64 v87; // r14
  __int64 v88; // rdi
  unsigned int v89; // eax
  unsigned __int8 *v90; // rsi
  __int64 v91; // r11
  __int64 v92; // r11
  __int64 v93; // rsi
  _QWORD *v94; // rax
  __int64 v95; // rdx
  __int64 *v96; // rsi
  __int64 v97; // rdi
  __int64 v98; // rax
  __int64 *v99; // r13
  __int64 *v100; // r12
  __int64 *v101; // rax
  int v102; // edx
  int v103; // edi
  __int64 *v104; // rsi
  int v105; // edi
  __int64 *v106; // rax
  __int64 v107; // rax
  __int64 v108; // rax
  _BYTE **v109; // r10
  int v110; // esi
  unsigned __int64 *v111; // rcx
  __int64 v112; // r15
  unsigned __int64 v113; // rdi
  __int64 v114; // r11
  __int64 v115; // r12
  unsigned int v116; // eax
  unsigned int v117; // eax
  unsigned __int64 *v118; // rax
  int v119; // esi
  int v120; // edi
  __int64 *v121; // rax
  int v122; // edx
  __int64 v123; // rax
  __int64 v124; // r13
  unsigned __int64 v125; // rdx
  int v126; // r13d
  _BYTE **v127; // r12
  unsigned __int8 *v128; // r11
  unsigned __int8 *v129; // rax
  __int64 v130; // rdi
  unsigned __int8 *v131; // rsi
  __int64 v132; // rax
  _QWORD *v133; // rax
  __int64 v134; // rdx
  __int64 *v135; // rsi
  __int64 v136; // rdi
  __int64 v137; // rax
  unsigned __int64 v138; // r14
  _BYTE *v139; // r13
  __int64 *v140; // rdx
  __int64 v141; // rax
  __int64 v142; // r13
  unsigned __int64 v143; // rdx
  int v144; // eax
  __int64 v145; // rax
  int v146; // edi
  int v147; // edi
  __int64 v148; // rax
  unsigned __int64 *v149; // rax
  int v150; // eax
  unsigned __int64 v151; // [rsp-10h] [rbp-340h]
  __int64 v152; // [rsp-8h] [rbp-338h]
  unsigned __int8 v153; // [rsp+8h] [rbp-328h]
  __int64 v154; // [rsp+10h] [rbp-320h]
  __int64 v156; // [rsp+30h] [rbp-300h]
  __int64 v157; // [rsp+30h] [rbp-300h]
  __int64 v158; // [rsp+38h] [rbp-2F8h]
  __int64 v159; // [rsp+38h] [rbp-2F8h]
  _BYTE **v160; // [rsp+40h] [rbp-2F0h]
  __int64 v161; // [rsp+40h] [rbp-2F0h]
  unsigned __int8 v162; // [rsp+48h] [rbp-2E8h]
  unsigned __int8 v163; // [rsp+50h] [rbp-2E0h]
  __int64 *v164; // [rsp+50h] [rbp-2E0h]
  __int64 v165; // [rsp+58h] [rbp-2D8h]
  __int64 v166; // [rsp+60h] [rbp-2D0h]
  __int64 *v167; // [rsp+70h] [rbp-2C0h]
  __int64 *v168; // [rsp+70h] [rbp-2C0h]
  __int64 v169; // [rsp+78h] [rbp-2B8h] BYREF
  __int64 v170; // [rsp+88h] [rbp-2A8h] BYREF
  _QWORD v171[2]; // [rsp+90h] [rbp-2A0h] BYREF
  _QWORD v172[2]; // [rsp+A0h] [rbp-290h] BYREF
  _QWORD v173[4]; // [rsp+B0h] [rbp-280h] BYREF
  __int64 v174; // [rsp+D0h] [rbp-260h] BYREF
  unsigned __int64 *v175; // [rsp+D8h] [rbp-258h]
  __int64 v176; // [rsp+E0h] [rbp-250h]
  unsigned int v177; // [rsp+E8h] [rbp-248h]
  _QWORD v178[6]; // [rsp+F0h] [rbp-240h] BYREF
  __int64 v179[4]; // [rsp+120h] [rbp-210h] BYREF
  char v180; // [rsp+140h] [rbp-1F0h]
  __int64 *v181; // [rsp+150h] [rbp-1E0h] BYREF
  unsigned int v182; // [rsp+158h] [rbp-1D8h]
  unsigned int v183; // [rsp+15Ch] [rbp-1D4h]
  _BYTE v184[32]; // [rsp+160h] [rbp-1D0h] BYREF
  unsigned __int64 *v185; // [rsp+180h] [rbp-1B0h] BYREF
  __int64 v186; // [rsp+188h] [rbp-1A8h]
  unsigned __int64 v187; // [rsp+190h] [rbp-1A0h] BYREF
  __int64 v188; // [rsp+198h] [rbp-198h]
  _BYTE *v189; // [rsp+1A0h] [rbp-190h] BYREF
  __int64 v190; // [rsp+1A8h] [rbp-188h]
  _BYTE v191[64]; // [rsp+1B0h] [rbp-180h] BYREF
  __int64 v192; // [rsp+1F0h] [rbp-140h] BYREF
  unsigned __int64 *v193; // [rsp+1F8h] [rbp-138h]
  __int64 v194; // [rsp+200h] [rbp-130h]
  __int64 v195; // [rsp+208h] [rbp-128h]
  _BYTE *v196; // [rsp+210h] [rbp-120h] BYREF
  __int64 v197; // [rsp+218h] [rbp-118h]
  _BYTE v198[64]; // [rsp+220h] [rbp-110h] BYREF
  __int64 v199; // [rsp+260h] [rbp-D0h] BYREF
  void *s; // [rsp+268h] [rbp-C8h]
  _BYTE v201[12]; // [rsp+270h] [rbp-C0h]
  char v202; // [rsp+27Ch] [rbp-B4h]
  char v203; // [rsp+280h] [rbp-B0h] BYREF

  v7 = &v192;
  v181 = (__int64 *)v184;
  s = &v203;
  v171[0] = a1;
  v169 = a2;
  v165 = i;
  v183 = 4;
  v199 = 0;
  *(_QWORD *)v201 = 16;
  *(_DWORD *)&v201[8] = 0;
  v202 = 1;
  v174 = 0;
  v175 = 0;
  v176 = 0;
  v177 = 0;
  v171[1] = &v174;
  v173[0] = &v174;
  v173[1] = a1;
  v173[2] = i;
  v163 = 0;
  while ( 2 )
  {
    v182 = 0;
    v8 = *(_QWORD *)(v169 + 56);
    v9 = v169 + 48;
    if ( v169 + 48 == v8 )
      goto LABEL_102;
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      if ( *(_BYTE *)(v8 - 24) != 84 || (*(_DWORD *)(v8 - 20) & 0x7FFFFFFu) > 0x80 )
        break;
      v10 = v8 - 24;
      if ( v202 )
      {
        v11 = s;
        i = (unsigned __int64)s + 8 * *(unsigned int *)&v201[4];
        if ( s != (void *)i )
        {
          while ( v10 != *v11 )
          {
            if ( (_QWORD *)i == ++v11 )
              goto LABEL_74;
          }
          goto LABEL_11;
        }
      }
      else
      {
        a2 = v8 - 24;
        if ( sub_C8CA60((__int64)&v199, v8 - 24) )
          goto LABEL_11;
      }
LABEL_74:
      a2 = *(_QWORD *)(v165 + 1984);
      v43 = *(_DWORD *)(v165 + 2000);
      if ( !v43 )
        goto LABEL_78;
      a4 = (unsigned int)(v43 - 1);
      v44 = a4 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      i = *(_QWORD *)(a2 + 8LL * v44);
      if ( v10 == i )
      {
LABEL_11:
        v8 = *(_QWORD *)(v8 + 8);
        if ( v9 == v8 )
          break;
      }
      else
      {
        v45 = 1;
        while ( i != -4096 )
        {
          a5 = (unsigned int)(v45 + 1);
          v44 = a4 & (v45 + v44);
          i = *(_QWORD *)(a2 + 8LL * v44);
          if ( v10 == i )
            goto LABEL_11;
          ++v45;
        }
LABEL_78:
        if ( !sub_2B08630(*(_QWORD *)(v8 - 16)) )
          goto LABEL_11;
        v46 = v182;
        a4 = v183;
        v47 = v182 + 1LL;
        if ( v47 > v183 )
        {
          a2 = (__int64)v184;
          sub_C8D5F0((__int64)&v181, v184, v47, 8u, a5, a6);
          v46 = v182;
        }
        i = (unsigned __int64)v181;
        v181[v46] = v10;
        ++v182;
        v8 = *(_QWORD *)(v8 + 8);
        if ( v9 == v8 )
          break;
      }
    }
    if ( v182 <= 1uLL )
    {
LABEL_102:
      v168 = &v192;
      goto LABEL_103;
    }
    v12 = (__int64)v181;
    v13 = v181;
    v167 = &v181[v182];
    do
    {
      while ( 1 )
      {
        v19 = *v13;
        if ( !v177 )
        {
          ++v174;
          goto LABEL_19;
        }
        v14 = 1;
        LODWORD(v15) = (v177 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v16 = 0;
        v17 = &v175[7 * (unsigned int)v15];
        v18 = *v17;
        if ( v19 != *v17 )
        {
          while ( v18 != -4096 )
          {
            if ( v18 == -8192 && !v16 )
              v16 = v17;
            a5 = (unsigned int)(v14 + 1);
            v15 = (v177 - 1) & ((_DWORD)v15 + v14);
            v12 = v15;
            v17 = &v175[7 * v15];
            v18 = *v17;
            if ( v19 == *v17 )
              goto LABEL_15;
            ++v14;
          }
          if ( v16 )
            v17 = v16;
          ++v174;
          v21 = v176 + 1;
          if ( 4 * ((int)v176 + 1) < 3 * v177 )
          {
            if ( v177 - HIDWORD(v176) - v21 <= v177 >> 3 )
            {
              sub_2B5BE90((__int64)&v174, v177);
              if ( !v177 )
              {
LABEL_425:
                LODWORD(v176) = v176 + 1;
                BUG();
              }
              v48 = 0;
              v49 = (v177 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
              v17 = &v175[7 * v49];
              v50 = *v17;
              v21 = v176 + 1;
              v51 = 1;
              if ( v19 != *v17 )
              {
                while ( v50 != -4096 )
                {
                  if ( v50 == -8192 && !v48 )
                    v48 = v17;
                  a5 = (unsigned int)(v51 + 1);
                  v148 = (v177 - 1) & (v49 + v51);
                  v49 = v148;
                  v17 = &v175[7 * v148];
                  v50 = *v17;
                  if ( v19 == *v17 )
                    goto LABEL_21;
                  v51 = a5;
                }
                if ( v48 )
                  v17 = v48;
              }
            }
            goto LABEL_21;
          }
LABEL_19:
          sub_2B5BE90((__int64)&v174, 2 * v177);
          if ( !v177 )
            goto LABEL_425;
          v20 = (v177 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v17 = &v175[7 * v20];
          v21 = v176 + 1;
          v22 = *v17;
          if ( v19 != *v17 )
          {
            v110 = 1;
            v111 = 0;
            while ( v22 != -4096 )
            {
              if ( !v111 && v22 == -8192 )
                v111 = v17;
              v12 = (unsigned int)(v110 + 1);
              v20 = (v177 - 1) & (v20 + v110);
              a5 = 7LL * v20;
              v17 = &v175[7 * v20];
              v22 = *v17;
              if ( v19 == *v17 )
                goto LABEL_21;
              ++v110;
            }
            if ( v111 )
              v17 = v111;
          }
LABEL_21:
          LODWORD(v176) = v21;
          if ( *v17 != -4096 )
            --HIDWORD(v176);
          *v17 = v19;
          v17[1] = (unsigned __int64)(v17 + 3);
          v17[2] = 0x400000000LL;
          break;
        }
LABEL_15:
        if ( !*((_DWORD *)v17 + 4) )
          break;
LABEL_16:
        if ( v167 == ++v13 )
          goto LABEL_36;
      }
      v187 = v19;
      v23 = 1;
      v186 = 0x400000001LL;
      v193 = (unsigned __int64 *)&v196;
      v24 = 1;
      v185 = &v187;
      v192 = 0;
      v194 = 4;
      LODWORD(v195) = 0;
      BYTE4(v195) = 1;
      do
      {
        v25 = v185;
        v26 = v185[v24 - 1];
        LODWORD(v186) = v24 - 1;
        if ( !(_BYTE)v23 )
          goto LABEL_47;
        v27 = v193;
        v25 = &v193[HIDWORD(v194)];
        if ( v193 != v25 )
        {
          while ( v26 != *v27 )
          {
            if ( v25 == ++v27 )
              goto LABEL_62;
          }
LABEL_30:
          v24 = v186;
          continue;
        }
LABEL_62:
        if ( HIDWORD(v194) < (unsigned int)v194 )
        {
          ++HIDWORD(v194);
          *v25 = v26;
          v23 = BYTE4(v195);
          ++v192;
        }
        else
        {
LABEL_47:
          sub_C8CC70((__int64)v7, v26, (__int64)v25, v23, a5, v12);
          v23 = BYTE4(v195);
          if ( !v34 )
            goto LABEL_30;
        }
        v35 = 4LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v26 + 7) & 0x40) != 0 )
        {
          v36 = *(__int64 **)(v26 - 8);
          v12 = (__int64)&v36[v35];
        }
        else
        {
          v12 = v26;
          v36 = (__int64 *)(v26 - v35 * 8);
        }
        if ( v36 == (__int64 *)v12 )
          goto LABEL_30;
        do
        {
          while ( 1 )
          {
            a5 = *v36;
            if ( *(_BYTE *)*v36 != 84 )
              break;
            v41 = (unsigned int)v186;
            v42 = (unsigned int)v186 + 1LL;
            if ( v42 > HIDWORD(v186) )
            {
              v156 = v12;
              v158 = *v36;
              sub_C8D5F0((__int64)&v185, &v187, v42, 8u, a5, v12);
              v41 = (unsigned int)v186;
              v12 = v156;
              a5 = v158;
            }
            v36 += 4;
            v185[v41] = a5;
            LODWORD(v186) = v186 + 1;
            if ( (__int64 *)v12 == v36 )
              goto LABEL_61;
          }
          v37 = *((unsigned int *)v17 + 4);
          v38 = *((unsigned int *)v17 + 5);
          v39 = *((_DWORD *)v17 + 4);
          if ( v37 >= v38 )
          {
            if ( v38 < v37 + 1 )
            {
              v157 = v12;
              v159 = *v36;
              sub_C8D5F0((__int64)(v17 + 1), v17 + 3, v37 + 1, 8u, a5, v12);
              v37 = *((unsigned int *)v17 + 4);
              v12 = v157;
              a5 = v159;
            }
            *(_QWORD *)(v17[1] + 8 * v37) = a5;
            ++*((_DWORD *)v17 + 4);
          }
          else
          {
            v40 = (__int64 *)(v17[1] + 8 * v37);
            if ( v40 )
            {
              *v40 = a5;
              v39 = *((_DWORD *)v17 + 4);
            }
            *((_DWORD *)v17 + 4) = v39 + 1;
          }
          v36 += 4;
        }
        while ( (__int64 *)v12 != v36 );
LABEL_61:
        v23 = BYTE4(v195);
        v24 = v186;
      }
      while ( v24 );
      if ( !(_BYTE)v23 )
        _libc_free((unsigned __int64)v193);
      if ( v185 == &v187 )
        goto LABEL_16;
      _libc_free((unsigned __int64)v185);
      ++v13;
    }
    while ( v167 != v13 );
LABEL_36:
    v192 = a1;
    a2 = (__int64)sub_2B6F180;
    v193 = (unsigned __int64 *)v165;
    v168 = v7;
    v28 = sub_2BC5990(
            (__int64)&v181,
            (__int64 (__fastcall *)(__int64, __int64, __int64))sub_2B6F180,
            (__int64)v171,
            (unsigned __int8 (__fastcall *)(__int64, unsigned __int8 *, unsigned __int8 *, __int64))sub_2B6ECA0,
            (__int64)v173,
            v165,
            (__int64 (__fastcall *)(__int64, __int64 *, __int64, __int64))sub_2BCF910,
            (__int64)v7);
    i = v151;
    a4 = v152;
    v29 = v28;
    v162 = v28 | v163;
    if ( !v28 )
    {
      v30 = v181;
      v31 = &v181[v182];
      if ( v31 == v181 )
        goto LABEL_103;
      break;
    }
    v52 = v175;
    v53 = v176;
    v54 = &v175[7 * v177];
    if ( (_DWORD)v176 && v175 != v54 )
    {
      v70 = v175;
      while ( 1 )
      {
        i = *v70;
        if ( *v70 != -8192 && i != -4096 )
          break;
        v70 += 7;
        if ( v54 == v70 )
          goto LABEL_100;
      }
      while ( v54 != v70 )
      {
        if ( *(_BYTE *)i != 84 )
          goto LABEL_151;
        v73 = *(_DWORD *)(v165 + 2000);
        a5 = *(_QWORD *)(v165 + 1984);
        if ( v73 )
        {
          a6 = (unsigned int)(v73 - 1);
          a4 = (unsigned int)a6 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          a2 = *(_QWORD *)(a5 + 8 * a4);
          if ( i == a2 )
          {
LABEL_151:
            if ( v54 != v70 )
            {
              i = 64;
              ++v174;
              v71 = 4 * v176;
              if ( (unsigned int)(4 * v176) < 0x40 )
                v71 = 64;
              if ( v177 > v71 )
              {
                v164 = v7;
                v112 = 56LL * v177;
                do
                {
                  if ( *v52 != -4096 && *v52 != -8192 )
                  {
                    v113 = v52[1];
                    if ( (unsigned __int64 *)v113 != v52 + 3 )
                      _libc_free(v113);
                  }
                  v52 += 7;
                }
                while ( v54 != v52 );
                v114 = v112;
                v115 = 64;
                v7 = v164;
                if ( v53 != 1 )
                {
                  _BitScanReverse(&v116, v53 - 1);
                  a4 = 33 - (v116 ^ 0x1F);
                  v115 = (unsigned int)(1 << (33 - (v116 ^ 0x1F)));
                  if ( (int)v115 < 64 )
                    v115 = 64;
                }
                if ( (_DWORD)v115 == v177 )
                {
                  v149 = v175;
                  v176 = 0;
                  i = (unsigned __int64)&v175[7 * v115];
                  do
                  {
                    if ( v149 )
                      *v149 = -4096;
                    v149 += 7;
                  }
                  while ( (unsigned __int64 *)i != v149 );
                }
                else
                {
                  a2 = v114;
                  sub_C7D6A0((__int64)v175, v114, 8);
                  v117 = sub_2B149A0(v115);
                  v177 = v117;
                  if ( v117 )
                  {
                    a2 = 8;
                    v118 = (unsigned __int64 *)sub_C7D670(56LL * v117, 8);
                    a4 = v177;
                    v176 = 0;
                    v175 = v118;
                    for ( i = (unsigned __int64)&v118[7 * v177]; (unsigned __int64 *)i != v118; v118 += 7 )
                    {
                      if ( v118 )
                        *v118 = -4096;
                    }
                  }
                  else
                  {
                    v175 = 0;
                    v176 = 0;
                  }
                }
              }
              else
              {
                do
                {
                  if ( *v52 != -4096 )
                  {
                    if ( *v52 != -8192 )
                    {
                      v72 = v52[1];
                      if ( (unsigned __int64 *)v72 != v52 + 3 )
                        _libc_free(v72);
                    }
                    *v52 = -4096;
                  }
                  v52 += 7;
                }
                while ( v54 != v52 );
                v176 = 0;
              }
            }
            break;
          }
          v74 = 1;
          while ( a2 != -4096 )
          {
            a4 = (unsigned int)a6 & (v74 + (_DWORD)a4);
            a2 = *(_QWORD *)(a5 + 8LL * (unsigned int)a4);
            if ( i == a2 )
              goto LABEL_151;
            ++v74;
          }
        }
        a4 = (__int64)(v70 + 7);
        if ( v54 == v70 + 7 )
          break;
        while ( 1 )
        {
          i = *(_QWORD *)a4;
          v70 = (unsigned __int64 *)a4;
          if ( *(_QWORD *)a4 != -8192 && i != -4096 )
            break;
          a4 += 56;
          if ( v54 == (unsigned __int64 *)a4 )
            goto LABEL_100;
        }
      }
    }
LABEL_100:
    v30 = v181;
    v31 = &v181[v182];
    if ( v31 == v181 )
    {
LABEL_46:
      v163 = 1;
      continue;
    }
    break;
  }
  v32 = v202;
  while ( 1 )
  {
LABEL_39:
    a2 = *v30;
    if ( !v32 )
    {
LABEL_68:
      ++v30;
      sub_C8CC70((__int64)&v199, a2, i, a4, a5, a6);
      v32 = v202;
      if ( v30 == v31 )
        goto LABEL_45;
      continue;
    }
    v33 = s;
    a4 = *(unsigned int *)&v201[4];
    i = (unsigned __int64)s + 8 * *(unsigned int *)&v201[4];
    if ( s != (void *)i )
      break;
LABEL_70:
    if ( *(_DWORD *)&v201[4] >= *(_DWORD *)v201 )
      goto LABEL_68;
    a4 = (unsigned int)(*(_DWORD *)&v201[4] + 1);
    ++v30;
    ++*(_DWORD *)&v201[4];
    *(_QWORD *)i = a2;
    v32 = v202;
    ++v199;
    if ( v30 == v31 )
      goto LABEL_45;
  }
  while ( a2 != *v33 )
  {
    if ( (_QWORD *)i == ++v33 )
      goto LABEL_70;
  }
  if ( ++v30 != v31 )
    goto LABEL_39;
LABEL_45:
  if ( v29 )
    goto LABEL_46;
  v163 = v162;
LABEL_103:
  ++v199;
  if ( v202 )
  {
LABEL_108:
    *(_QWORD *)&v201[4] = 0;
  }
  else
  {
    v55 = 4 * (*(_DWORD *)&v201[4] - *(_DWORD *)&v201[8]);
    if ( v55 < 0x20 )
      v55 = 32;
    if ( *(_DWORD *)v201 <= v55 )
    {
      memset(s, -1, 8LL * *(unsigned int *)v201);
      goto LABEL_108;
    }
    sub_C8C990((__int64)&v199, a2);
  }
  v185 = 0;
  v189 = v191;
  v190 = 0x800000000LL;
  v197 = 0x800000000LL;
  v178[1] = &v169;
  v186 = 0;
  v178[2] = v165;
  v187 = 0;
  v178[4] = v168;
  v172[0] = v168;
  v188 = 0;
  v192 = 0;
  v193 = 0;
  v194 = 0;
  v195 = 0;
  v196 = v198;
  v56 = *(_QWORD *)(v169 + 56);
  v178[0] = &v185;
  v178[3] = a1;
  v172[1] = &v185;
  v166 = v169 + 48;
  if ( v169 + 48 == v56 )
    goto LABEL_129;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( !v56 )
        BUG();
      v58 = v56 - 24;
      v59 = v56 - 24;
      if ( *(_BYTE *)(*(_QWORD *)(v56 - 16) + 8LL) == 18 )
        goto LABEL_112;
      v60 = *(_QWORD *)(v165 + 1984);
      v61 = *(_DWORD *)(v165 + 2000);
      if ( v61 )
      {
        a4 = (unsigned int)(v61 - 1);
        v57 = a4 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        i = *(_QWORD *)(v60 + 8LL * v57);
        if ( v58 == i )
          goto LABEL_112;
        v105 = 1;
        while ( i != -4096 )
        {
          a5 = (unsigned int)(v105 + 1);
          v57 = a4 & (v105 + v57);
          i = *(_QWORD *)(v60 + 8LL * v57);
          if ( v58 == i )
            goto LABEL_112;
          ++v105;
        }
      }
      if ( !v202 )
        goto LABEL_172;
      v62 = s;
      a4 = *(unsigned int *)&v201[4];
      i = (unsigned __int64)s + 8 * *(unsigned int *)&v201[4];
      if ( s != (void *)i )
      {
        while ( v58 != *v62 )
        {
          if ( (_QWORD *)i == ++v62 )
            goto LABEL_198;
        }
        goto LABEL_121;
      }
LABEL_198:
      if ( *(_DWORD *)&v201[4] >= *(_DWORD *)v201 )
      {
LABEL_172:
        sub_C8CC70((__int64)&v199, v56 - 24, i, a4, a5, a6);
        if ( (_BYTE)i )
        {
          v75 = *(_BYTE *)(v56 - 24);
          if ( v75 != 85 )
            goto LABEL_174;
LABEL_200:
          i = *(_QWORD *)(v56 - 56);
          if ( i )
          {
            if ( !*(_BYTE *)i && *(_QWORD *)(i + 24) == *(_QWORD *)(v56 + 56) && (*(_BYTE *)(i + 33) & 0x20) != 0 )
            {
              i = (unsigned int)(*(_DWORD *)(i + 36) - 68);
              if ( (unsigned int)i <= 3 )
                goto LABEL_112;
            }
          }
          if ( *(_QWORD *)(v56 - 8) )
            goto LABEL_204;
          goto LABEL_240;
        }
LABEL_121:
        if ( !*(_QWORD *)(v56 - 8) )
        {
          i = *(_QWORD *)(v56 - 16);
          v63 = *(unsigned __int8 *)(v56 - 24);
          if ( *(_BYTE *)(i + 8) == 7 || (_BYTE)v63 == 34 || (_BYTE)v63 == 85 )
          {
            v64 = sub_2BD7BC0((__int64)v178, (unsigned int)(v63 - 30) <= 0xA, a7, i, a4, a5, a6);
            if ( v64 )
              goto LABEL_126;
          }
        }
        goto LABEL_112;
      }
      a4 = (unsigned int)++*(_DWORD *)&v201[4];
      *(_QWORD *)i = v58;
      ++v199;
      v75 = *(_BYTE *)(v56 - 24);
      if ( v75 == 85 )
        goto LABEL_200;
LABEL_174:
      if ( v75 != 84 )
      {
        if ( *(_QWORD *)(v56 - 8) )
          goto LABEL_321;
        i = *(_QWORD *)(v56 - 16);
        if ( *(_BYTE *)(i + 8) != 7 && v75 != 34 )
          goto LABEL_321;
        if ( v75 != 62 )
          goto LABEL_240;
        LOBYTE(v126) = qword_5010188;
        v129 = sub_98ACB0(*(unsigned __int8 **)(v56 - 56), 6u);
        a4 = *(unsigned int *)(a1 + 96);
        v130 = *(_QWORD *)(a1 + 80);
        v131 = v129;
        if ( (_DWORD)a4 )
        {
          i = ((_DWORD)a4 - 1) & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
          v132 = v130 + 16 * i;
          a5 = *(_QWORD *)v132;
          if ( v131 == *(unsigned __int8 **)v132 )
          {
LABEL_315:
            i = v130 + 16LL * (unsigned int)a4;
            if ( v132 != i )
            {
              a4 = *(_QWORD *)(a1 + 104);
              i = a4 + 88LL * *(unsigned int *)(v132 + 8);
              if ( i != a4 + 88LL * *(unsigned int *)(a1 + 112) && *(_DWORD *)(i + 16) != 1 )
              {
LABEL_318:
                if ( !(_BYTE)v126 )
                {
LABEL_319:
                  v64 = v126
                      | sub_2BD7BC0(
                          (__int64)v178,
                          (unsigned int)*(unsigned __int8 *)(v56 - 24) - 30 <= 0xA,
                          a7,
                          i,
                          a4,
                          a5,
                          a6);
                  if ( v64 )
                    goto LABEL_126;
                  v75 = *(_BYTE *)(v56 - 24);
LABEL_321:
                  if ( v75 == 94 || v75 == 91 )
                  {
                    v170 = v58;
                    if ( !(_DWORD)v187 )
                    {
                      a4 = (unsigned int)v190;
                      v133 = v189;
                      v134 = 8LL * (unsigned int)v190;
                      v135 = (__int64 *)&v189[v134];
                      v136 = v134 >> 3;
                      i = v134 >> 5;
                      if ( i )
                      {
                        i = (unsigned __int64)&v189[32 * i];
                        while ( v58 != *v133 )
                        {
                          if ( v58 == v133[1] )
                          {
                            ++v133;
                            break;
                          }
                          if ( v58 == v133[2] )
                          {
                            v133 += 2;
                            break;
                          }
                          if ( v58 == v133[3] )
                          {
                            v133 += 3;
                            break;
                          }
                          v133 += 4;
                          if ( v133 == (_QWORD *)i )
                          {
                            v136 = v135 - v133;
                            goto LABEL_346;
                          }
                        }
LABEL_331:
                        if ( v135 != v133 )
                          goto LABEL_112;
                        goto LABEL_332;
                      }
LABEL_346:
                      if ( v136 != 2 )
                      {
                        if ( v136 != 3 )
                        {
                          if ( v136 != 1 )
                            goto LABEL_332;
                          goto LABEL_349;
                        }
                        if ( v58 == *v133 )
                          goto LABEL_331;
                        ++v133;
                      }
                      if ( v58 == *v133 )
                        goto LABEL_331;
                      ++v133;
LABEL_349:
                      if ( v58 == *v133 )
                        goto LABEL_331;
LABEL_332:
                      i = (unsigned int)v190 + 1LL;
                      if ( i > HIDWORD(v190) )
                      {
                        sub_C8D5F0((__int64)&v189, v191, i, 8u, a5, a6);
                        i = (unsigned int)v190;
                        v135 = (__int64 *)&v189[8 * (unsigned int)v190];
                      }
                      *v135 = v58;
                      v137 = (unsigned int)(v190 + 1);
                      LODWORD(v190) = v137;
                      if ( (unsigned int)v137 > 8 )
                      {
                        v138 = (unsigned __int64)v189;
                        v139 = &v189[8 * v137];
                        do
                        {
                          v140 = (__int64 *)v138;
                          v138 += 8LL;
                          sub_2400480((__int64)v179, (__int64)&v185, v140);
                        }
                        while ( v139 != (_BYTE *)v138 );
                      }
                      goto LABEL_112;
                    }
                    sub_2400480((__int64)v179, (__int64)&v185, &v170);
                    if ( v180 )
                    {
                      v141 = (unsigned int)v190;
                      a4 = HIDWORD(v190);
                      v142 = v170;
                      v143 = (unsigned int)v190 + 1LL;
                      if ( v143 > HIDWORD(v190) )
                      {
                        sub_C8D5F0((__int64)&v189, v191, v143, 8u, a5, a6);
                        v141 = (unsigned int)v190;
                      }
                      i = (unsigned __int64)v189;
                      *(_QWORD *)&v189[8 * v141] = v142;
                      LODWORD(v190) = v190 + 1;
                    }
LABEL_112:
                    v56 = *(_QWORD *)(v56 + 8);
                    if ( v166 == v56 )
                      goto LABEL_127;
                    continue;
                  }
LABEL_204:
                  if ( (unsigned __int8)(v75 - 82) > 1u )
                    goto LABEL_112;
                  a6 = (unsigned int)v194;
                  v170 = v58;
                  if ( !(_DWORD)v194 )
                  {
                    a4 = (unsigned int)v197;
                    v94 = v196;
                    v95 = 8LL * (unsigned int)v197;
                    v96 = (__int64 *)&v196[v95];
                    v97 = v95 >> 3;
                    i = v95 >> 5;
                    if ( !i )
                      goto LABEL_395;
                    i = (unsigned __int64)&v196[32 * i];
                    do
                    {
                      if ( v58 == *v94 )
                        goto LABEL_213;
                      if ( v58 == v94[1] )
                      {
                        ++v94;
                        goto LABEL_213;
                      }
                      if ( v58 == v94[2] )
                      {
                        v94 += 2;
                        goto LABEL_213;
                      }
                      if ( v58 == v94[3] )
                      {
                        v94 += 3;
                        goto LABEL_213;
                      }
                      v94 += 4;
                    }
                    while ( (_QWORD *)i != v94 );
                    v97 = v96 - v94;
LABEL_395:
                    if ( v97 == 2 )
                      goto LABEL_421;
                    if ( v97 != 3 )
                    {
                      if ( v97 == 1 )
                        goto LABEL_398;
                      goto LABEL_214;
                    }
                    if ( v58 == *v94 )
                      goto LABEL_213;
                    ++v94;
LABEL_421:
                    if ( v58 == *v94 )
                      goto LABEL_213;
                    ++v94;
LABEL_398:
                    if ( v58 == *v94 )
                    {
LABEL_213:
                      if ( v96 == v94 )
                        goto LABEL_214;
                      goto LABEL_112;
                    }
LABEL_214:
                    i = (unsigned int)v197 + 1LL;
                    if ( i > HIDWORD(v197) )
                    {
                      sub_C8D5F0((__int64)&v196, v198, i, 8u, a5, (unsigned int)v194);
                      i = (unsigned int)v197;
                      v96 = (__int64 *)&v196[8 * (unsigned int)v197];
                    }
                    *v96 = v58;
                    v98 = (unsigned int)(v197 + 1);
                    LODWORD(v197) = v98;
                    if ( (unsigned int)v98 <= 8 )
                      goto LABEL_112;
                    v99 = (__int64 *)v196;
                    v100 = (__int64 *)&v196[8 * v98];
                    while ( (_DWORD)v195 )
                    {
                      a6 = *v99;
                      i = ((_DWORD)v195 - 1) & (((unsigned int)*v99 >> 9) ^ ((unsigned int)*v99 >> 4));
                      v101 = (__int64 *)&v193[i];
                      a5 = *v101;
                      if ( *v101 != *v99 )
                      {
                        v146 = 1;
                        a4 = 0;
                        while ( a5 != -4096 )
                        {
                          if ( a5 == -8192 && !a4 )
                            a4 = (__int64)v101;
                          i = ((_DWORD)v195 - 1) & (unsigned int)(i + v146);
                          v101 = (__int64 *)&v193[i];
                          a5 = *v101;
                          if ( a6 == *v101 )
                            goto LABEL_219;
                          ++v146;
                        }
                        if ( a4 )
                          v101 = (__int64 *)a4;
                        ++v192;
                        v102 = v194 + 1;
                        if ( 4 * ((int)v194 + 1) < (unsigned int)(3 * v195) )
                        {
                          a4 = (unsigned int)(v195 - HIDWORD(v194) - v102);
                          if ( (unsigned int)a4 <= (unsigned int)v195 >> 3 )
                          {
                            sub_2BB7120((__int64)v168, v195);
                            if ( !(_DWORD)v195 )
                            {
LABEL_423:
                              LODWORD(v194) = v194 + 1;
                              BUG();
                            }
                            a6 = (unsigned int)(v195 - 1);
                            v104 = 0;
                            v102 = v194 + 1;
                            v147 = 1;
                            a4 = (unsigned int)a6 & (((unsigned int)*v99 >> 9) ^ ((unsigned int)*v99 >> 4));
                            v101 = (__int64 *)&v193[a4];
                            a5 = *v101;
                            if ( *v101 != *v99 )
                            {
                              while ( a5 != -4096 )
                              {
                                if ( !v104 && a5 == -8192 )
                                  v104 = v101;
                                a4 = (unsigned int)a6 & ((_DWORD)a4 + v147);
                                v101 = (__int64 *)&v193[a4];
                                a5 = *v101;
                                if ( *v99 == *v101 )
                                  goto LABEL_370;
                                ++v147;
                              }
LABEL_376:
                              if ( v104 )
                                v101 = v104;
                            }
                          }
LABEL_370:
                          LODWORD(v194) = v102;
                          if ( *v101 != -4096 )
                            --HIDWORD(v194);
                          i = *v99;
                          *v101 = *v99;
                          goto LABEL_219;
                        }
LABEL_222:
                        sub_2BB7120((__int64)v168, 2 * v195);
                        if ( !(_DWORD)v195 )
                          goto LABEL_423;
                        a6 = (unsigned int)(v195 - 1);
                        v102 = v194 + 1;
                        a4 = (unsigned int)a6 & (((unsigned int)*v99 >> 9) ^ ((unsigned int)*v99 >> 4));
                        v101 = (__int64 *)&v193[a4];
                        a5 = *v101;
                        if ( *v101 != *v99 )
                        {
                          v103 = 1;
                          v104 = 0;
                          while ( a5 != -4096 )
                          {
                            if ( !v104 && a5 == -8192 )
                              v104 = v101;
                            a4 = (unsigned int)a6 & ((_DWORD)a4 + v103);
                            v101 = (__int64 *)&v193[a4];
                            a5 = *v101;
                            if ( *v99 == *v101 )
                              goto LABEL_370;
                            ++v103;
                          }
                          goto LABEL_376;
                        }
                        goto LABEL_370;
                      }
LABEL_219:
                      if ( v100 == ++v99 )
                        goto LABEL_112;
                    }
                    ++v192;
                    goto LABEL_222;
                  }
                  v119 = v195;
                  if ( (_DWORD)v195 )
                  {
                    v120 = 1;
                    a4 = 0;
                    i = ((_DWORD)v195 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
                    v121 = (__int64 *)&v193[i];
                    a5 = *v121;
                    if ( v58 == *v121 )
                      goto LABEL_112;
                    while ( a5 != -4096 )
                    {
                      if ( !a4 && a5 == -8192 )
                        a4 = (__int64)v121;
                      i = ((_DWORD)v195 - 1) & (unsigned int)(i + v120);
                      v121 = (__int64 *)&v193[i];
                      a5 = *v121;
                      if ( v58 == *v121 )
                        goto LABEL_112;
                      ++v120;
                    }
                    v122 = v194 + 1;
                    if ( a4 )
                      v121 = (__int64 *)a4;
                    ++v192;
                    v179[0] = (__int64)v121;
                    if ( 4 * v122 < (unsigned int)(3 * v195) )
                    {
                      if ( (int)v195 - HIDWORD(v194) - v122 > (unsigned int)v195 >> 3 )
                      {
LABEL_289:
                        LODWORD(v194) = v122;
                        if ( *v121 != -4096 )
                          --HIDWORD(v194);
                        *v121 = v59;
                        v123 = (unsigned int)v197;
                        a4 = HIDWORD(v197);
                        v124 = v170;
                        v125 = (unsigned int)v197 + 1LL;
                        if ( v125 > HIDWORD(v197) )
                        {
                          sub_C8D5F0((__int64)&v196, v198, v125, 8u, a5, a6);
                          v123 = (unsigned int)v197;
                        }
                        i = (unsigned __int64)v196;
                        *(_QWORD *)&v196[8 * v123] = v124;
                        LODWORD(v197) = v197 + 1;
                        goto LABEL_112;
                      }
LABEL_296:
                      sub_2BB7120((__int64)v168, v119);
                      sub_2B49B10((__int64)v168, &v170, v179);
                      v59 = v170;
                      v122 = v194 + 1;
                      v121 = (__int64 *)v179[0];
                      goto LABEL_289;
                    }
                  }
                  else
                  {
                    ++v192;
                    v179[0] = 0;
                  }
                  v119 = 2 * v195;
                  goto LABEL_296;
                }
LABEL_240:
                v108 = 4LL * (*(_DWORD *)(v56 - 20) & 0x7FFFFFF);
                if ( (*(_BYTE *)(v56 - 17) & 0x40) != 0 )
                {
                  v109 = *(_BYTE ***)(v56 - 32);
                  v160 = &v109[v108];
                }
                else
                {
                  v160 = (_BYTE **)(v56 - 24);
                  v109 = (_BYTE **)(v58 - v108 * 8);
                }
                v126 = 0;
                if ( v109 != v160 )
                {
                  v127 = v109;
                  do
                  {
                    if ( **v127 > 0x1Cu && !(unsigned __int8)sub_2B1E780(v172, *v127, i, a4, (_BYTE *)a5) )
                      v126 |= sub_2BD7120(a1, 0, v128, v169, v165, a7);
                    v127 += 4;
                  }
                  while ( v160 != v127 );
                  v58 = v56 - 24;
                }
                goto LABEL_319;
              }
            }
          }
          else
          {
            v150 = 1;
            while ( a5 != -4096 )
            {
              a6 = (unsigned int)(v150 + 1);
              i = ((_DWORD)a4 - 1) & (unsigned int)(v150 + i);
              v132 = v130 + 16LL * (unsigned int)i;
              a5 = *(_QWORD *)v132;
              if ( v131 == *(unsigned __int8 **)v132 )
                goto LABEL_315;
              v150 = a6;
            }
          }
        }
        v107 = *(_QWORD *)(*(_QWORD *)(v56 - 88) + 16LL);
        if ( v107 && !*(_QWORD *)(v107 + 8) )
          goto LABEL_240;
        goto LABEL_318;
      }
      break;
    }
    v76 = *(_DWORD *)(v56 - 20) & 0x7FFFFFF;
    if ( (_DWORD)v76 != 2 )
      goto LABEL_187;
    v77 = *(__int64 **)(v56 - 32);
    a5 = *(_QWORD *)(a1 + 32);
    a4 = v169;
    i = 32LL * *(unsigned int *)(v56 + 48);
    v78 = *(_QWORD *)(a1 + 40);
    if ( v169 == *(__int64 *)((char *)v77 + i) )
    {
      v79 = *v77;
      if ( !*v77 )
        goto LABEL_253;
    }
    else
    {
      v79 = 0;
      if ( v169 != *(__int64 *)((char *)v77 + i + 8) )
        goto LABEL_178;
      v79 = v77[4];
      if ( !v79 )
        goto LABEL_253;
    }
    if ( *(_BYTE *)v79 <= 0x1Cu )
    {
      v79 = 0;
LABEL_178:
      v80 = *(_DWORD *)(a5 + 24);
      v81 = *(_QWORD *)(a5 + 8);
      if ( !v80 )
        goto LABEL_186;
      v82 = v80 - 1;
      i = v82 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v83 = (__int64 *)(v81 + 16 * i);
      v84 = *v83;
      if ( a4 != *v83 )
      {
        v144 = 1;
        while ( v84 != -4096 )
        {
          a5 = (unsigned int)(v144 + 1);
          v145 = v82 & (unsigned int)(i + v144);
          i = (unsigned int)v145;
          v83 = (__int64 *)(v81 + 16 * v145);
          v84 = *v83;
          if ( a4 == *v83 )
            goto LABEL_180;
          v144 = a5;
        }
LABEL_186:
        v76 = *(_DWORD *)(v56 - 20) & 0x7FFFFFF;
LABEL_187:
        if ( !v76 )
          goto LABEL_112;
        v87 = 0;
        while ( 1 )
        {
          a4 = *(_QWORD *)(v56 - 32);
          i = (unsigned int)v87;
          v92 = *(_QWORD *)(a4 + 32LL * *(unsigned int *)(v56 + 48) + 8LL * (unsigned int)v87);
          if ( v169 != v92 )
          {
            v93 = *(_QWORD *)(a1 + 40);
            if ( v92 )
            {
              v88 = (unsigned int)(*(_DWORD *)(v92 + 44) + 1);
              v89 = *(_DWORD *)(v92 + 44) + 1;
            }
            else
            {
              v88 = 0;
              v89 = 0;
            }
            if ( v89 < *(_DWORD *)(v93 + 32) )
            {
              if ( *(_QWORD *)(*(_QWORD *)(v93 + 24) + 8 * v88) )
              {
                i = 32LL * (unsigned int)v87;
                v90 = *(unsigned __int8 **)(a4 + i);
                if ( *v90 > 0x1Cu && !(unsigned __int8)sub_2B1E780(v172, v90, i, a4, (_BYTE *)a5) )
                {
                  v153 = sub_2BD7120(a1, 0, v90, v91, v165, a7);
                  if ( v153 )
                  {
                    v179[0] = v56 - 24;
                    v106 = sub_2B4B3F0(v165 + 1976, v179);
                    i = v153;
                    if ( v106 )
                    {
                      v163 = v153;
                      v56 = *(_QWORD *)(v169 + 56);
                      v166 = v169 + 48;
                      goto LABEL_112;
                    }
                    v163 = v153;
                  }
                }
              }
            }
          }
          if ( v76 == ++v87 )
            goto LABEL_112;
        }
      }
LABEL_180:
      v85 = v83[1];
      if ( !v85 )
        goto LABEL_186;
      v86 = sub_D47930(v85);
      if ( !v86 )
        goto LABEL_186;
      i = *(_QWORD *)(v56 - 32);
      a4 = 32LL * *(unsigned int *)(v56 + 48);
      if ( v86 == *(_QWORD *)(i + a4) )
      {
        v79 = *(_QWORD *)i;
        if ( !*(_QWORD *)i )
          goto LABEL_253;
      }
      else
      {
        if ( v86 != *(_QWORD *)(i + a4 + 8) )
        {
          if ( !v79 || *(_BYTE *)v79 <= 0x1Cu )
            goto LABEL_186;
          goto LABEL_245;
        }
        v79 = *(_QWORD *)(i + 32);
        if ( !v79 )
LABEL_253:
          BUG();
      }
      if ( *(_BYTE *)v79 <= 0x1Cu )
        goto LABEL_186;
LABEL_245:
      if ( !(unsigned __int8)sub_B19720(v78, *(_QWORD *)(v56 + 16), *(_QWORD *)(v79 + 40)) )
        goto LABEL_186;
      goto LABEL_246;
    }
    v154 = v169;
    v161 = *(_QWORD *)(a1 + 32);
    if ( !(unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 40), *(_QWORD *)(v56 + 16), *(_QWORD *)(v79 + 40)) )
    {
      a5 = v161;
      a4 = v154;
      goto LABEL_178;
    }
LABEL_246:
    v64 = sub_2BD7120(a1, (unsigned __int8 *)(v56 - 24), (unsigned __int8 *)v79, v169, v165, a7);
    if ( !v64 )
      goto LABEL_186;
LABEL_126:
    i = v169;
    v163 = v64;
    v166 = v169 + 48;
    v56 = *(_QWORD *)(*(_QWORD *)(v169 + 56) + 8LL);
    if ( v169 + 48 != v56 )
      continue;
    break;
  }
LABEL_127:
  if ( v196 != v198 )
    _libc_free((unsigned __int64)v196);
LABEL_129:
  sub_C7D6A0((__int64)v193, 8LL * (unsigned int)v195, 8);
  if ( v189 != v191 )
    _libc_free((unsigned __int64)v189);
  sub_C7D6A0(v186, 8LL * (unsigned int)v188, 8);
  v65 = v177;
  if ( v177 )
  {
    v66 = v175;
    v67 = &v175[7 * v177];
    do
    {
      if ( *v66 != -8192 && *v66 != -4096 )
      {
        v68 = v66[1];
        if ( (unsigned __int64 *)v68 != v66 + 3 )
          _libc_free(v68);
      }
      v66 += 7;
    }
    while ( v67 != v66 );
    v65 = v177;
  }
  sub_C7D6A0((__int64)v175, 56 * v65, 8);
  if ( !v202 )
    _libc_free((unsigned __int64)s);
  if ( v181 != (__int64 *)v184 )
    _libc_free((unsigned __int64)v181);
  return v163;
}
