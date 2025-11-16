// Function: sub_2E350C0
// Address: 0x2e350c0
//
__int64 __fastcall sub_2E350C0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // r13
  int v10; // eax
  __int64 *v11; // rbx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // rax
  __m128i *v23; // rax
  unsigned __int64 v24; // r13
  __int64 v25; // rbx
  _BYTE *v26; // r12
  __int64 v27; // rax
  _BYTE *v28; // r15
  _BYTE *v29; // rbx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  int v33; // r15d
  unsigned __int64 v34; // rax
  char *v35; // rdx
  char *v36; // rdi
  unsigned __int64 v37; // rcx
  __int64 v38; // rax
  __int64 v39; // rsi
  char *v40; // rax
  char *v41; // rsi
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  _BYTE *v44; // r15
  _QWORD *v45; // rax
  unsigned __int64 v47; // rax
  __int64 v48; // r9
  __int64 v49; // r8
  _BYTE **v50; // r10
  unsigned __int64 v51; // r15
  __int64 v52; // r12
  __int64 v53; // r13
  int v54; // ebx
  _DWORD *v55; // rax
  __int64 v56; // rdx
  _DWORD *v57; // rsi
  __int64 v58; // rdi
  __int64 v59; // rdx
  _DWORD *v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rax
  const __m128i *v65; // r12
  const __m128i *i; // rbx
  __m128i *v67; // rsi
  const __m128i *v68; // rdx
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // r12
  __int64 *v72; // rbx
  int v73; // r13d
  __int64 *v74; // r15
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // r15
  int v83; // ebx
  int v84; // r14d
  unsigned int v85; // edx
  __int64 *v86; // rax
  __int64 *v87; // rcx
  const void *v88; // r13
  __int64 v89; // rbx
  unsigned __int64 v90; // rax
  __int64 v91; // rdx
  int v92; // ecx
  __int64 v93; // rsi
  int v94; // ecx
  unsigned int v95; // edi
  __int64 *v96; // rax
  __int64 v97; // r9
  _QWORD *v98; // rdi
  unsigned int v99; // r8d
  __int64 *v100; // rax
  __int64 v101; // r10
  _QWORD *v102; // r8
  unsigned __int64 v103; // rax
  _BYTE *v104; // rsi
  __int64 v105; // rdi
  __int64 v106; // r13
  __int64 (*v107)(void); // rax
  unsigned __int64 *v108; // r13
  unsigned __int64 *v109; // r12
  unsigned __int64 v110; // rsi
  unsigned __int64 *v111; // r13
  unsigned __int64 *v112; // r12
  unsigned __int64 v113; // rsi
  unsigned __int64 v114; // rax
  unsigned __int64 **v115; // rdi
  __int64 v116; // rsi
  __int64 v117; // rcx
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // r13
  __int64 *v121; // rax
  __int64 *v122; // rax
  __int64 *v123; // rdx
  __int64 v124; // r8
  __int64 v125; // r9
  __int64 v126; // rcx
  __int64 j; // r13
  __int64 *v128; // rdx
  __int64 v129; // r8
  __int64 v130; // r9
  __int64 v131; // rax
  unsigned int v132; // ebx
  int v133; // r12d
  __int64 v134; // rdx
  __int64 v135; // r13
  int v136; // r9d
  unsigned __int64 v137; // rax
  unsigned int v138; // ecx
  __int64 v139; // r13
  __int64 *v140; // rdx
  __int64 *v141; // rax
  __int64 v142; // r8
  __int64 v143; // r9
  __int64 v144; // rdx
  __int64 v145; // rsi
  __int64 v146; // r9
  __int64 v147; // rdx
  __int64 v148; // r13
  __int64 v149; // r12
  __int64 v150; // r14
  unsigned int v151; // edx
  __int64 v152; // rdx
  __int64 *v153; // r13
  __int64 v154; // rax
  int v155; // eax
  int v156; // r8d
  int v157; // eax
  int v158; // r9d
  unsigned __int64 v159; // r10
  __int64 v160; // rcx
  __int64 v161; // rsi
  __int128 v162; // [rsp-20h] [rbp-250h]
  __int128 v163; // [rsp-20h] [rbp-250h]
  __int128 v164; // [rsp-20h] [rbp-250h]
  __int128 v165; // [rsp-20h] [rbp-250h]
  int v166; // [rsp+0h] [rbp-230h]
  unsigned __int64 v167; // [rsp+0h] [rbp-230h]
  __int64 v168; // [rsp+8h] [rbp-228h]
  unsigned int v169; // [rsp+8h] [rbp-228h]
  int v170; // [rsp+8h] [rbp-228h]
  __int64 v171; // [rsp+18h] [rbp-218h]
  __int64 v172; // [rsp+20h] [rbp-210h]
  __int64 v174; // [rsp+30h] [rbp-200h]
  __int64 v175; // [rsp+38h] [rbp-1F8h]
  __int64 v176; // [rsp+38h] [rbp-1F8h]
  __int64 v177; // [rsp+38h] [rbp-1F8h]
  __int64 v179; // [rsp+50h] [rbp-1E0h]
  unsigned int v180; // [rsp+50h] [rbp-1E0h]
  __int64 *v181; // [rsp+50h] [rbp-1E0h]
  __int64 v182; // [rsp+50h] [rbp-1E0h]
  __int64 v183; // [rsp+50h] [rbp-1E0h]
  __int64 *v184; // [rsp+50h] [rbp-1E0h]
  __int64 v185; // [rsp+50h] [rbp-1E0h]
  char v186; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v187; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v188; // [rsp+60h] [rbp-1D0h]
  __int64 v189; // [rsp+68h] [rbp-1C8h]
  __int64 v190; // [rsp+68h] [rbp-1C8h]
  __m128i *v191; // [rsp+70h] [rbp-1C0h]
  signed __int64 v192; // [rsp+70h] [rbp-1C0h]
  __int64 v195; // [rsp+88h] [rbp-1A8h]
  __int64 v196; // [rsp+90h] [rbp-1A0h]
  _BYTE **v197; // [rsp+90h] [rbp-1A0h]
  __int64 v199; // [rsp+A0h] [rbp-190h] BYREF
  __int64 v200; // [rsp+A8h] [rbp-188h] BYREF
  _BYTE *v201; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v202; // [rsp+B8h] [rbp-178h]
  _BYTE v203[16]; // [rsp+C0h] [rbp-170h] BYREF
  _BYTE *v204; // [rsp+D0h] [rbp-160h] BYREF
  __int64 v205; // [rsp+D8h] [rbp-158h]
  _BYTE v206[16]; // [rsp+E0h] [rbp-150h] BYREF
  void *v207; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v208; // [rsp+F8h] [rbp-138h]
  __int64 v209; // [rsp+100h] [rbp-130h]
  __int64 v210; // [rsp+108h] [rbp-128h]
  __int64 v211; // [rsp+110h] [rbp-120h]
  __int64 v212; // [rsp+118h] [rbp-118h]
  __int64 v213; // [rsp+120h] [rbp-110h]
  unsigned __int64 *v214; // [rsp+128h] [rbp-108h]
  __int64 v215; // [rsp+130h] [rbp-100h]
  _BYTE v216[24]; // [rsp+138h] [rbp-F8h] BYREF
  __int64 *v217; // [rsp+150h] [rbp-E0h] BYREF
  __int64 v218; // [rsp+158h] [rbp-D8h]
  __int64 v219[2]; // [rsp+160h] [rbp-D0h] BYREF
  __int64 v220; // [rsp+170h] [rbp-C0h]
  __int64 v221; // [rsp+178h] [rbp-B8h]
  __int64 v222; // [rsp+180h] [rbp-B0h]
  unsigned __int64 *v223; // [rsp+188h] [rbp-A8h] BYREF
  unsigned __int64 v224; // [rsp+190h] [rbp-A0h]
  _QWORD v225[2]; // [rsp+198h] [rbp-98h] BYREF
  __int64 v226; // [rsp+1A8h] [rbp-88h]

  v5 = 0;
  v186 = sub_2E325A0(a1, a2);
  if ( v186 )
  {
    v6 = *(_QWORD *)(a1 + 32);
    v7 = *(_QWORD *)(a1 + 8);
    LOBYTE(v218) = 0;
    v8 = v6 + 320;
    v179 = v6;
    v9 = v6;
    if ( v6 + 320 != v7 )
      v5 = v7;
    v189 = v5;
    v5 = sub_2E7AAE0(v6, 0, v217, (unsigned int)v218);
    *(_DWORD *)(v5 + 28) = *(_DWORD *)(a2 + 28);
    v10 = sub_2E31540(a1);
    if ( v10 >= 0 )
      sub_2E79D70(*(_QWORD *)(v9 + 64), (unsigned int)v10, a2, v5);
    else
      v186 = 0;
    v11 = *(__int64 **)(a1 + 8);
    sub_2E33BD0(v8, v5);
    v14 = *(_QWORD *)v5;
    v15 = *v11;
    *(_QWORD *)(v5 + 8) = v11;
    v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v5 = v16 | v14 & 7;
    *(_QWORD *)(v16 + 8) = v5;
    *v11 = v5 | *v11 & 7;
    v195 = *a3;
    v17 = *a3;
    if ( *a3 )
    {
      sub_2E34D50(*(_QWORD *)(*a3 + 32), (__int64 *)v5, v16, *a3, v12, v13);
      v20 = v17;
      v21 = *(unsigned int *)(v17 + 192);
      v22 = *(unsigned int *)(v20 + 352);
      if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 356) )
      {
        sub_C8D5F0(v195 + 344, (const void *)(v195 + 360), v22 + 1, 8u, v18, v19);
        v22 = *(unsigned int *)(v195 + 352);
      }
      *(_QWORD *)(*(_QWORD *)(v195 + 344) + 8 * v22) = v21;
      ++*(_DWORD *)(v195 + 352);
    }
    else
    {
      v105 = a3[1];
      if ( v105 )
        sub_2E34D50(v105, (__int64 *)v5, v16, 0, v12, v13);
    }
    v23 = (__m128i *)a3[2];
    v201 = v203;
    v202 = 0x400000000LL;
    v191 = v23;
    if ( v23 )
    {
      v24 = sub_2E318E0(a1);
      if ( v24 != a1 + 48 )
      {
        while ( 1 )
        {
          v25 = *(_QWORD *)(v24 + 32);
          v26 = (_BYTE *)(v25 + 40LL * (*(_DWORD *)(v24 + 40) & 0xFFFFFF));
          v27 = 5LL * (unsigned int)sub_2E88FE0(v24);
          if ( v26 != (_BYTE *)(v25 + 8 * v27) )
            break;
LABEL_49:
          v24 = *(_QWORD *)(v24 + 8);
          if ( v24 == a1 + 48 )
            goto LABEL_50;
        }
        v28 = (_BYTE *)(v25 + 8 * v27);
        while ( 1 )
        {
          v29 = v28;
          if ( (unsigned __int8)sub_2E2FA70(v28) )
            break;
          v28 += 40;
          if ( v26 == v28 )
            goto LABEL_49;
        }
        while ( 1 )
        {
          if ( v26 == v29 )
            goto LABEL_49;
          v33 = *((_DWORD *)v29 + 2);
          if ( v33 )
          {
            if ( (((v29[3] & 0x40) != 0) & ((v29[3] >> 4) ^ 1)) != 0 && (v29[4] & 1) == 0 )
              break;
          }
LABEL_35:
          v44 = v29 + 40;
          if ( v29 + 40 == v26 )
            goto LABEL_49;
          while ( 1 )
          {
            v29 = v44;
            if ( (unsigned __int8)sub_2E2FA70(v44) )
              break;
            v44 += 40;
            if ( v26 == v44 )
              goto LABEL_49;
          }
        }
        if ( (unsigned int)(v33 - 1) <= 0x3FFFFFFE )
        {
LABEL_32:
          v42 = (unsigned int)v202;
          v43 = (unsigned int)v202 + 1LL;
          if ( v43 > HIDWORD(v202) )
          {
            sub_C8D5F0((__int64)&v201, v203, v43, 4u, v31, v32);
            v42 = (unsigned int)v202;
          }
          *(_DWORD *)&v201[4 * v42] = v33;
          LODWORD(v202) = v202 + 1;
          v29[3] &= ~0x40u;
          goto LABEL_35;
        }
        v34 = sub_2E29D60(v191, v33, (v29[3] >> 4) ^ 1u, v30, v31, v32);
        v35 = *(char **)(v34 + 40);
        v36 = *(char **)(v34 + 32);
        v37 = v34;
        v38 = (v35 - v36) >> 5;
        v39 = (v35 - v36) >> 3;
        if ( v38 > 0 )
        {
          v40 = &v36[32 * v38];
          while ( *(_QWORD *)v36 != v24 )
          {
            if ( *((_QWORD *)v36 + 1) == v24 )
            {
              v36 += 8;
              goto LABEL_28;
            }
            if ( *((_QWORD *)v36 + 2) == v24 )
            {
              v36 += 16;
              goto LABEL_28;
            }
            if ( *((_QWORD *)v36 + 3) == v24 )
            {
              v36 += 24;
              goto LABEL_28;
            }
            v36 += 32;
            if ( v40 == v36 )
            {
              v39 = (v35 - v36) >> 3;
              goto LABEL_133;
            }
          }
          goto LABEL_28;
        }
LABEL_133:
        if ( v39 != 2 )
        {
          if ( v39 != 3 )
          {
            if ( v39 != 1 )
              goto LABEL_35;
LABEL_136:
            if ( *(_QWORD *)v36 != v24 )
              goto LABEL_35;
            goto LABEL_28;
          }
          if ( *(_QWORD *)v36 == v24 )
          {
LABEL_28:
            if ( v36 == v35 )
              goto LABEL_35;
            v41 = v36 + 8;
            if ( v35 != v36 + 8 )
            {
              v188 = v37;
              memmove(v36, v41, v35 - v41);
              v37 = v188;
              v41 = *(char **)(v188 + 40);
            }
            *(_QWORD *)(v37 + 40) = v41 - 8;
            goto LABEL_32;
          }
          v36 += 8;
        }
        if ( *(_QWORD *)v36 != v24 )
        {
          v36 += 8;
          goto LABEL_136;
        }
        goto LABEL_28;
      }
    }
LABEL_50:
    v204 = v206;
    v205 = 0x400000000LL;
    if ( v195 )
    {
      v47 = sub_2E318E0(a1);
      v48 = a1 + 48;
      if ( v47 != a1 + 48 )
      {
        v49 = v5;
        v50 = &v204;
        v51 = v47;
        do
        {
          v52 = *(_QWORD *)(v51 + 32);
          v53 = v52 + 40LL * (*(_DWORD *)(v51 + 40) & 0xFFFFFF);
          if ( v53 != v52 )
          {
            while ( 1 )
            {
              if ( *(_BYTE *)v52 )
                goto LABEL_64;
              v54 = *(_DWORD *)(v52 + 8);
              if ( !v54 )
                goto LABEL_64;
              v55 = v204;
              v56 = 4LL * (unsigned int)v205;
              v57 = &v204[v56];
              v58 = v56 >> 2;
              v59 = v56 >> 4;
              if ( v59 )
              {
                v60 = &v204[16 * v59];
                while ( v54 != *v55 )
                {
                  if ( v54 == v55[1] )
                  {
                    ++v55;
                    goto LABEL_63;
                  }
                  if ( v54 == v55[2] )
                  {
                    v55 += 2;
                    goto LABEL_63;
                  }
                  if ( v54 == v55[3] )
                  {
                    v55 += 3;
                    goto LABEL_63;
                  }
                  v55 += 4;
                  if ( v60 == v55 )
                  {
                    v58 = v57 - v55;
                    goto LABEL_118;
                  }
                }
                goto LABEL_63;
              }
LABEL_118:
              if ( v58 == 2 )
                goto LABEL_130;
              if ( v58 != 3 )
              {
                if ( v58 != 1 )
                  goto LABEL_122;
LABEL_121:
                if ( v54 != *v55 )
                {
LABEL_122:
                  if ( (unsigned __int64)(unsigned int)v205 + 1 > HIDWORD(v205) )
                  {
                    v177 = v49;
                    v197 = v50;
                    sub_C8D5F0((__int64)v50, v206, (unsigned int)v205 + 1LL, 4u, v49, v48);
                    v49 = v177;
                    v50 = v197;
                    v57 = &v204[4 * (unsigned int)v205];
                  }
                  *v57 = v54;
                  LODWORD(v205) = v205 + 1;
                  goto LABEL_64;
                }
                goto LABEL_63;
              }
              if ( v54 != *v55 )
                break;
LABEL_63:
              if ( v57 == v55 )
                goto LABEL_122;
LABEL_64:
              v52 += 40;
              if ( v53 == v52 )
                goto LABEL_65;
            }
            ++v55;
LABEL_130:
            if ( v54 != *v55 )
            {
              ++v55;
              goto LABEL_121;
            }
            goto LABEL_63;
          }
LABEL_65:
          v51 = *(_QWORD *)(v51 + 8);
        }
        while ( a1 + 48 != v51 );
        v5 = v49;
      }
    }
    sub_2E337A0(a1, a2, v5);
    v64 = v189;
    if ( a2 == v189 )
      v64 = v5;
    v175 = a3[1];
    if ( !v186 )
    {
      v219[1] = 0;
      v218 = v179;
      v217 = (__int64 *)&unk_4A288B0;
      v219[0] = v175;
      v220 = 0;
      v221 = 0;
      v222 = 0;
      v223 = v225;
      v224 = 0x200000000LL;
      *(_QWORD *)(v179 + 672) = &v217;
      sub_2E32A60(a1, v64);
      v217 = (__int64 *)&unk_4A288B0;
      *(_QWORD *)(v218 + 672) = 0;
      v111 = v223;
      v112 = &v223[(unsigned int)v224];
      if ( v223 != v112 )
      {
        do
        {
          v113 = *v111++;
          sub_2E192D0(v219[0], v113, 0);
        }
        while ( v112 != v111 );
        v112 = v223;
      }
      if ( v112 != v225 )
        _libc_free((unsigned __int64)v112);
      sub_C7D6A0(v220, 8LL * (unsigned int)v222, 8);
    }
    sub_2E33F80(v5, a2, -1, v61, v62, v63);
    if ( !sub_2E322F0(v5, a2) )
    {
      v106 = 0;
      v210 = 0;
      v208 = v179;
      v207 = &unk_4A288B0;
      v211 = 0;
      v209 = v175;
      v212 = 0;
      v213 = 0;
      v214 = (unsigned __int64 *)v216;
      v215 = 0x200000000LL;
      *(_QWORD *)(v179 + 672) = &v207;
      v217 = v219;
      v218 = 0x400000000LL;
      v107 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a1 + 32) + 16LL) + 128LL);
      if ( v107 != sub_2DAC790 )
        v106 = v107();
      v199 = 0;
      sub_2E32880(&v200, a1);
      if ( v200 && ((unsigned int)sub_B10CE0((__int64)&v200) || (unsigned int)sub_B10CF0((__int64)&v200)) )
      {
        v199 = v200;
        if ( v200 )
          sub_B96E90((__int64)&v199, v200, 1);
      }
      (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64 *, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v106 + 368LL))(
        v106,
        v5,
        a2,
        0,
        v217,
        (unsigned int)v218,
        &v199,
        0);
      if ( v200 )
        sub_B91220((__int64)&v200, v200);
      if ( v199 )
        sub_B91220((__int64)&v199, v199);
      if ( v217 != v219 )
        _libc_free((unsigned __int64)v217);
      v207 = &unk_4A288B0;
      *(_QWORD *)(v208 + 672) = 0;
      v108 = v214;
      v109 = &v214[(unsigned int)v215];
      if ( v214 != v109 )
      {
        do
        {
          v110 = *v108++;
          sub_2E192D0(v209, v110, 0);
        }
        while ( v109 != v108 );
        v109 = v214;
      }
      if ( v109 != (unsigned __int64 *)v216 )
        _libc_free((unsigned __int64)v109);
      sub_C7D6A0(v211, 8LL * (unsigned int)v213, 8);
    }
    sub_2E32770(a2, a1, v5);
    v65 = *(const __m128i **)(a2 + 192);
    for ( i = (const __m128i *)sub_2E33140(a2); v65 != i; *(_QWORD *)(v5 + 192) = (char *)v67 + 24 )
    {
      while ( 1 )
      {
        v67 = *(__m128i **)(v5 + 192);
        if ( v67 != *(__m128i **)(v5 + 200) )
          break;
        v68 = i;
        i = (const __m128i *)((char *)i + 24);
        sub_2E33890((unsigned __int64 *)(v5 + 184), v67, v68);
        if ( v65 == i )
          goto LABEL_78;
      }
      if ( v67 )
      {
        *v67 = _mm_loadu_si128(i);
        v67[1].m128i_i64[0] = i[1].m128i_i64[0];
        v67 = *(__m128i **)(v5 + 192);
      }
      i = (const __m128i *)((char *)i + 24);
    }
LABEL_78:
    v71 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v179 + 16) + 200LL))(*(_QWORD *)(v179 + 16));
    if ( v191 )
    {
LABEL_84:
      while ( (_DWORD)v202 )
      {
        v72 = (__int64 *)(a1 + 48);
        v73 = *(_DWORD *)&v201[4 * (unsigned int)v202 - 4];
        LODWORD(v202) = v202 - 1;
        v74 = *(__int64 **)(a1 + 56);
        while ( v72 != v74 )
        {
          v72 = (__int64 *)(*v72 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (unsigned __int8)sub_2E8F280(v72, (unsigned int)v73, v71, 0) )
          {
            if ( v73 < 0 )
            {
              v103 = sub_2E29D60(v191, v73, v75, v76, v77, v70);
              v217 = v72;
              v104 = *(_BYTE **)(v103 + 40);
              if ( v104 == *(_BYTE **)(v103 + 48) )
              {
                sub_2E26050(v103 + 32, v104, &v217);
              }
              else
              {
                if ( v104 )
                {
                  *(_QWORD *)v104 = v72;
                  v104 = *(_BYTE **)(v103 + 40);
                }
                *(_QWORD *)(v103 + 40) = v104 + 8;
              }
            }
            goto LABEL_84;
          }
        }
      }
      if ( a4 )
        sub_2E2C870(v191, v5, a1, a2, a4, v70);
      else
        sub_2E2BE10(v191, v5, a1, a2, 0, v70);
    }
    if ( v195 )
    {
      v78 = *(_QWORD *)(a1 + 32);
      v174 = v78 + 320;
      v172 = *(_QWORD *)(v5 + 8);
      v79 = *(_QWORD *)(v175 + 152);
      v80 = *(_QWORD *)(v79 + 16LL * *(unsigned int *)(a1 + 24) + 8);
      v196 = v80;
      if ( ((v80 >> 1) & 3) != 0 )
        v192 = v80 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v80 >> 1) & 3) - 1));
      else
        v192 = *(_QWORD *)(v80 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
      v81 = *(_QWORD *)(v79 + 16LL * *(unsigned int *)(v5 + 24) + 8);
      LODWORD(v223) = 0;
      v224 = 0;
      v190 = v81;
      v217 = v219;
      v218 = 0x800000000LL;
      v225[0] = &v223;
      v225[1] = &v223;
      v226 = 0;
      if ( a2 + 48 != *(_QWORD *)(a2 + 56) )
      {
        v82 = *(_QWORD *)(a2 + 56);
        do
        {
          if ( *(_WORD *)(v82 + 68) && *(_WORD *)(v82 + 68) != 68 )
            break;
          v132 = 1;
          v133 = *(_DWORD *)(v82 + 40) & 0xFFFFFF;
          if ( v133 != 1 )
          {
            do
            {
              while ( 1 )
              {
                v134 = *(_QWORD *)(v82 + 32);
                if ( v5 == *(_QWORD *)(v134 + 40LL * (v132 + 1) + 24) )
                {
                  v135 = v134 + 40LL * v132;
                  LODWORD(v200) = *(_DWORD *)(v135 + 8);
                  sub_2E34820((__int64)&v207, (__int64)&v217, (unsigned int *)&v200, v79, v69);
                  if ( (*(_BYTE *)(v135 + 4) & 1) == 0 )
                  {
                    v136 = v200;
                    v137 = *(unsigned int *)(v195 + 160);
                    v138 = v200 & 0x7FFFFFFF;
                    v139 = 8 * (v200 & 0x7FFFFFFF);
                    if ( ((unsigned int)v200 & 0x7FFFFFFF) >= (unsigned int)v137
                      || (v140 = *(__int64 **)(*(_QWORD *)(v195 + 152) + 8LL * v138)) == 0 )
                    {
                      v151 = v138 + 1;
                      if ( (unsigned int)v137 >= v138 + 1 || v151 == v137 )
                      {
                        v152 = *(_QWORD *)(v195 + 152);
                      }
                      else if ( v151 >= v137 )
                      {
                        v159 = v151 - v137;
                        v185 = *(_QWORD *)(v195 + 168);
                        if ( v151 > (unsigned __int64)*(unsigned int *)(v195 + 164) )
                        {
                          v167 = v151 - v137;
                          v170 = v200;
                          sub_C8D5F0(v195 + 152, (const void *)(v195 + 168), v151, 8u, v195 + 168, (unsigned int)v200);
                          v159 = v167;
                          v136 = v170;
                          v137 = *(unsigned int *)(v195 + 160);
                        }
                        v152 = *(_QWORD *)(v195 + 152);
                        v160 = v152 + 8 * v137;
                        v161 = v160 + 8 * v159;
                        if ( v160 != v161 )
                        {
                          do
                          {
                            v160 += 8;
                            *(_QWORD *)(v160 - 8) = v185;
                          }
                          while ( v161 != v160 );
                          LODWORD(v137) = *(_DWORD *)(v195 + 160);
                          v152 = *(_QWORD *)(v195 + 152);
                        }
                        *(_DWORD *)(v195 + 160) = v159 + v137;
                      }
                      else
                      {
                        *(_DWORD *)(v195 + 160) = v151;
                        v152 = *(_QWORD *)(v195 + 152);
                      }
                      v153 = (__int64 *)(v152 + v139);
                      v154 = sub_2E10F30(v136);
                      *v153 = v154;
                      v184 = (__int64 *)v154;
                      sub_2E11E80((_QWORD *)v195, v154);
                      v140 = v184;
                    }
                    v181 = v140;
                    v141 = (__int64 *)sub_2E09D00(v140, v192);
                    v144 = (__int64)v181;
                    v182 = 0;
                    if ( v141 != (__int64 *)(*(_QWORD *)v144 + 24LL * *(unsigned int *)(v144 + 8))
                      && (*(_DWORD *)((*v141 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v141 >> 1) & 3) <= (*(_DWORD *)((v192 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v192 >> 1) & 3) )
                    {
                      v182 = v141[2];
                    }
                    v145 = v182;
                    v168 = v144;
                    v208 = v190;
                    v209 = v182;
                    *((_QWORD *)&v164 + 1) = v190;
                    *(_QWORD *)&v164 = v196;
                    v207 = (void *)v196;
                    sub_2E0F080(v144, v182, v144, v190, v142, v143, v164, v182);
                    v147 = v168;
                    v148 = *(_QWORD *)(v168 + 104);
                    if ( v148 )
                      break;
                  }
                }
                v132 += 2;
                if ( v132 == v133 )
                  goto LABEL_215;
              }
              v169 = v132;
              v166 = v133;
              v149 = v182;
              v183 = v5;
              v150 = v148;
              do
              {
                v208 = v190;
                v209 = v149;
                *((_QWORD *)&v165 + 1) = v190;
                *(_QWORD *)&v165 = v196;
                v207 = (void *)v196;
                sub_2E0F080(v150, v145, v147, v79, v69, v146, v165, v149);
                v150 = *(_QWORD *)(v150 + 104);
              }
              while ( v150 );
              v133 = v166;
              v5 = v183;
              v132 += 2;
            }
            while ( v169 + 2 != v166 );
          }
LABEL_215:
          v82 = *(_QWORD *)(v82 + 8);
        }
        while ( a2 + 48 != v82 );
        v78 = *(_QWORD *)(a1 + 32);
      }
      v83 = *(_DWORD *)(*(_QWORD *)(v78 + 32) + 64LL);
      if ( v83 )
      {
        v176 = v5;
        v187 = v192 & 0xFFFFFFFFFFFFFFF8LL;
        v84 = 0;
        v180 = (v192 >> 1) & 3;
        while ( 1 )
        {
          v85 = v84 | 0x80000000;
          if ( v226 )
          {
            v114 = v224;
            if ( v224 )
            {
              v115 = &v223;
              do
              {
                while ( 1 )
                {
                  v116 = *(_QWORD *)(v114 + 16);
                  v117 = *(_QWORD *)(v114 + 24);
                  if ( v85 <= *(_DWORD *)(v114 + 32) )
                    break;
                  v114 = *(_QWORD *)(v114 + 24);
                  if ( !v117 )
                    goto LABEL_182;
                }
                v115 = (unsigned __int64 **)v114;
                v114 = *(_QWORD *)(v114 + 16);
              }
              while ( v116 );
LABEL_182:
              if ( v115 != &v223 && v85 >= *((_DWORD *)v115 + 8) )
                goto LABEL_103;
            }
          }
          else
          {
            v86 = v217;
            v87 = (__int64 *)((char *)v217 + 4 * (unsigned int)v218);
            if ( v217 != v87 )
            {
              while ( v85 != *(_DWORD *)v86 )
              {
                v86 = (__int64 *)((char *)v86 + 4);
                if ( v87 == v86 )
                  goto LABEL_184;
              }
              if ( v87 != v86 )
                goto LABEL_103;
            }
          }
LABEL_184:
          v118 = v84 & 0x7FFFFFFF;
          if ( *(_DWORD *)(v195 + 160) > (unsigned int)v118 )
          {
            v119 = *(_QWORD *)(v195 + 152);
            v120 = *(_QWORD *)(v119 + 8 * v118);
            if ( v120 )
            {
              v121 = (__int64 *)sub_2E09D00(*(__int64 **)(v119 + 8 * v118), v192);
              if ( v121 != (__int64 *)(*(_QWORD *)v120 + 24LL * *(unsigned int *)(v120 + 8))
                && (*(_DWORD *)((*v121 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v121 >> 1) & 3) <= (*(_DWORD *)(v187 + 24) | v180) )
              {
                v171 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v195 + 32) + 152LL) + 16LL * *(unsigned int *)(a2 + 24));
                v122 = (__int64 *)sub_2E09D00((__int64 *)v120, v171);
                if ( v122 == (__int64 *)(*(_QWORD *)v120 + 24LL * *(unsigned int *)(v120 + 8))
                  || (*(_DWORD *)((*v122 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v122 >> 1) & 3)) > (*(_DWORD *)((v171 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v171 >> 1) & 3) )
                {
                  if ( v174 != v172 )
                  {
                    do
                    {
                      sub_2E0C3B0(v120, v196, v190, 0);
                      v120 = *(_QWORD *)(v120 + 104);
                    }
                    while ( v120 );
                  }
                }
                else if ( v174 == v172 )
                {
                  v123 = (__int64 *)sub_2E09D00((__int64 *)v120, v192);
                  v126 = 0;
                  if ( v123 != (__int64 *)(*(_QWORD *)v120 + 24LL * *(unsigned int *)(v120 + 8))
                    && (*(_DWORD *)((*v123 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v123 >> 1) & 3) <= (*(_DWORD *)(v187 + 24) | v180) )
                  {
                    v126 = v123[2];
                  }
                  v209 = v126;
                  v208 = v190;
                  *((_QWORD *)&v162 + 1) = v190;
                  *(_QWORD *)&v162 = v196;
                  v207 = (void *)v196;
                  sub_2E0F080(v120, v190, (__int64)v123, v126, v124, v125, v162, v126);
                  for ( j = *(_QWORD *)(v120 + 104); j; j = *(_QWORD *)(j + 104) )
                  {
                    v128 = (__int64 *)sub_2E09D00((__int64 *)j, v192);
                    if ( v128 != (__int64 *)(*(_QWORD *)j + 24LL * *(unsigned int *)(j + 8))
                      && (*(_DWORD *)((*v128 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v128 >> 1) & 3) <= (*(_DWORD *)(v187 + 24) | v180) )
                    {
                      v131 = v128[2];
                      if ( v131 )
                      {
                        v209 = v128[2];
                        v208 = v190;
                        *((_QWORD *)&v163 + 1) = v190;
                        *(_QWORD *)&v163 = v196;
                        v207 = (void *)v196;
                        sub_2E0F080(j, v190, (__int64)v128, v196, v129, v130, v163, v131);
                      }
                    }
                  }
                }
              }
            }
          }
LABEL_103:
          if ( ++v84 == v83 )
          {
            v5 = v176;
            break;
          }
        }
      }
      v88 = v204;
      v89 = (unsigned int)v205;
      v90 = sub_2E313E0(a1);
      sub_2E17AE0(v195, a1, v90, a1 + 48, v88, v89);
      sub_2E30390(v224);
      if ( v217 != v219 )
        _libc_free((unsigned __int64)v217);
    }
    if ( a5 )
      sub_2E6BF20(a5, a1, a2, v5);
    v91 = a3[3];
    if ( v91 )
    {
      v92 = *(_DWORD *)(v91 + 24);
      v93 = *(_QWORD *)(v91 + 8);
      if ( v92 )
      {
        v94 = v92 - 1;
        v95 = v94 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v96 = (__int64 *)(v93 + 16LL * v95);
        v97 = *v96;
        if ( a1 == *v96 )
        {
LABEL_112:
          v98 = (_QWORD *)v96[1];
          if ( v98 )
          {
            v99 = v94 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v100 = (__int64 *)(v93 + 16LL * v99);
            v101 = *v100;
            if ( a2 == *v100 )
            {
LABEL_114:
              v102 = (_QWORD *)v100[1];
              if ( v102 )
              {
                v45 = (_QWORD *)v100[1];
                if ( v98 != v102 )
                {
                  while ( 1 )
                  {
                    v45 = (_QWORD *)*v45;
                    if ( v98 == v45 )
                      break;
                    if ( !v45 )
                    {
                      if ( v102 != v98 )
                      {
                        while ( 1 )
                        {
                          v98 = (_QWORD *)*v98;
                          if ( v102 == v98 )
                            break;
                          if ( !v98 )
                          {
                            v98 = (_QWORD *)*v102;
                            if ( *v102 )
                              goto LABEL_42;
                            goto LABEL_43;
                          }
                        }
                      }
                      sub_2EA77F0(v102, v5);
                      goto LABEL_43;
                    }
                  }
                }
LABEL_42:
                sub_2EA77F0(v98, v5);
              }
            }
            else
            {
              v157 = 1;
              while ( v101 != -4096 )
              {
                v158 = v157 + 1;
                v99 = v94 & (v157 + v99);
                v100 = (__int64 *)(v93 + 16LL * v99);
                v101 = *v100;
                if ( a2 == *v100 )
                  goto LABEL_114;
                v157 = v158;
              }
            }
          }
        }
        else
        {
          v155 = 1;
          while ( v97 != -4096 )
          {
            v156 = v155 + 1;
            v95 = v94 & (v155 + v95);
            v96 = (__int64 *)(v93 + 16LL * v95);
            v97 = *v96;
            if ( a1 == *v96 )
              goto LABEL_112;
            v155 = v156;
          }
        }
      }
    }
LABEL_43:
    if ( v204 != v206 )
      _libc_free((unsigned __int64)v204);
    if ( v201 != v203 )
      _libc_free((unsigned __int64)v201);
  }
  return v5;
}
