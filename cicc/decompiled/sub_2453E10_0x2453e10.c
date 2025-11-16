// Function: sub_2453E10
// Address: 0x2453e10
//
__int64 __fastcall sub_2453E10(__int64 a1, __int64 a2)
{
  _QWORD **v3; // r13
  unsigned __int8 *v4; // rax
  unsigned int v5; // esi
  __int64 v6; // rbx
  __int64 v7; // rdi
  int v8; // r10d
  unsigned int v9; // r12d
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // r9
  __int64 result; // rax
  int v15; // ecx
  int v16; // ecx
  _OWORD *v17; // rax
  _BYTE *v18; // rsi
  __int64 v19; // r12
  unsigned __int8 *v20; // rax
  unsigned int v21; // esi
  _QWORD *v22; // r8
  int v23; // r10d
  unsigned __int8 **v24; // rbx
  unsigned int v25; // edx
  _QWORD *v26; // rcx
  unsigned __int8 *v27; // rdi
  __int64 *v28; // r14
  __int64 v29; // rax
  __int64 v30; // rcx
  unsigned __int8 v31; // al
  __int64 v32; // rdx
  char *v33; // rsi
  size_t v34; // rdx
  const void *v35; // rax
  size_t v36; // rdx
  char *v37; // rbx
  size_t v38; // rdx
  __int64 v39; // rax
  int v40; // edx
  char *v41; // rbx
  size_t v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rbx
  unsigned __int8 v45; // r14
  __int64 v46; // rcx
  _BYTE *v47; // rax
  unsigned __int8 v48; // al
  _QWORD *v49; // rdx
  const char *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // r14
  __int64 v59; // rbx
  __int64 v60; // r13
  unsigned __int64 v61; // r12
  __int64 v62; // rsi
  char *v63; // rbx
  char *v64; // r12
  __int64 v65; // rsi
  unsigned __int64 v66; // rbx
  unsigned __int64 v67; // r12
  unsigned __int64 v68; // rdi
  char *v69; // rbx
  char *v70; // r12
  __int64 v71; // rsi
  char *v72; // rbx
  char *v73; // r12
  __int64 v74; // rsi
  char *v75; // rbx
  char *v76; // r12
  __int64 v77; // rsi
  __int64 v78; // rax
  int v79; // ecx
  int v80; // ecx
  __int64 *v81; // r14
  __int64 **v82; // rax
  unsigned __int64 v83; // rax
  __int64 v84; // rdx
  __int64 *v85; // rax
  _QWORD *v86; // rbx
  _QWORD *v87; // rax
  unsigned __int64 v88; // r12
  char v89; // al
  __int64 *v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  bool v93; // cc
  _QWORD *v94; // rax
  __int64 *v95; // rbx
  __int64 v96; // rax
  __int64 *v97; // rax
  __int64 **v98; // r12
  int v99; // eax
  char v100; // cl
  __int64 i; // r12
  _QWORD *v102; // rax
  _QWORD *v103; // rbx
  unsigned __int64 v104; // r12
  __int64 v105; // rsi
  __int64 v106; // rax
  __int64 v107; // rax
  unsigned __int8 *v108; // rax
  size_t v109; // rdx
  size_t v110; // r12
  __int64 v111; // rax
  __int64 v112; // rax
  int v113; // edx
  __int64 v114; // rax
  __int64 v115; // r15
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // rax
  __int64 v119; // rax
  char v120; // al
  _BYTE *v121; // rsi
  unsigned __int8 *v122; // rdx
  _BYTE *v123; // rsi
  int v124; // eax
  int v125; // esi
  _QWORD *v126; // rdi
  unsigned int v127; // edx
  __int64 v128; // r8
  int v129; // r10d
  __int64 *v130; // r9
  int v131; // eax
  int v132; // edx
  _QWORD *v133; // rdi
  int v134; // r9d
  __int64 *v135; // r8
  unsigned int v136; // r12d
  __int64 v137; // rsi
  unsigned __int8 v138; // dl
  __int64 v139; // rdx
  int v140; // eax
  char v141; // cl
  int v142; // edi
  int v143; // edi
  _QWORD *v144; // r8
  unsigned int v145; // edx
  unsigned __int8 *v146; // rsi
  int v147; // r10d
  unsigned __int8 **v148; // r9
  int v149; // esi
  int v150; // esi
  _QWORD *v151; // r8
  unsigned __int8 **v152; // r9
  int v153; // r10d
  unsigned int v154; // edx
  __int64 v155; // [rsp+8h] [rbp-358h]
  char v156; // [rsp+8h] [rbp-358h]
  __int64 v157; // [rsp+18h] [rbp-348h]
  unsigned __int64 v158; // [rsp+20h] [rbp-340h]
  __int64 v159; // [rsp+28h] [rbp-338h]
  __int64 v160; // [rsp+38h] [rbp-328h]
  bool v161; // [rsp+43h] [rbp-31Dh]
  char v162; // [rsp+44h] [rbp-31Ch]
  int v163; // [rsp+44h] [rbp-31Ch]
  __int64 v164; // [rsp+48h] [rbp-318h]
  __int64 **v165; // [rsp+50h] [rbp-310h]
  __int64 **v166; // [rsp+58h] [rbp-308h]
  int v167; // [rsp+60h] [rbp-300h]
  __int64 v168; // [rsp+60h] [rbp-300h]
  unsigned __int8 *v169; // [rsp+60h] [rbp-300h]
  __int64 **v170; // [rsp+68h] [rbp-2F8h]
  int *v171; // [rsp+68h] [rbp-2F8h]
  __int64 *v172; // [rsp+68h] [rbp-2F8h]
  __int64 v173; // [rsp+70h] [rbp-2F0h]
  __int64 v174; // [rsp+70h] [rbp-2F0h]
  __int64 v175; // [rsp+78h] [rbp-2E8h]
  __int64 v176; // [rsp+78h] [rbp-2E8h]
  unsigned __int8 *v177; // [rsp+78h] [rbp-2E8h]
  __int64 v178; // [rsp+78h] [rbp-2E8h]
  _BYTE *v179; // [rsp+80h] [rbp-2E0h]
  __int64 v180; // [rsp+80h] [rbp-2E0h]
  _QWORD **v181; // [rsp+88h] [rbp-2D8h]
  __int64 v182; // [rsp+90h] [rbp-2D0h]
  unsigned int *v183; // [rsp+90h] [rbp-2D0h]
  _QWORD *v184; // [rsp+98h] [rbp-2C8h]
  char v185; // [rsp+A7h] [rbp-2B9h] BYREF
  unsigned __int8 *v186; // [rsp+A8h] [rbp-2B8h] BYREF
  __int64 v187[4]; // [rsp+B0h] [rbp-2B0h] BYREF
  __int64 *v188; // [rsp+D0h] [rbp-290h] BYREF
  size_t v189; // [rsp+D8h] [rbp-288h]
  __int64 v190; // [rsp+E0h] [rbp-280h] BYREF
  __int64 v191[2]; // [rsp+F0h] [rbp-270h] BYREF
  __int64 v192; // [rsp+100h] [rbp-260h] BYREF
  __int64 *v193; // [rsp+110h] [rbp-250h] BYREF
  _QWORD *v194; // [rsp+118h] [rbp-248h]
  __int64 v195; // [rsp+120h] [rbp-240h] BYREF
  _QWORD *v196; // [rsp+130h] [rbp-230h] BYREF
  __int64 v197; // [rsp+138h] [rbp-228h]
  _QWORD v198[8]; // [rsp+140h] [rbp-220h] BYREF
  const char *v199; // [rsp+180h] [rbp-1E0h] BYREF
  __int64 v200; // [rsp+188h] [rbp-1D8h]
  _QWORD v201[2]; // [rsp+190h] [rbp-1D0h] BYREF
  __int64 v202; // [rsp+1A0h] [rbp-1C0h]
  __int64 v203; // [rsp+1A8h] [rbp-1B8h]
  __int64 v204; // [rsp+1B0h] [rbp-1B0h]
  __int64 v205; // [rsp+1B8h] [rbp-1A8h]
  __int64 v206; // [rsp+1C0h] [rbp-1A0h]
  char v207; // [rsp+1C8h] [rbp-198h] BYREF
  char *v208; // [rsp+1E8h] [rbp-178h]
  int v209; // [rsp+1F0h] [rbp-170h]
  char v210; // [rsp+1F8h] [rbp-168h] BYREF
  char *v211; // [rsp+218h] [rbp-148h]
  char v212; // [rsp+228h] [rbp-138h] BYREF
  char *v213; // [rsp+248h] [rbp-118h]
  char v214; // [rsp+258h] [rbp-108h] BYREF
  char *v215; // [rsp+278h] [rbp-E8h]
  int v216; // [rsp+280h] [rbp-E0h]
  char v217; // [rsp+288h] [rbp-D8h] BYREF
  __int64 v218; // [rsp+2B0h] [rbp-B0h]
  unsigned int v219; // [rsp+2C0h] [rbp-A0h]
  unsigned __int64 v220; // [rsp+2C8h] [rbp-98h]
  unsigned int v221; // [rsp+2D0h] [rbp-90h]
  char *v222; // [rsp+2D8h] [rbp-88h] BYREF
  int v223; // [rsp+2E0h] [rbp-80h]
  char v224; // [rsp+2E8h] [rbp-78h] BYREF
  __int64 v225; // [rsp+318h] [rbp-48h]
  unsigned int v226; // [rsp+328h] [rbp-38h]

  v3 = (_QWORD **)a1;
  v4 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a2);
  v5 = *(_DWORD *)(a1 + 176);
  v6 = (__int64)v4;
  v182 = a1 + 152;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_169;
  }
  v7 = *(_QWORD *)(a1 + 160);
  v8 = 1;
  v9 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
  v10 = (v5 - 1) & v9;
  v11 = (__int64 *)(v7 + 56LL * v10);
  v12 = 0;
  v13 = *v11;
  if ( v6 != *v11 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v12 )
        v12 = v11;
      v10 = (v5 - 1) & (v8 + v10);
      v11 = (__int64 *)(v7 + 56LL * v10);
      v13 = *v11;
      if ( v6 == *v11 )
        goto LABEL_3;
      ++v8;
    }
    v15 = *((_DWORD *)v3 + 42);
    if ( !v12 )
      v12 = v11;
    v3[19] = (_QWORD *)((char *)v3[19] + 1);
    v16 = v15 + 1;
    if ( 4 * v16 < 3 * v5 )
    {
      if ( v5 - *((_DWORD *)v3 + 43) - v16 > v5 >> 3 )
      {
LABEL_15:
        *((_DWORD *)v3 + 42) = v16;
        if ( *v12 != -4096 )
          --*((_DWORD *)v3 + 43);
        *v12 = v6;
        v17 = v12 + 1;
        *v17 = 0;
        v17[1] = 0;
        v17[2] = 0;
        v184 = v17;
LABEL_18:
        v18 = (_BYTE *)a2;
        v19 = sub_2451540(v3, a2, 1);
        v184[2] = v19;
        if ( unk_4FE76C8 || unk_4FE7468 == 1 )
        {
          v28 = (__int64 *)**v3;
          v29 = sub_B92180(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL));
          v179 = (_BYTE *)v29;
          if ( v29 )
          {
            v175 = v29 - 16;
            v30 = v29;
            v31 = *(_BYTE *)(v29 - 16);
            if ( (v31 & 2) != 0 )
              v32 = *(_QWORD *)(v30 - 32);
            else
              v32 = v175 - 8LL * ((v31 >> 2) & 0xF);
            sub_AE0470((__int64)&v199, *v3, 1, *(_QWORD *)(v32 + 40));
            if ( off_4C5D190 )
            {
              v33 = off_4C5D190;
              v34 = strlen(off_4C5D190);
            }
            else
            {
              v33 = 0;
              v34 = 0;
            }
            v188 = (__int64 *)sub_B9B140(v28, v33, v34);
            v35 = (const void *)sub_ED1E10(v6);
            v189 = sub_B9B140(v28, v35, v36);
            if ( off_4C5D188[0] )
            {
              v37 = off_4C5D188[0];
              v38 = strlen(off_4C5D188[0]);
            }
            else
            {
              v37 = 0;
              v38 = 0;
            }
            v39 = sub_B9B140(v28, v37, v38);
            v40 = *(_DWORD *)(a2 + 4);
            v191[0] = v39;
            v191[1] = (__int64)sub_B98A20(*(_QWORD *)(a2 + 32 * (1LL - (v40 & 0x7FFFFFF))), (__int64)v37);
            if ( off_4C5D180[0] )
            {
              v41 = off_4C5D180[0];
              v42 = strlen(off_4C5D180[0]);
            }
            else
            {
              v41 = 0;
              v42 = 0;
            }
            v193 = (__int64 *)sub_B9B140(v28, v41, v42);
            v43 = sub_B59B70(a2);
            v194 = sub_B98A20(v43, (__int64)v41);
            v196 = (_QWORD *)sub_B9C770(v28, (__int64 *)&v188, (__int64 *)2, 0, 1);
            v197 = sub_B9C770(v28, v191, (__int64 *)2, 0, 1);
            v198[0] = sub_B9C770(v28, (__int64 *)&v193, (__int64 *)2, 0, 1);
            v44 = sub_ADCD70((__int64)&v199, (__int64)&v196, 3);
            v45 = (*(_BYTE *)(v19 + 32) & 0xFu) - 7 <= 1;
            v46 = sub_ADC950((__int64)&v199, (__int64)"Profile Data Type", 17);
            v47 = v179;
            if ( *v179 != 16 )
            {
              v48 = *(v179 - 16);
              if ( (v48 & 2) != 0 )
                v49 = (_QWORD *)*((_QWORD *)v179 - 4);
              else
                v49 = (_QWORD *)(v175 - 8LL * ((v48 >> 2) & 0xF));
              v47 = (_BYTE *)*v49;
            }
            v176 = v46;
            v173 = (__int64)v47;
            v50 = sub_BD5D20(v19);
            v52 = sub_ADD600(
                    (__int64)&v199,
                    (int)v179,
                    (__int64)v50,
                    v51,
                    0,
                    0,
                    v173,
                    0,
                    v176,
                    v45,
                    1u,
                    0,
                    0,
                    0,
                    0,
                    v44);
            sub_B996C0(v19, v52);
            sub_ADCDB0((__int64)&v199, v52, v53, v54, v55, v56);
            v57 = v226;
            if ( v226 )
            {
              v58 = v225;
              v181 = v3;
              v59 = v225 + 56LL * v226;
              do
              {
                if ( *(_QWORD *)v58 != -4096 && *(_QWORD *)v58 != -8192 )
                {
                  v60 = *(_QWORD *)(v58 + 8);
                  v61 = v60 + 8LL * *(unsigned int *)(v58 + 16);
                  if ( v60 != v61 )
                  {
                    do
                    {
                      v62 = *(_QWORD *)(v61 - 8);
                      v61 -= 8LL;
                      if ( v62 )
                        sub_B91220(v61, v62);
                    }
                    while ( v60 != v61 );
                    v61 = *(_QWORD *)(v58 + 8);
                  }
                  if ( v61 != v58 + 24 )
                    _libc_free(v61);
                }
                v58 += 56;
              }
              while ( v59 != v58 );
              v3 = v181;
              v57 = v226;
            }
            sub_C7D6A0(v225, 56 * v57, 8);
            v63 = v222;
            v64 = &v222[8 * v223];
            if ( v222 != v64 )
            {
              do
              {
                v65 = *((_QWORD *)v64 - 1);
                v64 -= 8;
                if ( v65 )
                  sub_B91220((__int64)v64, v65);
              }
              while ( v63 != v64 );
              v64 = v222;
            }
            if ( v64 != &v224 )
              _libc_free((unsigned __int64)v64);
            v66 = v220;
            v67 = v220 + 56LL * v221;
            if ( v220 != v67 )
            {
              do
              {
                v67 -= 56LL;
                v68 = *(_QWORD *)(v67 + 40);
                if ( v68 != v67 + 56 )
                  _libc_free(v68);
                sub_C7D6A0(*(_QWORD *)(v67 + 16), 8LL * *(unsigned int *)(v67 + 32), 8);
              }
              while ( v66 != v67 );
              v67 = v220;
            }
            if ( (char **)v67 != &v222 )
              _libc_free(v67);
            sub_C7D6A0(v218, 16LL * v219, 8);
            v69 = v215;
            v70 = &v215[8 * v216];
            if ( v215 != v70 )
            {
              do
              {
                v71 = *((_QWORD *)v70 - 1);
                v70 -= 8;
                if ( v71 )
                  sub_B91220((__int64)v70, v71);
              }
              while ( v69 != v70 );
              v70 = v215;
            }
            if ( v70 != &v217 )
              _libc_free((unsigned __int64)v70);
            if ( v213 != &v214 )
              _libc_free((unsigned __int64)v213);
            if ( v211 != &v212 )
              _libc_free((unsigned __int64)v211);
            v72 = v208;
            v73 = &v208[8 * v209];
            if ( v208 != v73 )
            {
              do
              {
                v74 = *((_QWORD *)v73 - 1);
                v73 -= 8;
                if ( v74 )
                  sub_B91220((__int64)v73, v74);
              }
              while ( v72 != v73 );
              v73 = v208;
            }
            if ( v73 != &v210 )
              _libc_free((unsigned __int64)v73);
            v75 = (char *)v205;
            v76 = (char *)(v205 + 8LL * (unsigned int)v206);
            if ( (char *)v205 != v76 )
            {
              do
              {
                v77 = *((_QWORD *)v76 - 1);
                v76 -= 8;
                if ( v77 )
                  sub_B91220((__int64)v76, v77);
              }
              while ( v75 != v76 );
              v76 = (char *)v205;
            }
            if ( v76 != &v207 )
              _libc_free((unsigned __int64)v76);
          }
          v18 = v3[32];
          v78 = v184[2];
          v199 = (const char *)v78;
          if ( v18 == (_BYTE *)v3[33] )
          {
            sub_E48660((__int64)(v3 + 31), v18, &v199);
          }
          else
          {
            if ( v18 )
            {
              *(_QWORD *)v18 = v78;
              v18 = v3[32];
            }
            v18 += 8;
            v3[32] = v18;
          }
          if ( unk_4FE76C8 || unk_4FE7468 == 1 )
            return v184[2];
        }
        v20 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), (__int64)v18);
        v21 = *((_DWORD *)v3 + 44);
        v186 = v20;
        if ( v21 )
        {
          v22 = v3[20];
          v23 = 1;
          v24 = 0;
          v25 = (v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v26 = &v22[7 * v25];
          v27 = (unsigned __int8 *)*v26;
          if ( v20 == (unsigned __int8 *)*v26 )
          {
LABEL_22:
            v183 = (unsigned int *)(v26 + 1);
            if ( v26[4] )
              return v184[2];
LABEL_109:
            v162 = 0;
            v81 = (__int64 *)**v3;
            v167 = *((_DWORD *)v3 + 25);
            v180 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL);
            if ( v167 != 8 )
            {
              LOBYTE(v167) = v186[32] & 0xF;
              v162 = (v186[32] >> 4) & 3;
            }
            v161 = sub_ED2B40(v180, (__int64)*v3);
            sub_2451270((__int64 *)&v188, a2, "__profc_", (void *)8, &v185);
            sub_2451270(v191, a2, "__profd_", (void *)8, &v185);
            v82 = (__int64 **)sub_BCE3C0(v81, 0);
            v160 = sub_AC9EC0(v82);
            v155 = v183[2] + *v183 + (unsigned __int64)v183[1];
            if ( v155 )
            {
              if ( (_BYTE)qword_4FE71C8 )
              {
                v83 = *((unsigned int *)v3 + 25);
                if ( (unsigned int)v83 <= 8 )
                {
                  v84 = 426;
                  if ( _bittest64(&v84, v83) )
                  {
                    v85 = (__int64 *)sub_BCB2E0(v81);
                    v86 = sub_BCD420(v85, v155);
                    v174 = sub_AD6530((__int64)v86, v155);
                    sub_2451270((__int64 *)&v196, a2, "__profvp_", (void *)9, &v185);
                    LOWORD(v202) = 260;
                    BYTE4(v193) = 0;
                    v199 = (const char *)&v196;
                    v87 = sub_BD2C40(88, unk_3F0FAE8);
                    v88 = (unsigned __int64)v87;
                    if ( v87 )
                      sub_B30000((__int64)v87, (__int64)*v3, v86, 0, v167, v174, (__int64)&v199, 0, 0, (__int64)v193, 0);
                    if ( v196 != v198 )
                      j_j___libc_free_0((unsigned __int64)v196);
                    v89 = (16 * v162) | *(_BYTE *)(v88 + 32) & 0xCF;
                    *(_BYTE *)(v88 + 32) = v89;
                    if ( (v89 & 0xFu) - 7 <= 1 || (v89 & 0x30) != 0 && (v89 & 0xF) != 9 )
                      *(_BYTE *)(v88 + 33) |= 0x40u;
                    sub_29F3D50(v3 + 6, v88);
                    sub_ED12E0((__int64)&v199, 5, *((_DWORD *)v3 + 25), 1u);
                    sub_B31A00(v88, (__int64)v199, v200);
                    if ( v199 != (const char *)v201 )
                      j_j___libc_free_0((unsigned __int64)v199);
                    sub_B2F770(v88, 3u);
                    sub_24511A0((__int64)v3, v88, v180, v188, v189);
                    v90 = (__int64 *)sub_B2BE50(v180);
                    v91 = sub_BCE3C0(v90, 0);
                    v160 = sub_ADB060(v88, v91);
                  }
                }
              }
            }
            v92 = sub_B59B70(a2);
            v93 = *(_DWORD *)(v92 + 32) <= 0x40u;
            v94 = *(_QWORD **)(v92 + 24);
            if ( !v93 )
              v94 = (_QWORD *)*v94;
            v159 = (__int64)v94;
            v157 = v183[10];
            v158 = *((_QWORD *)v183 + 2);
            v170 = (__int64 **)sub_AE4420((__int64)(*v3 + 39), **v3, 0);
            v95 = (__int64 *)sub_BCB2C0(v81);
            v166 = (__int64 **)sub_BCD420(v95, 3);
            v196 = (_QWORD *)sub_BCB2E0(v81);
            v96 = sub_BCB2E0(v81);
            v198[0] = v170;
            v198[1] = v170;
            v197 = v96;
            v198[2] = sub_BCE3C0(v81, 0);
            v198[3] = sub_BCE3C0(v81, 0);
            v198[4] = sub_BCB2D0(v81);
            v198[5] = v166;
            v198[6] = sub_BCB2D0(v81);
            v165 = (__int64 **)sub_BD0B90(v81, &v196, 9, 0);
            v97 = (__int64 *)sub_B2BE50(v180);
            v98 = (__int64 **)sub_BCE3C0(v97, 0);
            if ( !(unsigned __int8)sub_2450400(*(_QWORD *)(v180 + 40)) )
              goto LABEL_134;
            v99 = *(_BYTE *)(v180 + 32) & 0xF;
            v100 = *(_BYTE *)(v180 + 32) & 0xF;
            if ( v99 == 2 )
            {
LABEL_191:
              if ( (unsigned __int8)sub_B2DDD0(v180, 0, 0, 1, 0, 0, 0) )
              {
                v138 = *(_BYTE *)(v180 + 32) & 0xF;
                goto LABEL_193;
              }
              v138 = *(_BYTE *)(v180 + 32) & 0xF;
              if ( (unsigned int)v138 - 2 <= 1 )
              {
LABEL_193:
                v164 = v180;
                if ( v138 == 1 )
                  goto LABEL_135;
LABEL_194:
                v164 = v180;
                if ( sub_B2FC80(v180)
                  || (*(_BYTE *)(v180 + 32) & 0xFu) - 7 <= 1
                  || (*(_BYTE *)(v180 + 7) & 0x20) != 0 && sub_B91C10(v180, 19)
                  || *(_QWORD *)(v180 + 48) && (v164 = v180, ((*(_BYTE *)(v180 + 32) >> 4) & 3) == 1)
                  || (v199 = sub_BD5D20(v180),
                      v200 = v139,
                      LOWORD(v202) = 773,
                      v201[0] = ".local",
                      v164 = sub_B305A0(8, (__int64)&v199, v180),
                      !*(_QWORD *)(v180 + 48)) )
                {
LABEL_135:
                  for ( i = 0; i != 3; ++i )
                    v187[i] = sub_ACD640((__int64)v95, v183[i], 0);
                  if ( sub_ED1700((__int64)*v3) )
                  {
                    v156 = 2;
                    LOBYTE(v167) = 0;
                    goto LABEL_144;
                  }
                  if ( !v155 )
                  {
                    if ( *((_BYTE *)v3 + 144) )
                    {
                      if ( v161 && !v185 || *((_DWORD *)v3 + 25) != 3 )
                        goto LABEL_143;
LABEL_215:
                      v156 = 0;
                      LOBYTE(v167) = 8;
                      goto LABEL_144;
                    }
                    if ( (*((_DWORD *)v3 + 25) & 0xFFFFFFFD) == 1 )
                      goto LABEL_215;
                  }
LABEL_143:
                  v156 = v162;
LABEL_144:
                  BYTE4(v193) = 0;
                  LOWORD(v202) = 260;
                  v199 = (const char *)v191;
                  v102 = sub_BD2C40(88, unk_3F0FAE8);
                  v103 = v102;
                  if ( v102 )
                    sub_B30000((__int64)v102, (__int64)*v3, v165, 0, v167, 0, (__int64)&v199, 0, 0, (__int64)v193, 0);
                  v104 = *((_QWORD *)v183 + 4);
                  v105 = (__int64)v170;
                  v168 = sub_ACD640((__int64)v170, 0, 0);
                  if ( unk_4FE7468 == 2 )
                  {
                    v163 = 11;
                    v178 = sub_AD4C50(v158, v170, 0);
                    if ( v104 )
                    {
                      v105 = (__int64)v170;
                      v168 = sub_AD4C50(v104, v170, 0);
                    }
                  }
                  else
                  {
                    v177 = (unsigned __int8 *)sub_AD4C50((unsigned __int64)v103, v170, 0);
                    v106 = sub_AD4C50(v158, v170, 0);
                    v105 = (__int64)v177;
                    v163 = 0;
                    v178 = sub_AD57F0(v106, v177, 0, 0);
                    if ( v104 )
                    {
                      v169 = (unsigned __int8 *)sub_AD4C50((unsigned __int64)v103, v170, 0);
                      v107 = sub_AD4C50(v104, v170, 0);
                      v105 = (__int64)v169;
                      v168 = sub_AD57F0(v107, v169, 0, 0);
                    }
                  }
                  v108 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v105);
                  v171 = (int *)sub_ED1E10((__int64)v108);
                  v110 = v109;
                  sub_C7D030(&v199);
                  sub_C7D280((int *)&v199, v171, v110);
                  sub_C7D290(&v199, &v193);
                  v172 = v193;
                  v111 = sub_BCB2E0(v81);
                  v112 = sub_ACD640(v111, (__int64)v172, 0);
                  v113 = *(_DWORD *)(a2 + 4);
                  v199 = (const char *)v112;
                  v114 = *(_QWORD *)(a2 + 32 * (1LL - (v113 & 0x7FFFFFF)));
                  if ( *(_DWORD *)(v114 + 32) <= 0x40u )
                    v115 = *(_QWORD *)(v114 + 24);
                  else
                    v115 = **(_QWORD **)(v114 + 24);
                  v116 = sub_BCB2E0(v81);
                  v200 = sub_ACD640(v116, v115, 0);
                  v201[0] = v178;
                  v201[1] = v168;
                  v202 = v164;
                  v203 = v160;
                  v117 = sub_BCB2D0(v81);
                  v204 = sub_ACD640(v117, v159, 0);
                  v205 = sub_AD1300(v166, v187, 3);
                  v118 = sub_BCB2D0(v81);
                  v206 = sub_ACD640(v118, v157, 0);
                  v119 = sub_AD24A0(v165, (__int64 *)&v199, 9);
                  sub_B30160((__int64)v103, v119);
                  v120 = (16 * v156) | v103[4] & 0xCF;
                  *((_BYTE *)v103 + 32) = v120;
                  if ( (v120 & 0xFu) - 7 <= 1 || (v120 & 0x30) != 0 && (v120 & 0xF) != 9 )
                    *((_BYTE *)v103 + 33) |= 0x40u;
                  sub_ED12E0((__int64)&v193, v163, *((_DWORD *)v3 + 25), 1u);
                  sub_B31A00((__int64)v103, (__int64)v193, (__int64)v194);
                  if ( v193 != &v195 )
                    j_j___libc_free_0((unsigned __int64)v193);
                  sub_B2F770((__int64)v103, 3u);
                  sub_24511A0((__int64)v3, (__int64)v103, v180, v188, v189);
                  *((_QWORD *)v183 + 3) = v103;
                  v121 = v3[32];
                  v193 = v103;
                  if ( v121 == (_BYTE *)v3[33] )
                  {
                    sub_E48660((__int64)(v3 + 31), v121, &v193);
                  }
                  else
                  {
                    if ( v121 )
                    {
                      *(_QWORD *)v121 = v103;
                      v121 = v3[32];
                    }
                    v3[32] = v121 + 8;
                  }
                  v122 = v186;
                  *((_WORD *)v186 + 16) = *((_WORD *)v186 + 16) & 0xBCC0 | 0x4008;
                  v123 = v3[38];
                  if ( v123 == (_BYTE *)v3[39] )
                  {
                    sub_2453710((__int64)(v3 + 37), v123, &v186);
                  }
                  else
                  {
                    if ( v123 )
                    {
                      *(_QWORD *)v123 = v122;
                      v123 = v3[38];
                    }
                    v3[38] = v123 + 8;
                  }
                  if ( (__int64 *)v191[0] != &v192 )
                    j_j___libc_free_0(v191[0]);
                  if ( v188 != &v190 )
                    j_j___libc_free_0((unsigned __int64)v188);
                  return v184[2];
                }
                v140 = *(_BYTE *)(v180 + 32) & 0xF;
                v141 = *(_BYTE *)(v180 + 32) & 0xF;
                if ( (unsigned int)(v140 - 7) > 1 )
                {
                  *(_BYTE *)(v164 + 32) = v141 | *(_BYTE *)(v164 + 32) & 0xF0;
                }
                else
                {
                  *(_WORD *)(v164 + 32) = *(_BYTE *)(v180 + 32) & 0xF | *(_WORD *)(v164 + 32) & 0xFCC0;
                  if ( v140 == 7 )
                    goto LABEL_203;
                }
                if ( v140 != 8 )
                {
                  if ( (*(_BYTE *)(v164 + 32) & 0x30) == 0 || v141 == 9 )
                  {
                    *(_BYTE *)(v164 + 32) = *(_BYTE *)(v164 + 32) & 0xCF | 0x10;
                    if ( v141 == 9 )
                      goto LABEL_135;
                  }
                  else
                  {
                    *(_WORD *)(v164 + 32) = *(_WORD *)(v164 + 32) & 0xBFCF | 0x4010;
                  }
                  goto LABEL_204;
                }
LABEL_203:
                *(_BYTE *)(v164 + 32) = *(_BYTE *)(v164 + 32) & 0xCF | 0x10;
LABEL_204:
                *(_BYTE *)(v164 + 33) |= 0x40u;
                goto LABEL_135;
              }
LABEL_134:
              v164 = sub_AC9EC0(v98);
              goto LABEL_135;
            }
            if ( v99 == 3 || ((v100 + 9) & 0xFu) <= 1 )
            {
              if ( v100 != 1 )
              {
LABEL_132:
                if ( (unsigned int)(v99 - 7) <= 1 && *(_QWORD *)(v180 + 48) )
                  goto LABEL_134;
                goto LABEL_191;
              }
            }
            else if ( v100 != 1 )
            {
              goto LABEL_194;
            }
            if ( (unsigned __int8)sub_B2D610(v180, 3) )
              goto LABEL_134;
            v99 = *(_BYTE *)(v180 + 32) & 0xF;
            goto LABEL_132;
          }
          while ( v27 != (unsigned __int8 *)-4096LL )
          {
            if ( v27 == (unsigned __int8 *)-8192LL && !v24 )
              v24 = (unsigned __int8 **)v26;
            v25 = (v21 - 1) & (v23 + v25);
            v26 = &v22[7 * v25];
            v27 = (unsigned __int8 *)*v26;
            if ( v20 == (unsigned __int8 *)*v26 )
              goto LABEL_22;
            ++v23;
          }
          if ( !v24 )
            v24 = (unsigned __int8 **)v26;
          v79 = *((_DWORD *)v3 + 42);
          v3[19] = (_QWORD *)((char *)v3[19] + 1);
          v80 = v79 + 1;
          if ( 4 * v80 < 3 * v21 )
          {
            if ( v21 - *((_DWORD *)v3 + 43) - v80 > v21 >> 3 )
            {
LABEL_106:
              *((_DWORD *)v3 + 42) = v80;
              if ( *v24 != (unsigned __int8 *)-4096LL )
                --*((_DWORD *)v3 + 43);
              *v24 = v20;
              v183 = (unsigned int *)(v24 + 1);
              *(_OWORD *)(v24 + 1) = 0;
              *(_OWORD *)(v24 + 3) = 0;
              *(_OWORD *)(v24 + 5) = 0;
              goto LABEL_109;
            }
            sub_24507C0(v182, v21);
            v149 = *((_DWORD *)v3 + 44);
            if ( v149 )
            {
              v150 = v149 - 1;
              v151 = v3[20];
              v152 = 0;
              v153 = 1;
              v154 = v150 & (((unsigned int)v186 >> 9) ^ ((unsigned int)v186 >> 4));
              v24 = (unsigned __int8 **)&v151[7 * v154];
              v20 = *v24;
              v80 = *((_DWORD *)v3 + 42) + 1;
              if ( v186 != *v24 )
              {
                while ( v20 != (unsigned __int8 *)-4096LL )
                {
                  if ( !v152 && v20 == (unsigned __int8 *)-8192LL )
                    v152 = v24;
                  v154 = v150 & (v153 + v154);
                  v24 = (unsigned __int8 **)&v151[7 * v154];
                  v20 = *v24;
                  if ( v186 == *v24 )
                    goto LABEL_106;
                  ++v153;
                }
                v20 = v186;
                if ( v152 )
                  v24 = v152;
              }
              goto LABEL_106;
            }
LABEL_262:
            ++*((_DWORD *)v3 + 42);
            BUG();
          }
        }
        else
        {
          v3[19] = (_QWORD *)((char *)v3[19] + 1);
        }
        sub_24507C0(v182, 2 * v21);
        v142 = *((_DWORD *)v3 + 44);
        if ( v142 )
        {
          v20 = v186;
          v143 = v142 - 1;
          v144 = v3[20];
          v145 = v143 & (((unsigned int)v186 >> 9) ^ ((unsigned int)v186 >> 4));
          v24 = (unsigned __int8 **)&v144[7 * v145];
          v146 = *v24;
          v80 = *((_DWORD *)v3 + 42) + 1;
          if ( *v24 != v186 )
          {
            v147 = 1;
            v148 = 0;
            while ( v146 != (unsigned __int8 *)-4096LL )
            {
              if ( v146 == (unsigned __int8 *)-8192LL && !v148 )
                v148 = v24;
              v145 = v143 & (v147 + v145);
              v24 = (unsigned __int8 **)&v144[7 * v145];
              v146 = *v24;
              if ( v186 == *v24 )
                goto LABEL_106;
              ++v147;
            }
            if ( v148 )
              v24 = v148;
          }
          goto LABEL_106;
        }
        goto LABEL_262;
      }
      sub_24507C0(v182, v5);
      v131 = *((_DWORD *)v3 + 44);
      if ( v131 )
      {
        v132 = v131 - 1;
        v133 = v3[20];
        v134 = 1;
        v135 = 0;
        v136 = (v131 - 1) & v9;
        v12 = &v133[7 * v136];
        v137 = *v12;
        v16 = *((_DWORD *)v3 + 42) + 1;
        if ( v6 != *v12 )
        {
          while ( v137 != -4096 )
          {
            if ( !v135 && v137 == -8192 )
              v135 = v12;
            v136 = v132 & (v134 + v136);
            v12 = &v133[7 * v136];
            v137 = *v12;
            if ( v6 == *v12 )
              goto LABEL_15;
            ++v134;
          }
          if ( v135 )
            v12 = v135;
        }
        goto LABEL_15;
      }
LABEL_261:
      ++*((_DWORD *)v3 + 42);
      BUG();
    }
LABEL_169:
    sub_24507C0(v182, 2 * v5);
    v124 = *((_DWORD *)v3 + 44);
    if ( v124 )
    {
      v125 = v124 - 1;
      v126 = v3[20];
      v127 = (v124 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v12 = &v126[7 * v127];
      v128 = *v12;
      v16 = *((_DWORD *)v3 + 42) + 1;
      if ( v6 != *v12 )
      {
        v129 = 1;
        v130 = 0;
        while ( v128 != -4096 )
        {
          if ( !v130 && v128 == -8192 )
            v130 = v12;
          v127 = v125 & (v129 + v127);
          v12 = &v126[7 * v127];
          v128 = *v12;
          if ( v6 == *v12 )
            goto LABEL_15;
          ++v129;
        }
        if ( v130 )
          v12 = v130;
      }
      goto LABEL_15;
    }
    goto LABEL_261;
  }
LABEL_3:
  v184 = v11 + 1;
  result = v11[3];
  if ( !result )
    goto LABEL_18;
  return result;
}
