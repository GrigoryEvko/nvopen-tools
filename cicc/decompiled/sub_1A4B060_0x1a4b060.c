// Function: sub_1A4B060
// Address: 0x1a4b060
//
__int64 __fastcall sub_1A4B060(
        __int64 a1,
        __m128i a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r12
  int v10; // r14d
  _QWORD *v11; // rbx
  __int64 v12; // rdx
  _QWORD *v13; // r13
  unsigned int v14; // eax
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 *v23; // rax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // rsi
  __int64 *v30; // rdi
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rdx
  __int64 *v35; // rax
  char v36; // r8
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // r8
  unsigned __int64 v39; // rbx
  __int64 v40; // rax
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rax
  char v44; // si
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // r13
  unsigned __int64 v47; // rdx
  _QWORD *v48; // r14
  __int64 v49; // rbx
  __int64 v50; // r13
  unsigned __int64 v51; // rax
  __int64 v52; // r8
  __int16 v53; // dx
  __int64 v54; // rdx
  int v55; // ecx
  __int64 v56; // rsi
  int v57; // eax
  __int64 *v58; // rax
  __int64 v59; // r14
  __int64 v60; // r15
  __int64 *v61; // rax
  char v62; // dl
  __int64 v63; // rbx
  __int64 *v64; // rax
  __int64 *v65; // rcx
  __int64 *v66; // rsi
  unsigned __int64 v67; // rcx
  char v68; // si
  char v69; // al
  bool v70; // al
  __int64 v72; // rcx
  int v73; // edx
  int v74; // edx
  _QWORD **v75; // rcx
  _QWORD *v76; // r14
  __int64 v77; // rdx
  int v78; // ecx
  int v79; // ecx
  _QWORD **v80; // rdx
  _QWORD *v81; // rsi
  __int64 v82; // r13
  __int64 v83; // r15
  __int64 *v84; // rax
  __int64 v85; // r8
  __int64 *v86; // r15
  unsigned int v87; // esi
  int v88; // r9d
  __int64 v89; // rdi
  unsigned int v90; // r13d
  unsigned int v91; // edx
  __int64 **v92; // r14
  __int64 *v93; // rcx
  __int64 v94; // rax
  __int64 *v95; // rax
  __int64 *v96; // rdx
  __int64 v97; // rdx
  int v98; // esi
  int v99; // esi
  _QWORD **v100; // rdx
  __int64 v101; // rdx
  __int64 v102; // r15
  __int64 *v103; // rax
  __int64 *v104; // r14
  __int64 v105; // rax
  __int64 v106; // rsi
  unsigned int v107; // edx
  __int64 v108; // rcx
  __int64 *v109; // rdi
  __int64 v110; // rax
  __int64 v111; // r15
  __int64 v112; // rbx
  __int64 v113; // r14
  __int64 v114; // r12
  char v115; // al
  __int64 v116; // r15
  char v117; // r14
  _QWORD *v118; // rax
  __int64 v119; // r8
  __int64 v120; // r13
  double v121; // xmm4_8
  double v122; // xmm5_8
  __int64 v123; // rdx
  int v124; // ecx
  int v125; // ecx
  _QWORD **v126; // rdx
  int v127; // ecx
  _QWORD **v128; // rdx
  unsigned __int64 v129; // rdi
  int v130; // eax
  __int64 **v131; // r10
  int v132; // eax
  int v133; // edx
  int v134; // edi
  int v135; // edi
  __int64 v136; // r10
  __int64 **v137; // rcx
  unsigned int v138; // eax
  int v139; // esi
  __int64 *v140; // r9
  int v141; // r9d
  int v142; // r9d
  __int64 v143; // r10
  unsigned int v144; // eax
  __int64 *v145; // rdi
  int v146; // esi
  int v147; // ecx
  int v148; // r9d
  __int64 v149; // rcx
  __int64 v150; // rdx
  int v151; // ebx
  unsigned int v152; // eax
  _QWORD *v153; // rdi
  unsigned __int64 v154; // rdx
  unsigned __int64 v155; // rax
  _QWORD *v156; // rax
  __int64 v157; // rdx
  _QWORD *i; // rdx
  _QWORD *v159; // rax
  int v160; // r11d
  __int64 v161; // rax
  unsigned __int8 v162; // [rsp+7h] [rbp-369h]
  __int64 v163; // [rsp+8h] [rbp-368h]
  __int64 v164; // [rsp+10h] [rbp-360h]
  __int64 v165; // [rsp+30h] [rbp-340h]
  __int64 v166; // [rsp+30h] [rbp-340h]
  __int64 v167; // [rsp+30h] [rbp-340h]
  __int64 v168; // [rsp+30h] [rbp-340h]
  __int64 v169; // [rsp+30h] [rbp-340h]
  __int64 v170; // [rsp+30h] [rbp-340h]
  __int64 v171; // [rsp+30h] [rbp-340h]
  __int64 v172; // [rsp+30h] [rbp-340h]
  __int64 v173; // [rsp+38h] [rbp-338h]
  _QWORD v174[16]; // [rsp+40h] [rbp-330h] BYREF
  unsigned __int64 v175; // [rsp+C0h] [rbp-2B0h] BYREF
  __int64 v176; // [rsp+C8h] [rbp-2A8h]
  _QWORD *v177; // [rsp+D0h] [rbp-2A0h] BYREF
  __int64 v178; // [rsp+D8h] [rbp-298h]
  int v179; // [rsp+E0h] [rbp-290h]
  _QWORD v180[8]; // [rsp+E8h] [rbp-288h] BYREF
  unsigned __int64 v181; // [rsp+128h] [rbp-248h] BYREF
  unsigned __int64 v182; // [rsp+130h] [rbp-240h]
  unsigned __int64 v183; // [rsp+138h] [rbp-238h]
  __int64 v184; // [rsp+140h] [rbp-230h] BYREF
  __int64 *v185; // [rsp+148h] [rbp-228h]
  __int64 *v186; // [rsp+150h] [rbp-220h]
  unsigned int v187; // [rsp+158h] [rbp-218h]
  unsigned int v188; // [rsp+15Ch] [rbp-214h]
  int v189; // [rsp+160h] [rbp-210h]
  _BYTE v190[64]; // [rsp+168h] [rbp-208h] BYREF
  unsigned __int64 v191; // [rsp+1A8h] [rbp-1C8h] BYREF
  unsigned __int64 v192; // [rsp+1B0h] [rbp-1C0h]
  unsigned __int64 v193; // [rsp+1B8h] [rbp-1B8h]
  __int64 v194; // [rsp+1C0h] [rbp-1B0h] BYREF
  __int64 v195; // [rsp+1C8h] [rbp-1A8h]
  unsigned __int64 v196; // [rsp+1D0h] [rbp-1A0h]
  _BYTE v197[64]; // [rsp+1E8h] [rbp-188h] BYREF
  unsigned __int64 v198; // [rsp+228h] [rbp-148h]
  unsigned __int64 v199; // [rsp+230h] [rbp-140h]
  unsigned __int64 v200; // [rsp+238h] [rbp-138h]
  _QWORD v201[2]; // [rsp+240h] [rbp-130h] BYREF
  unsigned __int64 v202; // [rsp+250h] [rbp-120h]
  char v203[64]; // [rsp+268h] [rbp-108h] BYREF
  __int64 *v204; // [rsp+2A8h] [rbp-C8h]
  __int64 *v205; // [rsp+2B0h] [rbp-C0h]
  unsigned __int64 v206; // [rsp+2B8h] [rbp-B8h]
  _QWORD v207[2]; // [rsp+2C0h] [rbp-B0h] BYREF
  unsigned __int64 v208; // [rsp+2D0h] [rbp-A0h]
  char v209[64]; // [rsp+2E8h] [rbp-88h] BYREF
  unsigned __int64 v210; // [rsp+328h] [rbp-48h]
  unsigned __int64 v211; // [rsp+330h] [rbp-40h]
  unsigned __int64 v212; // [rsp+338h] [rbp-38h]

  v9 = a1;
  v10 = *(_DWORD *)(a1 + 224);
  ++*(_QWORD *)(a1 + 208);
  v163 = a1 + 208;
  if ( v10 || *(_DWORD *)(a1 + 228) )
  {
    v11 = *(_QWORD **)(a1 + 216);
    v12 = *(unsigned int *)(a1 + 232);
    v13 = &v11[5 * v12];
    v14 = 4 * v10;
    if ( (unsigned int)(4 * v10) < 0x40 )
      v14 = 64;
    if ( (unsigned int)v12 <= v14 )
    {
      for ( ; v11 != v13; v11 += 5 )
      {
        if ( *v11 != -8 )
        {
          if ( *v11 != -16 )
          {
            v15 = v11[1];
            if ( (_QWORD *)v15 != v11 + 3 )
              _libc_free(v15);
          }
          *v11 = -8;
        }
      }
      goto LABEL_13;
    }
    do
    {
      if ( *v11 != -16 && *v11 != -8 )
      {
        v129 = v11[1];
        if ( (_QWORD *)v129 != v11 + 3 )
          _libc_free(v129);
      }
      v11 += 5;
    }
    while ( v11 != v13 );
    v150 = *(unsigned int *)(v9 + 232);
    if ( v10 )
    {
      v151 = 64;
      if ( v10 != 1 )
      {
        _BitScanReverse(&v152, v10 - 1);
        v151 = 1 << (33 - (v152 ^ 0x1F));
        if ( v151 < 64 )
          v151 = 64;
      }
      v153 = *(_QWORD **)(v9 + 216);
      if ( (_DWORD)v150 == v151 )
      {
        *(_QWORD *)(v9 + 224) = 0;
        v159 = &v153[5 * v150];
        do
        {
          if ( v153 )
            *v153 = -8;
          v153 += 5;
        }
        while ( v159 != v153 );
      }
      else
      {
        j___libc_free_0(v153);
        v154 = ((((((((4 * v151 / 3u + 1) | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 2)
                  | (4 * v151 / 3u + 1)
                  | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 4)
                | (((4 * v151 / 3u + 1) | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 2)
                | (4 * v151 / 3u + 1)
                | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 8)
              | (((((4 * v151 / 3u + 1) | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 2)
                | (4 * v151 / 3u + 1)
                | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 4)
              | (((4 * v151 / 3u + 1) | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 2)
              | (4 * v151 / 3u + 1)
              | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 16;
        v155 = (v154
              | (((((((4 * v151 / 3u + 1) | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 2)
                  | (4 * v151 / 3u + 1)
                  | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 4)
                | (((4 * v151 / 3u + 1) | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 2)
                | (4 * v151 / 3u + 1)
                | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 8)
              | (((((4 * v151 / 3u + 1) | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 2)
                | (4 * v151 / 3u + 1)
                | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 4)
              | (((4 * v151 / 3u + 1) | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1)) >> 2)
              | (4 * v151 / 3u + 1)
              | ((unsigned __int64)(4 * v151 / 3u + 1) >> 1))
             + 1;
        *(_DWORD *)(v9 + 232) = v155;
        v156 = (_QWORD *)sub_22077B0(40 * v155);
        v157 = *(unsigned int *)(v9 + 232);
        *(_QWORD *)(v9 + 224) = 0;
        *(_QWORD *)(v9 + 216) = v156;
        for ( i = &v156[5 * v157]; i != v156; v156 += 5 )
        {
          if ( v156 )
            *v156 = -8;
        }
      }
    }
    else
    {
      if ( !(_DWORD)v150 )
      {
LABEL_13:
        *(_QWORD *)(v9 + 224) = 0;
        goto LABEL_14;
      }
      j___libc_free_0(*(_QWORD *)(v9 + 216));
      *(_QWORD *)(v9 + 216) = 0;
      *(_QWORD *)(v9 + 224) = 0;
      *(_DWORD *)(v9 + 232) = 0;
    }
  }
LABEL_14:
  v16 = *(_QWORD *)(v9 + 168);
  memset(v174, 0, sizeof(v174));
  v174[1] = &v174[5];
  v174[2] = &v174[5];
  v17 = *(_QWORD *)(v16 + 56);
  v178 = 0x100000008LL;
  v176 = (__int64)v180;
  v177 = v180;
  v180[0] = v17;
  v201[0] = v17;
  LODWORD(v174[3]) = 8;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v179 = 0;
  v175 = 1;
  LOBYTE(v202) = 0;
  sub_13B8390(&v181, (__int64)v201);
  sub_16CCEE0(&v194, (__int64)v197, 8, (__int64)v174);
  v18 = v174[13];
  memset(&v174[13], 0, 24);
  v198 = v18;
  v199 = v174[14];
  v200 = v174[15];
  sub_16CCEE0(&v184, (__int64)v190, 8, (__int64)&v175);
  v19 = v181;
  v181 = 0;
  v191 = v19;
  v20 = v182;
  v182 = 0;
  v192 = v20;
  v21 = v183;
  v183 = 0;
  v193 = v21;
  sub_16CCEE0(v201, (__int64)v203, 8, (__int64)&v184);
  v22 = v191;
  v191 = 0;
  v204 = (__int64 *)v22;
  v23 = (__int64 *)v192;
  v192 = 0;
  v205 = v23;
  v24 = v193;
  v193 = 0;
  v206 = v24;
  sub_16CCEE0(v207, (__int64)v209, 8, (__int64)&v194);
  v25 = v198;
  v198 = 0;
  v210 = v25;
  v26 = v199;
  v199 = 0;
  v211 = v26;
  v27 = v200;
  v200 = 0;
  v212 = v27;
  if ( v191 )
    j_j___libc_free_0(v191, v193 - v191);
  if ( v186 != v185 )
    _libc_free((unsigned __int64)v186);
  if ( v198 )
    j_j___libc_free_0(v198, v200 - v198);
  if ( v196 != v195 )
    _libc_free(v196);
  if ( v181 )
    j_j___libc_free_0(v181, v183 - v181);
  if ( v177 != (_QWORD *)v176 )
    _libc_free((unsigned __int64)v177);
  if ( v174[13] )
    j_j___libc_free_0(v174[13], v174[15] - v174[13]);
  if ( v174[2] != v174[1] )
    _libc_free(v174[2]);
  sub_16CCCB0(&v184, (__int64)v190, (__int64)v201);
  v29 = v205;
  v30 = v204;
  v191 = 0;
  v192 = 0;
  v193 = 0;
  v31 = (char *)v205 - (char *)v204;
  if ( v205 == v204 )
  {
    v31 = 0;
    v33 = 0;
  }
  else
  {
    if ( v31 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_271;
    v32 = sub_22077B0((char *)v205 - (char *)v204);
    v29 = v205;
    v30 = v204;
    v33 = v32;
  }
  v191 = v33;
  v192 = v33;
  v193 = v33 + v31;
  if ( v30 != v29 )
  {
    v34 = v33;
    v35 = v30;
    do
    {
      if ( v34 )
      {
        *(_QWORD *)v34 = *v35;
        v36 = *((_BYTE *)v35 + 16);
        *(_BYTE *)(v34 + 16) = v36;
        if ( v36 )
          *(_QWORD *)(v34 + 8) = v35[1];
      }
      v35 += 3;
      v34 += 24LL;
    }
    while ( v35 != v29 );
    v33 += 8 * ((unsigned __int64)((char *)(v35 - 3) - (char *)v30) >> 3) + 24;
  }
  v29 = (__int64 *)v197;
  v30 = &v194;
  v192 = v33;
  sub_16CCCB0(&v194, (__int64)v197, (__int64)v207);
  v37 = v211;
  v38 = v210;
  v198 = 0;
  v199 = 0;
  v200 = 0;
  v39 = v211 - v210;
  if ( v211 != v210 )
  {
    if ( v39 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v40 = sub_22077B0(v211 - v210);
      v37 = v211;
      v38 = v210;
      v41 = v40;
      goto LABEL_43;
    }
LABEL_271:
    sub_4261EA(v30, v29, v28);
  }
  v41 = 0;
LABEL_43:
  v198 = v41;
  v199 = v41;
  v200 = v41 + v39;
  if ( v37 == v38 )
  {
    v45 = v41;
  }
  else
  {
    v42 = v41;
    v43 = v38;
    do
    {
      if ( v42 )
      {
        *(_QWORD *)v42 = *(_QWORD *)v43;
        v44 = *(_BYTE *)(v43 + 16);
        *(_BYTE *)(v42 + 16) = v44;
        if ( v44 )
          *(_QWORD *)(v42 + 8) = *(_QWORD *)(v43 + 8);
      }
      v43 += 24LL;
      v42 += 24LL;
    }
    while ( v43 != v37 );
    v45 = v41 + 8 * ((v43 - 24 - v38) >> 3) + 24;
  }
  v46 = v192;
  v47 = v191;
  v199 = v45;
  v162 = 0;
  if ( v192 - v191 == v45 - v41 )
    goto LABEL_88;
  do
  {
LABEL_51:
    v48 = *(_QWORD **)(v46 - 24);
    v49 = *(_QWORD *)(*v48 + 48LL);
    v173 = *v48 + 40LL;
    if ( v49 == v173 )
      goto LABEL_73;
    do
    {
      v50 = v49;
      v49 = *(_QWORD *)(v49 + 8);
      if ( !sub_1456C80(*(_QWORD *)(v9 + 176), *(_QWORD *)(v50 - 24)) )
        continue;
      v51 = *(unsigned __int8 *)(v50 - 8);
      v52 = v50 - 24;
      switch ( (_BYTE)v51 )
      {
        case 0x23:
          v72 = *(_QWORD *)(v50 - 72);
          v73 = *(unsigned __int8 *)(v72 + 16);
          if ( (unsigned __int8)v73 > 0x17u )
          {
            v74 = v73 - 24;
          }
          else
          {
            if ( (_BYTE)v73 != 5 )
              break;
            v74 = *(unsigned __int16 *)(v72 + 18);
          }
          if ( v74 == 38 )
          {
            v75 = (*(_BYTE *)(v72 + 23) & 0x40) != 0
                ? *(_QWORD ***)(v72 - 8)
                : (_QWORD **)(v72 - 24LL * (*(_DWORD *)(v72 + 20) & 0xFFFFFFF));
            v76 = *v75;
            if ( *v75 )
              goto LABEL_122;
          }
          break;
        case 5:
          v53 = *(_WORD *)(v50 - 6);
          if ( v53 == 11 )
          {
            v97 = *(_QWORD *)(v52 - 24LL * (*(_DWORD *)(v50 - 4) & 0xFFFFFFF));
            v98 = *(unsigned __int8 *)(v97 + 16);
            if ( (unsigned __int8)v98 > 0x17u )
            {
              v99 = v98 - 24;
            }
            else
            {
              if ( (_BYTE)v98 != 5 )
                break;
              v99 = *(unsigned __int16 *)(v97 + 18);
            }
            if ( v99 == 38 )
            {
              v100 = (*(_BYTE *)(v97 + 23) & 0x40) != 0
                   ? *(_QWORD ***)(v97 - 8)
                   : (_QWORD **)(v97 - 24LL * (*(_DWORD *)(v97 + 20) & 0xFFFFFFF));
              v76 = *v100;
              if ( *v100 )
              {
                v101 = 1LL - (*(_DWORD *)(v50 - 4) & 0xFFFFFFF);
LABEL_158:
                v77 = *(_QWORD *)(v52 + 24 * v101);
                v78 = *(unsigned __int8 *)(v77 + 16);
                if ( (unsigned __int8)v78 <= 0x17u )
                  goto LABEL_159;
LABEL_123:
                v79 = v78 - 24;
                goto LABEL_124;
              }
            }
          }
          else
          {
            if ( v53 != 13 )
              break;
            v54 = *(_QWORD *)(v52 - 24LL * (*(_DWORD *)(v50 - 4) & 0xFFFFFFF));
            v55 = *(unsigned __int8 *)(v54 + 16);
            if ( (unsigned __int8)v55 > 0x17u )
            {
              v127 = v55 - 24;
            }
            else
            {
              if ( (_BYTE)v55 != 5 )
                break;
              v127 = *(unsigned __int16 *)(v54 + 18);
            }
            if ( v127 == 38 )
            {
              v128 = (*(_BYTE *)(v54 + 23) & 0x40) != 0
                   ? *(_QWORD ***)(v54 - 8)
                   : (_QWORD **)(v54 - 24LL * (*(_DWORD *)(v54 + 20) & 0xFFFFFFF));
              v76 = *v128;
              if ( *v128 )
              {
                v101 = 1LL - (*(_DWORD *)(v50 - 4) & 0xFFFFFFF);
                goto LABEL_158;
              }
            }
          }
          break;
        case 0x25:
          v123 = *(_QWORD *)(v50 - 72);
          v124 = *(unsigned __int8 *)(v123 + 16);
          if ( (unsigned __int8)v124 > 0x17u )
          {
            v125 = v124 - 24;
          }
          else
          {
            if ( (_BYTE)v124 != 5 )
              break;
            v125 = *(unsigned __int16 *)(v123 + 18);
          }
          if ( v125 == 38 )
          {
            v126 = (*(_BYTE *)(v123 + 23) & 0x40) != 0
                 ? *(_QWORD ***)(v123 - 8)
                 : (_QWORD **)(v123 - 24LL * (*(_DWORD *)(v123 + 20) & 0xFFFFFFF));
            v76 = *v126;
            if ( *v126 )
            {
LABEL_122:
              v77 = *(_QWORD *)(v50 - 48);
              v78 = *(unsigned __int8 *)(v77 + 16);
              if ( (unsigned __int8)v78 > 0x17u )
                goto LABEL_123;
LABEL_159:
              if ( (_BYTE)v78 != 5 )
                break;
              v79 = *(unsigned __int16 *)(v77 + 18);
LABEL_124:
              if ( v79 != 38 )
                break;
              v80 = (*(_BYTE *)(v77 + 23) & 0x40) != 0
                  ? *(_QWORD ***)(v77 - 8)
                  : (_QWORD **)(v77 - 24LL * (*(_DWORD *)(v77 + 20) & 0xFFFFFFF));
              v81 = *v80;
              if ( !*v80 )
                break;
              if ( *v76 == *v81 )
              {
                v102 = *(_QWORD *)(v9 + 176);
                v166 = sub_145DC80(v102, (__int64)v81);
                v177 = (_QWORD *)sub_145DC80(*(_QWORD *)(v9 + 176), (__int64)v76);
                v175 = (unsigned __int64)&v177;
                v178 = v166;
                v176 = 0x200000002LL;
                v103 = sub_147DD40(v102, (__int64 *)&v175, 0, 0, a2, a3);
                v52 = v50 - 24;
                v104 = v103;
                if ( (_QWORD **)v175 != &v177 )
                {
                  _libc_free(v175);
                  v52 = v50 - 24;
                }
                v105 = *(unsigned int *)(v9 + 232);
                if ( (_DWORD)v105 )
                {
                  v106 = *(_QWORD *)(v9 + 216);
                  v107 = (v105 - 1) & (((unsigned int)v104 >> 9) ^ ((unsigned int)v104 >> 4));
                  v108 = v106 + 40LL * v107;
                  v109 = *(__int64 **)v108;
                  if ( v104 == *(__int64 **)v108 )
                  {
LABEL_165:
                    if ( v108 != v106 + 40 * v105 )
                    {
                      v110 = *(unsigned int *)(v108 + 16);
                      if ( (_DWORD)v110 )
                      {
                        v167 = v49;
                        v111 = v108;
                        v112 = v9;
                        v113 = v52;
                        while ( 1 )
                        {
                          v114 = *(_QWORD *)(*(_QWORD *)(v111 + 8) + 8 * v110 - 8);
                          v115 = sub_15CCEE0(*(_QWORD *)(v112 + 168), v114, v113);
                          if ( v115 )
                            break;
                          v110 = (unsigned int)(*(_DWORD *)(v111 + 16) - 1);
                          *(_DWORD *)(v111 + 16) = v110;
                          if ( !(_DWORD)v110 )
                          {
                            v9 = v112;
                            v49 = v167;
                            v52 = v113;
                            goto LABEL_175;
                          }
                        }
                        v116 = v114;
                        v52 = v113;
                        v9 = v112;
                        v117 = v115;
                        v49 = v167;
                        if ( !v116 )
                          goto LABEL_175;
                        v168 = v52;
                        v164 = *(_QWORD *)(v50 - 24);
                        LOWORD(v177) = 257;
                        v118 = sub_1648A60(56, 1u);
                        v119 = v168;
                        v120 = (__int64)v118;
                        if ( v118 )
                        {
                          sub_15FC810((__int64)v118, v116, v164, (__int64)&v175, v168);
                          v119 = v168;
                        }
                        v169 = v119;
                        sub_164B7C0(v120, v119);
                        sub_164D160(v169, v120, (__m128)a2, *(double *)a3.m128i_i64, a4, a5, v121, v122, a8, a9);
                        sub_1AEB370(v169, 0);
                        v162 = v117;
                        continue;
                      }
                    }
                  }
                  else
                  {
                    v147 = 1;
                    while ( v109 != (__int64 *)-8LL )
                    {
                      v148 = v147 + 1;
                      v149 = ((_DWORD)v105 - 1) & (v107 + v147);
                      v107 = v149;
                      v108 = v106 + 40 * v149;
                      v109 = *(__int64 **)v108;
                      if ( v104 == *(__int64 **)v108 )
                        goto LABEL_165;
                      v147 = v148;
                    }
                  }
                }
LABEL_175:
                v51 = *(unsigned __int8 *)(v50 - 8);
              }
LABEL_129:
              if ( (unsigned __int8)v51 > 0x2Fu )
                continue;
            }
          }
          break;
        default:
          goto LABEL_129;
      }
      v56 = 0x80A800000000LL;
      if ( !_bittest64(&v56, v51) )
        continue;
      if ( (unsigned __int8)v51 <= 0x17u )
      {
        if ( *(_WORD *)(v50 - 6) != 11 )
          goto LABEL_62;
LABEL_143:
        if ( (*(_BYTE *)(v50 - 7) & 4) != 0 )
        {
          if ( (*(_BYTE *)(v50 - 1) & 0x40) != 0 )
          {
            v96 = *(__int64 **)(v50 - 32);
            v59 = *v96;
            if ( *v96 )
            {
LABEL_146:
              v60 = v96[3];
              if ( v60 )
                goto LABEL_70;
            }
          }
          else
          {
            v96 = (__int64 *)(v52 - 24LL * (*(_DWORD *)(v50 - 4) & 0xFFFFFFF));
            v59 = *v96;
            if ( *v96 )
              goto LABEL_146;
          }
        }
        if ( (_BYTE)v51 == 35 )
          goto LABEL_148;
        goto LABEL_62;
      }
      if ( (_BYTE)v51 == 35 )
        goto LABEL_143;
LABEL_62:
      if ( (v51 & 0xFD) != 0x25 && (_BYTE)v51 != 47 )
        continue;
      if ( (unsigned __int8)v51 <= 0x17u )
      {
        v57 = *(unsigned __int16 *)(v50 - 6);
        goto LABEL_65;
      }
LABEL_148:
      v57 = v51 - 24;
LABEL_65:
      if ( v57 != 13 || (*(_BYTE *)(v50 - 7) & 4) == 0 )
        continue;
      if ( (*(_BYTE *)(v50 - 1) & 0x40) != 0 )
      {
        v58 = *(__int64 **)(v50 - 32);
        v59 = *v58;
        if ( !*v58 )
          continue;
LABEL_69:
        v60 = v58[3];
        if ( !v60 )
          continue;
LABEL_70:
        v165 = v52;
        if ( !(unsigned __int8)sub_14AEC90(v52) )
          continue;
        v82 = *(_QWORD *)(v9 + 176);
        v83 = sub_145DC80(v82, v60);
        v177 = (_QWORD *)sub_145DC80(*(_QWORD *)(v9 + 176), v59);
        v178 = v83;
        v175 = (unsigned __int64)&v177;
        v176 = 0x200000002LL;
        v84 = sub_147DD40(v82, (__int64 *)&v175, 0, 0, a2, a3);
        v85 = v165;
        v86 = v84;
        if ( (_QWORD **)v175 != &v177 )
        {
          _libc_free(v175);
          v85 = v165;
        }
        v87 = *(_DWORD *)(v9 + 232);
        if ( v87 )
        {
          v88 = v87 - 1;
          v89 = *(_QWORD *)(v9 + 216);
          v90 = ((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4);
          v91 = (v87 - 1) & v90;
          v92 = (__int64 **)(v89 + 40LL * v91);
          v93 = *v92;
          if ( v86 == *v92 )
          {
LABEL_139:
            v94 = *((unsigned int *)v92 + 4);
            if ( (unsigned int)v94 >= *((_DWORD *)v92 + 5) )
            {
              v170 = v85;
              sub_16CD150((__int64)(v92 + 1), v92 + 3, 0, 8, v85, v88);
              v85 = v170;
              v95 = &v92[1][*((unsigned int *)v92 + 4)];
            }
            else
            {
              v95 = &v92[1][v94];
            }
LABEL_141:
            *v95 = v85;
            ++*((_DWORD *)v92 + 4);
            continue;
          }
          v130 = 1;
          v131 = 0;
          while ( v93 != (__int64 *)-8LL )
          {
            if ( !v131 && v93 == (__int64 *)-16LL )
              v131 = v92;
            v160 = v130 + 1;
            v161 = v88 & (v91 + v130);
            v91 = v161;
            v92 = (__int64 **)(v89 + 40 * v161);
            v93 = *v92;
            if ( v86 == *v92 )
              goto LABEL_139;
            v130 = v160;
          }
          v132 = *(_DWORD *)(v9 + 224);
          if ( v131 )
            v92 = v131;
          ++*(_QWORD *)(v9 + 208);
          v133 = v132 + 1;
          if ( 4 * (v132 + 1) < 3 * v87 )
          {
            if ( v87 - *(_DWORD *)(v9 + 228) - v133 <= v87 >> 3 )
            {
              v171 = v85;
              sub_1A4AD70(v163, v87);
              v134 = *(_DWORD *)(v9 + 232);
              if ( !v134 )
                goto LABEL_274;
              v135 = v134 - 1;
              v136 = *(_QWORD *)(v9 + 216);
              v137 = 0;
              v138 = v135 & v90;
              v85 = v171;
              v92 = (__int64 **)(v136 + 40LL * (v135 & v90));
              v133 = *(_DWORD *)(v9 + 224) + 1;
              v139 = 1;
              v140 = *v92;
              if ( v86 != *v92 )
              {
                while ( v140 != (__int64 *)-8LL )
                {
                  if ( v140 == (__int64 *)-16LL && !v137 )
                    v137 = v92;
                  v138 = v135 & (v138 + v139);
                  v92 = (__int64 **)(v136 + 40LL * v138);
                  v140 = *v92;
                  if ( v86 == *v92 )
                    goto LABEL_219;
                  ++v139;
                }
                goto LABEL_227;
              }
            }
            goto LABEL_219;
          }
        }
        else
        {
          ++*(_QWORD *)(v9 + 208);
        }
        v172 = v85;
        sub_1A4AD70(v163, 2 * v87);
        v141 = *(_DWORD *)(v9 + 232);
        if ( !v141 )
        {
LABEL_274:
          ++*(_DWORD *)(v9 + 224);
          BUG();
        }
        v142 = v141 - 1;
        v143 = *(_QWORD *)(v9 + 216);
        v85 = v172;
        v144 = v142 & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
        v92 = (__int64 **)(v143 + 40LL * v144);
        v133 = *(_DWORD *)(v9 + 224) + 1;
        v145 = *v92;
        if ( v86 != *v92 )
        {
          v146 = 1;
          v137 = 0;
          while ( v145 != (__int64 *)-8LL )
          {
            if ( !v137 && v145 == (__int64 *)-16LL )
              v137 = v92;
            v144 = v142 & (v144 + v146);
            v92 = (__int64 **)(v143 + 40LL * v144);
            v145 = *v92;
            if ( v86 == *v92 )
              goto LABEL_219;
            ++v146;
          }
LABEL_227:
          if ( v137 )
            v92 = v137;
        }
LABEL_219:
        *(_DWORD *)(v9 + 224) = v133;
        if ( *v92 != (__int64 *)-8LL )
          --*(_DWORD *)(v9 + 228);
        v95 = (__int64 *)(v92 + 3);
        *v92 = v86;
        v92[1] = (__int64 *)(v92 + 3);
        v92[2] = (__int64 *)0x200000000LL;
        goto LABEL_141;
      }
      v58 = (__int64 *)(v52 - 24LL * (*(_DWORD *)(v50 - 4) & 0xFFFFFFF));
      v59 = *v58;
      if ( *v58 )
        goto LABEL_69;
    }
    while ( v49 != v173 );
    v46 = v192;
    v48 = *(_QWORD **)(v192 - 24);
LABEL_73:
    while ( 2 )
    {
      if ( !*(_BYTE *)(v46 - 8) )
      {
        v61 = (__int64 *)v48[3];
        *(_BYTE *)(v46 - 8) = 1;
        *(_QWORD *)(v46 - 16) = v61;
        goto LABEL_77;
      }
      while ( 1 )
      {
        v61 = *(__int64 **)(v46 - 16);
LABEL_77:
        if ( v61 == (__int64 *)v48[4] )
          break;
        *(_QWORD *)(v46 - 16) = v61 + 1;
        v63 = *v61;
        v64 = v185;
        if ( v186 == v185 )
        {
          v65 = &v185[v188];
          if ( v185 != v65 )
          {
            v66 = 0;
            while ( v63 != *v64 )
            {
              if ( *v64 == -2 )
              {
                v66 = v64;
                if ( v65 == v64 + 1 )
                  goto LABEL_85;
                ++v64;
              }
              else if ( v65 == ++v64 )
              {
                if ( !v66 )
                  goto LABEL_131;
LABEL_85:
                *v66 = v63;
                --v189;
                ++v184;
                goto LABEL_86;
              }
            }
            continue;
          }
LABEL_131:
          if ( v188 < v187 )
          {
            ++v188;
            *v65 = v63;
            ++v184;
LABEL_86:
            v175 = v63;
            LOBYTE(v177) = 0;
            sub_13B8390(&v191, (__int64)&v175);
            v47 = v191;
            v46 = v192;
            goto LABEL_87;
          }
        }
        sub_16CCBA0((__int64)&v184, v63);
        if ( v62 )
          goto LABEL_86;
      }
      v192 -= 24LL;
      v47 = v191;
      v46 = v192;
      if ( v192 != v191 )
      {
        v48 = *(_QWORD **)(v192 - 24);
        continue;
      }
      break;
    }
LABEL_87:
    v41 = v198;
  }
  while ( v46 - v47 != v199 - v198 );
LABEL_88:
  if ( v46 != v47 )
  {
    v67 = v41;
    while ( *(_QWORD *)v47 == *(_QWORD *)v67 )
    {
      v68 = *(_BYTE *)(v47 + 16);
      v69 = *(_BYTE *)(v67 + 16);
      if ( v68 && v69 )
        v70 = *(_QWORD *)(v47 + 8) == *(_QWORD *)(v67 + 8);
      else
        v70 = v68 == v69;
      if ( !v70 )
        break;
      v47 += 24LL;
      v67 += 24LL;
      if ( v46 == v47 )
        goto LABEL_96;
    }
    goto LABEL_51;
  }
LABEL_96:
  if ( v41 )
    j_j___libc_free_0(v41, v200 - v41);
  if ( v196 != v195 )
    _libc_free(v196);
  if ( v191 )
    j_j___libc_free_0(v191, v193 - v191);
  if ( v186 != v185 )
    _libc_free((unsigned __int64)v186);
  if ( v210 )
    j_j___libc_free_0(v210, v212 - v210);
  if ( v208 != v207[1] )
    _libc_free(v208);
  if ( v204 )
    j_j___libc_free_0(v204, v206 - (_QWORD)v204);
  if ( v202 != v201[1] )
    _libc_free(v202);
  return v162;
}
