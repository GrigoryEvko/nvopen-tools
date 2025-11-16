// Function: sub_2AC7F80
// Address: 0x2ac7f80
//
__int64 __fastcall sub_2AC7F80(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 result; // rax
  _QWORD *v6; // r15
  __int64 v7; // r13
  int v8; // edx
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 i; // rcx
  int v13; // edi
  char v14; // al
  __int64 v15; // r12
  bool v16; // zf
  __int64 v17; // rax
  int v18; // ebx
  unsigned int v19; // esi
  int v20; // edx
  char v21; // dl
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r12
  int v26; // r12d
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r8
  unsigned __int8 *v30; // rbx
  unsigned __int8 *v31; // r12
  __int64 **v32; // rax
  __int64 v33; // r15
  unsigned __int8 *v34; // r13
  __int64 v35; // rbx
  __int64 v36; // rax
  unsigned int v37; // edx
  unsigned int v38; // r12d
  signed __int64 v39; // rbx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rsi
  unsigned __int64 v46; // rax
  bool v47; // of
  unsigned __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 *v57; // r14
  __int64 *v58; // rbx
  __int64 v59; // r13
  __int64 v60; // rax
  unsigned __int64 v61; // rdx
  char v62; // al
  char v63; // al
  __int64 v64; // rdx
  __int64 v65; // rsi
  int v66; // ecx
  int v67; // r10d
  __int64 v68; // r11
  unsigned int v69; // ecx
  unsigned __int64 v70; // r14
  __int64 v71; // rax
  _QWORD *v72; // rbx
  unsigned __int8 *v73; // r15
  __int64 v74; // rdx
  unsigned int *v75; // rsi
  unsigned int *v76; // rax
  __int64 v77; // rdx
  unsigned int *v78; // r13
  char v79; // r12
  _QWORD *v80; // r14
  unsigned int *v81; // rbx
  __int64 *v82; // rax
  __int64 v83; // r11
  __int64 v84; // r9
  int v85; // eax
  __int64 v86; // rdx
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r9
  bool v92; // r12
  __int64 v93; // rbx
  unsigned __int64 v94; // rdi
  unsigned __int64 v95; // rdi
  unsigned __int64 v96; // rdi
  __int64 *v97; // rsi
  __int64 v98; // rax
  __int64 v99; // rbx
  int v100; // edx
  int v101; // r14d
  int v102; // eax
  int v103; // edx
  int v104; // r12d
  __int64 v105; // rax
  int v106; // ecx
  bool v107; // cc
  int v108; // edx
  _DWORD *v109; // rax
  char v110; // al
  int v111; // ecx
  unsigned __int64 v112; // rax
  int v113; // edi
  unsigned int v114; // esi
  int v115; // edx
  char v116; // dl
  unsigned __int64 v117; // rax
  __int64 v118; // rdi
  __int64 v119; // rax
  __int64 v120; // rax
  int v121; // eax
  int v122; // ebx
  __int64 v123; // r12
  int v124; // r13d
  unsigned __int64 v125; // rax
  int v126; // esi
  int v127; // edx
  unsigned int v128; // esi
  char v129; // dl
  unsigned __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 v133; // r10
  _QWORD *v134; // rax
  __int64 v135; // rax
  __int64 *v136; // rsi
  unsigned int v137; // eax
  __int64 v138; // rdx
  signed __int64 v139; // rax
  __int64 v140; // r8
  unsigned __int64 v141; // rdx
  int v142; // r8d
  size_t v143; // rax
  unsigned __int8 *v144; // rdi
  _BYTE *v145; // rdi
  __int64 v146; // [rsp+8h] [rbp-9D8h]
  __int64 v147; // [rsp+10h] [rbp-9D0h]
  int v148; // [rsp+18h] [rbp-9C8h]
  __int64 v149; // [rsp+18h] [rbp-9C8h]
  __int64 v150; // [rsp+18h] [rbp-9C8h]
  __int64 v151; // [rsp+20h] [rbp-9C0h]
  int v152; // [rsp+20h] [rbp-9C0h]
  int v153; // [rsp+20h] [rbp-9C0h]
  __int64 v154; // [rsp+20h] [rbp-9C0h]
  __int64 v155; // [rsp+28h] [rbp-9B8h]
  __int64 v156; // [rsp+28h] [rbp-9B8h]
  __int64 v157; // [rsp+28h] [rbp-9B8h]
  _BYTE *src; // [rsp+30h] [rbp-9B0h]
  _QWORD *dest; // [rsp+38h] [rbp-9A8h]
  void *desta; // [rsp+38h] [rbp-9A8h]
  __int64 v161; // [rsp+40h] [rbp-9A0h]
  unsigned __int64 v162; // [rsp+48h] [rbp-998h]
  __int64 v163; // [rsp+60h] [rbp-980h]
  __int64 v164; // [rsp+90h] [rbp-950h]
  int v165; // [rsp+98h] [rbp-948h]
  char v166; // [rsp+9Fh] [rbp-941h]
  __int64 v167; // [rsp+A0h] [rbp-940h]
  __int64 v168; // [rsp+A8h] [rbp-938h]
  unsigned int v169; // [rsp+B0h] [rbp-930h]
  __int64 v170; // [rsp+B0h] [rbp-930h]
  void *v171; // [rsp+B8h] [rbp-928h]
  _BYTE *v172; // [rsp+B8h] [rbp-928h]
  void *v173; // [rsp+B8h] [rbp-928h]
  _QWORD *v174; // [rsp+B8h] [rbp-928h]
  signed __int64 v175; // [rsp+C0h] [rbp-920h]
  __int64 **v176; // [rsp+C0h] [rbp-920h]
  int v177; // [rsp+C0h] [rbp-920h]
  __int64 v178; // [rsp+C8h] [rbp-918h]
  _QWORD *v179; // [rsp+E0h] [rbp-900h]
  int v180; // [rsp+E0h] [rbp-900h]
  char v181; // [rsp+E0h] [rbp-900h]
  unsigned __int8 *v183; // [rsp+F0h] [rbp-8F0h]
  unsigned __int8 *v184; // [rsp+F8h] [rbp-8E8h] BYREF
  _BYTE *v185; // [rsp+100h] [rbp-8E0h] BYREF
  __int64 v186; // [rsp+108h] [rbp-8D8h]
  _BYTE v187[32]; // [rsp+110h] [rbp-8D0h] BYREF
  _BYTE *v188; // [rsp+130h] [rbp-8B0h] BYREF
  __int64 v189; // [rsp+138h] [rbp-8A8h]
  _BYTE v190[32]; // [rsp+140h] [rbp-8A0h] BYREF
  __int64 *v191; // [rsp+160h] [rbp-880h] BYREF
  __int64 v192; // [rsp+168h] [rbp-878h]
  _BYTE v193[32]; // [rsp+170h] [rbp-870h] BYREF
  unsigned __int8 *v194; // [rsp+190h] [rbp-850h] BYREF
  __int64 v195; // [rsp+198h] [rbp-848h]
  _BYTE v196[32]; // [rsp+1A0h] [rbp-840h] BYREF
  unsigned __int8 *v197; // [rsp+1C0h] [rbp-820h] BYREF
  _BYTE *v198; // [rsp+1C8h] [rbp-818h] BYREF
  __int64 v199; // [rsp+1D0h] [rbp-810h]
  _BYTE v200[128]; // [rsp+1D8h] [rbp-808h] BYREF
  unsigned __int64 v201[2]; // [rsp+258h] [rbp-788h] BYREF
  _BYTE v202[16]; // [rsp+268h] [rbp-778h] BYREF
  unsigned __int64 v203[2]; // [rsp+278h] [rbp-768h] BYREF
  _BYTE v204[16]; // [rsp+288h] [rbp-758h] BYREF
  int v205; // [rsp+298h] [rbp-748h]
  unsigned __int8 *v206; // [rsp+2A0h] [rbp-740h] BYREF
  __int64 v207; // [rsp+2A8h] [rbp-738h]
  _BYTE v208[1840]; // [rsp+2B0h] [rbp-730h] BYREF

  v184 = (unsigned __int8 *)a2;
  v163 = sub_2ABFD00(a1 + 224, (__int64)&v184);
  if ( !v163 )
    v163 = *(_QWORD *)(a1 + 232) + 72LL * *(unsigned int *)(a1 + 248);
  v3 = *(_QWORD *)(a1 + 416);
  v4 = *(_QWORD *)(v3 + 40);
  result = *(_QWORD *)(v3 + 32);
  v161 = v4;
  v164 = result;
  if ( v4 != result )
  {
    while ( 1 )
    {
      v178 = *(_QWORD *)v164 + 48LL;
      if ( *(_QWORD *)(*(_QWORD *)v164 + 56LL) != v178 )
        break;
LABEL_29:
      v164 += 8;
      result = v164;
      if ( v161 == v164 )
        return result;
    }
    v6 = *(_QWORD **)(*(_QWORD *)v164 + 56LL);
    while ( 1 )
    {
      if ( !v6 )
        BUG();
      if ( *((_BYTE *)v6 - 24) != 85 )
        goto LABEL_28;
      v7 = *(v6 - 2);
      v188 = v190;
      v189 = 0x400000000LL;
      v192 = 0x400000000LL;
      v191 = (__int64 *)v193;
      v8 = *((unsigned __int8 *)v6 - 24);
      v183 = (unsigned __int8 *)(v6 - 3);
      if ( v8 == 40 )
      {
        v9 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v183);
      }
      else
      {
        v9 = -32;
        if ( v8 != 85 )
        {
          if ( v8 != 34 )
            BUG();
          v9 = -96;
        }
      }
      if ( *((char *)v6 - 17) < 0 )
      {
        v23 = sub_BD2BC0((__int64)v183);
        v25 = v23 + v24;
        if ( *((char *)v6 - 17) >= 0 )
        {
          if ( (unsigned int)(v25 >> 4) )
LABEL_232:
            BUG();
        }
        else if ( (unsigned int)((v25 - sub_BD2BC0((__int64)v183)) >> 4) )
        {
          if ( *((char *)v6 - 17) >= 0 )
            goto LABEL_232;
          v26 = *(_DWORD *)(sub_BD2BC0((__int64)v183) + 8);
          if ( *((char *)v6 - 17) >= 0 )
            BUG();
          v27 = sub_BD2BC0((__int64)v183);
          v9 -= 32LL * (unsigned int)(*(_DWORD *)(v27 + v28 - 4) - v26);
        }
      }
      v29 = (unsigned int)v192;
      v30 = &v183[v9];
      v31 = &v183[-32 * (*((_DWORD *)v6 - 5) & 0x7FFFFFF)];
      if ( v31 != v30 )
      {
        v179 = v6;
        v32 = &v191;
        v33 = v7;
        v34 = v30;
        do
        {
          v35 = *(_QWORD *)(*(_QWORD *)v31 + 8LL);
          if ( v29 + 1 > (unsigned __int64)HIDWORD(v192) )
          {
            v176 = v32;
            sub_C8D5F0((__int64)v32, v193, v29 + 1, 8u, v29, v2);
            v29 = (unsigned int)v192;
            v32 = v176;
          }
          v31 += 32;
          v191[v29] = v35;
          v29 = (unsigned int)(v192 + 1);
          LODWORD(v192) = v192 + 1;
        }
        while ( v34 != v31 );
        v7 = v33;
        v6 = v179;
      }
      v36 = sub_DFD7B0(*(_QWORD *)(a1 + 448));
      v38 = v37;
      v39 = v36;
      v43 = sub_2AC04F0(a1, (__int64)v183, (unsigned __int64)v184, v40, v41, v42);
      i = v44;
      v45 = v43;
      v46 = (unsigned int)v184 * v39;
      v13 = (int)v184;
      if ( !is_mul_ok((unsigned int)v184, v39) )
      {
        if ( v39 <= 0 || (v46 = 0x7FFFFFFFFFFFFFFFLL, !(_DWORD)v184) )
          v46 = 0x8000000000000000LL;
      }
      v11 = 1;
      if ( (_DWORD)i != 1 )
        v11 = v38;
      v47 = __OFADD__(v45, v46);
      v48 = v45 + v46;
      v180 = v11;
      if ( v47 )
      {
        v11 = 0x7FFFFFFFFFFFFFFFLL;
        v48 = 0x8000000000000000LL;
        if ( v45 > 0 )
          v48 = 0x7FFFFFFFFFFFFFFFLL;
      }
      v175 = v48;
      if ( BYTE4(v184) )
      {
        if ( !(_DWORD)v184 )
          goto LABEL_50;
      }
      else if ( (unsigned int)v184 <= 1 )
      {
        goto LABEL_50;
      }
      v11 = 9LL * *(unsigned int *)(a1 + 248);
      if ( v163 != *(_QWORD *)(a1 + 232) + 72LL * *(unsigned int *)(a1 + 248) )
      {
        v63 = sub_B19060(v163 + 8, (__int64)v183, v11, i);
        v13 = (int)v184;
        if ( v63 )
          goto LABEL_16;
      }
      if ( *((_BYTE *)v6 - 24) == 85 )
      {
        v131 = *(v6 - 7);
        if ( v131 )
        {
          if ( !*(_BYTE *)v131
            && *(_QWORD *)(v131 + 24) == v6[7]
            && (*(_BYTE *)(v131 + 33) & 0x20) != 0
            && *(_DWORD *)(v131 + 36) == 291 )
          {
LABEL_50:
            v166 = sub_B19060(*(_QWORD *)(a1 + 440) + 440LL, (__int64)v183, v11, i);
            v206 = v184;
            if ( *(_BYTE *)(v7 + 8) == 15 )
              v167 = (__int64)sub_E454C0(v7, (__int64)v184, v49, v50, v51, v52);
            else
              v167 = sub_2AAEDF0(v7, (__int64)v184);
            v57 = v191;
            v58 = &v191[(unsigned int)v192];
            if ( v58 != v191 )
            {
              do
              {
                v59 = *v57;
                v197 = v184;
                v62 = *(_BYTE *)(v59 + 8);
                if ( v62 == 15 )
                {
                  v59 = (__int64)sub_E454C0(v59, (__int64)v184, v53, v54, v55, v56);
                }
                else
                {
                  v206 = v184;
                  if ( ((v62 - 7) & 0xFD) != 0 && (BYTE4(v184) || (_DWORD)v184 != 1) )
                    v59 = sub_BCE1B0((__int64 *)v59, (__int64)v206);
                }
                v60 = (unsigned int)v189;
                v54 = HIDWORD(v189);
                v61 = (unsigned int)v189 + 1LL;
                if ( v61 > HIDWORD(v189) )
                {
                  sub_C8D5F0((__int64)&v188, v190, v61, 8u, v55, v56);
                  v60 = (unsigned int)v189;
                }
                v53 = (__int64)v188;
                ++v57;
                *(_QWORD *)&v188[8 * v60] = v59;
                LODWORD(v189) = v189 + 1;
              }
              while ( v58 != v57 );
            }
            v168 = a1 + 384;
            if ( *((_BYTE *)v6 - 24) != 85
              || (v120 = *(v6 - 7)) == 0
              || *(_BYTE *)v120
              || *(_QWORD *)(v120 + 24) != v6[7]
              || (*(_BYTE *)(v120 + 33) & 0x20) == 0
              || *(_DWORD *)(v120 + 36) != 174
              || (sub_2AB5570((__int64)&v206, a1, (__int64)v183, (__int64)v184, v167), !v208[0]) )
            {
              LODWORD(v197) = 0;
              v198 = v200;
              v201[0] = (unsigned __int64)v202;
              v206 = v208;
              v199 = 0x800000000LL;
              v207 = 0x800000000LL;
              BYTE4(v197) = 0;
              v201[1] = 0;
              v202[0] = 0;
              v203[0] = (unsigned __int64)v204;
              v203[1] = 0;
              v204[0] = 0;
              sub_D39570((__int64)v183, (unsigned int *)&v206);
              v70 = (unsigned __int64)v206;
              v71 = 224LL * (unsigned int)v207;
              if ( &v206[v71] != v206 )
              {
                v72 = v6;
                v73 = &v206[v71];
LABEL_79:
                if ( *(_DWORD *)v70 != (_DWORD)v184 || *(_BYTE *)(v70 + 4) != BYTE4(v184) )
                  goto LABEL_78;
                v74 = *(unsigned int *)(v70 + 16);
                if ( v166 )
                {
                  if ( !(_DWORD)v74 )
                    goto LABEL_78;
                  v75 = *(unsigned int **)(v70 + 8);
                  v76 = v75 + 1;
                  while ( *v76 != 10 )
                  {
                    v76 += 4;
                    if ( &v75[4 * (unsigned int)(v74 - 1) + 5] == v76 )
                      goto LABEL_78;
                  }
                }
                else
                {
                  v75 = *(unsigned int **)(v70 + 8);
                }
                v77 = 4 * v74;
                if ( &v75[v77] == v75 )
                  goto LABEL_100;
                v78 = v75;
                v162 = v70;
                v79 = 1;
                v80 = v72;
                v81 = &v75[v77];
                while ( 1 )
                {
LABEL_94:
                  v85 = v78[1];
                  v86 = *v78;
                  if ( v85 == 9 )
                  {
                    v118 = *(_QWORD *)(a1 + 424);
                    v173 = *(void **)(a1 + 416);
                    v170 = *(_QWORD *)(v118 + 112);
                    v119 = sub_DEEF40(v118, *(_QWORD *)&v183[32 * (v86 - (*((_DWORD *)v80 - 5) & 0x7FFFFFF))]);
                    if ( !sub_DADE90(v170, v119, (__int64)v173) )
                      goto LABEL_92;
                    goto LABEL_93;
                  }
                  if ( v85 <= 9 )
                    break;
                  if ( v85 != 10 )
                    v79 = 0;
                  v78 += 4;
                  if ( v81 == v78 )
                  {
LABEL_99:
                    v72 = v80;
                    v70 = v162;
                    if ( v79 )
                    {
LABEL_100:
                      v6 = v72;
                      v87 = sub_B43CA0((__int64)v183);
                      v172 = sub_BA8CB0(v87, *(_QWORD *)(v70 + 184), *(_QWORD *)(v70 + 192));
                      LODWORD(v197) = *(_DWORD *)v70;
                      BYTE4(v197) = *(_BYTE *)(v70 + 4);
                      sub_2AA8CF0((__int64)&v198, v70 + 8, v88, v89, v90, v91);
                      sub_2240AE0(v201, (unsigned __int64 *)(v70 + 152));
                      sub_2240AE0(v203, (unsigned __int64 *)(v70 + 184));
                      v92 = v172 != 0;
                      v205 = *(_DWORD *)(v70 + 216);
                      goto LABEL_101;
                    }
LABEL_78:
                    v70 += 224LL;
                    if ( v73 != (unsigned __int8 *)v70 )
                      goto LABEL_79;
                    v6 = v72;
                    v92 = 0;
                    v172 = 0;
LABEL_101:
                    v93 = (__int64)v206;
                    v70 = (unsigned __int64)&v206[224 * (unsigned int)v207];
                    if ( v206 != (unsigned __int8 *)v70 )
                    {
                      do
                      {
                        v70 -= 224LL;
                        v94 = *(_QWORD *)(v70 + 184);
                        if ( v94 != v70 + 200 )
                          j_j___libc_free_0(v94);
                        v95 = *(_QWORD *)(v70 + 152);
                        if ( v95 != v70 + 168 )
                          j_j___libc_free_0(v95);
                        v96 = *(_QWORD *)(v70 + 8);
                        if ( v96 != v70 + 24 )
                          _libc_free(v96);
                      }
                      while ( v93 != v70 );
                      v70 = (unsigned __int64)v206;
                    }
                    if ( (_BYTE *)v70 != v208 )
LABEL_111:
                      _libc_free(v70);
                    v97 = *(__int64 **)(a1 + 456);
                    if ( v97 && v92 )
                    {
                      if ( !(unsigned __int8)sub_A73ED0(v6 + 6, 23) && !(unsigned __int8)sub_B49560((__int64)v183, 23)
                        || (unsigned __int8)sub_A73ED0(v6 + 6, 4)
                        || (unsigned __int8)sub_B49560((__int64)v183, 4) )
                      {
                        v98 = sub_DFD7B0(*(_QWORD *)(a1 + 448));
                        v97 = *(__int64 **)(a1 + 456);
                        v99 = v98;
                        v101 = v100;
                      }
                      else
                      {
                        v101 = 1;
                        v99 = 0;
                        v97 = *(__int64 **)(a1 + 456);
                      }
                    }
                    else
                    {
                      v101 = 1;
                      v99 = 0;
                    }
                    goto LABEL_117;
                  }
                }
                if ( !v85 )
                  goto LABEL_93;
                if ( v85 != 1 )
                  goto LABEL_92;
                v169 = v78[2];
                v171 = *(void **)(*(_QWORD *)(a1 + 424) + 112LL);
                v82 = sub_DD8400((__int64)v171, *(_QWORD *)&v183[32 * (v86 - (*((_DWORD *)v80 - 5) & 0x7FFFFFF))]);
                if ( *((_WORD *)v82 + 12) != 8 )
                  goto LABEL_92;
                v83 = v82[6];
                v84 = (__int64)v171;
                if ( *(_QWORD *)(a1 + 416) != v83 )
                  goto LABEL_92;
                v132 = v82[5];
                v133 = v82[4];
                if ( v132 == 2 )
                {
                  v134 = *(_QWORD **)(v133 + 8);
LABEL_187:
                  if ( *((_WORD *)v134 + 12) )
                    goto LABEL_92;
                  v135 = v134[4];
                  v136 = *(__int64 **)(v135 + 24);
                  v137 = *(_DWORD *)(v135 + 32);
                  if ( v137 > 0x40 )
                  {
                    v138 = *v136;
                  }
                  else
                  {
                    v138 = 0;
                    if ( v137 )
                      v138 = (__int64)((_QWORD)v136 << (64 - (unsigned __int8)v137)) >> (64 - (unsigned __int8)v137);
                  }
                  if ( v169 != v138 )
LABEL_92:
                    v79 = 0;
LABEL_93:
                  v78 += 4;
                  if ( v81 == v78 )
                    goto LABEL_99;
                  goto LABEL_94;
                }
                v185 = v187;
                v186 = 0x300000000LL;
                v139 = 8 * v132 - 8;
                v140 = v139 >> 3;
                if ( (unsigned __int64)v139 > 0x18 )
                {
                  v150 = v133;
                  v154 = 8 * v132 - 8;
                  v157 = v83;
                  sub_C8D5F0((__int64)&v185, v187, v154 >> 3, 8u, v140, (__int64)v171);
                  LODWORD(v140) = v154 >> 3;
                  v84 = (__int64)v171;
                  v83 = v157;
                  v139 = v154;
                  v145 = &v185[8 * (unsigned int)v186];
                  v133 = v150;
                }
                else
                {
                  src = v187;
                  if ( 8 * v132 == 8 )
                    goto LABEL_200;
                  v145 = v187;
                }
                v152 = v140;
                v156 = v83;
                desta = (void *)v84;
                memcpy(v145, (const void *)(v133 + 8), v139);
                LODWORD(v140) = v152;
                v83 = v156;
                v84 = (__int64)desta;
                src = v185;
                LODWORD(v139) = v186;
LABEL_200:
                v141 = (unsigned int)(v140 + v139);
                v194 = v196;
                v142 = v141;
                LODWORD(v186) = v141;
                v143 = 8 * v141;
                v195 = 0x400000000LL;
                if ( v141 > 4 )
                {
                  v146 = 8 * v141;
                  v147 = v83;
                  v149 = v84;
                  v153 = v141;
                  sub_C8D5F0((__int64)&v194, v196, v141, 8u, v141, v84);
                  v142 = v153;
                  v84 = v149;
                  v83 = v147;
                  v144 = &v194[8 * (unsigned int)v195];
                  v143 = v146;
                }
                else
                {
                  if ( !v143 )
                  {
LABEL_202:
                    LODWORD(v195) = v142 + v143;
                    v134 = sub_DBFF60(v84, (unsigned int *)&v194, v83, 0);
                    if ( v194 != v196 )
                    {
                      dest = v134;
                      _libc_free((unsigned __int64)v194);
                      v134 = dest;
                    }
                    if ( v185 != v187 )
                    {
                      v174 = v134;
                      _libc_free((unsigned __int64)v185);
                      v134 = v174;
                    }
                    goto LABEL_187;
                  }
                  v144 = v196;
                }
                v148 = v142;
                v151 = v83;
                v155 = v84;
                memcpy(v144, src, v143);
                LODWORD(v143) = v195;
                v142 = v148;
                v83 = v151;
                v84 = v155;
                goto LABEL_202;
              }
              if ( &v206[v71] != v208 )
              {
                v92 = 0;
                v172 = 0;
                goto LABEL_111;
              }
              v101 = 1;
              v99 = 0;
              v172 = 0;
              v97 = *(__int64 **)(a1 + 456);
LABEL_117:
              v102 = sub_9B78C0((__int64)v183, v97);
              v103 = 1;
              v104 = v102;
              v105 = 0;
              if ( v104 )
                v105 = sub_2AB3340(a1, v183, (__int64)v184);
              v106 = 6;
              v107 = v180 < v101;
              if ( v180 == v101 )
              {
                if ( v99 > v175 )
                {
                  v99 = v175;
                  v106 = 5;
                }
              }
              else
              {
                if ( v180 < v101 )
                  v101 = v180;
                if ( v107 )
                {
                  v99 = v175;
                  v106 = 5;
                }
              }
              if ( v103 == v101 )
              {
                if ( v105 <= v99 )
                {
                  v99 = v105;
                  v106 = 7;
                }
              }
              else if ( v101 >= v103 )
              {
                v101 = v103;
                v99 = v105;
                v106 = 7;
              }
              if ( (_DWORD)v199 )
              {
                v108 = 0;
                v109 = v198 + 4;
                while ( *v109 != 10 )
                {
                  ++v108;
                  v109 += 4;
                  if ( (_DWORD)v199 == v108 )
                    goto LABEL_152;
                }
                v165 = v108;
                v181 = 1;
              }
              else
              {
LABEL_152:
                v181 = 0;
              }
              v177 = v106;
              v206 = v183;
              LODWORD(v207) = (_DWORD)v184;
              BYTE4(v207) = BYTE4(v184);
              v110 = sub_2ABE520(v168, (__int64 *)&v206, &v185);
              v111 = v177;
              v16 = v110 == 0;
              v112 = (unsigned __int64)v185;
              if ( !v16 )
                goto LABEL_138;
              v194 = v185;
              v113 = *(_DWORD *)(a1 + 400);
              ++*(_QWORD *)(a1 + 384);
              v114 = *(_DWORD *)(a1 + 408);
              v115 = v113 + 1;
              v2 = (unsigned int)(4 * (v113 + 1));
              if ( (unsigned int)v2 >= 3 * v114 )
              {
                v114 *= 2;
              }
              else if ( v114 - *(_DWORD *)(a1 + 404) - v115 > v114 >> 3 )
              {
LABEL_135:
                *(_DWORD *)(a1 + 400) = v115;
                if ( *(_QWORD *)v112 != -4096 || *(_DWORD *)(v112 + 8) != -1 || !*(_BYTE *)(v112 + 12) )
                  --*(_DWORD *)(a1 + 404);
                *(_QWORD *)v112 = v206;
                *(_DWORD *)(v112 + 8) = v207;
                v116 = BYTE4(v207);
                *(_OWORD *)(v112 + 16) = 0;
                *(_BYTE *)(v112 + 12) = v116;
                *(_OWORD *)(v112 + 32) = 0;
                *(_OWORD *)(v112 + 48) = 0;
LABEL_138:
                *(_DWORD *)(v112 + 16) = v111;
                v117 = v112 + 16;
                *(_DWORD *)(v117 + 16) = v104;
                *(_QWORD *)(v117 + 8) = v172;
                *(_DWORD *)(v117 + 20) = v165;
                *(_BYTE *)(v117 + 24) = v181;
                *(_QWORD *)(v117 + 32) = v99;
                *(_DWORD *)(v117 + 40) = v101;
                if ( (_BYTE *)v203[0] != v204 )
                  j_j___libc_free_0(v203[0]);
                if ( (_BYTE *)v201[0] != v202 )
                  j_j___libc_free_0(v201[0]);
                if ( v198 != v200 )
                  _libc_free((unsigned __int64)v198);
                goto LABEL_24;
              }
              sub_2AC7CC0(v168, v114);
              sub_2ABE520(v168, (__int64 *)&v206, &v194);
              v111 = v177;
              v115 = *(_DWORD *)(a1 + 400) + 1;
              v112 = (unsigned __int64)v194;
              goto LABEL_135;
            }
            v121 = sub_9B78C0((__int64)v183, *(__int64 **)(a1 + 456));
            v197 = v183;
            v122 = v121;
            v123 = (__int64)v206;
            v124 = v207;
            LODWORD(v198) = (_DWORD)v184;
            BYTE4(v198) = BYTE4(v184);
            v16 = (unsigned __int8)sub_2ABE520(v168, (__int64 *)&v197, &v185) == 0;
            v125 = (unsigned __int64)v185;
            if ( !v16 )
              goto LABEL_172;
            v194 = v185;
            v126 = *(_DWORD *)(a1 + 400);
            ++*(_QWORD *)(a1 + 384);
            v127 = v126 + 1;
            v128 = *(_DWORD *)(a1 + 408);
            if ( 4 * v127 >= 3 * v128 )
            {
              v128 *= 2;
            }
            else if ( v128 - *(_DWORD *)(a1 + 404) - v127 > v128 >> 3 )
            {
LABEL_169:
              *(_DWORD *)(a1 + 400) = v127;
              if ( *(_QWORD *)v125 != -4096 || *(_DWORD *)(v125 + 8) != -1 || !*(_BYTE *)(v125 + 12) )
                --*(_DWORD *)(a1 + 404);
              *(_QWORD *)v125 = v197;
              *(_DWORD *)(v125 + 8) = (_DWORD)v198;
              v129 = BYTE4(v198);
              *(_OWORD *)(v125 + 16) = 0;
              *(_BYTE *)(v125 + 12) = v129;
              *(_OWORD *)(v125 + 32) = 0;
              *(_OWORD *)(v125 + 48) = 0;
LABEL_172:
              v130 = v125 + 16;
              *(_DWORD *)v130 = 7;
              *(_QWORD *)(v130 + 8) = 0;
              *(_DWORD *)(v130 + 16) = v122;
              *(_BYTE *)(v130 + 24) = 0;
              *(_QWORD *)(v130 + 32) = v123;
              *(_DWORD *)(v130 + 40) = v124;
              goto LABEL_24;
            }
            sub_2AC7CC0(v168, v128);
            sub_2ABE520(v168, (__int64 *)&v197, &v194);
            v127 = *(_DWORD *)(a1 + 400) + 1;
            v125 = (unsigned __int64)v194;
            goto LABEL_169;
          }
        }
      }
      v64 = BYTE4(v184);
      v14 = BYTE4(v184);
      if ( BYTE4(v184) )
        break;
      if ( v13 != 1 )
      {
        i = a1;
        v65 = *(unsigned int *)(a1 + 184);
        v10 = *(_QWORD *)(a1 + 168);
        if ( (_DWORD)v65 )
        {
          v66 = 37 * v13;
LABEL_71:
          v67 = 1;
          for ( i = ((_DWORD)v65 - 1) & (unsigned int)v66; ; i = ((_DWORD)v65 - 1) & v69 )
          {
            v68 = v10 + 72LL * (unsigned int)i;
            if ( *(_DWORD *)v68 == v13 && BYTE4(v184) == *(_BYTE *)(v68 + 4) )
            {
              v10 += 72LL * (unsigned int)i;
              goto LABEL_14;
            }
            if ( *(_DWORD *)v68 == -1 && *(_BYTE *)(v68 + 4) )
              break;
            v69 = v67 + i;
            ++v67;
          }
          v64 = 9 * v65;
          v10 += 72 * v65;
        }
        goto LABEL_14;
      }
LABEL_17:
      LODWORD(v207) = v13;
      BYTE4(v207) = v14;
      v15 = a1 + 384;
      v206 = v183;
      v16 = (unsigned __int8)sub_2ABE520(a1 + 384, (__int64 *)&v206, &v194) == 0;
      v17 = (__int64)v194;
      if ( v16 )
      {
        v197 = v194;
        v18 = *(_DWORD *)(a1 + 400);
        v19 = *(_DWORD *)(a1 + 408);
        ++*(_QWORD *)(a1 + 384);
        v20 = v18 + 1;
        if ( 4 * (v18 + 1) >= 3 * v19 )
        {
          v19 *= 2;
        }
        else if ( v19 - *(_DWORD *)(a1 + 404) - v20 > v19 >> 3 )
        {
          goto LABEL_20;
        }
        sub_2AC7CC0(v15, v19);
        sub_2ABE520(v15, (__int64 *)&v206, &v197);
        v20 = *(_DWORD *)(a1 + 400) + 1;
        v17 = (__int64)v197;
LABEL_20:
        *(_DWORD *)(a1 + 400) = v20;
        if ( *(_QWORD *)v17 != -4096 || *(_DWORD *)(v17 + 8) != -1 || !*(_BYTE *)(v17 + 12) )
          --*(_DWORD *)(a1 + 404);
        *(_QWORD *)v17 = v206;
        *(_DWORD *)(v17 + 8) = v207;
        v21 = BYTE4(v207);
        *(_OWORD *)(v17 + 16) = 0;
        *(_BYTE *)(v17 + 12) = v21;
        *(_OWORD *)(v17 + 32) = 0;
        *(_OWORD *)(v17 + 48) = 0;
      }
      *(_DWORD *)(v17 + 16) = 5;
      v22 = v17 + 16;
      *(_QWORD *)(v22 + 8) = 0;
      *(_DWORD *)(v22 + 16) = 0;
      *(_BYTE *)(v22 + 24) = 0;
      *(_QWORD *)(v22 + 32) = v175;
      *(_DWORD *)(v22 + 40) = v180;
LABEL_24:
      if ( v191 != (__int64 *)v193 )
        _libc_free((unsigned __int64)v191);
      if ( v188 != v190 )
        _libc_free((unsigned __int64)v188);
LABEL_28:
      v6 = (_QWORD *)v6[1];
      if ( (_QWORD *)v178 == v6 )
        goto LABEL_29;
    }
    v65 = *(unsigned int *)(a1 + 184);
    v10 = *(_QWORD *)(a1 + 168);
    if ( (_DWORD)v65 )
    {
      v66 = 37 * v13 - 1;
      goto LABEL_71;
    }
LABEL_14:
    if ( !(unsigned __int8)sub_B19060(v10 + 8, (__int64)v183, v64, i) )
      goto LABEL_50;
    v13 = (int)v184;
LABEL_16:
    v14 = BYTE4(v184);
    goto LABEL_17;
  }
  return result;
}
