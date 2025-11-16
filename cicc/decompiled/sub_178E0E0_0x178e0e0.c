// Function: sub_178E0E0
// Address: 0x178e0e0
//
__int64 __fastcall sub_178E0E0(
        __int64 *a1,
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
  _QWORD *v10; // rdx
  unsigned int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // r8
  unsigned int v14; // esi
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r12
  __int64 i; // r14
  _QWORD *v24; // rax
  int v25; // r8d
  int v26; // r9d
  char v27; // dl
  _QWORD *v28; // r15
  char v29; // al
  __int64 v30; // rdi
  int v31; // r8d
  _QWORD *v32; // r9
  _QWORD *v33; // r15
  __int64 v34; // rax
  _QWORD *v35; // r15
  __int64 v36; // rax
  __int64 v37; // r15
  _QWORD *v38; // rax
  _QWORD *v39; // rsi
  _QWORD *v40; // rcx
  __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // rax
  __int64 v46; // r14
  __int64 ***v47; // r15
  __int64 v48; // rbx
  _QWORD *v49; // rax
  double v50; // xmm4_8
  double v51; // xmm5_8
  __m128i *v52; // r15
  __int64 v53; // r12
  __int64 v54; // rdi
  __int64 ***v55; // r13
  __int64 v56; // r12
  __int64 v57; // r14
  __int64 v58; // rbx
  _QWORD *v59; // rax
  double v60; // xmm4_8
  double v61; // xmm5_8
  __int64 v62; // rax
  unsigned int v63; // r12d
  __int64 v64; // r15
  __int64 v65; // r14
  __int64 v66; // r13
  __int64 v67; // rbx
  _QWORD *v68; // rax
  double v69; // xmm4_8
  double v70; // xmm5_8
  __int64 v71; // rsi
  __int64 v72; // r14
  __int64 v73; // rbx
  _QWORD *v74; // rax
  double v75; // xmm4_8
  double v76; // xmm5_8
  char v77; // al
  __int64 v78; // rdx
  _QWORD *v79; // rdx
  int v80; // r13d
  __int64 v81; // rax
  int v82; // r13d
  __int64 v83; // rbx
  __int64 v84; // rcx
  __int64 v85; // r13
  __int64 v86; // r8
  unsigned int v87; // r15d
  __int64 v88; // rdx
  __int64 v89; // r9
  __int64 *v90; // r14
  __int64 v91; // rcx
  __int64 v92; // r15
  int v93; // eax
  __int64 v94; // rax
  int v95; // edx
  __int64 v96; // rdx
  __int64 *v97; // rax
  __int64 v98; // rcx
  unsigned __int64 v99; // rsi
  __int64 v100; // rcx
  __int64 v101; // rax
  __int64 v102; // rcx
  char v103; // al
  __int64 *v104; // rdi
  int v105; // eax
  __int64 v106; // rax
  __int64 *v107; // r15
  __int64 v108; // rdx
  __int64 v109; // rcx
  __int64 v110; // r8
  __int64 v111; // r9
  int v112; // eax
  __int64 v113; // rax
  int v114; // ecx
  __int64 v115; // rcx
  __int64 *v116; // rax
  __int64 v117; // rsi
  unsigned __int64 v118; // rdi
  __int64 v119; // rsi
  unsigned __int64 v120; // rax
  __int64 *v121; // r8
  __int64 v122; // rsi
  __int64 v123; // rsi
  unsigned __int8 *v124; // rsi
  __int64 v125; // rdi
  unsigned __int8 *v126; // rax
  __int64 v127; // rdx
  __int64 v128; // rcx
  __int64 v129; // r8
  __int64 v130; // r9
  unsigned __int8 *v131; // r15
  int v132; // eax
  __int64 v133; // rax
  int v134; // edx
  __int64 v135; // rdx
  unsigned __int8 **v136; // rax
  unsigned __int8 *v137; // rcx
  unsigned __int64 v138; // rsi
  __int64 v139; // rcx
  __int64 v140; // rax
  __int64 v141; // rcx
  __int64 v142; // rdx
  __int64 v143; // r13
  int v144; // r8d
  int v145; // r9d
  __int64 v146; // rsi
  char *v147; // rax
  char *v148; // rdx
  __int64 v149; // r13
  __int64 v150; // rax
  __int64 *v151; // rax
  int v152; // eax
  __int64 v153; // rax
  int v154; // edx
  __int64 v155; // rdx
  __int64 *v156; // rax
  __int64 v157; // rcx
  unsigned __int64 v158; // rsi
  __int64 v159; // rcx
  __int64 v160; // rax
  _QWORD *v161; // rdx
  _QWORD *v162; // rax
  __int64 v163; // r11
  __int64 v164; // rax
  unsigned int v165; // ecx
  unsigned int v166; // edx
  int v167; // r13d
  unsigned __int64 v168; // rdx
  int v169; // esi
  __int64 v170; // rdi
  int v171; // esi
  signed __int64 v172; // rsi
  __int64 *v173; // [rsp+8h] [rbp-328h]
  __int64 *v174; // [rsp+8h] [rbp-328h]
  __int64 v175; // [rsp+8h] [rbp-328h]
  __int64 v176; // [rsp+18h] [rbp-318h]
  __int64 v177; // [rsp+20h] [rbp-310h]
  __m128i *v178; // [rsp+28h] [rbp-308h]
  __int64 **v179; // [rsp+38h] [rbp-2F8h]
  int v180; // [rsp+44h] [rbp-2ECh]
  __int64 v181; // [rsp+48h] [rbp-2E8h]
  __int64 v182; // [rsp+50h] [rbp-2E0h]
  _QWORD *v183; // [rsp+58h] [rbp-2D8h]
  __int64 v184; // [rsp+58h] [rbp-2D8h]
  __int64 v185; // [rsp+58h] [rbp-2D8h]
  unsigned int v186; // [rsp+60h] [rbp-2D0h]
  int v187; // [rsp+60h] [rbp-2D0h]
  _QWORD v190[2]; // [rsp+80h] [rbp-2B0h] BYREF
  __m128i v191; // [rsp+90h] [rbp-2A0h] BYREF
  __int64 v192; // [rsp+A0h] [rbp-290h]
  _QWORD *v193; // [rsp+B0h] [rbp-280h] BYREF
  __int16 v194; // [rsp+C0h] [rbp-270h]
  __m128 v195; // [rsp+D0h] [rbp-260h] BYREF
  __int64 v196; // [rsp+E0h] [rbp-250h]
  __int64 v197; // [rsp+F0h] [rbp-240h] BYREF
  _QWORD *v198; // [rsp+F8h] [rbp-238h]
  __int64 v199; // [rsp+100h] [rbp-230h]
  unsigned int v200; // [rsp+108h] [rbp-228h]
  __int64 v201; // [rsp+110h] [rbp-220h] BYREF
  __int64 v202; // [rsp+118h] [rbp-218h]
  __int64 v203; // [rsp+120h] [rbp-210h]
  int v204; // [rsp+128h] [rbp-208h]
  _QWORD *v205; // [rsp+130h] [rbp-200h] BYREF
  __int64 v206; // [rsp+138h] [rbp-1F8h]
  _QWORD v207[8]; // [rsp+140h] [rbp-1F0h] BYREF
  __int64 v208; // [rsp+180h] [rbp-1B0h] BYREF
  _QWORD *v209; // [rsp+188h] [rbp-1A8h]
  _QWORD *v210; // [rsp+190h] [rbp-1A0h]
  __int64 v211; // [rsp+198h] [rbp-198h]
  int v212; // [rsp+1A0h] [rbp-190h]
  _QWORD v213[9]; // [rsp+1A8h] [rbp-188h] BYREF
  void *base; // [rsp+1F0h] [rbp-140h] BYREF
  __int64 v215; // [rsp+1F8h] [rbp-138h]
  _BYTE v216[304]; // [rsp+200h] [rbp-130h] BYREF

  v10 = v207;
  v11 = 0;
  base = v216;
  v215 = 0x1000000000LL;
  v209 = v213;
  v210 = v213;
  v205 = v207;
  v207[0] = a2;
  v212 = 0;
  v213[0] = a2;
  v208 = 1;
  v206 = 0x800000001LL;
  v211 = 0x100000008LL;
  v12 = 0;
  while ( 1 )
  {
    v13 = v10[v12];
    v14 = *(_DWORD *)(v13 + 20) & 0xFFFFFFF;
    if ( v14 )
    {
      v15 = 3LL * v14;
      v16 = 8LL * v14;
      v17 = v13 - 8 * v15;
      v18 = 0;
      do
      {
        v19 = v17;
        if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
          v19 = *(_QWORD *)(v13 - 8);
        v20 = *(_QWORD *)(v19 + 3 * v18);
        if ( *(_BYTE *)(v20 + 16) == 29
          && *(_QWORD *)(v20 + 40) == *(_QWORD *)(v18 + v19 + 24LL * *(unsigned int *)(v13 + 56) + 8) )
        {
          goto LABEL_9;
        }
        v18 += 8;
      }
      while ( v18 != v16 );
    }
    for ( i = *(_QWORD *)(v13 + 8); i; i = *(_QWORD *)(i + 8) )
    {
      v28 = sub_1648700(i);
      v29 = *((_BYTE *)v28 + 16);
      switch ( v29 )
      {
        case 'M':
          v24 = v209;
          if ( v210 == v209 )
          {
            v39 = &v209[HIDWORD(v211)];
            if ( v209 != v39 )
            {
              v40 = 0;
              while ( v28 != (_QWORD *)*v24 )
              {
                if ( *v24 == -2 )
                  v40 = v24;
                if ( v39 == ++v24 )
                {
                  if ( !v40 )
                    goto LABEL_51;
                  *v40 = v28;
                  --v212;
                  ++v208;
                  goto LABEL_45;
                }
              }
              continue;
            }
LABEL_51:
            if ( HIDWORD(v211) < (unsigned int)v211 )
            {
              ++HIDWORD(v211);
              *v39 = v28;
              ++v208;
LABEL_45:
              v41 = (unsigned int)v206;
              if ( (unsigned int)v206 >= HIDWORD(v206) )
              {
                sub_16CD150((__int64)&v205, v207, 0, 8, v25, v26);
                v41 = (unsigned int)v206;
              }
              v205[v41] = v28;
              LODWORD(v206) = v206 + 1;
              continue;
            }
          }
          sub_16CCBA0((__int64)&v208, (__int64)v28);
          if ( v27 )
            goto LABEL_45;
          break;
        case '<':
          v42 = v11;
          v43 = (unsigned int)v215;
          if ( (unsigned int)v215 >= HIDWORD(v215) )
          {
            sub_16CD150((__int64)&base, v216, 0, 16, v25, v11);
            v43 = (unsigned int)v215;
            v42 = v11;
          }
          v44 = (char *)base + 16 * v43;
          *v44 = v42;
          v44[1] = v28;
          LODWORD(v215) = v215 + 1;
          break;
        case '0':
          v30 = v28[1];
          if ( !v30 )
            goto LABEL_9;
          if ( *(_QWORD *)(v30 + 8) )
            goto LABEL_9;
          v32 = sub_1648700(v30);
          if ( *((_BYTE *)v32 + 16) != 60 )
            goto LABEL_9;
          v33 = (*((_BYTE *)v28 + 23) & 0x40) != 0
              ? (_QWORD *)*(v28 - 1)
              : &v28[-3 * (*((_DWORD *)v28 + 5) & 0xFFFFFFF)];
          v34 = v33[3];
          if ( *(_BYTE *)(v34 + 16) != 13 )
            goto LABEL_9;
          v35 = *(_QWORD **)(v34 + 24);
          if ( *(_DWORD *)(v34 + 32) > 0x40u )
            v35 = (_QWORD *)*v35;
          v36 = (unsigned int)v215;
          v37 = v11 | ((_QWORD)v35 << 32);
          if ( (unsigned int)v215 >= HIDWORD(v215) )
          {
            v183 = v32;
            sub_16CD150((__int64)&base, v216, 0, 16, v31, (int)v32);
            v36 = (unsigned int)v215;
            v32 = v183;
          }
          v38 = (char *)base + 16 * v36;
          *v38 = v37;
          v38[1] = v32;
          LODWORD(v215) = v215 + 1;
          break;
        default:
          goto LABEL_9;
      }
    }
    v12 = v11 + 1;
    v11 = v12;
    if ( (_DWORD)v12 == (_DWORD)v206 )
      break;
    v10 = v205;
  }
  v180 = v215;
  if ( (_DWORD)v215 )
  {
    if ( (unsigned int)v215 == 1 )
    {
      v197 = 0;
      v198 = 0;
      v199 = 0;
      v200 = 0;
      v201 = 0;
      v202 = 0;
      v203 = 0;
      v204 = 0;
    }
    else
    {
      qsort(base, (16LL * (unsigned int)v215) >> 4, 0x10u, (__compar_fn_t)sub_17890F0);
      v197 = 0;
      v198 = 0;
      v180 = v215;
      v199 = 0;
      v200 = 0;
      v201 = 0;
      v202 = 0;
      v203 = 0;
      v204 = 0;
      if ( !(_DWORD)v215 )
      {
LABEL_73:
        v62 = sub_1599EF0(*(__int64 ***)a2);
        v184 = v62;
        v187 = v206;
        if ( (_DWORD)v206 != 1 )
        {
          v63 = 1;
          v64 = v62;
          do
          {
            v65 = v205[v63];
            v66 = *(_QWORD *)(v65 + 8);
            if ( v66 )
            {
              v67 = *a1;
              do
              {
                v68 = sub_1648700(v66);
                sub_170B990(v67, (__int64)v68);
                v66 = *(_QWORD *)(v66 + 8);
              }
              while ( v66 );
              v71 = v64;
              if ( v65 == v64 )
                v71 = sub_1599EF0(*(__int64 ***)v65);
              sub_164D160(v65, v71, a3, a4, a5, a6, v69, v70, a9, a10);
            }
            ++v63;
          }
          while ( v63 != v187 );
        }
        v72 = *(_QWORD *)(a2 + 8);
        v21 = a2;
        if ( v72 )
        {
          v73 = *a1;
          do
          {
            v74 = sub_1648700(v72);
            sub_170B990(v73, (__int64)v74);
            v72 = *(_QWORD *)(v72 + 8);
          }
          while ( v72 );
          if ( a2 == v184 )
            v184 = sub_1599EF0(*(__int64 ***)a2);
          sub_164D160(a2, v184, a3, a4, a5, a6, v75, v76, a9, a10);
        }
        else
        {
          v21 = 0;
        }
        j___libc_free_0(v202);
        j___libc_free_0(v198);
        goto LABEL_10;
      }
    }
    v52 = (__m128i *)&v195;
    v186 = 0;
    while ( 1 )
    {
      v181 = 16LL * v186;
      v53 = v205[*(unsigned int *)((char *)base + v181)];
      LODWORD(v182) = *(_DWORD *)((char *)base + v181 + 4);
      v54 = **(_QWORD **)((char *)base + v181 + 8);
      v195.m128_i32[2] = v182;
      v195.m128_u64[0] = v53;
      v179 = (__int64 **)v54;
      v195.m128_i32[3] = sub_1643030(v54);
      v55 = (__int64 ***)sub_17894D0((__int64)&v201, v52)[2];
      if ( !v55 )
        break;
LABEL_66:
      v56 = *(_QWORD *)((char *)base + v181 + 8);
      v57 = *(_QWORD *)(v56 + 8);
      if ( v57 )
      {
        v58 = *a1;
        do
        {
          v59 = sub_1648700(v57);
          sub_170B990(v58, (__int64)v59);
          v57 = *(_QWORD *)(v57 + 8);
        }
        while ( v57 );
        if ( v55 == (__int64 ***)v56 )
          v55 = (__int64 ***)sub_1599EF0(*v55);
        sub_164D160(v56, (__int64)v55, a3, a4, a5, a6, v60, v61, a9, a10);
      }
      if ( v180 == ++v186 )
        goto LABEL_73;
    }
    v194 = 265;
    LODWORD(v193) = v182;
    v190[0] = sub_1649960(v53);
    v191.m128i_i64[0] = (__int64)v190;
    v191.m128i_i64[1] = (__int64)".off";
    v77 = v194;
    v190[1] = v78;
    LOWORD(v192) = 773;
    if ( (_BYTE)v194 )
    {
      if ( (_BYTE)v194 == 1 )
      {
        a3 = (__m128)_mm_load_si128(&v191);
        v195 = a3;
        v196 = v192;
      }
      else
      {
        if ( HIBYTE(v194) == 1 )
        {
          v79 = v193;
        }
        else
        {
          v79 = &v193;
          v77 = 2;
        }
        v195.m128_u64[1] = (unsigned __int64)v79;
        v195.m128_u64[0] = (unsigned __int64)&v191;
        LOBYTE(v196) = 2;
        BYTE1(v196) = v77;
      }
    }
    else
    {
      LOWORD(v196) = 256;
    }
    v80 = *(_DWORD *)(v53 + 20);
    v81 = sub_1648B60(64);
    v82 = v80 & 0xFFFFFFF;
    v83 = v81;
    if ( v81 )
    {
      sub_15F1EA0(v81, v54, 53, 0, 0, v53);
      *(_DWORD *)(v83 + 56) = v82;
      sub_164B780(v83, v52->m128i_i64);
      sub_1648880(v83, *(_DWORD *)(v83 + 56), 1);
    }
    if ( (*(_DWORD *)(v53 + 20) & 0xFFFFFFF) != 0 )
    {
      v178 = v52;
      v185 = 0;
      v177 = 8LL * (*(_DWORD *)(v53 + 20) & 0xFFFFFFF);
      while ( 1 )
      {
        v103 = *(_BYTE *)(v53 + 23) & 0x40;
        v84 = v103 ? *(_QWORD *)(v53 - 8) : v53 - 24LL * (*(_DWORD *)(v53 + 20) & 0xFFFFFFF);
        v85 = *(_QWORD *)(v185 + v84 + 24LL * *(unsigned int *)(v53 + 56) + 8);
        if ( !v200 )
          break;
        v86 = (__int64)v198;
        v87 = ((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4);
        v88 = (v200 - 1) & v87;
        v89 = 16 * v88;
        v90 = &v198[2 * v88];
        v91 = *v90;
        if ( v85 != *v90 )
        {
          v104 = 0;
          v89 = 1;
          while ( v91 != -8 )
          {
            if ( v91 == -16 && !v104 )
              v104 = v90;
            v88 = (v200 - 1) & ((_DWORD)v89 + (_DWORD)v88);
            v90 = &v198[2 * (unsigned int)v88];
            v91 = *v90;
            if ( v85 == *v90 )
              goto LABEL_101;
            v89 = (unsigned int)(v89 + 1);
          }
          if ( v104 )
            v90 = v104;
          ++v197;
          v105 = v199 + 1;
          if ( 4 * ((int)v199 + 1) < 3 * v200 )
          {
            v88 = v200 - HIDWORD(v199) - v105;
            v91 = v200 >> 3;
            if ( (unsigned int)v88 <= (unsigned int)v91 )
            {
              sub_141A900((__int64)&v197, v200);
              if ( !v200 )
                goto LABEL_265;
              v86 = v200 - 1;
              v91 = 0;
              v88 = (unsigned int)v86 & v87;
              v169 = 1;
              v89 = 16 * v88;
              v105 = v199 + 1;
              v90 = &v198[2 * v88];
              v170 = *v90;
              if ( v85 != *v90 )
              {
                while ( v170 != -8 )
                {
                  if ( v170 == -16 && !v91 )
                    v91 = (__int64)v90;
                  v89 = (unsigned int)(v169 + 1);
                  v88 = (unsigned int)v86 & (v169 + (_DWORD)v88);
                  v90 = &v198[2 * (unsigned int)v88];
                  v170 = *v90;
                  if ( v85 == *v90 )
                    goto LABEL_124;
                  ++v169;
                }
                goto LABEL_220;
              }
            }
            goto LABEL_124;
          }
LABEL_224:
          sub_141A900((__int64)&v197, 2 * v200);
          if ( !v200 )
          {
LABEL_265:
            LODWORD(v199) = v199 + 1;
            BUG();
          }
          v88 = (v200 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
          v105 = v199 + 1;
          v89 = 16 * v88;
          v90 = &v198[2 * v88];
          v86 = *v90;
          if ( v85 != *v90 )
          {
            v91 = 0;
            v171 = 1;
            while ( v86 != -8 )
            {
              if ( !v91 && v86 == -16 )
                v91 = (__int64)v90;
              v89 = (unsigned int)(v171 + 1);
              v88 = (v200 - 1) & (v171 + (_DWORD)v88);
              v90 = &v198[2 * (unsigned int)v88];
              v86 = *v90;
              if ( v85 == *v90 )
                goto LABEL_124;
              ++v171;
            }
LABEL_220:
            if ( v91 )
              v90 = (__int64 *)v91;
          }
LABEL_124:
          LODWORD(v199) = v105;
          if ( *v90 != -8 )
            --HIDWORD(v199);
          *v90 = v85;
          v90[1] = 0;
          v103 = *(_BYTE *)(v53 + 23) & 0x40;
          goto LABEL_127;
        }
LABEL_101:
        v92 = v90[1];
        if ( v92 )
        {
          v93 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
          if ( v93 == *(_DWORD *)(v83 + 56) )
          {
            sub_15F55D0(v83, v200, v88, v91, (__int64)v198, v89);
            v93 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
          }
          v94 = (v93 + 1) & 0xFFFFFFF;
          v95 = v94 | *(_DWORD *)(v83 + 20) & 0xF0000000;
          *(_DWORD *)(v83 + 20) = v95;
          if ( (v95 & 0x40000000) != 0 )
            v96 = *(_QWORD *)(v83 - 8);
          else
            v96 = v83 - 24 * v94;
          v97 = (__int64 *)(v96 + 24LL * (unsigned int)(v94 - 1));
          if ( *v97 )
          {
            v98 = v97[1];
            v99 = v97[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v99 = v98;
            if ( v98 )
              *(_QWORD *)(v98 + 16) = v99 | *(_QWORD *)(v98 + 16) & 3LL;
          }
          *v97 = v92;
          v100 = *(_QWORD *)(v92 + 8);
          v97[1] = v100;
          if ( v100 )
            *(_QWORD *)(v100 + 16) = (unsigned __int64)(v97 + 1) | *(_QWORD *)(v100 + 16) & 3LL;
          v97[2] = (v92 + 8) | v97[2] & 3;
          *(_QWORD *)(v92 + 8) = v97;
          goto LABEL_112;
        }
LABEL_127:
        if ( v103 )
        {
          v106 = *(_QWORD *)(v53 - 8);
        }
        else
        {
          v88 = 24LL * (*(_DWORD *)(v53 + 20) & 0xFFFFFFF);
          v106 = v53 - v88;
        }
        v107 = *(__int64 **)(v106 + 3 * v185);
        if ( v107 && (__int64 *)v53 == v107 )
        {
          v90[1] = v83;
          v152 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
          if ( v152 == *(_DWORD *)(v83 + 56) )
          {
            sub_15F55D0(v83, 3 * v185, v88, v91, v86, v89);
            v152 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
          }
          v153 = (v152 + 1) & 0xFFFFFFF;
          v154 = v153 | *(_DWORD *)(v83 + 20) & 0xF0000000;
          *(_DWORD *)(v83 + 20) = v154;
          if ( (v154 & 0x40000000) != 0 )
            v155 = *(_QWORD *)(v83 - 8);
          else
            v155 = v83 - 24 * v153;
          v156 = (__int64 *)(v155 + 24LL * (unsigned int)(v153 - 1));
          if ( *v156 )
          {
            v157 = v156[1];
            v158 = v156[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v158 = v157;
            if ( v157 )
              *(_QWORD *)(v157 + 16) = v158 | *(_QWORD *)(v157 + 16) & 3LL;
          }
          *v156 = v83;
          v159 = *(_QWORD *)(v83 + 8);
          v156[1] = v159;
          if ( v159 )
            *(_QWORD *)(v159 + 16) = (unsigned __int64)(v156 + 1) | *(_QWORD *)(v159 + 16) & 3LL;
          v156[2] = v156[2] & 3 | (v83 + 8);
          *(_QWORD *)(v83 + 8) = v156;
        }
        else
        {
          v195.m128_u64[0] = v53;
          v195.m128_i32[2] = v182;
          v195.m128_i32[3] = sub_1643030((__int64)v179);
          v108 = sub_17894D0((__int64)&v201, v178)[2];
          if ( !v108 )
          {
            v173 = (__int64 *)a1[1];
            v120 = sub_157EBA0(v85);
            v121 = v173;
            v173[1] = *(_QWORD *)(v120 + 40);
            v173[2] = v120 + 24;
            v122 = *(_QWORD *)(v120 + 48);
            v195.m128_u64[0] = v122;
            if ( v122 )
            {
              sub_1623A60((__int64)v178, v122, 2);
              v121 = v173;
              v123 = *v173;
              if ( *v173 )
                goto LABEL_146;
LABEL_147:
              v124 = (unsigned __int8 *)v195.m128_u64[0];
              *v121 = v195.m128_u64[0];
              if ( v124 )
              {
                sub_1623210((__int64)v178, v124, (__int64)v121);
              }
              else if ( v195.m128_u64[0] )
              {
                sub_161E7C0((__int64)v178, v195.m128_i64[0]);
              }
            }
            else
            {
              v123 = *v173;
              if ( *v173 )
              {
LABEL_146:
                v174 = v121;
                sub_161E7C0((__int64)v121, v123);
                v121 = v174;
                goto LABEL_147;
              }
            }
            if ( (_DWORD)v182 )
            {
              LOWORD(v196) = 259;
              v163 = a1[1];
              v195.m128_u64[0] = (unsigned __int64)"extract";
              v175 = v163;
              v164 = sub_15A0680(*v107, (unsigned int)v182, 0);
              v107 = sub_172C310(v175, (__int64)v107, v164, v178->m128i_i64, 0, *(double *)a3.m128_u64, a4, a5);
            }
            v125 = a1[1];
            LOWORD(v196) = 259;
            v195.m128_u64[0] = (unsigned __int64)"extract.t";
            v126 = sub_1708970(v125, 36, (__int64)v107, v179, v178->m128i_i64);
            v90[1] = (__int64)v126;
            v131 = v126;
            v132 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
            if ( v132 == *(_DWORD *)(v83 + 56) )
            {
              sub_15F55D0(v83, 36, v127, v128, v129, v130);
              v132 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
            }
            v133 = (v132 + 1) & 0xFFFFFFF;
            v134 = v133 | *(_DWORD *)(v83 + 20) & 0xF0000000;
            *(_DWORD *)(v83 + 20) = v134;
            if ( (v134 & 0x40000000) != 0 )
              v135 = *(_QWORD *)(v83 - 8);
            else
              v135 = v83 - 24 * v133;
            v136 = (unsigned __int8 **)(v135 + 24LL * (unsigned int)(v133 - 1));
            if ( *v136 )
            {
              v137 = v136[1];
              v138 = (unsigned __int64)v136[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v138 = v137;
              if ( v137 )
                *((_QWORD *)v137 + 2) = v138 | *((_QWORD *)v137 + 2) & 3LL;
            }
            *v136 = v131;
            if ( v131 )
            {
              v139 = *((_QWORD *)v131 + 1);
              v136[1] = (unsigned __int8 *)v139;
              if ( v139 )
                *(_QWORD *)(v139 + 16) = (unsigned __int64)(v136 + 1) | *(_QWORD *)(v139 + 16) & 3LL;
              v136[2] = (unsigned __int8 *)((unsigned __int64)(v131 + 8) | (unsigned __int64)v136[2] & 3);
              *((_QWORD *)v131 + 1) = v136;
            }
            v140 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
            if ( (*(_BYTE *)(v83 + 23) & 0x40) != 0 )
              v141 = *(_QWORD *)(v83 - 8);
            else
              v141 = v83 - 24 * v140;
            *(_QWORD *)(v141 + 8LL * (unsigned int)(v140 - 1) + 24LL * *(unsigned int *)(v83 + 56) + 8) = v85;
            if ( (*(_BYTE *)(v53 + 23) & 0x40) != 0 )
              v142 = *(_QWORD *)(v53 - 8);
            else
              v142 = v53 - 24LL * (*(_DWORD *)(v53 + 20) & 0xFFFFFFF);
            v143 = *(_QWORD *)(v142 + 3 * v185);
            if ( !v143 )
              BUG();
            if ( *(_BYTE *)(v143 + 16) != 77 || !sub_17898B0((__int64)&v208, *(_QWORD *)(v142 + 3 * v185)) )
              goto LABEL_115;
            v146 = (unsigned int)v206;
            if ( (unsigned __int64)(unsigned int)v206 >> 2 )
            {
              v147 = (char *)v205;
              v148 = (char *)&v205[4 * ((unsigned __int64)(unsigned int)v206 >> 2)];
              while ( v143 != *(_QWORD *)v147 )
              {
                if ( v143 == *((_QWORD *)v147 + 1) )
                {
                  v147 += 8;
                  goto LABEL_176;
                }
                if ( v143 == *((_QWORD *)v147 + 2) )
                {
                  v147 += 16;
                  goto LABEL_176;
                }
                if ( v143 == *((_QWORD *)v147 + 3) )
                {
                  v147 += 24;
                  goto LABEL_176;
                }
                v147 += 32;
                if ( v148 == v147 )
                  goto LABEL_247;
              }
              goto LABEL_176;
            }
            v148 = (char *)v205;
LABEL_247:
            v147 = (char *)&v205[v146];
            v172 = (char *)&v205[v146] - v148;
            if ( v172 == 16 )
              goto LABEL_256;
            if ( v172 != 24 )
            {
              if ( v172 == 8 )
              {
LABEL_250:
                if ( v143 == *(_QWORD *)v148 )
                  v147 = v148;
              }
LABEL_176:
              v149 = (v182 << 32) | (unsigned int)((v147 - (char *)v205) >> 3);
              v150 = (unsigned int)v215;
              if ( (unsigned int)v215 >= HIDWORD(v215) )
              {
                sub_16CD150((__int64)&base, v216, 0, 16, v144, v145);
                v150 = (unsigned int)v215;
              }
              v151 = (__int64 *)((char *)base + 16 * v150);
              ++v180;
              *v151 = v149;
              v151[1] = (__int64)v131;
              LODWORD(v215) = v215 + 1;
              goto LABEL_115;
            }
            if ( v143 != *(_QWORD *)v148 )
            {
              v148 += 8;
LABEL_256:
              if ( v143 != *(_QWORD *)v148 )
              {
                v148 += 8;
                goto LABEL_250;
              }
            }
            v147 = v148;
            goto LABEL_176;
          }
          v90[1] = v108;
          v112 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
          if ( v112 == *(_DWORD *)(v83 + 56) )
          {
            v176 = v108;
            sub_15F55D0(v83, (__int64)v178, v108, v109, v110, v111);
            v108 = v176;
            v112 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
          }
          v113 = (v112 + 1) & 0xFFFFFFF;
          v114 = v113 | *(_DWORD *)(v83 + 20) & 0xF0000000;
          *(_DWORD *)(v83 + 20) = v114;
          if ( (v114 & 0x40000000) != 0 )
            v115 = *(_QWORD *)(v83 - 8);
          else
            v115 = v83 - 24 * v113;
          v116 = (__int64 *)(v115 + 24LL * (unsigned int)(v113 - 1));
          if ( *v116 )
          {
            v117 = v116[1];
            v118 = v116[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v118 = v117;
            if ( v117 )
              *(_QWORD *)(v117 + 16) = v118 | *(_QWORD *)(v117 + 16) & 3LL;
          }
          *v116 = v108;
          v119 = *(_QWORD *)(v108 + 8);
          v116[1] = v119;
          if ( v119 )
            *(_QWORD *)(v119 + 16) = (unsigned __int64)(v116 + 1) | *(_QWORD *)(v119 + 16) & 3LL;
          v116[2] = (v108 + 8) | v116[2] & 3;
          *(_QWORD *)(v108 + 8) = v116;
        }
LABEL_112:
        v101 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v83 + 23) & 0x40) != 0 )
          v102 = *(_QWORD *)(v83 - 8);
        else
          v102 = v83 - 24 * v101;
        *(_QWORD *)(v102 + 8LL * (unsigned int)(v101 - 1) + 24LL * *(unsigned int *)(v83 + 56) + 8) = v85;
LABEL_115:
        v185 += 8;
        if ( v177 == v185 )
        {
          v52 = v178;
          goto LABEL_194;
        }
      }
      ++v197;
      goto LABEL_224;
    }
LABEL_194:
    ++v197;
    if ( !(_DWORD)v199 )
    {
      if ( HIDWORD(v199) )
      {
        v160 = v200;
        if ( v200 <= 0x40 )
        {
LABEL_197:
          v161 = v198;
          v162 = &v198[2 * v160];
          if ( v198 != v162 )
          {
            do
            {
              *v161 = -8;
              v161 += 2;
            }
            while ( v162 != v161 );
          }
          v199 = 0;
          goto LABEL_200;
        }
        j___libc_free_0(v198);
        v198 = 0;
        v199 = 0;
        v200 = 0;
      }
LABEL_200:
      v195.m128_u64[0] = v53;
      v55 = (__int64 ***)v83;
      v195.m128_i32[2] = v182;
      v195.m128_i32[3] = sub_1643030((__int64)v179);
      sub_17894D0((__int64)&v201, v52)[2] = v83;
      goto LABEL_66;
    }
    v165 = 4 * v199;
    v160 = v200;
    if ( (unsigned int)(4 * v199) < 0x40 )
      v165 = 64;
    if ( v165 >= v200 )
      goto LABEL_197;
    if ( (_DWORD)v199 == 1 )
    {
      v167 = 64;
    }
    else
    {
      _BitScanReverse(&v166, v199 - 1);
      v167 = 1 << (33 - (v166 ^ 0x1F));
      if ( v167 < 64 )
        v167 = 64;
      if ( v167 == v200 )
        goto LABEL_216;
    }
    j___libc_free_0(v198);
    v168 = ((((((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
              | (4 * v167 / 3u + 1)
              | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
            | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
            | (4 * v167 / 3u + 1)
            | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
            | (4 * v167 / 3u + 1)
            | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
          | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
          | (4 * v167 / 3u + 1)
          | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 16;
    v200 = (v168
          | (((((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
              | (4 * v167 / 3u + 1)
              | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
            | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
            | (4 * v167 / 3u + 1)
            | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
            | (4 * v167 / 3u + 1)
            | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
          | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
          | (4 * v167 / 3u + 1)
          | ((4 * v167 / 3u + 1) >> 1))
         + 1;
    v198 = (_QWORD *)sub_22077B0(
                       16
                     * ((v168
                       | (((((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                           | (4 * v167 / 3u + 1)
                           | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
                         | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                         | (4 * v167 / 3u + 1)
                         | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 8)
                       | (((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                         | (4 * v167 / 3u + 1)
                         | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
                       | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                       | (4 * v167 / 3u + 1)
                       | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1))
                      + 1));
LABEL_216:
    sub_178CA70((__int64)&v197);
    goto LABEL_200;
  }
  v21 = a2;
  v45 = sub_1599EF0(*(__int64 ***)a2);
  v46 = *(_QWORD *)(a2 + 8);
  v47 = (__int64 ***)v45;
  if ( !v46 )
  {
LABEL_9:
    v21 = 0;
    goto LABEL_10;
  }
  v48 = *a1;
  do
  {
    v49 = sub_1648700(v46);
    sub_170B990(v48, (__int64)v49);
    v46 = *(_QWORD *)(v46 + 8);
  }
  while ( v46 );
  if ( v47 == (__int64 ***)a2 )
    v47 = (__int64 ***)sub_1599EF0(*v47);
  sub_164D160(a2, (__int64)v47, a3, a4, a5, a6, v50, v51, a9, a10);
LABEL_10:
  if ( v210 != v209 )
    _libc_free((unsigned __int64)v210);
  if ( v205 != v207 )
    _libc_free((unsigned __int64)v205);
  if ( base != v216 )
    _libc_free((unsigned __int64)base);
  return v21;
}
