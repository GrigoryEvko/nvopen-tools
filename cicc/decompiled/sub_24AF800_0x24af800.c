// Function: sub_24AF800
// Address: 0x24af800
//
void __fastcall sub_24AF800(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        char a9,
        unsigned __int8 a10,
        char a11,
        char a12)
{
  _QWORD *v12; // r15
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int8 v20; // al
  bool v21; // zf
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r15
  unsigned __int64 v27; // rax
  int v28; // edx
  unsigned __int64 v29; // rax
  bool v30; // cf
  __int64 v31; // rdx
  __int64 v32; // r14
  __int64 *v33; // rdi
  unsigned __int64 v34; // r12
  unsigned int v35; // r13d
  __int64 v36; // rax
  int v37; // esi
  __int64 v38; // r9
  int v39; // esi
  unsigned int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // rdi
  __int64 *v43; // r10
  __int64 v44; // r10
  __int64 v45; // rdx
  __int64 v46; // rsi
  char v47; // al
  unsigned __int64 v48; // rcx
  __int64 v49; // rax
  unsigned __int64 v50; // rsi
  unsigned __int64 v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rbx
  __int64 v54; // r9
  __int64 v55; // rdi
  __int64 v56; // rdx
  _QWORD *v57; // rax
  _QWORD *v58; // rdx
  __int64 v59; // r8
  unsigned int v60; // ecx
  int v61; // r10d
  int v62; // r11d
  unsigned __int64 *v63; // rbx
  unsigned __int64 *v64; // r8
  __int64 v65; // r14
  unsigned __int64 *v66; // r15
  __int64 v67; // r13
  unsigned __int64 *v68; // rax
  unsigned __int64 *v69; // r12
  unsigned __int64 *v70; // r8
  __int64 v71; // rcx
  unsigned __int64 *v72; // r14
  unsigned __int64 *v73; // rax
  unsigned __int64 v74; // rdx
  unsigned __int64 *v75; // rax
  unsigned __int64 v76; // rdx
  unsigned __int64 v77; // rdi
  unsigned __int64 *v78; // rbx
  __int64 **v79; // r12
  __int64 **v80; // rbx
  __int64 *v81; // rax
  __int64 v82; // rdi
  __int64 **v83; // r12
  __int64 **v84; // rbx
  __int64 *v85; // rax
  __int64 v86; // rsi
  __int64 v87; // rax
  __int64 *v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // rax
  __int64 v91; // r12
  __int64 v92; // r14
  __int64 v93; // rax
  __int64 v94; // rbx
  __int64 v95; // r13
  __int64 v96; // rdx
  int v97; // eax
  __int64 v98; // rax
  unsigned __int64 v99; // rdi
  unsigned __int64 *p_src; // rdi
  __int64 v101; // rcx
  unsigned __int64 v102; // rdx
  unsigned __int64 v103; // rsi
  unsigned __int64 *v104; // rdi
  unsigned __int64 v105; // rdx
  __int64 v106; // rcx
  unsigned __int64 v107; // rsi
  __int64 v108; // rax
  unsigned __int64 v109; // rax
  __int64 v110; // r12
  int v111; // ebx
  unsigned int j; // r13d
  __int64 v113; // rax
  __int64 v114; // rdi
  __int64 v115; // rcx
  __int64 v116; // rax
  unsigned int v117; // esi
  __int64 *v118; // rdx
  __int64 v119; // r10
  __int64 v120; // rax
  int v121; // r14d
  unsigned int v122; // r15d
  __int64 v123; // r13
  int v124; // r12d
  unsigned int v125; // ebx
  __int64 v126; // rsi
  char *v127; // rsi
  __int64 v128; // rdx
  __int64 v129; // rax
  __int64 v130; // rax
  unsigned __int64 v131; // rax
  int v132; // eax
  __int64 v133; // rsi
  int v134; // edx
  int v135; // r8d
  __m128i v136; // rax
  __int64 v137; // rsi
  __int64 v138; // rax
  __int64 v139; // rbx
  unsigned __int8 *v140; // rax
  size_t v141; // rdx
  void *v142; // rdi
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // r12
  __int64 v151; // rbx
  unsigned __int64 v152; // rdi
  unsigned __int64 *v153; // rax
  unsigned __int64 v154; // rdi
  __int64 v155; // rax
  unsigned __int64 v156; // rdi
  __int64 v157; // rdx
  char *v158; // rsi
  __int64 v159; // rdi
  __int64 v160; // rdx
  __int64 v161; // rcx
  __int64 v162; // r9
  unsigned __int8 *v163; // rdi
  __int64 v164; // rdx
  _BYTE *v165; // rsi
  __int64 v166; // rdx
  __int64 v167; // rcx
  __int64 v168; // r8
  __int64 v169; // r9
  unsigned __int64 *v170; // rdi
  __int64 v171; // rdx
  size_t v172; // rcx
  __int64 v173; // rsi
  __int64 v174; // rcx
  _QWORD *v175; // rdi
  __int64 v176; // rdx
  __int64 v177; // r8
  __int64 v178; // r9
  __int64 v179; // r12
  __int64 v180; // rax
  __int64 v181; // rdx
  __int64 v182; // rcx
  _QWORD *v183; // r13
  _QWORD *v184; // r14
  int i; // eax
  int v186; // ecx
  __int64 v187; // rdx
  __int64 v188; // rdx
  size_t v189; // rdx
  __int64 v190; // rax
  __int64 v191; // rcx
  __int64 v192; // rdx
  __int64 v193; // rsi
  __int64 v194; // [rsp+0h] [rbp-230h]
  unsigned __int64 *v195; // [rsp+8h] [rbp-228h]
  unsigned __int64 *v197; // [rsp+20h] [rbp-210h]
  __int64 *v199; // [rsp+30h] [rbp-200h]
  __int64 v201; // [rsp+40h] [rbp-1F0h]
  _QWORD *v202; // [rsp+48h] [rbp-1E8h]
  __int64 v203; // [rsp+50h] [rbp-1E0h]
  unsigned __int64 v204; // [rsp+50h] [rbp-1E0h]
  unsigned __int64 v205; // [rsp+58h] [rbp-1D8h]
  __int64 v206; // [rsp+60h] [rbp-1D0h]
  __int64 v207; // [rsp+70h] [rbp-1C0h]
  __int64 v208; // [rsp+78h] [rbp-1B8h]
  __int64 v209; // [rsp+80h] [rbp-1B0h]
  int v210; // [rsp+8Ch] [rbp-1A4h]
  unsigned __int64 v211; // [rsp+90h] [rbp-1A0h]
  __int64 v212; // [rsp+98h] [rbp-198h]
  unsigned __int64 v213; // [rsp+A0h] [rbp-190h]
  void *v214; // [rsp+A0h] [rbp-190h]
  unsigned __int64 v215; // [rsp+A8h] [rbp-188h]
  __int64 v216; // [rsp+A8h] [rbp-188h]
  __int64 v217; // [rsp+B0h] [rbp-180h]
  __int64 v218; // [rsp+B0h] [rbp-180h]
  __int64 v219; // [rsp+B0h] [rbp-180h]
  unsigned __int64 v220; // [rsp+B8h] [rbp-178h]
  __int64 v221; // [rsp+C0h] [rbp-170h]
  __int64 v222; // [rsp+C0h] [rbp-170h]
  char v223; // [rsp+C0h] [rbp-170h]
  __int64 v224; // [rsp+C0h] [rbp-170h]
  __int64 v225; // [rsp+C0h] [rbp-170h]
  size_t v226; // [rsp+C0h] [rbp-170h]
  _QWORD *v228; // [rsp+C8h] [rbp-168h]
  unsigned __int64 *v229; // [rsp+C8h] [rbp-168h]
  _QWORD *v230; // [rsp+C8h] [rbp-168h]
  __int64 v231[2]; // [rsp+D0h] [rbp-160h] BYREF
  _BYTE v232[16]; // [rsp+E0h] [rbp-150h] BYREF
  void *v233[2]; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v234; // [rsp+100h] [rbp-130h] BYREF
  __int64 v235[2]; // [rsp+110h] [rbp-120h] BYREF
  _QWORD v236[2]; // [rsp+120h] [rbp-110h] BYREF
  unsigned __int64 *v237; // [rsp+130h] [rbp-100h] BYREF
  size_t v238; // [rsp+138h] [rbp-F8h]
  _QWORD v239[2]; // [rsp+140h] [rbp-F0h] BYREF
  __m128i v240; // [rsp+150h] [rbp-E0h] BYREF
  char *v241; // [rsp+160h] [rbp-D0h]
  __int16 v242; // [rsp+170h] [rbp-C0h]
  __m128i v243[2]; // [rsp+180h] [rbp-B0h] BYREF
  __int16 v244; // [rsp+1A0h] [rbp-90h]
  __m128i v245; // [rsp+1B0h] [rbp-80h] BYREF
  unsigned __int64 src; // [rsp+1C0h] [rbp-70h] BYREF
  __int64 v247; // [rsp+1C8h] [rbp-68h]
  __int64 v248; // [rsp+1D0h] [rbp-60h]
  unsigned int v249; // [rsp+1D8h] [rbp-58h]
  __int64 v250; // [rsp+1E0h] [rbp-50h]
  __int64 v251; // [rsp+1E8h] [rbp-48h]
  __int64 v252; // [rsp+1F0h] [rbp-40h]
  int v253; // [rsp+1F8h] [rbp-38h]

  v12 = a1;
  *a1 = a2;
  *((_BYTE *)a1 + 8) = a9;
  a1[2] = a4;
  v199 = a1 + 3;
  sub_24DAB80(a1 + 3);
  a1[4] = a3;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  v15 = (_QWORD *)sub_22077B0(0x48u);
  v16 = v15 + 9;
  a1[5] = (__int64)v15;
  a1[6] = (__int64)v15;
  a1[7] = (__int64)(v15 + 9);
  do
  {
    if ( v15 )
    {
      *v15 = 0;
      v15[1] = 0;
      v15[2] = 0;
    }
    v15 += 3;
  }
  while ( v15 != v16 );
  a1[6] = (__int64)v15;
  v202 = a1 + 8;
  a1[9] = 0;
  a1[8] = a2;
  a1[10] = 0;
  *((_BYTE *)a1 + 120) = a12;
  v197 = (unsigned __int64 *)(a1 + 18);
  a1[16] = (__int64)(a1 + 18);
  v195 = (unsigned __int64 *)(a1 + 22);
  a1[20] = (__int64)(a1 + 22);
  v212 = (__int64)(a1 + 26);
  v17 = *a1;
  *((_DWORD *)a1 + 22) = 0;
  a1[26] = v17;
  a1[12] = 0;
  a1[13] = 0;
  a1[14] = 0;
  a1[17] = 0;
  *((_BYTE *)a1 + 144) = 0;
  a1[21] = 0;
  *((_BYTE *)a1 + 176) = 0;
  a1[25] = 0;
  a1[27] = 0;
  a1[28] = 0;
  a1[29] = 0;
  a1[30] = 0;
  a1[31] = 0;
  a1[32] = 0;
  *((_DWORD *)a1 + 66) = 0;
  *((_BYTE *)a1 + 272) = 0;
  a1[35] = a6;
  a1[36] = a7;
  *((_BYTE *)a1 + 304) = a10;
  a1[37] = a8;
  *((_BYTE *)a1 + 305) = a11;
  v18 = *(_QWORD *)(v17 + 80);
  v19 = v18 - 24;
  if ( !v18 )
    v19 = 0;
  v208 = v19;
  if ( a7 )
  {
    v203 = sub_FDC4B0(a7);
    v20 = *((_BYTE *)a1 + 304);
  }
  else
  {
    v203 = 2;
    v20 = a10;
  }
  v21 = v20 == 0;
  v22 = 0;
  if ( v21 )
    v22 = v203;
  v204 = v22;
  v194 = sub_24A7690(v212, 0, v208, v22);
  v23 = *(_QWORD *)(v208 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v23 == v208 + 48 )
    goto LABEL_255;
  if ( !v23 )
LABEL_59:
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 > 0xA || !(unsigned int)sub_B46E30(v23 - 24) )
  {
LABEL_255:
    sub_24A7690(v212, v208, 0, v204);
    goto LABEL_74;
  }
  v24 = a1[26];
  v211 = 0;
  v206 = v24 + 72;
  v207 = *(_QWORD *)(v24 + 80);
  if ( v207 == v24 + 72 )
  {
    v217 = 0;
    v220 = 0;
    v209 = 0;
LABEL_162:
    if ( 2 * v220 < 3 * v211 )
    {
      *(_QWORD *)(v217 + 16) = v211;
      *(_QWORD *)(v209 + 16) = v220 + 1;
    }
    goto LABEL_74;
  }
  v220 = 0;
  v209 = 0;
  v201 = 0;
  v217 = 0;
  v205 = 0;
  do
  {
    if ( !v207 )
LABEL_274:
      BUG();
    v26 = v207 - 24;
    v27 = *(_QWORD *)(v207 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v27 == v207 + 24 )
    {
      v32 = 0;
    }
    else
    {
      if ( !v27 )
        BUG();
      v28 = *(unsigned __int8 *)(v27 - 24);
      v29 = v27 - 24;
      v30 = (unsigned int)(v28 - 30) < 0xB;
      v31 = 0;
      if ( v30 )
        v31 = v29;
      v32 = v31;
    }
    v215 = 2;
    v33 = (__int64 *)a1[36];
    if ( v33 )
      v215 = sub_FDD860(v33, v207 - 24);
    v210 = sub_B46E30(v32);
    if ( v210 )
    {
      v34 = 2;
      v35 = 0;
      while ( 1 )
      {
        v53 = sub_B46EC0(v32, v35);
        v223 = sub_D0E970(v32, v35, 0);
        v54 = v215;
        if ( v223 )
        {
          v54 = -1;
          if ( v215 <= 0x4189374BC6A7EELL )
            v54 = 1000 * v215;
        }
        v213 = v54;
        v55 = a1[35];
        if ( v55 )
        {
          v245.m128i_i32[0] = sub_FF0430(v55, v26, v53);
          v34 = sub_F02E20((unsigned int *)&v245, v213);
          if ( !*((_BYTE *)a1 + 305) )
            goto LABEL_35;
        }
        else if ( !*((_BYTE *)a1 + 305) )
        {
          goto LABEL_37;
        }
        v36 = a1[37];
        v37 = *(_DWORD *)(v36 + 24);
        v38 = *(_QWORD *)(v36 + 8);
        if ( !v37 )
          goto LABEL_35;
        v39 = v37 - 1;
        v40 = v39 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
        v41 = (__int64 *)(v38 + 16LL * v40);
        v42 = *v41;
        v43 = v41;
        if ( v53 != *v41 )
        {
          v59 = *v41;
          v60 = v39 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
          v61 = 1;
          while ( v59 != -4096 )
          {
            v62 = v61 + 1;
            v60 = v39 & (v61 + v60);
            v43 = (__int64 *)(v38 + 16LL * v60);
            v59 = *v43;
            if ( v53 == *v43 )
              goto LABEL_33;
            v61 = v62;
          }
          goto LABEL_35;
        }
LABEL_33:
        v44 = v43[1];
        if ( !v44 || v53 != **(_QWORD **)(v44 + 32) )
          goto LABEL_35;
        if ( v53 != v42 )
        {
          for ( i = 1; ; i = v186 )
          {
            if ( v42 == -4096 )
              BUG();
            v186 = i + 1;
            v40 = v39 & (i + v40);
            v41 = (__int64 *)(v38 + 16LL * v40);
            v42 = *v41;
            if ( v53 == *v41 )
              break;
          }
        }
        v56 = v41[1];
        if ( *(_BYTE *)(v56 + 84) )
        {
          v57 = *(_QWORD **)(v56 + 64);
          v58 = &v57[*(unsigned int *)(v56 + 76)];
          if ( v57 != v58 )
          {
            while ( v26 != *v57 )
            {
              if ( v58 == ++v57 )
                goto LABEL_65;
            }
LABEL_35:
            if ( !v34 )
              v34 = 1;
            goto LABEL_37;
          }
LABEL_65:
          v34 = 1;
        }
        else
        {
          if ( sub_C8CA60(v56 + 56, v26) )
            goto LABEL_35;
          v34 = 1;
        }
LABEL_37:
        v45 = sub_24A7690(v212, v26, v53, v34);
        v46 = *(_QWORD *)(v45 + 8);
        *(_BYTE *)(v45 + 26) = v223;
        if ( v46 )
        {
          v221 = v45;
          v47 = sub_F35EF0(*(_QWORD *)v45, v46);
          v45 = v221;
          if ( v47 )
            *(_BYTE *)(v221 + 25) = 1;
        }
        if ( v208 == v26 )
        {
          v48 = v220;
          v49 = v217;
          if ( v34 > v220 )
          {
            v48 = v34;
            v49 = v45;
          }
          v220 = v48;
          v217 = v49;
        }
        v50 = *(_QWORD *)(v53 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v50 != v53 + 48 )
        {
          if ( !v50 )
            goto LABEL_59;
          v222 = v45;
          if ( (unsigned int)*(unsigned __int8 *)(v50 - 24) - 30 <= 0xA && !(unsigned int)sub_B46E30(v50 - 24) )
          {
            v51 = v211;
            v52 = v222;
            if ( v34 > v211 )
              v51 = v34;
            else
              v52 = v209;
            v209 = v52;
            v211 = v51;
          }
        }
        if ( v210 == ++v35 )
          goto LABEL_19;
      }
    }
    *((_BYTE *)a1 + 272) = 1;
    v25 = sub_24A7690(v212, v26, 0, v215);
    if ( v215 > v205 )
    {
      v205 = v215;
      v201 = v25;
    }
LABEL_19:
    v207 = *(_QWORD *)(v207 + 8);
  }
  while ( v206 != v207 );
  v12 = a1;
  if ( v204 < v205 || 2 * v204 >= 3 * v205 )
  {
    if ( v220 < v211 )
      goto LABEL_74;
    goto LABEL_162;
  }
  *(_QWORD *)(v194 + 16) = v205;
  *(_QWORD *)(v201 + 16) = v204 + 1;
  if ( v220 >= v211 )
    goto LABEL_162;
LABEL_74:
  v63 = (unsigned __int64 *)v12[28];
  v64 = (unsigned __int64 *)v12[27];
  if ( (char *)v63 - (char *)v64 <= 0 )
  {
LABEL_249:
    v69 = 0;
    sub_24A4A70(v64, v63);
  }
  else
  {
    v228 = v12;
    v65 = (__int64)(v12[28] - (_QWORD)v64) >> 3;
    v66 = (unsigned __int64 *)v12[27];
    while ( 1 )
    {
      v67 = v65;
      v68 = (unsigned __int64 *)sub_2207800(8 * v65);
      v69 = v68;
      if ( v68 )
        break;
      v65 >>= 1;
      if ( !v65 )
      {
        v64 = v66;
        v12 = v228;
        goto LABEL_249;
      }
    }
    v70 = v66;
    v71 = v65;
    v72 = &v68[v67];
    v12 = v228;
    *v68 = *v70;
    v73 = v68 + 1;
    *v70 = 0;
    if ( v72 == v69 + 1 )
    {
      v75 = v69;
    }
    else
    {
      do
      {
        v74 = *(v73 - 1);
        *(v73++ - 1) = 0;
        *(v73 - 1) = v74;
      }
      while ( v72 != v73 );
      v75 = &v69[v67 - 1];
    }
    v76 = *v75;
    *v75 = 0;
    v77 = *v70;
    *v70 = v76;
    if ( v77 )
    {
      v224 = v71;
      v229 = v70;
      j_j___libc_free_0(v77);
      sub_24A68E0(v229, v63, v69, v224);
    }
    else
    {
      sub_24A68E0(v70, v63, v69, v71);
    }
    v78 = v69;
    do
    {
      if ( *v78 )
        j_j___libc_free_0(*v78);
      ++v78;
    }
    while ( v72 != v78 );
  }
  j_j___libc_free_0((unsigned __int64)v69);
  v79 = (__int64 **)v12[28];
  v80 = (__int64 **)v12[27];
  if ( v80 != v79 )
  {
    do
    {
      v81 = *v80;
      if ( !*((_BYTE *)*v80 + 25) )
      {
        if ( *((_BYTE *)v81 + 26) )
        {
          v82 = v81[1];
          if ( v82 )
          {
            if ( sub_AA5E90(v82) && (unsigned __int8)sub_24A43F0(v212, **v80, (*v80)[1]) )
              *((_BYTE *)*v80 + 24) = 1;
          }
        }
      }
      ++v80;
    }
    while ( v79 != v80 );
    v83 = (__int64 **)v12[27];
    v84 = (__int64 **)v12[28];
    if ( v83 != v84 )
    {
      do
      {
        v85 = *v83;
        if ( !*((_BYTE *)*v83 + 25) )
        {
          v86 = *v85;
          if ( *((_BYTE *)v12 + 272) || v86 )
          {
            if ( (unsigned __int8)sub_24A43F0(v212, v86, v85[1]) )
              *((_BYTE *)*v83 + 24) = 1;
          }
        }
        ++v83;
      }
      while ( v84 != v83 );
      v87 = v12[28];
      v88 = (__int64 *)v12[27];
      if ( (unsigned __int64)(v87 - (_QWORD)v88) > 8 && a10 )
      {
        v89 = *v88;
        *v88 = *(_QWORD *)(v87 - 8);
        *(_QWORD *)(v87 - 8) = v89;
      }
    }
  }
  v214 = v12 + 39;
  if ( a12 )
  {
    sub_315C560(&v245, a2, a10);
    v147 = v245.m128i_i64[0];
    v12[41] = 1;
    v12[39] = v147;
    LOBYTE(v147) = v245.m128i_i8[8];
    v12[45] = 1;
    *((_BYTE *)v12 + 320) = v147;
    v148 = v247;
    *((_BYTE *)v12 + 392) = 1;
    v12[42] = v148;
    ++src;
    v12[43] = v248;
    ++v250;
    *((_DWORD *)v12 + 88) = v249;
    v247 = 0;
    v12[46] = v251;
    v248 = 0;
    v12[47] = v252;
    v249 = 0;
    *((_DWORD *)v12 + 96) = v253;
    v251 = 0;
    v252 = 0;
    v253 = 0;
    sub_C7D6A0(0, 0, 8);
    v149 = v249;
    if ( v249 )
    {
      v150 = v247;
      v151 = v247 + 88LL * v249;
      do
      {
        if ( *(_QWORD *)v150 != -8192 && *(_QWORD *)v150 != -4096 )
        {
          v152 = *(_QWORD *)(v150 + 40);
          if ( v152 != v150 + 56 )
            _libc_free(v152);
          sub_C7D6A0(*(_QWORD *)(v150 + 16), 8LL * *(unsigned int *)(v150 + 32), 8);
        }
        v150 += 88;
      }
      while ( v151 != v150 );
      v149 = v249;
    }
    sub_C7D6A0(v247, 88 * v149, 8);
    if ( *((_BYTE *)v12 + 392) && (_BYTE)qword_4FEB408 )
      sub_315DEA0(v214, 0);
  }
  else
  {
    memset(v214, 0, 0x58u);
  }
  v90 = v12[8];
  v12[9] = 0;
  if ( v90 + 72 != *(_QWORD *)(v90 + 80) )
  {
    v91 = v90 + 72;
    v92 = *(_QWORD *)(v90 + 80);
    do
    {
      v93 = v92;
      v92 = *(_QWORD *)(v92 + 8);
      v94 = *(_QWORD *)(v93 + 32);
      v95 = v93 + 24;
LABEL_108:
      while ( v95 != v94 )
      {
        while ( 1 )
        {
          v96 = v94;
          v94 = *(_QWORD *)(v94 + 8);
          v97 = *(unsigned __int8 *)(v96 - 24);
          if ( v97 == 86 )
            break;
          if ( (unsigned int)(v97 - 29) <= 0x39 )
          {
            if ( (unsigned int)(v97 - 30) > 0x37 )
              goto LABEL_274;
            goto LABEL_108;
          }
          if ( (unsigned int)(v97 - 87) > 9 )
            goto LABEL_274;
          if ( v95 == v94 )
            goto LABEL_113;
        }
        if ( (_BYTE)qword_4FEBC88
          && !(_BYTE)qword_4FEB5C8
          && !*((_BYTE *)v12 + 120)
          && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v96 - 120) + 8LL) + 8LL) - 17 > 1 )
        {
          v132 = *((_DWORD *)v12 + 19);
          v133 = v96 - 24;
          if ( v132 == 1 )
          {
            sub_24AAB30((__int64)v202, v133);
          }
          else if ( v132 == 2 )
          {
            sub_24ADFC0(v202, v133);
          }
          else
          {
            if ( v132 )
              goto LABEL_274;
            ++*((_DWORD *)v12 + 18);
          }
        }
      }
LABEL_113:
      ;
    }
    while ( v91 != v92 );
  }
  sub_24DB2C0(&v245, v199, 1);
  v98 = v12[5];
  v99 = *(_QWORD *)(v98 + 24);
  *(__m128i *)(v98 + 24) = v245;
  *(_QWORD *)(v98 + 40) = src;
  v245 = 0u;
  src = 0;
  if ( v99 )
  {
    j_j___libc_free_0(v99);
    if ( v245.m128i_i64[0] )
      j_j___libc_free_0(v245.m128i_u64[0]);
  }
  if ( !a9 )
  {
    sub_24DB2C0(&v245, v199, 0);
    v153 = (unsigned __int64 *)v12[5];
    v154 = *v153;
    *(__m128i *)v153 = v245;
    v153[2] = src;
    v245 = 0u;
    src = 0;
    if ( v154 )
    {
      j_j___libc_free_0(v154);
      if ( v245.m128i_i64[0] )
        j_j___libc_free_0(v245.m128i_u64[0]);
    }
    if ( LOBYTE(qword_4F8A568[8]) )
    {
      sub_24DB2C0(&v245, v199, 2);
      v155 = v12[5];
      v156 = *(_QWORD *)(v155 + 48);
      *(__m128i *)(v155 + 48) = v245;
      *(_QWORD *)(v155 + 64) = src;
      v245 = 0u;
      src = 0;
      if ( v156 )
      {
        j_j___libc_free_0(v156);
        if ( v245.m128i_i64[0] )
          j_j___libc_free_0(v245.m128i_u64[0]);
      }
    }
  }
  sub_ED29C0(v245.m128i_i64, *v12, 0);
  p_src = (unsigned __int64 *)v12[16];
  if ( (unsigned __int64 *)v245.m128i_i64[0] == &src )
  {
    v188 = v245.m128i_i64[1];
    if ( v245.m128i_i64[1] )
    {
      if ( v245.m128i_i64[1] == 1 )
        *(_BYTE *)p_src = src;
      else
        memcpy(p_src, &src, v245.m128i_u64[1]);
      v188 = v245.m128i_i64[1];
      p_src = (unsigned __int64 *)v12[16];
    }
    v12[17] = v188;
    *((_BYTE *)p_src + v188) = 0;
    p_src = (unsigned __int64 *)v245.m128i_i64[0];
  }
  else
  {
    v101 = v245.m128i_i64[1];
    v102 = src;
    if ( v197 == p_src )
    {
      v12[16] = v245.m128i_i64[0];
      v12[17] = v101;
      v12[18] = v102;
    }
    else
    {
      v103 = v12[18];
      v12[16] = v245.m128i_i64[0];
      v12[17] = v101;
      v12[18] = v102;
      if ( p_src )
      {
        v245.m128i_i64[0] = (__int64)p_src;
        src = v103;
        goto LABEL_122;
      }
    }
    v245.m128i_i64[0] = (__int64)&src;
    p_src = &src;
  }
LABEL_122:
  v245.m128i_i64[1] = 0;
  *(_BYTE *)p_src = 0;
  if ( (unsigned __int64 *)v245.m128i_i64[0] != &src )
    j_j___libc_free_0(v245.m128i_u64[0]);
  sub_ED2A00(v245.m128i_i64, *v12, 0);
  v104 = (unsigned __int64 *)v12[20];
  if ( (unsigned __int64 *)v245.m128i_i64[0] == &src )
  {
    v187 = v245.m128i_i64[1];
    if ( v245.m128i_i64[1] )
    {
      if ( v245.m128i_i64[1] == 1 )
        *(_BYTE *)v104 = src;
      else
        memcpy(v104, &src, v245.m128i_u64[1]);
      v187 = v245.m128i_i64[1];
      v104 = (unsigned __int64 *)v12[20];
    }
    v12[21] = v187;
    *((_BYTE *)v104 + v187) = 0;
    v104 = (unsigned __int64 *)v245.m128i_i64[0];
  }
  else
  {
    v105 = src;
    v106 = v245.m128i_i64[1];
    if ( v195 == v104 )
    {
      v12[20] = v245.m128i_i64[0];
      v12[21] = v106;
      v12[22] = v105;
    }
    else
    {
      v107 = v12[22];
      v12[20] = v245.m128i_i64[0];
      v12[21] = v106;
      v12[22] = v105;
      if ( v104 )
      {
        v245.m128i_i64[0] = (__int64)v104;
        src = v107;
        goto LABEL_128;
      }
    }
    v245.m128i_i64[0] = (__int64)&src;
    v104 = &src;
  }
LABEL_128:
  v245.m128i_i64[1] = 0;
  *(_BYTE *)v104 = 0;
  if ( (unsigned __int64 *)v245.m128i_i64[0] != &src )
    j_j___libc_free_0(v245.m128i_u64[0]);
  v108 = *v12;
  v245 = 0u;
  src = 0;
  LODWORD(v237) = -1;
  v216 = v108 + 72;
  v218 = *(_QWORD *)(v108 + 80);
  if ( v218 == v108 + 72 )
  {
    v128 = 0;
    v127 = 0;
  }
  else
  {
    do
    {
      if ( !v218 )
        goto LABEL_274;
      v109 = *(_QWORD *)(v218 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v109 != v218 + 24 )
      {
        if ( !v109 )
          goto LABEL_59;
        v110 = v109 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v109 - 24) - 30 <= 0xA )
        {
          v111 = sub_B46E30(v110);
          if ( v111 )
          {
            for ( j = 0; j != v111; ++j )
            {
              v113 = sub_B46EC0(v110, j);
              v114 = v12[31];
              v115 = v113;
              v116 = *((unsigned int *)v12 + 66);
              if ( (_DWORD)v116 )
              {
                v117 = (v116 - 1) & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
                v118 = (__int64 *)(v114 + 16LL * v117);
                v119 = *v118;
                if ( v115 == *v118 )
                {
LABEL_139:
                  if ( v118 != (__int64 *)(v114 + 16 * v116) )
                  {
                    v120 = v118[1];
                    if ( v120 )
                    {
                      v230 = v12;
                      v121 = 0;
                      v122 = j;
                      v123 = v110;
                      v124 = v111;
                      v125 = *(_DWORD *)(v120 + 8);
                      do
                      {
                        v126 = v245.m128i_i64[1];
                        v243[0].m128i_i8[0] = v125 >> v121;
                        if ( v245.m128i_i64[1] == src )
                        {
                          sub_C8FB10((__int64)&v245, (const void *)v245.m128i_i64[1], v243[0].m128i_i8);
                        }
                        else
                        {
                          if ( v245.m128i_i64[1] )
                          {
                            *(_BYTE *)v245.m128i_i64[1] = v125 >> v121;
                            v126 = v245.m128i_i64[1];
                          }
                          v245.m128i_i64[1] = v126 + 1;
                        }
                        v121 += 8;
                      }
                      while ( v121 != 32 );
                      v111 = v124;
                      v110 = v123;
                      j = v122;
                      v12 = v230;
                    }
                  }
                }
                else
                {
                  v134 = 1;
                  while ( v119 != -4096 )
                  {
                    v135 = v134 + 1;
                    v117 = (v116 - 1) & (v134 + v117);
                    v118 = (__int64 *)(v114 + 16LL * v117);
                    v119 = *v118;
                    if ( v115 == *v118 )
                      goto LABEL_139;
                    v134 = v135;
                  }
                }
              }
            }
          }
        }
      }
      v218 = *(_QWORD *)(v218 + 8);
    }
    while ( v216 != v218 );
    v127 = (char *)v245.m128i_i64[0];
    v128 = v245.m128i_i64[1] - v245.m128i_i64[0];
  }
  sub_1098F90((unsigned int *)&v237, v127, v128);
  v129 = *((unsigned int *)v12 + 18);
  v240.m128i_i32[0] = -1;
  v243[0].m128i_i64[0] = v129;
  sub_1098F90((unsigned int *)&v240, v243[0].m128i_i8, 8);
  v243[0].m128i_i64[0] = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(v12[5] + 8LL) - *(_QWORD *)v12[5]) >> 3);
  sub_1098F90((unsigned int *)&v240, v243[0].m128i_i8, 8);
  v243[0].m128i_i64[0] = 0xAAAAAAAAAAAAAAABLL
                       * ((__int64)(*(_QWORD *)(v12[5] + 32LL) - *(_QWORD *)(v12[5] + 24LL)) >> 3);
  sub_1098F90((unsigned int *)&v240, v243[0].m128i_i8, 8);
  if ( *((_BYTE *)v12 + 392) )
    v130 = sub_3158260(v214);
  else
    v130 = (__int64)(v12[28] - v12[27]) >> 3;
  v243[0].m128i_i64[0] = v130;
  sub_1098F90((unsigned int *)&v240, v243[0].m128i_i8, 8);
  v131 = ((unsigned int)v237 + ((unsigned __int64)v240.m128i_u32[0] << 28)) & 0xFFFFFFFFFFFFFFFLL;
  v21 = *((_BYTE *)v12 + 8) == 0;
  v12[25] = v131;
  if ( !v21 )
    v12[25] = v131 | 0x1000000000000000LL;
  if ( sub_2241AC0((__int64)&qword_4FEADC8, "-") )
  {
    v136.m128i_i64[0] = (__int64)sub_BD5D20(*v12);
    v137 = qword_4FEADC8;
    v243[0] = v136;
    if ( sub_C931B0(v243[0].m128i_i64, (_WORD *)qword_4FEADC8, qword_4FEADD0, 0) != -1 )
    {
      v138 = sub_C5F790((__int64)v243, v137);
      v139 = sub_904010(v138, "Funcname=");
      v140 = (unsigned __int8 *)sub_BD5D20(*v12);
      v142 = *(void **)(v139 + 32);
      if ( v141 > *(_QWORD *)(v139 + 24) - (_QWORD)v142 )
      {
        v139 = sub_CB6200(v139, v140, v141);
      }
      else if ( v141 )
      {
        v226 = v141;
        memcpy(v142, v140, v141);
        *(_QWORD *)(v139 + 32) += v226;
      }
      v143 = sub_904010(v139, ", Hash=");
      v144 = sub_CB59D0(v143, v12[25]);
      v145 = sub_904010(v144, " in building ");
      v146 = sub_CB6200(
               v145,
               *(unsigned __int8 **)(*(_QWORD *)(*v12 + 40LL) + 200LL),
               *(_QWORD *)(*(_QWORD *)(*v12 + 40LL) + 208LL));
      sub_904010(v146, "\n");
    }
  }
  if ( v245.m128i_i64[0] )
    j_j___libc_free_0(v245.m128i_u64[0]);
  if ( *(_QWORD *)(a4 + 24) && (unsigned __int8)sub_24ABC70(*v12, (_QWORD *)v12[2]) )
  {
    v158 = (char *)sub_BD5D20(*v12);
    if ( v158 )
    {
      v231[0] = (__int64)v232;
      sub_24A2F70(v231, v158, (__int64)&v158[v157]);
    }
    else
    {
      v232[0] = 0;
      v231[0] = (__int64)v232;
      v231[1] = 0;
    }
    v244 = 267;
    v159 = *v12;
    v243[0].m128i_i64[0] = (__int64)(v12 + 25);
    v240.m128i_i64[0] = (__int64)sub_BD5D20(v159);
    v240.m128i_i64[1] = v160;
    v242 = 773;
    v241 = ".";
    sub_9C6370(&v245, &v240, v243, v161, 773, v162);
    sub_CA0F50((__int64 *)v233, (void **)&v245);
    v163 = (unsigned __int8 *)*v12;
    LOWORD(v248) = 260;
    v245.m128i_i64[0] = (__int64)v233;
    sub_BD6B50(v163, (const char **)&v245);
    v164 = *v12;
    LOWORD(v248) = 260;
    v245.m128i_i64[0] = (__int64)v231;
    sub_B305A0(4, (__int64)&v245, v164);
    v165 = (_BYTE *)v12[16];
    v166 = (__int64)&v165[v12[17]];
    v235[0] = (__int64)v236;
    v244 = 267;
    v243[0].m128i_i64[0] = (__int64)(v12 + 25);
    sub_24A3020(v235, v165, v166);
    if ( v235[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)v235, ".", 1u);
    v242 = 260;
    v240.m128i_i64[0] = (__int64)v235;
    sub_9C6370(&v245, &v240, v243, v167, v168, v169);
    sub_CA0F50((__int64 *)&v237, (void **)&v245);
    v170 = (unsigned __int64 *)v12[16];
    if ( v237 == v239 )
    {
      v189 = v238;
      if ( v238 )
      {
        if ( v238 == 1 )
          *(_BYTE *)v170 = v239[0];
        else
          memcpy(v170, v239, v238);
        v189 = v238;
        v170 = (unsigned __int64 *)v12[16];
      }
      v12[17] = v189;
      *((_BYTE *)v170 + v189) = 0;
      v170 = v237;
    }
    else
    {
      v171 = v239[0];
      v172 = v238;
      if ( v197 == v170 )
      {
        v12[16] = v237;
        v12[17] = v172;
        v12[18] = v171;
      }
      else
      {
        v173 = v12[18];
        v12[16] = v237;
        v12[17] = v172;
        v12[18] = v171;
        if ( v170 )
        {
          v237 = v170;
          v239[0] = v173;
          goto LABEL_220;
        }
      }
      v237 = v239;
      v170 = v239;
    }
LABEL_220:
    v238 = 0;
    *(_BYTE *)v170 = 0;
    if ( v237 != v239 )
      j_j___libc_free_0((unsigned __int64)v237);
    if ( (_QWORD *)v235[0] != v236 )
      j_j___libc_free_0(v235[0]);
    v174 = *(_QWORD *)(*v12 + 48LL);
    if ( v174 )
    {
      v175 = *(_QWORD **)(*v12 + 48LL);
      v219 = *(_QWORD *)(*v12 + 40LL);
      v244 = 267;
      v235[0] = v174;
      v225 = v174;
      v243[0].m128i_i64[0] = (__int64)(v12 + 25);
      v240.m128i_i64[0] = sub_AA8810(v175);
      v240.m128i_i64[1] = v176;
      v242 = 773;
      v241 = ".";
      sub_9C6370(&v245, &v240, v243, 773, v177, v178);
      sub_CA0F50((__int64 *)&v237, (void **)&v245);
      v179 = sub_BAA410(v219, v237, v238);
      *(_DWORD *)(v179 + 8) = *(_DWORD *)(v225 + 8);
      v180 = sub_24ABBE0((_QWORD *)v12[2], (unsigned __int64 *)v235);
      v183 = (_QWORD *)v181;
      v184 = (_QWORD *)v180;
      if ( v180 != v181 )
      {
        do
        {
          sub_B2F990(v184[2], v179, v181, v182);
          v184 = (_QWORD *)*v184;
        }
        while ( v183 != v184 );
      }
      if ( v237 != v239 )
        j_j___libc_free_0((unsigned __int64)v237);
      if ( v233[0] != &v234 )
        j_j___libc_free_0((unsigned __int64)v233[0]);
      if ( (_BYTE *)v231[0] != v232 )
        j_j___libc_free_0(v231[0]);
    }
    else
    {
      v190 = sub_BAA410(*(_QWORD *)(*v12 + 40LL), v233[0], (size_t)v233[1]);
      v192 = *v12;
      v193 = v190;
      LOBYTE(v190) = *(_BYTE *)(*v12 + 32LL) & 0xF0 | 3;
      *(_BYTE *)(*v12 + 32LL) = v190;
      if ( (v190 & 0x30) != 0 )
        *(_BYTE *)(v192 + 33) |= 0x40u;
      sub_B2F990(*v12, v193, v192, v191);
      sub_2240A30((unsigned __int64 *)v233);
      sub_2240A30((unsigned __int64 *)v231);
    }
  }
  if ( a5 )
    v12[24] = sub_ED18C0(*v12, (char *)v12[16], v12[17]);
}
