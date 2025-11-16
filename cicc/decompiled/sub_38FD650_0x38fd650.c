// Function: sub_38FD650
// Address: 0x38fd650
//
__int64 __fastcall sub_38FD650(
        __int64 a1,
        __m128 a2,
        __m128i a3,
        __m128i a4,
        __int64 a5,
        unsigned __int64 *a6,
        unsigned int *a7,
        unsigned int *a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        _QWORD *a12,
        __int64 *a13,
        __int64 a14)
{
  __int64 v14; // r13
  size_t v15; // rdx
  __int64 v16; // r9
  int v17; // eax
  __int64 v18; // rbx
  int v19; // eax
  int v20; // r8d
  unsigned int v21; // edx
  int v22; // r12d
  char *v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // ebx
  int v26; // r15d
  _QWORD *v27; // r13
  __int64 v28; // rax
  __int64 (*v29)(); // rdx
  __int64 v30; // r14
  __int64 (*v31)(); // r12
  unsigned int v32; // eax
  int v33; // r8d
  int v34; // r9d
  int v35; // r12d
  __int64 v36; // rax
  _WORD *v37; // r8
  int v38; // eax
  _WORD *v39; // rcx
  unsigned __int64 v40; // rbx
  _QWORD *v41; // r12
  __int64 v42; // rdi
  __int64 v43; // rcx
  char *v44; // r8
  __int64 v45; // rsi
  char *v46; // rsi
  char *v47; // rbx
  __int64 v48; // rbx
  __int64 v49; // r14
  __int64 *v50; // r15
  unsigned __int64 *v51; // r14
  __int64 *v52; // r14
  __int64 *i; // r12
  __int64 v54; // r13
  __int64 *v55; // rdx
  __int64 v56; // r8
  __int64 v57; // rdx
  __int64 v58; // rcx
  unsigned int v59; // eax
  unsigned int v60; // ebx
  unsigned __int64 v61; // r15
  unsigned int v62; // r14d
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rdx
  unsigned __int64 *v65; // rbx
  __int64 v66; // r15
  unsigned int v67; // r14d
  unsigned int v68; // r12d
  __int64 v69; // rdi
  char v70; // cl
  __int64 v71; // rax
  unsigned __int64 *v72; // rsi
  unsigned int v73; // r12d
  __int64 v74; // rsi
  __int64 v75; // rdi
  char v76; // cl
  __int64 v77; // rdx
  unsigned __int64 *v78; // rsi
  __m128i *v79; // r9
  char *v80; // r14
  __int64 **v81; // rax
  __int64 v82; // rax
  char *v83; // r11
  __int64 v84; // rdx
  char *v85; // rsi
  char *v86; // r13
  __int64 v87; // rbx
  int v88; // r14d
  unsigned int v89; // edx
  __int64 (*v90)(); // rax
  __int64 v91; // r14
  __int64 v92; // rdx
  int v93; // r12d
  __int64 (*v94)(); // rax
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rax
  unsigned __int64 v99; // rdx
  __int64 v100; // r9
  __int64 (*v101)(); // rax
  __int64 v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rdx
  __m128 *v105; // rax
  unsigned int v106; // edx
  char *v107; // rax
  __int64 v108; // rax
  unsigned __int64 v109; // rdx
  int v110; // r9d
  __int64 (*v111)(); // rax
  __int64 v112; // rax
  __int64 v113; // rax
  unsigned __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  unsigned int v118; // edx
  __m128 *v119; // rax
  unsigned __int64 v120; // rbx
  __int64 v121; // rdi
  char *v122; // rax
  unsigned __int64 v123; // r14
  size_t v124; // r14
  __int64 v125; // rax
  _BYTE **v126; // r10
  void *v127; // rsi
  __int64 v128; // rax
  char *v129; // r14
  size_t v130; // rax
  size_t v131; // rdx
  _BYTE **v132; // rdi
  __int64 v133; // rax
  __int64 v134; // rax
  char *v135; // rsi
  size_t v136; // rdx
  unsigned __int64 v137; // rax
  _BYTE **v138; // r14
  size_t v139; // rdx
  void *v140; // rsi
  __int64 v141; // rax
  __int64 v142; // rax
  _BYTE **v143; // rdi
  signed __int64 v144; // r15
  size_t v145; // rdx
  unsigned int v146; // r12d
  unsigned __int64 *v147; // rbx
  unsigned __int64 *v148; // r13
  unsigned __int64 *v149; // rbx
  unsigned __int64 *v150; // r13
  __int64 v152; // r14
  _BYTE **p_p_dest; // rdi
  _BYTE **v154; // rdi
  __int64 v155; // rax
  __int64 v156; // r10
  __int64 v157; // r12
  unsigned __int64 v158; // rax
  __m128i v159; // xmm0
  __int64 *v160; // rdi
  __m128i *v161; // rax
  unsigned int v162; // r8d
  size_t v163; // rdx
  __m128i *v164; // rdi
  __int64 v165; // r12
  __int64 v166; // rbx
  unsigned __int64 v167; // rdi
  unsigned __int64 v168; // rbx
  __int64 v169; // rax
  _QWORD *v170; // r12
  __int64 v171; // rdi
  __int64 v172; // rax
  __int64 v173; // rax
  __int64 j; // rdx
  _BYTE **v175; // rdi
  __int64 v176; // r15
  __int64 k; // rax
  __int64 v178; // rax
  char *v179; // rsi
  size_t v180; // rdx
  unsigned int v181; // eax
  __int64 v182; // rcx
  unsigned __int8 v187; // [rsp+78h] [rbp-508h]
  char v188; // [rsp+78h] [rbp-508h]
  __int64 v189; // [rsp+78h] [rbp-508h]
  __int64 v190; // [rsp+78h] [rbp-508h]
  unsigned int v191; // [rsp+8Ch] [rbp-4F4h]
  __m128i *v192; // [rsp+90h] [rbp-4F0h]
  __int64 v193; // [rsp+90h] [rbp-4F0h]
  unsigned int v194; // [rsp+90h] [rbp-4F0h]
  unsigned int v195; // [rsp+90h] [rbp-4F0h]
  size_t v196; // [rsp+90h] [rbp-4F0h]
  size_t v197; // [rsp+90h] [rbp-4F0h]
  _BYTE **v198; // [rsp+90h] [rbp-4F0h]
  __int64 v199; // [rsp+98h] [rbp-4E8h]
  __int64 v200; // [rsp+A0h] [rbp-4E0h]
  __int64 v201; // [rsp+A0h] [rbp-4E0h]
  char *v202; // [rsp+A0h] [rbp-4E0h]
  __int64 v203; // [rsp+B0h] [rbp-4D0h]
  unsigned int v204; // [rsp+B0h] [rbp-4D0h]
  __int64 v205; // [rsp+B0h] [rbp-4D0h]
  unsigned __int64 v206; // [rsp+B8h] [rbp-4C8h]
  __int64 v207; // [rsp+B8h] [rbp-4C8h]
  char *v208; // [rsp+B8h] [rbp-4C8h]
  char *v209; // [rsp+B8h] [rbp-4C8h]
  unsigned __int64 v210; // [rsp+C0h] [rbp-4C0h] BYREF
  __int64 v211; // [rsp+C8h] [rbp-4B8h]
  _BYTE *v212; // [rsp+D0h] [rbp-4B0h] BYREF
  __int64 v213; // [rsp+D8h] [rbp-4A8h]
  _BYTE v214[16]; // [rsp+E0h] [rbp-4A0h] BYREF
  _BYTE *v215; // [rsp+F0h] [rbp-490h] BYREF
  __int64 v216; // [rsp+F8h] [rbp-488h]
  _BYTE v217[16]; // [rsp+100h] [rbp-480h] BYREF
  _QWORD v218[2]; // [rsp+110h] [rbp-470h] BYREF
  __int16 v219; // [rsp+120h] [rbp-460h]
  void *v220; // [rsp+130h] [rbp-450h] BYREF
  __int64 v221; // [rsp+138h] [rbp-448h]
  _BYTE v222[16]; // [rsp+140h] [rbp-440h] BYREF
  _BYTE *v223; // [rsp+150h] [rbp-430h] BYREF
  __int64 v224; // [rsp+158h] [rbp-428h]
  _BYTE v225[32]; // [rsp+160h] [rbp-420h] BYREF
  _BYTE *v226; // [rsp+180h] [rbp-400h] BYREF
  __int64 v227; // [rsp+188h] [rbp-3F8h]
  _BYTE v228[32]; // [rsp+190h] [rbp-3F0h] BYREF
  _QWORD *v229; // [rsp+1B0h] [rbp-3D0h] BYREF
  __int64 v230; // [rsp+1B8h] [rbp-3C8h]
  _BYTE v231[64]; // [rsp+1C0h] [rbp-3C0h] BYREF
  unsigned int v232; // [rsp+200h] [rbp-380h]
  char v233; // [rsp+204h] [rbp-37Ch]
  void **p_base; // [rsp+208h] [rbp-378h]
  __m128i *p_dest; // [rsp+210h] [rbp-370h] BYREF
  __m128i *v236; // [rsp+218h] [rbp-368h] BYREF
  __m128i dest; // [rsp+220h] [rbp-360h] BYREF
  int v238; // [rsp+230h] [rbp-350h]
  unsigned __int64 *v239; // [rsp+238h] [rbp-348h]
  __m128i v240; // [rsp+268h] [rbp-318h]
  unsigned __int64 *v241; // [rsp+280h] [rbp-300h] BYREF
  __int64 v242; // [rsp+288h] [rbp-2F8h]
  _BYTE v243[128]; // [rsp+290h] [rbp-2F0h] BYREF
  unsigned __int64 *v244; // [rsp+310h] [rbp-270h] BYREF
  __int64 v245; // [rsp+318h] [rbp-268h]
  _BYTE v246[128]; // [rsp+320h] [rbp-260h] BYREF
  void *base; // [rsp+3A0h] [rbp-1E0h] BYREF
  __int64 v248; // [rsp+3A8h] [rbp-1D8h]
  _BYTE v249[464]; // [rsp+3B0h] [rbp-1D0h] BYREF

  v14 = a1;
  v226 = v228;
  v212 = v214;
  v215 = v217;
  v223 = v225;
  v241 = (unsigned __int64 *)v243;
  v244 = (unsigned __int64 *)v246;
  v220 = v222;
  v224 = 0x400000000LL;
  v227 = 0x400000000LL;
  v213 = 0x400000000LL;
  v216 = 0x400000000LL;
  v242 = 0x400000000LL;
  v245 = 0x400000000LL;
  v221 = 0x400000000LL;
  base = v249;
  v248 = 0x400000000LL;
  sub_38EB180(a1);
  v17 = **(_DWORD **)(a1 + 152);
  v191 = 0;
  v203 = a1 + 144;
  if ( !v17 )
  {
LABEL_39:
    v43 = (__int64)a7;
    v44 = (char *)v220;
    *a7 = v227;
    *a8 = v224;
    v45 = 4LL * (unsigned int)v221;
    if ( (unsigned int)v221 > 1uLL )
    {
      qsort(v44, v45 >> 2, 4u, (__compar_fn_t)sub_1DC3280);
      v44 = (char *)v220;
      v47 = (char *)v220;
      v46 = (char *)v220 + 4 * (unsigned int)v221;
      if ( v46 != v220 )
      {
LABEL_140:
        while ( 1 )
        {
          v122 = v47;
          v47 += 4;
          if ( v46 == v47 )
            break;
          v43 = *((unsigned int *)v47 - 1);
          if ( (_DWORD)v43 == *(_DWORD *)v47 )
          {
            if ( v46 == v122 )
            {
              v47 = v46;
            }
            else
            {
              v15 = (size_t)(v122 + 8);
              if ( v46 != v122 + 8 )
              {
                while ( 1 )
                {
                  if ( *(_DWORD *)v15 != (_DWORD)v43 )
                  {
                    *((_DWORD *)v122 + 1) = *(_DWORD *)v15;
                    v122 += 4;
                  }
                  v15 += 4LL;
                  if ( v46 == (char *)v15 )
                    break;
                  v43 = *(unsigned int *)v122;
                }
                v44 = (char *)v220;
                v15 = (_BYTE *)v220 + 4 * (unsigned int)v221 - v46;
                v47 = &v122[v15 + 4];
                if ( v46 != (char *)v220 + 4 * (unsigned int)v221 )
                {
                  memmove(v122 + 4, v46, v15);
                  v44 = (char *)v220;
                }
              }
            }
            break;
          }
        }
      }
    }
    else
    {
      v46 = &v44[v45];
      v47 = v44;
      if ( v46 != v44 )
        goto LABEL_140;
    }
    v236 = 0;
    v48 = (v47 - v44) >> 2;
    p_dest = &dest;
    LODWORD(v221) = v48;
    v49 = *(unsigned int *)(a11 + 8);
    v50 = *(__int64 **)a11;
    dest.m128i_i8[0] = 0;
    v51 = (unsigned __int64 *)&v50[4 * v49];
    while ( v50 != (__int64 *)v51 )
    {
      while ( 1 )
      {
        v51 -= 4;
        if ( (unsigned __int64 *)*v51 == v51 + 2 )
          break;
        j_j___libc_free_0(*v51);
        if ( v50 == (__int64 *)v51 )
          goto LABEL_45;
      }
    }
LABEL_45:
    *(_DWORD *)(a11 + 8) = 0;
    if ( (unsigned int)v48 > (unsigned __int64)*(unsigned int *)(a11 + 12) )
      sub_12BE710(a11, (unsigned int)v48, v15, v43, (__int64)v44, v16);
    v52 = *(__int64 **)a11;
    *(_DWORD *)(a11 + 8) = v48;
    for ( i = &v52[4 * (unsigned int)v48]; i != v52; v52 += 4 )
    {
      if ( v52 )
      {
        *v52 = (__int64)(v52 + 2);
        sub_38E3500(v52, p_dest, (__int64)v236->m128i_i64 + (_QWORD)p_dest);
      }
    }
    if ( p_dest != &dest )
      j_j___libc_free_0((unsigned __int64)p_dest);
    if ( (_DWORD)v221 )
    {
      v201 = v14;
      v207 = 4LL * (unsigned int)v221;
      v54 = 0;
      do
      {
        v55 = *(__int64 **)a11;
        v56 = *a13;
        p_dest = (__m128i *)&unk_49EFBE0;
        v238 = 1;
        v239 = (unsigned __int64 *)&v55[v54];
        dest = 0u;
        v236 = 0;
        v57 = *(unsigned int *)((char *)v220 + v54);
        v54 += 4;
        (*(void (__fastcall **)(__int64 *, __m128i **, __int64))(v56 + 24))(a13, &p_dest, v57);
        sub_16E7BC0((__int64 *)&p_dest);
      }
      while ( v207 != v54 );
      v14 = v201;
    }
    v58 = (__int64)a8;
    v59 = *a7;
    v60 = *a8;
    if ( !(*a8 | *a7) )
      goto LABEL_73;
    v61 = v59 + v60;
    v62 = v59 + v60;
    v63 = *(unsigned int *)(a9 + 8);
    if ( v61 >= v63 )
    {
      if ( v61 > v63 )
      {
        if ( v61 > *(unsigned int *)(a9 + 12) )
        {
          sub_16CD150(a9, (const void *)(a9 + 16), v61, 16, (int)v44, v16);
          v63 = *(unsigned int *)(a9 + 8);
        }
        v58 = *(_QWORD *)a9;
        v173 = *(_QWORD *)a9 + 16 * v63;
        for ( j = *(_QWORD *)a9 + 16 * v61; j != v173; v173 += 16 )
        {
          if ( v173 )
          {
            *(_QWORD *)v173 = 0;
            *(_BYTE *)(v173 + 8) = 0;
          }
        }
        *(_DWORD *)(a9 + 8) = v61;
      }
    }
    else
    {
      *(_DWORD *)(a9 + 8) = v61;
    }
    v64 = *(unsigned int *)(a10 + 8);
    if ( v61 >= v64 )
    {
      if ( v61 <= v64 )
        goto LABEL_66;
      if ( v61 > *(unsigned int *)(a10 + 12) )
      {
        sub_12BE710(a10, v61, v64, v58, (__int64)v44, v16);
        v64 = *(unsigned int *)(a10 + 8);
      }
      v176 = *(_QWORD *)a10 + 32 * v61;
      for ( k = *(_QWORD *)a10 + 32 * v64; v176 != k; k += 32 )
      {
        if ( k )
        {
          *(_QWORD *)(k + 8) = 0;
          *(_QWORD *)k = k + 16;
          *(_BYTE *)(k + 16) = 0;
        }
      }
    }
    else
    {
      v65 = (unsigned __int64 *)(*(_QWORD *)a10 + 32 * v64);
      v66 = *(_QWORD *)a10 + 32 * v61;
      while ( (unsigned __int64 *)v66 != v65 )
      {
        v65 -= 4;
        if ( (unsigned __int64 *)*v65 != v65 + 2 )
          j_j___libc_free_0(*v65);
      }
    }
    *(_DWORD *)(a10 + 8) = v62;
LABEL_66:
    v67 = *a7;
    if ( *a7 )
    {
      v68 = 0;
      do
      {
        v69 = v68++;
        v70 = v215[v69];
        v71 = *(_QWORD *)a9 + 16 * v69;
        *(_QWORD *)v71 = *(_QWORD *)&v226[8 * v69];
        v72 = v244;
        *(_BYTE *)(v71 + 8) = v70;
        sub_2240AE0((unsigned __int64 *)(*(_QWORD *)a10 + 32 * v69), &v72[4 * v69]);
      }
      while ( *a7 > v68 );
      v67 = *a7;
    }
    if ( *a8 )
    {
      v73 = 0;
      do
      {
        v74 = v73;
        v75 = v73 + v67;
        ++v73;
        v76 = v212[v74];
        v77 = *(_QWORD *)a9 + 16 * v75;
        *(_QWORD *)v77 = *(_QWORD *)&v223[8 * v74];
        v78 = &v241[4 * v74];
        *(_BYTE *)(v77 + 8) = v76;
        sub_2240AE0((unsigned __int64 *)(*(_QWORD *)a10 + 32 * v75), v78);
      }
      while ( *a8 > v73 );
    }
LABEL_73:
    v79 = 0;
    v230 = 0;
    v231[0] = 0;
    v80 = (char *)base;
    v229 = v231;
    v238 = 1;
    p_dest = (__m128i *)&unk_49EFBE0;
    dest = 0u;
    v239 = (unsigned __int64 *)&v229;
    v81 = *(__int64 ***)(v14 + 344);
    v236 = 0;
    v82 = **v81;
    v83 = *(char **)(v82 + 8);
    v202 = *(char **)(v82 + 16);
    v84 = 104LL * (unsigned int)v248;
    if ( 13 * (unsigned __int64)(unsigned int)v248 > 0xD )
    {
      v208 = *(char **)(v82 + 8);
      qsort(base, 0x4EC4EC4EC4EC4EC5LL * ((104LL * (unsigned int)v248) >> 3), 0x68u, (__compar_fn_t)sub_38E30D0);
      v80 = (char *)base;
      v79 = (__m128i *)dest.m128i_i64[1];
      v83 = v208;
      v84 = 104LL * (unsigned int)v248;
    }
    v209 = &v80[v84];
    if ( &v80[v84] == v80 )
    {
LABEL_186:
      if ( v202 != v83 )
      {
        v144 = v202 - v83;
        v145 = v202 - v83;
        if ( (unsigned __int64)(v202 - v83) > dest.m128i_i64[0] - (__int64)v79 )
        {
          sub_16E7EE0((__int64)&p_dest, v83, v145);
          v79 = (__m128i *)dest.m128i_i64[1];
        }
        else
        {
          memcpy(v79, v83, v145);
          v79 = (__m128i *)(v144 + dest.m128i_i64[1]);
          dest.m128i_i64[1] += v144;
        }
      }
      if ( v236 != v79 )
        sub_16E7BA0((__int64 *)&p_dest);
      sub_2240AE0(a6, v239);
      sub_16E7BC0((__int64 *)&p_dest);
      if ( v229 != (_QWORD *)v231 )
        j_j___libc_free_0((unsigned __int64)v229);
      v146 = 0;
      goto LABEL_194;
    }
    v204 = 0;
    v85 = v83;
    v199 = v14;
    v86 = v80;
    while ( 2 )
    {
      v87 = *((_QWORD *)v86 + 1);
      v88 = *(_DWORD *)v86;
      v89 = v87 - (_DWORD)v85;
      if ( (_DWORD)v87 != (_DWORD)v85 )
      {
        if ( (unsigned __int64)v89 > dest.m128i_i64[0] - (__int64)v79 )
        {
          sub_16E7EE0((__int64)&p_dest, v85, v89);
          v79 = (__m128i *)dest.m128i_i64[1];
        }
        else if ( v89 )
        {
          v193 = v89;
          memcpy(v79, v85, v89);
          dest.m128i_i64[1] += v193;
          v79 = (__m128i *)dest.m128i_i64[1];
          if ( v88 == 8 )
          {
LABEL_243:
            v85 = (char *)(v87 + *((unsigned int *)v86 + 4));
            goto LABEL_184;
          }
          goto LABEL_79;
        }
      }
      if ( v88 == 8 )
        goto LABEL_243;
LABEL_79:
      switch ( v88 )
      {
        case 0:
          if ( dest.m128i_i64[0] - (__int64)v79 <= 5uLL )
          {
            sub_16E7EE0((__int64)&p_dest, (char *)".align", 6u);
            v79 = (__m128i *)dest.m128i_i64[1];
          }
          else
          {
            v79->m128i_i32[0] = 1768710446;
            v79->m128i_i16[2] = 28263;
            v79 = (__m128i *)(dest.m128i_i64[1] + 6);
            dest.m128i_i64[1] += 6;
          }
          v133 = 0;
          if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v199 + 320) + 16LL) + 283LL) )
          {
            v152 = *((_QWORD *)v86 + 3);
            if ( dest.m128i_i64[0] <= (unsigned __int64)v79 )
            {
              p_p_dest = (_BYTE **)sub_16E7DE0((__int64)&p_dest, 32);
            }
            else
            {
              p_p_dest = (_BYTE **)&p_dest;
              dest.m128i_i64[1] = (__int64)v79->m128i_i64 + 1;
              v79->m128i_i8[0] = 32;
            }
            sub_16E7A90((__int64)p_p_dest, (unsigned int)v152);
            v79 = (__m128i *)dest.m128i_i64[1];
            v133 = 2;
            if ( (unsigned int)v152 > 3 )
              v133 = ((unsigned int)v152 > 6) + 3LL;
          }
          goto LABEL_183;
        case 1:
          if ( dest.m128i_i64[0] - (__int64)v79 <= 4uLL )
          {
            sub_16E7EE0((__int64)&p_dest, ".even", 5u);
            v79 = (__m128i *)dest.m128i_i64[1];
            v133 = 0;
          }
          else
          {
            v79->m128i_i32[0] = 1702257966;
            v79->m128i_i8[4] = 110;
            v79 = (__m128i *)(dest.m128i_i64[1] + 5);
            v133 = 0;
            dest.m128i_i64[1] += 5;
          }
          goto LABEL_183;
        case 2:
          if ( dest.m128i_i64[0] - (__int64)v79 <= 4uLL )
          {
            sub_16E7EE0((__int64)&p_dest, ".byte", 5u);
            v79 = (__m128i *)dest.m128i_i64[1];
            v133 = 0;
          }
          else
          {
            v79->m128i_i32[0] = 1954112046;
            v79->m128i_i8[4] = 101;
            v79 = (__m128i *)(dest.m128i_i64[1] + 5);
            v133 = 0;
            dest.m128i_i64[1] += 5;
          }
          goto LABEL_183;
        case 3:
          if ( dest.m128i_i64[0] <= (unsigned __int64)v79 )
          {
            v154 = (_BYTE **)sub_16E7DE0((__int64)&p_dest, 36);
          }
          else
          {
            v154 = (_BYTE **)&p_dest;
            dest.m128i_i64[1] = (__int64)v79->m128i_i64 + 1;
            v79->m128i_i8[0] = 36;
          }
          sub_16E7A90((__int64)v154, v191++);
          v79 = (__m128i *)dest.m128i_i64[1];
          v133 = 0;
          goto LABEL_183;
        case 4:
          if ( dest.m128i_i64[0] <= (unsigned __int64)v79 )
          {
            v143 = (_BYTE **)sub_16E7DE0((__int64)&p_dest, 36);
          }
          else
          {
            v143 = (_BYTE **)&p_dest;
            dest.m128i_i64[1] = (__int64)v79->m128i_i64 + 1;
            v79->m128i_i8[0] = 36;
          }
          sub_16E7A90((__int64)v143, v204++);
          v79 = (__m128i *)dest.m128i_i64[1];
          v133 = 0;
          goto LABEL_183;
        case 5:
          v141 = *((_QWORD *)v86 + 3);
          if ( v141 == 64 )
          {
            if ( dest.m128i_i64[0] - (__int64)v79 <= 9uLL )
            {
              sub_16E7EE0((__int64)&p_dest, "qword ptr ", 0xAu);
              v79 = (__m128i *)dest.m128i_i64[1];
              v133 = 0;
              goto LABEL_183;
            }
            v142 = 0x74702064726F7771LL;
LABEL_280:
            v79->m128i_i64[0] = v142;
            v79->m128i_i16[4] = 8306;
            v79 = (__m128i *)(dest.m128i_i64[1] + 10);
            v133 = 0;
            dest.m128i_i64[1] += 10;
            goto LABEL_183;
          }
          if ( v141 <= 64 )
          {
            if ( v141 == 16 )
            {
              if ( dest.m128i_i64[0] - (__int64)v79 <= 8uLL )
              {
                sub_16E7EE0((__int64)&p_dest, "word ptr ", 9u);
                v79 = (__m128i *)dest.m128i_i64[1];
                v133 = 0;
                goto LABEL_183;
              }
              v172 = 0x7274702064726F77LL;
LABEL_284:
              v79->m128i_i64[0] = v172;
              v79->m128i_i8[8] = 32;
              v79 = (__m128i *)(dest.m128i_i64[1] + 9);
              v133 = 0;
              dest.m128i_i64[1] += 9;
              goto LABEL_183;
            }
            if ( v141 != 32 )
            {
              if ( v141 != 8 )
                goto LABEL_164;
              if ( dest.m128i_i64[0] - (__int64)v79 <= 8uLL )
              {
                sub_16E7EE0((__int64)&p_dest, "byte ptr ", 9u);
                v79 = (__m128i *)dest.m128i_i64[1];
                v133 = 0;
                goto LABEL_183;
              }
              v172 = 0x7274702065747962LL;
              goto LABEL_284;
            }
            if ( dest.m128i_i64[0] - (__int64)v79 <= 9uLL )
            {
              sub_16E7EE0((__int64)&p_dest, "dword ptr ", 0xAu);
              v79 = (__m128i *)dest.m128i_i64[1];
              v133 = 0;
              goto LABEL_183;
            }
            v142 = 0x74702064726F7764LL;
            goto LABEL_280;
          }
          switch ( v141 )
          {
            case 128LL:
              if ( dest.m128i_i64[0] - (__int64)v79 <= 0xBuLL )
              {
                sub_16E7EE0((__int64)&p_dest, "xmmword ptr ", 0xCu);
                v79 = (__m128i *)dest.m128i_i64[1];
                v133 = 0;
                goto LABEL_183;
              }
              v155 = 0x2064726F776D6D78LL;
              break;
            case 256LL:
              if ( dest.m128i_i64[0] - (__int64)v79 <= 0xBuLL )
              {
                sub_16E7EE0((__int64)&p_dest, "ymmword ptr ", 0xCu);
                v79 = (__m128i *)dest.m128i_i64[1];
                v133 = 0;
                goto LABEL_183;
              }
              v155 = 0x2064726F776D6D79LL;
              break;
            case 80LL:
              if ( dest.m128i_i64[0] - (__int64)v79 <= 9uLL )
              {
                sub_16E7EE0((__int64)&p_dest, "xword ptr ", 0xAu);
                v79 = (__m128i *)dest.m128i_i64[1];
                v133 = 0;
                goto LABEL_183;
              }
              v142 = 0x74702064726F7778LL;
              goto LABEL_280;
            default:
              goto LABEL_164;
          }
          v79->m128i_i64[0] = v155;
          v79->m128i_i32[2] = 544371824;
          v79 = (__m128i *)(dest.m128i_i64[1] + 12);
          v133 = 0;
          dest.m128i_i64[1] += 12;
LABEL_183:
          v85 = (char *)(v87 + v133 + *((unsigned int *)v86 + 4));
LABEL_184:
          v86 += 104;
          if ( v209 == v86 )
          {
            v83 = v85;
            goto LABEL_186;
          }
          continue;
        case 6:
          v134 = *(_QWORD *)(*(_QWORD *)(v199 + 320) + 16LL);
          v135 = *(char **)(v134 + 96);
          v136 = *(_QWORD *)(v134 + 104);
          v137 = dest.m128i_i64[0] - (_QWORD)v79;
          if ( v136 > dest.m128i_i64[0] - (__int64)v79 )
          {
            v178 = sub_16E7EE0((__int64)&p_dest, v135, v136);
            v139 = *((_QWORD *)v86 + 5);
            v140 = (void *)*((_QWORD *)v86 + 4);
            v79 = *(__m128i **)(v178 + 24);
            v138 = (_BYTE **)v178;
            if ( v139 <= *(_QWORD *)(v178 + 16) - (_QWORD)v79 )
              goto LABEL_171;
          }
          else
          {
            v138 = (_BYTE **)&p_dest;
            if ( v136 )
            {
              v197 = v136;
              memcpy(v79, v135, v136);
              dest.m128i_i64[1] += v197;
              v79 = (__m128i *)dest.m128i_i64[1];
              v137 = dest.m128i_i64[0] - dest.m128i_i64[1];
            }
            v139 = *((_QWORD *)v86 + 5);
            v140 = (void *)*((_QWORD *)v86 + 4);
            if ( v139 <= v137 )
            {
LABEL_171:
              if ( v139 )
              {
                v196 = v139;
                memcpy(v79, v140, v139);
                v138[3] += v196;
              }
              v79 = (__m128i *)dest.m128i_i64[1];
              v133 = 0;
              goto LABEL_183;
            }
          }
          sub_16E7EE0((__int64)v138, (char *)v140, v139);
          v79 = (__m128i *)dest.m128i_i64[1];
          v133 = 0;
          goto LABEL_183;
        case 7:
          if ( dest.m128i_i64[0] - (__int64)v79 <= 1uLL )
          {
            sub_16E7EE0((__int64)&p_dest, "\n\t", 2u);
            v79 = (__m128i *)dest.m128i_i64[1];
            v133 = 0;
          }
          else
          {
            v79->m128i_i16[0] = 2314;
            v79 = (__m128i *)(dest.m128i_i64[1] + 2);
            v133 = 0;
            dest.m128i_i64[1] += 2;
          }
          goto LABEL_183;
        case 9:
          if ( v86[48] )
          {
            if ( (__m128i *)dest.m128i_i64[0] == v79 )
            {
              sub_16E7EE0((__int64)&p_dest, "[", 1u);
              v79 = (__m128i *)dest.m128i_i64[1];
            }
            else
            {
              v79->m128i_i8[0] = 91;
              v79 = (__m128i *)++dest.m128i_i64[1];
            }
          }
          v123 = *((_QWORD *)v86 + 9);
          if ( v123 )
          {
            v179 = (char *)*((_QWORD *)v86 + 8);
            v180 = *((_QWORD *)v86 + 9);
            if ( v123 > dest.m128i_i64[0] - (__int64)v79 )
            {
              sub_16E7EE0((__int64)&p_dest, v179, v180);
              v79 = (__m128i *)dest.m128i_i64[1];
            }
            else
            {
              memcpy(v79, v179, v180);
              v79 = (__m128i *)(v123 + dest.m128i_i64[1]);
              dest.m128i_i64[1] += v123;
            }
          }
          v124 = *((_QWORD *)v86 + 11);
          if ( !v124 )
            goto LABEL_153;
          if ( *((_QWORD *)v86 + 9) )
          {
            if ( dest.m128i_i64[0] - (__int64)v79 > 2uLL )
            {
              v79->m128i_i8[2] = 32;
              v126 = (_BYTE **)&p_dest;
              v79->m128i_i16[0] = 11040;
              v79 = (__m128i *)(dest.m128i_i64[1] + 3);
              dest.m128i_i64[1] += 3;
              v124 = *((_QWORD *)v86 + 11);
            }
            else
            {
              v125 = sub_16E7EE0((__int64)&p_dest, " + ", 3u);
              v124 = *((_QWORD *)v86 + 11);
              v79 = *(__m128i **)(v125 + 24);
              v126 = (_BYTE **)v125;
            }
            v127 = (void *)*((_QWORD *)v86 + 10);
            if ( v126[2] - (_BYTE *)v79 >= v124 )
            {
              if ( !v124 )
              {
LABEL_152:
                v79 = (__m128i *)dest.m128i_i64[1];
LABEL_153:
                if ( *((_DWORD *)v86 + 24) > 1u )
                {
                  if ( dest.m128i_i64[0] - (__int64)v79 <= 4uLL )
                  {
                    v175 = (_BYTE **)sub_16E7EE0((__int64)&p_dest, " * $$", 5u);
                  }
                  else
                  {
                    v79->m128i_i32[0] = 606087712;
                    v175 = (_BYTE **)&p_dest;
                    v79->m128i_i8[4] = 36;
                    dest.m128i_i64[1] += 5;
                  }
                  sub_16E7A90((__int64)v175, *((unsigned int *)v86 + 24));
                  v79 = (__m128i *)dest.m128i_i64[1];
                  v128 = *((_QWORD *)v86 + 9);
                  if ( *((_QWORD *)v86 + 7) )
                  {
LABEL_155:
                    v129 = " + $$";
                    if ( !v128 && !*((_QWORD *)v86 + 11) )
                      v129 = "$$";
LABEL_158:
                    v192 = v79;
                    v130 = strlen(v129);
                    v131 = v130;
                    if ( dest.m128i_i64[0] - (__int64)v192 >= v130 )
                    {
                      if ( (_DWORD)v130 )
                      {
                        v181 = 0;
                        do
                        {
                          v182 = v181++;
                          v192->m128i_i8[v182] = v129[v182];
                        }
                        while ( v181 < (unsigned int)v131 );
                      }
                      dest.m128i_i64[1] += v131;
                      v132 = (_BYTE **)&p_dest;
                    }
                    else
                    {
                      v132 = (_BYTE **)sub_16E7EE0((__int64)&p_dest, v129, v130);
                    }
                    sub_16E7AB0((__int64)v132, *((_QWORD *)v86 + 7));
                    v79 = (__m128i *)dest.m128i_i64[1];
LABEL_161:
                    v133 = 0;
                    if ( v86[48] )
                    {
                      if ( (__m128i *)dest.m128i_i64[0] != v79 )
                      {
                        v79->m128i_i8[0] = 93;
                        v79 = (__m128i *)++dest.m128i_i64[1];
                        goto LABEL_164;
                      }
                      sub_16E7EE0((__int64)&p_dest, "]", 1u);
                      v79 = (__m128i *)dest.m128i_i64[1];
                      v133 = 0;
                    }
                    goto LABEL_183;
                  }
                }
                else
                {
                  v128 = *((_QWORD *)v86 + 9);
                  if ( *((_QWORD *)v86 + 7) )
                    goto LABEL_155;
                }
                if ( v128 )
                  goto LABEL_161;
                v129 = "$$";
                if ( *((_QWORD *)v86 + 11) )
                  goto LABEL_161;
                goto LABEL_158;
              }
LABEL_326:
              v198 = v126;
              memcpy(v79, v127, v124);
              v198[3] += v124;
              goto LABEL_152;
            }
          }
          else
          {
            v127 = (void *)*((_QWORD *)v86 + 10);
            v126 = (_BYTE **)&p_dest;
            if ( v124 <= dest.m128i_i64[0] - (__int64)v79 )
              goto LABEL_326;
          }
          sub_16E7EE0((__int64)v126, (char *)v127, v124);
          goto LABEL_152;
        default:
LABEL_164:
          v133 = 0;
          goto LABEL_183;
      }
    }
  }
  while ( 1 )
  {
    if ( (unsigned int)(v17 - 21) <= 1 )
    {
      v18 = sub_3909290(v203);
      sub_38EB180(v14);
      if ( **(_DWORD **)(v14 + 152) == 9 )
        sub_38EB180(v14);
      v19 = sub_3909290(v203);
      v21 = v248;
      v22 = v19;
      if ( (unsigned int)v248 >= HIDWORD(v248) )
      {
        sub_16CD150((__int64)&base, v249, 0, 104, v20, v16);
        v21 = v248;
      }
      v23 = (char *)base + 104 * v21;
      if ( v23 )
      {
        *((_QWORD *)v23 + 3) = 0;
        *((_DWORD *)v23 + 4) = v22 - v18;
        *((_QWORD *)v23 + 4) = 0;
        *((_QWORD *)v23 + 5) = 0;
        v23[48] = 0;
        *((_QWORD *)v23 + 7) = 0;
        *((_QWORD *)v23 + 8) = 0;
        *((_QWORD *)v23 + 9) = 0;
        *((_QWORD *)v23 + 10) = 0;
        *((_QWORD *)v23 + 11) = 0;
        *((_DWORD *)v23 + 24) = 1;
        *(_DWORD *)v23 = 8;
        *((_QWORD *)v23 + 1) = v18;
        v21 = v248;
      }
      v15 = v21 + 1;
      LODWORD(v248) = v15;
      goto LABEL_10;
    }
    v232 = -1;
    v233 = 0;
    v229 = v231;
    v230 = 0x800000000LL;
    p_base = &base;
    if ( (unsigned __int8)sub_38F9390(v14, (__int64)&v229, a14, a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64)
      || v233 )
    {
      break;
    }
    if ( v232 == -1 )
    {
      v120 = (unsigned __int64)v229;
      v41 = &v229[(unsigned int)v230];
      if ( v229 == v41 )
        goto LABEL_37;
      do
      {
        v121 = *--v41;
        if ( v121 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v121 + 8LL))(v121);
      }
      while ( (_QWORD *)v120 != v41 );
    }
    else
    {
      v24 = (__int64)a12;
      v206 = *a12 + ((unsigned __int64)v232 << 6);
      if ( (_DWORD)v230 != 1 )
      {
        v200 = v14;
        v25 = 1;
        v26 = v230;
        while ( 1 )
        {
          v27 = (_QWORD *)v229[v25];
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *))(*v27 + 40LL))(v27) )
            break;
LABEL_26:
          if ( ++v25 == v26 )
          {
            v14 = v200;
            goto LABEL_28;
          }
        }
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD *))(*v27 + 48LL))(v27) )
        {
          v28 = *v27;
          v29 = *(__int64 (**)())(*v27 + 88LL);
          if ( v29 != sub_38E29B0 )
          {
            if ( ((unsigned __int8 (__fastcall *)(_QWORD *))v29)(v27) )
              goto LABEL_81;
            v28 = *v27;
          }
          v24 = v200;
          v30 = *(_QWORD *)(v200 + 8);
          v31 = *(__int64 (**)())(*(_QWORD *)v30 + 80LL);
          v32 = (*(__int64 (__fastcall **)(_QWORD *))(v28 + 56))(v27);
          if ( v31 == sub_38E29D0 || (v24 = v32, !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v31)(v30, v32)) )
          {
            if ( *(_BYTE *)(v206 + 4) && (unsigned int)*(unsigned __int8 *)(v206 + 4) > *((_DWORD *)v27 + 2) )
            {
              v35 = (*(__int64 (__fastcall **)(_QWORD *))(*v27 + 56LL))(v27);
              v36 = (unsigned int)v221;
              if ( (unsigned int)v221 >= HIDWORD(v221) )
              {
                v24 = (__int64)v222;
                sub_16CD150((__int64)&v220, v222, 0, 4, v33, v34);
                v36 = (unsigned int)v221;
              }
              *((_DWORD *)v220 + v36) = v35;
              LODWORD(v221) = v221 + 1;
            }
            goto LABEL_26;
          }
        }
LABEL_81:
        v90 = *(__int64 (**)())(*v27 + 16LL);
        if ( v90 == sub_38E2990 )
          goto LABEL_26;
        v91 = ((__int64 (__fastcall *)(_QWORD *))v90)(v27);
        v93 = v92;
        if ( !v92 )
          goto LABEL_26;
        v94 = *(__int64 (**)())(*v27 + 24LL);
        if ( v94 == sub_38E29A0 )
          goto LABEL_26;
        v97 = ((__int64 (__fastcall *)(_QWORD *))v94)(v27);
        if ( !v97 )
          goto LABEL_26;
        if ( v25 == 1 && (*(_BYTE *)(v206 + 10) & 2) != 0 )
        {
          ++v191;
          v108 = (unsigned int)v227;
          if ( (unsigned int)v227 >= HIDWORD(v227) )
          {
            v24 = (__int64)v228;
            v189 = v97;
            sub_16CD150((__int64)&v226, v228, 0, 8, v96, v97);
            v108 = (unsigned int)v227;
            v97 = v189;
          }
          v109 = (unsigned __int64)v226;
          *(_QWORD *)&v226[8 * v108] = v97;
          v110 = 0;
          LODWORD(v227) = v227 + 1;
          v111 = *(__int64 (**)())(*v27 + 88LL);
          if ( v111 == sub_38E29B0 )
          {
            v112 = (unsigned int)v216;
            if ( (unsigned int)v216 < HIDWORD(v216) )
              goto LABEL_111;
          }
          else
          {
            v110 = ((__int64 (__fastcall *)(_QWORD *, __int64, unsigned __int64, __int64, __int64, _QWORD))v111)(
                     v27,
                     v24,
                     v109,
                     v95,
                     v96,
                     0);
            v112 = (unsigned int)v216;
            if ( (unsigned int)v216 < HIDWORD(v216) )
              goto LABEL_111;
          }
          v188 = v110;
          sub_16CD150((__int64)&v215, v217, 0, 1, v96, v110);
          v112 = (unsigned int)v216;
          LOBYTE(v110) = v188;
LABEL_111:
          v24 = (__int64)v218;
          v215[v112] = v110;
          v113 = v27[3];
          v114 = v27[2];
          LODWORD(v216) = v216 + 1;
          v211 = v113;
          v218[0] = "=";
          v210 = v114;
          v219 = 1283;
          v218[1] = &v210;
          sub_16E2FC0((__int64 *)&p_dest, (__int64)v218);
          v118 = v245;
          if ( (unsigned int)v245 >= HIDWORD(v245) )
          {
            v24 = 0;
            sub_12BE710((__int64)&v244, 0, (unsigned int)v245, v115, v116, v117);
            v118 = v245;
          }
          v119 = (__m128 *)&v244[4 * v118];
          if ( v119 )
          {
            v119->m128_u64[0] = (unsigned __int64)&v119[1];
            if ( p_dest == &dest )
            {
              a3 = _mm_load_si128(&dest);
              v119[1] = (__m128)a3;
            }
            else
            {
              v119->m128_u64[0] = (unsigned __int64)p_dest;
              v119[1].m128_u64[0] = dest.m128i_i64[0];
            }
            v119->m128_u64[1] = (unsigned __int64)v236;
            LODWORD(v245) = v245 + 1;
          }
          else
          {
            LODWORD(v245) = v118 + 1;
            if ( p_dest != &dest )
            {
              v24 = dest.m128i_i64[0] + 1;
              j_j___libc_free_0((unsigned __int64)p_dest);
              v106 = v248;
              if ( (unsigned int)v248 < HIDWORD(v248) )
                goto LABEL_118;
              goto LABEL_122;
            }
          }
          v106 = v248;
          if ( (unsigned int)v248 < HIDWORD(v248) )
          {
LABEL_118:
            v107 = (char *)base + 104 * v106;
            if ( !v107 )
              goto LABEL_100;
            *(_DWORD *)v107 = 4;
            goto LABEL_99;
          }
LABEL_122:
          v24 = (__int64)v249;
          sub_16CD150((__int64)&base, v249, 0, 104, v116, v117);
          v106 = v248;
          goto LABEL_118;
        }
        v98 = (unsigned int)v224;
        if ( (unsigned int)v224 >= HIDWORD(v224) )
        {
          v24 = (__int64)v225;
          v190 = v97;
          sub_16CD150((__int64)&v223, v225, 0, 8, v96, v97);
          v98 = (unsigned int)v224;
          v97 = v190;
        }
        v99 = (unsigned __int64)v223;
        *(_QWORD *)&v223[8 * v98] = v97;
        v100 = 0;
        LODWORD(v224) = v224 + 1;
        v101 = *(__int64 (**)())(*v27 + 88LL);
        if ( v101 == sub_38E29B0 )
        {
          v102 = (unsigned int)v213;
          if ( (unsigned int)v213 < HIDWORD(v213) )
            goto LABEL_90;
        }
        else
        {
          v100 = ((unsigned int (__fastcall *)(_QWORD *, __int64, unsigned __int64, __int64, __int64, _QWORD))v101)(
                   v27,
                   v24,
                   v99,
                   v95,
                   v96,
                   0);
          v102 = (unsigned int)v213;
          if ( (unsigned int)v213 < HIDWORD(v213) )
            goto LABEL_90;
        }
        v187 = v100;
        sub_16CD150((__int64)&v212, v214, 0, 1, v96, v100);
        v102 = (unsigned int)v213;
        v100 = v187;
LABEL_90:
        v212[v102] = v100;
        v24 = v27[2];
        LODWORD(v213) = v213 + 1;
        if ( v24 )
        {
          v103 = v27[3];
          p_dest = &dest;
          sub_38E3110((__int64 *)&p_dest, (_BYTE *)v24, v24 + v103);
          v104 = (unsigned int)v242;
          if ( (unsigned int)v242 < HIDWORD(v242) )
            goto LABEL_92;
        }
        else
        {
          dest.m128i_i8[0] = 0;
          v104 = (unsigned int)v242;
          p_dest = &dest;
          v236 = 0;
          if ( (unsigned int)v242 < HIDWORD(v242) )
            goto LABEL_92;
        }
        v24 = 0;
        sub_12BE710((__int64)&v241, 0, v104, (__int64)&dest, v96, v100);
        LODWORD(v104) = v242;
LABEL_92:
        v105 = (__m128 *)&v241[4 * (unsigned int)v104];
        if ( v105 )
        {
          v105->m128_u64[0] = (unsigned __int64)&v105[1];
          if ( p_dest == &dest )
          {
            a4 = _mm_load_si128(&dest);
            v105[1] = (__m128)a4;
          }
          else
          {
            v105->m128_u64[0] = (unsigned __int64)p_dest;
            v105[1].m128_u64[0] = dest.m128i_i64[0];
          }
          v105->m128_u64[1] = (unsigned __int64)v236;
          LODWORD(v242) = v242 + 1;
        }
        else
        {
          LODWORD(v242) = v104 + 1;
          if ( p_dest != &dest )
          {
            v24 = dest.m128i_i64[0] + 1;
            j_j___libc_free_0((unsigned __int64)p_dest);
            v106 = v248;
            if ( (unsigned int)v248 < HIDWORD(v248) )
              goto LABEL_97;
            goto LABEL_103;
          }
        }
        v106 = v248;
        if ( (unsigned int)v248 < HIDWORD(v248) )
        {
LABEL_97:
          v107 = (char *)base + 104 * v106;
          if ( !v107 )
          {
LABEL_100:
            LODWORD(v248) = v106 + 1;
            goto LABEL_26;
          }
          *(_DWORD *)v107 = 3;
LABEL_99:
          *((_QWORD *)v107 + 1) = v91;
          *((_DWORD *)v107 + 4) = v93;
          *((_QWORD *)v107 + 3) = 0;
          *((_QWORD *)v107 + 4) = 0;
          *((_QWORD *)v107 + 5) = 0;
          v107[48] = 0;
          *((_QWORD *)v107 + 7) = 0;
          *((_QWORD *)v107 + 8) = 0;
          *((_QWORD *)v107 + 9) = 0;
          *((_QWORD *)v107 + 10) = 0;
          *((_QWORD *)v107 + 11) = 0;
          *((_DWORD *)v107 + 24) = 1;
          v106 = v248;
          goto LABEL_100;
        }
LABEL_103:
        v24 = (__int64)v249;
        sub_16CD150((__int64)&base, v249, 0, 104, v96, v100);
        v106 = v248;
        goto LABEL_97;
      }
LABEL_28:
      v37 = *(_WORD **)(v206 + 32);
      if ( v37 )
      {
        if ( *v37 )
        {
          v38 = 0;
          do
            v39 = &v37[++v38];
          while ( *v39 );
        }
        else
        {
          v39 = *(_WORD **)(v206 + 32);
        }
      }
      else
      {
        v39 = 0;
      }
      sub_38E8C60((__int64)&v220, (char *)v220 + 4 * (unsigned int)v221, *(_QWORD *)(v206 + 32), (__int64)v39);
      v40 = (unsigned __int64)v229;
      v41 = &v229[(unsigned int)v230];
      if ( v229 == v41 )
        goto LABEL_37;
      do
      {
        v42 = *--v41;
        if ( v42 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v42 + 8LL))(v42);
      }
      while ( (_QWORD *)v40 != v41 );
    }
    v41 = v229;
LABEL_37:
    if ( v41 == (_QWORD *)v231 )
    {
LABEL_10:
      v17 = **(_DWORD **)(v14 + 152);
      if ( !v17 )
        goto LABEL_39;
    }
    else
    {
      _libc_free((unsigned __int64)v41);
      v17 = **(_DWORD **)(v14 + 152);
      if ( !v17 )
        goto LABEL_39;
    }
  }
  v156 = *(_QWORD *)(v14 + 24);
  v205 = v156 + 104LL * *(unsigned int *)(v14 + 32);
  if ( v156 != v205 )
  {
    v157 = *(_QWORD *)(v14 + 24);
    do
    {
      v161 = *(__m128i **)v157;
      v236 = (__m128i *)&dest.m128i_u64[1];
      p_dest = v161;
      dest.m128i_i64[0] = 0x4000000000LL;
      v162 = *(_DWORD *)(v157 + 16);
      if ( v162 && &v236 != (__m128i **)(v157 + 8) )
      {
        v163 = v162;
        v164 = (__m128i *)&dest.m128i_u64[1];
        if ( v162 <= 0x40
          || (v195 = *(_DWORD *)(v157 + 16),
              sub_16CD150((__int64)&v236, &dest.m128i_u64[1], v162, 1, v162, v16),
              v163 = *(unsigned int *)(v157 + 16),
              v164 = v236,
              v162 = v195,
              *(_DWORD *)(v157 + 16)) )
        {
          v194 = v162;
          memcpy(v164, *(const void **)(v157 + 8), v163);
          v162 = v194;
        }
        dest.m128i_i32[0] = v162;
      }
      v158 = *(_QWORD *)(v157 + 88);
      v159 = _mm_loadu_si128((const __m128i *)(v157 + 88));
      *(_BYTE *)(v14 + 17) = 1;
      v160 = *(__int64 **)(v14 + 344);
      v240 = v159;
      v218[0] = &v236;
      v210 = v158;
      v219 = 262;
      v211 = v159.m128i_i64[1];
      sub_16D14E0(v160, (unsigned __int64)p_dest, 0, (__int64)v218, &v210, 1, 0, 0, 1u);
      sub_38E35B0((_QWORD *)v14);
      if ( v236 != (__m128i *)&dest.m128i_u64[1] )
        _libc_free((unsigned __int64)v236);
      v157 += 104;
    }
    while ( v205 != v157 );
    v165 = *(_QWORD *)(v14 + 24);
    v166 = v165 + 104LL * *(unsigned int *)(v14 + 32);
    while ( v166 != v165 )
    {
      v166 -= 104;
      v167 = *(_QWORD *)(v166 + 8);
      if ( v167 != v166 + 24 )
        _libc_free(v167);
    }
  }
  v168 = (unsigned __int64)v229;
  v169 = (unsigned int)v230;
  *(_DWORD *)(v14 + 32) = 0;
  v170 = (_QWORD *)(v168 + 8 * v169);
  if ( (_QWORD *)v168 != v170 )
  {
    do
    {
      v171 = *--v170;
      if ( v171 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v171 + 8LL))(v171);
    }
    while ( (_QWORD *)v168 != v170 );
    v170 = v229;
  }
  if ( v170 != (_QWORD *)v231 )
    _libc_free((unsigned __int64)v170);
  v146 = 1;
LABEL_194:
  if ( base != v249 )
    _libc_free((unsigned __int64)base);
  if ( v220 != v222 )
    _libc_free((unsigned __int64)v220);
  v147 = v244;
  v148 = &v244[4 * (unsigned int)v245];
  if ( v244 != v148 )
  {
    do
    {
      v148 -= 4;
      if ( (unsigned __int64 *)*v148 != v148 + 2 )
        j_j___libc_free_0(*v148);
    }
    while ( v147 != v148 );
    v148 = v244;
  }
  if ( v148 != (unsigned __int64 *)v246 )
    _libc_free((unsigned __int64)v148);
  v149 = v241;
  v150 = &v241[4 * (unsigned int)v242];
  if ( v241 != v150 )
  {
    do
    {
      v150 -= 4;
      if ( (unsigned __int64 *)*v150 != v150 + 2 )
        j_j___libc_free_0(*v150);
    }
    while ( v149 != v150 );
    v150 = v241;
  }
  if ( v150 != (unsigned __int64 *)v243 )
    _libc_free((unsigned __int64)v150);
  if ( v215 != v217 )
    _libc_free((unsigned __int64)v215);
  if ( v212 != v214 )
    _libc_free((unsigned __int64)v212);
  if ( v226 != v228 )
    _libc_free((unsigned __int64)v226);
  if ( v223 != v225 )
    _libc_free((unsigned __int64)v223);
  return v146;
}
