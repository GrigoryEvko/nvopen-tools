// Function: sub_2967C10
// Address: 0x2967c10
//
void __fastcall sub_2967C10(__int64 a1, char **a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 **v12; // rax
  char **v13; // r12
  __int64 *v14; // r15
  char *v15; // rdx
  __int64 *v16; // rax
  char *v17; // rbx
  __int64 v18; // rsi
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 *v21; // rbx
  char *v22; // rdx
  unsigned int v23; // eax
  unsigned int v24; // esi
  __int64 v25; // rax
  __int64 *v26; // rbx
  __int64 *v27; // rdi
  __int64 *v28; // r12
  __int64 *v29; // rsi
  __int64 v30; // r12
  __int64 v31; // rdx
  __int64 *v32; // r14
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned int v35; // ecx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rdx
  __int64 *v40; // r15
  __int64 *v41; // rsi
  __int64 v42; // rcx
  char *v43; // rax
  char *v44; // r12
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // r15
  __int64 v50; // r14
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rcx
  int v54; // esi
  __int64 v55; // rdi
  int v56; // esi
  unsigned int v57; // edx
  __int64 *v58; // rax
  __int64 v59; // r10
  __int64 v60; // rdi
  char *k; // r15
  _BYTE *v62; // rsi
  __int64 *v63; // rdx
  __int64 *v64; // rbx
  __int64 *v65; // rdx
  __int64 v66; // rdx
  __int64 *v67; // r14
  __int64 *v68; // r12
  __int64 *v69; // r15
  unsigned int v70; // r12d
  __int64 *v71; // rbx
  int v72; // ecx
  __int64 **v73; // rsi
  unsigned int v74; // edx
  __int64 **v75; // rax
  __int64 *v76; // rdi
  __int64 *v77; // r14
  __int64 v78; // rax
  unsigned __int64 v79; // rdx
  unsigned int v80; // ecx
  __int64 v81; // rdx
  __int64 *v82; // rax
  __int64 v83; // rbx
  __int64 v84; // rdx
  __int64 v85; // r12
  _QWORD *v86; // rdx
  _QWORD *v87; // rax
  int v88; // esi
  __int64 **v89; // rdi
  unsigned int v90; // edx
  __int64 *v91; // rax
  __int64 v92; // rax
  unsigned __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 *v95; // rax
  unsigned int v96; // eax
  int i; // edi
  __int64 *v98; // r14
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // r15
  __int64 *v102; // r14
  __int64 *v103; // r13
  void **v104; // rdi
  __int64 *v105; // r15
  __int64 (__fastcall *v106)(__int64); // rax
  __int64 *v107; // rax
  __int64 v108; // rdi
  __int64 v109; // rsi
  __int64 **v110; // rdi
  int v111; // ecx
  unsigned int v112; // edx
  __int64 **v113; // rax
  __int64 *v114; // r8
  __int64 *v115; // rdi
  __int64 *v116; // r15
  void **v117; // rdi
  __int64 *v118; // r12
  __int64 (__fastcall *v119)(__int64); // rax
  __int64 v120; // rdi
  __int64 *v121; // rbx
  __int64 **v122; // rdi
  int v123; // esi
  unsigned int v124; // ecx
  __int64 **v125; // rax
  __int64 *v126; // r9
  __int64 v127; // rsi
  __int64 v128; // r8
  __int64 v129; // r9
  _QWORD *v130; // r15
  __int64 v131; // rax
  __int64 v132; // r15
  __int64 v133; // r13
  __int64 v134; // rdx
  __int64 v135; // rcx
  unsigned int v136; // esi
  unsigned int v137; // eax
  unsigned int v138; // edx
  unsigned int v139; // ecx
  __int64 v140; // rdx
  __int64 v141; // rax
  int v142; // r10d
  int v143; // eax
  int v144; // eax
  int v145; // r9d
  int v146; // eax
  int v147; // edi
  __int64 *v148; // r14
  size_t v149; // rdx
  __int64 v150; // rbx
  __int64 *v151; // r12
  unsigned __int64 v152; // rax
  __int64 *v153; // rbx
  __int64 v154; // r13
  __int64 *n; // r12
  __int64 v156; // r14
  int v157; // eax
  int v158; // eax
  int v159; // edx
  __int64 *dest; // [rsp+18h] [rbp-518h]
  __int64 *v162; // [rsp+28h] [rbp-508h]
  __int64 *v163; // [rsp+28h] [rbp-508h]
  __int64 v164; // [rsp+28h] [rbp-508h]
  char **v166; // [rsp+38h] [rbp-4F8h]
  __int64 *v167; // [rsp+40h] [rbp-4F0h]
  __int64 *j; // [rsp+40h] [rbp-4F0h]
  __int64 *m; // [rsp+40h] [rbp-4F0h]
  __int64 *v170; // [rsp+40h] [rbp-4F0h]
  char *v172; // [rsp+58h] [rbp-4D8h]
  __int64 *v173; // [rsp+58h] [rbp-4D8h]
  __int64 *v175; // [rsp+68h] [rbp-4C8h]
  __int64 *v176; // [rsp+78h] [rbp-4B8h] BYREF
  void *src; // [rsp+80h] [rbp-4B0h] BYREF
  __int64 v178; // [rsp+88h] [rbp-4A8h]
  _BYTE v179[32]; // [rsp+90h] [rbp-4A0h] BYREF
  void *v180; // [rsp+B0h] [rbp-480h] BYREF
  __int64 v181; // [rsp+B8h] [rbp-478h]
  _BYTE v182[32]; // [rsp+C0h] [rbp-470h] BYREF
  char *v183; // [rsp+E0h] [rbp-450h] BYREF
  __int64 *v184; // [rsp+E8h] [rbp-448h]
  void **v185; // [rsp+F0h] [rbp-440h]
  char *v186; // [rsp+F8h] [rbp-438h]
  __int64 *v187; // [rsp+100h] [rbp-430h]
  void **p_src; // [rsp+108h] [rbp-428h]
  __int64 v189[6]; // [rsp+110h] [rbp-420h] BYREF
  __int64 *v190; // [rsp+140h] [rbp-3F0h] BYREF
  __int64 *v191; // [rsp+148h] [rbp-3E8h]
  __int64 (__fastcall *v192)(__int64); // [rsp+150h] [rbp-3E0h]
  __int64 v193; // [rsp+158h] [rbp-3D8h]
  __int64 (__fastcall *v194)(_QWORD *); // [rsp+160h] [rbp-3D0h]
  __int64 v195; // [rsp+168h] [rbp-3C8h]
  _BYTE *v196; // [rsp+170h] [rbp-3C0h] BYREF
  __int64 v197; // [rsp+178h] [rbp-3B8h]
  _BYTE v198[128]; // [rsp+180h] [rbp-3B0h] BYREF
  __int64 v199; // [rsp+200h] [rbp-330h] BYREF
  char *v200; // [rsp+208h] [rbp-328h]
  __int64 v201; // [rsp+210h] [rbp-320h]
  unsigned int v202; // [rsp+218h] [rbp-318h]
  char v203; // [rsp+21Ch] [rbp-314h]
  char v204; // [rsp+220h] [rbp-310h] BYREF
  char *v205; // [rsp+2A0h] [rbp-290h] BYREF
  _BYTE *v206; // [rsp+2A8h] [rbp-288h]
  __int64 v207; // [rsp+2B0h] [rbp-280h]
  int v208; // [rsp+2B8h] [rbp-278h]
  char v209; // [rsp+2BCh] [rbp-274h]
  _BYTE v210[128]; // [rsp+2C0h] [rbp-270h] BYREF
  char *v211; // [rsp+340h] [rbp-1F0h] BYREF
  __int64 v212; // [rsp+348h] [rbp-1E8h]
  __int64 v213; // [rsp+350h] [rbp-1E0h]
  __int64 v214; // [rsp+358h] [rbp-1D8h]
  __int64 *v215; // [rsp+360h] [rbp-1D0h] BYREF
  __int64 v216; // [rsp+368h] [rbp-1C8h]
  _BYTE v217[128]; // [rsp+370h] [rbp-1C0h] BYREF
  unsigned __int64 v218; // [rsp+3F0h] [rbp-140h] BYREF
  __int64 v219; // [rsp+3F8h] [rbp-138h]
  __int64 *v220; // [rsp+400h] [rbp-130h] BYREF
  unsigned int v221; // [rsp+408h] [rbp-128h]
  char v222; // [rsp+500h] [rbp-30h] BYREF

  v7 = sub_D4B130(a1);
  v8 = **(_QWORD **)(a1 + 32);
  v211 = (char *)v7;
  sub_D696B0(&v218, a4, (__int64 *)&v211);
  v176 = v220;
  sub_D68D70(&v218);
  v211 = (char *)v8;
  sub_D696B0(&v218, a4, (__int64 *)&v211);
  v175 = v220;
  sub_D68D70(&v218);
  v218 = 0;
  src = v179;
  v178 = 0x400000000LL;
  v12 = &v220;
  v219 = 1;
  do
  {
    *v12 = (__int64 *)-4096LL;
    v12 += 2;
  }
  while ( v12 != (__int64 **)&v222 );
  if ( a3 > 4 )
  {
    sub_C8D5F0((__int64)&src, v179, a3, 8u, v10, v11);
    v166 = &a2[a3];
    if ( v166 != a2 )
      goto LABEL_5;
  }
  else
  {
    v166 = &a2[a3];
    if ( v166 != a2 )
    {
LABEL_5:
      v13 = a2;
      v14 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v17 = *v13;
          v205 = *v13;
          sub_D696B0((unsigned __int64 *)&v211, a4, (__int64 *)&v205);
          v199 = v213;
          sub_D68D70(&v211);
          if ( v199 )
          {
            v9 = *(unsigned int *)(a5 + 24);
            v18 = *(_QWORD *)(a5 + 8);
            if ( (_DWORD)v9 )
            {
              v9 = (unsigned int)(v9 - 1);
              v19 = v9 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
              v20 = (__int64 *)(v18 + 16LL * v19);
              v10 = *v20;
              if ( v17 != (char *)*v20 )
              {
                v146 = 1;
                while ( v10 != -4096 )
                {
                  v147 = v146 + 1;
                  v19 = v9 & (v146 + v19);
                  v20 = (__int64 *)(v18 + 16LL * v19);
                  v10 = *v20;
                  if ( v17 == (char *)*v20 )
                    goto LABEL_13;
                  v146 = v147;
                }
                goto LABEL_9;
              }
LABEL_13:
              v21 = (__int64 *)v20[1];
              if ( v21 )
                break;
            }
          }
LABEL_9:
          if ( v166 == ++v13 )
            goto LABEL_24;
        }
        if ( (unsigned __int8)sub_295D140((__int64)&v218, &v199, &v205) )
        {
          v15 = v205 + 8;
          goto LABEL_7;
        }
        v22 = v205;
        ++v218;
        v211 = v205;
        v23 = ((unsigned int)v219 >> 1) + 1;
        if ( (v219 & 1) != 0 )
        {
          v24 = 16;
          if ( 4 * v23 >= 0x30 )
          {
LABEL_115:
            v24 *= 2;
LABEL_116:
            sub_295E3B0((__int64)&v218, v24);
            sub_295D140((__int64)&v218, &v199, &v211);
            v22 = v211;
            v23 = ((unsigned int)v219 >> 1) + 1;
            goto LABEL_18;
          }
        }
        else
        {
          v24 = v221;
          if ( 4 * v23 >= 3 * v221 )
            goto LABEL_115;
        }
        if ( v24 - (v23 + HIDWORD(v219)) <= v24 >> 3 )
          goto LABEL_116;
LABEL_18:
        LODWORD(v219) = v219 & 1 | (2 * v23);
        if ( *(_QWORD *)v22 != -4096 )
          --HIDWORD(v219);
        v25 = v199;
        v15 = v22 + 8;
        *(_QWORD *)v15 = 0;
        *((_QWORD *)v15 - 1) = v25;
LABEL_7:
        *(_QWORD *)v15 = v21;
        sub_B1A4E0((__int64)&src, v199);
        if ( v14 )
        {
          v16 = v21;
          if ( v14 == v21 )
            goto LABEL_9;
          while ( 1 )
          {
            v16 = (__int64 *)*v16;
            if ( v14 == v16 )
              break;
            if ( !v16 )
              goto LABEL_9;
          }
        }
        v14 = v21;
        if ( v166 == ++v13 )
          goto LABEL_24;
      }
    }
  }
  v14 = 0;
LABEL_24:
  v211 = 0;
  v26 = &v199;
  v215 = (__int64 *)v217;
  v216 = 0x1000000000LL;
  v212 = 0;
  v27 = *(__int64 **)(a1 + 40);
  v28 = *(__int64 **)(a1 + 32);
  v213 = 0;
  v214 = 0;
  if ( v27 != v28 )
  {
    while ( 1 )
    {
      v199 = *v28;
      sub_D696B0((unsigned __int64 *)&v205, a4, v26);
      v196 = (_BYTE *)v207;
      sub_D68D70(&v205);
      v10 = (__int64)v196;
      if ( !v196 )
        goto LABEL_27;
      if ( !(_DWORD)v213 )
      {
        v29 = &v215[(unsigned int)v216];
        if ( v29 == sub_2957710(v215, (__int64)v29, (__int64 *)&v196) )
        {
          sub_B1A4E0((__int64)&v215, v10);
          if ( (unsigned int)v216 > 0x10 )
          {
            dest = v26;
            v64 = v215;
            v163 = &v215[(unsigned int)v216];
            do
            {
              v65 = v64++;
              sub_D6CB10((__int64)&v205, (__int64)&v211, v65);
            }
            while ( v163 != v64 );
            v26 = dest;
          }
        }
        goto LABEL_27;
      }
      sub_D6CB10((__int64)&v205, (__int64)&v211, (__int64 *)&v196);
      if ( v210[0] )
      {
        ++v28;
        sub_B1A4E0((__int64)&v215, (__int64)v196);
        if ( v27 == v28 )
          break;
      }
      else
      {
LABEL_27:
        if ( v27 == ++v28 )
          break;
      }
    }
  }
  v199 = 0;
  v196 = v198;
  v197 = 0x1000000000LL;
  v200 = &v204;
  v201 = 16;
  v202 = 0;
  v203 = 1;
  v30 = v175[2];
  if ( !v30 )
    goto LABEL_76;
  while ( 1 )
  {
    v31 = *(_QWORD *)(v30 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v31 - 30) <= 0xAu )
      break;
    v30 = *(_QWORD *)(v30 + 8);
    if ( !v30 )
      goto LABEL_76;
  }
LABEL_36:
  v32 = *(__int64 **)(v31 + 40);
  if ( v176 != v32 )
  {
    sub_D695C0((__int64)&v205, (__int64)&v199, *(__int64 **)(v31 + 40), v9, v10, v11);
    if ( v210[0] )
    {
      if ( v32 != v175 )
        sub_B1A4E0((__int64)&v196, (__int64)v32);
    }
  }
  while ( 1 )
  {
    v30 = *(_QWORD *)(v30 + 8);
    if ( !v30 )
      break;
    v31 = *(_QWORD *)(v30 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v31 - 30) <= 0xAu )
      goto LABEL_36;
  }
  if ( HIDWORD(v201) == v202 )
  {
LABEL_76:
    v205 = 0;
    v206 = v210;
    v207 = 16;
    v208 = 0;
    v209 = 1;
  }
  else
  {
    sub_D695C0((__int64)&v205, (__int64)&v199, v175, v9, v10, v11);
    v35 = v197;
    if ( (_DWORD)v197 )
    {
      v167 = v14;
      do
      {
        v36 = v35--;
        v37 = *(_QWORD *)&v196[8 * v36 - 8];
        LODWORD(v197) = v35;
        v38 = *(_QWORD *)(v37 + 16);
        if ( v38 )
        {
          while ( 1 )
          {
            v39 = *(_QWORD *)(v38 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v39 - 30) <= 0xAu )
              break;
            v38 = *(_QWORD *)(v38 + 8);
            if ( !v38 )
              goto LABEL_52;
          }
LABEL_48:
          v40 = *(__int64 **)(v39 + 40);
          v190 = v40;
          if ( (_DWORD)v213 )
          {
            if ( !(_DWORD)v214 )
              goto LABEL_50;
            v96 = (v214 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
            v42 = *(_QWORD *)(v212 + 8LL * v96);
            if ( v40 != (__int64 *)v42 )
            {
              for ( i = 1; ; ++i )
              {
                if ( v42 == -4096 )
                  goto LABEL_50;
                v33 = (unsigned int)(i + 1);
                v96 = (v214 - 1) & (i + v96);
                v42 = *(_QWORD *)(v212 + 8LL * v96);
                if ( v40 == (__int64 *)v42 )
                  break;
              }
            }
          }
          else
          {
            v41 = &v215[(unsigned int)v216];
            if ( v41 == sub_2957650(v215, (__int64)v41, (__int64 *)&v190) )
              goto LABEL_50;
          }
          sub_D695C0((__int64)&v205, (__int64)&v199, v40, v42, v33, v34);
          if ( v210[0] )
            sub_B1A4E0((__int64)&v196, (__int64)v40);
LABEL_50:
          while ( 1 )
          {
            v38 = *(_QWORD *)(v38 + 8);
            if ( !v38 )
              break;
            v39 = *(_QWORD *)(v38 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v39 - 30) <= 0xAu )
              goto LABEL_48;
          }
          v35 = v197;
        }
LABEL_52:
        ;
      }
      while ( v35 );
      v14 = v167;
    }
    v43 = (char *)sub_2965130((_QWORD *)a5);
    v44 = v43;
    if ( v14 )
    {
      sub_D4F330(v14, (__int64)v176, a5);
      v205 = v44;
      *(_QWORD *)v44 = v14;
      sub_D4C980((__int64)(v14 + 1), &v205);
    }
    else
    {
      v205 = v43;
      sub_D4C980(a5 + 32, &v205);
    }
    sub_295C830(a6, (__int64)v44, v45, v46, v47, v48);
    sub_D4ACD0((__int64)(v44 + 32), HIDWORD(v201) - v202);
    v162 = *(__int64 **)(a1 + 40);
    for ( j = *(__int64 **)(a1 + 32); v162 != j; ++j )
    {
      v49 = *j;
      v190 = (__int64 *)*j;
      sub_D696B0((unsigned __int64 *)&v205, a4, (__int64 *)&v190);
      v50 = v207;
      if ( v207 )
      {
        sub_D68D70(&v205);
        if ( (unsigned __int8)sub_B19060((__int64)&v199, v50, v51, v52) )
        {
          v54 = *(_DWORD *)(a5 + 24);
          v55 = *(_QWORD *)(a5 + 8);
          if ( !v54 )
            goto LABEL_62;
          v56 = v54 - 1;
          v57 = v56 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
          v58 = (__int64 *)(v55 + 16LL * v57);
          v59 = *v58;
          if ( v49 != *v58 )
          {
            v158 = 1;
            while ( v59 != -4096 )
            {
              v53 = (unsigned int)(v158 + 1);
              v57 = v56 & (v158 + v57);
              v58 = (__int64 *)(v55 + 16LL * v57);
              v59 = *v58;
              if ( v49 == *v58 )
                goto LABEL_61;
              v158 = v53;
            }
LABEL_62:
            if ( v44 )
            {
              v60 = (__int64)(v44 + 32);
              for ( k = v44; ; v60 = (__int64)(k + 32) )
              {
                v190 = (__int64 *)v50;
                v62 = (_BYTE *)*((_QWORD *)k + 5);
                if ( v62 == *((_BYTE **)k + 6) )
                {
                  sub_9319A0(v60, v62, &v190);
                  v63 = v190;
                }
                else
                {
                  if ( v62 )
                  {
                    *(_QWORD *)v62 = v50;
                    v62 = (_BYTE *)*((_QWORD *)k + 5);
                  }
                  v63 = (__int64 *)v50;
                  *((_QWORD *)k + 5) = v62 + 8;
                }
                sub_26C2C80((__int64)&v205, (__int64)(k + 56), v63, v53, v10, v11);
                k = *(char **)k;
                if ( !k )
                  break;
              }
            }
            continue;
          }
LABEL_61:
          if ( a1 != v58[1] )
            goto LABEL_62;
          sub_D4F330((__int64 *)v44, v50, a5);
        }
      }
      else
      {
        sub_D68D70(&v205);
      }
    }
    v98 = *(__int64 **)(a1 + 8);
    for ( m = *(__int64 **)(a1 + 16); m != v98; ++v98 )
    {
      while ( 1 )
      {
        v101 = *v98;
        v190 = **(__int64 ***)(*v98 + 32);
        sub_D696B0((unsigned __int64 *)&v205, a4, (__int64 *)&v190);
        if ( v207 )
          break;
        ++v98;
        sub_D68D70(&v205);
        if ( m == v98 )
          goto LABEL_136;
      }
      v164 = v207;
      sub_D68D70(&v205);
      if ( (unsigned __int8)sub_B19060((__int64)&v199, v164, v99, v100) )
        sub_29651B0(v101, (__int64)v44, a4, (_QWORD *)a5);
    }
LABEL_136:
    v66 = v202;
    v9 = (__int64)v210;
    v205 = 0;
    v206 = v210;
    v207 = 16;
    v208 = 0;
    v209 = 1;
    if ( v202 != HIDWORD(v201) )
      goto LABEL_78;
  }
  sub_D695C0((__int64)&v190, (__int64)&v205, v176, v9, v10, v11);
LABEL_78:
  v67 = v215;
  v68 = &v215[(unsigned int)v216];
  if ( v68 != v215 )
  {
    do
    {
      while ( 1 )
      {
        v69 = (__int64 *)*v67;
        if ( !(unsigned __int8)sub_B19060((__int64)&v199, *v67, v66, v9) )
          break;
        if ( v68 == ++v67 )
          goto LABEL_83;
      }
      ++v67;
      sub_D695C0((__int64)&v190, (__int64)&v205, v69, v9, v10, v11);
    }
    while ( v68 != v67 );
  }
LABEL_83:
  v70 = v178;
  v180 = v182;
  v181 = 0x400000000LL;
  if ( (_DWORD)v178 )
  {
    v148 = (__int64 *)v182;
    v149 = 8LL * (unsigned int)v178;
    if ( (unsigned int)v178 <= 4
      || (sub_C8D5F0((__int64)&v180, v182, (unsigned int)v178, 8u, v10, v11),
          v148 = (__int64 *)v180,
          (v149 = 8LL * (unsigned int)v178) != 0) )
    {
      memcpy(v148, src, v149);
      v148 = (__int64 *)v180;
    }
    v150 = v70;
    LODWORD(v181) = v70;
    v151 = &v148[v150];
    _BitScanReverse64(&v152, (v150 * 8) >> 3);
    v170 = &v148[v150];
    sub_2959530(v148, (char *)&v148[v150], 2LL * (int)(63 - (v152 ^ 0x3F)), (__int64)&v218);
    if ( (unsigned __int64)v150 <= 16 )
    {
      sub_2959170(v148, v170, (__int64)&v218);
    }
    else
    {
      v153 = v148 + 16;
      sub_2959170(v148, v148 + 16, (__int64)&v218);
      if ( v151 != v148 + 16 )
      {
        do
        {
          v154 = *v153;
          for ( n = v153; ; --n )
          {
            v156 = *(n - 1);
            if ( !sub_2959010((__int64)&v218, v154, v156) )
              break;
            *n = v156;
          }
          *n = v154;
          ++v153;
        }
        while ( v170 != v153 );
      }
    }
  }
  while ( HIDWORD(v207) != v208 )
  {
    if ( !(_DWORD)v181 )
      break;
    v71 = (__int64 *)*((_QWORD *)v180 + (unsigned int)v181 - 1);
    LODWORD(v181) = v181 - 1;
    if ( (v219 & 1) != 0 )
    {
      v72 = 15;
      v73 = &v220;
    }
    else
    {
      v73 = (__int64 **)v220;
      if ( !v221 )
        goto LABEL_202;
      v72 = v221 - 1;
    }
    v74 = v72 & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
    v75 = &v73[2 * v74];
    v76 = *v75;
    if ( v71 != *v75 )
    {
      v143 = 1;
      while ( v76 != (__int64 *)-4096LL )
      {
        v10 = (unsigned int)(v143 + 1);
        v74 = v72 & (v143 + v74);
        v75 = &v73[2 * v74];
        v76 = *v75;
        if ( v71 == *v75 )
          goto LABEL_89;
        v143 = v10;
      }
LABEL_202:
      v77 = 0;
      goto LABEL_90;
    }
LABEL_89:
    v77 = v75[1];
LABEL_90:
    v78 = (unsigned int)v197;
    v79 = (unsigned int)v197 + 1LL;
    if ( v79 > HIDWORD(v197) )
    {
      sub_C8D5F0((__int64)&v196, v198, v79, 8u, v10, v11);
      v78 = (unsigned int)v197;
    }
    *(_QWORD *)&v196[8 * v78] = v71;
    v80 = v197 + 1;
    LODWORD(v197) = v197 + 1;
LABEL_94:
    while ( 2 )
    {
      v81 = v80--;
      v82 = *(__int64 **)&v196[8 * v81 - 8];
      LODWORD(v197) = v80;
      if ( v176 == v82 )
        goto LABEL_93;
      v83 = v82[2];
      if ( !v83 )
        goto LABEL_93;
      while ( 1 )
      {
        v84 = *(_QWORD *)(v83 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v84 - 30) <= 0xAu )
          break;
        v83 = *(_QWORD *)(v83 + 8);
        if ( !v83 )
        {
          if ( !v80 )
            goto LABEL_140;
          goto LABEL_94;
        }
      }
      v85 = *(_QWORD *)(v84 + 40);
      if ( v209 )
      {
LABEL_98:
        v86 = &v206[8 * HIDWORD(v207)];
        v87 = v206;
        if ( v206 != (_BYTE *)v86 )
        {
          while ( v85 != *v87 )
          {
            if ( v86 == ++v87 )
              goto LABEL_109;
          }
          --HIDWORD(v207);
          *v87 = *(_QWORD *)&v206[8 * HIDWORD(v207)];
          ++v205;
LABEL_103:
          v190 = (__int64 *)v85;
          v191 = v77;
          if ( (v219 & 1) != 0 )
          {
            v88 = 15;
            v89 = &v220;
          }
          else
          {
            v136 = v221;
            v89 = (__int64 **)v220;
            if ( !v221 )
            {
              v137 = v219;
              ++v218;
              v189[0] = 0;
              v138 = ((unsigned int)v219 >> 1) + 1;
              goto LABEL_183;
            }
            v88 = v221 - 1;
          }
          v90 = v88 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
          v91 = (__int64 *)&v89[2 * v90];
          v10 = *v91;
          if ( *v91 == v85 )
            goto LABEL_106;
          v142 = 1;
          v11 = 0;
          while ( v10 != -4096 )
          {
            if ( v11 || v10 != -8192 )
              v91 = (__int64 *)v11;
            v11 = (unsigned int)(v142 + 1);
            v90 = v88 & (v142 + v90);
            v10 = (__int64)v89[2 * v90];
            if ( v85 == v10 )
              goto LABEL_106;
            ++v142;
            v11 = (__int64)v91;
            v91 = (__int64 *)&v89[2 * v90];
          }
          if ( !v11 )
            v11 = (__int64)v91;
          v137 = v219;
          ++v218;
          v189[0] = v11;
          v138 = ((unsigned int)v219 >> 1) + 1;
          if ( (v219 & 1) == 0 )
          {
            v136 = v221;
LABEL_183:
            if ( 3 * v136 <= 4 * v138 )
              goto LABEL_194;
            goto LABEL_184;
          }
          v136 = 16;
          if ( 4 * v138 >= 0x30 )
          {
LABEL_194:
            v136 *= 2;
LABEL_195:
            sub_295E3B0((__int64)&v218, v136);
            sub_295D140((__int64)&v218, (__int64 *)&v190, v189);
            v140 = (__int64)v190;
            v137 = v219;
            goto LABEL_185;
          }
LABEL_184:
          v139 = v136 - HIDWORD(v219) - v138;
          v140 = v85;
          if ( v139 <= v136 >> 3 )
            goto LABEL_195;
LABEL_185:
          LODWORD(v219) = (2 * (v137 >> 1) + 2) | v137 & 1;
          v141 = v189[0];
          if ( *(_QWORD *)v189[0] != -4096 )
            --HIDWORD(v219);
          *(_QWORD *)v189[0] = v140;
          *(_QWORD *)(v141 + 8) = v191;
LABEL_106:
          v92 = (unsigned int)v197;
          v93 = (unsigned int)v197 + 1LL;
          if ( v93 > HIDWORD(v197) )
          {
            sub_C8D5F0((__int64)&v196, v198, v93, 8u, v10, v11);
            v92 = (unsigned int)v197;
          }
          *(_QWORD *)&v196[8 * v92] = v85;
          LODWORD(v197) = v197 + 1;
        }
        goto LABEL_109;
      }
LABEL_112:
      v95 = sub_C8CA60((__int64)&v205, v85);
      if ( v95 )
      {
        *v95 = -2;
        ++v208;
        ++v205;
        goto LABEL_103;
      }
LABEL_109:
      while ( 1 )
      {
        v83 = *(_QWORD *)(v83 + 8);
        if ( !v83 )
          break;
        v94 = *(_QWORD *)(v83 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v94 - 30) <= 0xAu )
        {
          v85 = *(_QWORD *)(v94 + 40);
          if ( v209 )
            goto LABEL_98;
          goto LABEL_112;
        }
      }
      v80 = v197;
LABEL_93:
      if ( v80 )
        continue;
      break;
    }
LABEL_140:
    ;
  }
  v185 = (void **)&v176;
  v183 = (char *)src;
  v184 = v215;
  v172 = (char *)src + 8 * (unsigned int)v178;
  v102 = &v215[(unsigned int)v216];
  v186 = v172;
  v187 = v102;
  p_src = &src;
  do
  {
    v103 = v189;
    v189[3] = 0;
    v189[5] = 0;
    v104 = (void **)&v183;
    v189[2] = (__int64)sub_2957410;
    v105 = v189;
    v189[4] = (__int64)sub_2957430;
    v106 = sub_29573F0;
    if ( ((unsigned __int8)sub_29573F0 & 1) != 0 )
LABEL_143:
      v106 = *(__int64 (__fastcall **)(__int64))((char *)v106 + (_QWORD)*v104 - 1);
    v107 = (__int64 *)v106((__int64)v104);
    if ( !v107 )
    {
      while ( 1 )
      {
        v103 += 2;
        if ( &v190 == (__int64 **)v103 )
          break;
        v108 = v105[3];
        v106 = (__int64 (__fastcall *)(__int64))v105[2];
        v105 = v103;
        v104 = (void **)((char *)&v183 + v108);
        if ( ((unsigned __int8)v106 & 1) != 0 )
          goto LABEL_143;
        v107 = (__int64 *)v106((__int64)v104);
        if ( v107 )
          goto LABEL_148;
      }
LABEL_260:
      BUG();
    }
LABEL_148:
    v109 = *v107;
    if ( (v219 & 1) != 0 )
    {
      v110 = &v220;
      v111 = 15;
    }
    else
    {
      v110 = (__int64 **)v220;
      if ( !v221 )
        goto LABEL_153;
      v111 = v221 - 1;
    }
    v112 = v111 & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
    v113 = &v110[2 * v112];
    v114 = *v113;
    if ( (__int64 *)v109 == *v113 )
    {
LABEL_151:
      v115 = v113[1];
      if ( v115 )
        sub_D4F330(v115, v109, a5);
    }
    else
    {
      v144 = 1;
      while ( v114 != (__int64 *)-4096LL )
      {
        v145 = v144 + 1;
        v112 = v111 & (v144 + v112);
        v113 = &v110[2 * v112];
        v114 = *v113;
        if ( (__int64 *)v109 == *v113 )
          goto LABEL_151;
        v144 = v145;
      }
    }
LABEL_153:
    v116 = (__int64 *)&v190;
    v193 = 0;
    v195 = 0;
    v117 = (void **)&v183;
    v192 = sub_2957390;
    v118 = (__int64 *)&v190;
    v194 = sub_29573C0;
    v119 = sub_2957360;
    if ( ((unsigned __int8)sub_2957360 & 1) != 0 )
LABEL_154:
      v119 = *(__int64 (__fastcall **)(__int64))((char *)v119 + (_QWORD)*v117 - 1);
    while ( !(unsigned __int8)v119((__int64)v117) )
    {
      v116 += 2;
      if ( &v196 == (_BYTE **)v116 )
        goto LABEL_260;
      v120 = v118[3];
      v119 = (__int64 (__fastcall *)(__int64))v118[2];
      v118 = v116;
      v117 = (void **)((char *)&v183 + v120);
      if ( ((unsigned __int8)v119 & 1) != 0 )
        goto LABEL_154;
    }
  }
  while ( v185 != &src || v102 != v184 || v172 != v183 || p_src != &src || v102 != v187 || v172 != v186 );
  v121 = *(__int64 **)(a1 + 8);
  if ( *(__int64 **)(a1 + 16) != v121 )
  {
    v173 = *(__int64 **)(a1 + 16);
    while ( 1 )
    {
      while ( 1 )
      {
        v132 = *v121;
        v189[0] = **(_QWORD **)(*v121 + 32);
        sub_D696B0((unsigned __int64 *)&v190, a4, v189);
        v133 = (__int64)v192;
        if ( v192 )
          break;
        sub_D68D70(&v190);
        if ( v173 == ++v121 )
          goto LABEL_234;
      }
      sub_D68D70(&v190);
      if ( !(unsigned __int8)sub_B19060((__int64)&v199, v133, v134, v135) )
        break;
LABEL_173:
      if ( v173 == ++v121 )
        goto LABEL_234;
    }
    if ( (v219 & 1) != 0 )
    {
      v122 = &v220;
      v123 = 15;
      goto LABEL_168;
    }
    v122 = (__int64 **)v220;
    if ( v221 )
    {
      v123 = v221 - 1;
LABEL_168:
      v124 = v123 & (((unsigned int)v133 >> 9) ^ ((unsigned int)v133 >> 4));
      v125 = &v122[2 * v124];
      v126 = *v125;
      if ( (__int64 *)v133 == *v125 )
      {
LABEL_169:
        v127 = (__int64)v125[1];
LABEL_170:
        v130 = sub_29651B0(v132, v127, a4, (_QWORD *)a5);
        v131 = *(unsigned int *)(a6 + 8);
        if ( v131 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
        {
          sub_C8D5F0(a6, (const void *)(a6 + 16), v131 + 1, 8u, v128, v129);
          v131 = *(unsigned int *)(a6 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a6 + 8 * v131) = v130;
        ++*(_DWORD *)(a6 + 8);
        goto LABEL_173;
      }
      v157 = 1;
      while ( v126 != (__int64 *)-4096LL )
      {
        v159 = v157 + 1;
        v124 = v123 & (v157 + v124);
        v125 = &v122[2 * v124];
        v126 = *v125;
        if ( (__int64 *)v133 == *v125 )
          goto LABEL_169;
        v157 = v159;
      }
    }
    v127 = 0;
    goto LABEL_170;
  }
LABEL_234:
  if ( v180 != v182 )
    _libc_free((unsigned __int64)v180);
  if ( !v209 )
    _libc_free((unsigned __int64)v206);
  if ( !v203 )
    _libc_free((unsigned __int64)v200);
  if ( v196 != v198 )
    _libc_free((unsigned __int64)v196);
  if ( v215 != (__int64 *)v217 )
    _libc_free((unsigned __int64)v215);
  sub_C7D6A0(v212, 8LL * (unsigned int)v214, 8);
  if ( (v219 & 1) == 0 )
    sub_C7D6A0((__int64)v220, 16LL * v221, 8);
  if ( src != v179 )
    _libc_free((unsigned __int64)src);
}
