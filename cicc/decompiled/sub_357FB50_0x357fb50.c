// Function: sub_357FB50
// Address: 0x357fb50
//
__int64 __fastcall sub_357FB50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int64 *v9; // r12
  __int64 v10; // rax
  unsigned __int64 **v11; // r15
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rax
  __int64 *v16; // rsi
  __int64 *v17; // r14
  unsigned __int64 v18; // rbx
  __int64 *v19; // r13
  __int64 v20; // rbx
  __int64 v21; // rax
  int v22; // edx
  int v23; // r8d
  __int64 v24; // rsi
  __int64 *v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rbx
  __int64 *v29; // rax
  __int64 *v30; // rsi
  unsigned __int64 v31; // r9
  __int64 *v32; // r8
  unsigned __int64 **v33; // rcx
  __int64 *v34; // r13
  unsigned __int64 v35; // r15
  __int64 v36; // r14
  __int64 v37; // r12
  unsigned int v38; // ebx
  int v39; // esi
  __int64 v40; // rax
  int v41; // edx
  char *v42; // rsi
  unsigned __int64 p_src; // rdi
  __int64 *v44; // r12
  __int64 *v45; // r13
  __int64 v46; // r14
  int v47; // eax
  int v48; // eax
  __int64 v49; // rax
  __int64 *v50; // rcx
  __int64 v51; // rax
  unsigned int v52; // edx
  __int64 v53; // r8
  __int64 v54; // r11
  int v55; // edx
  unsigned __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // r10
  __int64 v59; // rsi
  __int64 v60; // rax
  __int64 v61; // rbx
  unsigned int v62; // r8d
  unsigned int v63; // esi
  __int64 v64; // r14
  unsigned int v65; // edx
  _QWORD *v66; // rax
  void **v67; // r9
  unsigned __int64 v68; // r10
  unsigned int v69; // ecx
  __int64 v70; // rax
  unsigned __int64 v71; // r9
  unsigned int v72; // r9d
  __int64 *v73; // rdx
  __int64 *v74; // rax
  __int64 *v75; // rbx
  __int64 *v76; // r14
  bool v77; // zf
  unsigned __int64 *v78; // rax
  unsigned __int64 *v79; // rsi
  unsigned __int64 v80; // rdx
  _QWORD *v81; // r8
  __int64 *v82; // rax
  __int64 v83; // rax
  unsigned __int64 v84; // rcx
  __int64 v85; // rdx
  __int64 *v86; // r12
  __int64 v87; // rbx
  __int64 v88; // r13
  int v89; // ebx
  __int64 v90; // rax
  unsigned __int64 v91; // rdx
  unsigned __int64 *v92; // r8
  unsigned __int64 *v93; // rax
  unsigned __int64 v94; // rsi
  unsigned __int64 v95; // rcx
  _BYTE *v96; // rax
  _BYTE *v97; // rsi
  __int64 *v98; // rdx
  __int64 v99; // rax
  __int64 *v100; // rcx
  size_t v101; // r15
  _BYTE *v102; // rax
  unsigned __int64 v103; // r12
  __int64 v104; // rax
  __int64 *v105; // rcx
  size_t v106; // r14
  char v107; // bl
  char v108; // al
  __int64 v109; // rsi
  bool v110; // r9
  char v111; // r10
  _BYTE *v112; // rax
  _BYTE *v113; // r8
  char v114; // dl
  _BYTE *v116; // rsi
  __int64 *v117; // r10
  unsigned __int64 v118; // rax
  __int64 v119; // rcx
  __int64 v120; // rdx
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // r12
  _BYTE *v125; // rsi
  int v126; // r15d
  __int64 *v127; // r12
  __int64 v128; // rdi
  unsigned __int64 *v129; // r12
  __int64 v130; // rax
  unsigned __int64 v131; // rdx
  __int64 v132; // [rsp+8h] [rbp-238h]
  unsigned int v133; // [rsp+10h] [rbp-230h]
  __int64 *v134; // [rsp+10h] [rbp-230h]
  unsigned int v135; // [rsp+18h] [rbp-228h]
  unsigned __int64 v136; // [rsp+18h] [rbp-228h]
  unsigned __int64 v137; // [rsp+28h] [rbp-218h]
  __int64 v138; // [rsp+38h] [rbp-208h]
  unsigned int v139; // [rsp+40h] [rbp-200h]
  int v140; // [rsp+44h] [rbp-1FCh]
  __int64 v141; // [rsp+48h] [rbp-1F8h]
  int v142; // [rsp+50h] [rbp-1F0h]
  char v143; // [rsp+56h] [rbp-1EAh]
  unsigned __int8 v144; // [rsp+57h] [rbp-1E9h]
  __int64 v145; // [rsp+58h] [rbp-1E8h]
  unsigned __int64 v146; // [rsp+60h] [rbp-1E0h]
  __int64 v147; // [rsp+68h] [rbp-1D8h]
  unsigned __int64 **v148; // [rsp+68h] [rbp-1D8h]
  unsigned __int64 *v149; // [rsp+70h] [rbp-1D0h]
  __int64 v150; // [rsp+78h] [rbp-1C8h]
  __int64 v151; // [rsp+78h] [rbp-1C8h]
  unsigned __int8 v152; // [rsp+78h] [rbp-1C8h]
  __int64 *v153; // [rsp+78h] [rbp-1C8h]
  __int64 *v154; // [rsp+78h] [rbp-1C8h]
  unsigned __int64 **v155; // [rsp+80h] [rbp-1C0h]
  unsigned int v156; // [rsp+80h] [rbp-1C0h]
  unsigned __int64 **v157; // [rsp+80h] [rbp-1C0h]
  unsigned int v158; // [rsp+80h] [rbp-1C0h]
  __int64 v159; // [rsp+88h] [rbp-1B8h]
  __int64 v160; // [rsp+90h] [rbp-1B0h] BYREF
  __int64 v161; // [rsp+98h] [rbp-1A8h] BYREF
  __int64 v162; // [rsp+A0h] [rbp-1A0h] BYREF
  int v163; // [rsp+A8h] [rbp-198h]
  __int64 *v164; // [rsp+B0h] [rbp-190h] BYREF
  __int64 *v165; // [rsp+B8h] [rbp-188h]
  __int64 *v166; // [rsp+C0h] [rbp-180h]
  void *src; // [rsp+D0h] [rbp-170h] BYREF
  _BYTE *v168; // [rsp+D8h] [rbp-168h]
  _BYTE *v169; // [rsp+E0h] [rbp-160h]
  unsigned __int64 v170; // [rsp+F0h] [rbp-150h] BYREF
  char *v171; // [rsp+F8h] [rbp-148h]
  char *v172; // [rsp+100h] [rbp-140h]
  __int64 *v173; // [rsp+110h] [rbp-130h] BYREF
  __int64 *v174; // [rsp+118h] [rbp-128h]
  char *v175; // [rsp+120h] [rbp-120h]
  __int64 *v176; // [rsp+130h] [rbp-110h] BYREF
  __int64 *v177; // [rsp+138h] [rbp-108h]
  char *v178; // [rsp+140h] [rbp-100h]
  _QWORD v179[2]; // [rsp+150h] [rbp-F0h] BYREF
  __int64 (__fastcall *v180)(_QWORD *, _QWORD *, int); // [rsp+160h] [rbp-E0h]
  __int64 (__fastcall *v181)(__int64); // [rsp+168h] [rbp-D8h]
  __int64 *v182; // [rsp+170h] [rbp-D0h] BYREF
  __int64 (__fastcall *v183)(unsigned __int64 **, unsigned __int64 **, int); // [rsp+180h] [rbp-C0h]
  __int64 (__fastcall *v184)(__int64); // [rsp+188h] [rbp-B8h]
  __int64 *v185; // [rsp+190h] [rbp-B0h] BYREF
  __int64 *v186; // [rsp+198h] [rbp-A8h] BYREF
  unsigned __int64 *v187; // [rsp+1A0h] [rbp-A0h]
  __int64 **v188; // [rsp+1A8h] [rbp-98h]
  __int64 **v189; // [rsp+1B0h] [rbp-90h]
  __int64 v190; // [rsp+1B8h] [rbp-88h]
  unsigned __int64 *v191; // [rsp+1C0h] [rbp-80h] BYREF
  __int64 v192; // [rsp+1C8h] [rbp-78h] BYREF
  _BYTE *v193; // [rsp+1D0h] [rbp-70h] BYREF
  __int64 *v194; // [rsp+1D8h] [rbp-68h]
  __int64 *v195; // [rsp+1E0h] [rbp-60h]
  __int64 v196; // [rsp+1E8h] [rbp-58h]

  if ( a1 + 40 == (_QWORD *)(a1[40] & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  v6 = a1[41];
  v192 = 0x800000000LL;
  v191 = (unsigned __int64 *)&v193;
  sub_357E7D0((__int64)&v191, v6, (__int64)(a1 + 40), a4, a5, a6);
  v7 = (unsigned __int64)v191;
  v8 = (unsigned int)v192;
  v9 = &v191[v8];
  if ( v191 == &v191[v8] )
  {
    v141 = 0;
    v137 = 0;
  }
  else
  {
    v137 = 0;
    v141 = 0;
    if ( v8 * 8 )
    {
      v129 = v9 - 1;
      v130 = sub_22077B0(8LL * (unsigned int)v192);
      v137 = v130;
      v141 = v130 + v8 * 8;
      do
      {
        v131 = *v129;
        v130 += 8;
        --v129;
        *(_QWORD *)(v130 - 8) = v131;
      }
      while ( v130 != v141 );
      v7 = (unsigned __int64)v191;
    }
    v9 = (unsigned __int64 *)v7;
  }
  if ( v9 != (unsigned __int64 *)&v193 )
    _libc_free((unsigned __int64)v9);
  v10 = a1[4];
  v163 = 0;
  v162 = v10;
  if ( v137 == v141 )
  {
    v144 = 0;
    goto LABEL_214;
  }
  v144 = 0;
  v142 = 0;
  v146 = v137;
  v11 = (unsigned __int64 **)&v182;
  do
  {
    v140 = v142++;
    v145 = *(_QWORD *)v146;
    v12 = *(_QWORD *)v146 + 48LL;
    v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v146 + 32LL) + 32LL);
    v185 = 0;
    v186 = 0;
    v187 = 0;
    v14 = *(_QWORD *)(v145 + 56);
    v159 = v13;
    v15 = v14;
    if ( v145 + 48 == v14 )
    {
      v143 = 0;
      goto LABEL_32;
    }
    v16 = 0;
    do
    {
      while ( *(_WORD *)(v14 + 68) != 20 )
      {
LABEL_11:
        v14 = *(_QWORD *)(v14 + 8);
        if ( v12 == v14 )
          goto LABEL_17;
      }
      v191 = (unsigned __int64 *)v14;
      if ( v187 == (unsigned __int64 *)v16 )
      {
        sub_2E26050((__int64)&v185, v16, &v191);
        v16 = v186;
        goto LABEL_11;
      }
      if ( v16 )
      {
        *v16 = v14;
        v16 = v186;
      }
      v186 = ++v16;
      v14 = *(_QWORD *)(v14 + 8);
    }
    while ( v12 != v14 );
LABEL_17:
    v17 = v185;
    v18 = (unsigned __int64)v16;
    if ( v185 == v16 )
    {
      v143 = 0;
    }
    else
    {
      v143 = 0;
      v155 = v11;
      v150 = v12;
      v19 = v16;
      do
      {
        v20 = *v17;
        v21 = *(_QWORD *)(*v17 + 32);
        if ( !*(_BYTE *)v21 && !*(_BYTE *)(v21 + 40) )
        {
          v22 = *(_DWORD *)(v21 + 8);
          if ( v22 < 0 )
          {
            v23 = *(_DWORD *)(v21 + 48);
            if ( v23 < 0 )
            {
              v24 = *(_QWORD *)(v159 + 56);
              v25 = (__int64 *)(v24 + 16LL * (v22 & 0x7FFFFFFF));
              v26 = *v25;
              if ( *v25 )
              {
                if ( (v26 & 4) == 0 )
                {
                  v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v27 )
                  {
                    if ( v27 == (*(_QWORD *)(v24 + 16LL * (v23 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) )
                    {
                      v191 = 0;
                      v192 = 0;
                      v193 = 0;
                      v124 = v25[1];
                      if ( v124 )
                      {
                        if ( (*(_BYTE *)(v124 + 3) & 0x10) != 0 )
                        {
                          while ( 1 )
                          {
                            v124 = *(_QWORD *)(v124 + 32);
                            if ( !v124 )
                              break;
                            if ( (*(_BYTE *)(v124 + 3) & 0x10) == 0 )
                              goto LABEL_253;
                          }
                        }
                        else
                        {
LABEL_253:
                          v182 = (__int64 *)v124;
                          v125 = 0;
                          v126 = v23;
LABEL_268:
                          sub_357DB10((__int64)&v191, v125, v155);
                          v125 = (_BYTE *)v192;
                          while ( 1 )
                          {
                            v124 = *(_QWORD *)(v124 + 32);
                            if ( !v124 )
                              break;
                            if ( (*(_BYTE *)(v124 + 3) & 0x10) == 0 )
                            {
                              v182 = (__int64 *)v124;
                              if ( v125 == v193 )
                                goto LABEL_268;
                              if ( v125 )
                              {
                                *(_QWORD *)v125 = v124;
                                v125 = (_BYTE *)v192;
                              }
                              v125 += 8;
                              v192 = (__int64)v125;
                            }
                          }
                          v127 = (__int64 *)v191;
                          if ( v125 != (_BYTE *)v191 )
                          {
                            do
                            {
                              v128 = *v127++;
                              sub_2EAB0C0(v128, v126);
                            }
                            while ( v125 != (_BYTE *)v127 );
                          }
                        }
                      }
                      sub_2E88E20(v20);
                      if ( v191 )
                        j_j___libc_free_0((unsigned __int64)v191);
                      v143 = 1;
                    }
                  }
                }
              }
            }
          }
        }
        ++v17;
      }
      while ( v19 != v17 );
      v18 = (unsigned __int64)v185;
      v12 = v150;
      v11 = v155;
    }
    if ( v18 )
      j_j___libc_free_0(v18);
    v15 = *(_QWORD *)(v145 + 56);
LABEL_32:
    v28 = v15;
    v164 = 0;
    v165 = 0;
    v160 = v145;
    v166 = 0;
    if ( v15 != v12 )
    {
      v29 = 0;
      v30 = 0;
      while ( 1 )
      {
        v191 = (unsigned __int64 *)v28;
        if ( v30 == v29 )
        {
          sub_2E26050((__int64)&v164, v30, &v191);
        }
        else
        {
          if ( v30 )
          {
            *v30 = v28;
            v30 = v165;
          }
          v165 = v30 + 1;
        }
        if ( !v28 )
          goto LABEL_292;
        if ( (*(_BYTE *)v28 & 4) != 0 )
        {
          v28 = *(_QWORD *)(v28 + 8);
          v30 = v165;
          if ( v12 == v28 )
            goto LABEL_43;
        }
        else
        {
          while ( (*(_BYTE *)(v28 + 44) & 8) != 0 )
            v28 = *(_QWORD *)(v28 + 8);
          v28 = *(_QWORD *)(v28 + 8);
          v30 = v165;
          if ( v12 == v28 )
          {
LABEL_43:
            v31 = (unsigned __int64)v164;
            v32 = v30;
            goto LABEL_44;
          }
        }
        v29 = v166;
      }
    }
    v32 = 0;
    v31 = 0;
LABEL_44:
    LODWORD(v186) = 0;
    v188 = &v186;
    v189 = &v186;
    v187 = 0;
    v190 = 0;
    LODWORD(v192) = 0;
    v193 = 0;
    v194 = &v192;
    v195 = &v192;
    v196 = 0;
    src = 0;
    v168 = 0;
    v169 = 0;
    v170 = 0;
    v171 = 0;
    v172 = 0;
    if ( (__int64 *)v31 == v32 )
    {
      v152 = 0;
      v87 = v160;
      goto LABEL_186;
    }
    v151 = v12;
    v33 = v11;
    v34 = v32;
    v35 = v31;
    do
    {
      v36 = *(_QWORD *)v35;
      v37 = 40;
      v38 = 1;
      v39 = *(_DWORD *)(*(_QWORD *)v35 + 40LL);
      if ( (v39 & 0xFFFFFFu) > 1 )
      {
        do
        {
          v40 = v37 + *(_QWORD *)(v36 + 32);
          if ( !*(_BYTE *)v40 )
          {
            v41 = *(_DWORD *)(v40 + 8);
            if ( v41 >= 0 && (*(_BYTE *)(v40 + 3) & 0x10) != 0 )
            {
              LODWORD(v182) = *(_DWORD *)(v40 + 8);
              v42 = v171;
              if ( v171 == v172 )
              {
                v157 = v33;
                sub_34B5B30(&v170, v171, v33);
                v39 = *(_DWORD *)(v36 + 40);
                v33 = v157;
              }
              else
              {
                if ( v171 )
                {
                  *(_DWORD *)v171 = v41;
                  v42 = v171;
                }
                v171 = v42 + 4;
                v39 = *(_DWORD *)(v36 + 40);
              }
            }
          }
          ++v38;
          v37 += 40;
        }
        while ( (v39 & 0xFFFFFFu) > v38 );
      }
      v35 += 8LL;
    }
    while ( v34 != (__int64 *)v35 );
    v11 = v33;
    v12 = v151;
    p_src = (unsigned __int64)v165;
    if ( v164 == v165 )
    {
      v152 = 0;
      v86 = v194;
      goto LABEL_160;
    }
    v147 = v151;
    v44 = v164;
    v45 = v165;
    v156 = 0;
    v152 = 0;
    while ( 2 )
    {
      while ( 2 )
      {
        v46 = *v44;
        v173 = (__int64 *)v46;
        if ( (*(_DWORD *)(v46 + 40) & 0xFFFFFF) == 0
          || (unsigned int)*(unsigned __int16 *)(v46 + 68) - 1 <= 1
          && (*(_BYTE *)(*(_QWORD *)(v46 + 32) + 64LL) & 8) != 0 )
        {
          goto LABEL_60;
        }
        v47 = *(_DWORD *)(v46 + 44);
        if ( (v47 & 4) != 0 || (v47 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v46 + 16) + 24LL) & 0x80000LL) != 0 )
            goto LABEL_60;
        }
        else
        {
          p_src = v46;
          if ( sub_2E88A90(v46, 0x80000, 1) )
            goto LABEL_60;
        }
        if ( (unsigned int)*(unsigned __int16 *)(v46 + 68) - 1 <= 1
          && (*(_BYTE *)(*(_QWORD *)(v46 + 32) + 64LL) & 0x10) != 0 )
        {
          goto LABEL_60;
        }
        v48 = *(_DWORD *)(v46 + 44);
        if ( (v48 & 4) != 0 || (v48 & 8) == 0 )
        {
          v49 = (*(_QWORD *)(*(_QWORD *)(v46 + 16) + 24LL) >> 20) & 1LL;
        }
        else
        {
          p_src = v46;
          LOBYTE(v49) = sub_2E88A90(v46, 0x100000, 1);
        }
        if ( (_BYTE)v49 )
          goto LABEL_60;
        v50 = v173;
        v51 = v173[4];
        if ( *(_BYTE *)v51 )
          goto LABEL_60;
        p_src = *(unsigned int *)(v51 + 8);
        if ( (p_src & 0x80000000) == 0LL || (*(_BYTE *)(v51 + 3) & 0x10) == 0 )
          goto LABEL_60;
        v52 = v173[5] & 0xFFFFFF;
        if ( v52 <= 1 )
        {
LABEL_232:
          v116 = v168;
          if ( v168 == v169 )
          {
            p_src = (unsigned __int64)&src;
            sub_2E997F0((__int64)&src, v168, &v173);
          }
          else
          {
            if ( v168 )
            {
              *(_QWORD *)v168 = v173;
              v116 = v168;
            }
            v168 = v116 + 8;
          }
LABEL_60:
          if ( v45 == ++v44 )
            goto LABEL_159;
          continue;
        }
        break;
      }
      v53 = v51 + 40;
      v54 = v51 + 40LL * (v52 - 2) + 80;
      while ( 2 )
      {
        if ( *(_BYTE *)v53 == 1 )
          goto LABEL_231;
        if ( !*(_BYTE *)v53 )
        {
          v55 = *(_DWORD *)(v53 + 8);
          if ( v55 >= 0 )
          {
            v56 = v170;
            v57 = (__int64)&v171[-v170] >> 4;
            v58 = (__int64)&v171[-v170] >> 2;
            if ( v57 > 0 )
            {
              v59 = v170 + 16 * v57;
              while ( v55 != *(_DWORD *)v56 )
              {
                if ( v55 == *(_DWORD *)(v56 + 4) )
                {
                  v56 += 4LL;
                  goto LABEL_87;
                }
                if ( v55 == *(_DWORD *)(v56 + 8) )
                {
                  v56 += 8LL;
                  goto LABEL_87;
                }
                if ( v55 == *(_DWORD *)(v56 + 12) )
                {
                  v56 += 12LL;
                  goto LABEL_87;
                }
                v56 += 16LL;
                if ( v59 == v56 )
                {
                  v58 = (__int64)&v171[-v56] >> 2;
                  goto LABEL_227;
                }
              }
              goto LABEL_87;
            }
LABEL_227:
            switch ( v58 )
            {
              case 2LL:
LABEL_278:
                if ( v55 != *(_DWORD *)v56 )
                {
                  v56 += 4LL;
LABEL_230:
                  if ( v55 != *(_DWORD *)v56 )
                  {
LABEL_231:
                    v53 += 40;
                    if ( v54 == v53 )
                      goto LABEL_232;
                    continue;
                  }
                }
                break;
              case 3LL:
                if ( v55 != *(_DWORD *)v56 )
                {
                  v56 += 4LL;
                  goto LABEL_278;
                }
                break;
              case 1LL:
                goto LABEL_230;
              default:
                goto LABEL_231;
            }
LABEL_87:
            if ( v171 != (char *)v56 )
              break;
            goto LABEL_231;
          }
        }
        break;
      }
      v60 = v160;
      v176 = v173;
      v179[0] = 0;
      v61 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v160 + 32) + 32LL) + 56LL) + 16 * (p_src & 0x7FFFFFFF) + 8);
      if ( v61 )
      {
        while ( (*(_BYTE *)(v61 + 3) & 0x10) != 0 || (*(_BYTE *)(v61 + 4) & 8) != 0 )
        {
          v61 = *(_QWORD *)(v61 + 32);
          if ( !v61 )
            goto LABEL_128;
        }
        v62 = v156;
        v63 = -1;
LABEL_94:
        p_src = v50[3];
        v64 = *(_QWORD *)(v61 + 16);
        v65 = 0;
        v66 = *(_QWORD **)(p_src + 56);
        v67 = (void **)(p_src + 48);
        if ( v66 == (_QWORD *)(p_src + 48) )
        {
LABEL_101:
          v65 = -1;
        }
        else
        {
          while ( v50 != v66 )
          {
            ++v65;
            if ( !v66 )
              goto LABEL_292;
            if ( (*(_BYTE *)v66 & 4) != 0 )
            {
              v66 = (_QWORD *)v66[1];
              if ( v67 == v66 )
                goto LABEL_101;
            }
            else
            {
              while ( (*((_BYTE *)v66 + 44) & 8) != 0 )
                v66 = (_QWORD *)v66[1];
              v66 = (_QWORD *)v66[1];
              if ( v67 == v66 )
                goto LABEL_101;
            }
          }
        }
        v68 = *(_QWORD *)(v64 + 24);
        v69 = 0;
        v70 = *(_QWORD *)(v68 + 56);
        v71 = v68 + 48;
        if ( v70 == v68 + 48 )
        {
LABEL_109:
          v69 = -1;
        }
        else
        {
          while ( v64 != v70 )
          {
            ++v69;
            if ( !v70 )
              goto LABEL_292;
            if ( (*(_BYTE *)v70 & 4) != 0 )
            {
              v70 = *(_QWORD *)(v70 + 8);
              if ( v71 == v70 )
                goto LABEL_109;
            }
            else
            {
              while ( (*(_BYTE *)(v70 + 44) & 8) != 0 )
                v70 = *(_QWORD *)(v70 + 8);
              v70 = *(_QWORD *)(v70 + 8);
              if ( v71 == v70 )
                goto LABEL_109;
            }
          }
        }
        v72 = v69 - v65;
        if ( v68 == p_src && v69 > v65 && v72 < v63 )
        {
          v179[0] = *(_QWORD *)(v61 + 16);
          v117 = &v192;
          v158 = v62 + 1;
          v118 = (unsigned __int64)v193;
          if ( !v193 )
            goto LABEL_245;
          do
          {
            while ( 1 )
            {
              v119 = *(_QWORD *)(v118 + 16);
              v120 = *(_QWORD *)(v118 + 24);
              if ( *(_DWORD *)(v118 + 32) >= v62 )
                break;
              v118 = *(_QWORD *)(v118 + 24);
              if ( !v120 )
                goto LABEL_243;
            }
            v117 = (__int64 *)v118;
            v118 = *(_QWORD *)(v118 + 16);
          }
          while ( v119 );
LABEL_243:
          if ( v117 == &v192 || *((_DWORD *)v117 + 8) > v62 )
          {
LABEL_245:
            v139 = v72;
            v132 = (__int64)v117;
            v135 = v62;
            v121 = sub_22077B0(0x30u);
            *(_QWORD *)(v121 + 40) = 0;
            *(_DWORD *)(v121 + 32) = v135;
            v133 = v135;
            v136 = v121;
            v122 = sub_357DD40(&v191, v132, (unsigned int *)(v121 + 32));
            if ( v123 )
            {
              p_src = &v192 == (__int64 *)v123 || v122 || *(_DWORD *)(v123 + 32) > v133;
              sub_220F040(p_src, v136, (_QWORD *)v123, &v192);
              ++v196;
              v117 = (__int64 *)v136;
              v72 = v139;
            }
            else
            {
              p_src = v136;
              v134 = (__int64 *)v122;
              j_j___libc_free_0(v136);
              v72 = v139;
              v117 = v134;
            }
          }
          v117[5] = v64;
          v62 = v158;
          v63 = v72;
        }
        while ( 1 )
        {
          v61 = *(_QWORD *)(v61 + 32);
          if ( !v61 )
            break;
          if ( (*(_BYTE *)(v61 + 3) & 0x10) == 0 && (*(_BYTE *)(v61 + 4) & 8) == 0 )
          {
            v50 = v176;
            goto LABEL_94;
          }
        }
        v156 = v62;
        v60 = v160;
      }
LABEL_128:
      v73 = (__int64 *)(v60 + 48);
      v74 = *(__int64 **)(v60 + 56);
      if ( v73 == v74 )
        goto LABEL_60;
      v75 = v73;
      v76 = v73;
      while ( 2 )
      {
        if ( v176 == v74 )
        {
          v74 = (__int64 *)v74[1];
          v76 = v176;
          if ( v73 == v74 )
            goto LABEL_223;
LABEL_134:
          if ( v73 != v75 && v73 != v76 )
            goto LABEL_136;
          continue;
        }
        break;
      }
      v77 = v179[0] == (_QWORD)v74;
      v74 = (__int64 *)v74[1];
      if ( v77 )
        v75 = (__int64 *)v179[0];
      if ( v73 != v74 )
        goto LABEL_134;
LABEL_223:
      if ( v73 == v75 || v73 == v76 )
        goto LABEL_60;
LABEL_136:
      v78 = v187;
      v79 = (unsigned __int64 *)&v186;
      if ( !v187 )
        goto LABEL_143;
      do
      {
        while ( 1 )
        {
          p_src = v78[2];
          v80 = v78[3];
          if ( v78[4] >= v179[0] )
            break;
          v78 = (unsigned __int64 *)v78[3];
          if ( !v80 )
            goto LABEL_141;
        }
        v79 = v78;
        v78 = (unsigned __int64 *)v78[2];
      }
      while ( p_src );
LABEL_141:
      if ( v79 == (unsigned __int64 *)&v186 || v79[4] > v179[0] )
      {
LABEL_143:
        p_src = (unsigned __int64)&v185;
        v182 = v179;
        v79 = sub_357DFE0(&v185, (__int64)v79, v11);
      }
      v81 = (_QWORD *)v79[6];
      if ( v81 == (_QWORD *)v79[7] )
      {
        p_src = (unsigned __int64)(v79 + 5);
        sub_2E997F0((__int64)(v79 + 5), (_BYTE *)v79[6], &v176);
      }
      else
      {
        if ( v81 )
        {
          *v81 = v176;
          v81 = (_QWORD *)v79[6];
        }
        v79[6] = (unsigned __int64)(v81 + 1);
      }
      if ( v76 != v75 )
      {
        if ( v76 )
        {
          v82 = v76;
          if ( (*(_BYTE *)v76 & 4) == 0 && (*((_BYTE *)v76 + 44) & 8) != 0 )
          {
            do
              v82 = (__int64 *)v82[1];
            while ( (*((_BYTE *)v82 + 44) & 8) != 0 );
          }
          v83 = v82[1];
          p_src = v160;
          if ( v76 != (__int64 *)v83 && v75 != (__int64 *)v83 )
          {
            p_src = v160 + 40;
            v153 = (__int64 *)v83;
            sub_2E310C0((__int64 *)p_src, (__int64 *)p_src, (__int64)v76, v83);
            if ( v153 != v75 && v153 != v76 )
            {
              v84 = *v153 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)((*v76 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v153;
              *v153 = *v153 & 7 | *v76 & 0xFFFFFFFFFFFFFFF8LL;
              v85 = *v75;
              *(_QWORD *)(v84 + 8) = v75;
              v85 &= 0xFFFFFFFFFFFFFFF8LL;
              *v76 = v85 | *v76 & 7;
              *(_QWORD *)(v85 + 8) = v76;
              *v75 = v84 | *v75 & 7;
            }
          }
          goto LABEL_158;
        }
LABEL_292:
        BUG();
      }
LABEL_158:
      ++v44;
      v152 = 1;
      if ( v45 != v44 )
        continue;
      break;
    }
LABEL_159:
    v12 = v147;
    v86 = v194;
LABEL_160:
    v87 = v160;
    if ( v86 != &v192 )
    {
      v148 = v11;
      v138 = v12;
      v88 = v160;
      v89 = v152;
      do
      {
        v90 = *(_QWORD *)(v88 + 56);
        if ( v88 + 48 != v90 )
        {
          v91 = v86[5];
          while ( v91 != v90 )
          {
            v90 = *(_QWORD *)(v90 + 8);
            if ( v88 + 48 == v90 )
              goto LABEL_184;
          }
          v161 = v86[5];
          v92 = (unsigned __int64 *)&v186;
          v179[0] = &v161;
          v181 = sub_357C450;
          v180 = sub_357C690;
          v93 = v187;
          if ( !v187 )
            goto LABEL_173;
          do
          {
            while ( 1 )
            {
              v94 = v93[2];
              v95 = v93[3];
              if ( v93[4] >= v91 )
                break;
              v93 = (unsigned __int64 *)v93[3];
              if ( !v95 )
                goto LABEL_171;
            }
            v92 = v93;
            v93 = (unsigned __int64 *)v93[2];
          }
          while ( v94 );
LABEL_171:
          if ( v92 == (unsigned __int64 *)&v186 || v92[4] > v91 )
          {
LABEL_173:
            p_src = (unsigned __int64)&v185;
            v182 = v86 + 5;
            v92 = sub_357DFE0(&v185, (__int64)v92, v148);
          }
          v96 = (_BYTE *)v92[6];
          v97 = (_BYTE *)v92[5];
          v173 = 0;
          v174 = 0;
          v175 = 0;
          v98 = (__int64 *)(v96 - v97);
          if ( v96 == v97 )
          {
            v101 = 0;
            v100 = 0;
          }
          else
          {
            if ( (unsigned __int64)v98 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_290;
            v149 = v92;
            v154 = (__int64 *)(v96 - v97);
            v99 = sub_22077B0((unsigned __int64)v98);
            v98 = v154;
            v100 = (__int64 *)v99;
            v96 = (_BYTE *)v149[6];
            v97 = (_BYTE *)v149[5];
            v101 = v96 - v97;
          }
          v173 = v100;
          v174 = v100;
          v175 = (char *)v98 + (_QWORD)v100;
          if ( v96 != v97 )
            v100 = (__int64 *)memmove(v100, v97, v101);
          v174 = (__int64 *)((char *)v100 + v101);
          v89 |= sub_357F410(&v173, v88, (__int64)v179);
          if ( v173 )
            j_j___libc_free_0((unsigned __int64)v173);
          if ( v180 )
            v180(v179, v179, 3);
          v88 = v160;
        }
LABEL_184:
        p_src = (unsigned __int64)v86;
        v86 = (__int64 *)sub_220EEE0((__int64)v86);
      }
      while ( v86 != &v192 );
      v152 = v89;
      v11 = v148;
      v87 = v88;
      v12 = v138;
    }
LABEL_186:
    v102 = v168;
    v97 = src;
    v98 = &v160;
    v184 = sub_357C460;
    p_src = (unsigned __int64)sub_357C6C0;
    v182 = &v160;
    v183 = (__int64 (__fastcall *)(unsigned __int64 **, unsigned __int64 **, int))sub_357C6C0;
    v176 = 0;
    v177 = 0;
    v178 = 0;
    v103 = v168 - (_BYTE *)src;
    if ( v168 == src )
    {
      v106 = 0;
      v105 = 0;
    }
    else
    {
      if ( v103 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_290:
        sub_4261EA(p_src, v97, v98);
      v104 = sub_22077B0(v168 - (_BYTE *)src);
      v97 = src;
      v105 = (__int64 *)v104;
      v102 = v168;
      v106 = v168 - (_BYTE *)src;
    }
    v176 = v105;
    v177 = v105;
    v178 = (char *)v105 + v103;
    if ( v102 != v97 )
      v105 = (__int64 *)memmove(v105, v97, v106);
    v177 = (__int64 *)((char *)v105 + v106);
    v107 = sub_357F410(&v176, v87, (__int64)v11);
    if ( v176 )
      j_j___libc_free_0((unsigned __int64)v176);
    if ( v183 )
      v183(v11, v11, 3);
    if ( v170 )
      j_j___libc_free_0(v170);
    if ( src )
      j_j___libc_free_0((unsigned __int64)src);
    sub_357C6F0((unsigned __int64)v193);
    sub_357C8C0(v187);
    if ( v164 )
      j_j___libc_free_0((unsigned __int64)v164);
    v163 = v140;
    v108 = sub_3592050(&v162, v145);
    v109 = *(_QWORD *)(v145 + 56);
    v110 = 0;
    v111 = v108;
    if ( v109 != v12 )
    {
      while ( 1 )
      {
        v112 = *(_BYTE **)(v109 + 32);
        v113 = &v112[40 * (*(_DWORD *)(v109 + 40) & 0xFFFFFF)];
        if ( v112 != v113 )
          break;
LABEL_211:
        if ( (*(_BYTE *)v109 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v109 + 44) & 8) != 0 )
            v109 = *(_QWORD *)(v109 + 8);
        }
        v109 = *(_QWORD *)(v109 + 8);
        if ( v109 == v12 )
          goto LABEL_213;
      }
      while ( 1 )
      {
LABEL_207:
        if ( *v112 )
          goto LABEL_206;
        v114 = v112[3];
        if ( (v114 & 0x10) == 0 )
          break;
        if ( (v112[3] & 0x40) != 0 )
          goto LABEL_205;
        v112 += 40;
        if ( v113 == v112 )
          goto LABEL_211;
      }
      if ( (v112[3] & 0x40) != 0 )
      {
LABEL_205:
        v110 = (v112[3] & 0x40) != 0;
        v112[3] = v114 & 0xBF;
      }
LABEL_206:
      v112 += 40;
      if ( v113 == v112 )
        goto LABEL_211;
      goto LABEL_207;
    }
LABEL_213:
    v146 += 8LL;
    v144 |= v110 | v111 | v107 | v143 | v152;
  }
  while ( v141 != v146 );
LABEL_214:
  if ( v137 )
    j_j___libc_free_0(v137);
  return v144;
}
