// Function: sub_20C5FB0
// Address: 0x20c5fb0
//
__int64 __fastcall sub_20C5FB0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // rbx
  _BOOL8 v12; // rdi
  __int64 v13; // r13
  unsigned __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r14
  _QWORD *v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // edx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // r13
  unsigned __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // r14
  _QWORD *v28; // rbx
  __int64 v29; // r15
  __int64 v30; // rax
  unsigned __int64 v31; // r14
  __int64 v32; // r15
  _QWORD *v33; // rax
  _QWORD *v34; // rdx
  __int64 v35; // rax
  __int64 i; // r14
  int v38; // r8d
  int *v39; // rax
  int *v40; // rsi
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 v45; // rbx
  __int64 v46; // r14
  __int64 v47; // rax
  unsigned int v48; // r15d
  __int64 v49; // rdx
  unsigned __int64 *v50; // rdi
  unsigned __int64 *v51; // rsi
  unsigned __int64 *v52; // rax
  _BOOL4 v53; // r14d
  __int64 v54; // rax
  bool v55; // zf
  unsigned int v56; // eax
  unsigned int *v57; // rbx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r15
  __int64 v61; // rax
  __int64 v62; // rdx
  _BOOL4 v63; // r15d
  __int64 v64; // rax
  _BYTE *v65; // rsi
  __int64 v66; // rsi
  __int64 v67; // r8
  int v68; // r9d
  unsigned int v69; // edx
  _QWORD *v70; // rax
  __int64 v71; // rdx
  _QWORD *v72; // r10
  __int64 v73; // rax
  __int64 v74; // rdx
  _BOOL4 v75; // r9d
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rdx
  _QWORD *v79; // r14
  _QWORD *v80; // r15
  __int64 v81; // rbx
  __int64 v82; // rdx
  __int64 v83; // r12
  __int64 v84; // rax
  __int64 (*v85)(); // rax
  unsigned int v86; // r13d
  __int64 v87; // rax
  __int64 v88; // rax
  int *v89; // rsi
  __int64 v90; // rcx
  __int64 v91; // rdx
  unsigned int v92; // eax
  __int64 v93; // rax
  __int64 v94; // rcx
  __int64 v95; // rdi
  unsigned __int64 v96; // r12
  __int64 v97; // rdx
  __int64 v98; // rsi
  __int64 v99; // rdx
  _QWORD *v100; // rdx
  unsigned int v101; // ecx
  __int16 v102; // ax
  _WORD *v103; // rcx
  __int64 *v104; // rsi
  unsigned __int64 v105; // rcx
  __int64 v106; // r8
  __int64 v107; // rax
  __int64 v108; // rdi
  __int64 v109; // r8
  unsigned __int64 v110; // rdx
  __int64 v111; // rax
  __int16 *v112; // rax
  __int16 v113; // cx
  __int16 *v114; // rax
  unsigned __int16 v115; // dx
  __int16 *v116; // rsi
  __int16 v117; // cx
  __int16 *v118; // rax
  __int64 v119; // rax
  unsigned int v120; // esi
  unsigned int v121; // edx
  __int64 v122; // rbx
  __int64 v123; // rax
  _QWORD *v124; // rsi
  __int64 v125; // rcx
  _QWORD *v126; // rdx
  unsigned int v127; // r14d
  int v128; // r13d
  unsigned __int64 v129; // r12
  unsigned int v130; // eax
  __int64 v131; // r15
  unsigned int v132; // r12d
  __int64 v133; // rax
  __int64 v134; // rdx
  __int64 v135; // r14
  int v136; // r15d
  int *v137; // r9
  unsigned __int64 v138; // rsi
  int *v139; // rax
  __int64 v140; // rcx
  __int64 v141; // rdx
  __int64 v142; // r9
  __int64 v143; // r10
  __int64 v144; // rax
  __int64 v145; // rcx
  __int64 v146; // rdx
  __int64 v147; // rdi
  __int16 v148; // ax
  __int64 v149; // rax
  __int64 v150; // [rsp+0h] [rbp-250h]
  __int64 v151; // [rsp+8h] [rbp-248h]
  __int64 v152; // [rsp+10h] [rbp-240h]
  __int64 v153; // [rsp+18h] [rbp-238h]
  __int64 v154; // [rsp+20h] [rbp-230h]
  _QWORD *v155; // [rsp+30h] [rbp-220h]
  _QWORD *v156; // [rsp+40h] [rbp-210h]
  __int64 v157; // [rsp+50h] [rbp-200h]
  size_t n; // [rsp+60h] [rbp-1F0h]
  __int64 v160; // [rsp+68h] [rbp-1E8h]
  _QWORD *v161; // [rsp+70h] [rbp-1E0h]
  unsigned int v162; // [rsp+78h] [rbp-1D8h]
  unsigned int v163; // [rsp+7Ch] [rbp-1D4h]
  char *s; // [rsp+80h] [rbp-1D0h]
  unsigned __int64 v165; // [rsp+88h] [rbp-1C8h]
  __int64 v166; // [rsp+98h] [rbp-1B8h]
  __int64 v167; // [rsp+A0h] [rbp-1B0h]
  __int64 v168; // [rsp+A0h] [rbp-1B0h]
  _BOOL4 v169; // [rsp+A0h] [rbp-1B0h]
  __int64 *v170; // [rsp+A0h] [rbp-1B0h]
  __int64 v171; // [rsp+A8h] [rbp-1A8h]
  __int64 v172; // [rsp+B0h] [rbp-1A0h]
  __int64 v173; // [rsp+B0h] [rbp-1A0h]
  unsigned __int64 v174; // [rsp+B8h] [rbp-198h]
  __int64 v176; // [rsp+C8h] [rbp-188h]
  int v177; // [rsp+D0h] [rbp-180h]
  _QWORD *v180; // [rsp+E0h] [rbp-170h]
  unsigned __int64 v181; // [rsp+E0h] [rbp-170h]
  __int64 v182; // [rsp+E0h] [rbp-170h]
  __int64 *v183; // [rsp+E8h] [rbp-168h]
  __int64 v184; // [rsp+E8h] [rbp-168h]
  int v185; // [rsp+E8h] [rbp-168h]
  __int64 v186; // [rsp+F8h] [rbp-158h]
  unsigned int v187; // [rsp+F8h] [rbp-158h]
  __int64 v188; // [rsp+F8h] [rbp-158h]
  __int64 v189; // [rsp+F8h] [rbp-158h]
  _QWORD *v190; // [rsp+F8h] [rbp-158h]
  unsigned int v191; // [rsp+108h] [rbp-148h] BYREF
  unsigned int v192; // [rsp+10Ch] [rbp-144h] BYREF
  unsigned __int64 v193; // [rsp+110h] [rbp-140h] BYREF
  unsigned __int64 *v194; // [rsp+118h] [rbp-138h] BYREF
  __int64 v195; // [rsp+120h] [rbp-130h] BYREF
  _BYTE *v196; // [rsp+128h] [rbp-128h]
  _BYTE *v197; // [rsp+130h] [rbp-120h]
  __int64 v198; // [rsp+140h] [rbp-110h] BYREF
  int v199; // [rsp+148h] [rbp-108h] BYREF
  __int64 v200; // [rsp+150h] [rbp-100h]
  int *v201; // [rsp+158h] [rbp-F8h]
  int *v202; // [rsp+160h] [rbp-F0h]
  __int64 v203; // [rsp+168h] [rbp-E8h]
  __int64 v204; // [rsp+170h] [rbp-E0h] BYREF
  int v205; // [rsp+178h] [rbp-D8h] BYREF
  int *v206; // [rsp+180h] [rbp-D0h]
  int *v207; // [rsp+188h] [rbp-C8h]
  int *v208; // [rsp+190h] [rbp-C0h]
  __int64 v209; // [rsp+198h] [rbp-B8h]
  char v210[8]; // [rsp+1A0h] [rbp-B0h] BYREF
  int v211; // [rsp+1A8h] [rbp-A8h] BYREF
  __int64 v212; // [rsp+1B0h] [rbp-A0h]
  int *v213; // [rsp+1B8h] [rbp-98h]
  int *v214; // [rsp+1C0h] [rbp-90h]
  __int64 v215; // [rsp+1C8h] [rbp-88h]
  unsigned __int64 *v216; // [rsp+1D0h] [rbp-80h] BYREF
  __int64 v217; // [rsp+1D8h] [rbp-78h] BYREF
  __int64 v218; // [rsp+1E0h] [rbp-70h] BYREF
  __int64 *v219; // [rsp+1E8h] [rbp-68h]
  __int64 *v220; // [rsp+1F0h] [rbp-60h] BYREF
  __int64 v221; // [rsp+1F8h] [rbp-58h] BYREF
  __int64 v222; // [rsp+200h] [rbp-50h]
  __int64 *v223; // [rsp+208h] [rbp-48h]
  __int64 *v224; // [rsp+210h] [rbp-40h]
  __int64 v225; // [rsp+218h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)a2;
  v183 = (__int64 *)a2;
  v177 = a5;
  if ( *(_QWORD *)a2 == v6 )
  {
    return 0;
  }
  else
  {
    v8 = a1;
    v9 = *(_QWORD *)(a1 + 72);
    v201 = &v199;
    v202 = &v199;
    v207 = &v205;
    v208 = &v205;
    v10 = 0xF0F0F0F0F0F0F0F1LL * ((v6 - v7) >> 4);
    v157 = v9;
    v199 = 0;
    v200 = 0;
    v203 = 0;
    v205 = 0;
    v206 = 0;
    v209 = 0;
    if ( !(_DWORD)v10 )
      goto LABEL_12;
    v11 = 0;
    v186 = 272LL * (unsigned int)(v10 - 1);
    while ( 1 )
    {
      v13 = v7 + v11;
      v14 = *(_QWORD *)(v7 + v11 + 8);
      v15 = sub_22077B0(48);
      *(_QWORD *)(v15 + 32) = v14;
      v16 = v15;
      *(_QWORD *)(v15 + 40) = v13;
      v17 = sub_20C4E40((__int64)&v204, (unsigned __int64 *)(v15 + 32));
      if ( v18 )
        break;
      a2 = 48;
      j_j___libc_free_0(v16, 48);
      if ( v186 == v11 )
        goto LABEL_11;
LABEL_8:
      v11 += 272;
      v7 = *v183;
    }
    v12 = v17 || (int *)v18 == &v205 || v14 < *(_QWORD *)(v18 + 32);
    a2 = v16;
    sub_220F040(v12, v16, v18, &v205);
    ++v209;
    if ( v186 != v11 )
      goto LABEL_8;
LABEL_11:
    v8 = a1;
LABEL_12:
    v161 = (_QWORD *)(v8 + 48);
    v19 = (unsigned int)(*(_DWORD *)(v8 + 64) + 63) >> 6;
    if ( v19 )
    {
      v20 = *(_QWORD **)(v8 + 48);
      v21 = (__int64)&v20[v19];
      while ( !*v20 )
      {
        if ( (_QWORD *)v21 == ++v20 )
          goto LABEL_29;
      }
      v22 = (__int64)v183;
      v23 = 0xF0F0F0F0F0F0F0F1LL;
      v24 = *v183;
      v25 = 0xF0F0F0F0F0F0F0F1LL * ((v183[1] - *v183) >> 4);
      if ( !(_DWORD)v25 )
        BUG();
      v184 = v8;
      v26 = 272;
      v27 = 272LL * (unsigned int)v25;
      v28 = (_QWORD *)v22;
      while ( v27 != v26 )
      {
        v29 = v26 + *v28;
        if ( v24 )
        {
          if ( (*(_BYTE *)(v29 + 236) & 1) == 0 )
            sub_1F01DD0(v26 + *v28, (_QWORD *)a2, v23, v22, a5, (int)a6);
          v23 = *(_DWORD *)(v29 + 240) + (unsigned int)*(unsigned __int16 *)(v29 + 226);
          if ( (*(_BYTE *)(v24 + 236) & 1) == 0 )
          {
            v187 = *(_DWORD *)(v29 + 240) + *(unsigned __int16 *)(v29 + 226);
            sub_1F01DD0(v24, (_QWORD *)a2, v23, v22, a5, (int)a6);
            v23 = v187;
          }
          if ( (unsigned int)v23 > *(_DWORD *)(v24 + 240) + (unsigned int)*(unsigned __int16 *)(v24 + 226) )
            v24 = v29;
        }
        else
        {
          v24 = v26 + *v28;
        }
        v26 += 272;
      }
      v165 = v24;
      v8 = v184;
      v171 = *(_QWORD *)(v24 + 8);
    }
    else
    {
LABEL_29:
      v171 = 0;
      v165 = 0;
    }
    v162 = (unsigned int)(*(_DWORD *)(*(_QWORD *)(v8 + 32) + 16LL) + 63) >> 6;
    n = 8LL * v162;
    s = (char *)malloc(n);
    if ( !s )
    {
      if ( n || (v149 = malloc(1u)) == 0 )
        sub_16BD1C0("Allocation failed", 1u);
      else
        s = (char *)v149;
    }
    if ( v162 )
      memset(s, 0, n);
    v163 = 0;
    v185 = v177 - 1;
    if ( a4 != a3 )
    {
      v30 = v8;
      v31 = a4;
      v32 = v30;
      while ( 1 )
      {
        v33 = (_QWORD *)(*(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL);
        v34 = v33;
        if ( !v33 )
          BUG();
        v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
        v35 = *v33;
        if ( (v35 & 4) == 0 && (*((_BYTE *)v34 + 46) & 4) != 0 )
        {
          for ( i = v35; ; i = *(_QWORD *)v31 )
          {
            v31 = i & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v31 + 46) & 4) == 0 )
              break;
          }
        }
        if ( (unsigned __int16)(**(_WORD **)(v31 + 16) - 12) <= 1u )
          goto LABEL_42;
        v213 = &v211;
        v214 = &v211;
        v211 = 0;
        v212 = 0;
        v215 = 0;
        sub_20C47C0(v32, v31, (__int64)v210);
        sub_20C3910((__int64 *)v32, v31, v185, (__int64)v210);
        v39 = v206;
        v195 = 0;
        v194 = (unsigned __int64 *)v31;
        v40 = &v205;
        v196 = 0;
        v197 = 0;
        if ( !v206 )
          goto LABEL_52;
        do
        {
          while ( 1 )
          {
            v41 = *((_QWORD *)v39 + 2);
            v42 = *((_QWORD *)v39 + 3);
            if ( *((_QWORD *)v39 + 4) >= v31 )
              break;
            v39 = (int *)*((_QWORD *)v39 + 3);
            if ( !v42 )
              goto LABEL_50;
          }
          v40 = v39;
          v39 = (int *)*((_QWORD *)v39 + 2);
        }
        while ( v41 );
LABEL_50:
        if ( v40 == &v205 || *((_QWORD *)v40 + 4) > v31 )
        {
LABEL_52:
          v216 = (unsigned __int64 *)&v194;
          v40 = (int *)sub_20C4FE0(&v204, v40, &v216);
        }
        v43 = *((_QWORD *)v40 + 5);
        LODWORD(v221) = 0;
        v216 = (unsigned __int64 *)&v218;
        v217 = 0x400000000LL;
        v222 = 0;
        v223 = &v221;
        v224 = &v221;
        v225 = 0;
        v44 = *(_QWORD *)(v43 + 32);
        v166 = v43;
        v45 = 16LL * *(unsigned int *)(v43 + 40);
        v188 = v44 + v45;
        if ( v44 != v44 + v45 )
        {
          v174 = v31;
          v46 = *(_QWORD *)(v43 + 32);
          v172 = v32;
          while ( 1 )
          {
            while ( 1 )
            {
              v47 = (*(__int64 *)v46 >> 1) & 3;
              if ( v47 != 2 && v47 != 1 )
                goto LABEL_56;
              v48 = *(_DWORD *)(v46 + 8);
              LODWORD(v194) = v48;
              if ( v225 )
                break;
              v49 = (unsigned int)v217;
              v50 = v216;
              v51 = (unsigned __int64 *)((char *)v216 + 4 * (unsigned int)v217);
              if ( v216 != v51 )
              {
                v52 = v216;
                while ( v48 != *(_DWORD *)v52 )
                {
                  v52 = (unsigned __int64 *)((char *)v52 + 4);
                  if ( v51 == v52 )
                    goto LABEL_64;
                }
                if ( v51 != v52 )
                  goto LABEL_56;
              }
LABEL_64:
              if ( (unsigned int)v217 <= 3uLL )
              {
                if ( (unsigned int)v217 >= HIDWORD(v217) )
                {
                  sub_16CD150((__int64)&v216, &v218, 0, 4, v38, v217);
                  v48 = (unsigned int)v194;
                  v51 = (unsigned __int64 *)((char *)v216 + 4 * (unsigned int)v217);
                }
                *(_DWORD *)v51 = v48;
                LODWORD(v217) = v217 + 1;
              }
              else
              {
                v167 = v46;
                while ( 1 )
                {
                  v57 = (unsigned int *)v50 + v49 - 1;
                  v58 = sub_B996D0((__int64)&v220, v57);
                  v60 = v59;
                  if ( v59 )
                  {
                    v53 = v58 || (__int64 *)v59 == &v221 || *v57 < *(_DWORD *)(v59 + 32);
                    v54 = sub_22077B0(40);
                    *(_DWORD *)(v54 + 32) = *v57;
                    sub_220F040(v53, v54, v60, &v221);
                    ++v225;
                  }
                  v55 = (_DWORD)v217 == 1;
                  v56 = v217 - 1;
                  LODWORD(v217) = v217 - 1;
                  if ( v55 )
                    break;
                  v50 = v216;
                  v49 = v56;
                }
                v46 = v167;
                v73 = sub_B996D0((__int64)&v220, (unsigned int *)&v194);
                if ( v74 )
                {
                  v75 = v73 || (__int64 *)v74 == &v221 || (unsigned int)v194 < *(_DWORD *)(v74 + 32);
                  v160 = v74;
                  v169 = v75;
                  v76 = sub_22077B0(40);
                  *(_DWORD *)(v76 + 32) = (_DWORD)v194;
                  sub_220F040(v169, v76, v160, &v221);
                  ++v225;
                }
              }
LABEL_79:
              v194 = (unsigned __int64 *)v46;
              v65 = v196;
              if ( v196 == v197 )
              {
                sub_20C4A60((__int64)&v195, v196, &v194);
                goto LABEL_56;
              }
              if ( v196 )
              {
                *(_QWORD *)v196 = v46;
                v65 = v196;
              }
              v46 += 16;
              v196 = v65 + 8;
              if ( v188 == v46 )
              {
LABEL_83:
                v31 = v174;
                v32 = v172;
                v66 = v222;
                goto LABEL_84;
              }
            }
            v61 = sub_B996D0((__int64)&v220, (unsigned int *)&v194);
            if ( v62 )
            {
              v63 = v61 || (__int64 *)v62 == &v221 || v48 < *(_DWORD *)(v62 + 32);
              v168 = v62;
              v64 = sub_22077B0(40);
              *(_DWORD *)(v64 + 32) = (_DWORD)v194;
              sub_220F040(v63, v64, v168, &v221);
              ++v225;
              goto LABEL_79;
            }
LABEL_56:
            v46 += 16;
            if ( v188 == v46 )
              goto LABEL_83;
          }
        }
        v66 = 0;
LABEL_84:
        sub_20C31F0((__int64)&v220, v66);
        if ( v216 != (unsigned __int64 *)&v218 )
          _libc_free((unsigned __int64)v216);
        if ( v171 != v31 )
          break;
        if ( !v165 )
          goto LABEL_191;
        v122 = *(_QWORD *)(v165 + 32);
        v123 = 16LL * *(unsigned int *)(v165 + 40);
        v124 = (_QWORD *)(v122 + v123);
        if ( v122 == v122 + v123 )
          goto LABEL_191;
        v125 = 0;
        v181 = v31;
        v126 = 0;
        v127 = 0;
        do
        {
          while ( 1 )
          {
            v128 = *(_DWORD *)(v122 + 12);
            v129 = *(_QWORD *)v122 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v129 + 236) & 1) == 0 )
            {
              v190 = v126;
              sub_1F01DD0(*(_QWORD *)v122 & 0xFFFFFFFFFFFFFFF8LL, v124, (__int64)v126, v125, v67, v68);
              v126 = v190;
            }
            v130 = v128 + *(_DWORD *)(v129 + 240);
            if ( v127 >= v130 )
              break;
            v126 = (_QWORD *)v122;
            v122 += 16;
            v127 = v128 + *(_DWORD *)(v129 + 240);
            if ( v124 == (_QWORD *)v122 )
              goto LABEL_183;
          }
          if ( v127 == v130 && ((*(__int64 *)v122 >> 1) & 3) == 1 )
            v126 = (_QWORD *)v122;
          v122 += 16;
        }
        while ( v124 != (_QWORD *)v122 );
LABEL_183:
        v31 = v181;
        if ( v126 && (v165 = *v126 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v72 = 0;
          v171 = *(_QWORD *)((*v126 & 0xFFFFFFFFFFFFFFF8LL) + 8);
        }
        else
        {
LABEL_191:
          v171 = 0;
          v72 = 0;
          v165 = 0;
        }
LABEL_92:
        if ( **(_WORD **)(v31 + 16) != 6 )
          goto LABEL_102;
LABEL_93:
        sub_20C3FC0(v32, v31, v185);
        if ( v195 )
          j_j___libc_free_0(v195, &v197[-v195]);
        sub_20C31F0((__int64)v210, v212);
LABEL_42:
        --v185;
        if ( a3 == v31 )
          goto LABEL_43;
      }
      v69 = (unsigned int)(*(_DWORD *)(v32 + 64) + 63) >> 6;
      if ( v69 )
      {
        v70 = *(_QWORD **)(v32 + 48);
        v71 = (__int64)&v70[v69];
        while ( !*v70 )
        {
          if ( (_QWORD *)v71 == ++v70 )
            goto LABEL_101;
        }
        v72 = v161;
        goto LABEL_92;
      }
LABEL_101:
      v72 = 0;
      if ( **(_WORD **)(v31 + 16) == 6 )
        goto LABEL_93;
LABEL_102:
      v77 = v195;
      v78 = (__int64)&v196[-v195] >> 3;
      if ( !(_DWORD)v78 )
        goto LABEL_93;
      v176 = v31;
      v79 = (_QWORD *)v32;
      v80 = v72;
      v189 = 8LL * (unsigned int)(v78 - 1);
      v81 = 0;
      while ( 2 )
      {
        v82 = *(_QWORD *)(v77 + v81);
        v83 = *(_QWORD *)v82;
        v84 = (*(__int64 *)v82 >> 1) & 3;
        if ( v84 == 1 || v84 == 2 )
        {
          v180 = (_QWORD *)v79[2];
          v85 = *(__int64 (**)())(**(_QWORD **)(*v180 + 16LL) + 112LL);
          if ( v85 == sub_1D00B10 )
            BUG();
          v86 = *(_DWORD *)(v82 + 8);
          if ( *(_BYTE *)(*(_QWORD *)(v85() + 232) + 8LL * v86 + 4) )
          {
            v87 = v86 >> 6;
            if ( (*(_QWORD *)(v180[38] + 8 * v87) & (1LL << v86)) == 0
              && (!v80 || (*(_QWORD *)(*v80 + 8 * v87) & (1LL << v86)) == 0) )
            {
              v88 = v212;
              if ( !v212 )
                goto LABEL_233;
              v89 = &v211;
              do
              {
                while ( 1 )
                {
                  v90 = *(_QWORD *)(v88 + 16);
                  v91 = *(_QWORD *)(v88 + 24);
                  if ( v86 <= *(_DWORD *)(v88 + 32) )
                    break;
                  v88 = *(_QWORD *)(v88 + 24);
                  if ( !v91 )
                    goto LABEL_118;
                }
                v89 = (int *)v88;
                v88 = *(_QWORD *)(v88 + 16);
              }
              while ( v90 );
LABEL_118:
              if ( v89 == &v211 || v86 < v89[8] )
              {
LABEL_233:
                v92 = sub_1E16810(v176, v86, 0, 0, 0);
                if ( v92 != -1 )
                {
                  v93 = *(_QWORD *)(v176 + 32) + 40LL * v92;
                  if ( v93 )
                  {
                    if ( (*(_BYTE *)(v93 + 3) & 0x20) == 0 )
                    {
                      v94 = *(_QWORD *)(v166 + 32);
                      v95 = v94 + 16LL * *(unsigned int *)(v166 + 40);
                      if ( v94 != v95 )
                      {
                        v96 = v83 & 0xFFFFFFFFFFFFFFF8LL;
                        v97 = *(_QWORD *)(v166 + 32);
                        while ( 1 )
                        {
                          v98 = (*(__int64 *)v97 >> 1) & 3;
                          if ( v96 != (*(_QWORD *)v97 & 0xFFFFFFFFFFFFFFF8LL) )
                            break;
                          if ( v98 != 1 || v86 != *(_DWORD *)(v97 + 8) )
                            goto LABEL_131;
LABEL_127:
                          v97 += 16;
                          if ( v97 == v95 )
                            goto LABEL_135;
                        }
                        if ( v98 || v86 != *(_DWORD *)(v97 + 8) )
                          goto LABEL_127;
LABEL_131:
                        v86 = 0;
                        do
                        {
LABEL_135:
                          while ( 1 )
                          {
                            v99 = (*(__int64 *)v94 >> 1) & 3;
                            if ( v96 == (*(_QWORD *)v94 & 0xFFFFFFFFFFFFFFF8LL) )
                              break;
                            if ( !v99 && *(_DWORD *)(v94 + 8) == v86 )
                              goto LABEL_105;
LABEL_134:
                            v94 += 16;
                            if ( v94 == v95 )
                              goto LABEL_139;
                          }
                          if ( v99 == 2 )
                            goto LABEL_134;
                          if ( v99 != 1 )
                            goto LABEL_105;
                          v94 += 16;
                        }
                        while ( v94 != v95 );
                      }
LABEL_139:
                      if ( v86 )
                      {
                        if ( v162 )
                          memset(s, 0, n);
                        v100 = (_QWORD *)v79[4];
                        if ( !v100 )
                        {
                          LODWORD(v216) = v86;
                          v217 = 0;
                          LOBYTE(v218) = 1;
                          LOWORD(v219) = 0;
                          v220 = 0;
                          LODWORD(v221) = 0;
                          LOWORD(v222) = 0;
                          v223 = 0;
                          BUG();
                        }
                        LODWORD(v216) = v86;
                        v217 = (__int64)(v100 + 1);
                        LOWORD(v219) = 0;
                        v220 = 0;
                        LOBYTE(v218) = 1;
                        LODWORD(v221) = 0;
                        LOWORD(v222) = 0;
                        v223 = 0;
                        v101 = *(_DWORD *)(v100[1] + 24LL * v86 + 16);
                        v102 = v86 * (v101 & 0xF);
                        v103 = (_WORD *)(v100[7] + 2LL * (v101 >> 4));
                        v104 = (__int64 *)(v103 + 1);
                        LOWORD(v219) = *v103 + v102;
                        v220 = (__int64 *)(v103 + 1);
                        while ( v104 )
                        {
                          LODWORD(v221) = *(_DWORD *)(v100[6] + 4LL * (unsigned __int16)v219);
                          v105 = (unsigned __int16)v221;
                          if ( (_WORD)v221 )
                          {
                            while ( 1 )
                            {
                              v106 = *(unsigned int *)(v100[1] + 24LL * (unsigned __int16)v105 + 8);
                              v107 = v100[7];
                              LOWORD(v222) = v105;
                              v223 = (__int64 *)(v107 + 2 * v106);
                              if ( v223 )
                                break;
                              v105 = WORD1(v221);
                              LODWORD(v221) = WORD1(v221);
                              if ( !(_WORD)v105 )
                                goto LABEL_221;
                            }
                            while ( 1 )
                            {
                              *(_QWORD *)&s[(v105 >> 3) & 0x1FF8] |= 1LL << v105;
                              sub_1E1D5E0((__int64)&v216);
                              if ( !v220 )
                                break;
                              v105 = (unsigned __int16)v222;
                            }
                            break;
                          }
LABEL_221:
                          v104 = (__int64 *)((char *)v104 + 2);
                          v220 = v104;
                          v148 = *((_WORD *)v104 - 1);
                          LOWORD(v219) = v148 + (_WORD)v219;
                          if ( !v148 )
                          {
                            v220 = 0;
                            break;
                          }
                        }
                        v108 = *(_QWORD *)(v166 + 112);
                        v109 = v108 + 16LL * *(unsigned int *)(v166 + 120);
                        if ( v108 == v109 )
                        {
LABEL_162:
                          v119 = v79[9];
                          v120 = *(_DWORD *)(*(_QWORD *)(v119 + 32) + 4LL * v86);
                          do
                          {
                            v121 = v120;
                            v120 = *(_DWORD *)(*(_QWORD *)(v119 + 8) + 4LL * v120);
                          }
                          while ( v121 != v120 );
                          if ( v120 )
                          {
                            LODWORD(v217) = 0;
                            v218 = 0;
                            v219 = &v217;
                            v220 = &v217;
                            v221 = 0;
                            if ( (unsigned __int8)sub_20C5300((__int64)v79, v120, &v198, &v216) )
                            {
                              v173 = (__int64)v219;
                              v170 = (__int64 *)(v157 + 56);
                              if ( v219 != &v217 )
                              {
                                v156 = v79;
                                v155 = v80;
                                v131 = v157;
                                do
                                {
                                  v132 = *(_DWORD *)(v173 + 36);
                                  v191 = *(_DWORD *)(v173 + 32);
                                  v192 = v132;
                                  v133 = sub_20C3470((__int64)v170, &v191);
                                  v182 = v134;
                                  v135 = v133;
                                  if ( v134 != v133 )
                                  {
                                    v154 = v131;
                                    v136 = v132;
                                    do
                                    {
                                      sub_1E310D0(*(_QWORD *)(v135 + 40), v136);
                                      v137 = &v205;
                                      v138 = *(_QWORD *)(*(_QWORD *)(v135 + 40) + 16LL);
                                      v139 = v206;
                                      v193 = v138;
                                      if ( !v206 )
                                        goto LABEL_203;
                                      do
                                      {
                                        while ( 1 )
                                        {
                                          v140 = *((_QWORD *)v139 + 2);
                                          v141 = *((_QWORD *)v139 + 3);
                                          if ( *((_QWORD *)v139 + 4) >= v138 )
                                            break;
                                          v139 = (int *)*((_QWORD *)v139 + 3);
                                          if ( !v141 )
                                            goto LABEL_201;
                                        }
                                        v137 = v139;
                                        v139 = (int *)*((_QWORD *)v139 + 2);
                                      }
                                      while ( v140 );
LABEL_201:
                                      if ( v137 == &v205 || *((_QWORD *)v137 + 4) > v138 )
                                      {
LABEL_203:
                                        v194 = &v193;
                                        v137 = (int *)sub_20C4FE0(&v204, v137, &v194);
                                      }
                                      v136 = v192;
                                      if ( *((_QWORD *)v137 + 5) )
                                      {
                                        v142 = *a6;
                                        v143 = *(_QWORD *)(*(_QWORD *)(v135 + 40) + 16LL);
                                        v144 = a6[1];
                                        if ( *a6 != v144 )
                                        {
                                          v145 = 0;
                                          do
                                          {
                                            v146 = *(_QWORD *)(v144 - 8);
                                            if ( v143 == v146 || v145 == v146 )
                                            {
                                              v145 = *(_QWORD *)(v144 - 16);
                                              v147 = *(_QWORD *)(v145 + 32);
                                              if ( !*(_BYTE *)v147 && v86 == *(_DWORD *)(v147 + 8) )
                                              {
                                                v150 = v142;
                                                v151 = *(_QWORD *)(v144 - 16);
                                                v152 = v144;
                                                v153 = v143;
                                                sub_1E310D0(v147, v136);
                                                v143 = v153;
                                                v144 = v152;
                                                v145 = v151;
                                                v142 = v150;
                                              }
                                            }
                                            else if ( v145 )
                                            {
                                              break;
                                            }
                                            v144 -= 16;
                                          }
                                          while ( v142 != v144 );
                                          v136 = v192;
                                        }
                                      }
                                      v135 = sub_220EEE0(v135);
                                    }
                                    while ( v182 != v135 );
                                    v132 = v136;
                                    v131 = v154;
                                  }
                                  sub_20C2470((_QWORD *)v156[9], v132, 0);
                                  sub_20C3510(v170, &v192);
                                  *(_DWORD *)(*(_QWORD *)(v131 + 128) + 4LL * v192) = *(_DWORD *)(*(_QWORD *)(v131 + 128)
                                                                                                + 4LL * v191);
                                  *(_DWORD *)(*(_QWORD *)(v131 + 104) + 4LL * v192) = *(_DWORD *)(*(_QWORD *)(v131 + 104)
                                                                                                + 4LL * v191);
                                  sub_20C2470((_QWORD *)v156[9], v191, 0);
                                  sub_20C3510(v170, &v191);
                                  *(_DWORD *)(*(_QWORD *)(v131 + 128) + 4LL * v191) = *(_DWORD *)(*(_QWORD *)(v131 + 104)
                                                                                                + 4LL * v191);
                                  *(_DWORD *)(*(_QWORD *)(v131 + 104) + 4LL * v191) = -1;
                                  v173 = sub_220EEE0(v173);
                                }
                                while ( (__int64 *)v173 != &v217 );
                                v80 = v155;
                                v79 = v156;
                              }
                              ++v163;
                            }
                            sub_1ECD820((__int64)&v216, v218);
                          }
                        }
                        else
                        {
                          while ( 1 )
                          {
                            if ( ((*(_BYTE *)v108 ^ 6) & 6) != 0 )
                            {
                              v110 = *(unsigned int *)(v108 + 8);
                              v111 = *(_QWORD *)&s[8 * (*(_DWORD *)(v108 + 8) >> 6)];
                              if ( _bittest64(&v111, v110) )
                              {
                                if ( v86 != (_DWORD)v110 )
                                  break;
                              }
                            }
LABEL_161:
                            v108 += 16;
                            if ( v109 == v108 )
                              goto LABEL_162;
                          }
                          v112 = (__int16 *)(*(_QWORD *)(v79[4] + 56LL)
                                           + 2LL
                                           * *(unsigned int *)(*(_QWORD *)(v79[4] + 8LL) + 24LL * (unsigned int)v110 + 8));
                          v113 = *v112;
                          v114 = v112 + 1;
                          v115 = v113 + v110;
                          if ( !v113 )
                            v114 = 0;
                          v116 = v114;
LABEL_158:
                          v118 = v116;
                          while ( v118 )
                          {
                            if ( v86 == v115 )
                              goto LABEL_161;
                            v117 = *v118;
                            v116 = 0;
                            ++v118;
                            v115 += v117;
                            if ( !v117 )
                              goto LABEL_158;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
LABEL_105:
        if ( v189 == v81 )
        {
          v32 = (__int64)v79;
          v31 = v176;
          goto LABEL_93;
        }
        v77 = v195;
        v81 += 8;
        continue;
      }
    }
LABEL_43:
    _libc_free((unsigned __int64)s);
    sub_20C45E0((__int64)&v204, (__int64)v206);
    sub_20C4400((__int64)&v198, v200);
  }
  return v163;
}
