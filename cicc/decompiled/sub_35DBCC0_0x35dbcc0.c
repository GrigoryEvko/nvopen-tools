// Function: sub_35DBCC0
// Address: 0x35dbcc0
//
__int64 __fastcall sub_35DBCC0(
        __int64 a1,
        _QWORD *a2,
        unsigned int a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned int v10; // ebx
  unsigned int v11; // esi
  __int64 v12; // r8
  __int64 v13; // rcx
  int v14; // r11d
  __int64 v15; // rdi
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // r13
  __int64 v20; // r8
  unsigned int v21; // edx
  int v22; // eax
  unsigned int v23; // r13d
  __int64 v24; // r9
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // r15
  unsigned int v28; // eax
  __int64 v29; // rax
  const void **v30; // r14
  unsigned int v31; // r14d
  _QWORD *v32; // r15
  __int64 v33; // rcx
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // r13
  __int64 v37; // r15
  _QWORD *v38; // rbx
  __int64 v39; // r12
  __int64 *v40; // rsi
  __int64 v41; // r14
  unsigned int v42; // esi
  __int64 v43; // rdi
  int v44; // ebx
  _QWORD *v45; // rdx
  unsigned int v46; // ecx
  __int64 *v47; // rax
  __int64 v48; // r8
  __int64 v49; // rdi
  unsigned int *v50; // rdx
  unsigned int v51; // eax
  int v52; // ebx
  __int64 v53; // r15
  __int64 v54; // r13
  __int64 *v55; // r13
  unsigned __int64 v56; // r14
  unsigned int v57; // r13d
  unsigned __int64 *v58; // r15
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 *v61; // rbx
  __int64 v62; // r15
  __int64 v63; // r12
  __int64 *v64; // rax
  __int64 *v65; // rdx
  void (__fastcall *v66)(__int64, __int64, __int64, _QWORD); // r8
  int v67; // r11d
  __int64 *v68; // rdx
  unsigned int v69; // ecx
  _QWORD *v70; // rax
  __int64 v71; // rdi
  unsigned int *v72; // rax
  __int64 *v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 *v77; // rax
  __int64 v78; // r9
  int v79; // r11d
  __int64 v80; // r10
  unsigned int v81; // edx
  int v82; // eax
  __int64 v83; // r8
  unsigned int v84; // ecx
  int v85; // ebx
  _QWORD *v86; // r10
  unsigned __int64 v87; // rdi
  __int64 v88; // rcx
  __int64 v89; // r9
  __int64 *v90; // r9
  unsigned int v91; // edx
  __int64 v92; // rcx
  __int64 v93; // r12
  __int64 i; // rbx
  __int64 *v95; // rsi
  __int64 v96; // r9
  __int64 *v97; // r9
  unsigned __int64 v98; // rdi
  __int64 v99; // r9
  __int64 v100; // r9
  unsigned __int64 v101; // r13
  int v103; // eax
  unsigned __int64 *v104; // rax
  __int64 v105; // rdx
  int v106; // eax
  int v107; // ebx
  __int64 v108; // rdx
  __int64 v109; // rbx
  __int64 v110; // rax
  __int64 v111; // rdx
  unsigned int v112; // eax
  __int64 v113; // rdx
  unsigned int v114; // edx
  __int64 v115; // r13
  bool v116; // zf
  char v117; // al
  __int64 v118; // r13
  int v119; // ecx
  __int64 v120; // rsi
  __int64 v121; // rbx
  int v122; // ebx
  _QWORD *v123; // r10
  unsigned int v124; // ecx
  __int64 v125; // r8
  unsigned __int8 *v126; // rsi
  unsigned int v127; // eax
  __int64 v128; // rsi
  __int64 v129; // rsi
  unsigned int v130; // eax
  int v131; // eax
  unsigned int v132; // ecx
  __int64 v133; // rdi
  int v134; // r10d
  __int64 *v135; // rsi
  __int64 *v136; // rcx
  int v137; // r9d
  unsigned int v138; // r14d
  __int64 v139; // rsi
  int v140; // r11d
  __int64 v141; // r10
  unsigned int v145; // [rsp+18h] [rbp-1A8h]
  void (__fastcall *v146)(__int64, __int64, __int64, _QWORD); // [rsp+18h] [rbp-1A8h]
  void (__fastcall *v147)(__int64, __int64, __int64, _QWORD); // [rsp+18h] [rbp-1A8h]
  unsigned int v149; // [rsp+28h] [rbp-198h]
  _QWORD *v150; // [rsp+28h] [rbp-198h]
  _QWORD *v151; // [rsp+28h] [rbp-198h]
  __int64 v152; // [rsp+30h] [rbp-190h]
  _QWORD *v153; // [rsp+30h] [rbp-190h]
  _QWORD *v154; // [rsp+30h] [rbp-190h]
  __int64 v155; // [rsp+30h] [rbp-190h]
  __int64 v156; // [rsp+30h] [rbp-190h]
  __int64 *v157; // [rsp+30h] [rbp-190h]
  __int64 v159; // [rsp+38h] [rbp-188h]
  __int64 v160; // [rsp+40h] [rbp-180h]
  const void **v161; // [rsp+48h] [rbp-178h]
  unsigned int v162; // [rsp+48h] [rbp-178h]
  __int64 v163; // [rsp+48h] [rbp-178h]
  __int64 v164; // [rsp+48h] [rbp-178h]
  __int64 *v165; // [rsp+50h] [rbp-170h]
  unsigned int v166; // [rsp+5Ch] [rbp-164h]
  __int64 v167; // [rsp+60h] [rbp-160h] BYREF
  unsigned int v168; // [rsp+68h] [rbp-158h]
  unsigned __int64 v169; // [rsp+70h] [rbp-150h] BYREF
  unsigned int v170; // [rsp+78h] [rbp-148h]
  __int64 *v171; // [rsp+80h] [rbp-140h] BYREF
  __int64 *v172; // [rsp+88h] [rbp-138h]
  __int64 *v173; // [rsp+90h] [rbp-130h]
  __int64 v174; // [rsp+A0h] [rbp-120h] BYREF
  __int64 v175; // [rsp+A8h] [rbp-118h]
  __int64 v176; // [rsp+B0h] [rbp-110h]
  unsigned int v177; // [rsp+B8h] [rbp-108h]
  int v178; // [rsp+C0h] [rbp-100h] BYREF
  int v179; // [rsp+C4h] [rbp-FCh]
  __int64 v180; // [rsp+C8h] [rbp-F8h]
  __int64 v181; // [rsp+D0h] [rbp-F0h]
  unsigned __int8 *v182; // [rsp+D8h] [rbp-E8h] BYREF
  unsigned int v183; // [rsp+E0h] [rbp-E0h]
  char v184; // [rsp+E8h] [rbp-D8h]
  _QWORD *v185; // [rsp+F0h] [rbp-D0h] BYREF
  unsigned int v186; // [rsp+F8h] [rbp-C8h]
  unsigned __int64 v187; // [rsp+100h] [rbp-C0h]
  unsigned int v188; // [rsp+108h] [rbp-B8h]
  __int64 v189; // [rsp+110h] [rbp-B0h]
  __int64 v190; // [rsp+118h] [rbp-A8h]
  __int16 v191; // [rsp+120h] [rbp-A0h]
  unsigned __int64 v192; // [rsp+130h] [rbp-90h] BYREF
  __int64 *v193; // [rsp+138h] [rbp-88h]
  __int64 v194; // [rsp+140h] [rbp-80h]
  int v195; // [rsp+148h] [rbp-78h]
  char v196; // [rsp+14Ch] [rbp-74h]
  char v197; // [rsp+150h] [rbp-70h] BYREF

  v171 = 0;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v175 = 0;
  v176 = 0;
  v177 = 0;
  if ( a3 <= a4 )
  {
    v10 = a3;
    v11 = 0;
    v12 = 0;
    while ( 1 )
    {
      v19 = *a2 + 40LL * v10;
      if ( !v11 )
        break;
      v13 = *(_QWORD *)(v19 + 24);
      v14 = 1;
      v15 = 0;
      v16 = (v11 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v17 = v12 + 16LL * v16;
      v18 = *(_QWORD *)v17;
      if ( v13 == *(_QWORD *)v17 )
      {
LABEL_4:
        ++v10;
        *(_DWORD *)(v17 + 8) = 0;
        if ( a4 < v10 )
          goto LABEL_13;
        goto LABEL_5;
      }
      while ( v18 != -4096 )
      {
        if ( v18 == -8192 && !v15 )
          v15 = v17;
        v16 = (v11 - 1) & (v14 + v16);
        v17 = v12 + 16LL * v16;
        v18 = *(_QWORD *)v17;
        if ( v13 == *(_QWORD *)v17 )
          goto LABEL_4;
        ++v14;
      }
      if ( !v15 )
        v15 = v17;
      ++v174;
      v22 = v176 + 1;
      if ( 4 * ((int)v176 + 1) >= 3 * v11 )
        goto LABEL_8;
      if ( v11 - (v22 + HIDWORD(v176)) <= v11 >> 3 )
      {
        sub_35DBAE0((__int64)&v174, v11);
        if ( !v177 )
        {
LABEL_268:
          LODWORD(v176) = v176 + 1;
          BUG();
        }
        v78 = *(_QWORD *)(v19 + 24);
        v79 = 1;
        v80 = 0;
        v81 = (v177 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
        v22 = v176 + 1;
        v15 = v175 + 16LL * v81;
        v13 = *(_QWORD *)v15;
        if ( v78 != *(_QWORD *)v15 )
        {
          while ( v13 != -4096 )
          {
            if ( v13 == -8192 && !v80 )
              v80 = v15;
            v81 = (v177 - 1) & (v79 + v81);
            v15 = v175 + 16LL * v81;
            v13 = *(_QWORD *)v15;
            if ( v78 == *(_QWORD *)v15 )
              goto LABEL_10;
            ++v79;
          }
          v13 = *(_QWORD *)(v19 + 24);
          if ( v80 )
            v15 = v80;
        }
      }
LABEL_10:
      LODWORD(v176) = v22;
      if ( *(_QWORD *)v15 != -4096 )
        --HIDWORD(v176);
      ++v10;
      *(_QWORD *)v15 = v13;
      *(_DWORD *)(v15 + 8) = -1;
      *(_DWORD *)(v15 + 8) = 0;
      if ( a4 < v10 )
      {
LABEL_13:
        v23 = a3;
        v166 = 0;
        v145 = 0;
        while ( 1 )
        {
          v24 = *a2;
          v25 = 40LL * v23;
          v26 = *a2 + v25;
          v27 = *(_QWORD *)(v26 + 16);
          v28 = v166 + *(_DWORD *)(v26 + 32);
          if ( v166 + (unsigned __int64)*(unsigned int *)(v26 + 32) > 0x80000000 )
            v28 = 0x80000000;
          v161 = (const void **)(v27 + 24);
          v166 = v28;
          v29 = *(_QWORD *)(v26 + 8);
          v30 = (const void **)(v29 + 24);
          if ( *(_DWORD *)(v29 + 32) <= 0x40u )
          {
            v88 = *(_QWORD *)(v29 + 24);
            if ( v88 == *(_QWORD *)(v27 + 24) )
            {
              ++v145;
              if ( a3 == v23 )
                goto LABEL_19;
              v89 = *(_QWORD *)(v24 + 40LL * (v23 - 1) + 16);
              v186 = *(_DWORD *)(v29 + 32);
              v90 = (__int64 *)(v89 + 24);
            }
            else
            {
              v145 += 2;
              if ( a3 == v23 )
              {
LABEL_19:
                v186 = *(_DWORD *)(v27 + 32);
                if ( v186 <= 0x40 )
                  goto LABEL_20;
                goto LABEL_116;
              }
              v99 = *(_QWORD *)(v24 + 40LL * (v23 - 1) + 16);
              v186 = *(_DWORD *)(v29 + 32);
              v88 = *(_QWORD *)(v29 + 24);
              v90 = (__int64 *)(v99 + 24);
            }
            v185 = (_QWORD *)v88;
          }
          else
          {
            v149 = *(_DWORD *)(v29 + 32);
            v152 = *a2;
            if ( sub_C43C50(v29 + 24, v161) )
            {
              ++v145;
              if ( a3 == v23 )
                goto LABEL_19;
              v100 = *(_QWORD *)(v152 + 40LL * (v23 - 1) + 16);
              v186 = v149;
              v97 = (__int64 *)(v100 + 24);
            }
            else
            {
              v145 += 2;
              if ( a3 == v23 )
                goto LABEL_19;
              v96 = *(_QWORD *)(v152 + 40LL * (v23 - 1) + 16);
              v186 = v149;
              v97 = (__int64 *)(v96 + 24);
            }
            v157 = v97;
            sub_C43780((__int64)&v185, v30);
            v90 = v157;
          }
          sub_C46B40((__int64)&v185, v90);
          v91 = v186;
          v186 = 0;
          LODWORD(v193) = v91;
          v192 = (unsigned __int64)v185;
          if ( v91 <= 0x40 )
          {
            v92 = (__int64)v185 - 1;
            goto LABEL_108;
          }
          v151 = v185;
          if ( v91 - (unsigned int)sub_C444A0((__int64)&v192) > 0x40 )
            break;
          v155 = *v151 - 1LL;
          j_j___libc_free_0_0((unsigned __int64)v151);
          v92 = v155;
          if ( v186 > 0x40 )
          {
            v98 = (unsigned __int64)v185;
            if ( v185 )
              goto LABEL_124;
          }
LABEL_108:
          if ( !v92 )
            goto LABEL_19;
LABEL_109:
          v154 = a2;
          v93 = v92;
          for ( i = 0; i != v93; ++i )
          {
            while ( 1 )
            {
              v95 = v172;
              if ( v172 != v173 )
                break;
              ++i;
              sub_2E33A40((__int64)&v171, v172, &a7);
              if ( i == v93 )
                goto LABEL_115;
            }
            if ( v172 )
            {
              *v172 = a7;
              v95 = v172;
            }
            v172 = v95 + 1;
          }
LABEL_115:
          a2 = v154;
          v25 = 40LL * v23;
          v186 = *(_DWORD *)(v27 + 32);
          if ( v186 <= 0x40 )
          {
LABEL_20:
            v185 = *(_QWORD **)(v27 + 24);
            goto LABEL_21;
          }
LABEL_116:
          sub_C43780((__int64)&v185, v161);
LABEL_21:
          sub_C46B40((__int64)&v185, (__int64 *)v30);
          v31 = v186;
          v32 = v185;
          v186 = 0;
          LODWORD(v193) = v31;
          v192 = (unsigned __int64)v185;
          if ( v31 <= 0x40 )
          {
            v33 = (__int64)v185 + 1;
            goto LABEL_23;
          }
          if ( v31 - (unsigned int)sub_C444A0((__int64)&v192) > 0x40 )
          {
            if ( !v32
              || (j_j___libc_free_0_0((unsigned __int64)v32), v186 <= 0x40)
              || (v87 = (unsigned __int64)v185, v33 = 0, !v185) )
            {
              v34 = *a2;
LABEL_90:
              v42 = v177;
              v41 = v25 + v34;
              if ( !v177 )
                goto LABEL_91;
              goto LABEL_31;
            }
LABEL_101:
            v164 = v33;
            j_j___libc_free_0_0(v87);
            v33 = v164;
            goto LABEL_23;
          }
          v163 = *v32 + 1LL;
          j_j___libc_free_0_0((unsigned __int64)v32);
          v33 = v163;
          if ( v186 > 0x40 )
          {
            v87 = (unsigned __int64)v185;
            if ( v185 )
              goto LABEL_101;
          }
LABEL_23:
          v34 = *a2;
          if ( !v33 )
            goto LABEL_90;
          v162 = v23;
          v35 = v34 + v25;
          v36 = v25;
          v37 = 0;
          v38 = a2;
          v39 = v33;
          do
          {
            while ( 1 )
            {
              v40 = v172;
              v41 = v35;
              if ( v172 != v173 )
                break;
              ++v37;
              sub_2E33A40((__int64)&v171, v172, (_QWORD *)(v35 + 24));
              v35 = v36 + *v38;
              v41 = v35;
              if ( v37 == v39 )
                goto LABEL_30;
            }
            if ( v172 )
            {
              *v172 = *(_QWORD *)(v35 + 24);
              v40 = v172;
              v35 = v36 + *v38;
              v41 = v35;
            }
            ++v37;
            v172 = v40 + 1;
          }
          while ( v37 != v39 );
LABEL_30:
          v42 = v177;
          v23 = v162;
          a2 = v38;
          if ( !v177 )
          {
LABEL_91:
            ++v174;
            goto LABEL_92;
          }
LABEL_31:
          v43 = *(_QWORD *)(v41 + 24);
          v44 = 1;
          v45 = 0;
          v46 = (v42 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v47 = (__int64 *)(v175 + 16LL * v46);
          v48 = *v47;
          if ( v43 != *v47 )
          {
            while ( v48 != -4096 )
            {
              if ( v48 == -8192 && !v45 )
                v45 = v47;
              v46 = (v42 - 1) & (v44 + v46);
              v47 = (__int64 *)(v175 + 16LL * v46);
              v48 = *v47;
              if ( v43 == *v47 )
                goto LABEL_32;
              ++v44;
            }
            if ( !v45 )
              v45 = v47;
            ++v174;
            v82 = v176 + 1;
            if ( 4 * ((int)v176 + 1) >= 3 * v42 )
            {
LABEL_92:
              sub_35DBAE0((__int64)&v174, 2 * v42);
              if ( !v177 )
                goto LABEL_269;
              v83 = *(_QWORD *)(v41 + 24);
              v84 = (v177 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
              v82 = v176 + 1;
              v45 = (_QWORD *)(v175 + 16LL * v84);
              v43 = *v45;
              if ( v83 != *v45 )
              {
                v85 = 1;
                v86 = 0;
                while ( v43 != -4096 )
                {
                  if ( !v86 && v43 == -8192 )
                    v86 = v45;
                  v84 = (v177 - 1) & (v85 + v84);
                  v45 = (_QWORD *)(v175 + 16LL * v84);
                  v43 = *v45;
                  if ( v83 == *v45 )
                    goto LABEL_84;
                  ++v85;
                }
                v43 = *(_QWORD *)(v41 + 24);
                if ( v86 )
                  v45 = v86;
              }
            }
            else if ( v42 - HIDWORD(v176) - v82 <= v42 >> 3 )
            {
              sub_35DBAE0((__int64)&v174, v42);
              if ( !v177 )
              {
LABEL_269:
                LODWORD(v176) = v176 + 1;
                BUG();
              }
              v43 = *(_QWORD *)(v41 + 24);
              v122 = 1;
              v123 = 0;
              v124 = (v177 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
              v82 = v176 + 1;
              v45 = (_QWORD *)(v175 + 16LL * v124);
              v125 = *v45;
              if ( *v45 != v43 )
              {
                while ( v125 != -4096 )
                {
                  if ( !v123 && v125 == -8192 )
                    v123 = v45;
                  v124 = (v177 - 1) & (v122 + v124);
                  v45 = (_QWORD *)(v175 + 16LL * v124);
                  v125 = *v45;
                  if ( v43 == *v45 )
                    goto LABEL_84;
                  ++v122;
                }
                if ( v123 )
                  v45 = v123;
              }
            }
LABEL_84:
            LODWORD(v176) = v82;
            if ( *v45 != -4096 )
              --HIDWORD(v176);
            *v45 = v43;
            v51 = 0x80000000;
            v50 = (unsigned int *)(v45 + 1);
            *v50 = -1;
            goto LABEL_34;
          }
LABEL_32:
          v49 = *((unsigned int *)v47 + 2);
          v50 = (unsigned int *)(v47 + 1);
          v51 = v49 + *(_DWORD *)(v41 + 32);
          if ( v49 + (unsigned __int64)*(unsigned int *)(v41 + 32) > 0x80000000 )
            v51 = 0x80000000;
LABEL_34:
          *v50 = v51;
          if ( a4 < ++v23 )
          {
            v52 = v176;
            goto LABEL_36;
          }
        }
        if ( !v151 || (j_j___libc_free_0_0((unsigned __int64)v151), v186 <= 0x40) )
        {
          v92 = -2;
          goto LABEL_109;
        }
        v98 = (unsigned __int64)v185;
        v92 = -2;
        if ( !v185 )
          goto LABEL_109;
LABEL_124:
        v156 = v92;
        j_j___libc_free_0_0(v98);
        v92 = v156;
        goto LABEL_108;
      }
LABEL_5:
      v12 = v175;
      v11 = v177;
    }
    ++v174;
LABEL_8:
    sub_35DBAE0((__int64)&v174, 2 * v11);
    if ( !v177 )
      goto LABEL_268;
    v20 = *(_QWORD *)(v19 + 24);
    v21 = (v177 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v22 = v176 + 1;
    v15 = v175 + 16LL * v21;
    v13 = *(_QWORD *)v15;
    if ( v20 != *(_QWORD *)v15 )
    {
      v140 = 1;
      v141 = 0;
      while ( v13 != -4096 )
      {
        if ( !v141 && v13 == -8192 )
          v141 = v15;
        v21 = (v177 - 1) & (v140 + v21);
        v15 = v175 + 16LL * v21;
        v13 = *(_QWORD *)v15;
        if ( v20 == *(_QWORD *)v15 )
          goto LABEL_10;
        ++v140;
      }
      v13 = *(_QWORD *)(v19 + 24);
      if ( v141 )
        v15 = v141;
    }
    goto LABEL_10;
  }
  v166 = 0;
  v52 = 0;
  v145 = 0;
LABEL_36:
  v53 = *(_QWORD *)(*a2 + 40LL * a4 + 16);
  v160 = 40LL * a4;
  v54 = *(_QWORD *)(*a2 + 40LL * a3 + 8);
  v159 = 40LL * a3;
  v55 = (__int64 *)(v54 + 24);
  v56 = (unsigned int)sub_AE2980(*(_QWORD *)(a1 + 96), 0)[3];
  LODWORD(v193) = *(_DWORD *)(v53 + 32);
  if ( (unsigned int)v193 > 0x40 )
    sub_C43780((__int64)&v192, (const void **)(v53 + 24));
  else
    v192 = *(_QWORD *)(v53 + 24);
  sub_C46B40((__int64)&v192, v55);
  v57 = (unsigned int)v193;
  v58 = (unsigned __int64 *)v192;
  LODWORD(v193) = 0;
  v186 = v57;
  v185 = (_QWORD *)v192;
  if ( v57 > 0x40 )
  {
    if ( v57 - (unsigned int)sub_C444A0((__int64)&v185) > 0x40 )
    {
      v101 = -1;
    }
    else
    {
      v101 = *v58;
      if ( *v58 == -1 )
      {
        j_j___libc_free_0_0((unsigned __int64)v58);
        if ( (unsigned int)v193 <= 0x40 )
          goto LABEL_40;
        goto LABEL_134;
      }
      ++v101;
    }
    if ( !v58 )
      goto LABEL_136;
    j_j___libc_free_0_0((unsigned __int64)v58);
    if ( (unsigned int)v193 <= 0x40 )
      goto LABEL_136;
LABEL_134:
    if ( v192 )
      j_j___libc_free_0_0(v192);
    goto LABEL_136;
  }
  if ( v192 == -1 )
    goto LABEL_40;
  v101 = v192 + 1;
LABEL_136:
  if ( v56 < v101 )
    goto LABEL_40;
  LOBYTE(v101) = v145 > 2 && v52 == 1;
  if ( (_BYTE)v101 )
  {
    LODWORD(v101) = 0;
    goto LABEL_139;
  }
  if ( (v145 <= 4 || v52 != 2) && (v52 != 3 || v145 <= 5) )
  {
LABEL_40:
    v59 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL);
    LOBYTE(v193) = 0;
    v153 = (_QWORD *)v59;
    v60 = sub_2E7AAE0(v59, *(_QWORD *)(a5 + 40), v192, 0);
    v61 = v171;
    v195 = 0;
    v62 = v60;
    v194 = 8;
    v193 = (__int64 *)&v197;
    v192 = 0;
    v196 = 1;
    v165 = v172;
    if ( v172 == v171 )
      goto LABEL_145;
    v150 = a2;
    v63 = *v171;
LABEL_42:
    v64 = v193;
    v65 = &v193[HIDWORD(v194)];
    if ( v193 != v65 )
    {
      do
      {
        if ( v63 == *v64 )
          goto LABEL_46;
        ++v64;
      }
      while ( v65 != v64 );
    }
LABEL_49:
    v66 = **(void (__fastcall ***)(__int64, __int64, __int64, _QWORD))a1;
    if ( v177 )
    {
      v67 = 1;
      v68 = 0;
      v69 = (v177 - 1) & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
      v70 = (_QWORD *)(v175 + 16LL * v69);
      v71 = *v70;
      if ( v63 == *v70 )
      {
LABEL_51:
        v72 = (unsigned int *)(v70 + 1);
        goto LABEL_52;
      }
      while ( v71 != -4096 )
      {
        if ( v71 == -8192 && !v68 )
          v68 = v70;
        v69 = (v177 - 1) & (v67 + v69);
        v70 = (_QWORD *)(v175 + 16LL * v69);
        v71 = *v70;
        if ( v63 == *v70 )
          goto LABEL_51;
        ++v67;
      }
      if ( !v68 )
        v68 = v70;
      ++v174;
      v131 = v176 + 1;
      if ( 4 * ((int)v176 + 1) < 3 * v177 )
      {
        if ( v177 - HIDWORD(v176) - v131 > v177 >> 3 )
        {
LABEL_204:
          LODWORD(v176) = v131;
          if ( *v68 != -4096 )
            --HIDWORD(v176);
          *v68 = v63;
          v72 = (unsigned int *)(v68 + 1);
          *((_DWORD *)v68 + 2) = -1;
LABEL_52:
          v66(a1, v62, v63, *v72);
          if ( v196 )
          {
            v77 = v193;
            v74 = HIDWORD(v194);
            v73 = &v193[HIDWORD(v194)];
            if ( v193 != v73 )
            {
              while ( v63 != *v77 )
              {
                if ( v73 == ++v77 )
                  goto LABEL_166;
              }
LABEL_46:
              while ( 1 )
              {
                if ( v165 == ++v61 )
                  goto LABEL_144;
LABEL_47:
                v63 = *v61;
                if ( v196 )
                  goto LABEL_42;
                if ( !sub_C8CA60((__int64)&v192, v63) )
                  goto LABEL_49;
              }
            }
LABEL_166:
            if ( HIDWORD(v194) < (unsigned int)v194 )
            {
              ++HIDWORD(v194);
              *v73 = v63;
              ++v192;
              goto LABEL_46;
            }
          }
          ++v61;
          sub_C8CC70((__int64)&v192, v63, (__int64)v73, v74, v75, v76);
          if ( v165 != v61 )
            goto LABEL_47;
LABEL_144:
          a2 = v150;
LABEL_145:
          sub_2E33470(*(unsigned int **)(v62 + 144), *(unsigned int **)(v62 + 152));
          v103 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 80) + 1912LL))(*(_QWORD *)(a1 + 80));
          v104 = (unsigned __int64 *)sub_2E7BE50(v153, v103);
          v106 = sub_2E7C5A0(v104, (const void **)&v171, v105);
          LOBYTE(v187) = 0;
          v107 = v106;
          if ( !*(_BYTE *)(a6 + 16) )
          {
            v178 = 0;
            v179 = v106;
            v180 = v62;
            v181 = 0;
            v184 = 0;
LABEL_147:
            v108 = *a2;
            v109 = **(_QWORD **)(a5 - 8);
            v110 = *(_QWORD *)(*a2 + v160 + 16);
            v170 = *(_DWORD *)(v110 + 32);
            if ( v170 > 0x40 )
            {
              sub_C43780((__int64)&v169, (const void **)(v110 + 24));
              v108 = *a2;
            }
            else
            {
              v169 = *(_QWORD *)(v110 + 24);
            }
            v111 = *(_QWORD *)(v108 + v159 + 8);
            v112 = *(_DWORD *)(v111 + 32);
            v168 = v112;
            if ( v112 > 0x40 )
            {
              sub_C43780((__int64)&v167, (const void **)(v111 + 24));
              v112 = v168;
              v113 = v167;
            }
            else
            {
              v113 = *(_QWORD *)(v111 + 24);
            }
            v185 = (_QWORD *)v113;
            v189 = v109;
            v114 = v170;
            v187 = v169;
            v186 = v112;
            v188 = v170;
            v115 = *(_QWORD *)(a1 + 40);
            v190 = 0;
            v191 = 0;
            if ( v115 == *(_QWORD *)(a1 + 48) )
            {
              sub_35D9270((unsigned __int64 *)(a1 + 32), v115, (__int64 *)&v185, (__int64)&v178);
              v114 = v188;
              v118 = *(_QWORD *)(a1 + 40);
            }
            else
            {
              if ( v115 )
              {
                *(_DWORD *)(v115 + 8) = v112;
                *(_QWORD *)v115 = v185;
                v186 = 0;
                *(_DWORD *)(v115 + 24) = v188;
                *(_QWORD *)(v115 + 16) = v187;
                v188 = 0;
                *(_QWORD *)(v115 + 32) = v189;
                *(_QWORD *)(v115 + 40) = v190;
                v116 = v184 == 0;
                *(_BYTE *)(v115 + 48) = v191;
                v117 = HIBYTE(v191);
                *(_BYTE *)(v115 + 96) = 0;
                *(_BYTE *)(v115 + 49) = v117;
                *(_DWORD *)(v115 + 56) = v178;
                *(_DWORD *)(v115 + 60) = v179;
                *(_QWORD *)(v115 + 64) = v180;
                *(_QWORD *)(v115 + 72) = v181;
                if ( v116 )
                {
                  v115 = *(_QWORD *)(a1 + 40);
                  v114 = 0;
                }
                else
                {
                  v126 = v182;
                  v114 = 0;
                  *(_QWORD *)(v115 + 80) = v182;
                  if ( v126 )
                  {
                    sub_B976B0((__int64)&v182, v126, v115 + 80);
                    v114 = v188;
                    v182 = 0;
                  }
                  v127 = v183;
                  *(_BYTE *)(v115 + 96) = 1;
                  *(_DWORD *)(v115 + 88) = v127;
                  v115 = *(_QWORD *)(a1 + 40);
                }
              }
              v118 = v115 + 104;
              *(_QWORD *)(a1 + 40) = v118;
            }
            v119 = -991146299 * ((v118 - *(_QWORD *)(a1 + 32)) >> 3) - 1;
            v120 = *(_QWORD *)(*a2 + v160 + 16);
            v121 = a8;
            *(_QWORD *)(a8 + 8) = *(_QWORD *)(*a2 + v159 + 8);
            *(_DWORD *)v121 = 1;
            *(_QWORD *)(v121 + 16) = v120;
            *(_DWORD *)(v121 + 24) = v119;
            *(_DWORD *)(v121 + 32) = v166;
            if ( v114 > 0x40 && v187 )
              j_j___libc_free_0_0(v187);
            if ( v186 > 0x40 && v185 )
              j_j___libc_free_0_0((unsigned __int64)v185);
            if ( v184 )
            {
              v184 = 0;
              if ( v182 )
                sub_B91220((__int64)&v182, (__int64)v182);
            }
            if ( !v196 )
              _libc_free((unsigned __int64)v193);
            LODWORD(v101) = 1;
            goto LABEL_139;
          }
          v128 = *(_QWORD *)a6;
          v185 = (_QWORD *)v128;
          if ( v128 )
            sub_B96E90((__int64)&v185, v128, 1);
          v179 = v107;
          v129 = (__int64)v185;
          LOBYTE(v187) = 1;
          v130 = *(_DWORD *)(a6 + 8);
          v178 = 0;
          v180 = v62;
          v186 = v130;
          v181 = 0;
          v184 = 0;
          v182 = (unsigned __int8 *)v185;
          if ( v185 )
          {
            sub_B96E90((__int64)&v182, (__int64)v185, 1);
            v184 = 1;
            v183 = v186;
            if ( !(_BYTE)v187 )
              goto LABEL_147;
            v129 = (__int64)v185;
          }
          else
          {
            v183 = v130;
            v184 = 1;
          }
          LOBYTE(v187) = 0;
          if ( v129 )
            sub_B91220((__int64)&v185, v129);
          goto LABEL_147;
        }
        v147 = v66;
        sub_35DBAE0((__int64)&v174, v177);
        if ( v177 )
        {
          v136 = 0;
          v137 = 1;
          v138 = (v177 - 1) & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
          v66 = v147;
          v131 = v176 + 1;
          v68 = (__int64 *)(v175 + 16LL * v138);
          v139 = *v68;
          if ( v63 != *v68 )
          {
            while ( v139 != -4096 )
            {
              if ( v139 == -8192 && !v136 )
                v136 = v68;
              v138 = (v177 - 1) & (v137 + v138);
              v68 = (__int64 *)(v175 + 16LL * v138);
              v139 = *v68;
              if ( v63 == *v68 )
                goto LABEL_204;
              ++v137;
            }
            if ( v136 )
              v68 = v136;
          }
          goto LABEL_204;
        }
LABEL_267:
        LODWORD(v176) = v176 + 1;
        BUG();
      }
    }
    else
    {
      ++v174;
    }
    v146 = v66;
    sub_35DBAE0((__int64)&v174, 2 * v177);
    if ( v177 )
    {
      v66 = v146;
      v132 = (v177 - 1) & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
      v131 = v176 + 1;
      v68 = (__int64 *)(v175 + 16LL * v132);
      v133 = *v68;
      if ( v63 != *v68 )
      {
        v134 = 1;
        v135 = 0;
        while ( v133 != -4096 )
        {
          if ( !v135 && v133 == -8192 )
            v135 = v68;
          v132 = (v177 - 1) & (v134 + v132);
          v68 = (__int64 *)(v175 + 16LL * v132);
          v133 = *v68;
          if ( v63 == *v68 )
            goto LABEL_204;
          ++v134;
        }
        if ( v135 )
          v68 = v135;
      }
      goto LABEL_204;
    }
    goto LABEL_267;
  }
LABEL_139:
  sub_C7D6A0(v175, 16LL * v177, 8);
  if ( v171 )
    j_j___libc_free_0((unsigned __int64)v171);
  return (unsigned int)v101;
}
