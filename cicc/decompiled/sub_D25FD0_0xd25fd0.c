// Function: sub_D25FD0
// Address: 0xd25fd0
//
__int64 __fastcall sub_D25FD0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        void (__fastcall *a4)(__int64, __int64 *, signed __int64),
        __int64 a5)
{
  __int64 *v5; // r15
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 *v10; // r13
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r12
  __int64 v18; // r13
  __int64 *v19; // rax
  __int64 *v20; // r14
  __int64 *v21; // rbx
  unsigned __int64 v22; // rdx
  int v23; // edi
  __int64 v24; // r8
  int v25; // edi
  unsigned int v26; // esi
  __int64 *v27; // rax
  __int64 v28; // r10
  __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 *v33; // r13
  __int64 *v34; // r12
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 *v38; // r14
  __int64 v39; // rsi
  _QWORD *v40; // rax
  _QWORD *v41; // rcx
  __int64 v42; // rsi
  __int64 *v43; // r8
  _QWORD *v44; // rcx
  __int64 v45; // rbx
  __int64 v46; // r15
  __int64 v47; // r14
  __int64 *v48; // r8
  __int64 v49; // r9
  __int64 v50; // rcx
  __int64 v51; // rsi
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // r14
  __int64 v57; // r11
  __int64 *v58; // rsi
  int v59; // ecx
  unsigned int v60; // edx
  __int64 *v61; // rax
  unsigned int v62; // eax
  __int64 v63; // rbx
  __int64 v64; // rax
  const void *v65; // r13
  __int64 *v66; // r12
  __int64 *v67; // r11
  unsigned int v68; // esi
  __int64 *v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rbx
  __int64 v72; // rdx
  int v73; // esi
  __int64 *v74; // rcx
  int v75; // esi
  __int64 v76; // rax
  unsigned __int64 v77; // rcx
  __int64 v78; // rdx
  __int64 v79; // r12
  char *v80; // rax
  __int64 *v81; // r12
  __int64 v82; // rax
  __int64 *v83; // r13
  __int64 v84; // rax
  _DWORD *v85; // rax
  __int64 v86; // rsi
  unsigned int v87; // r12d
  __int64 *v89; // rsi
  int v90; // r9d
  __int64 *v91; // rax
  __int64 v92; // r14
  __int64 v93; // r8
  __int64 v94; // rax
  __int64 v95; // rsi
  _QWORD *v96; // rcx
  __int64 v97; // rsi
  __int64 *v98; // rcx
  __int64 *v99; // rax
  __int64 *v100; // rax
  __int64 *v101; // rax
  __int64 *v102; // rax
  __int64 *v103; // rax
  int v104; // eax
  int v105; // r9d
  int v106; // ecx
  int v107; // eax
  __int64 v108; // rax
  int v109; // eax
  int v110; // eax
  int v111; // edi
  unsigned int v112; // eax
  __int64 v113; // rsi
  _QWORD *v114; // rdi
  int v115; // r8d
  unsigned int v116; // eax
  __int64 v117; // rdx
  __int64 v118; // rdx
  __int64 v119; // r14
  __int64 v120; // rdx
  __int64 v121; // r12
  __int64 v122; // rax
  __int64 v123; // rbx
  int v124; // r13d
  unsigned __int64 v125; // rax
  int v126; // esi
  __int64 v127; // rdi
  __int64 v128; // rcx
  __int64 *v129; // rdx
  __int64 v130; // r10
  _QWORD *v131; // r8
  __int64 v132; // r13
  __int64 v133; // rbx
  __int64 *v134; // r12
  __int64 *v135; // r13
  __int64 v136; // rax
  __int64 v137; // rcx
  __int64 *v138; // r14
  __int64 v139; // rsi
  char *v140; // rax
  char *v141; // rcx
  _QWORD *v142; // rdx
  __int64 v143; // rsi
  __int64 *v144; // r8
  _QWORD *v145; // rdx
  __int64 v146; // rsi
  __int64 *v147; // r8
  _QWORD *v148; // rdx
  int v149; // edi
  _QWORD *v150; // rax
  _QWORD *v151; // r11
  _QWORD *v152; // rax
  __int64 v153; // rax
  unsigned __int64 v154; // rdx
  __int64 v155; // rax
  int v156; // edx
  int v157; // r8d
  char v158; // dl
  __int64 v159; // rax
  int v160; // eax
  __int64 v161; // r15
  __int64 v162; // rbx
  __int64 v163; // r14
  __int64 v164; // rax
  __int64 *v165; // r8
  __int64 v166; // r9
  int v167; // r12d
  __int64 *v168; // rcx
  unsigned int v169; // edx
  __int64 *v170; // rax
  __int64 v171; // r14
  __int64 v172; // rax
  __int64 *v173; // rax
  __int64 *v174; // rax
  int v175; // r10d
  __int64 v176; // rax
  int v177; // eax
  char v178; // al
  int v179; // r10d
  int v180; // r10d
  char v181; // al
  __int64 v184; // [rsp+20h] [rbp-130h]
  __int64 v185; // [rsp+20h] [rbp-130h]
  int v187; // [rsp+38h] [rbp-118h]
  __int64 v188; // [rsp+40h] [rbp-110h]
  int v190; // [rsp+50h] [rbp-100h]
  int v191; // [rsp+50h] [rbp-100h]
  __int64 v192; // [rsp+50h] [rbp-100h]
  __int64 v193; // [rsp+58h] [rbp-F8h]
  __int64 *v194; // [rsp+58h] [rbp-F8h]
  int v195; // [rsp+58h] [rbp-F8h]
  __int64 *v196; // [rsp+58h] [rbp-F8h]
  int v197; // [rsp+58h] [rbp-F8h]
  int v198; // [rsp+60h] [rbp-F0h]
  __int64 v199; // [rsp+60h] [rbp-F0h]
  __int64 v200; // [rsp+68h] [rbp-E8h]
  __int64 v201; // [rsp+68h] [rbp-E8h]
  __int64 v202; // [rsp+68h] [rbp-E8h]
  __int64 v203; // [rsp+68h] [rbp-E8h]
  __int64 v204; // [rsp+68h] [rbp-E8h]
  char *dest; // [rsp+70h] [rbp-E0h]
  void *desta; // [rsp+70h] [rbp-E0h]
  __int64 *destb; // [rsp+70h] [rbp-E0h]
  void *src; // [rsp+80h] [rbp-D0h]
  __int64 *srcc; // [rsp+80h] [rbp-D0h]
  __int64 *srca; // [rsp+80h] [rbp-D0h]
  void *srce; // [rsp+80h] [rbp-D0h]
  __int64 *srcb; // [rsp+80h] [rbp-D0h]
  void *srcd; // [rsp+80h] [rbp-D0h]
  __int64 v214; // [rsp+88h] [rbp-C8h]
  __int64 *v215; // [rsp+88h] [rbp-C8h]
  __int64 v216; // [rsp+88h] [rbp-C8h]
  __int64 *v217; // [rsp+88h] [rbp-C8h]
  _QWORD *v218; // [rsp+88h] [rbp-C8h]
  __int64 *v219; // [rsp+88h] [rbp-C8h]
  _QWORD *v220; // [rsp+88h] [rbp-C8h]
  _BYTE *v221; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v222; // [rsp+98h] [rbp-B8h]
  _BYTE v223[16]; // [rsp+A0h] [rbp-B0h] BYREF
  _QWORD *v224; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v225; // [rsp+B8h] [rbp-98h]
  _QWORD v226[4]; // [rsp+C0h] [rbp-90h] BYREF
  __int64 *v227; // [rsp+E0h] [rbp-70h] BYREF
  void *s; // [rsp+E8h] [rbp-68h]
  _BYTE v229[12]; // [rsp+F0h] [rbp-60h]
  char v230; // [rsp+FCh] [rbp-54h]
  char v231; // [rsp+100h] [rbp-50h] BYREF

  v5 = a1;
  v7 = *a1;
  v221 = v223;
  v222 = 0x100000000LL;
  v8 = sub_D23C40(v7, a2);
  v200 = sub_D23C40(v7, a3);
  if ( v8 == v200
    || (v227 = (__int64 *)v8,
        v188 = (__int64)(a1 + 7),
        v187 = *(_DWORD *)sub_D25BD0((__int64)(a1 + 7), (__int64 *)&v227),
        v227 = (__int64 *)v200,
        v9 = *(int *)sub_D25BD0((__int64)(a1 + 7), (__int64 *)&v227),
        v187 > (int)v9) )
  {
    v199 = a2 + 24;
LABEL_143:
    v86 = a3;
    v87 = 0;
    sub_D23D60(v199, a3, 1u);
    goto LABEL_83;
  }
  v10 = (__int64 *)&v227;
  v227 = (__int64 *)v8;
  v190 = *(_DWORD *)sub_D25BD0(v188, (__int64 *)&v227);
  v227 = (__int64 *)v200;
  v11 = *(_DWORD *)sub_D25BD0(v188, (__int64 *)&v227);
  v227 = 0;
  *(_QWORD *)v229 = 4;
  v198 = v11;
  s = &v231;
  *(_DWORD *)&v229[8] = 0;
  v230 = 1;
  sub_AE6EC0((__int64)&v227, v8);
  v13 = a1[1];
  v193 = v13 + 8 * v9 + 8;
  if ( v193 == v13 + 8LL * v187 + 8 )
    goto LABEL_26;
  dest = (char *)(v13 + 8LL * v187 + 8);
  do
  {
    v14 = *(_QWORD *)(*(_QWORD *)dest + 8LL);
    v15 = *(unsigned int *)(*(_QWORD *)dest + 16LL);
    if ( v14 == v14 + 8 * v15 )
      goto LABEL_24;
    src = *(void **)dest;
    v16 = (__int64)v10;
    v17 = v14 + 8 * v15;
    v18 = *(_QWORD *)(*(_QWORD *)dest + 8LL);
    do
    {
LABEL_7:
      v19 = *(__int64 **)(*(_QWORD *)v18 + 24LL);
      v12 = *(unsigned int *)(*(_QWORD *)v18 + 32LL);
      v20 = &v19[v12];
      if ( v19 != v20 )
      {
        while ( 1 )
        {
          v12 = *v19;
          v21 = v19;
          if ( (*v19 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*v19 & 0xFFFFFFFFFFFFFFF8LL) && (v12 & 4) != 0 )
            break;
          if ( v20 == ++v19 )
            goto LABEL_12;
        }
        while ( v20 != v21 )
        {
          v22 = v12 & 0xFFFFFFFFFFFFFFF8LL;
          v23 = *(_DWORD *)(*v5 + 328);
          v24 = *(_QWORD *)(*v5 + 312);
          if ( v23 )
          {
            v25 = v23 - 1;
            v26 = v25 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
            v27 = (__int64 *)(v24 + 16LL * v26);
            v28 = *v27;
            if ( v22 == *v27 )
            {
LABEL_17:
              v29 = v27[1];
              goto LABEL_18;
            }
            v104 = 1;
            while ( v28 != -4096 )
            {
              v105 = v104 + 1;
              v26 = v25 & (v104 + v26);
              v27 = (__int64 *)(v24 + 16LL * v26);
              v28 = *v27;
              if ( v22 == *v27 )
                goto LABEL_17;
              v104 = v105;
            }
          }
          v29 = 0;
LABEL_18:
          if ( v230 )
          {
            v30 = s;
            v12 = (__int64)s + 8 * *(unsigned int *)&v229[4];
            if ( s != (void *)v12 )
            {
              while ( *v30 != v29 )
              {
                if ( (_QWORD *)v12 == ++v30 )
                  goto LABEL_121;
              }
LABEL_23:
              v10 = (__int64 *)v16;
              sub_AE6EC0(v16, (__int64)src);
              goto LABEL_24;
            }
          }
          else
          {
            v216 = v16;
            v102 = sub_C8CA60(v16, v29);
            v16 = v216;
            if ( v102 )
              goto LABEL_23;
          }
LABEL_121:
          v103 = v21 + 1;
          if ( v21 + 1 == v20 )
            break;
          while ( 1 )
          {
            v12 = *v103;
            v21 = v103;
            if ( (*v103 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*v103 & 0xFFFFFFFFFFFFFFF8LL) && (v12 & 4) != 0 )
              break;
            if ( v20 == ++v103 )
            {
              v18 += 8;
              if ( v17 != v18 )
                goto LABEL_7;
              goto LABEL_13;
            }
          }
        }
      }
LABEL_12:
      v18 += 8;
    }
    while ( v17 != v18 );
LABEL_13:
    v10 = (__int64 *)v16;
LABEL_24:
    dest += 8;
  }
  while ( (char *)v193 != dest );
  v13 = v5[1];
LABEL_26:
  v214 = v198;
  v31 = 8LL * v190;
  v32 = 8LL * v198 + 8;
  v33 = (__int64 *)(v13 + v32);
  v184 = v32;
  v34 = (__int64 *)(v31 + v13);
  v35 = v32 - v31;
  v36 = (v32 - v31) >> 5;
  v37 = v35 >> 3;
  if ( v36 <= 0 )
    goto LABEL_110;
  v38 = &v34[4 * v36];
  while ( 2 )
  {
    v39 = *v34;
    if ( v230 )
    {
      v40 = s;
      v12 = (__int64)s + 8 * *(unsigned int *)&v229[4];
      if ( s != (void *)v12 )
      {
        v41 = s;
        while ( v39 != *v41 )
        {
          if ( (_QWORD *)v12 == ++v41 )
            goto LABEL_34;
        }
        goto LABEL_40;
      }
LABEL_34:
      v42 = v34[1];
      v43 = v34 + 1;
LABEL_35:
      if ( (_QWORD *)v12 != v40 )
      {
        v44 = v40;
        while ( *v44 != v42 )
        {
          if ( ++v44 == (_QWORD *)v12 )
            goto LABEL_92;
        }
        goto LABEL_39;
      }
LABEL_92:
      v95 = v34[2];
      v43 = v34 + 2;
LABEL_93:
      if ( v40 != (_QWORD *)v12 )
      {
        v96 = v40;
        while ( *v96 != v95 )
        {
          if ( ++v96 == (_QWORD *)v12 )
            goto LABEL_101;
        }
LABEL_39:
        v34 = v43;
        goto LABEL_40;
      }
LABEL_101:
      v97 = v34[3];
      v98 = v34 + 3;
LABEL_102:
      if ( (_QWORD *)v12 != v40 )
      {
        while ( *v40 != v97 )
        {
          if ( ++v40 == (_QWORD *)v12 )
            goto LABEL_108;
        }
LABEL_106:
        v34 = v98;
        goto LABEL_40;
      }
    }
    else
    {
      if ( sub_C8CA60((__int64)&v227, v39) )
        goto LABEL_40;
      v42 = v34[1];
      v43 = v34 + 1;
      if ( v230 )
      {
        v40 = s;
        v12 = (__int64)s + 8 * *(unsigned int *)&v229[4];
        goto LABEL_35;
      }
      v100 = sub_C8CA60((__int64)&v227, v42);
      v43 = v34 + 1;
      if ( v100 )
        goto LABEL_39;
      v95 = v34[2];
      v43 = v34 + 2;
      if ( v230 )
      {
        v40 = s;
        v12 = (__int64)s + 8 * *(unsigned int *)&v229[4];
        goto LABEL_93;
      }
      v101 = sub_C8CA60((__int64)&v227, v95);
      v43 = v34 + 2;
      if ( v101 )
        goto LABEL_39;
      v97 = v34[3];
      v98 = v34 + 3;
      if ( v230 )
      {
        v40 = s;
        v12 = (__int64)s + 8 * *(unsigned int *)&v229[4];
        goto LABEL_102;
      }
      v99 = sub_C8CA60((__int64)&v227, v97);
      v98 = v34 + 3;
      if ( v99 )
        goto LABEL_106;
    }
LABEL_108:
    v34 += 4;
    if ( v34 != v38 )
      continue;
    break;
  }
  v37 = v33 - v34;
LABEL_110:
  if ( v37 != 2 )
  {
    if ( v37 != 3 )
    {
      if ( v37 != 1 )
        goto LABEL_113;
LABEL_279:
      if ( (unsigned __int8)sub_B19060((__int64)&v227, *v34, v12, v37) )
        goto LABEL_40;
LABEL_113:
      v34 = v33;
      goto LABEL_46;
    }
    v178 = sub_B19060((__int64)&v227, *v34, v12, 3);
    v37 = 3;
    if ( !v178 )
    {
      ++v34;
      goto LABEL_277;
    }
    if ( v33 == v34 )
      goto LABEL_46;
LABEL_42:
    v45 = v37;
    v194 = v5;
    v46 = v37;
    do
    {
      v47 = 8 * v45;
      v48 = (__int64 *)sub_2207800(8 * v45, &unk_435FF63);
      if ( v48 )
      {
        v37 = v46;
        v49 = v45;
        v31 = 8LL * v190;
        v5 = v194;
        goto LABEL_45;
      }
      v45 >>= 1;
    }
    while ( v45 );
    v37 = v46;
    v31 = 8LL * v190;
    v5 = v194;
LABEL_285:
    v47 = 0;
    v48 = 0;
    v49 = 0;
LABEL_45:
    srcc = v48;
    v34 = (__int64 *)sub_D23910(v34, v33, (__int64)&v227, v37, v48, v49);
    j_j___libc_free_0(srcc, v47);
    goto LABEL_46;
  }
LABEL_277:
  if ( !(unsigned __int8)sub_B19060((__int64)&v227, *v34, v12, v37) )
  {
    ++v34;
    goto LABEL_279;
  }
LABEL_40:
  if ( v33 != v34 )
  {
    v37 = v33 - v34;
    if ( (char *)v33 - (char *)v34 > 0 )
      goto LABEL_42;
    goto LABEL_285;
  }
LABEL_46:
  v50 = (unsigned int)v190;
  if ( v190 <= v198 )
  {
    while ( 1 )
    {
      v93 = *(_QWORD *)(v5[1] + v31);
      if ( (v5[8] & 1) != 0 )
      {
        v89 = v5 + 9;
        v90 = 3;
      }
      else
      {
        v94 = *((unsigned int *)v5 + 20);
        v89 = (__int64 *)v5[9];
        if ( !(_DWORD)v94 )
          goto LABEL_145;
        v90 = v94 - 1;
      }
      v12 = v90 & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
      v91 = &v89[2 * v12];
      v92 = *v91;
      if ( v93 != *v91 )
        break;
LABEL_88:
      *((_DWORD *)v91 + 2) = v50;
      v50 = (unsigned int)(v50 + 1);
      v31 += 8;
      if ( (_DWORD)v50 == v198 + 1 )
        goto LABEL_47;
    }
    v109 = 1;
    while ( v92 != -4096 )
    {
      v179 = v109 + 1;
      v12 = v90 & (unsigned int)(v109 + v12);
      v91 = &v89[2 * (unsigned int)v12];
      v92 = *v91;
      if ( v93 == *v91 )
        goto LABEL_88;
      v109 = v179;
    }
    if ( (v5[8] & 1) != 0 )
    {
      v108 = 8;
    }
    else
    {
      v94 = *((unsigned int *)v5 + 20);
LABEL_145:
      v108 = 2 * v94;
    }
    v91 = &v89[v108];
    goto LABEL_88;
  }
LABEL_47:
  v51 = v200;
  if ( !(unsigned __int8)sub_B19060((__int64)&v227, v200, v12, v50) )
  {
    destb = v34 - 1;
    srca = v34 - 1;
    if ( v230 )
      goto LABEL_51;
    goto LABEL_161;
  }
  v54 = v5[1];
  v55 = ((__int64)v34 - v54) >> 3;
  v191 = v55;
  v195 = v55 + 1;
  if ( v198 <= (int)v55 + 1 )
  {
    desta = (void *)(int)v55;
    goto LABEL_50;
  }
  v227 = (__int64 *)((char *)v227 + 1);
  if ( v230 )
  {
LABEL_167:
    *(_QWORD *)&v229[4] = 0;
  }
  else
  {
    v112 = 4 * (*(_DWORD *)&v229[4] - *(_DWORD *)&v229[8]);
    if ( v112 < 0x20 )
      v112 = 32;
    if ( v112 >= *(_DWORD *)v229 )
    {
      memset(s, -1, 8LL * *(unsigned int *)v229);
      goto LABEL_167;
    }
    sub_C8C990((__int64)&v227, v200);
  }
  v113 = v200;
  sub_AE6EC0((__int64)&v227, v200);
  v114 = v226;
  v226[0] = v200;
  v115 = v187;
  v224 = v226;
  v225 = 0x400000001LL;
  v116 = 1;
  while ( 2 )
  {
    v117 = v116--;
    v118 = v114[v117 - 1];
    LODWORD(v225) = v116;
    v119 = *(_QWORD *)(v118 + 8);
    v120 = *(unsigned int *)(v118 + 16);
    v53 = v119 + 8 * v120;
    if ( v119 == v53 )
      goto LABEL_187;
    while ( 2 )
    {
      v120 = *(_QWORD *)(*(_QWORD *)v119 + 24LL);
      v121 = v120 + 8LL * *(unsigned int *)(*(_QWORD *)v119 + 32LL);
      if ( v120 == v121 )
        goto LABEL_185;
      while ( 2 )
      {
        v122 = *(_QWORD *)v120;
        v123 = v120;
        if ( (*(_QWORD *)v120 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*(_QWORD *)v120 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          if ( v121 == v120 )
            break;
          v124 = v115;
          while ( 1 )
          {
            if ( (v122 & 4) == 0 )
              goto LABEL_181;
            v125 = v122 & 0xFFFFFFFFFFFFFFF8LL;
            v126 = *(_DWORD *)(*v5 + 328);
            v127 = *(_QWORD *)(*v5 + 312);
            if ( !v126 )
              goto LABEL_303;
            v113 = (unsigned int)(v126 - 1);
            v128 = (unsigned int)v113 & (((unsigned int)v125 >> 9) ^ ((unsigned int)v125 >> 4));
            v129 = (__int64 *)(v127 + 16 * v128);
            v130 = *v129;
            if ( *v129 != v125 )
            {
              v156 = 1;
              while ( v130 != -4096 )
              {
                v157 = v156 + 1;
                v128 = (unsigned int)v113 & (v156 + (_DWORD)v128);
                v129 = (__int64 *)(v127 + 16LL * (unsigned int)v128);
                v130 = *v129;
                if ( v125 == *v129 )
                  goto LABEL_180;
                v156 = v157;
              }
LABEL_303:
              BUG();
            }
LABEL_180:
            v131 = (_QWORD *)v129[1];
            if ( v5 != (__int64 *)*v131 )
              goto LABEL_181;
            v113 = v5[8] & 1;
            if ( (v5[8] & 1) != 0 )
            {
              v148 = v5 + 9;
              v149 = 3;
            }
            else
            {
              v155 = *((unsigned int *)v5 + 20);
              v148 = (_QWORD *)v5[9];
              if ( !(_DWORD)v155 )
                goto LABEL_234;
              v149 = v155 - 1;
            }
            v128 = v149 & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
            v150 = &v148[2 * v128];
            v151 = (_QWORD *)*v150;
            if ( (_QWORD *)*v150 == v131 )
              goto LABEL_213;
            v160 = 1;
            while ( v151 != (_QWORD *)-4096LL )
            {
              v175 = v160 + 1;
              v128 = v149 & (unsigned int)(v160 + v128);
              v150 = &v148[2 * (unsigned int)v128];
              v151 = (_QWORD *)*v150;
              if ( v131 == (_QWORD *)*v150 )
                goto LABEL_213;
              v160 = v175;
            }
            if ( (_BYTE)v113 )
            {
              v159 = 8;
              goto LABEL_235;
            }
            v155 = *((unsigned int *)v5 + 20);
LABEL_234:
            v159 = 2 * v155;
LABEL_235:
            v150 = &v148[v159];
LABEL_213:
            if ( v124 >= *((_DWORD *)v150 + 2) )
              goto LABEL_181;
            if ( !v230 )
              goto LABEL_231;
            v152 = s;
            v128 = *(unsigned int *)&v229[4];
            v148 = (char *)s + 8 * *(unsigned int *)&v229[4];
            if ( s != v148 )
            {
              while ( v131 != (_QWORD *)*v152 )
              {
                if ( v148 == ++v152 )
                  goto LABEL_218;
              }
              goto LABEL_181;
            }
LABEL_218:
            if ( *(_DWORD *)&v229[4] < *(_DWORD *)v229 )
            {
              ++*(_DWORD *)&v229[4];
              *v148 = v131;
              v227 = (__int64 *)((char *)v227 + 1);
            }
            else
            {
LABEL_231:
              v113 = (__int64)v131;
              srce = (void *)v53;
              v218 = v131;
              sub_C8CC70((__int64)&v227, (__int64)v131, (__int64)v148, v128, (__int64)v131, v53);
              v131 = v218;
              v53 = (__int64)srce;
              if ( !v158 )
                goto LABEL_181;
            }
            v153 = (unsigned int)v225;
            v154 = (unsigned int)v225 + 1LL;
            if ( v154 > HIDWORD(v225) )
            {
              v113 = (__int64)v226;
              srcd = (void *)v53;
              v220 = v131;
              sub_C8D5F0((__int64)&v224, v226, v154, 8u, (__int64)v131, v53);
              v153 = (unsigned int)v225;
              v53 = (__int64)srcd;
              v131 = v220;
            }
            v224[v153] = v131;
            LODWORD(v225) = v225 + 1;
LABEL_181:
            v120 = v123 + 8;
            if ( v121 != v123 + 8 )
            {
              while ( 1 )
              {
                v122 = *(_QWORD *)v120;
                v123 = v120;
                if ( (*(_QWORD *)v120 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                {
                  if ( *(_QWORD *)(*(_QWORD *)v120 & 0xFFFFFFFFFFFFFFF8LL) )
                    break;
                }
                v120 += 8;
                if ( v121 == v120 )
                  goto LABEL_184;
              }
              if ( v121 != v120 )
                continue;
            }
LABEL_184:
            v115 = v124;
            goto LABEL_185;
          }
        }
        v120 += 8;
        if ( v121 != v120 )
          continue;
        break;
      }
LABEL_185:
      v119 += 8;
      if ( v53 != v119 )
        continue;
      break;
    }
    v116 = v225;
    v114 = v224;
LABEL_187:
    if ( v116 )
      continue;
    break;
  }
  if ( v114 != v226 )
    _libc_free(v114, v113);
  v132 = v5[1];
  v133 = 8LL * v191 + 8;
  desta = (void *)v191;
  v134 = (__int64 *)(v132 + v184);
  v135 = (__int64 *)(v133 + v132);
  v136 = (v184 - v133) >> 5;
  v137 = (v184 - v133) >> 3;
  if ( v136 <= 0 )
  {
LABEL_206:
    if ( v137 != 2 )
    {
      if ( v137 != 3 )
      {
        if ( v137 != 1 )
        {
LABEL_209:
          v135 = v134;
          goto LABEL_247;
        }
LABEL_297:
        if ( !(unsigned __int8)sub_B19060((__int64)&v227, *v135, v120, v137) )
          goto LABEL_241;
        goto LABEL_209;
      }
      v181 = sub_B19060((__int64)&v227, *v135, v120, 3);
      v137 = 3;
      if ( !v181 )
      {
        if ( v134 != v135 )
          goto LABEL_243;
        goto LABEL_247;
      }
      ++v135;
    }
    if ( !(unsigned __int8)sub_B19060((__int64)&v227, *v135, v120, v137) )
      goto LABEL_241;
    ++v135;
    goto LABEL_297;
  }
  v138 = v135;
  v217 = &v135[4 * v136];
  while ( 1 )
  {
    v139 = *v138;
    if ( v230 )
    {
      v140 = (char *)s;
      v141 = (char *)s + 8 * *(unsigned int *)&v229[4];
      if ( s == v141 )
      {
LABEL_240:
        v133 = 8LL * v191 + 8;
        v135 = v138;
        goto LABEL_241;
      }
      v142 = s;
      while ( v139 != *v142 )
      {
        if ( v141 == (char *)++v142 )
          goto LABEL_240;
      }
      v143 = v138[1];
      v144 = v138 + 1;
      v145 = s;
      do
      {
LABEL_199:
        if ( v143 == *(_QWORD *)v140 )
        {
          v146 = v138[2];
          v147 = v138 + 2;
          goto LABEL_202;
        }
        v140 += 8;
      }
      while ( v141 != v140 );
      goto LABEL_258;
    }
    if ( !sub_C8CA60((__int64)&v227, v139) )
      goto LABEL_240;
    v143 = v138[1];
    v144 = v138 + 1;
    if ( v230 )
    {
      v140 = (char *)s;
      v141 = (char *)s + 8 * *(unsigned int *)&v229[4];
      if ( v141 != s )
      {
        v145 = s;
        goto LABEL_199;
      }
LABEL_258:
      v133 = 8LL * v191 + 8;
      v135 = v144;
      goto LABEL_241;
    }
    v173 = sub_C8CA60((__int64)&v227, v143);
    v144 = v138 + 1;
    if ( !v173 )
      goto LABEL_258;
    v146 = v138[2];
    v147 = v138 + 2;
    if ( v230 )
    {
      v145 = s;
      v141 = (char *)s + 8 * *(unsigned int *)&v229[4];
      if ( v141 == s )
        goto LABEL_262;
LABEL_202:
      while ( *v145 != v146 )
      {
        if ( ++v145 == (_QWORD *)v141 )
          goto LABEL_262;
      }
    }
    else
    {
      v174 = sub_C8CA60((__int64)&v227, v146);
      v147 = v138 + 2;
      if ( !v174 )
      {
LABEL_262:
        v133 = 8LL * v191 + 8;
        v135 = v147;
        goto LABEL_241;
      }
    }
    if ( !(unsigned __int8)sub_B19060((__int64)&v227, v138[3], (__int64)v145, (__int64)v141) )
      break;
    v138 += 4;
    if ( v217 == v138 )
    {
      v133 = 8LL * v191 + 8;
      v135 = v138;
      v137 = v134 - v138;
      goto LABEL_206;
    }
  }
  v133 = 8LL * v191 + 8;
  v135 = v138 + 3;
LABEL_241:
  if ( v134 != v135 )
  {
    v137 = v134 - v135;
    if ( (char *)v134 - (char *)v135 > 0 )
    {
LABEL_243:
      srcb = v5;
      v161 = v133;
      v162 = v137;
      do
      {
        v163 = 8 * v162;
        v192 = v137;
        v164 = sub_2207800(8 * v162, &unk_435FF63);
        v137 = v192;
        v165 = (__int64 *)v164;
        if ( v164 )
        {
          v166 = v162;
          v133 = v161;
          v5 = srcb;
          goto LABEL_246;
        }
        v162 >>= 1;
      }
      while ( v162 );
      v133 = v161;
      v5 = srcb;
    }
    v163 = 0;
    v165 = 0;
    v166 = 0;
LABEL_246:
    v219 = v165;
    v135 = (__int64 *)sub_D23A60(v135, v134, (__int64)&v227, v137, v165, v166);
    j_j___libc_free_0(v219, v163);
  }
LABEL_247:
  LODWORD(v51) = v195;
  v167 = v198 + 1;
  while ( 1 )
  {
LABEL_251:
    v52 = *(_QWORD *)(v5[1] + v133);
    if ( (v5[8] & 1) != 0 )
    {
      v168 = v5 + 9;
      v53 = 3;
    }
    else
    {
      v172 = *((unsigned int *)v5 + 20);
      v168 = (__int64 *)v5[9];
      if ( !(_DWORD)v172 )
        goto LABEL_268;
      v53 = (unsigned int)(v172 - 1);
    }
    v169 = v53 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
    v170 = &v168[2 * v169];
    v171 = *v170;
    if ( v52 != *v170 )
      break;
LABEL_250:
    *((_DWORD *)v170 + 2) = v51;
    v51 = (unsigned int)(v51 + 1);
    v133 += 8;
    if ( v167 == (_DWORD)v51 )
      goto LABEL_270;
  }
  v177 = 1;
  while ( v171 != -4096 )
  {
    v180 = v177 + 1;
    v169 = v53 & (v177 + v169);
    v170 = &v168[2 * v169];
    v171 = *v170;
    if ( v52 == *v170 )
      goto LABEL_250;
    v177 = v180;
  }
  if ( (v5[8] & 1) == 0 )
  {
    v172 = *((unsigned int *)v5 + 20);
LABEL_268:
    v176 = 2 * v172;
    goto LABEL_269;
  }
  v176 = 8;
LABEL_269:
  v133 += 8;
  LODWORD(v168[v176 + 1]) = v51;
  v51 = (unsigned int)(v51 + 1);
  if ( v167 != (_DWORD)v51 )
    goto LABEL_251;
LABEL_270:
  v54 = v5[1];
  v214 = (int)(((__int64)v135 - v54 - 8) >> 3);
LABEL_50:
  srca = (__int64 *)(v54 + 8 * v214);
  destb = (__int64 *)(v54 + 8LL * (_QWORD)desta);
  if ( !v230 )
LABEL_161:
    _libc_free(s, v51);
LABEL_51:
  if ( a4 )
    a4(a5, destb, srca - destb);
  v199 = a2 + 24;
  if ( srca == destb )
    goto LABEL_143;
  v56 = v200;
  v215 = destb;
  v185 = v200 + 8;
  do
  {
    v57 = *v215;
    if ( (v5[8] & 1) != 0 )
    {
      v58 = v5 + 9;
      v59 = 3;
    }
    else
    {
      v106 = *((_DWORD *)v5 + 20);
      v58 = (__int64 *)v5[9];
      if ( !v106 )
        goto LABEL_59;
      v59 = v106 - 1;
    }
    v60 = v59 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
    v61 = &v58[2 * v60];
    v52 = *v61;
    if ( v57 == *v61 )
    {
LABEL_58:
      *v61 = -8192;
      v62 = *((_DWORD *)v5 + 16);
      ++*((_DWORD *)v5 + 17);
      *((_DWORD *)v5 + 16) = (2 * (v62 >> 1) - 2) | v62 & 1;
    }
    else
    {
      v110 = 1;
      while ( v52 != -4096 )
      {
        v111 = v110 + 1;
        v60 = v59 & (v110 + v60);
        v61 = &v58[2 * v60];
        v52 = *v61;
        if ( v57 == *v61 )
          goto LABEL_58;
        v110 = v111;
      }
    }
LABEL_59:
    v63 = *(unsigned int *)(v57 + 16);
    v64 = *(unsigned int *)(v56 + 16);
    v65 = *(const void **)(v57 + 8);
    if ( v63 + v64 > (unsigned __int64)*(unsigned int *)(v56 + 20) )
    {
      v203 = v57;
      sub_C8D5F0(v185, (const void *)(v56 + 24), v63 + v64, 8u, v52, v53);
      v64 = *(unsigned int *)(v56 + 16);
      v57 = v203;
    }
    if ( 8 * v63 )
    {
      v201 = v57;
      memcpy((void *)(*(_QWORD *)(v56 + 8) + 8 * v64), v65, 8 * v63);
      LODWORD(v64) = *(_DWORD *)(v56 + 16);
      v57 = v201;
    }
    *(_DWORD *)(v56 + 16) = v63 + v64;
    v66 = *(__int64 **)(v57 + 8);
    if ( &v66[*(unsigned int *)(v57 + 16)] != v66 )
    {
      v202 = v57;
      v67 = &v66[*(unsigned int *)(v57 + 16)];
      while ( 1 )
      {
        v71 = *v5;
        v72 = *v66;
        v224 = (_QWORD *)*v66;
        v52 = *(unsigned int *)(v71 + 328);
        if ( !(_DWORD)v52 )
          break;
        v53 = *(_QWORD *)(v71 + 312);
        v68 = (v52 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
        v69 = (__int64 *)(v53 + 16LL * v68);
        v70 = *v69;
        if ( v72 == *v69 )
        {
LABEL_66:
          ++v66;
          v69[1] = v56;
          if ( v67 == v66 )
            goto LABEL_74;
        }
        else
        {
          v197 = 1;
          v74 = 0;
          while ( v70 != -4096 )
          {
            if ( !v74 && v70 == -8192 )
              v74 = v69;
            v68 = (v52 - 1) & (v197 + v68);
            v53 = (unsigned int)(v197 + 1);
            v69 = (__int64 *)(*(_QWORD *)(v71 + 312) + 16LL * v68);
            v70 = *v69;
            if ( v72 == *v69 )
              goto LABEL_66;
            ++v197;
          }
          if ( !v74 )
            v74 = v69;
          v227 = v74;
          v107 = *(_DWORD *)(v71 + 320);
          ++*(_QWORD *)(v71 + 304);
          v75 = v107 + 1;
          if ( 4 * (v107 + 1) >= (unsigned int)(3 * v52) )
            goto LABEL_69;
          if ( (int)v52 - *(_DWORD *)(v71 + 324) - v75 > (unsigned int)v52 >> 3 )
            goto LABEL_71;
          v196 = v67;
          v73 = v52;
LABEL_70:
          sub_D25CB0(v71 + 304, v73);
          sub_D24C50(v71 + 304, (__int64 *)&v224, &v227);
          v74 = v227;
          v72 = (__int64)v224;
          v67 = v196;
          v75 = *(_DWORD *)(v71 + 320) + 1;
LABEL_71:
          *(_DWORD *)(v71 + 320) = v75;
          if ( *v74 != -4096 )
            --*(_DWORD *)(v71 + 324);
          ++v66;
          *v74 = v72;
          v74[1] = 0;
          v74[1] = v56;
          if ( v67 == v66 )
          {
LABEL_74:
            v57 = v202;
            goto LABEL_75;
          }
        }
      }
      v227 = 0;
      ++*(_QWORD *)(v71 + 304);
LABEL_69:
      v196 = v67;
      v73 = 2 * v52;
      goto LABEL_70;
    }
LABEL_75:
    *(_QWORD *)v57 = 0;
    v76 = (unsigned int)v222;
    v77 = HIDWORD(v222);
    *(_DWORD *)(v57 + 16) = 0;
    if ( v76 + 1 > v77 )
    {
      v204 = v57;
      sub_C8D5F0((__int64)&v221, v223, v76 + 1, 8u, v52, v53);
      v76 = (unsigned int)v222;
      v57 = v204;
    }
    ++v215;
    *(_QWORD *)&v221[8 * v76] = v57;
    LODWORD(v222) = v222 + 1;
  }
  while ( srca != v215 );
  v78 = v5[1];
  v79 = v78 + 8LL * *((unsigned int *)v5 + 4) - (_QWORD)srca;
  if ( srca != (__int64 *)(v78 + 8LL * *((unsigned int *)v5 + 4)) )
  {
    memmove(destb, srca, v78 + 8LL * *((unsigned int *)v5 + 4) - (_QWORD)srca);
    v78 = v5[1];
  }
  v80 = (char *)destb + v79;
  v81 = destb;
  v82 = (__int64)&v80[-v78] >> 3;
  *((_DWORD *)v5 + 4) = v82;
  v83 = (__int64 *)(v78 + 8LL * (unsigned int)v82);
  if ( destb != v83 )
  {
    do
    {
      v84 = *v81++;
      v227 = (__int64 *)v84;
      v85 = (_DWORD *)sub_D25AF0(v188, (__int64 *)&v227);
      *v85 -= srca - destb;
    }
    while ( v83 != v81 );
  }
  v86 = a3;
  v87 = 1;
  sub_D23D60(v199, a3, 1u);
LABEL_83:
  if ( v221 != v223 )
    _libc_free(v221, v86);
  return v87;
}
