// Function: sub_2AFD350
// Address: 0x2afd350
//
char __fastcall sub_2AFD350(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, int a5)
{
  __int64 v5; // r12
  unsigned int v7; // eax
  unsigned __int8 *v8; // r10
  unsigned int v9; // r13d
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  char result; // al
  bool v17; // zf
  __int64 *v18; // r14
  __int64 *v19; // rax
  _QWORD *v20; // rbx
  _QWORD *v21; // r13
  unsigned __int64 v22; // r12
  _QWORD *v23; // rax
  __int64 *v24; // rdi
  __int64 *v25; // r15
  _QWORD *v26; // rbx
  unsigned __int8 v27; // al
  __int64 v28; // rdx
  unsigned __int8 *v29; // r15
  unsigned __int8 **v30; // r15
  __int64 v31; // rbx
  unsigned __int8 *v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r14
  int v36; // eax
  unsigned __int8 **v37; // r13
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rax
  __int64 v40; // r14
  unsigned __int8 v41; // si
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rax
  __int64 v44; // rbx
  unsigned __int8 v45; // si
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  char *v48; // rdi
  unsigned __int64 v49; // rdx
  __int64 *v50; // r15
  _QWORD *v51; // rbx
  _QWORD *v52; // r12
  unsigned __int64 v53; // r13
  _QWORD *v54; // rdx
  unsigned __int8 *v55; // r9
  unsigned __int8 v56; // al
  unsigned __int8 *v57; // r14
  unsigned __int8 v58; // dl
  __int64 v59; // rdx
  unsigned int v60; // ecx
  __int64 v61; // rdi
  __int64 v62; // r8
  unsigned __int64 v63; // rdi
  unsigned __int64 v64; // rsi
  __int64 v65; // rax
  char v66; // bl
  __int64 v67; // rax
  __int64 v68; // rdx
  unsigned __int64 v69; // r15
  __int64 *v70; // rcx
  unsigned __int8 *v71; // r14
  unsigned __int8 *v72; // r15
  unsigned __int8 v73; // dl
  __int64 v74; // rax
  _QWORD *v75; // rdx
  unsigned __int64 v76; // rbx
  bool v77; // cc
  unsigned __int64 v78; // rcx
  __int64 v79; // rax
  _QWORD *v80; // r10
  _QWORD *v81; // rax
  char v82; // di
  __int64 v83; // rax
  _QWORD *v84; // r10
  int v85; // eax
  unsigned __int64 v86; // rdi
  unsigned __int64 v87; // rax
  int v88; // eax
  int v89; // eax
  unsigned __int64 v90; // rax
  int v91; // eax
  unsigned __int8 *v92; // rt0
  int v93; // eax
  int v94; // eax
  __int64 v95; // rax
  int v96; // eax
  unsigned __int64 v97; // rdx
  char v98; // bl
  __int64 *v99; // rax
  __int64 *v100; // r14
  __int64 *v101; // r15
  _QWORD *v102; // r14
  __int64 *v103; // rdi
  __int64 *v104; // r12
  __int64 v105; // rdx
  unsigned __int8 *v106; // rdx
  __int64 v107; // rax
  bool v108; // r12
  unsigned int v109; // eax
  __int64 v110; // rdx
  unsigned __int8 *v111; // rcx
  __int64 v112; // rsi
  bool v113; // bl
  __int64 v114; // rax
  __int64 v115; // r14
  __int64 v116; // rbx
  unsigned __int8 v117; // al
  __int64 v118; // r13
  _QWORD *v119; // rdi
  __int64 v120; // rcx
  _QWORD *v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // rsi
  __int64 v124; // r9
  __int64 v125; // rax
  __int64 v126; // rdx
  __int64 v127; // r8
  __int64 v128; // r13
  _QWORD *v129; // rsi
  __int64 v130; // rcx
  unsigned int v131; // edi
  __int64 *v132; // rax
  __int64 v133; // rax
  __int64 v134; // rdx
  unsigned int v135; // r9d
  __int64 *v136; // rax
  __int64 v137; // rax
  __int64 v138; // rcx
  __int64 v139; // [rsp+0h] [rbp-140h]
  __int64 v140; // [rsp+8h] [rbp-138h]
  __int64 v141; // [rsp+10h] [rbp-130h]
  unsigned __int64 v142; // [rsp+18h] [rbp-128h]
  unsigned __int64 v143; // [rsp+20h] [rbp-120h]
  unsigned __int64 v144; // [rsp+20h] [rbp-120h]
  _QWORD *v145; // [rsp+20h] [rbp-120h]
  _QWORD *v146; // [rsp+20h] [rbp-120h]
  _QWORD *v147; // [rsp+20h] [rbp-120h]
  __int64 v148; // [rsp+20h] [rbp-120h]
  _QWORD *v149; // [rsp+28h] [rbp-118h]
  _QWORD *v150; // [rsp+28h] [rbp-118h]
  __int64 v151; // [rsp+28h] [rbp-118h]
  _QWORD *v152; // [rsp+28h] [rbp-118h]
  __int64 v153; // [rsp+28h] [rbp-118h]
  __int64 v154; // [rsp+28h] [rbp-118h]
  bool v155; // [rsp+28h] [rbp-118h]
  unsigned int v156; // [rsp+30h] [rbp-110h]
  __int64 v157; // [rsp+38h] [rbp-108h]
  _QWORD *v158; // [rsp+38h] [rbp-108h]
  _QWORD *v159; // [rsp+38h] [rbp-108h]
  char *v160; // [rsp+38h] [rbp-108h]
  unsigned __int64 *v161; // [rsp+40h] [rbp-100h]
  _QWORD *v162; // [rsp+40h] [rbp-100h]
  _QWORD *v163; // [rsp+48h] [rbp-F8h]
  __int64 v164; // [rsp+48h] [rbp-F8h]
  _QWORD *v165; // [rsp+48h] [rbp-F8h]
  _QWORD *v166; // [rsp+48h] [rbp-F8h]
  unsigned __int8 *v168; // [rsp+50h] [rbp-F0h]
  __int64 v169; // [rsp+50h] [rbp-F0h]
  unsigned __int8 *v170; // [rsp+58h] [rbp-E8h]
  unsigned __int8 *v171; // [rsp+58h] [rbp-E8h]
  unsigned int v172; // [rsp+58h] [rbp-E8h]
  unsigned int v173; // [rsp+58h] [rbp-E8h]
  __int64 v174; // [rsp+58h] [rbp-E8h]
  __int64 *v176; // [rsp+60h] [rbp-E0h]
  unsigned __int8 **v177; // [rsp+60h] [rbp-E0h]
  unsigned __int8 v178; // [rsp+60h] [rbp-E0h]
  unsigned __int8 *v179; // [rsp+60h] [rbp-E0h]
  unsigned __int8 *v180; // [rsp+60h] [rbp-E0h]
  __int64 v181; // [rsp+60h] [rbp-E0h]
  unsigned __int8 *v182; // [rsp+68h] [rbp-D8h]
  char v183; // [rsp+68h] [rbp-D8h]
  char v184; // [rsp+68h] [rbp-D8h]
  char v185; // [rsp+68h] [rbp-D8h]
  char v186; // [rsp+68h] [rbp-D8h]
  char v187; // [rsp+68h] [rbp-D8h]
  unsigned __int8 **v188; // [rsp+68h] [rbp-D8h]
  char v189; // [rsp+68h] [rbp-D8h]
  __int64 v190; // [rsp+68h] [rbp-D8h]
  __int64 v191; // [rsp+68h] [rbp-D8h]
  unsigned __int8 *v192; // [rsp+68h] [rbp-D8h]
  char v193; // [rsp+68h] [rbp-D8h]
  __int64 *v194; // [rsp+68h] [rbp-D8h]
  char *v195; // [rsp+70h] [rbp-D0h] BYREF
  unsigned int v196; // [rsp+78h] [rbp-C8h]
  char *v197; // [rsp+80h] [rbp-C0h] BYREF
  unsigned int v198; // [rsp+88h] [rbp-B8h]
  unsigned __int64 v199; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v200; // [rsp+98h] [rbp-A8h]
  unsigned __int64 *v201; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned int v202; // [rsp+A8h] [rbp-98h]
  unsigned __int64 v203; // [rsp+B0h] [rbp-90h] BYREF
  unsigned int v204; // [rsp+B8h] [rbp-88h]
  char *v205; // [rsp+C0h] [rbp-80h] BYREF
  unsigned int v206; // [rsp+C8h] [rbp-78h]
  unsigned __int64 v207; // [rsp+D0h] [rbp-70h] BYREF
  unsigned __int64 v208; // [rsp+D8h] [rbp-68h] BYREF
  unsigned int v209; // [rsp+E0h] [rbp-60h]
  unsigned __int64 v210; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v211; // [rsp+F8h] [rbp-48h]
  __int64 *v212; // [rsp+100h] [rbp-40h] BYREF
  _QWORD *v213; // [rsp+108h] [rbp-38h]

  v5 = a1;
  v7 = sub_AE43F0(*(_QWORD *)(a1 + 48), *(_QWORD *)(a2 + 8));
  v8 = (unsigned __int8 *)a2;
  v196 = v7;
  v9 = v7;
  if ( v7 > 0x40 )
  {
    sub_C43690((__int64)&v195, 0, 0);
    v198 = v9;
    sub_C43690((__int64)&v197, 0, 0);
    v8 = (unsigned __int8 *)a2;
  }
  else
  {
    v198 = v7;
    v195 = 0;
    v197 = 0;
  }
  v170 = sub_BD45C0(v8, *(_QWORD *)(a1 + 48), (__int64)&v195, 0, 0, 0, 0, 0);
  v182 = sub_BD45C0(a3, *(_QWORD *)(a1 + 48), (__int64)&v197, 0, 0, 0, 0, 0);
  v10 = sub_9208B0(*(_QWORD *)(a1 + 48), *((_QWORD *)v170 + 1));
  v211 = v11;
  v210 = (v10 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v12 = sub_CA1930(&v210);
  v13 = sub_9208B0(*(_QWORD *)(a1 + 48), *((_QWORD *)v182 + 1));
  v211 = v14;
  v210 = (v13 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v15 = sub_CA1930(&v210);
  result = 0;
  if ( v15 != v12 )
    goto LABEL_4;
  sub_C44B10((__int64)&v210, &v195, v12);
  if ( v196 > 0x40 && v195 )
    j_j___libc_free_0_0((unsigned __int64)v195);
  v195 = (char *)v210;
  v196 = v211;
  sub_C44B10((__int64)&v210, &v197, v12);
  if ( v198 > 0x40 && v197 )
    j_j___libc_free_0_0((unsigned __int64)v197);
  v197 = (char *)v210;
  v198 = v211;
  sub_C44B10((__int64)&v210, (char **)a4, v12);
  if ( *(_DWORD *)(a4 + 8) > 0x40u && *(_QWORD *)a4 )
    j_j___libc_free_0_0(*(_QWORD *)a4);
  *(_QWORD *)a4 = v210;
  *(_DWORD *)(a4 + 8) = v211;
  LODWORD(v211) = v198;
  if ( v198 > 0x40 )
    sub_C43780((__int64)&v210, (const void **)&v197);
  else
    v210 = (unsigned __int64)v197;
  sub_C46B40((__int64)&v210, (__int64 *)&v195);
  v200 = v211;
  v199 = v210;
  if ( v182 != v170 )
  {
    LODWORD(v211) = *(_DWORD *)(a4 + 8);
    if ( (unsigned int)v211 > 0x40 )
      sub_C43780((__int64)&v210, (const void **)a4);
    else
      v210 = *(_QWORD *)a4;
    sub_C46B40((__int64)&v210, (__int64 *)&v199);
    v17 = *(_BYTE *)(a1 + 236) == 0;
    v202 = v211;
    v201 = (unsigned __int64 *)v210;
    if ( v17 )
    {
      v18 = sub_DD8400(*(_QWORD *)(a1 + 32), (__int64)v170);
      v176 = sub_DD8400(*(_QWORD *)(a1 + 32), (__int64)v182);
      v23 = sub_DA26C0(*(__int64 **)(a1 + 32), (__int64)&v201);
      v24 = *(__int64 **)(a1 + 32);
      v213 = v23;
      v210 = (unsigned __int64)&v212;
      v212 = v18;
      v211 = 0x200000002LL;
      v25 = sub_DC7EB0(v24, (__int64)&v210, 0, 0);
      if ( (__int64 **)v210 != &v212 )
        _libc_free(v210);
LABEL_43:
      if ( v176 == v25
        || (_BYTE)qword_500EE68
        && (v26 = sub_DA26C0(*(__int64 **)(v5 + 32), (__int64)&v201),
            v26 == sub_DCC810(*(__int64 **)(v5 + 32), (__int64)v176, (__int64)v18, 0, 0)) )
      {
        result = 1;
        goto LABEL_54;
      }
      v204 = v202;
      if ( v202 > 0x40 )
        sub_C43780((__int64)&v203, (const void **)&v201);
      else
        v203 = (unsigned __int64)v201;
      v27 = *v182;
      if ( *v170 != 63 )
      {
        if ( v27 != 63 && *v170 == 86 && v27 == 86 && a5 != 3 && *((_QWORD *)v170 - 12) == *((_QWORD *)v182 - 12) )
        {
          LODWORD(v211) = v204;
          if ( v204 > 0x40 )
            sub_C43780((__int64)&v210, (const void **)&v203);
          else
            v210 = v203;
          result = sub_2AFD350(v5, *((_QWORD *)v170 - 8), *((_QWORD *)v182 - 8), &v210, (unsigned int)(a5 + 1));
          if ( result )
          {
            LODWORD(v208) = v204;
            if ( v204 > 0x40 )
              sub_C43780((__int64)&v207, (const void **)&v203);
            else
              v207 = v203;
            result = sub_2AFD350(v5, *((_QWORD *)v170 - 4), *((_QWORD *)v182 - 4), &v207, (unsigned int)(a5 + 1));
            if ( (unsigned int)v208 > 0x40 )
            {
              if ( v207 )
              {
                v193 = result;
                j_j___libc_free_0_0(v207);
                result = v193;
              }
            }
          }
          if ( (unsigned int)v211 <= 0x40 )
            goto LABEL_51;
          v48 = (char *)v210;
          if ( !v210 )
            goto LABEL_51;
          goto LABEL_100;
        }
        goto LABEL_50;
      }
      if ( v27 != 63 )
        goto LABEL_50;
      v28 = *((_DWORD *)v170 + 1) & 0x7FFFFFF;
      if ( (_DWORD)v28 != (*((_DWORD *)v182 + 1) & 0x7FFFFFF)
        || *(_QWORD *)&v170[-32 * v28] != *(_QWORD *)&v182[-32 * v28] )
      {
        goto LABEL_50;
      }
      if ( (v170[7] & 0x40) != 0 )
        v29 = (unsigned __int8 *)*((_QWORD *)v170 - 1);
      else
        v29 = &v170[-32 * v28];
      v30 = (unsigned __int8 **)(v29 + 32);
      v31 = sub_BB5290((__int64)v170) & 0xFFFFFFFFFFFFFFF9LL | 4;
      if ( (v182[7] & 0x40) != 0 )
        v32 = (unsigned __int8 *)*((_QWORD *)v182 - 1);
      else
        v32 = &v182[-32 * (*((_DWORD *)v182 + 1) & 0x7FFFFFF)];
      v168 = v32;
      v177 = (unsigned __int8 **)(v32 + 32);
      v33 = sub_BB5290((__int64)v182);
      v34 = (__int64)v177;
      v35 = v33 & 0xFFFFFFFFFFFFFFF9LL | 4;
      v36 = *((_DWORD *)v170 + 1) & 0x7FFFFFF;
      if ( v36 != 2 )
      {
        v37 = v177;
        v188 = (unsigned __int8 **)&v168[32 * (v36 - 3) + 64];
        while ( 1 )
        {
          if ( *v30 != *v37 )
            goto LABEL_50;
          v42 = v31 & 0xFFFFFFFFFFFFFFF8LL;
          v43 = v31 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v31 )
            goto LABEL_84;
          v44 = (v31 >> 1) & 3;
          if ( v44 == 2 )
            break;
          if ( v44 != 1 || !v42 )
            goto LABEL_84;
          v43 = *(_QWORD *)(v42 + 24);
LABEL_80:
          v45 = *(_BYTE *)(v43 + 8);
          if ( v45 == 16 )
          {
            v31 = *(_QWORD *)(v43 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
          }
          else
          {
            v46 = v43 & 0xFFFFFFFFFFFFFFF9LL;
            if ( (unsigned int)v45 - 17 > 1 )
            {
              v31 = 0;
              if ( v45 == 15 )
                v31 = v46;
            }
            else
            {
              v31 = v46 | 2;
            }
          }
          v30 += 4;
          v38 = v35 & 0xFFFFFFFFFFFFFFF8LL;
          v39 = v35 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v35 )
            goto LABEL_85;
          v40 = (v35 >> 1) & 3;
          if ( v40 != 2 )
          {
            if ( v40 == 1 && v38 )
            {
              v39 = *(_QWORD *)(v38 + 24);
              goto LABEL_73;
            }
LABEL_85:
            v39 = sub_BCBAE0(v38, *v37, v34);
            goto LABEL_73;
          }
          if ( !v38 )
            goto LABEL_85;
LABEL_73:
          v41 = *(_BYTE *)(v39 + 8);
          if ( v41 == 16 )
          {
            v35 = *(_QWORD *)(v39 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
          }
          else
          {
            v47 = v39 & 0xFFFFFFFFFFFFFFF9LL;
            if ( (unsigned int)v41 - 17 > 1 )
            {
              v35 = 0;
              if ( v41 == 15 )
                v35 = v47;
            }
            else
            {
              v35 = v47 | 2;
            }
          }
          v37 += 4;
          if ( v37 == v188 )
            goto LABEL_138;
        }
        if ( v42 )
          goto LABEL_80;
LABEL_84:
        v43 = sub_BCBAE0(v42, *v30, v34);
        goto LABEL_80;
      }
      v188 = v177;
LABEL_138:
      v55 = *v30;
      v56 = **v30;
      if ( v56 <= 0x1Cu
        || (v57 = *v188, v58 = **v188, v58 <= 0x1Cu)
        || v58 != v56
        || *((_QWORD *)v55 + 1) != *((_QWORD *)v57 + 1) )
      {
LABEL_50:
        result = 0;
        goto LABEL_51;
      }
      v59 = 1;
      v60 = v204 - 1;
      v61 = 1LL << ((unsigned __int8)v204 - 1);
      if ( v204 > 0x40 )
      {
        v59 = v60 >> 6;
        if ( (*(_QWORD *)(v203 + 8 * v59) & v61) == 0 )
          goto LABEL_144;
        v172 = v204 - 1;
        v179 = *v30;
        if ( v172 == (unsigned int)sub_C44590((__int64)&v203) )
          goto LABEL_50;
        sub_C43D10((__int64)&v203);
        v55 = v179;
      }
      else
      {
        if ( (v61 & v203) == 0 )
          goto LABEL_144;
        if ( 1LL << v60 == v203 )
          goto LABEL_50;
        v97 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v204 - 1) & 0x3F));
        if ( !v204 )
          v97 = 0;
        v203 = v97 & ~v203;
      }
      v180 = v55;
      sub_C46250((__int64)&v203);
      v92 = v57;
      v57 = v180;
      v55 = v92;
LABEL_144:
      v62 = *(_QWORD *)(v5 + 48);
      v63 = v31 & 0xFFFFFFFFFFFFFFF8LL;
      v64 = v31 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v31 )
      {
        v65 = (v31 >> 1) & 3;
        if ( v65 == 2 )
        {
          if ( v63 )
            goto LABEL_147;
        }
        else if ( v65 == 1 && v63 )
        {
          v64 = *(_QWORD *)(v63 + 24);
LABEL_147:
          v171 = v55;
          v190 = v62;
          v66 = sub_AE5020(v62, v64);
          v67 = sub_9208B0(v190, v64);
          v211 = v68;
          v210 = ((1LL << v66) + ((unsigned __int64)(v67 + 7) >> 3) - 1) >> v66 << v66;
          v69 = sub_CA1930(&v210);
          if ( !sub_C459C0((__int64)&v203, v69) )
          {
            sub_C45850((__int64)&v205, (unsigned __int64 **)&v203, v69);
            v178 = *v171;
            if ( (unsigned __int8)(*v171 - 68) > 1u
              || ((v171[7] & 0x40) == 0
                ? (v70 = (__int64 *)&v171[-32 * (*((_DWORD *)v171 + 1) & 0x7FFFFFF)])
                : (v70 = (__int64 *)*((_QWORD *)v171 - 1)),
                  (v57[7] & 0x40) == 0
                ? (v71 = &v57[-32 * (*((_DWORD *)v57 + 1) & 0x7FFFFFF)])
                : (v71 = (unsigned __int8 *)*((_QWORD *)v57 - 1)),
                  v72 = *(unsigned __int8 **)v71,
                  v73 = **(_BYTE **)v71,
                  v73 <= 0x1Cu) )
            {
              result = 0;
              goto LABEL_155;
            }
            v191 = *v70;
            v164 = *(_QWORD *)(*v70 + 8);
            result = 0;
            if ( v164 != *((_QWORD *)v72 + 1) )
              goto LABEL_155;
            v98 = v178 == 69;
            if ( v73 != 42 )
              goto LABEL_225;
            if ( (v72[7] & 0x40) != 0 )
              v106 = (unsigned __int8 *)*((_QWORD *)v72 - 1);
            else
              v106 = &v72[-32 * (*((_DWORD *)v72 + 1) & 0x7FFFFFF)];
            v107 = *((_QWORD *)v106 + 4);
            if ( *(_BYTE *)v107 == 17 )
            {
              v114 = sub_2AF7540(*(_QWORD *)(v107 + 24), *(_DWORD *)(v107 + 32));
              if ( !sub_AAD930((__int64)&v205, v114) && sub_2AF6930((__int64)v72, v98) )
              {
                v173 = sub_BCB060(v164);
                goto LABEL_229;
              }
            }
            if ( *(_BYTE *)v191 != 42 || !sub_2AF6930(v191, v98) || !(v155 = sub_2AF6930((__int64)v72, v98)) )
            {
LABEL_225:
              v173 = sub_BCB060(v164);
              goto LABEL_226;
            }
            v210 = 0x100000000LL;
            v156 = v206;
            v160 = v205;
            v139 = v5;
            v108 = 0;
            v169 = 0;
            v148 = (__int64)((_QWORD)v205 << (64 - (unsigned __int8)v206)) >> (64 - (unsigned __int8)v206);
            while ( 1 )
            {
              v174 = 0;
              v109 = *(_DWORD *)((char *)&v210 + v169);
              v207 = 0x100000000LL;
              v141 = 32LL * v109;
              v140 = 32LL * (v109 != 1);
              do
              {
                if ( v108 )
                  goto LABEL_251;
                v110 = (*(_BYTE *)(v191 + 7) & 0x40) != 0
                     ? *(_QWORD *)(v191 - 8)
                     : v191 - 32LL * (*(_DWORD *)(v191 + 4) & 0x7FFFFFF);
                v111 = (v72[7] & 0x40) != 0
                     ? (unsigned __int8 *)*((_QWORD *)v72 - 1)
                     : &v72[-32 * (*((_DWORD *)v72 + 1) & 0x7FFFFFF)];
                v112 = *(unsigned int *)((char *)&v207 + v174);
                if ( *(_QWORD *)(v110 + v141) != *(_QWORD *)&v111[32 * v112] )
                  goto LABEL_251;
                v115 = *(_QWORD *)(v110 + v140);
                v116 = *(_QWORD *)&v111[32 * ((_DWORD)v112 != 1)];
                v117 = *(_BYTE *)v116;
                if ( *(_BYTE *)v115 <= 0x1Cu )
                {
                  if ( v117 != 42 )
                    goto LABEL_251;
                  v128 = 0;
LABEL_286:
                  if ( v178 == 69 )
                  {
                    if ( sub_B44900(v116) )
                    {
LABEL_288:
                      if ( (*(_BYTE *)(v116 + 7) & 0x40) != 0 )
                        v129 = *(_QWORD **)(v116 - 8);
                      else
                        v129 = (_QWORD *)(v116 - 32LL * (*(_DWORD *)(v116 + 4) & 0x7FFFFFF));
                      v130 = v129[4];
                      if ( *(_BYTE *)v130 == 17 )
                      {
                        v131 = *(_DWORD *)(v130 + 32);
                        v132 = *(__int64 **)(v130 + 24);
                        if ( v131 > 0x40 )
                          v133 = *v132;
                        else
                          v133 = v131
                               ? (__int64)((_QWORD)v132 << (64 - (unsigned __int8)v131)) >> (64 - (unsigned __int8)v131)
                               : 0LL;
                        if ( v115 == *v129 )
                        {
                          if ( v156 > 0x40 )
                          {
                            v134 = *(_QWORD *)v160;
                          }
                          else
                          {
                            v134 = 0;
                            if ( v156 )
                              v134 = v148;
                          }
                          if ( v133 == v134 )
                          {
LABEL_306:
                            v108 = v155;
                            goto LABEL_251;
                          }
                        }
                      }
                    }
                  }
                  else if ( sub_B448F0(v116) )
                  {
                    goto LABEL_288;
                  }
                  v115 = v128;
                  if ( !v128 )
                    goto LABEL_251;
LABEL_263:
                  v118 = v116;
                  goto LABEL_264;
                }
                if ( v117 > 0x1Cu )
                {
                  if ( v117 == 42 )
                  {
                    v128 = *(_QWORD *)(v110 + v140);
                    goto LABEL_286;
                  }
                  goto LABEL_263;
                }
                v118 = 0;
LABEL_264:
                if ( *(_BYTE *)v115 != 42 )
                  goto LABEL_251;
                if ( v178 == 69 )
                {
                  if ( !sub_B44900(v115) )
                    goto LABEL_270;
                }
                else if ( !sub_B448F0(v115) )
                {
                  goto LABEL_270;
                }
                if ( (*(_BYTE *)(v115 + 7) & 0x40) != 0 )
                  v119 = *(_QWORD **)(v115 - 8);
                else
                  v119 = (_QWORD *)(v115 - 32LL * (*(_DWORD *)(v115 + 4) & 0x7FFFFFF));
                v120 = v119[4];
                if ( *(_BYTE *)v120 == 17 )
                {
                  v135 = *(_DWORD *)(v120 + 32);
                  v136 = *(__int64 **)(v120 + 24);
                  if ( v135 > 0x40 )
                    v137 = *v136;
                  else
                    v137 = v135
                         ? (__int64)((_QWORD)v136 << (64 - (unsigned __int8)v135)) >> (64 - (unsigned __int8)v135)
                         : 0LL;
                  if ( v116 == *v119 )
                  {
                    if ( v156 > 0x40 )
                    {
                      v138 = *(_QWORD *)v160;
                    }
                    else
                    {
                      v138 = 0;
                      if ( v156 )
                        v138 = v148;
                    }
                    if ( !(v137 + v138) )
                      goto LABEL_306;
                  }
                }
LABEL_270:
                if ( v118 && *(_BYTE *)v118 == 42 )
                {
                  if ( v178 == 69 )
                  {
                    if ( !sub_B44900(v115) || !sub_B44900(v118) )
                    {
LABEL_330:
                      v108 = 0;
                      goto LABEL_251;
                    }
                  }
                  else if ( !sub_B448F0(v115) || !sub_B448F0(v118) )
                  {
                    goto LABEL_330;
                  }
                  if ( (*(_BYTE *)(v115 + 7) & 0x40) != 0 )
                    v121 = *(_QWORD **)(v115 - 8);
                  else
                    v121 = (_QWORD *)(v115 - 32LL * (*(_DWORD *)(v115 + 4) & 0x7FFFFFF));
                  v122 = v121[4];
                  if ( *(_BYTE *)v122 == 17 )
                  {
                    v123 = (*(_BYTE *)(v118 + 7) & 0x40) != 0
                         ? *(_QWORD *)(v118 - 8)
                         : v118 - 32LL * (*(_DWORD *)(v118 + 4) & 0x7FFFFFF);
                    if ( **(_BYTE **)(v123 + 32) == 17 && *v121 == *(_QWORD *)v123 )
                    {
                      sub_2AF7540(*(_QWORD *)(v122 + 24), *(_DWORD *)(v122 + 32));
                      sub_2AF7540(*(_QWORD *)(v124 + 24), *(_DWORD *)(v124 + 32));
                      v125 = sub_2AF7540((__int64)v160, v156);
                      v108 = v126 - v127 == v125;
                    }
                  }
                }
LABEL_251:
                v174 += 4;
              }
              while ( v174 != 8 );
              v169 += 4;
              if ( v169 == 8 )
              {
                v113 = v108;
                v5 = v139;
                v173 = sub_BCB060(v164);
                if ( v113 )
                {
LABEL_229:
                  v194 = sub_DD8400(*(_QWORD *)(v5 + 32), v191);
                  v99 = sub_DD8400(*(_QWORD *)(v5 + 32), (__int64)v72);
                  v100 = *(__int64 **)(v5 + 32);
                  v101 = v99;
                  sub_C44740((__int64)&v210, &v205, v173);
                  v102 = sub_DA26C0(v100, (__int64)&v210);
                  sub_969240((__int64 *)&v210);
                  v103 = *(__int64 **)(v5 + 32);
                  v210 = (unsigned __int64)&v212;
                  v213 = v102;
                  v212 = v194;
                  v211 = 0x200000002LL;
                  v104 = sub_DC7EB0(v103, (__int64)&v210, 0, 0);
                  if ( (__int64 **)v210 != &v212 )
                    _libc_free(v210);
                  result = v101 == v104;
                  goto LABEL_155;
                }
LABEL_226:
                sub_9878D0((__int64)&v210, v173);
                sub_9AC1B0(
                  v191,
                  &v210,
                  *(_QWORD *)(v5 + 48),
                  0,
                  *(_QWORD *)(v5 + 16),
                  (__int64)v72,
                  *(_QWORD *)(v5 + 24),
                  1);
                sub_C449B0((__int64)&v207, (const void **)&v210, v206);
                if ( v178 == 69 )
                {
                  v105 = ~(1LL << ((unsigned __int8)v173 - 1));
                  if ( (unsigned int)v208 > 0x40 )
                    *(_QWORD *)(v207 + 8LL * ((v173 - 1) >> 6)) &= v105;
                  else
                    v207 &= v105;
                }
                if ( (int)sub_C49970((__int64)&v207, (unsigned __int64 *)&v205) >= 0 )
                {
                  sub_969240((__int64 *)&v207);
                  sub_969240((__int64 *)&v212);
                  sub_969240((__int64 *)&v210);
                  goto LABEL_229;
                }
                sub_969240((__int64 *)&v207);
                sub_969240((__int64 *)&v212);
                sub_969240((__int64 *)&v210);
                result = 0;
LABEL_155:
                if ( v206 <= 0x40 || (v48 = v205) == 0 )
                {
LABEL_51:
                  if ( v204 > 0x40 && v203 )
                  {
                    v186 = result;
                    j_j___libc_free_0_0(v203);
                    result = v186;
                  }
LABEL_54:
                  if ( v202 > 0x40 && v201 )
                  {
                    v187 = result;
                    j_j___libc_free_0_0((unsigned __int64)v201);
                    result = v187;
                  }
                  if ( v200 > 0x40 )
                  {
LABEL_38:
                    if ( v199 )
                    {
                      v185 = result;
                      j_j___libc_free_0_0(v199);
                      result = v185;
                    }
                  }
                  goto LABEL_4;
                }
LABEL_100:
                v189 = result;
                j_j___libc_free_0_0((unsigned __int64)v48);
                result = v189;
                goto LABEL_51;
              }
            }
          }
          goto LABEL_50;
        }
      }
      v181 = *(_QWORD *)(v5 + 48);
      v192 = v55;
      v95 = sub_BCBAE0(v63, *v30, v59);
      v62 = v181;
      v55 = v192;
      v64 = v95;
      goto LABEL_147;
    }
    v205 = (char *)a1;
    v18 = sub_2AFD010((__int64 *)&v205, (__int64)v170);
    v19 = sub_2AFD010((__int64 *)&v205, (__int64)v182);
    v207 = (unsigned __int64)v18;
    v176 = v19;
    v209 = v202;
    if ( v202 > 0x40 )
      sub_C43780((__int64)&v208, (const void **)&v201);
    else
      v208 = (unsigned __int64)v201;
    v20 = *(_QWORD **)(a1 + 288);
    v163 = (_QWORD *)(a1 + 280);
    if ( v20 )
    {
      v21 = (_QWORD *)(a1 + 280);
      v22 = v207;
      do
      {
        if ( v20[4] < v22 || v20[4] == v22 && (int)sub_C49970((__int64)(v20 + 5), &v208) < 0 )
        {
          v20 = (_QWORD *)v20[3];
        }
        else
        {
          v21 = v20;
          v20 = (_QWORD *)v20[2];
        }
      }
      while ( v20 );
      v49 = v22;
      v5 = a1;
      if ( v163 != v21 && v49 >= v21[4] && (v49 != v21[4] || (int)sub_C49970((__int64)&v208, v21 + 5) >= 0) )
      {
        v25 = (__int64 *)v21[7];
LABEL_127:
        if ( v209 > 0x40 && v208 )
          j_j___libc_free_0_0(v208);
        goto LABEL_43;
      }
    }
    v50 = *(__int64 **)(v5 + 32);
    v213 = sub_DA26C0(v50, (__int64)&v201);
    v210 = (unsigned __int64)&v212;
    v212 = v18;
    v211 = 0x200000002LL;
    v25 = sub_DC7EB0(v50, (__int64)&v210, 0, 0);
    if ( (__int64 **)v210 != &v212 )
      _libc_free(v210);
    v51 = *(_QWORD **)(v5 + 288);
    if ( v51 )
    {
      v157 = v5;
      v52 = (_QWORD *)(a1 + 280);
      v53 = v207;
      do
      {
        if ( v51[4] < v53 || v51[4] == v53 && (int)sub_C49970((__int64)(v51 + 5), &v208) < 0 )
        {
          v51 = (_QWORD *)v51[3];
        }
        else
        {
          v52 = v51;
          v51 = (_QWORD *)v51[2];
        }
      }
      while ( v51 );
      v54 = v52;
      v5 = v157;
      if ( v163 != v54 && v53 >= v54[4] )
      {
        if ( v53 != v54[4] || (v162 = v54, v89 = sub_C49970((__int64)&v208, v54 + 5), v54 = v162, v89 >= 0) )
        {
LABEL_135:
          v54[7] = v25;
          goto LABEL_127;
        }
      }
    }
    else
    {
      v54 = (_QWORD *)(a1 + 280);
    }
    v158 = v54;
    v74 = sub_22077B0(0x40u);
    v75 = v158;
    v76 = v74;
    *(_QWORD *)(v74 + 32) = v207;
    v161 = (unsigned __int64 *)(v74 + 40);
    v77 = v209 <= 0x40;
    *(_DWORD *)(v74 + 48) = v209;
    if ( v77 )
    {
      *(_QWORD *)(v74 + 40) = v208;
    }
    else
    {
      sub_C43780((__int64)v161, (const void **)&v208);
      v75 = v158;
    }
    *(_QWORD *)(v76 + 56) = 0;
    if ( v163 == v75 )
    {
      if ( !*(_QWORD *)(v5 + 312)
        || (v75 = *(_QWORD **)(v5 + 304), v87 = *(_QWORD *)(v76 + 32), v75[4] >= v87)
        && (v75[4] != v87
         || (v152 = *(_QWORD **)(v5 + 304), v88 = sub_C49970((__int64)(v75 + 5), v161), v75 = v152, v88 >= 0)) )
      {
LABEL_170:
        v81 = sub_2AF7A40(v5 + 272, (unsigned __int64 *)(v76 + 32));
        if ( v75 )
          goto LABEL_171;
        v75 = v81;
LABEL_182:
        if ( *(_DWORD *)(v76 + 48) > 0x40u )
        {
          v86 = *(_QWORD *)(v76 + 40);
          if ( v86 )
          {
            v165 = v75;
            j_j___libc_free_0_0(v86);
            v75 = v165;
          }
        }
        v166 = v75;
        j_j___libc_free_0(v76);
        v54 = v166;
        goto LABEL_135;
      }
LABEL_190:
      v81 = 0;
      goto LABEL_171;
    }
    v78 = *(_QWORD *)(v76 + 32);
    if ( v78 >= v75[4] )
    {
      if ( v78 != v75[4] )
      {
        if ( v78 <= v75[4] )
          goto LABEL_182;
        goto LABEL_167;
      }
      v142 = *(_QWORD *)(v76 + 32);
      v146 = v75;
      v153 = (__int64)(v75 + 5);
      v93 = sub_C49970((__int64)v161, v75 + 5);
      v75 = v146;
      v78 = v142;
      if ( v93 >= 0 )
      {
        v94 = sub_C49970(v153, v161);
        v75 = v146;
        v78 = v142;
        if ( v94 >= 0 )
          goto LABEL_182;
LABEL_167:
        v143 = v78;
        if ( v75 != *(_QWORD **)(v5 + 304) )
        {
          v149 = v75;
          v79 = sub_220EEE0((__int64)v75);
          v75 = v149;
          v80 = (_QWORD *)v79;
          if ( v143 < *(_QWORD *)(v79 + 32)
            || v143 == *(_QWORD *)(v79 + 32)
            && (v147 = v149,
                v154 = v79,
                v96 = sub_C49970((__int64)v161, (unsigned __int64 *)(v79 + 40)),
                v80 = (_QWORD *)v154,
                v75 = v147,
                v96 < 0) )
          {
            v81 = (_QWORD *)v75[3];
            if ( v81 )
            {
              v75 = v80;
              goto LABEL_179;
            }
            goto LABEL_171;
          }
          goto LABEL_170;
        }
        goto LABEL_190;
      }
    }
    v81 = *(_QWORD **)(v5 + 296);
    v144 = v78;
    if ( v75 == v81 )
      goto LABEL_171;
    v150 = v75;
    v83 = sub_220EF80((__int64)v75);
    v75 = v150;
    v84 = (_QWORD *)v83;
    if ( v144 > *(_QWORD *)(v83 + 32)
      || v144 == *(_QWORD *)(v83 + 32)
      && (v145 = v150, v151 = v83, v85 = sub_C49970(v83 + 40, v161), v84 = (_QWORD *)v151, v75 = v145, v85 < 0) )
    {
      v81 = (_QWORD *)v84[3];
      if ( v81 )
      {
LABEL_179:
        v81 = v75;
        goto LABEL_171;
      }
      v75 = v84;
LABEL_171:
      v82 = 1;
      if ( v163 != v75 && !v81 )
      {
        v90 = v75[4];
        if ( *(_QWORD *)(v76 + 32) >= v90 )
        {
          v82 = 0;
          if ( *(_QWORD *)(v76 + 32) == v90 )
          {
            v159 = v75;
            v91 = sub_C49970((__int64)v161, v75 + 5);
            v75 = v159;
            v82 = v91 < 0;
          }
        }
      }
      sub_220F040(v82, v76, v75, v163);
      v54 = (_QWORD *)v76;
      ++*(_QWORD *)(v5 + 312);
      goto LABEL_135;
    }
    goto LABEL_170;
  }
  if ( (unsigned int)v211 > 0x40 )
  {
    result = sub_C43C50((__int64)&v199, (const void **)a4);
    goto LABEL_38;
  }
  result = *(_QWORD *)a4 == v210;
LABEL_4:
  if ( v198 > 0x40 && v197 )
  {
    v183 = result;
    j_j___libc_free_0_0((unsigned __int64)v197);
    result = v183;
  }
  if ( v196 > 0x40 )
  {
    if ( v195 )
    {
      v184 = result;
      j_j___libc_free_0_0((unsigned __int64)v195);
      return v184;
    }
  }
  return result;
}
