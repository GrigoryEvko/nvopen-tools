// Function: sub_1B0FB90
// Address: 0x1b0fb90
//
void __fastcall sub_1B0FB90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 a18,
        char a19)
{
  __int64 v21; // r13
  __int64 v22; // rsi
  __int64 v23; // rbx
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // r13
  _QWORD *v28; // r12
  __int64 v29; // rcx
  __int64 v30; // rbx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rdx
  __int64 v34; // rdx
  int v35; // ecx
  __int64 v36; // rcx
  __int64 *v37; // rdx
  __int64 v38; // rsi
  unsigned __int64 v39; // rcx
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rcx
  char v43; // di
  __int64 v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned __int64 *v49; // rbx
  int v50; // eax
  __int64 v51; // rax
  int v52; // ecx
  _QWORD *v53; // rcx
  unsigned __int64 **v54; // rax
  unsigned __int64 v55; // rcx
  unsigned __int64 v56; // rcx
  __int64 v57; // rdx
  _QWORD *v58; // rcx
  char v59; // cl
  unsigned int v60; // r10d
  __int64 v61; // r11
  __int64 v62; // rax
  _QWORD *v63; // rdx
  _QWORD *v64; // r12
  __int64 v65; // rax
  unsigned __int64 v66; // rdi
  __int64 i; // rax
  __int64 v68; // r14
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // r15
  __int64 v72; // rdx
  __int64 v73; // rax
  __int64 v74; // r8
  __int64 v75; // rsi
  __int64 v76; // r9
  __int64 v77; // r14
  char v78; // di
  __int64 v79; // rsi
  __int64 v80; // rdx
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rcx
  int v86; // eax
  __int64 v87; // rax
  int v88; // edx
  __int64 v89; // rdx
  __int64 *v90; // rax
  __int64 v91; // rsi
  unsigned __int64 v92; // rdx
  __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 v95; // rcx
  char v96; // di
  __int64 v97; // rsi
  __int64 v98; // rdx
  __int64 v99; // rax
  __int64 v100; // rcx
  __int64 v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rcx
  int v104; // eax
  __int64 v105; // rax
  int v106; // edx
  __int64 v107; // rdx
  __int64 *v108; // rax
  __int64 v109; // rsi
  unsigned __int64 v110; // rdx
  __int64 v111; // rdx
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // rsi
  char v115; // r8
  unsigned int v116; // edi
  __int64 v117; // rdx
  __int64 v118; // rax
  __int64 v119; // rcx
  __int64 v120; // rax
  __int64 v121; // rsi
  __int64 *v122; // rsi
  __int64 v123; // rdx
  unsigned __int64 v124; // rax
  __int64 v125; // rax
  __int64 v126; // rax
  __int64 v127; // rax
  __int64 v128; // rcx
  unsigned __int8 *v129; // rsi
  double v130; // xmm4_8
  double v131; // xmm5_8
  __int64 v132; // r14
  int v133; // r8d
  int v134; // r9d
  __int64 v135; // r13
  __int64 v136; // r15
  signed __int64 v137; // r15
  __int64 *v138; // r13
  _QWORD *v139; // rax
  __int64 *v140; // rsi
  int v141; // eax
  __int64 v142; // rdx
  _QWORD *v143; // rax
  _QWORD *v144; // r15
  unsigned __int64 *v145; // r12
  __int64 v146; // rax
  unsigned __int64 v147; // rcx
  __int64 v148; // rsi
  unsigned __int8 *v149; // rsi
  double v150; // xmm4_8
  double v151; // xmm5_8
  __int64 v152; // rsi
  __int64 v153; // rax
  int v154; // edi
  unsigned int v155; // ecx
  __int64 *v156; // rdx
  __int64 v157; // r8
  __int64 *v158; // rax
  __int64 v159; // r12
  unsigned int v160; // ecx
  __int64 *v161; // rdx
  __int64 v162; // r9
  __int64 v163; // r14
  __int64 v164; // rax
  _BYTE *v165; // rax
  __int64 v166; // rdx
  __int64 v167; // rcx
  int v168; // r8d
  int v169; // r9d
  _BYTE *v170; // rsi
  int v171; // edx
  int v172; // r8d
  int v173; // edx
  int v174; // r9d
  __int64 v177; // [rsp+10h] [rbp-140h]
  int v178; // [rsp+1Ch] [rbp-134h]
  unsigned __int64 v179; // [rsp+20h] [rbp-130h]
  __int64 v181; // [rsp+28h] [rbp-128h]
  __int64 v183; // [rsp+38h] [rbp-118h]
  __int64 v185; // [rsp+48h] [rbp-108h]
  unsigned int v186; // [rsp+48h] [rbp-108h]
  _QWORD *v187; // [rsp+48h] [rbp-108h]
  __int64 v188; // [rsp+50h] [rbp-100h]
  __int64 v189; // [rsp+50h] [rbp-100h]
  __int64 v190; // [rsp+50h] [rbp-100h]
  __int64 v191; // [rsp+50h] [rbp-100h]
  __int64 v192; // [rsp+58h] [rbp-F8h]
  __int64 v193; // [rsp+58h] [rbp-F8h]
  __int64 v194; // [rsp+58h] [rbp-F8h]
  __int64 v195; // [rsp+68h] [rbp-E8h] BYREF
  __int64 *v196; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v197; // [rsp+78h] [rbp-D8h]
  _BYTE v198[32]; // [rsp+80h] [rbp-D0h] BYREF
  char *v199; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v200; // [rsp+A8h] [rbp-A8h]
  __int64 v201[4]; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned __int8 **v202; // [rsp+D0h] [rbp-80h] BYREF
  const char *v203; // [rsp+D8h] [rbp-78h]
  unsigned __int64 *v204; // [rsp+E0h] [rbp-70h]
  __int64 v205; // [rsp+E8h] [rbp-68h]
  __int64 v206; // [rsp+F0h] [rbp-60h]
  int v207; // [rsp+F8h] [rbp-58h]
  __int64 v208; // [rsp+100h] [rbp-50h]
  __int64 v209; // [rsp+108h] [rbp-48h]

  v21 = a15;
  v22 = sub_13FCB50(a1);
  v23 = v22;
  v192 = sub_1B0E720(a16, v22)[2];
  v24 = sub_157F280(a3);
  v188 = v25;
  v185 = a1 + 56;
  if ( v24 != v25 )
  {
    v26 = v22;
    v27 = a6;
    while ( 1 )
    {
      v28 = sub_1648700(*(_QWORD *)(v24 + 8));
      v30 = sub_1599EF0(*(__int64 ***)v24);
      v33 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
      if ( (_DWORD)v33 == *(_DWORD *)(v24 + 56) )
      {
        sub_15F55D0(v24, v22, v33, v29, v31, v32);
        LODWORD(v33) = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
      }
      v34 = ((_DWORD)v33 + 1) & 0xFFFFFFF;
      v35 = v34 | *(_DWORD *)(v24 + 20) & 0xF0000000;
      *(_DWORD *)(v24 + 20) = v35;
      if ( (v35 & 0x40000000) != 0 )
        v36 = *(_QWORD *)(v24 - 8);
      else
        v36 = v24 - 24 * v34;
      v37 = (__int64 *)(v36 + 24LL * (unsigned int)(v34 - 1));
      if ( *v37 )
      {
        v38 = v37[1];
        v39 = v37[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v39 = v38;
        if ( v38 )
          *(_QWORD *)(v38 + 16) = *(_QWORD *)(v38 + 16) & 3LL | v39;
      }
      *v37 = v30;
      if ( v30 )
      {
        v40 = *(_QWORD *)(v30 + 8);
        v37[1] = v40;
        if ( v40 )
          *(_QWORD *)(v40 + 16) = (unsigned __int64)(v37 + 1) | *(_QWORD *)(v40 + 16) & 3LL;
        v37[2] = v37[2] & 3 | (v30 + 8);
        *(_QWORD *)(v30 + 8) = v37;
      }
      v41 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v24 + 23) & 0x40) != 0 )
        v42 = *(_QWORD *)(v24 - 8);
      else
        v42 = v24 - 24 * v41;
      *(_QWORD *)(v42 + 8LL * (unsigned int)(v41 - 1) + 24LL * *(unsigned int *)(v24 + 56) + 8) = a5;
      v43 = *(_BYTE *)(v24 + 23) & 0x40;
      v44 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
      if ( (*(_DWORD *)(v24 + 20) & 0xFFFFFFF) != 0 )
      {
        v45 = 24LL * *(unsigned int *)(v24 + 56) + 8;
        v46 = 0;
        while ( 1 )
        {
          v42 = v24 - 24LL * (unsigned int)v44;
          if ( v43 )
            v42 = *(_QWORD *)(v24 - 8);
          if ( v26 == *(_QWORD *)(v42 + v45) )
            break;
          ++v46;
          v45 += 8;
          if ( (_DWORD)v44 == (_DWORD)v46 )
            goto LABEL_131;
        }
        v47 = 24 * v46;
        if ( v43 )
        {
LABEL_23:
          v48 = *(_QWORD *)(v24 - 8);
          goto LABEL_24;
        }
      }
      else
      {
LABEL_131:
        v47 = 0x17FFFFFFE8LL;
        if ( v43 )
          goto LABEL_23;
      }
      v44 = (unsigned int)v44;
      v42 = 24LL * (unsigned int)v44;
      v48 = v24 - v42;
LABEL_24:
      v49 = *(unsigned __int64 **)(v48 + v47);
      if ( *((_BYTE *)v49 + 16) > 0x17u )
      {
        v44 = v49[5];
        if ( sub_1377F70(v185, v44) )
        {
          v44 = a16;
          v199 = (char *)v49;
          sub_1A51850((unsigned __int64 *)&v202, a16, (__int64 *)&v199);
          v49 = v204;
          LOBYTE(v42) = v204 + 1 != 0;
          if ( ((v204 != 0) & (unsigned __int8)v42) != 0 && v204 != (unsigned __int64 *)-16LL )
            sub_1649B30(&v202);
        }
      }
      v50 = *((_DWORD *)v28 + 5) & 0xFFFFFFF;
      if ( v50 == *((_DWORD *)v28 + 14) )
      {
        sub_15F55D0((__int64)v28, v44, v48, v42, v31, v32);
        v50 = *((_DWORD *)v28 + 5) & 0xFFFFFFF;
      }
      v51 = (v50 + 1) & 0xFFFFFFF;
      v22 = (unsigned int)(v51 - 1);
      v52 = v51 | *((_DWORD *)v28 + 5) & 0xF0000000;
      *((_DWORD *)v28 + 5) = v52;
      if ( (v52 & 0x40000000) != 0 )
        v53 = (_QWORD *)*(v28 - 1);
      else
        v53 = &v28[-3 * v51];
      v54 = (unsigned __int64 **)&v53[3 * (unsigned int)v22];
      if ( *v54 )
      {
        v22 = (__int64)v54[1];
        v55 = (unsigned __int64)v54[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v55 = v22;
        if ( v22 )
          *(_QWORD *)(v22 + 16) = *(_QWORD *)(v22 + 16) & 3LL | v55;
      }
      *v54 = v49;
      if ( v49 )
      {
        v56 = v49[1];
        v54[1] = (unsigned __int64 *)v56;
        if ( v56 )
        {
          v22 = (unsigned __int64)(v54 + 1) | *(_QWORD *)(v56 + 16) & 3LL;
          *(_QWORD *)(v56 + 16) = v22;
        }
        v54[2] = (unsigned __int64 *)((unsigned __int64)v54[2] & 3 | (unsigned __int64)(v49 + 1));
        v49[1] = (unsigned __int64)v54;
      }
      v57 = *((_DWORD *)v28 + 5) & 0xFFFFFFF;
      if ( (*((_BYTE *)v28 + 23) & 0x40) != 0 )
        v58 = (_QWORD *)*(v28 - 1);
      else
        v58 = &v28[-3 * v57];
      v58[3 * *((unsigned int *)v28 + 14) + 1 + (unsigned int)(v57 - 1)] = v192;
      v59 = *((_BYTE *)v28 + 23) & 0x40;
      v60 = *((_DWORD *)v28 + 5) & 0xFFFFFFF;
      if ( v60 )
      {
        v22 = (__int64)&v28[-3 * v60];
        v61 = 24LL * *((unsigned int *)v28 + 14);
        v62 = v61 + 8;
        while ( 1 )
        {
          v63 = &v28[-3 * v60];
          if ( v59 )
            v63 = (_QWORD *)*(v28 - 1);
          if ( v27 == *(_QWORD *)((char *)v63 + v62) )
            break;
          v62 += 8;
          if ( v62 == v61 + 8LL * (v60 - 1) + 16 )
          {
            v62 = v61 + 0x800000000LL;
            break;
          }
        }
      }
      else
      {
        v62 = 24LL * *((unsigned int *)v28 + 14) + 0x800000000LL;
      }
      if ( v59 )
        v64 = (_QWORD *)*(v28 - 1);
      else
        v64 = &v28[-3 * v60];
      *(_QWORD *)((char *)v64 + v62) = a3;
      v65 = *(_QWORD *)(v24 + 32);
      if ( !v65 )
        BUG();
      v24 = 0;
      if ( *(_BYTE *)(v65 - 8) == 77 )
        v24 = v65 - 24;
      if ( v188 == v24 )
      {
        a6 = v27;
        v21 = a15;
        v23 = v26;
        break;
      }
    }
  }
  v66 = sub_157EBA0(v23);
  if ( v66 )
  {
    v178 = sub_15F4D60(v66);
    v179 = sub_157EBA0(v23);
    if ( v178 )
    {
      v186 = 0;
      v177 = a1 + 56;
      for ( i = sub_15F4DF0(v179, 0); ; i = sub_15F4DF0(v179, v186) )
      {
        v68 = i;
        if ( sub_1377F70(v177, i) )
        {
          v69 = sub_157F280(v68);
          v181 = v70;
          v71 = v69;
          if ( v70 != v69 )
            break;
        }
LABEL_55:
        if ( v178 == ++v186 )
          goto LABEL_142;
      }
      while ( 1 )
      {
        v189 = sub_157ED20(a3);
        v199 = (char *)sub_1649960(v71);
        v200 = v72;
        v202 = (unsigned __int8 **)&v199;
        LOWORD(v204) = 773;
        v203 = ".unr";
        v193 = *(_QWORD *)v71;
        v73 = sub_1648B60(64);
        v75 = v193;
        v76 = v189;
        v77 = v73;
        if ( v73 )
        {
          v194 = v73;
          sub_15F1EA0(v73, v75, 53, 0, 0, v189);
          *(_DWORD *)(v77 + 56) = 2;
          sub_164B780(v77, (__int64 *)&v202);
          sub_1648880(v77, *(_DWORD *)(v77 + 56), 1);
        }
        else
        {
          v194 = 0;
        }
        v78 = *(_BYTE *)(v71 + 23) & 0x40;
        v79 = *(_DWORD *)(v71 + 20) & 0xFFFFFFF;
        if ( (*(_DWORD *)(v71 + 20) & 0xFFFFFFF) != 0 )
        {
          v74 = v71 - 24LL * (unsigned int)v79;
          v80 = 24LL * *(unsigned int *)(v71 + 56) + 8;
          v81 = 0;
          while ( 1 )
          {
            v82 = v71 - 24LL * (unsigned int)v79;
            if ( v78 )
              v82 = *(_QWORD *)(v71 - 8);
            if ( v21 == *(_QWORD *)(v82 + v80) )
              break;
            ++v81;
            v80 += 8;
            if ( (_DWORD)v79 == (_DWORD)v81 )
              goto LABEL_125;
          }
          v83 = 24 * v81;
          if ( v78 )
          {
LABEL_68:
            v84 = *(_QWORD *)(v71 - 8);
            goto LABEL_69;
          }
        }
        else
        {
LABEL_125:
          v83 = 0x17FFFFFFE8LL;
          if ( v78 )
            goto LABEL_68;
        }
        v79 = (unsigned int)v79;
        v84 = v71 - 24LL * (unsigned int)v79;
LABEL_69:
        v85 = *(_QWORD *)(v84 + v83);
        v86 = *(_DWORD *)(v77 + 20) & 0xFFFFFFF;
        if ( v86 == *(_DWORD *)(v77 + 56) )
        {
          v191 = v85;
          sub_15F55D0(v77, v79, v84, v85, v74, v76);
          v85 = v191;
          v86 = *(_DWORD *)(v77 + 20) & 0xFFFFFFF;
        }
        v87 = (v86 + 1) & 0xFFFFFFF;
        v88 = v87 | *(_DWORD *)(v77 + 20) & 0xF0000000;
        *(_DWORD *)(v77 + 20) = v88;
        if ( (v88 & 0x40000000) != 0 )
          v89 = *(_QWORD *)(v77 - 8);
        else
          v89 = v194 - 24 * v87;
        v90 = (__int64 *)(v89 + 24LL * (unsigned int)(v87 - 1));
        if ( *v90 )
        {
          v91 = v90[1];
          v92 = v90[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v92 = v91;
          if ( v91 )
            *(_QWORD *)(v91 + 16) = *(_QWORD *)(v91 + 16) & 3LL | v92;
        }
        *v90 = v85;
        if ( v85 )
        {
          v93 = *(_QWORD *)(v85 + 8);
          v90[1] = v93;
          if ( v93 )
          {
            v74 = (__int64)(v90 + 1);
            *(_QWORD *)(v93 + 16) = (unsigned __int64)(v90 + 1) | *(_QWORD *)(v93 + 16) & 3LL;
          }
          v90[2] = (v85 + 8) | v90[2] & 3;
          *(_QWORD *)(v85 + 8) = v90;
        }
        v94 = *(_DWORD *)(v77 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v77 + 23) & 0x40) != 0 )
          v95 = *(_QWORD *)(v77 - 8);
        else
          v95 = v194 - 24 * v94;
        *(_QWORD *)(v95 + 8LL * (unsigned int)(v94 - 1) + 24LL * *(unsigned int *)(v77 + 56) + 8) = a5;
        v96 = *(_BYTE *)(v71 + 23) & 0x40;
        v97 = *(_DWORD *)(v71 + 20) & 0xFFFFFFF;
        if ( (*(_DWORD *)(v71 + 20) & 0xFFFFFFF) != 0 )
        {
          v74 = v71 - 24LL * (unsigned int)v97;
          v98 = 24LL * *(unsigned int *)(v71 + 56) + 8;
          v99 = 0;
          while ( 1 )
          {
            v100 = v71 - 24LL * (unsigned int)v97;
            if ( v96 )
              v100 = *(_QWORD *)(v71 - 8);
            if ( v23 == *(_QWORD *)(v100 + v98) )
              break;
            ++v99;
            v98 += 8;
            if ( (_DWORD)v97 == (_DWORD)v99 )
              goto LABEL_123;
          }
          v101 = 24 * v99;
          if ( v96 )
          {
LABEL_89:
            v102 = *(_QWORD *)(v71 - 8);
            goto LABEL_90;
          }
        }
        else
        {
LABEL_123:
          v101 = 0x17FFFFFFE8LL;
          if ( v96 )
            goto LABEL_89;
        }
        v97 = (unsigned int)v97;
        v102 = v71 - 24LL * (unsigned int)v97;
LABEL_90:
        v103 = *(_QWORD *)(v102 + v101);
        v104 = *(_DWORD *)(v77 + 20) & 0xFFFFFFF;
        if ( v104 == *(_DWORD *)(v77 + 56) )
        {
          v190 = v103;
          sub_15F55D0(v77, v97, v102, v103, v74, v76);
          v103 = v190;
          v104 = *(_DWORD *)(v77 + 20) & 0xFFFFFFF;
        }
        v105 = (v104 + 1) & 0xFFFFFFF;
        v106 = v105 | *(_DWORD *)(v77 + 20) & 0xF0000000;
        *(_DWORD *)(v77 + 20) = v106;
        if ( (v106 & 0x40000000) != 0 )
          v107 = *(_QWORD *)(v77 - 8);
        else
          v107 = v194 - 24 * v105;
        v108 = (__int64 *)(v107 + 24LL * (unsigned int)(v105 - 1));
        if ( *v108 )
        {
          v109 = v108[1];
          v110 = v108[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v110 = v109;
          if ( v109 )
            *(_QWORD *)(v109 + 16) = *(_QWORD *)(v109 + 16) & 3LL | v110;
        }
        *v108 = v103;
        if ( v103 )
        {
          v111 = *(_QWORD *)(v103 + 8);
          v108[1] = v111;
          if ( v111 )
            *(_QWORD *)(v111 + 16) = (unsigned __int64)(v108 + 1) | *(_QWORD *)(v111 + 16) & 3LL;
          v108[2] = (v103 + 8) | v108[2] & 3;
          *(_QWORD *)(v103 + 8) = v108;
        }
        v112 = *(_DWORD *)(v77 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v77 + 23) & 0x40) != 0 )
          v113 = *(_QWORD *)(v77 - 8);
        else
          v113 = v194 - 24 * v112;
        *(_QWORD *)(v113 + 8LL * (unsigned int)(v112 - 1) + 24LL * *(unsigned int *)(v77 + 56) + 8) = v23;
        v114 = sub_1B0E720(a16, v71)[2];
        v115 = *(_BYTE *)(v114 + 23) & 0x40;
        v116 = *(_DWORD *)(v114 + 20) & 0xFFFFFFF;
        if ( v116 )
        {
          v117 = 24LL * *(unsigned int *)(v114 + 56) + 8;
          v118 = 0;
          while ( 1 )
          {
            v119 = v114 - 24LL * v116;
            if ( v115 )
              v119 = *(_QWORD *)(v114 - 8);
            if ( a6 == *(_QWORD *)(v119 + v117) )
              break;
            ++v118;
            v117 += 8;
            if ( v116 == (_DWORD)v118 )
              goto LABEL_121;
          }
          v120 = 24 * v118;
          if ( v115 )
          {
LABEL_110:
            v121 = *(_QWORD *)(v114 - 8);
            goto LABEL_111;
          }
        }
        else
        {
LABEL_121:
          v120 = 0x17FFFFFFE8LL;
          if ( v115 )
            goto LABEL_110;
        }
        v121 = v114 - 24LL * v116;
LABEL_111:
        v122 = (__int64 *)(v120 + v121);
        if ( *v122 )
        {
          v123 = v122[1];
          v124 = v122[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v124 = v123;
          if ( v123 )
            *(_QWORD *)(v123 + 16) = *(_QWORD *)(v123 + 16) & 3LL | v124;
        }
        *v122 = v77;
        v125 = *(_QWORD *)(v77 + 8);
        v122[1] = v125;
        if ( v125 )
          *(_QWORD *)(v125 + 16) = (unsigned __int64)(v122 + 1) | *(_QWORD *)(v125 + 16) & 3LL;
        v122[2] = (v77 + 8) | v122[2] & 3;
        *(_QWORD *)(v77 + 8) = v122;
        v126 = *(_QWORD *)(v71 + 32);
        if ( !v126 )
          BUG();
        v71 = 0;
        if ( *(_BYTE *)(v126 - 8) == 77 )
          v71 = v126 - 24;
        if ( v181 == v71 )
          goto LABEL_55;
      }
    }
  }
LABEL_142:
  v187 = (_QWORD *)sub_157EBA0(a3);
  v127 = sub_16498A0((__int64)v187);
  v202 = 0;
  v205 = v127;
  v206 = 0;
  v207 = 0;
  v208 = 0;
  v209 = 0;
  v203 = (const char *)v187[5];
  v204 = v187 + 3;
  v129 = (unsigned __int8 *)v187[6];
  v199 = (char *)v129;
  if ( v129 )
  {
    sub_1623A60((__int64)&v199, (__int64)v129, 2);
    if ( v202 )
      sub_161E7C0((__int64)&v202, (__int64)v202);
    v202 = (unsigned __int8 **)v199;
    if ( v199 )
      sub_1623210((__int64)&v199, (unsigned __int8 *)v199, (__int64)&v202);
  }
  v199 = "lcmp.mod";
  LOWORD(v201[0]) = 259;
  v183 = sub_1B0E120((__int64 *)&v202, a2, (__int64 *)&v199, v128);
  v132 = *(_QWORD *)(a4 + 8);
  if ( v132 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v132) + 16) - 25) > 9u )
    {
      v132 = *(_QWORD *)(v132 + 8);
      if ( !v132 )
        goto LABEL_194;
    }
    v135 = v132;
    v136 = 0;
    v196 = (__int64 *)v198;
    v197 = 0x400000000LL;
    while ( 1 )
    {
      v135 = *(_QWORD *)(v135 + 8);
      if ( !v135 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v135) + 16) - 25) <= 9u )
      {
        v135 = *(_QWORD *)(v135 + 8);
        ++v136;
        if ( !v135 )
          goto LABEL_153;
      }
    }
LABEL_153:
    v137 = v136 + 1;
    if ( v137 > 4 )
    {
      sub_16CD150((__int64)&v196, v198, v137, 8, v133, v134);
      v138 = &v196[(unsigned int)v197];
    }
    else
    {
      v138 = (__int64 *)v198;
    }
    v139 = sub_1648700(v132);
LABEL_157:
    if ( v138 )
      *v138 = v139[5];
    v132 = *(_QWORD *)(v132 + 8);
    if ( v132 )
    {
      do
      {
        v139 = sub_1648700(v132);
        if ( (unsigned __int8)(*((_BYTE *)v139 + 16) - 25) <= 9u )
        {
          ++v138;
          goto LABEL_157;
        }
        v132 = *(_QWORD *)(v132 + 8);
      }
      while ( v132 );
      v140 = v196;
      v141 = v197 + v137;
      v142 = (unsigned int)(v197 + v137);
    }
    else
    {
      v140 = v196;
      v141 = v197 + v137;
      v142 = (unsigned int)(v197 + v137);
    }
  }
  else
  {
LABEL_194:
    v142 = 0;
    v141 = 0;
    HIDWORD(v197) = 4;
    v196 = (__int64 *)v198;
    v140 = (__int64 *)v198;
  }
  LODWORD(v197) = v141;
  sub_1AAB350(a4, v140, v142, ".epilog-lcssa", a17, a18, a7, a8, a9, a10, v130, v131, a13, a14, a19);
  LOWORD(v201[0]) = 257;
  v143 = sub_1648A60(56, 3u);
  v144 = v143;
  if ( v143 )
    sub_15F83E0((__int64)v143, a6, a4, v183, 0);
  if ( v203 )
  {
    v145 = v204;
    sub_157E9D0((__int64)(v203 + 40), (__int64)v144);
    v146 = v144[3];
    v147 = *v145;
    v144[4] = v145;
    v147 &= 0xFFFFFFFFFFFFFFF8LL;
    v144[3] = v147 | v146 & 7;
    *(_QWORD *)(v147 + 8) = v144 + 3;
    *v145 = *v145 & 7 | (unsigned __int64)(v144 + 3);
  }
  sub_164B780((__int64)v144, (__int64 *)&v199);
  if ( v202 )
  {
    v195 = (__int64)v202;
    sub_1623A60((__int64)&v195, (__int64)v202, 2);
    v148 = v144[6];
    if ( v148 )
      sub_161E7C0((__int64)(v144 + 6), v148);
    v149 = (unsigned __int8 *)v195;
    v144[6] = v195;
    if ( v149 )
      sub_1623210((__int64)&v195, v149, (__int64)(v144 + 6));
  }
  sub_15F20C0(v187);
  if ( a17 )
  {
    v152 = *(_QWORD *)(a17 + 32);
    v153 = *(unsigned int *)(a17 + 48);
    if ( !(_DWORD)v153 )
      goto LABEL_214;
    v154 = v153 - 1;
    v155 = (v153 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v156 = (__int64 *)(v152 + 16LL * v155);
    v157 = *v156;
    if ( a3 == *v156 )
    {
LABEL_175:
      v158 = (__int64 *)(v152 + 16 * v153);
      if ( v156 != v158 )
      {
        v159 = v156[1];
        goto LABEL_177;
      }
    }
    else
    {
      v173 = 1;
      while ( v157 != -8 )
      {
        v174 = v173 + 1;
        v155 = v154 & (v173 + v155);
        v156 = (__int64 *)(v152 + 16LL * v155);
        v157 = *v156;
        if ( a3 == *v156 )
          goto LABEL_175;
        v173 = v174;
      }
      v158 = (__int64 *)(v152 + 16 * v153);
    }
    v159 = 0;
LABEL_177:
    v160 = v154 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v161 = (__int64 *)(v152 + 16LL * v160);
    v162 = *v161;
    if ( a4 == *v161 )
    {
LABEL_178:
      if ( v158 != v161 )
      {
        v163 = v161[1];
        *(_BYTE *)(a17 + 72) = 0;
        v164 = *(_QWORD *)(v163 + 8);
        if ( v164 != v159 )
        {
          v199 = (char *)v163;
          v165 = sub_1B0DF00(*(_QWORD **)(v164 + 24), *(_QWORD *)(v164 + 32), (__int64 *)&v199);
          sub_15CDF70(*(_QWORD *)(v163 + 8) + 24LL, v165);
          *(_QWORD *)(v163 + 8) = v159;
          v199 = (char *)v163;
          v170 = *(_BYTE **)(v159 + 32);
          if ( v170 == *(_BYTE **)(v159 + 40) )
          {
            sub_15CE310(v159 + 24, v170, &v199);
          }
          else
          {
            if ( v170 )
            {
              *(_QWORD *)v170 = v163;
              v170 = *(_BYTE **)(v159 + 32);
            }
            v170 += 8;
            *(_QWORD *)(v159 + 32) = v170;
          }
          if ( *(_DWORD *)(v163 + 16) != *(_DWORD *)(*(_QWORD *)(v163 + 8) + 16LL) + 1 )
            sub_1B0DDF0(v163, (__int64)v170, v166, v167, v168, v169);
        }
        goto LABEL_186;
      }
    }
    else
    {
      v171 = 1;
      while ( v162 != -8 )
      {
        v172 = v171 + 1;
        v160 = v154 & (v171 + v160);
        v161 = (__int64 *)(v152 + 16LL * v160);
        v162 = *v161;
        if ( a4 == *v161 )
          goto LABEL_178;
        v171 = v172;
      }
    }
LABEL_214:
    *(_BYTE *)(a17 + 72) = 0;
    BUG();
  }
LABEL_186:
  v199 = (char *)v201;
  v201[0] = v23;
  v200 = 0x400000001LL;
  sub_1AAB350(a3, v201, 1, ".loopexit", a17, a18, a7, a8, a9, a10, v150, v151, a13, a14, a19);
  if ( v199 != (char *)v201 )
    _libc_free((unsigned __int64)v199);
  if ( v196 != (__int64 *)v198 )
    _libc_free((unsigned __int64)v196);
  if ( v202 )
    sub_161E7C0((__int64)&v202, (__int64)v202);
}
