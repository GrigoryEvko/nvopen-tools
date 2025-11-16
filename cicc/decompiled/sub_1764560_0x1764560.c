// Function: sub_1764560
// Address: 0x1764560
//
__int64 __fastcall sub_1764560(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12)
{
  _BYTE *v14; // rdi
  unsigned __int8 v15; // al
  int v16; // r12d
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // r14
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  __int64 v23; // rax
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // rsi
  char v27; // dl
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  char v31; // dl
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r9
  __int64 v35; // rax
  unsigned int v36; // r8d
  unsigned int v39; // eax
  unsigned int v40; // r14d
  unsigned __int64 v41; // rcx
  unsigned int v42; // edx
  const char *v43; // rax
  bool v44; // al
  __int64 v45; // rcx
  int v46; // edi
  int v47; // eax
  __int64 v48; // rax
  double v49; // xmm4_8
  double v50; // xmm5_8
  unsigned int v51; // eax
  bool v52; // dl
  int v53; // eax
  __int64 v54; // r12
  unsigned int v55; // edx
  __int64 *v56; // r8
  bool v57; // al
  bool v58; // r14
  unsigned int v59; // edx
  __int64 *v60; // r12
  bool v61; // al
  bool v62; // r14
  __int64 v63; // r12
  _QWORD *v64; // rax
  __int64 v65; // rax
  __int64 v66; // r12
  _QWORD *v67; // rax
  unsigned int v68; // ebx
  __int64 *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rbx
  _QWORD *v72; // rax
  _QWORD *v73; // rax
  __int64 *v74; // rax
  __int64 v75; // rax
  __int64 *v76; // r14
  const char *v77; // rsi
  __int64 v78; // rsi
  unsigned __int8 *v79; // rsi
  const char *v80; // rax
  unsigned __int8 *v81; // rdx
  const char *v82; // rax
  unsigned __int8 *v83; // rdx
  unsigned __int8 *v84; // rax
  unsigned __int8 *v85; // rax
  unsigned __int8 *v86; // rax
  double v87; // xmm4_8
  double v88; // xmm5_8
  _QWORD *v89; // rax
  _QWORD *v90; // r12
  __int64 v91; // rax
  __int64 v92; // rdx
  unsigned __int64 v93; // rcx
  __int64 v94; // rcx
  unsigned int v95; // [rsp+8h] [rbp-138h]
  __int64 v96; // [rsp+8h] [rbp-138h]
  __int64 v97; // [rsp+8h] [rbp-138h]
  __int64 v98; // [rsp+8h] [rbp-138h]
  unsigned int v99; // [rsp+10h] [rbp-130h]
  const void **v100; // [rsp+10h] [rbp-130h]
  __int64 *v101; // [rsp+10h] [rbp-130h]
  unsigned int v102; // [rsp+10h] [rbp-130h]
  __int64 i; // [rsp+10h] [rbp-130h]
  unsigned int v104; // [rsp+10h] [rbp-130h]
  __int64 v105; // [rsp+10h] [rbp-130h]
  __int64 v106; // [rsp+18h] [rbp-128h]
  __int16 v107; // [rsp+18h] [rbp-128h]
  unsigned int v108; // [rsp+18h] [rbp-128h]
  unsigned int v109; // [rsp+18h] [rbp-128h]
  __int64 v110; // [rsp+20h] [rbp-120h]
  __int64 v111; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v112; // [rsp+20h] [rbp-120h]
  __int64 v113; // [rsp+28h] [rbp-118h]
  __int64 v114; // [rsp+28h] [rbp-118h]
  __int64 *v115; // [rsp+28h] [rbp-118h]
  __int64 v116; // [rsp+30h] [rbp-110h]
  __int64 v117; // [rsp+30h] [rbp-110h]
  unsigned int v118; // [rsp+30h] [rbp-110h]
  __int64 v119; // [rsp+38h] [rbp-108h]
  char v120; // [rsp+4Fh] [rbp-F1h] BYREF
  __int64 v121; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v122; // [rsp+58h] [rbp-E8h] BYREF
  __int64 *v123; // [rsp+60h] [rbp-E0h] BYREF
  unsigned int v124; // [rsp+68h] [rbp-D8h]
  __int64 *v125; // [rsp+70h] [rbp-D0h] BYREF
  unsigned int v126; // [rsp+78h] [rbp-C8h]
  __int16 v127; // [rsp+80h] [rbp-C0h]
  __int64 v128; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v129; // [rsp+98h] [rbp-A8h]
  __int64 v130; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned int v131; // [rsp+A8h] [rbp-98h]
  __int64 **v132; // [rsp+B0h] [rbp-90h] BYREF
  unsigned int v133; // [rsp+B8h] [rbp-88h]
  __int64 v134; // [rsp+C0h] [rbp-80h] BYREF
  unsigned int v135; // [rsp+C8h] [rbp-78h]
  __int64 *v136; // [rsp+D0h] [rbp-70h] BYREF
  unsigned __int8 *v137; // [rsp+D8h] [rbp-68h]
  __int64 *v138; // [rsp+E0h] [rbp-60h] BYREF
  unsigned int v139; // [rsp+E8h] [rbp-58h]
  const char *v140; // [rsp+F0h] [rbp-50h] BYREF
  char *v141; // [rsp+F8h] [rbp-48h]
  __int64 *v142; // [rsp+100h] [rbp-40h] BYREF
  unsigned int v143; // [rsp+108h] [rbp-38h]

  v14 = *(_BYTE **)(a2 - 24);
  v119 = *(_QWORD *)(a2 - 48);
  v15 = v14[16];
  v16 = *(_WORD *)(a2 + 18) & 0x7FFF;
  if ( v15 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 16 )
      return 0;
    if ( v15 > 0x10u )
      return 0;
    v23 = sub_15A1020(v14, a2, *(_QWORD *)v14, a12);
    if ( !v23 || *(_BYTE *)(v23 + 16) != 13 )
      return 0;
  }
  v121 = 0;
  v122 = 0;
  if ( v16 != 34 )
    goto LABEL_3;
  v140 = (const char *)&v121;
  v141 = (char *)&v122;
  v142 = &v128;
  v24 = *(_BYTE *)(v119 + 16);
  if ( v24 != 35 )
  {
    if ( v24 != 5 || *(_WORD *)(v119 + 18) != 11 )
      goto LABEL_3;
    v25 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
    v26 = *(_QWORD *)(v119 - 24 * v25);
    v27 = *(_BYTE *)(v26 + 16);
    if ( v27 == 35 )
    {
      if ( !*(_QWORD *)(v26 - 48) )
        goto LABEL_3;
      v121 = *(_QWORD *)(v26 - 48);
      if ( !*(_QWORD *)(v26 - 24) )
        goto LABEL_3;
      v122 = *(_QWORD *)(v26 - 24);
    }
    else
    {
      if ( v27 != 5 || *(_WORD *)(v26 + 18) != 11 || !(unsigned __int8)sub_17570E0((_QWORD **)&v140, v26) )
        goto LABEL_3;
      v25 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
    }
    v28 = *(_QWORD *)(v119 + 24 * (1 - v25));
    if ( *(_BYTE *)(v28 + 16) == 13 )
    {
      *v142 = v28;
      v29 = v128;
      goto LABEL_32;
    }
LABEL_3:
    v17 = *(_QWORD *)(a2 - 24);
    goto LABEL_4;
  }
  v30 = *(_QWORD *)(v119 - 48);
  v31 = *(_BYTE *)(v30 + 16);
  if ( v31 == 35 )
  {
    if ( !*(_QWORD *)(v30 - 48) )
      goto LABEL_3;
    v121 = *(_QWORD *)(v30 - 48);
    v33 = *(_QWORD *)(v30 - 24);
    if ( !v33 )
      goto LABEL_3;
  }
  else
  {
    if ( v31 != 5 )
      goto LABEL_3;
    if ( *(_WORD *)(v30 + 18) != 11 )
      goto LABEL_3;
    v32 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
    if ( !*(_QWORD *)(v30 - 24 * v32) )
      goto LABEL_3;
    v121 = *(_QWORD *)(v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF));
    v33 = *(_QWORD *)(v30 + 24 * (1 - v32));
    if ( !v33 )
      goto LABEL_3;
  }
  v122 = v33;
  v29 = *(_QWORD *)(v119 - 24);
  if ( *(_BYTE *)(v29 + 16) != 13 )
    goto LABEL_3;
  v128 = *(_QWORD *)(v119 - 24);
LABEL_32:
  v34 = *(_QWORD *)(a2 - 24);
  v113 = *(_QWORD *)(a2 - 48);
  v35 = *(_QWORD *)(v113 + 8);
  v17 = v34;
  if ( !v35 || *(_QWORD *)(v35 + 8) )
    goto LABEL_4;
  v36 = *(_DWORD *)(v29 + 32);
  if ( v36 > 0x40 )
  {
    v117 = *(_QWORD *)(a2 - 24);
    if ( (unsigned int)sub_16A5940(v29 + 24) != 1 )
      goto LABEL_4;
    v51 = sub_16A58A0(v29 + 24);
    v34 = v117;
    v36 = v51;
  }
  else
  {
    _RAX = *(_QWORD *)(v29 + 24);
    if ( !_RAX || (_RAX & (_RAX - 1)) != 0 )
      goto LABEL_4;
    __asm { tzcnt   rax, rax }
    if ( (unsigned int)_RAX <= v36 )
      v36 = _RAX;
  }
  if ( (v36 & 0xFFFFFFF7) != 7 && v36 != 31 || (v39 = *(_DWORD *)(v34 + 32), v40 = v36 + 1, v36 + 1 == v39) )
  {
LABEL_4:
    v18 = 0;
    if ( *(_BYTE *)(v17 + 16) != 13 )
      return v18;
    v19 = *(_QWORD *)(a2 + 40);
    v20 = sub_157F0B0(v19);
    if ( v20 )
    {
      v21 = sub_157EBA0(v20);
      if ( *(_BYTE *)(v21 + 16) == 26 && (*(_DWORD *)(v21 + 20) & 0xFFFFFFF) == 3 )
      {
        v45 = *(_QWORD *)(v21 - 72);
        if ( *(_BYTE *)(v45 + 16) == 75 && v119 == *(_QWORD *)(v45 - 48) )
        {
          v96 = *(_QWORD *)(v45 - 24);
          if ( *(_BYTE *)(v96 + 16) == 13 )
          {
            v111 = *(_QWORD *)(v21 - 24);
            if ( v111 != *(_QWORD *)(v21 - 48) )
            {
              v107 = *(_WORD *)(v45 + 18);
              v100 = (const void **)(v17 + 24);
              LODWORD(v137) = *(_DWORD *)(v17 + 32);
              if ( (unsigned int)v137 > 0x40 )
                sub_16A4FD0((__int64)&v136, v100);
              else
                v136 = *(__int64 **)(v17 + 24);
              sub_1589870((__int64)&v140, (__int64 *)&v136);
              sub_158AE10((__int64)&v128, v16, (__int64)&v140);
              if ( v143 > 0x40 && v142 )
                j_j___libc_free_0_0(v142);
              if ( (unsigned int)v141 > 0x40 && v140 )
                j_j___libc_free_0_0(v140);
              if ( (unsigned int)v137 > 0x40 && v136 )
                j_j___libc_free_0_0(v136);
              v46 = v107 & 0x7FFF;
              if ( v19 == v111 )
              {
                sub_158B890((__int64)&v132, v46, v96 + 24);
              }
              else
              {
                v47 = sub_15FF0F0(v46);
                sub_158B890((__int64)&v132, v47, v96 + 24);
              }
              sub_158BE00((__int64)&v136, (__int64)&v132, (__int64)&v128);
              sub_1590FF0((__int64)&v140, (__int64)&v132, (__int64)&v128);
              if ( sub_158A120((__int64)&v136) )
              {
                v48 = sub_159C540(*(__int64 **)(a1[1] + 24));
LABEL_71:
                v18 = sub_170E100(a1, a2, v48, a3, a4, a5, a6, v49, v50, a9, a10);
                goto LABEL_72;
              }
              if ( sub_158A120((__int64)&v140) )
              {
                v48 = sub_159C4F0(*(__int64 **)(a1[1] + 24));
                goto LABEL_71;
              }
              v52 = sub_1757FA0(v16, (__int64)v100, &v120);
              v53 = *(unsigned __int16 *)(a2 + 18);
              BYTE1(v53) &= ~0x80u;
              if ( (unsigned int)(v53 - 32) <= 1 )
              {
LABEL_132:
                v18 = 0;
LABEL_72:
                if ( v143 > 0x40 && v142 )
                  j_j___libc_free_0_0(v142);
                if ( (unsigned int)v141 > 0x40 && v140 )
                  j_j___libc_free_0_0(v140);
                if ( v139 > 0x40 && v138 )
                  j_j___libc_free_0_0(v138);
                if ( (unsigned int)v137 > 0x40 && v136 )
                  j_j___libc_free_0_0(v136);
                if ( v135 > 0x40 && v134 )
                  j_j___libc_free_0_0(v134);
                if ( v133 > 0x40 && v132 )
                  j_j___libc_free_0_0(v132);
                if ( v131 > 0x40 && v130 )
                  j_j___libc_free_0_0(v130);
                if ( v129 > 0x40 && v128 )
                  j_j___libc_free_0_0(v128);
                return v18;
              }
              if ( v52 )
              {
                v54 = *(_QWORD *)(a2 + 8);
                if ( v54 )
                {
                  while ( *((_BYTE *)sub_1648700(v54) + 16) != 26 )
                  {
                    v54 = *(_QWORD *)(v54 + 8);
                    if ( !v54 )
                      goto LABEL_110;
                  }
                  goto LABEL_132;
                }
              }
LABEL_110:
              v124 = (unsigned int)v137;
              if ( (unsigned int)v137 > 0x40 )
                sub_16A4FD0((__int64)&v123, (const void **)&v136);
              else
                v123 = v136;
              sub_16A7490((__int64)&v123, 1);
              v55 = v124;
              v56 = v123;
              v124 = 0;
              v126 = v55;
              v125 = v123;
              if ( v139 <= 0x40 )
              {
                v58 = v138 == v123;
              }
              else
              {
                v101 = v123;
                v108 = v55;
                v57 = sub_16A5220((__int64)&v138, (const void **)&v125);
                v56 = v101;
                v55 = v108;
                v58 = v57;
              }
              if ( v55 > 0x40 )
              {
                if ( v56 )
                {
                  j_j___libc_free_0_0(v56);
                  if ( v124 > 0x40 )
                  {
                    if ( v123 )
                      j_j___libc_free_0_0(v123);
                  }
                }
              }
              if ( v58 )
              {
                v65 = sub_159C0E0(*(__int64 **)(a1[1] + 24), (__int64)&v136);
                v127 = 257;
                v66 = v65;
                v67 = sub_1648A60(56, 2u);
                v18 = (__int64)v67;
                if ( v67 )
                  sub_17582E0((__int64)v67, 32, v119, v66, (__int64)&v125);
                goto LABEL_72;
              }
              v124 = (unsigned int)v141;
              if ( (unsigned int)v141 > 0x40 )
                sub_16A4FD0((__int64)&v123, (const void **)&v140);
              else
                v123 = (__int64 *)v140;
              sub_16A7490((__int64)&v123, 1);
              v59 = v124;
              v60 = v123;
              v124 = 0;
              v126 = v59;
              v125 = v123;
              if ( v143 <= 0x40 )
              {
                v62 = v142 == v123;
              }
              else
              {
                v109 = v59;
                v61 = sub_16A5220((__int64)&v142, (const void **)&v125);
                v59 = v109;
                v62 = v61;
              }
              if ( v59 > 0x40 )
              {
                if ( v60 )
                {
                  j_j___libc_free_0_0(v60);
                  if ( v124 > 0x40 )
                  {
                    if ( v123 )
                      j_j___libc_free_0_0(v123);
                  }
                }
              }
              if ( v62 )
              {
                v63 = sub_159C0E0(*(__int64 **)(a1[1] + 24), (__int64)&v140);
                v127 = 257;
                v64 = sub_1648A60(56, 2u);
                v18 = (__int64)v64;
                if ( v64 )
                  sub_17582E0((__int64)v64, 33, v119, v63, (__int64)&v125);
                goto LABEL_72;
              }
              sub_135E100((__int64 *)&v142);
              sub_135E100((__int64 *)&v140);
              sub_135E100((__int64 *)&v138);
              sub_135E100((__int64 *)&v136);
              sub_135E100(&v134);
              sub_135E100((__int64 *)&v132);
              sub_135E100(&v130);
              sub_135E100(&v128);
            }
          }
        }
      }
    }
    return 0;
  }
  LODWORD(v141) = *(_DWORD *)(v34 + 32);
  v106 = v122;
  v110 = v121;
  if ( v39 > 0x40 )
  {
    v97 = v34;
    v102 = v36;
    sub_16A4EF0((__int64)&v140, 0, 0);
    v36 = v102;
    v34 = v97;
    if ( v40 )
    {
      if ( v40 > 0x40 )
      {
LABEL_169:
        v98 = v34;
        v104 = v36;
        sub_16A5260(&v140, 0, v40);
        v42 = (unsigned int)v141;
        v34 = v98;
        v36 = v104;
        goto LABEL_47;
      }
      v42 = (unsigned int)v141;
      v41 = 0xFFFFFFFFFFFFFFFFLL >> (63 - (unsigned __int8)v102);
      v43 = v140;
      if ( (unsigned int)v141 <= 0x40 )
        goto LABEL_46;
      *(_QWORD *)v140 |= v41;
    }
    v42 = (unsigned int)v141;
    goto LABEL_47;
  }
  v140 = 0;
  if ( v36 == -1 )
  {
    if ( *(_QWORD *)(v34 + 24) )
    {
LABEL_171:
      v17 = v34;
      goto LABEL_4;
    }
    goto LABEL_142;
  }
  if ( v40 > 0x40 )
    goto LABEL_169;
  v41 = 0xFFFFFFFFFFFFFFFFLL >> (63 - (unsigned __int8)v36);
  v42 = v39;
  v43 = 0;
LABEL_46:
  v140 = (const char *)(v41 | (unsigned __int64)v43);
LABEL_47:
  if ( *(_DWORD *)(v34 + 32) <= 0x40u )
  {
    if ( v140 != *(const char **)(v34 + 24) )
      goto LABEL_49;
  }
  else
  {
    v95 = v36;
    v99 = v42;
    v116 = v34;
    v44 = sub_16A5220(v34 + 24, (const void **)&v140);
    v34 = v116;
    v42 = v99;
    v36 = v95;
    if ( !v44 )
    {
LABEL_49:
      if ( v42 > 0x40 )
      {
        if ( v140 )
        {
          j_j___libc_free_0_0(v140);
          v17 = *(_QWORD *)(a2 - 24);
          goto LABEL_4;
        }
        goto LABEL_3;
      }
      v34 = *(_QWORD *)(a2 - 24);
      goto LABEL_171;
    }
  }
  if ( v42 > 0x40 && v140 )
  {
    v105 = v34;
    v118 = v36;
    j_j___libc_free_0_0(v140);
    v34 = v105;
    v36 = v118;
  }
LABEL_142:
  v68 = *(_DWORD *)(v34 + 32) - v36;
  if ( v68 > (unsigned int)sub_14C23D0(v110, a1[333], 0, a1[330], a2, a1[332])
    || v68 > (unsigned int)sub_14C23D0(v106, a1[333], 0, a1[330], a2, a1[332]) )
  {
    goto LABEL_3;
  }
  if ( (*(_BYTE *)(v113 + 23) & 0x40) != 0 )
    v69 = *(__int64 **)(v113 - 8);
  else
    v69 = (__int64 *)(v113 - 24LL * (*(_DWORD *)(v113 + 20) & 0xFFFFFFF));
  v70 = *v69;
  v71 = *(_QWORD *)(v70 + 8);
  for ( i = v70; v71; v71 = *(_QWORD *)(v71 + 8) )
  {
    v72 = sub_1648700(v71);
    if ( (_QWORD *)v113 != v72 && (*((_BYTE *)v72 + 16) != 60 || v40 < (unsigned int)sub_1643030(*v72)) )
      goto LABEL_3;
  }
  v73 = (_QWORD *)sub_16498A0(i);
  v132 = (__int64 **)sub_1644900(v73, v40);
  v74 = (__int64 *)sub_15F2050(a2);
  v75 = sub_15E26F0(v74, 189, (__int64 *)&v132, 1);
  v76 = (__int64 *)a1[1];
  v114 = v75;
  v76[1] = *(_QWORD *)(i + 40);
  v76[2] = i + 24;
  v77 = *(const char **)(i + 48);
  v140 = v77;
  if ( v77 )
  {
    sub_1623A60((__int64)&v140, (__int64)v77, 2);
    v78 = *v76;
    if ( !*v76 )
      goto LABEL_154;
  }
  else
  {
    v78 = *v76;
    if ( !*v76 )
      goto LABEL_156;
  }
  sub_161E7C0((__int64)v76, v78);
LABEL_154:
  v79 = (unsigned __int8 *)v140;
  *v76 = (__int64)v140;
  if ( v79 )
  {
    sub_1623210((__int64)&v140, v79, (__int64)v76);
  }
  else if ( v140 )
  {
    sub_161E7C0((__int64)&v140, (__int64)v140);
  }
LABEL_156:
  v80 = sub_1649960(v110);
  v137 = v81;
  v141 = ".trunc";
  v136 = (__int64 *)v80;
  LOWORD(v142) = 773;
  v140 = (const char *)&v136;
  v112 = sub_1708970((__int64)v76, 36, v110, v132, (__int64 *)&v140);
  v82 = sub_1649960(v106);
  v137 = v83;
  v141 = ".trunc";
  v136 = (__int64 *)v82;
  LOWORD(v142) = 773;
  v140 = (const char *)&v136;
  v84 = sub_1708970((__int64)v76, 36, v106, v132, (__int64 *)&v140);
  LOWORD(v142) = 259;
  v140 = "sadd";
  v136 = (__int64 *)v112;
  v137 = v84;
  v115 = (__int64 *)sub_172C570(
                      (__int64)v76,
                      *(_QWORD *)(*(_QWORD *)v114 + 24LL),
                      v114,
                      (__int64 *)&v136,
                      2,
                      (__int64 *)&v140,
                      0);
  v140 = "sadd.result";
  LOWORD(v142) = 259;
  LODWORD(v136) = 0;
  v85 = sub_1759FE0((__int64)v76, (__int64)v115, (unsigned int *)&v136, 1, (__int64 *)&v140);
  LOWORD(v142) = 257;
  v86 = sub_1708970((__int64)v76, 37, (__int64)v85, *(__int64 ***)i, (__int64 *)&v140);
  sub_170E100(a1, i, (__int64)v86, a3, a4, a5, a6, v87, v88, a9, a10);
  v140 = "sadd.overflow";
  LOWORD(v142) = 259;
  LODWORD(v136) = 1;
  v89 = sub_1648A60(88, 1u);
  v18 = (__int64)v89;
  if ( !v89 )
    goto LABEL_3;
  v90 = v89 - 3;
  v91 = sub_15FB2A0(*v115, (unsigned int *)&v136, 1);
  sub_15F1EA0(v18, v91, 62, v18 - 24, 1, 0);
  if ( *(_QWORD *)(v18 - 24) )
  {
    v92 = *(_QWORD *)(v18 - 16);
    v93 = *(_QWORD *)(v18 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v93 = v92;
    if ( v92 )
      *(_QWORD *)(v92 + 16) = v93 | *(_QWORD *)(v92 + 16) & 3LL;
  }
  *(_QWORD *)(v18 - 24) = v115;
  v94 = v115[1];
  *(_QWORD *)(v18 - 16) = v94;
  if ( v94 )
    *(_QWORD *)(v94 + 16) = (v18 - 16) | *(_QWORD *)(v94 + 16) & 3LL;
  *(_QWORD *)(v18 - 8) = (unsigned __int64)(v115 + 1) | *(_QWORD *)(v18 - 8) & 3LL;
  v115[1] = (__int64)v90;
  *(_QWORD *)(v18 + 56) = v18 + 72;
  *(_QWORD *)(v18 + 64) = 0x400000000LL;
  sub_15FB110(v18, &v136, 1, (__int64)&v140);
  return v18;
}
