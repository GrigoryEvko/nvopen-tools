// Function: sub_3927220
// Address: 0x3927220
//
unsigned __int64 __fastcall sub_3927220(
        __int64 a1,
        __int64 *a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 *v9; // r8
  char v10; // al
  __int64 v14; // rdx
  unsigned __int64 v15; // rbx
  __int64 v16; // rdi
  _QWORD *v17; // rax
  unsigned __int64 *v18; // rax
  char *v19; // rax
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // rbx
  __int64 v24; // r9
  unsigned int v25; // edi
  _QWORD *v26; // rax
  __int64 v27; // rcx
  _QWORD *v28; // r14
  __int64 v29; // rbx
  __int64 v30; // rsi
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  _QWORD *v33; // rdx
  const char *v34; // rax
  int v35; // eax
  __int64 v36; // r8
  char v37; // dl
  unsigned __int64 v38; // rax
  unsigned int v39; // esi
  __int64 v40; // r15
  int v41; // edi
  int v42; // edi
  __int64 v43; // rdx
  unsigned int v44; // r9d
  int v45; // ecx
  _QWORD *v46; // r11
  __int64 v47; // rax
  int v48; // esi
  _QWORD *v49; // r10
  unsigned int v50; // esi
  __int64 v51; // r9
  unsigned int v52; // r15d
  unsigned int v53; // edi
  __int64 *v54; // rax
  __int64 v55; // rcx
  __int64 v56; // r15
  __int32 v57; // r13d
  __int16 v58; // bx
  __int16 v59; // ax
  __int64 v60; // rdi
  unsigned __int64 result; // rax
  __m128i *v62; // r12
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rdi
  __int64 v67; // rdi
  __int64 v68; // rcx
  _QWORD *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rax
  unsigned __int64 *v72; // rax
  unsigned __int64 v73; // rdx
  _QWORD *v74; // rax
  int v75; // r10d
  _QWORD *v76; // rdx
  int v77; // eax
  int v78; // ecx
  unsigned __int64 *v79; // rdx
  int v80; // edx
  __int64 *v81; // r10
  unsigned __int64 v82; // r8
  __int8 *v83; // rsi
  unsigned __int64 v84; // rax
  unsigned __int64 v85; // rdx
  bool v86; // cf
  unsigned __int64 v87; // rax
  unsigned __int64 v88; // rcx
  __int64 v89; // rax
  __m128i *v90; // rdx
  unsigned __int64 v91; // rcx
  char *v92; // rsi
  __m128i *v93; // rsi
  const __m128i *v94; // rax
  int v95; // r9d
  int v96; // r9d
  __int64 v97; // rsi
  unsigned int v98; // r11d
  __int64 v99; // rax
  int v100; // r10d
  _QWORD *v101; // r14
  int v102; // r9d
  int v103; // r9d
  __int64 v104; // rsi
  _QWORD *v105; // r11
  unsigned int v106; // r14d
  int v107; // r10d
  __int64 v108; // rax
  int v109; // eax
  __int64 *v110; // rdx
  int v111; // eax
  int v112; // ecx
  int v113; // eax
  int v114; // eax
  __int64 v115; // rdi
  __int64 v116; // r11
  __int64 v117; // rsi
  int v118; // r15d
  __int64 *v119; // r13
  int v120; // r9d
  int v121; // r9d
  int v122; // r13d
  __int64 *v123; // r11
  __int64 v124; // rsi
  __int64 v125; // r15
  __int64 v126; // rax
  int v127; // r9d
  int v128; // r9d
  __int64 v129; // rdx
  unsigned int v130; // edi
  __int64 v131; // rax
  int v132; // r10d
  _QWORD *v133; // rsi
  __int64 *v134; // r10
  int v135; // eax
  unsigned int v136; // [rsp+8h] [rbp-B8h]
  __int64 *v137; // [rsp+8h] [rbp-B8h]
  __int64 v138; // [rsp+10h] [rbp-B0h]
  __int64 v139; // [rsp+18h] [rbp-A8h]
  __int64 v140; // [rsp+18h] [rbp-A8h]
  __int64 *v141; // [rsp+18h] [rbp-A8h]
  __int64 *v142; // [rsp+18h] [rbp-A8h]
  int v143; // [rsp+18h] [rbp-A8h]
  int v144; // [rsp+18h] [rbp-A8h]
  __int64 v145; // [rsp+18h] [rbp-A8h]
  __int64 v146; // [rsp+18h] [rbp-A8h]
  __int64 v147; // [rsp+18h] [rbp-A8h]
  __int64 *v148; // [rsp+20h] [rbp-A0h]
  int v149; // [rsp+20h] [rbp-A0h]
  __int64 v150; // [rsp+20h] [rbp-A0h]
  __int64 *v151; // [rsp+20h] [rbp-A0h]
  __int64 *v152; // [rsp+20h] [rbp-A0h]
  __int8 *v154; // [rsp+28h] [rbp-98h]
  __m128i *v155; // [rsp+28h] [rbp-98h]
  unsigned __int64 v157; // [rsp+30h] [rbp-90h]
  unsigned __int64 v158; // [rsp+30h] [rbp-90h]
  unsigned __int64 v160; // [rsp+38h] [rbp-88h]
  unsigned __int64 v161; // [rsp+38h] [rbp-88h]
  _QWORD *v162; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v163; // [rsp+48h] [rbp-78h]
  char *v164; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v165; // [rsp+58h] [rbp-68h]
  __int16 v166; // [rsp+60h] [rbp-60h]
  _QWORD v167[2]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v168; // [rsp+80h] [rbp-40h]

  v9 = *(__int64 **)(a7 + 24);
  v10 = *((_BYTE *)v9 + 8);
  if ( (v10 & 8) == 0 )
  {
    v16 = *a2;
    if ( (*(_BYTE *)v9 & 4) != 0 )
    {
      v72 = (unsigned __int64 *)*(v9 - 1);
      v73 = *v72;
      v74 = v72 + 2;
    }
    else
    {
      v73 = 0;
      v74 = 0;
    }
    v163 = v73;
    v162 = v74;
    v19 = "symbol '";
    goto LABEL_61;
  }
  if ( (v10 & 1) != 0 )
  {
    v14 = *v9;
    v15 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v15 )
    {
      if ( (*((_BYTE *)v9 + 9) & 0xC) != 8
        || (v20 = v9[3],
            v148 = *(__int64 **)(a7 + 24),
            *((_BYTE *)v9 + 8) = v10 | 4,
            v21 = (unsigned __int64)sub_38CE440(v20),
            v9 = v148,
            v14 = v21 | *v148 & 7,
            *v148 = v14,
            !v21) )
      {
        v16 = *a2;
        v17 = 0;
        if ( (v14 & 4) != 0 )
        {
          v18 = (unsigned __int64 *)*(v9 - 1);
          v15 = *v18;
          v17 = v18 + 2;
        }
        v162 = v17;
        v19 = "assembler label '";
        v163 = v15;
LABEL_61:
        v164 = v19;
        v166 = 1283;
        v165 = &v162;
        v167[0] = &v164;
        v34 = "' can not be undefined";
LABEL_62:
        v167[1] = v34;
        v168 = 770;
        return (unsigned __int64)sub_38BE3D0(v16, *(_QWORD *)(a5 + 16), (__int64)v167);
      }
    }
  }
  v22 = *(_DWORD *)(a1 + 184);
  v23 = *(_QWORD *)(a4 + 24);
  v138 = a1 + 160;
  if ( !v22 )
  {
    ++*(_QWORD *)(a1 + 160);
LABEL_100:
    v151 = v9;
    sub_3925B80(v138, 2 * v22);
    v95 = *(_DWORD *)(a1 + 184);
    if ( !v95 )
      goto LABEL_199;
    v96 = v95 - 1;
    v9 = v151;
    v97 = *(_QWORD *)(a1 + 168);
    v98 = v96 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
    v78 = *(_DWORD *)(a1 + 176) + 1;
    v76 = (_QWORD *)(v97 + 16LL * v98);
    v99 = *v76;
    if ( v23 != *v76 )
    {
      v100 = 1;
      v101 = 0;
      while ( v99 != -8 )
      {
        if ( !v101 && v99 == -16 )
          v101 = v76;
        v98 = v96 & (v100 + v98);
        v76 = (_QWORD *)(v97 + 16LL * v98);
        v99 = *v76;
        if ( v23 == *v76 )
          goto LABEL_70;
        ++v100;
      }
      if ( v101 )
        v76 = v101;
    }
    goto LABEL_70;
  }
  v24 = *(_QWORD *)(a1 + 168);
  v25 = (v22 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
  v26 = (_QWORD *)(v24 + 16LL * v25);
  v27 = *v26;
  if ( v23 == *v26 )
  {
    v28 = (_QWORD *)v26[1];
    goto LABEL_12;
  }
  v75 = 1;
  v76 = 0;
  while ( v27 != -8 )
  {
    if ( v76 || v27 != -16 )
      v26 = v76;
    v80 = v75 + 1;
    v25 = (v22 - 1) & (v75 + v25);
    v81 = (__int64 *)(v24 + 16LL * v25);
    v27 = *v81;
    if ( v23 == *v81 )
    {
      v28 = (_QWORD *)v81[1];
      goto LABEL_12;
    }
    v75 = v80;
    v76 = v26;
    v26 = (_QWORD *)(v24 + 16LL * v25);
  }
  if ( !v76 )
    v76 = v26;
  v77 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 160);
  v78 = v77 + 1;
  if ( 4 * (v77 + 1) >= 3 * v22 )
    goto LABEL_100;
  if ( v22 - *(_DWORD *)(a1 + 180) - v78 <= v22 >> 3 )
  {
    v152 = v9;
    sub_3925B80(v138, v22);
    v102 = *(_DWORD *)(a1 + 184);
    if ( !v102 )
      goto LABEL_199;
    v103 = v102 - 1;
    v104 = *(_QWORD *)(a1 + 168);
    v105 = 0;
    v106 = v103 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
    v9 = v152;
    v107 = 1;
    v78 = *(_DWORD *)(a1 + 176) + 1;
    v76 = (_QWORD *)(v104 + 16LL * v106);
    v108 = *v76;
    if ( v23 != *v76 )
    {
      while ( v108 != -8 )
      {
        if ( v108 == -16 && !v105 )
          v105 = v76;
        v106 = v103 & (v107 + v106);
        v76 = (_QWORD *)(v104 + 16LL * v106);
        v108 = *v76;
        if ( v23 == *v76 )
          goto LABEL_70;
        ++v107;
      }
      if ( v105 )
        v76 = v105;
    }
  }
LABEL_70:
  *(_DWORD *)(a1 + 176) = v78;
  if ( *v76 != -8 )
    --*(_DWORD *)(a1 + 180);
  *v76 = v23;
  v28 = 0;
  v76[1] = 0;
LABEL_12:
  v29 = a8;
  if ( !a8 )
  {
    *a6 = a9;
    goto LABEL_19;
  }
  v30 = *(_QWORD *)(a8 + 24);
  v31 = *(_QWORD *)v30;
  v32 = *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v32 )
  {
    if ( (*(_BYTE *)(v30 + 9) & 0xC) != 8 )
      goto LABEL_15;
    *(_BYTE *)(v30 + 8) |= 4u;
    v141 = v9;
    v63 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v30 + 24));
    v9 = v141;
    v64 = v63;
    v32 = 0;
    v31 = v63 | *(_QWORD *)v30 & 7LL;
    *(_QWORD *)v30 = v31;
    if ( !v64 )
    {
LABEL_15:
      v33 = 0;
      v16 = *a2;
      if ( (v31 & 4) != 0 )
      {
        v79 = *(unsigned __int64 **)(v30 - 8);
        v32 = *v79;
        v33 = v79 + 2;
      }
      v162 = v33;
      v164 = "symbol '";
      v165 = &v162;
      v167[0] = &v164;
      v34 = "' can not be undefined in a subtraction expression";
      v163 = v32;
      v166 = 1283;
      goto LABEL_62;
    }
  }
  v142 = v9;
  v150 = sub_38D0440(a3, v30);
  v65 = sub_38D01B0((__int64)a3, a4);
  v9 = v142;
  *a6 = a9 + v65 + *(unsigned int *)(a5 + 8) - v150;
LABEL_19:
  v139 = (__int64)v9;
  v35 = sub_38D01B0((__int64)a3, a4);
  v36 = v139;
  v149 = v35;
  v37 = *(_BYTE *)(v139 + 8);
  if ( (v37 & 1) == 0 )
  {
    v50 = *(_DWORD *)(a1 + 216);
    if ( v50 )
    {
      v51 = *(_QWORD *)(a1 + 200);
      v52 = ((unsigned int)v139 >> 4) ^ ((unsigned int)v139 >> 9);
      v53 = (v50 - 1) & v52;
      v54 = (__int64 *)(v51 + 16LL * v53);
      v55 = *v54;
      if ( v139 == *v54 )
      {
        v56 = v54[1];
        goto LABEL_34;
      }
      v144 = 1;
      v110 = 0;
      while ( v55 != -8 )
      {
        if ( v110 || v55 != -16 )
          v54 = v110;
        v53 = (v50 - 1) & (v144 + v53);
        v134 = (__int64 *)(v51 + 16LL * v53);
        v55 = *v134;
        if ( v36 == *v134 )
        {
          v56 = v134[1];
          goto LABEL_34;
        }
        ++v144;
        v110 = v54;
        v54 = (__int64 *)(v51 + 16LL * v53);
      }
      if ( !v110 )
        v110 = v54;
      v111 = *(_DWORD *)(a1 + 208);
      ++*(_QWORD *)(a1 + 192);
      v112 = v111 + 1;
      if ( 4 * (v111 + 1) < 3 * v50 )
      {
        if ( v50 - *(_DWORD *)(a1 + 212) - v112 > v50 >> 3 )
        {
LABEL_128:
          *(_DWORD *)(a1 + 208) = v112;
          if ( *v110 != -8 )
            --*(_DWORD *)(a1 + 212);
          *v110 = v36;
          v56 = 0;
          v110[1] = 0;
          goto LABEL_34;
        }
        v146 = v36;
        sub_3925D30(a1 + 192, v50);
        v120 = *(_DWORD *)(a1 + 216);
        if ( v120 )
        {
          v121 = v120 - 1;
          v122 = 1;
          v123 = 0;
          v124 = *(_QWORD *)(a1 + 200);
          LODWORD(v125) = v121 & v52;
          v36 = v146;
          v112 = *(_DWORD *)(a1 + 208) + 1;
          v110 = (__int64 *)(v124 + 16LL * (unsigned int)v125);
          v126 = *v110;
          if ( v146 != *v110 )
          {
            while ( v126 != -8 )
            {
              if ( !v123 && v126 == -16 )
                v123 = v110;
              v125 = v121 & (unsigned int)(v125 + v122);
              v110 = (__int64 *)(v124 + 16 * v125);
              v126 = *v110;
              if ( v146 == *v110 )
                goto LABEL_128;
              ++v122;
            }
            if ( v123 )
              v110 = v123;
          }
          goto LABEL_128;
        }
LABEL_200:
        ++*(_DWORD *)(a1 + 208);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 192);
    }
    v145 = v36;
    sub_3925D30(a1 + 192, 2 * v50);
    v113 = *(_DWORD *)(a1 + 216);
    if ( v113 )
    {
      v36 = v145;
      v114 = v113 - 1;
      v115 = *(_QWORD *)(a1 + 200);
      v112 = *(_DWORD *)(a1 + 208) + 1;
      LODWORD(v116) = v114 & (((unsigned int)v145 >> 9) ^ ((unsigned int)v145 >> 4));
      v110 = (__int64 *)(v115 + 16LL * (unsigned int)v116);
      v117 = *v110;
      if ( v145 != *v110 )
      {
        v118 = 1;
        v119 = 0;
        while ( v117 != -8 )
        {
          if ( !v119 && v117 == -16 )
            v119 = v110;
          v116 = v114 & (unsigned int)(v116 + v118);
          v110 = (__int64 *)(v115 + 16 * v116);
          v117 = *v110;
          if ( v145 == *v110 )
            goto LABEL_128;
          ++v118;
        }
        if ( v119 )
          v110 = v119;
      }
      goto LABEL_128;
    }
    goto LABEL_200;
  }
  v38 = *(_QWORD *)v139 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v38
    || (*(_BYTE *)(v139 + 9) & 0xC) == 8
    && (v66 = *(_QWORD *)(v139 + 24),
        *(_BYTE *)(v139 + 8) = v37 | 4,
        v38 = (unsigned __int64)sub_38CE440(v66),
        v36 = v139,
        *(_QWORD *)v139 = v38 | *(_QWORD *)v139 & 7LL,
        v38) )
  {
    v39 = *(_DWORD *)(a1 + 184);
    v40 = *(_QWORD *)(v38 + 24);
    if ( v39 )
      goto LABEL_50;
LABEL_23:
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_24;
  }
  v39 = *(_DWORD *)(a1 + 184);
  v40 = 0;
  if ( !v39 )
    goto LABEL_23;
LABEL_50:
  v67 = *(_QWORD *)(a1 + 168);
  v68 = (v39 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
  v69 = (_QWORD *)(v67 + 16 * v68);
  v70 = *v69;
  if ( v40 == *v69 )
  {
    v71 = v69[1];
    goto LABEL_52;
  }
  v143 = 1;
  v46 = 0;
  while ( v70 != -8 )
  {
    if ( v46 || v70 != -16 )
      v69 = v46;
    LODWORD(v68) = (v39 - 1) & (v143 + v68);
    v137 = (__int64 *)(v67 + 16LL * (unsigned int)v68);
    v70 = *v137;
    if ( v40 == *v137 )
    {
      v71 = v137[1];
      goto LABEL_52;
    }
    ++v143;
    v46 = v69;
    v69 = (_QWORD *)(v67 + 16LL * (unsigned int)v68);
  }
  if ( !v46 )
    v46 = v69;
  v109 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 160);
  v45 = v109 + 1;
  if ( 4 * (v109 + 1) < 3 * v39 )
  {
    if ( v39 - *(_DWORD *)(a1 + 180) - v45 > v39 >> 3 )
      goto LABEL_119;
    v136 = ((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4);
    v147 = v36;
    sub_3925B80(v138, v39);
    v127 = *(_DWORD *)(a1 + 184);
    if ( v127 )
    {
      v128 = v127 - 1;
      v129 = *(_QWORD *)(a1 + 168);
      v36 = v147;
      v130 = v128 & v136;
      v45 = *(_DWORD *)(a1 + 176) + 1;
      v46 = (_QWORD *)(v129 + 16LL * (v128 & v136));
      v131 = *v46;
      if ( v40 != *v46 )
      {
        v132 = 1;
        v133 = 0;
        while ( v131 != -8 )
        {
          if ( !v133 && v131 == -16 )
            v133 = v46;
          v135 = v132++;
          v130 = v128 & (v135 + v130);
          v46 = (_QWORD *)(v129 + 16LL * v130);
          v131 = *v46;
          if ( v40 == *v46 )
            goto LABEL_119;
        }
        if ( v133 )
          v46 = v133;
      }
      goto LABEL_119;
    }
LABEL_199:
    ++*(_DWORD *)(a1 + 176);
    BUG();
  }
LABEL_24:
  v140 = v36;
  sub_3925B80(v138, 2 * v39);
  v41 = *(_DWORD *)(a1 + 184);
  if ( !v41 )
    goto LABEL_199;
  v42 = v41 - 1;
  v36 = v140;
  v43 = *(_QWORD *)(a1 + 168);
  v44 = v42 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
  v45 = *(_DWORD *)(a1 + 176) + 1;
  v46 = (_QWORD *)(v43 + 16LL * v44);
  v47 = *v46;
  if ( *v46 != v40 )
  {
    v48 = 1;
    v49 = 0;
    while ( v47 != -8 )
    {
      if ( !v49 && v47 == -16 )
        v49 = v46;
      v44 = v42 & (v48 + v44);
      v46 = (_QWORD *)(v43 + 16LL * v44);
      v47 = *v46;
      if ( v40 == *v46 )
        goto LABEL_119;
      ++v48;
    }
    if ( v49 )
      v46 = v49;
  }
LABEL_119:
  *(_DWORD *)(a1 + 176) = v45;
  if ( *v46 != -8 )
    --*(_DWORD *)(a1 + 180);
  *v46 = v40;
  v71 = 0;
  v46[1] = 0;
LABEL_52:
  v56 = *(_QWORD *)(v71 + 88);
  *a6 += sub_38D0440(a3, v36);
LABEL_34:
  ++*(_DWORD *)(v56 + 112);
  v57 = *(_DWORD *)(a5 + 8) + v149;
  v58 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64 *, __int64, bool, __int64))(**(_QWORD **)(a1 + 24) + 32LL))(
          *(_QWORD *)(a1 + 24),
          *a2,
          &a7,
          a5,
          v29 != 0,
          a2[1]);
  v59 = *(_WORD *)(a1 + 32);
  if ( v59 == -31132 )
  {
    if ( v58 != 4 )
      goto LABEL_38;
    goto LABEL_57;
  }
  if ( v59 == 332 && v58 == 20 )
  {
LABEL_57:
    *a6 += 4LL;
    v59 = *(_WORD *)(a1 + 32);
  }
  if ( v59 == 452 && (v58 == 18 || (unsigned __int16)(v58 - 20) <= 1u) )
    *a6 += 4LL;
LABEL_38:
  if ( *(_DWORD *)(a5 + 12) == 17 )
    *a6 = 0;
  v60 = *(_QWORD *)(a1 + 24);
  result = *(_QWORD *)(*(_QWORD *)v60 + 40LL);
  if ( (__int64 (*)())result == sub_3924E40
    || (result = ((__int64 (__fastcall *)(__int64, __int64))result)(v60, a5), (_BYTE)result) )
  {
    v62 = (__m128i *)v28[13];
    if ( v62 != (__m128i *)v28[14] )
    {
      if ( v62 )
      {
        v62->m128i_i32[0] = v57;
        v62->m128i_i32[1] = 0;
        v62->m128i_i16[4] = v58;
        v62[1].m128i_i64[0] = v56;
        v62 = (__m128i *)v28[13];
      }
      v28[13] = (char *)v62 + 24;
      return result;
    }
    v82 = v28[12];
    v83 = &v62->m128i_i8[-v82];
    v84 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v62->m128i_i64 - v82) >> 3);
    if ( v84 == 0x555555555555555LL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v85 = 1;
    if ( v84 )
      v85 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v62->m128i_i64 - v82) >> 3);
    v86 = __CFADD__(v85, v84);
    v87 = v85 - 0x5555555555555555LL * ((__int64)((__int64)v62->m128i_i64 - v82) >> 3);
    if ( v86 )
    {
      v88 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v87 )
      {
        result = 24;
        v91 = 0;
        v90 = 0;
        goto LABEL_88;
      }
      if ( v87 > 0x555555555555555LL )
        v87 = 0x555555555555555LL;
      v88 = 24 * v87;
    }
    v154 = &v62->m128i_i8[-v82];
    v157 = v28[12];
    v160 = v88;
    v89 = sub_22077B0(v88);
    v82 = v157;
    v83 = v154;
    v90 = (__m128i *)v89;
    v91 = v89 + v160;
    result = v89 + 24;
LABEL_88:
    v92 = &v83[(_QWORD)v90];
    if ( v92 )
    {
      *(_DWORD *)v92 = v57;
      *((_DWORD *)v92 + 1) = 0;
      *((_WORD *)v92 + 4) = v58;
      *((_QWORD *)v92 + 2) = v56;
    }
    if ( v62 != (__m128i *)v82 )
    {
      v93 = v90;
      v94 = (const __m128i *)v82;
      do
      {
        if ( v93 )
        {
          *v93 = _mm_loadu_si128(v94);
          v93[1].m128i_i64[0] = v94[1].m128i_i64[0];
        }
        v94 = (const __m128i *)((char *)v94 + 24);
        v93 = (__m128i *)((char *)v93 + 24);
      }
      while ( v62 != v94 );
      result = (unsigned __int64)&v90[3] + 8 * (((unsigned __int64)&v62[-2].m128i_u64[1] - v82) >> 3);
    }
    if ( v82 )
    {
      v155 = v90;
      v158 = v91;
      v161 = result;
      j_j___libc_free_0(v82);
      v90 = v155;
      v91 = v158;
      result = v161;
    }
    v28[12] = v90;
    v28[13] = result;
    v28[14] = v91;
  }
  return result;
}
