// Function: sub_1BDCC60
// Address: 0x1bdcc60
//
__int64 __fastcall sub_1BDCC60(
        __int64 *a1,
        __m128i a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 *a14)
{
  __int64 *v14; // r14
  __int64 *v15; // r15
  unsigned int v16; // eax
  __int64 v17; // rbx
  __int64 v18; // r11
  __int64 v19; // r14
  __int64 *v20; // r15
  int v21; // r8d
  __int64 *v22; // rdi
  unsigned __int64 v23; // rax
  int v24; // esi
  __int64 *v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // r12
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 *v31; // rsi
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v34; // r14
  int v35; // r13d
  _QWORD *v36; // r10
  unsigned int v37; // edx
  _QWORD *v38; // rdi
  __int64 v39; // r8
  int v40; // edx
  unsigned int v41; // ecx
  __int64 v42; // r8
  __int64 v43; // rax
  _BYTE *v44; // rsi
  __int64 *v45; // r10
  __int64 v46; // rax
  int v47; // r14d
  int v48; // ecx
  unsigned int v49; // edx
  _BYTE *v50; // rax
  _BYTE *v51; // rsi
  _BYTE *v52; // rax
  _QWORD *v53; // rdi
  __int64 v54; // rbx
  __int64 v55; // rsi
  unsigned __int64 v56; // r10
  int v57; // r11d
  __int64 v58; // rbx
  unsigned int v59; // ecx
  __int64 v60; // r8
  __int64 v61; // r12
  __int64 *v62; // r14
  __int64 v63; // r13
  __int64 v64; // rax
  _QWORD *v65; // rsi
  unsigned int v66; // eax
  unsigned int v67; // ecx
  _QWORD *v68; // rdi
  _BYTE *v69; // r8
  _BYTE *v70; // rax
  _BYTE *v71; // rsi
  __int64 v72; // rcx
  unsigned int v73; // edx
  __int64 *v74; // rdi
  __int64 v75; // r8
  _BYTE *v76; // rax
  _BYTE *v77; // rsi
  unsigned __int64 v78; // rax
  unsigned __int64 v79; // rax
  __int64 v80; // rax
  _QWORD *v81; // rax
  __int64 *v82; // r14
  _QWORD *k; // rdx
  __int64 *v84; // rax
  __int64 v85; // rdi
  unsigned int v86; // edx
  _QWORD *v87; // rsi
  __int64 v88; // r8
  unsigned int v89; // ecx
  unsigned int v90; // ecx
  __int64 *v91; // r8
  unsigned int v92; // r13d
  __int64 v93; // rsi
  unsigned int v95; // edx
  _QWORD *v96; // rsi
  _BYTE *v97; // rdi
  int v98; // edi
  int v99; // edi
  unsigned __int64 v100; // r12
  _BYTE *v101; // rdi
  _QWORD *v102; // r9
  __int64 v103; // rax
  unsigned __int64 v104; // r8
  int v105; // esi
  _QWORD *v106; // rcx
  _QWORD *v107; // rdx
  _QWORD *v108; // rsi
  _QWORD *v109; // rdx
  _QWORD *v110; // r10
  int v111; // r10d
  __int64 v112; // [rsp+8h] [rbp-158h]
  __int64 v114; // [rsp+18h] [rbp-148h]
  unsigned __int8 v115; // [rsp+23h] [rbp-13Dh]
  unsigned int v116; // [rsp+24h] [rbp-13Ch]
  __int64 v117; // [rsp+28h] [rbp-138h]
  __int64 v118; // [rsp+30h] [rbp-130h]
  __int64 v119; // [rsp+30h] [rbp-130h]
  __int64 v120; // [rsp+30h] [rbp-130h]
  __int64 v121; // [rsp+30h] [rbp-130h]
  int v122; // [rsp+30h] [rbp-130h]
  unsigned int v123; // [rsp+38h] [rbp-128h]
  signed int v124; // [rsp+3Ch] [rbp-124h]
  __int64 v125; // [rsp+40h] [rbp-120h]
  int v126; // [rsp+48h] [rbp-118h]
  __int64 *v127; // [rsp+50h] [rbp-110h]
  __int64 *v128; // [rsp+50h] [rbp-110h]
  __int64 v129; // [rsp+58h] [rbp-108h]
  __int64 *v130; // [rsp+58h] [rbp-108h]
  __int64 v131; // [rsp+60h] [rbp-100h]
  _QWORD *v132; // [rsp+68h] [rbp-F8h]
  __int64 v133; // [rsp+70h] [rbp-F0h]
  unsigned int v134; // [rsp+78h] [rbp-E8h]
  _QWORD *v135; // [rsp+80h] [rbp-E0h] BYREF
  _BYTE *v136; // [rsp+88h] [rbp-D8h]
  _BYTE *v137; // [rsp+90h] [rbp-D0h]
  _BYTE *v138; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v139; // [rsp+A8h] [rbp-B8h]
  _BYTE s[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v114 = a1[20];
  v112 = a1[21];
  if ( v114 == v112 )
    return 0;
  v115 = 0;
  v14 = a1;
  do
  {
    v116 = *(_DWORD *)(v114 + 16);
    if ( v116 <= 1 )
      goto LABEL_97;
    v123 = 0;
    v15 = v14;
    do
    {
      v16 = 16;
      if ( v116 - v123 <= 0x10 )
        v16 = v116 - v123;
      v124 = v16;
      v131 = 0;
      v132 = 0;
      v133 = 0;
      v17 = *(_QWORD *)(v114 + 8) + 24LL * v123;
      v129 = v17;
      v134 = 0;
      v18 = v17 + 24LL * v16;
      v135 = 0;
      v136 = 0;
      v137 = 0;
      if ( v17 == v18 )
        goto LABEL_111;
      v127 = v15;
      v19 = 0;
      v20 = 0;
      do
      {
        while ( 1 )
        {
          v27 = *(_QWORD *)(v17 + 16);
          if ( !(_DWORD)v19 )
          {
            ++v131;
LABEL_13:
            v118 = v18;
            v28 = (((((((unsigned int)(2 * v19 - 1) | ((unsigned __int64)(unsigned int)(2 * v19 - 1) >> 1)) >> 2)
                    | (unsigned int)(2 * v19 - 1)
                    | ((unsigned __int64)(unsigned int)(2 * v19 - 1) >> 1)) >> 4)
                  | (((unsigned int)(2 * v19 - 1) | ((unsigned __int64)(unsigned int)(2 * v19 - 1) >> 1)) >> 2)
                  | (unsigned int)(2 * v19 - 1)
                  | ((unsigned __int64)(unsigned int)(2 * v19 - 1) >> 1)) >> 8)
                | (((((unsigned int)(2 * v19 - 1) | ((unsigned __int64)(unsigned int)(2 * v19 - 1) >> 1)) >> 2)
                  | (unsigned int)(2 * v19 - 1)
                  | ((unsigned __int64)(unsigned int)(2 * v19 - 1) >> 1)) >> 4)
                | (((unsigned int)(2 * v19 - 1) | ((unsigned __int64)(unsigned int)(2 * v19 - 1) >> 1)) >> 2)
                | (unsigned int)(2 * v19 - 1)
                | ((unsigned __int64)(unsigned int)(2 * v19 - 1) >> 1);
            v29 = ((v28 >> 16) | v28) + 1;
            if ( (unsigned int)v29 < 0x40 )
              LODWORD(v29) = 64;
            v134 = v29;
            v30 = (_QWORD *)sub_22077B0(8LL * (unsigned int)v29);
            v18 = v118;
            v132 = v30;
            if ( v20 )
            {
              v31 = &v20[v19];
              v133 = 0;
              for ( i = &v30[v134]; i != v30; ++v30 )
              {
                if ( v30 )
                  *v30 = -8;
              }
              for ( j = v20; v31 != j; ++j )
              {
                v34 = *j;
                if ( *j != -8 && v34 != -16 )
                {
                  v35 = 1;
                  v36 = 0;
                  v37 = (v134 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
                  v38 = &v132[v37];
                  v39 = *v38;
                  if ( v34 != *v38 )
                  {
                    while ( v39 != -8 )
                    {
                      if ( v36 || v39 != -16 )
                        v38 = v36;
                      v37 = (v134 - 1) & (v35 + v37);
                      v39 = v132[v37];
                      if ( v34 == v39 )
                      {
                        v38 = &v132[v37];
                        goto LABEL_24;
                      }
                      ++v35;
                      v36 = v38;
                      v38 = &v132[v37];
                    }
                    if ( v36 )
                      v38 = v36;
                  }
LABEL_24:
                  *v38 = v34;
                  LODWORD(v133) = v133 + 1;
                }
              }
              j___libc_free_0(v20);
              v30 = v132;
              v18 = v118;
              v40 = v133 + 1;
            }
            else
            {
              HIDWORD(v133) = 0;
              v106 = &v30[v134];
              if ( v30 != v106 )
              {
                v107 = v30;
                do
                {
                  if ( v107 )
                    *v107 = -8;
                  ++v107;
                }
                while ( v106 != v107 );
              }
              v40 = 1;
            }
            v41 = (v134 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v22 = &v30[v41];
            v42 = *v22;
            if ( v27 != *v22 )
            {
              v111 = 1;
              a14 = 0;
              while ( v42 != -8 )
              {
                if ( v42 == -16 && !a14 )
                  a14 = v22;
                v41 = (v134 - 1) & (v111 + v41);
                v22 = &v30[v41];
                v42 = *v22;
                if ( v27 == *v22 )
                  goto LABEL_28;
                ++v111;
              }
              if ( a14 )
                v22 = a14;
            }
            goto LABEL_28;
          }
          v21 = 1;
          v22 = 0;
          v23 = (unsigned int)(v19 - 1);
          v24 = v23 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v25 = &v20[v24];
          v26 = *v25;
          if ( v27 != *v25 )
            break;
LABEL_10:
          v17 += 24;
          if ( v18 == v17 )
            goto LABEL_35;
        }
        while ( v26 != -8 )
        {
          if ( v22 || v26 != -16 )
            v25 = v22;
          v24 = v23 & (v21 + v24);
          a14 = &v20[v24];
          v26 = *a14;
          if ( v27 == *a14 )
            goto LABEL_10;
          ++v21;
          v22 = v25;
          v25 = &v20[v24];
        }
        if ( !v22 )
          v22 = v25;
        ++v131;
        v40 = v133 + 1;
        if ( 4 * ((int)v133 + 1) >= (unsigned int)(3 * v19) )
          goto LABEL_13;
        if ( (int)v19 - (v40 + HIDWORD(v133)) <= (unsigned int)v19 >> 3 )
        {
          v119 = v18;
          v78 = (((v23 >> 1) | v23) >> 2) | (v23 >> 1) | v23;
          v79 = (((v78 >> 4) | v78) >> 8) | (v78 >> 4) | v78;
          v80 = ((v79 >> 16) | v79) + 1;
          if ( (unsigned int)v80 < 0x40 )
            LODWORD(v80) = 64;
          v134 = v80;
          v81 = (_QWORD *)sub_22077B0(8LL * (unsigned int)v80);
          v18 = v119;
          v132 = v81;
          if ( v20 )
          {
            v82 = &v20[v19];
            v133 = 0;
            for ( k = &v81[v134]; k != v81; ++v81 )
            {
              if ( v81 )
                *v81 = -8;
            }
            v84 = v20;
            do
            {
              v85 = *v84;
              if ( *v84 != -8 && v85 != -16 )
              {
                v86 = (v134 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
                v87 = &v132[v86];
                v88 = *v87;
                if ( v85 != *v87 )
                {
                  v122 = 1;
                  v110 = 0;
                  while ( v88 != -8 )
                  {
                    if ( v88 == -16 && !v110 )
                      v110 = v87;
                    v86 = (v134 - 1) & (v122 + v86);
                    v87 = &v132[v86];
                    v88 = *v87;
                    if ( v85 == *v87 )
                      goto LABEL_80;
                    ++v122;
                  }
                  if ( v110 )
                    v87 = v110;
                }
LABEL_80:
                *v87 = v85;
                LODWORD(v133) = v133 + 1;
              }
              ++v84;
            }
            while ( v82 != v84 );
            v120 = v18;
            j___libc_free_0(v20);
            v81 = v132;
            v89 = v134;
            v18 = v120;
            v40 = v133 + 1;
          }
          else
          {
            HIDWORD(v133) = 0;
            v108 = &v81[v134];
            v89 = v134;
            if ( v81 != v108 )
            {
              v109 = v81;
              do
              {
                if ( v109 )
                  *v109 = -8;
                ++v109;
              }
              while ( v108 != v109 );
            }
            v40 = 1;
          }
          v90 = v89 - 1;
          LODWORD(a14) = 1;
          v91 = 0;
          v92 = v90 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v22 = &v81[v92];
          v93 = *v22;
          if ( v27 != *v22 )
          {
            while ( v93 != -8 )
            {
              if ( v93 == -16 && !v91 )
                v91 = v22;
              v92 = v90 & ((_DWORD)a14 + v92);
              v22 = &v81[v92];
              v93 = *v22;
              if ( v27 == *v22 )
                goto LABEL_28;
              LODWORD(a14) = (_DWORD)a14 + 1;
            }
            if ( v91 )
              v22 = v91;
          }
        }
LABEL_28:
        LODWORD(v133) = v40;
        if ( *v22 != -8 )
          --HIDWORD(v133);
        *v22 = v27;
        v43 = *(_QWORD *)(v17 + 16);
        v44 = v136;
        v138 = (_BYTE *)v43;
        if ( v136 == v137 )
        {
          v121 = v18;
          sub_12879C0((__int64)&v135, v136, &v138);
          v18 = v121;
        }
        else
        {
          if ( v136 )
          {
            *(_QWORD *)v136 = v43;
            v44 = v136;
          }
          v136 = v44 + 8;
        }
        v17 += 24;
        v20 = v132;
        v19 = v134;
      }
      while ( v18 != v17 );
LABEL_35:
      v138 = 0;
      v45 = v20;
      v15 = v127;
      if ( !(_DWORD)v19 )
        goto LABEL_111;
      v46 = *v45;
      v47 = v19 - 1;
      v48 = 1;
      v49 = 0;
      if ( *v45 )
      {
        while ( v46 != -8 )
        {
          v49 = v47 & (v48 + v49);
          v46 = v45[v49];
          if ( !v46 )
          {
            v45 += v49;
            goto LABEL_37;
          }
          ++v48;
        }
LABEL_111:
        v52 = v136;
        goto LABEL_40;
      }
LABEL_37:
      *v45 = -16;
      LODWORD(v133) = v133 - 1;
      ++HIDWORD(v133);
      v50 = sub_1BB95F0(v135, (__int64)v136, (__int64 *)&v138);
      v51 = v50 + 8;
      if ( v136 != v50 + 8 )
      {
        memmove(v50, v51, v136 - v51);
        v51 = v136;
      }
      v52 = v51 - 8;
      v136 = v51 - 8;
LABEL_40:
      v53 = v135;
      v126 = 1;
      v125 = 0;
      v54 = v129 + 64;
      v130 = (__int64 *)(v129 + 40);
      v117 = v54;
      if ( v116 == v123 )
        goto LABEL_113;
      v55 = v52 - (_BYTE *)v135;
      v56 = v52 - (_BYTE *)v135;
      if ( (unsigned __int64)(v52 - (_BYTE *)v135) <= 8 )
        goto LABEL_93;
      while ( 2 )
      {
        if ( v134 )
        {
          LODWORD(a14) = (_DWORD)v132;
          v57 = 1;
          v58 = *(v130 - 3);
          v59 = (v134 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
          v60 = v132[v59];
          if ( v58 == v60 )
          {
LABEL_44:
            v61 = sub_146F1B0(*v15, *(v130 - 3));
            if ( v124 <= v126 )
            {
              v53 = v135;
              v52 = v136;
LABEL_113:
              v56 = v52 - (_BYTE *)v53;
              v100 = (v52 - (_BYTE *)v53) >> 3;
              if ( (unsigned __int64)(v52 - (_BYTE *)v53) > 8 )
                goto LABEL_114;
              goto LABEL_93;
            }
            v52 = v136;
            v62 = v130;
            v128 = (__int64 *)(v117 + 24 * (v125 + (unsigned int)(v124 - 2 - v125)));
            while ( 1 )
            {
              v53 = v135;
              if ( (unsigned __int64)(v52 - (_BYTE *)v135) <= 8 )
                goto LABEL_92;
              v63 = *v62;
              v64 = sub_146F1B0(*v15, *v62);
              if ( *(_WORD *)(sub_14806B0(*v15, v61, v64, 0, 0) + 24) )
              {
                if ( *(_QWORD *)(v63 + 24 * (1LL - (*(_DWORD *)(v63 + 20) & 0xFFFFFFF))) != *(_QWORD *)(v58 + 24 * (1LL - (*(_DWORD *)(v58 + 20) & 0xFFFFFFF))) )
                  goto LABEL_47;
                v138 = (_BYTE *)*v62;
                v95 = (v134 - 1) & (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4));
                v96 = &v132[v95];
                v97 = (_BYTE *)*v96;
                if ( v138 != (_BYTE *)*v96 )
                {
                  v105 = 1;
                  while ( v97 != (_BYTE *)-8LL )
                  {
                    LODWORD(a14) = v105 + 1;
                    v95 = (v134 - 1) & (v105 + v95);
                    v96 = &v132[v95];
                    v97 = (_BYTE *)*v96;
                    if ( v138 == (_BYTE *)*v96 )
                      goto LABEL_100;
                    v105 = (int)a14;
                  }
LABEL_47:
                  v52 = v136;
                  v62 += 3;
                  if ( v128 == v62 )
                    goto LABEL_59;
                  continue;
                }
LABEL_100:
                *v96 = -16;
              }
              else
              {
                v65 = v132;
                v138 = (_BYTE *)*(v130 - 3);
                v66 = v134 - 1;
                v67 = (v134 - 1) & (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4));
                v68 = &v132[v67];
                v69 = (_BYTE *)*v68;
                if ( v138 == (_BYTE *)*v68 )
                {
LABEL_51:
                  *v68 = -16;
                  LODWORD(v133) = v133 - 1;
                  ++HIDWORD(v133);
                  v70 = sub_1BB95F0(v135, (__int64)v136, (__int64 *)&v138);
                  v71 = v70 + 8;
                  if ( v136 != v70 + 8 )
                  {
                    memmove(v70, v71, v136 - v71);
                    v71 = v136;
                  }
                  v136 = v71 - 8;
                  v72 = *v62;
                  v65 = v132;
                  v138 = (_BYTE *)*v62;
                  v66 = v134 - 1;
                }
                else
                {
                  v99 = 1;
                  while ( v69 != (_BYTE *)-8LL )
                  {
                    LODWORD(a14) = v99 + 1;
                    v67 = v66 & (v99 + v67);
                    v68 = &v132[v67];
                    v69 = (_BYTE *)*v68;
                    if ( v138 == (_BYTE *)*v68 )
                      goto LABEL_51;
                    v99 = (int)a14;
                  }
                  v72 = *v62;
                  v138 = (_BYTE *)*v62;
                }
                v73 = v66 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
                v74 = &v65[v73];
                v75 = *v74;
                if ( *v74 != v72 )
                {
                  v98 = 1;
                  while ( v75 != -8 )
                  {
                    LODWORD(a14) = v98 + 1;
                    v73 = v66 & (v98 + v73);
                    v74 = &v65[v73];
                    v75 = *v74;
                    if ( *v74 == v72 )
                      goto LABEL_55;
                    v98 = (int)a14;
                  }
                  goto LABEL_47;
                }
LABEL_55:
                *v74 = -16;
              }
              LODWORD(v133) = v133 - 1;
              ++HIDWORD(v133);
              v76 = sub_1BB95F0(v135, (__int64)v136, (__int64 *)&v138);
              v77 = v76 + 8;
              if ( v136 != v76 + 8 )
              {
                memmove(v76, v77, v136 - v77);
                v77 = v136;
              }
              v52 = v77 - 8;
              v62 += 3;
              v136 = v77 - 8;
              if ( v128 == v62 )
              {
LABEL_59:
                v53 = v135;
                goto LABEL_92;
              }
            }
          }
          while ( v60 != -8 )
          {
            v59 = (v134 - 1) & (v57 + v59);
            v60 = v132[v59];
            if ( v58 == v60 )
              goto LABEL_44;
            ++v57;
          }
        }
        if ( v124 > v126 )
        {
LABEL_92:
          ++v125;
          v55 = v52 - (_BYTE *)v53;
          ++v126;
          v130 += 3;
          v56 = v52 - (_BYTE *)v53;
          if ( (unsigned __int64)(v52 - (_BYTE *)v53) <= 8 )
            goto LABEL_93;
          continue;
        }
        break;
      }
      v100 = v55 >> 3;
LABEL_114:
      v138 = s;
      v101 = s;
      v139 = 0x1000000000LL;
      if ( v56 > 0x80 )
      {
        sub_16CD150((__int64)&v138, s, v100, 8, (int)&v138, (int)a14);
        v101 = v138;
      }
      LODWORD(v139) = v100;
      if ( 8LL * (unsigned int)v100 )
      {
        memset(v101, 0, 8LL * (unsigned int)v100);
        v101 = v138;
      }
      v102 = v135;
      if ( v136 != (_BYTE *)v135 )
      {
        v103 = 0;
        v104 = (unsigned __int64)(v136 - 8 - (_BYTE *)v135) >> 3;
        while ( 1 )
        {
          *(_QWORD *)&v101[8 * (unsigned int)v103] = *(_QWORD *)(v102[v103]
                                                               + 24
                                                               * (1LL - (*(_DWORD *)(v102[v103] + 20LL) & 0xFFFFFFF)));
          v101 = v138;
          if ( v104 == v103 )
            break;
          ++v103;
        }
      }
      v115 |= sub_1BDB410(
                (__int64)v15,
                (__int64 ***)v101,
                (unsigned int)v139,
                a11,
                0,
                0,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                a8,
                a9);
      if ( v138 != s )
        _libc_free((unsigned __int64)v138);
      v53 = v135;
LABEL_93:
      if ( v53 )
        j_j___libc_free_0(v53, v137 - (_BYTE *)v53);
      j___libc_free_0(v132);
      v123 += 16;
    }
    while ( v123 < v116 );
    v14 = v15;
LABEL_97:
    v114 += 216;
  }
  while ( v112 != v114 );
  return v115;
}
