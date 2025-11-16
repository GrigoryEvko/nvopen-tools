// Function: sub_1F97850
// Address: 0x1f97850
//
__int64 *__fastcall sub_1F97850(
        __int64 **a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 *v12; // r14
  unsigned int v13; // ecx
  __m128 v14; // xmm0
  __m128i v15; // xmm1
  unsigned int v16; // r12d
  __int64 v17; // r13
  __int64 v18; // r15
  __int16 v19; // dx
  __int64 v20; // rax
  char v21; // di
  const void **v22; // rax
  int v23; // eax
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // r10
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 *v30; // rcx
  unsigned int v31; // edx
  __int16 v32; // ax
  bool v33; // al
  __int64 *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rax
  char v38; // r11
  __int64 v39; // rdx
  __int64 *v40; // rsi
  __int64 v41; // rcx
  __int64 v42; // rdx
  char v43; // r15
  const void **v44; // rdx
  unsigned int v45; // r8d
  unsigned int v46; // r8d
  int v47; // r10d
  char v48; // r11
  unsigned int v49; // r15d
  unsigned int v50; // ecx
  unsigned int v51; // r8d
  int v52; // r9d
  int v53; // r10d
  unsigned int v54; // r15d
  __int64 v55; // rax
  char *v56; // rdx
  char *v57; // rdi
  char *v58; // rsi
  unsigned int v59; // ecx
  __int64 v60; // r9
  unsigned int v61; // edx
  bool v62; // al
  __int64 v63; // rcx
  __int64 v64; // rdx
  int v65; // eax
  __int64 v66; // rdx
  _QWORD *v67; // rax
  __int64 *v68; // r12
  __int16 *v69; // rdx
  __int16 *v70; // r13
  __int64 v71; // rsi
  __int64 *v72; // r10
  __int64 v73; // rbx
  unsigned __int8 v74; // al
  unsigned int v75; // eax
  unsigned int v76; // ecx
  __int64 *v77; // rdi
  __int64 (*v78)(); // rax
  __int64 v79; // rsi
  _QWORD *v80; // rax
  int v81; // r10d
  _QWORD *v82; // r9
  int v83; // edx
  int v84; // ebx
  __int64 v85; // rdx
  int v86; // r8d
  _BYTE *v87; // rax
  _BYTE *v88; // rdx
  __int64 v89; // rax
  __int64 *v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r15
  __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 v95; // rax
  bool v96; // al
  __int64 *v97; // rax
  __int64 v98; // rcx
  __int64 v99; // rax
  __int64 v100; // rax
  const void **v101; // rax
  unsigned int v102; // eax
  bool v103; // al
  int v104; // r8d
  unsigned int v105; // r12d
  __int64 *v106; // rdi
  unsigned __int8 *v107; // rax
  __int64 v108; // r8
  unsigned int v109; // ecx
  _QWORD *v110; // rax
  int v111; // edx
  int v112; // r8d
  int v113; // r9d
  _QWORD *v114; // r14
  int v115; // eax
  __int64 v116; // rdx
  __int64 v117; // r13
  int v118; // ecx
  unsigned int *v119; // r15
  _BYTE *v120; // rdx
  unsigned __int64 v121; // rcx
  __int64 v122; // r13
  __int64 v123; // rax
  char v124; // dl
  const void **v125; // rax
  bool v126; // al
  __int32 v127; // edx
  __int64 v128; // rsi
  unsigned int *v129; // r15
  unsigned int v130; // eax
  __int64 v131; // rdx
  unsigned int v132; // eax
  unsigned int v133; // eax
  const __m128i *v134; // r12
  __int64 v135; // rdx
  __int64 v136; // r14
  const __m128i *v137; // r13
  unsigned __int64 v138; // r14
  __m128 *v139; // rax
  const void **v140; // rdx
  __int64 v141; // rax
  char v142; // al
  char v143; // r8
  __int128 v144; // [rsp-10h] [rbp-230h]
  __int128 v145; // [rsp-10h] [rbp-230h]
  _QWORD *v146; // [rsp+8h] [rbp-218h]
  __int64 v147; // [rsp+10h] [rbp-210h]
  _QWORD *v148; // [rsp+18h] [rbp-208h]
  _QWORD *v149; // [rsp+18h] [rbp-208h]
  int v150; // [rsp+18h] [rbp-208h]
  __int64 v151; // [rsp+20h] [rbp-200h]
  __int64 v152; // [rsp+28h] [rbp-1F8h]
  int v153; // [rsp+30h] [rbp-1F0h]
  int v154; // [rsp+30h] [rbp-1F0h]
  int v155; // [rsp+30h] [rbp-1F0h]
  __int64 v156; // [rsp+30h] [rbp-1F0h]
  unsigned int v157; // [rsp+38h] [rbp-1E8h]
  unsigned int v158; // [rsp+38h] [rbp-1E8h]
  unsigned int v159; // [rsp+38h] [rbp-1E8h]
  int v160; // [rsp+38h] [rbp-1E8h]
  unsigned int v161; // [rsp+40h] [rbp-1E0h]
  int v162; // [rsp+40h] [rbp-1E0h]
  __int64 v163; // [rsp+40h] [rbp-1E0h]
  unsigned int v164; // [rsp+40h] [rbp-1E0h]
  unsigned int s; // [rsp+48h] [rbp-1D8h]
  char sa; // [rsp+48h] [rbp-1D8h]
  int v167; // [rsp+50h] [rbp-1D0h]
  const void **v168; // [rsp+50h] [rbp-1D0h]
  int v169; // [rsp+50h] [rbp-1D0h]
  int v170; // [rsp+50h] [rbp-1D0h]
  int v171; // [rsp+50h] [rbp-1D0h]
  int v172; // [rsp+50h] [rbp-1D0h]
  int v173; // [rsp+50h] [rbp-1D0h]
  int v174; // [rsp+58h] [rbp-1C8h]
  _QWORD *v175; // [rsp+58h] [rbp-1C8h]
  int v176; // [rsp+60h] [rbp-1C0h]
  __int64 v177; // [rsp+60h] [rbp-1C0h]
  int v178; // [rsp+60h] [rbp-1C0h]
  __int64 v179; // [rsp+60h] [rbp-1C0h]
  __int64 v180; // [rsp+60h] [rbp-1C0h]
  __int64 v181; // [rsp+60h] [rbp-1C0h]
  __int64 v182; // [rsp+60h] [rbp-1C0h]
  __int64 v183; // [rsp+68h] [rbp-1B8h]
  unsigned int v184; // [rsp+80h] [rbp-1A0h]
  unsigned int v185; // [rsp+80h] [rbp-1A0h]
  __int64 v186; // [rsp+80h] [rbp-1A0h]
  int v187; // [rsp+80h] [rbp-1A0h]
  int v188; // [rsp+80h] [rbp-1A0h]
  int v189; // [rsp+88h] [rbp-198h]
  unsigned int v190; // [rsp+88h] [rbp-198h]
  int v191; // [rsp+88h] [rbp-198h]
  int v192; // [rsp+88h] [rbp-198h]
  int v193; // [rsp+88h] [rbp-198h]
  int v194; // [rsp+88h] [rbp-198h]
  int v195; // [rsp+88h] [rbp-198h]
  unsigned int v196; // [rsp+A0h] [rbp-180h]
  int v197; // [rsp+A0h] [rbp-180h]
  int v198; // [rsp+A0h] [rbp-180h]
  __int64 *v200; // [rsp+A8h] [rbp-178h]
  __int64 v201; // [rsp+C0h] [rbp-160h] BYREF
  int v202; // [rsp+C8h] [rbp-158h]
  __int64 v203; // [rsp+D0h] [rbp-150h] BYREF
  const void **v204; // [rsp+D8h] [rbp-148h]
  __int64 v205; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v206; // [rsp+E8h] [rbp-138h]
  unsigned int v207; // [rsp+F0h] [rbp-130h] BYREF
  const void **v208; // [rsp+F8h] [rbp-128h]
  __int64 v209; // [rsp+100h] [rbp-120h] BYREF
  int v210; // [rsp+108h] [rbp-118h]
  char *v211; // [rsp+110h] [rbp-110h] BYREF
  __int64 v212; // [rsp+118h] [rbp-108h]
  _BYTE v213[64]; // [rsp+120h] [rbp-100h] BYREF
  _BYTE *v214; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v215; // [rsp+168h] [rbp-B8h]
  _BYTE v216[176]; // [rsp+170h] [rbp-B0h] BYREF

  v10 = *(_QWORD *)(a2 + 32);
  v11 = *(_QWORD *)(a2 + 72);
  v12 = *(__int64 **)v10;
  v13 = *(_DWORD *)(v10 + 48);
  v201 = v11;
  v14 = (__m128)_mm_loadu_si128((const __m128i *)(v10 + 40));
  v15 = _mm_loadu_si128((const __m128i *)(v10 + 80));
  v16 = *(_DWORD *)(v10 + 8);
  v17 = *(_QWORD *)(v10 + 40);
  v196 = v13;
  v18 = *(_QWORD *)(v10 + 80);
  v189 = *(_DWORD *)(v10 + 88);
  if ( v11 )
    sub_1623A60((__int64)&v201, v11, 2);
  v19 = *(_WORD *)(v17 + 24);
  v202 = *(_DWORD *)(a2 + 64);
  if ( v19 != 48 )
  {
    v20 = v12[5] + 16LL * v16;
    v21 = *(_BYTE *)v20;
    v22 = *(const void ***)(v20 + 8);
    LOBYTE(v203) = v21;
    v204 = v22;
    if ( v19 != 106
      || (v25 = *(_QWORD *)(v17 + 32), v12 != *(__int64 **)v25)
      || *(_DWORD *)(v25 + 8) != v16
      || v18 != *(_QWORD *)(v25 + 40)
      || v189 != *(_DWORD *)(v25 + 48) )
    {
      v23 = *(unsigned __int16 *)(v18 + 24);
      if ( v23 != 32 && v23 != 10 )
        goto LABEL_7;
      v26 = *(_QWORD *)(v18 + 88);
      v27 = *(_QWORD **)(v26 + 24);
      if ( *(_DWORD *)(v26 + 32) > 0x40u )
        v27 = (_QWORD *)*v27;
      v28 = *(_QWORD *)(a2 + 32);
      v190 = (unsigned int)v27;
      v29 = *(_QWORD *)(v28 + 40);
      if ( *(_WORD *)(v29 + 24) != 158 )
        goto LABEL_19;
      v176 = (int)v27;
      v33 = sub_1D18C00(*(_QWORD *)(v28 + 40), 1, *(_DWORD *)(v28 + 48));
      LODWORD(v27) = v176;
      if ( !v33 )
        goto LABEL_19;
      v34 = *(__int64 **)(v29 + 32);
      v35 = *v34;
      v36 = *((unsigned int *)v34 + 2);
      v174 = v36;
      v37 = *(_QWORD *)(v35 + 40) + 16 * v36;
      v38 = *(_BYTE *)v37;
      v39 = *(_QWORD *)(v37 + 8);
      v177 = v35;
      LOBYTE(v214) = v38;
      v215 = v39;
      if ( v38 )
      {
        if ( (unsigned __int8)(v38 - 14) > 0x5Fu )
          goto LABEL_19;
        v40 = *(__int64 **)(a2 + 32);
        v152 = *v40;
        v151 = v40[1];
        v41 = *v40;
        LODWORD(v40) = *((_DWORD *)v40 + 2);
        v206 = v39;
        LOBYTE(v205) = v38;
        v42 = *(_QWORD *)(v41 + 40) + 16LL * (unsigned int)v40;
        v43 = *(_BYTE *)v42;
        v44 = *(const void ***)(v42 + 8);
        LOBYTE(v207) = v43;
        v208 = v44;
        v45 = word_42FA680[(unsigned __int8)(v38 - 14)];
      }
      else
      {
        v163 = v39;
        v169 = (int)v27;
        v96 = sub_1F58D20((__int64)&v214);
        LODWORD(v27) = v169;
        if ( !v96 )
          goto LABEL_19;
        v97 = *(__int64 **)(a2 + 32);
        v152 = *v97;
        v151 = v97[1];
        v98 = *v97;
        v99 = *((unsigned int *)v97 + 2);
        LOBYTE(v205) = 0;
        v206 = v163;
        v100 = *(_QWORD *)(v98 + 40) + 16 * v99;
        v43 = *(_BYTE *)v100;
        v101 = *(const void ***)(v100 + 8);
        LOBYTE(v207) = v43;
        v208 = v101;
        v102 = sub_1F58D30((__int64)&v205);
        LODWORD(v27) = v169;
        v38 = 0;
        v45 = v102;
      }
      if ( v43 )
      {
        v49 = sub_1F6C8D0(v43);
      }
      else
      {
        v164 = v45;
        sa = v38;
        v172 = (int)v27;
        v133 = sub_1F58D40((__int64)&v207);
        v46 = v164;
        v48 = sa;
        v47 = v172;
        v49 = v133;
      }
      if ( v48 )
      {
        v50 = sub_1F6C8D0(v48);
      }
      else
      {
        s = v46;
        v171 = v47;
        v132 = sub_1F58D40((__int64)&v205);
        v51 = s;
        v53 = v171;
        v50 = v132;
      }
      v161 = v49 / v50;
      v54 = v51 * (v49 / v50);
      v211 = v213;
      v212 = 0x1000000000LL;
      v55 = 4LL * v54;
      if ( v54 > 0x10uLL )
      {
        v156 = 4LL * v54;
        v159 = v51;
        v173 = v53;
        sub_16CD150((__int64)&v211, v213, v54, 4, v51, v52);
        v58 = v211;
        LODWORD(v212) = v54;
        v53 = v173;
        v51 = v159;
        v56 = &v211[v156];
        v57 = v211;
        if ( v211 == &v211[v156] )
        {
LABEL_41:
          v59 = 0;
          while ( 1 )
          {
            v60 = v59;
            v61 = v54 + v59 % v51;
            if ( v190 != v59 / v51 )
              v61 = v59;
            ++v59;
            *(_DWORD *)&v58[4 * v60] = v61;
            if ( v54 == v59 )
              break;
            v58 = v211;
          }
LABEL_46:
          if ( (_BYTE)v205 )
          {
            switch ( (char)v205 )
            {
              case 14:
              case 15:
              case 16:
              case 17:
              case 18:
              case 19:
              case 20:
              case 21:
              case 22:
              case 23:
              case 56:
              case 57:
              case 58:
              case 59:
              case 60:
              case 61:
                v74 = 2;
                break;
              case 24:
              case 25:
              case 26:
              case 27:
              case 28:
              case 29:
              case 30:
              case 31:
              case 32:
              case 62:
              case 63:
              case 64:
              case 65:
              case 66:
              case 67:
                v74 = 3;
                break;
              case 33:
              case 34:
              case 35:
              case 36:
              case 37:
              case 38:
              case 39:
              case 40:
              case 68:
              case 69:
              case 70:
              case 71:
              case 72:
              case 73:
                v74 = 4;
                break;
              case 41:
              case 42:
              case 43:
              case 44:
              case 45:
              case 46:
              case 47:
              case 48:
              case 74:
              case 75:
              case 76:
              case 77:
              case 78:
              case 79:
                v74 = 5;
                break;
              case 49:
              case 50:
              case 51:
              case 52:
              case 53:
              case 54:
              case 80:
              case 81:
              case 82:
              case 83:
              case 84:
              case 85:
                v74 = 6;
                break;
              case 55:
                v74 = 7;
                break;
              case 86:
              case 87:
              case 88:
              case 98:
              case 99:
              case 100:
                v74 = 8;
                break;
              case 89:
              case 90:
              case 91:
              case 92:
              case 93:
              case 101:
              case 102:
              case 103:
              case 104:
              case 105:
                v74 = 9;
                break;
              case 94:
              case 95:
              case 96:
              case 97:
              case 106:
              case 107:
              case 108:
              case 109:
                v74 = 10;
                break;
            }
            v147 = 0;
          }
          else
          {
            v170 = v53;
            v74 = sub_1F596B0((__int64)&v205);
            v53 = v170;
            v147 = v131;
          }
          v153 = v53;
          v158 = v74;
          v148 = (_QWORD *)(*a1)[6];
          LOBYTE(v75) = sub_1D15020(v74, v54);
          v168 = 0;
          LODWORD(v27) = v153;
          if ( !(_BYTE)v75 )
          {
            v75 = sub_1F593D0(v148, v158, v147, v54);
            LODWORD(v27) = v153;
            v184 = v75;
            v168 = v140;
          }
          v76 = v184;
          LOBYTE(v76) = v75;
          v185 = v76;
          v77 = a1[1];
          v78 = *(__int64 (**)())(*v77 + 336);
          if ( v78 == sub_1F3CA80
            || (v160 = (int)v27,
                v142 = ((__int64 (__fastcall *)(__int64 *, char *, _QWORD, _QWORD, const void **))v78)(
                         v77,
                         v211,
                         (unsigned int)v212,
                         v76,
                         v168),
                LODWORD(v27) = v160,
                v143 = v142,
                v95 = 0,
                v143) )
          {
            v79 = *(_QWORD *)(a2 + 72);
            v209 = v79;
            if ( v79 )
            {
              v154 = (int)v27;
              sub_1623A60((__int64)&v209, v79, 2);
              LODWORD(v27) = v154;
            }
            v155 = (int)v27;
            v210 = *(_DWORD *)(a2 + 64);
            v214 = 0;
            LODWORD(v215) = 0;
            v80 = sub_1D2B300(*a1, 0x30u, (__int64)&v214, v205, v206, a9);
            v81 = v155;
            v82 = v80;
            v84 = v83;
            if ( v214 )
            {
              v149 = v80;
              sub_161E7C0((__int64)&v214, (__int64)v214);
              v82 = v149;
              v81 = v155;
            }
            v85 = v161;
            v86 = v84;
            v215 = 0x800000000LL;
            v87 = v216;
            v214 = v216;
            if ( v161 > 8 )
            {
              v146 = v82;
              v150 = v81;
              sub_16CD150((__int64)&v214, v216, v161, 16, v84, (int)v82);
              v87 = v214;
              v82 = v146;
              v86 = v84;
              v81 = v150;
              v85 = v161;
            }
            v88 = &v87[16 * v85];
            for ( LODWORD(v215) = v161; v88 != v87; v87 += 16 )
            {
              if ( v87 )
              {
                *(_QWORD *)v87 = v82;
                *((_DWORD *)v87 + 2) = v86;
              }
            }
            v89 = (__int64)v214;
            v162 = v81;
            *(_QWORD *)v214 = v177;
            *(_DWORD *)(v89 + 8) = v174;
            *((_QWORD *)&v144 + 1) = (unsigned int)v215;
            *(_QWORD *)&v144 = v214;
            v90 = sub_1D359D0(
                    *a1,
                    107,
                    (__int64)&v209,
                    v185,
                    v168,
                    0,
                    *(double *)v14.m128_u64,
                    *(double *)v15.m128i_i64,
                    a5,
                    v144);
            v183 = v91;
            v179 = (__int64)v90;
            v92 = sub_1D32840(
                    *a1,
                    v185,
                    v168,
                    v152,
                    v151,
                    *(double *)v14.m128_u64,
                    *(double *)v15.m128i_i64,
                    *(double *)a5.m128i_i64);
            v175 = sub_1D41320(
                     (__int64)*a1,
                     v185,
                     v168,
                     (__int64)&v209,
                     v92,
                     v93,
                     *(double *)v14.m128_u64,
                     *(double *)v15.m128i_i64,
                     a5,
                     v179,
                     v183,
                     v211,
                     (unsigned int)v212);
            v186 = v94;
            sub_1F81BC0((__int64)a1, v179);
            sub_1F81BC0((__int64)a1, v92);
            sub_1F81BC0((__int64)a1, (__int64)v175);
            v95 = sub_1D32840(
                    *a1,
                    v207,
                    v208,
                    (__int64)v175,
                    v186,
                    *(double *)v14.m128_u64,
                    *(double *)v15.m128i_i64,
                    *(double *)a5.m128i_i64);
            LODWORD(v27) = v162;
            if ( v214 != v216 )
            {
              v180 = v95;
              _libc_free((unsigned __int64)v214);
              LODWORD(v27) = v162;
              v95 = v180;
            }
            if ( v209 )
            {
              v187 = (int)v27;
              v181 = v95;
              sub_161E7C0((__int64)&v209, v209);
              LODWORD(v27) = v187;
              v95 = v181;
            }
          }
          if ( v211 != v213 )
          {
            v188 = (int)v27;
            v182 = v95;
            _libc_free((unsigned __int64)v211);
            LODWORD(v27) = v188;
            v95 = v182;
          }
          if ( v95 )
          {
            v12 = (__int64 *)v95;
            goto LABEL_8;
          }
LABEL_19:
          if ( *((_WORD *)v12 + 12) == 105 )
          {
            v178 = (int)v27;
            v62 = sub_1D18C00((__int64)v12, 1, v16);
            LODWORD(v27) = v178;
            if ( v62 )
            {
              v63 = v12[4];
              v64 = *(_QWORD *)(v63 + 80);
              v65 = *(unsigned __int16 *)(v64 + 24);
              if ( v65 == 10 || v65 == 32 )
              {
                v66 = *(_QWORD *)(v64 + 88);
                v67 = *(_QWORD **)(v66 + 24);
                if ( *(_DWORD *)(v66 + 32) > 0x40u )
                  v67 = (_QWORD *)*v67;
                if ( v190 < (unsigned int)v67 )
                {
                  v68 = sub_1D3A900(
                          *a1,
                          0x69u,
                          (__int64)&v201,
                          (unsigned int)v203,
                          v204,
                          0,
                          v14,
                          *(double *)v15.m128i_i64,
                          a5,
                          *(_QWORD *)v63,
                          *(__int16 **)(v63 + 8),
                          *(_OWORD *)&v14,
                          v15.m128i_i64[0],
                          v15.m128i_i64[1]);
                  v70 = v69;
                  sub_1F81BC0((__int64)a1, (__int64)v68);
                  v71 = v12[9];
                  v72 = *a1;
                  v73 = v12[4];
                  v214 = (_BYTE *)v71;
                  if ( v71 )
                  {
                    v200 = v72;
                    sub_1623A60((__int64)&v214, v71, 2);
                    v72 = v200;
                  }
                  LODWORD(v215) = *((_DWORD *)v12 + 16);
                  v12 = sub_1D3A900(
                          v72,
                          0x69u,
                          (__int64)&v214,
                          (unsigned int)v203,
                          v204,
                          0,
                          v14,
                          *(double *)v15.m128i_i64,
                          a5,
                          (unsigned __int64)v68,
                          v70,
                          *(_OWORD *)(v73 + 40),
                          *(_QWORD *)(v73 + 80),
                          *(_QWORD *)(v73 + 88));
                  if ( v214 )
                    sub_161E7C0((__int64)&v214, (__int64)v214);
                  goto LABEL_8;
                }
              }
            }
          }
          if ( *((_BYTE *)a1 + 24) )
          {
            if ( (v30 = a1[1], v31 = 1, (_BYTE)v203 != 1)
              && (!(_BYTE)v203 || (v31 = (unsigned __int8)v203, !v30[(unsigned __int8)v203 + 15]))
              || *((_BYTE *)v30 + 259 * v31 + 2526) )
            {
LABEL_7:
              v12 = 0;
              goto LABEL_8;
            }
          }
          v214 = v216;
          v215 = 0x800000000LL;
          v32 = *((_WORD *)v12 + 12);
          if ( v32 == 104 )
          {
            v191 = (int)v27;
            v103 = sub_1D18C00((__int64)v12, 1, v16);
            LODWORD(v27) = v191;
            if ( v103 )
            {
              v134 = (const __m128i *)v12[4];
              v135 = (unsigned int)v215;
              v136 = 40LL * *((unsigned int *)v12 + 14);
              v137 = (const __m128i *)((char *)v134 + v136);
              v138 = 0xCCCCCCCCCCCCCCCDLL * (v136 >> 3);
              if ( v138 > HIDWORD(v215) - (unsigned __int64)(unsigned int)v215 )
              {
                sub_16CD150((__int64)&v214, v216, v138 + (unsigned int)v215, 16, v104, a9);
                v135 = (unsigned int)v215;
                LODWORD(v27) = v191;
              }
              v119 = (unsigned int *)v214;
              v139 = (__m128 *)&v214[16 * v135];
              if ( v134 != v137 )
              {
                do
                {
                  if ( v139 )
                  {
                    a5 = _mm_loadu_si128(v134);
                    *v139 = (__m128)a5;
                  }
                  v134 = (const __m128i *)((char *)v134 + 40);
                  ++v139;
                }
                while ( v137 != v134 );
                v119 = (unsigned int *)v214;
                LODWORD(v135) = v215;
              }
              LODWORD(v215) = v138 + v135;
              v121 = (unsigned int)(v138 + v135);
              goto LABEL_105;
            }
            v32 = *((_WORD *)v12 + 12);
          }
          v12 = 0;
          if ( v32 != 48 )
          {
LABEL_27:
            if ( v214 != v216 )
              _libc_free((unsigned __int64)v214);
            goto LABEL_8;
          }
          if ( (_BYTE)v203 )
          {
            v105 = word_42FA680[(unsigned __int8)(v203 - 14)];
          }
          else
          {
            v194 = (int)v27;
            v130 = sub_1F58D30((__int64)&v203);
            LODWORD(v27) = v194;
            v105 = v130;
          }
          v192 = (int)v27;
          v106 = *a1;
          v107 = (unsigned __int8 *)(*(_QWORD *)(v17 + 40) + 16LL * v196);
          v108 = *((_QWORD *)v107 + 1);
          v109 = *v107;
          v211 = 0;
          LODWORD(v212) = 0;
          v110 = sub_1D2B300(v106, 0x30u, (__int64)&v211, v109, v108, a9);
          LODWORD(v27) = v192;
          v114 = v110;
          v115 = v111;
          if ( v211 )
          {
            v197 = v192;
            v193 = v111;
            sub_161E7C0((__int64)&v211, (__int64)v211);
            v115 = v193;
            LODWORD(v27) = v197;
          }
          v116 = (unsigned int)v215;
          v117 = v105;
          v118 = v215;
          if ( v105 > HIDWORD(v215) - (unsigned __int64)(unsigned int)v215 )
          {
            v195 = v115;
            v198 = (int)v27;
            sub_16CD150((__int64)&v214, v216, v105 + (unsigned __int64)(unsigned int)v215, 16, v112, v113);
            v116 = (unsigned int)v215;
            v115 = v195;
            LODWORD(v27) = v198;
            v118 = v215;
          }
          v119 = (unsigned int *)v214;
          v120 = &v214[16 * v116];
          if ( v105 )
          {
            do
            {
              if ( v120 )
              {
                *(_QWORD *)v120 = v114;
                *((_DWORD *)v120 + 2) = v115;
              }
              v120 += 16;
              --v117;
            }
            while ( v117 );
            v118 = v215;
            v119 = (unsigned int *)v214;
          }
          v121 = v105 + v118;
          LODWORD(v215) = v121;
LABEL_105:
          v122 = (unsigned int)v27;
          if ( (unsigned int)v27 < v121 )
          {
            v123 = *(_QWORD *)(*(_QWORD *)v119 + 40LL) + 16LL * v119[2];
            v124 = *(_BYTE *)v123;
            v125 = *(const void ***)(v123 + 8);
            LOBYTE(v211) = v124;
            v212 = (__int64)v125;
            if ( v124 )
              v126 = (unsigned __int8)(v124 - 14) <= 0x47u || (unsigned __int8)(v124 - 2) <= 5u;
            else
              v126 = sub_1F58CF0((__int64)&v211);
            v127 = v14.m128_i32[2];
            v128 = v14.m128_u64[0];
            if ( v126 )
            {
              v141 = sub_1D321C0(
                       *a1,
                       v14.m128_i64[0],
                       v14.m128_i64[1],
                       (__int64)&v201,
                       (unsigned int)v211,
                       (const void **)v212,
                       *(double *)v14.m128_u64,
                       *(double *)v15.m128i_i64,
                       *(double *)a5.m128i_i64);
              v119 = (unsigned int *)v214;
              v128 = v141;
            }
            v129 = &v119[4 * v122];
            *(_QWORD *)v129 = v128;
            v129[2] = v127;
            v119 = (unsigned int *)v214;
            v121 = (unsigned int)v215;
          }
          *((_QWORD *)&v145 + 1) = v121;
          *(_QWORD *)&v145 = v119;
          v12 = sub_1D359D0(
                  *a1,
                  104,
                  (__int64)&v201,
                  v203,
                  v204,
                  0,
                  *(double *)v14.m128_u64,
                  *(double *)v15.m128i_i64,
                  a5,
                  v145);
          goto LABEL_27;
        }
      }
      else
      {
        LODWORD(v212) = v54;
        v56 = &v213[v55];
        v57 = v213;
        if ( &v213[v55] == v213 )
          goto LABEL_40;
      }
      v157 = v51;
      v167 = v53;
      memset(v57, 0, v56 - v57);
      v53 = v167;
      v51 = v157;
LABEL_40:
      v58 = v211;
      if ( !v54 )
        goto LABEL_46;
      goto LABEL_41;
    }
  }
LABEL_8:
  if ( v201 )
    sub_161E7C0((__int64)&v201, v201);
  return v12;
}
