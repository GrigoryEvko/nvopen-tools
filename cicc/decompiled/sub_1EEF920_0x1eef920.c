// Function: sub_1EEF920
// Address: 0x1eef920
//
__int64 __fastcall sub_1EEF920(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 *v16; // rsi
  __int64 v18; // r12
  __int64 v19; // rax
  unsigned __int8 *v20; // rsi
  __int64 v21; // r13
  __int64 **v22; // rsi
  __int64 v23; // r14
  __int64 v24; // r15
  __int64 v25; // rdi
  __int64 v26; // rsi
  unsigned __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r13
  _QWORD *v32; // rax
  __int64 v33; // r15
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int8 *v38; // rsi
  __int64 v39; // r15
  unsigned int v40; // eax
  unsigned int v41; // edx
  __int64 v42; // rdi
  __int64 **v43; // r13
  __int64 v44; // r14
  unsigned __int8 v45; // al
  _QWORD *v46; // rax
  _QWORD *v47; // r13
  unsigned __int64 *v48; // r14
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  __int64 v52; // rdx
  unsigned __int8 *v53; // rsi
  _QWORD *v54; // rax
  _QWORD *v55; // r13
  unsigned __int64 *v56; // r14
  __int64 v57; // rax
  unsigned __int64 v58; // rcx
  __int64 v59; // rsi
  __int64 v60; // rdx
  unsigned __int8 *v61; // rsi
  __int64 v62; // rsi
  double v63; // xmm4_8
  double v64; // xmm5_8
  int v66; // eax
  unsigned int v67; // eax
  __int64 v68; // rdi
  __int64 v69; // r8
  __int64 v70; // rsi
  unsigned __int64 v71; // r10
  _QWORD *v72; // rax
  __int64 v73; // rax
  int v74; // eax
  __int64 v75; // rax
  __int64 *v76; // r13
  __int64 v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rsi
  unsigned __int8 *v80; // rsi
  __int64 v81; // rax
  __int64 *v82; // r15
  __int64 v83; // rax
  __int64 v84; // rcx
  __int64 v85; // rsi
  unsigned __int8 *v86; // rsi
  __int64 v87; // rax
  __int64 *v88; // r14
  __int64 v89; // rax
  __int64 v90; // rcx
  __int64 v91; // rsi
  unsigned __int8 *v92; // rsi
  unsigned int v93; // edx
  __int64 v94; // rax
  __int64 v95; // rcx
  __int64 v96; // rax
  __int64 v97; // rsi
  __int64 v98; // rdx
  unsigned __int8 *v99; // rsi
  __int64 v100; // rax
  __int64 *v101; // r13
  __int64 v102; // rax
  __int64 v103; // rcx
  __int64 v104; // rsi
  unsigned __int8 *v105; // rsi
  __int64 v106; // rax
  __int64 *v107; // r13
  __int64 v108; // rax
  __int64 v109; // rcx
  __int64 v110; // rsi
  unsigned __int8 *v111; // rsi
  __int64 v112; // rax
  __int64 *v113; // r14
  __int64 v114; // rax
  __int64 v115; // rcx
  __int64 v116; // rsi
  unsigned __int8 *v117; // rsi
  __int64 v118; // rbx
  __int64 v119; // r13
  __int64 v120; // rcx
  __int64 j; // r12
  __int64 v122; // rax
  __int64 v123; // rax
  _QWORD *v124; // r14
  int v125; // eax
  __int64 v126; // rax
  __int64 v127; // rcx
  __int64 v128; // rax
  unsigned __int8 *v129; // rsi
  unsigned __int8 *v130; // rsi
  __int64 v131; // r11
  _QWORD *v132; // rax
  __int64 v133; // r9
  __int64 v134; // rsi
  __int64 v135; // rax
  __int64 v136; // rsi
  __int64 v137; // r9
  __int64 v138; // rsi
  __int64 v139; // rdx
  unsigned __int8 *v140; // rsi
  __int64 v141; // rax
  unsigned int v142; // esi
  int v143; // eax
  _QWORD *v144; // rax
  __int64 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rax
  unsigned __int8 *v148; // rsi
  _QWORD *v149; // rax
  __int64 v150; // r11
  __int64 v151; // rax
  __int64 v152; // rsi
  __int64 v153; // r11
  __int64 v154; // rsi
  __int64 v155; // rdx
  unsigned __int8 *v156; // rsi
  double v157; // xmm4_8
  double v158; // xmm5_8
  unsigned __int64 v160; // [rsp+48h] [rbp-328h]
  __int64 v161; // [rsp+50h] [rbp-320h]
  unsigned __int64 v162; // [rsp+50h] [rbp-320h]
  unsigned __int64 v163; // [rsp+50h] [rbp-320h]
  __int64 *v165; // [rsp+60h] [rbp-310h]
  __int64 v166; // [rsp+60h] [rbp-310h]
  __int64 v167; // [rsp+60h] [rbp-310h]
  __int64 v168; // [rsp+60h] [rbp-310h]
  __int64 *v170; // [rsp+70h] [rbp-300h]
  __int64 v172; // [rsp+90h] [rbp-2E0h]
  __int64 **v173; // [rsp+90h] [rbp-2E0h]
  unsigned int v174; // [rsp+90h] [rbp-2E0h]
  unsigned __int64 v175; // [rsp+90h] [rbp-2E0h]
  __int64 v176; // [rsp+90h] [rbp-2E0h]
  unsigned __int64 v177; // [rsp+90h] [rbp-2E0h]
  unsigned __int64 v178; // [rsp+90h] [rbp-2E0h]
  int v179; // [rsp+90h] [rbp-2E0h]
  __int64 *v180; // [rsp+90h] [rbp-2E0h]
  __int64 *v181; // [rsp+98h] [rbp-2D8h]
  __int64 *v182; // [rsp+98h] [rbp-2D8h]
  __int64 *i; // [rsp+A0h] [rbp-2D0h]
  __int64 v184; // [rsp+A0h] [rbp-2D0h]
  __int64 v185; // [rsp+A0h] [rbp-2D0h]
  _QWORD *v186; // [rsp+A0h] [rbp-2D0h]
  __int64 v187; // [rsp+A0h] [rbp-2D0h]
  _QWORD *v188; // [rsp+A0h] [rbp-2D0h]
  __int64 v189; // [rsp+A0h] [rbp-2D0h]
  __int64 v190; // [rsp+A8h] [rbp-2C8h]
  __int64 v191; // [rsp+A8h] [rbp-2C8h]
  __int64 v192; // [rsp+A8h] [rbp-2C8h]
  __int64 v193; // [rsp+A8h] [rbp-2C8h]
  __int64 v194; // [rsp+A8h] [rbp-2C8h]
  __int64 v195; // [rsp+A8h] [rbp-2C8h]
  __int64 v196; // [rsp+A8h] [rbp-2C8h]
  unsigned __int8 *v197; // [rsp+B8h] [rbp-2B8h] BYREF
  __int64 v198[2]; // [rsp+C0h] [rbp-2B0h] BYREF
  __int16 v199; // [rsp+D0h] [rbp-2A0h]
  __int64 v200[2]; // [rsp+E0h] [rbp-290h] BYREF
  __int16 v201; // [rsp+F0h] [rbp-280h]
  unsigned __int8 *v202[2]; // [rsp+100h] [rbp-270h] BYREF
  __int16 v203; // [rsp+110h] [rbp-260h]
  unsigned __int8 *v204; // [rsp+120h] [rbp-250h] BYREF
  __int64 v205; // [rsp+128h] [rbp-248h]
  __int64 *v206; // [rsp+130h] [rbp-240h]
  __int64 v207; // [rsp+138h] [rbp-238h]
  __int64 v208; // [rsp+140h] [rbp-230h]
  int v209; // [rsp+148h] [rbp-228h]
  __int64 v210; // [rsp+150h] [rbp-220h]
  __int64 v211; // [rsp+158h] [rbp-218h]
  _QWORD v212[64]; // [rsp+170h] [rbp-200h] BYREF

  v16 = *(__int64 **)(a2 + 40);
  sub_15A5590((__int64)v212, v16, 1, 0);
  v170 = &a5[a6];
  if ( a5 != v170 )
  {
    for ( i = a5; v170 != i; ++i )
    {
      v18 = *i;
      v19 = sub_16498A0(*i);
      v204 = 0;
      v207 = v19;
      v208 = 0;
      v209 = 0;
      v210 = 0;
      v211 = 0;
      v205 = *(_QWORD *)(v18 + 40);
      v206 = (__int64 *)(v18 + 24);
      v20 = *(unsigned __int8 **)(v18 + 48);
      v202[0] = v20;
      if ( v20 )
      {
        sub_1623A60((__int64)v202, (__int64)v20, 2);
        if ( v204 )
          sub_161E7C0((__int64)&v204, (__int64)v204);
        v204 = v202[0];
        if ( v202[0] )
          sub_1623210((__int64)v202, v202[0], (__int64)&v204);
      }
      v21 = *(_QWORD *)(v18 - 24);
      v22 = (__int64 **)a1[5];
      if ( v22 != *(__int64 ***)v21 )
      {
        v201 = 257;
        if ( v22 != *(__int64 ***)v21 )
        {
          if ( *(_BYTE *)(v21 + 16) > 0x10u )
          {
            v203 = 257;
            v112 = sub_15FE0A0((_QWORD *)v21, (__int64)v22, 0, (__int64)v202, 0);
            v21 = v112;
            if ( v205 )
            {
              v113 = v206;
              sub_157E9D0(v205 + 40, v112);
              v114 = *(_QWORD *)(v21 + 24);
              v115 = *v113;
              *(_QWORD *)(v21 + 32) = v113;
              v115 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v21 + 24) = v115 | v114 & 7;
              *(_QWORD *)(v115 + 8) = v21 + 24;
              *v113 = *v113 & 7 | (v21 + 24);
            }
            sub_164B780(v21, v200);
            if ( v204 )
            {
              v198[0] = (__int64)v204;
              sub_1623A60((__int64)v198, (__int64)v204, 2);
              v116 = *(_QWORD *)(v21 + 48);
              if ( v116 )
                sub_161E7C0(v21 + 48, v116);
              v117 = (unsigned __int8 *)v198[0];
              *(_QWORD *)(v21 + 48) = v198[0];
              if ( v117 )
                sub_1623210((__int64)v198, v117, v21 + 48);
            }
          }
          else
          {
            v21 = sub_15A4750((__int64 ***)v21, v22, 0);
          }
        }
      }
      v23 = *(_QWORD *)(v18 + 56);
      v24 = 1;
      v172 = a1[2];
      v25 = v172;
      v26 = v23;
      v27 = (unsigned int)sub_15A9FE0(v172, v23);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v26 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v73 = *(_QWORD *)(v26 + 32);
            v26 = *(_QWORD *)(v26 + 24);
            v24 *= v73;
            continue;
          case 1:
            v28 = 16;
            break;
          case 2:
            v28 = 32;
            break;
          case 3:
          case 9:
            v28 = 64;
            break;
          case 4:
            v28 = 80;
            break;
          case 5:
          case 6:
            v28 = 128;
            break;
          case 7:
            v175 = v27;
            v66 = sub_15A9520(v25, 0);
            v27 = v175;
            v28 = (unsigned int)(8 * v66);
            break;
          case 0xB:
            v28 = *(_DWORD *)(v26 + 8) >> 8;
            break;
          case 0xD:
            v177 = v27;
            v72 = (_QWORD *)sub_15A9930(v25, v26);
            v27 = v177;
            v28 = 8LL * *v72;
            break;
          case 0xE:
            v160 = v27;
            v166 = v172;
            v161 = *(_QWORD *)(v26 + 24);
            v176 = *(_QWORD *)(v26 + 32);
            v67 = sub_15A9FE0(v25, v161);
            v68 = v166;
            v27 = v160;
            v69 = 1;
            v70 = v161;
            v71 = v67;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v70 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v145 = *(_QWORD *)(v70 + 32);
                  v70 = *(_QWORD *)(v70 + 24);
                  v69 *= v145;
                  continue;
                case 1:
                  v141 = 16;
                  goto LABEL_179;
                case 2:
                  v141 = 32;
                  goto LABEL_179;
                case 3:
                case 9:
                  v141 = 64;
                  goto LABEL_179;
                case 4:
                  v141 = 80;
                  goto LABEL_179;
                case 5:
                case 6:
                  v141 = 128;
                  goto LABEL_179;
                case 7:
                  v142 = 0;
                  v162 = v71;
                  v167 = v69;
                  goto LABEL_182;
                case 0xB:
                  v141 = *(_DWORD *)(v70 + 8) >> 8;
                  goto LABEL_179;
                case 0xD:
                  v163 = v71;
                  v168 = v69;
                  v144 = (_QWORD *)sub_15A9930(v68, v70);
                  v69 = v168;
                  v71 = v163;
                  v27 = v160;
                  v141 = 8LL * *v144;
                  goto LABEL_179;
                case 0xE:
                  sub_15A9FE0(v166, *(_QWORD *)(v70 + 24));
                  JUMPOUT(0x1EF0DB9);
                case 0xF:
                  v162 = v71;
                  v167 = v69;
                  v142 = *(_DWORD *)(v70 + 8) >> 8;
LABEL_182:
                  v143 = sub_15A9520(v68, v142);
                  v69 = v167;
                  v71 = v162;
                  v27 = v160;
                  v141 = (unsigned int)(8 * v143);
LABEL_179:
                  v28 = 8 * v176 * v71 * ((v71 + ((unsigned __int64)(v141 * v69 + 7) >> 3) - 1) / v71);
                  break;
              }
              break;
            }
            break;
          case 0xF:
            v178 = v27;
            v74 = sub_15A9520(v25, *(_DWORD *)(v26 + 8) >> 8);
            v27 = v178;
            v28 = (unsigned int)(8 * v74);
            break;
        }
        break;
      }
      v29 = a1[5];
      v201 = 257;
      v30 = sub_15A0680(v29, v27 * ((v27 + ((unsigned __int64)(v28 * v24 + 7) >> 3) - 1) / v27), 0);
      if ( *(_BYTE *)(v21 + 16) > 0x10u || *(_BYTE *)(v30 + 16) > 0x10u )
      {
        v203 = 257;
        v81 = sub_15FB440(15, (__int64 *)v21, v30, (__int64)v202, 0);
        v31 = v81;
        if ( v205 )
        {
          v82 = v206;
          sub_157E9D0(v205 + 40, v81);
          v83 = *(_QWORD *)(v31 + 24);
          v84 = *v82;
          *(_QWORD *)(v31 + 32) = v82;
          v84 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v31 + 24) = v84 | v83 & 7;
          *(_QWORD *)(v84 + 8) = v31 + 24;
          *v82 = *v82 & 7 | (v31 + 24);
        }
        sub_164B780(v31, v200);
        if ( v204 )
        {
          v198[0] = (__int64)v204;
          sub_1623A60((__int64)v198, (__int64)v204, 2);
          v85 = *(_QWORD *)(v31 + 48);
          if ( v85 )
            sub_161E7C0(v31 + 48, v85);
          v86 = (unsigned __int8 *)v198[0];
          *(_QWORD *)(v31 + 48) = v198[0];
          if ( v86 )
            sub_1623210((__int64)v198, v86, v31 + 48);
        }
      }
      else
      {
        v31 = sub_15A2C20((__int64 *)v21, v30, 0, 0, *(double *)a7.m128_u64, a8, a9);
      }
      v201 = 257;
      v173 = (__int64 **)a1[5];
      v199 = 257;
      v32 = sub_1648A60(64, 1u);
      v33 = (__int64)v32;
      if ( v32 )
        sub_15F9210((__int64)v32, *(_QWORD *)(*(_QWORD *)a3 + 24LL), a3, 0, 0, 0);
      if ( v205 )
      {
        v165 = v206;
        sub_157E9D0(v205 + 40, v33);
        v34 = *v165;
        v35 = *(_QWORD *)(v33 + 24) & 7LL;
        *(_QWORD *)(v33 + 32) = v165;
        v34 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v33 + 24) = v34 | v35;
        *(_QWORD *)(v34 + 8) = v33 + 24;
        *v165 = *v165 & 7 | (v33 + 24);
      }
      sub_164B780(v33, v198);
      if ( v204 )
      {
        v202[0] = v204;
        sub_1623A60((__int64)v202, (__int64)v204, 2);
        v36 = *(_QWORD *)(v33 + 48);
        v37 = v33 + 48;
        if ( v36 )
        {
          sub_161E7C0(v33 + 48, v36);
          v37 = v33 + 48;
        }
        v38 = v202[0];
        *(unsigned __int8 **)(v33 + 48) = v202[0];
        if ( v38 )
          sub_1623210((__int64)v202, v38, v37);
      }
      if ( v173 != *(__int64 ***)v33 )
      {
        if ( *(_BYTE *)(v33 + 16) > 0x10u )
        {
          v203 = 257;
          v94 = sub_15FDBD0(45, v33, (__int64)v173, (__int64)v202, 0);
          v33 = v94;
          if ( v205 )
          {
            v180 = v206;
            sub_157E9D0(v205 + 40, v94);
            v95 = *v180;
            v96 = *(_QWORD *)(v33 + 24) & 7LL;
            *(_QWORD *)(v33 + 32) = v180;
            v95 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v33 + 24) = v95 | v96;
            *(_QWORD *)(v95 + 8) = v33 + 24;
            *v180 = *v180 & 7 | (v33 + 24);
          }
          sub_164B780(v33, v200);
          if ( v204 )
          {
            v197 = v204;
            sub_1623A60((__int64)&v197, (__int64)v204, 2);
            v97 = *(_QWORD *)(v33 + 48);
            v98 = v33 + 48;
            if ( v97 )
            {
              sub_161E7C0(v33 + 48, v97);
              v98 = v33 + 48;
            }
            v99 = v197;
            *(_QWORD *)(v33 + 48) = v197;
            if ( v99 )
              sub_1623210((__int64)&v197, v99, v98);
          }
        }
        else
        {
          v33 = sub_15A46C0(45, (__int64 ***)v33, v173, 0);
        }
      }
      v201 = 257;
      if ( *(_BYTE *)(v33 + 16) > 0x10u || *(_BYTE *)(v31 + 16) > 0x10u )
      {
        v203 = 257;
        v75 = sub_15FB440(13, (__int64 *)v33, v31, (__int64)v202, 0);
        v39 = v75;
        if ( v205 )
        {
          v76 = v206;
          sub_157E9D0(v205 + 40, v75);
          v77 = *(_QWORD *)(v39 + 24);
          v78 = *v76;
          *(_QWORD *)(v39 + 32) = v76;
          v78 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v39 + 24) = v78 | v77 & 7;
          *(_QWORD *)(v78 + 8) = v39 + 24;
          *v76 = *v76 & 7 | (v39 + 24);
        }
        sub_164B780(v39, v200);
        if ( v204 )
        {
          v198[0] = (__int64)v204;
          sub_1623A60((__int64)v198, (__int64)v204, 2);
          v79 = *(_QWORD *)(v39 + 48);
          if ( v79 )
            sub_161E7C0(v39 + 48, v79);
          v80 = (unsigned __int8 *)v198[0];
          *(_QWORD *)(v39 + 48) = v198[0];
          if ( v80 )
            sub_1623210((__int64)v198, v80, v39 + 48);
        }
      }
      else
      {
        v39 = sub_15A2B60((__int64 *)v33, v31, 0, 0, *(double *)a7.m128_u64, a8, a9);
      }
      v174 = (unsigned int)(1 << *(_WORD *)(v18 + 18)) >> 1;
      v40 = sub_15AAE50(a1[2], v23);
      v41 = v174;
      v42 = a1[5];
      v199 = 257;
      v43 = (__int64 **)a1[4];
      v201 = 257;
      if ( v174 < 0x10 )
        v41 = 16;
      if ( v41 < v40 )
        v41 = v40;
      v44 = sub_15A0680(v42, ~(unsigned __int64)(v41 - 1), 0);
      v45 = *(_BYTE *)(v44 + 16);
      if ( v45 > 0x10u )
      {
LABEL_99:
        v203 = 257;
        v87 = sub_15FB440(26, (__int64 *)v39, v44, (__int64)v202, 0);
        v39 = v87;
        if ( v205 )
        {
          v88 = v206;
          sub_157E9D0(v205 + 40, v87);
          v89 = *(_QWORD *)(v39 + 24);
          v90 = *v88;
          *(_QWORD *)(v39 + 32) = v88;
          v90 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v39 + 24) = v90 | v89 & 7;
          *(_QWORD *)(v90 + 8) = v39 + 24;
          *v88 = *v88 & 7 | (v39 + 24);
        }
        sub_164B780(v39, v198);
        if ( v204 )
        {
          v197 = v204;
          sub_1623A60((__int64)&v197, (__int64)v204, 2);
          v91 = *(_QWORD *)(v39 + 48);
          if ( v91 )
            sub_161E7C0(v39 + 48, v91);
          v92 = v197;
          *(_QWORD *)(v39 + 48) = v197;
          if ( v92 )
            sub_1623210((__int64)&v197, v92, v39 + 48);
        }
        goto LABEL_41;
      }
      if ( v45 == 13 )
      {
        v93 = *(_DWORD *)(v44 + 32);
        if ( v93 <= 0x40 )
        {
          if ( *(_QWORD *)(v44 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v93) )
            goto LABEL_41;
        }
        else
        {
          v179 = *(_DWORD *)(v44 + 32);
          if ( v179 == (unsigned int)sub_16A58F0(v44 + 24) )
            goto LABEL_41;
        }
      }
      if ( *(_BYTE *)(v39 + 16) > 0x10u )
        goto LABEL_99;
      v39 = sub_15A2CF0((__int64 *)v39, v44, *(double *)a7.m128_u64, a8, a9);
LABEL_41:
      if ( v43 != *(__int64 ***)v39 )
      {
        if ( *(_BYTE *)(v39 + 16) > 0x10u )
        {
          v203 = 257;
          v100 = sub_15FDBD0(46, v39, (__int64)v43, (__int64)v202, 0);
          v39 = v100;
          if ( v205 )
          {
            v101 = v206;
            sub_157E9D0(v205 + 40, v100);
            v102 = *(_QWORD *)(v39 + 24);
            v103 = *v101;
            *(_QWORD *)(v39 + 32) = v101;
            v103 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v39 + 24) = v103 | v102 & 7;
            *(_QWORD *)(v103 + 8) = v39 + 24;
            *v101 = *v101 & 7 | (v39 + 24);
          }
          sub_164B780(v39, v200);
          if ( v204 )
          {
            v197 = v204;
            sub_1623A60((__int64)&v197, (__int64)v204, 2);
            v104 = *(_QWORD *)(v39 + 48);
            if ( v104 )
              sub_161E7C0(v39 + 48, v104);
            v105 = v197;
            *(_QWORD *)(v39 + 48) = v197;
            if ( v105 )
              sub_1623210((__int64)&v197, v105, v39 + 48);
          }
        }
        else
        {
          v39 = sub_15A46C0(46, (__int64 ***)v39, v43, 0);
        }
      }
      v203 = 257;
      v46 = sub_1648A60(64, 2u);
      v47 = v46;
      if ( v46 )
        sub_15F9650((__int64)v46, v39, a3, 0, 0);
      if ( v205 )
      {
        v48 = (unsigned __int64 *)v206;
        sub_157E9D0(v205 + 40, (__int64)v47);
        v49 = v47[3];
        v50 = *v48;
        v47[4] = v48;
        v50 &= 0xFFFFFFFFFFFFFFF8LL;
        v47[3] = v50 | v49 & 7;
        *(_QWORD *)(v50 + 8) = v47 + 3;
        *v48 = *v48 & 7 | (unsigned __int64)(v47 + 3);
      }
      sub_164B780((__int64)v47, (__int64 *)v202);
      if ( v204 )
      {
        v200[0] = (__int64)v204;
        sub_1623A60((__int64)v200, (__int64)v204, 2);
        v51 = v47[6];
        v52 = (__int64)(v47 + 6);
        if ( v51 )
        {
          sub_161E7C0((__int64)(v47 + 6), v51);
          v52 = (__int64)(v47 + 6);
        }
        v53 = (unsigned __int8 *)v200[0];
        v47[6] = v200[0];
        if ( v53 )
          sub_1623210((__int64)v200, v53, v52);
      }
      if ( a4 )
      {
        v203 = 257;
        v54 = sub_1648A60(64, 2u);
        v55 = v54;
        if ( v54 )
          sub_15F9650((__int64)v54, v39, a4, 0, 0);
        if ( v205 )
        {
          v56 = (unsigned __int64 *)v206;
          sub_157E9D0(v205 + 40, (__int64)v55);
          v57 = v55[3];
          v58 = *v56;
          v55[4] = v56;
          v58 &= 0xFFFFFFFFFFFFFFF8LL;
          v55[3] = v58 | v57 & 7;
          *(_QWORD *)(v58 + 8) = v55 + 3;
          *v56 = *v56 & 7 | (unsigned __int64)(v55 + 3);
        }
        sub_164B780((__int64)v55, (__int64 *)v202);
        if ( v204 )
        {
          v200[0] = (__int64)v204;
          sub_1623A60((__int64)v200, (__int64)v204, 2);
          v59 = v55[6];
          v60 = (__int64)(v55 + 6);
          if ( v59 )
          {
            sub_161E7C0((__int64)(v55 + 6), v59);
            v60 = (__int64)(v55 + 6);
          }
          v61 = (unsigned __int8 *)v200[0];
          v55[6] = v200[0];
          if ( v61 )
            sub_1623210((__int64)v200, v61, v60);
        }
      }
      v201 = 257;
      v62 = *(_QWORD *)v18;
      if ( *(_QWORD *)v18 != *(_QWORD *)v39 )
      {
        if ( *(_BYTE *)(v39 + 16) > 0x10u )
        {
          v203 = 257;
          v106 = sub_15FDFF0(v39, v62, (__int64)v202, 0);
          v39 = v106;
          if ( v205 )
          {
            v107 = v206;
            sub_157E9D0(v205 + 40, v106);
            v108 = *(_QWORD *)(v39 + 24);
            v109 = *v107;
            *(_QWORD *)(v39 + 32) = v107;
            v109 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v39 + 24) = v109 | v108 & 7;
            *(_QWORD *)(v109 + 8) = v39 + 24;
            *v107 = *v107 & 7 | (v39 + 24);
          }
          sub_164B780(v39, v200);
          if ( v204 )
          {
            v198[0] = (__int64)v204;
            sub_1623A60((__int64)v198, (__int64)v204, 2);
            v110 = *(_QWORD *)(v39 + 48);
            if ( v110 )
              sub_161E7C0(v39 + 48, v110);
            v111 = (unsigned __int8 *)v198[0];
            *(_QWORD *)(v39 + 48) = v198[0];
            if ( v111 )
              sub_1623210((__int64)v198, v111, v39 + 48);
          }
        }
        else
        {
          v39 = sub_15A4A70((__int64 ***)v39, v62);
        }
      }
      if ( (*(_BYTE *)(v18 + 23) & 0x20) != 0 && *(_BYTE *)(v39 + 16) > 0x17u )
        sub_164B7C0(v39, v18);
      sub_1AEA710(v18, v39, v212, 0, 0, 0);
      sub_164D160(v18, v39, a7, a8, a9, a10, v63, v64, a13, a14);
      sub_15F20C0((_QWORD *)v18);
      v16 = (__int64 *)v204;
      if ( v204 )
        sub_161E7C0((__int64)&v204, (__int64)v204);
    }
  }
  if ( !a6 )
    return sub_129E320((__int64)v212, (__int64)v16);
  v118 = *(_QWORD *)(a2 + 80);
  v119 = a2 + 72;
  if ( a2 + 72 == v118 )
    return sub_129E320((__int64)v212, (__int64)v16);
  if ( !v118 )
    BUG();
  while ( 1 )
  {
    v120 = *(_QWORD *)(v118 + 24);
    if ( v120 != v118 + 16 )
      break;
    v118 = *(_QWORD *)(v118 + 8);
    if ( v119 == v118 )
      return sub_129E320((__int64)v212, (__int64)v16);
    if ( !v118 )
      BUG();
  }
  if ( v118 == v119 )
    return sub_129E320((__int64)v212, (__int64)v16);
LABEL_146:
  for ( j = *(_QWORD *)(v120 + 8); ; j = *(_QWORD *)(v118 + 24) )
  {
    v122 = v118 - 24;
    if ( !v118 )
      v122 = 0;
    if ( j != v122 + 40 )
    {
      if ( *(_BYTE *)(v120 - 8) == 78 )
      {
        v123 = *(_QWORD *)(v120 - 48);
        v124 = (_QWORD *)(v120 - 24);
        if ( !*(_BYTE *)(v123 + 16) && (*(_BYTE *)(v123 + 33) & 0x20) != 0 )
        {
LABEL_159:
          v125 = *(_DWORD *)(v123 + 36);
          if ( v125 == 202 )
          {
            v193 = v120;
            v146 = sub_16498A0((__int64)v124);
            v206 = 0;
            v204 = 0;
            v207 = v146;
            v208 = 0;
            v209 = 0;
            v210 = 0;
            v211 = 0;
            v147 = *(_QWORD *)(v193 + 16);
            v206 = (__int64 *)v193;
            v205 = v147;
            v148 = *(unsigned __int8 **)(v193 + 24);
            v202[0] = v148;
            if ( v148 )
            {
              sub_1623A60((__int64)v202, (__int64)v148, 2);
              if ( v204 )
                sub_161E7C0((__int64)&v204, (__int64)v204);
              v204 = v202[0];
              if ( v202[0] )
                sub_1623210((__int64)v202, v202[0], (__int64)&v204);
            }
            v203 = 257;
            v149 = sub_1648A60(64, 1u);
            v150 = (__int64)v149;
            if ( v149 )
            {
              v188 = v149;
              sub_15F9210((__int64)v149, *(_QWORD *)(*(_QWORD *)a3 + 24LL), a3, 0, 0, 0);
              v150 = (__int64)v188;
            }
            if ( v205 )
            {
              v189 = v150;
              v182 = v206;
              sub_157E9D0(v205 + 40, v150);
              v150 = v189;
              v151 = *(_QWORD *)(v189 + 24);
              v152 = *v182;
              *(_QWORD *)(v189 + 32) = v182;
              v152 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v189 + 24) = v152 | v151 & 7;
              *(_QWORD *)(v152 + 8) = v189 + 24;
              *v182 = *v182 & 7 | (v189 + 24);
            }
            v194 = v150;
            sub_164B780(v150, (__int64 *)v202);
            v153 = v194;
            if ( v204 )
            {
              v200[0] = (__int64)v204;
              sub_1623A60((__int64)v200, (__int64)v204, 2);
              v153 = v194;
              v154 = *(_QWORD *)(v194 + 48);
              v155 = v194 + 48;
              if ( v154 )
              {
                sub_161E7C0(v194 + 48, v154);
                v153 = v194;
                v155 = v194 + 48;
              }
              v156 = (unsigned __int8 *)v200[0];
              *(_QWORD *)(v153 + 48) = v200[0];
              if ( v156 )
              {
                v195 = v153;
                sub_1623210((__int64)v200, v156, v155);
                v153 = v195;
              }
            }
            v196 = v153;
            sub_164B7C0(v153, (__int64)v124);
            sub_164D160((__int64)v124, v196, a7, a8, a9, a10, v157, v158, a13, a14);
            goto LABEL_176;
          }
          if ( v125 == 201 )
          {
            v190 = v120;
            v126 = sub_16498A0((__int64)v124);
            v127 = v190;
            v206 = 0;
            v204 = 0;
            v207 = v126;
            v208 = 0;
            v209 = 0;
            v210 = 0;
            v211 = 0;
            v128 = *(_QWORD *)(v190 + 16);
            v206 = (__int64 *)v190;
            v205 = v128;
            v129 = *(unsigned __int8 **)(v190 + 24);
            v202[0] = v129;
            if ( v129 )
            {
              sub_1623A60((__int64)v202, (__int64)v129, 2);
              v127 = v190;
              if ( v204 )
              {
                sub_161E7C0((__int64)&v204, (__int64)v204);
                v130 = v202[0];
                v127 = v190;
              }
              else
              {
                v130 = v202[0];
              }
              v204 = v130;
              if ( v130 )
              {
                v184 = v127;
                sub_1623210((__int64)v202, v130, (__int64)&v204);
                v127 = v184;
              }
            }
            v131 = v124[-3 * (*(_DWORD *)(v127 - 4) & 0xFFFFFFF)];
            v203 = 257;
            v185 = v131;
            v132 = sub_1648A60(64, 2u);
            v133 = (__int64)v132;
            if ( v132 )
            {
              v134 = v185;
              v186 = v132;
              sub_15F9650((__int64)v132, v134, a3, 0, 0);
              v133 = (__int64)v186;
            }
            if ( v205 )
            {
              v187 = v133;
              v181 = v206;
              sub_157E9D0(v205 + 40, v133);
              v133 = v187;
              v135 = *(_QWORD *)(v187 + 24);
              v136 = *v181;
              *(_QWORD *)(v187 + 32) = v181;
              v136 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v187 + 24) = v136 | v135 & 7;
              *(_QWORD *)(v136 + 8) = v187 + 24;
              *v181 = *v181 & 7 | (v187 + 24);
            }
            v191 = v133;
            sub_164B780(v133, (__int64 *)v202);
            v137 = v191;
            if ( v204 )
            {
              v200[0] = (__int64)v204;
              sub_1623A60((__int64)v200, (__int64)v204, 2);
              v137 = v191;
              v138 = *(_QWORD *)(v191 + 48);
              v139 = v191 + 48;
              if ( v138 )
              {
                sub_161E7C0(v191 + 48, v138);
                v137 = v191;
                v139 = v191 + 48;
              }
              v140 = (unsigned __int8 *)v200[0];
              *(_QWORD *)(v137 + 48) = v200[0];
              if ( v140 )
              {
                v192 = v137;
                sub_1623210((__int64)v200, v140, v139);
                v137 = v192;
              }
            }
            sub_164B7C0(v137, (__int64)v124);
LABEL_176:
            sub_15F20C0(v124);
            v16 = (__int64 *)v204;
            if ( v204 )
              sub_161E7C0((__int64)&v204, (__int64)v204);
          }
        }
      }
      if ( v119 == v118 )
        return sub_129E320((__int64)v212, (__int64)v16);
      v120 = j;
      goto LABEL_146;
    }
    v118 = *(_QWORD *)(v118 + 8);
    if ( v119 == v118 )
      break;
    if ( !v118 )
      BUG();
  }
  if ( *(_BYTE *)(v120 - 8) == 78 )
  {
    v123 = *(_QWORD *)(v120 - 48);
    v124 = (_QWORD *)(v120 - 24);
    if ( !*(_BYTE *)(v123 + 16) && (*(_BYTE *)(v123 + 33) & 0x20) != 0 )
      goto LABEL_159;
  }
  return sub_129E320((__int64)v212, (__int64)v16);
}
