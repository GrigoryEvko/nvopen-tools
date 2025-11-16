// Function: sub_9A4320
// Address: 0x9a4320
//
__int64 __fastcall sub_9A4320(unsigned __int8 *a1, __int64 a2, unsigned int a3, const __m128i *a4)
{
  unsigned __int8 *v5; // r12
  __int64 v6; // r14
  int v7; // eax
  unsigned int v8; // ebx
  unsigned __int8 v9; // dl
  int v10; // eax
  _BYTE **v11; // rax
  __int32 v12; // ebx
  unsigned __int8 *v14; // rdi
  _QWORD *v15; // rax
  __int64 v16; // rcx
  unsigned __int8 v17; // r14
  int v18; // r12d
  bool v19; // al
  __int64 v20; // rax
  int v21; // eax
  int v22; // eax
  int v23; // eax
  int v24; // eax
  int v25; // eax
  __int64 *v26; // rax
  __int64 v27; // r14
  __int64 v28; // r15
  __int64 v29; // r12
  unsigned int v30; // ebx
  _QWORD *v31; // rsi
  __int64 v32; // rsi
  int v33; // eax
  __int64 v34; // r12
  __int64 v35; // rax
  char v36; // dl
  unsigned int v37; // ebx
  _QWORD *v38; // rcx
  __int64 v39; // r12
  unsigned __int8 v40; // r14
  char v41; // r14
  __int64 v42; // rax
  unsigned __int8 *v43; // r12
  int v44; // eax
  unsigned __int8 *v45; // r12
  int v46; // eax
  _QWORD *v47; // rax
  int v48; // eax
  _BYTE **v49; // rax
  __int64 v50; // rax
  bool v51; // zf
  __m128i v52; // xmm1
  __m128i v53; // xmm2
  unsigned __int64 v54; // xmm3_8
  __int64 v55; // rax
  int *v56; // rax
  __int64 v57; // r8
  __m128i *v58; // r9
  int v59; // eax
  unsigned int v60; // eax
  unsigned __int8 *v61; // r13
  __int64 v62; // rax
  unsigned __int8 *v63; // r12
  __int64 v64; // rcx
  unsigned __int8 **v65; // rdx
  __int64 v66; // rax
  unsigned __int8 *v67; // rbx
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // rcx
  int v81; // eax
  __int64 v82; // r12
  unsigned int v83; // ecx
  unsigned __int8 *v84; // rdi
  _QWORD *v85; // rax
  __int64 v86; // rax
  __m128i *v87; // rdx
  __m128i v88; // rax
  _QWORD *v89; // rax
  __int64 v90; // rax
  __m128i *v91; // rdx
  __m128i v92; // rax
  __int64 v93; // rdi
  __int64 v94; // rax
  __int64 v95; // rdx
  unsigned __int8 *v96; // r9
  unsigned __int8 *v97; // r15
  signed __int64 v98; // r8
  __int64 v99; // r9
  unsigned __int64 v100; // r10
  unsigned __int64 v101; // rbx
  _QWORD *v102; // rbx
  __int64 v103; // rax
  __int64 v104; // rdx
  unsigned __int64 v105; // r10
  unsigned __int8 v106; // al
  __int8 v107; // al
  unsigned int v108; // r13d
  __int64 v109; // rax
  __int64 v110; // rax
  __int64 v111; // rdi
  unsigned __int8 v112; // r14
  char v113; // al
  _BOOL4 v114; // r14d
  char v115; // r15
  _QWORD *v116; // rax
  int v117; // eax
  __int64 *v118; // rax
  __int64 v119; // r8
  __int64 v120; // rcx
  int v121; // eax
  __int64 v122; // rax
  unsigned __int8 *v123; // r9
  unsigned __int8 *v124; // r12
  _BYTE *v125; // rdi
  unsigned int v126; // r15d
  unsigned __int8 *v128; // r15
  int v129; // eax
  unsigned int v130; // r12d
  __int64 v132; // rax
  unsigned int v133; // eax
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // rdi
  __int64 v137; // rax
  __int64 v138; // rdx
  __int64 v139; // rax
  __int64 v140; // rdx
  __int64 v141; // rax
  unsigned __int8 *v142; // r14
  unsigned __int8 *v143; // r12
  __int64 v144; // r12
  unsigned __int8 **v145; // rax
  int v146; // eax
  unsigned int v147; // r15d
  unsigned int v148; // esi
  int v149; // eax
  __int64 v150; // rax
  _BYTE *v151; // rax
  __int64 v152; // rax
  unsigned __int64 v153; // rsi
  __m128i v154; // rax
  signed __int64 v155; // r8
  unsigned __int64 v156; // rax
  __int64 v157; // rax
  __m128i v158; // rax
  unsigned int *v159; // rax
  __int64 v160; // rdx
  unsigned int v161; // eax
  __int64 v162; // rax
  __int64 v163; // rax
  bool v164; // r12
  __int64 v165; // rax
  unsigned __int64 v166; // [rsp+8h] [rbp-128h]
  signed __int64 v167; // [rsp+10h] [rbp-120h]
  unsigned __int64 v168; // [rsp+18h] [rbp-118h]
  unsigned __int64 v169; // [rsp+18h] [rbp-118h]
  __int64 v170; // [rsp+18h] [rbp-118h]
  __int64 v171; // [rsp+18h] [rbp-118h]
  __int64 v172; // [rsp+20h] [rbp-110h]
  int v173; // [rsp+20h] [rbp-110h]
  unsigned __int64 v174; // [rsp+20h] [rbp-110h]
  signed __int64 v175; // [rsp+20h] [rbp-110h]
  signed __int64 v176; // [rsp+20h] [rbp-110h]
  __int64 v177; // [rsp+20h] [rbp-110h]
  unsigned __int8 *v178; // [rsp+28h] [rbp-108h]
  char v179; // [rsp+28h] [rbp-108h]
  signed __int64 v180; // [rsp+28h] [rbp-108h]
  signed __int64 v181; // [rsp+28h] [rbp-108h]
  unsigned __int64 v182; // [rsp+28h] [rbp-108h]
  signed __int64 v183; // [rsp+28h] [rbp-108h]
  unsigned __int8 v184; // [rsp+30h] [rbp-100h]
  __int64 v185; // [rsp+30h] [rbp-100h]
  unsigned __int64 v186; // [rsp+30h] [rbp-100h]
  unsigned __int8 *v187; // [rsp+30h] [rbp-100h]
  unsigned int v189; // [rsp+38h] [rbp-F8h]
  unsigned int v190; // [rsp+38h] [rbp-F8h]
  unsigned int v191; // [rsp+44h] [rbp-ECh] BYREF
  unsigned __int8 *v192; // [rsp+48h] [rbp-E8h] BYREF
  unsigned __int8 *v193; // [rsp+58h] [rbp-D8h] BYREF
  unsigned __int8 *v194; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v195; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v196; // [rsp+70h] [rbp-C0h] BYREF
  const __m128i *v197; // [rsp+78h] [rbp-B8h]
  unsigned int *v198; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v199; // [rsp+88h] [rbp-A8h]
  unsigned __int8 **v200; // [rsp+90h] [rbp-A0h] BYREF
  __m128i *v201; // [rsp+98h] [rbp-98h]
  __int64 v202; // [rsp+A0h] [rbp-90h] BYREF
  __int64 *v203; // [rsp+A8h] [rbp-88h]
  __m128i v204; // [rsp+B0h] [rbp-80h] BYREF
  __m128i v205; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v206; // [rsp+D0h] [rbp-60h]
  __int128 v207; // [rsp+E0h] [rbp-50h]
  __int64 v208; // [rsp+F0h] [rbp-40h]

  v5 = a1;
  v6 = *((_QWORD *)a1 + 1);
  v192 = a1;
  v7 = *(unsigned __int8 *)(v6 + 8);
  v191 = a3;
  if ( (unsigned int)(v7 - 17) <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  v8 = sub_BCB060(v6);
  if ( !v8 )
  {
    v5 = v192;
    v8 = sub_AE43A0(a4->m128i_i64[0], v6);
  }
  v9 = *v5;
  if ( *v5 <= 0x1Cu )
    v10 = *((unsigned __int16 *)v5 + 1);
  else
    v10 = v9 - 29;
  switch ( v10 )
  {
    case 5:
    case 56:
      if ( *(_BYTE *)(*((_QWORD *)v5 + 1) + 8LL) == 14 )
      {
        if ( (unsigned __int8)sub_B493B0(v5) )
          goto LABEL_94;
        v136 = sub_98AC40((__int64)v5, 1);
        if ( v136 )
        {
LABEL_209:
          LODWORD(v6) = sub_9B6260(v136, a4, v191);
          return (unsigned int)v6;
        }
      }
      else
      {
        if ( a4[4].m128i_i8[0] )
        {
          if ( (v5[7] & 0x20) != 0 )
          {
            v6 = sub_B91C10(v5, 4);
            if ( v6 )
            {
              sub_9691E0((__int64)&v204, v8, 0, 0, 0);
              LODWORD(v6) = sub_984D50(v6, (__int64)&v204);
              sub_969240(v204.m128i_i64);
              return (unsigned int)v6;
            }
          }
        }
        sub_B492D0(&v204, v5);
        LODWORD(v6) = v206.m128i_u8[0];
        if ( v206.m128i_i8[0] )
        {
          sub_9691E0((__int64)&v200, v204.m128i_u32[2], 0, 0, 0);
          if ( !(unsigned __int8)sub_AB1B10(&v204, &v200) )
          {
            sub_969240((__int64 *)&v200);
            if ( v206.m128i_i8[0] )
              sub_9963D0((__int64)&v204);
            return (unsigned int)v6;
          }
          sub_969240((__int64 *)&v200);
          if ( v206.m128i_i8[0] )
          {
            v206.m128i_i8[0] = 0;
            if ( v205.m128i_i32[2] > 0x40u && v205.m128i_i64[0] )
              j_j___libc_free_0_0(v205.m128i_i64[0]);
            if ( v204.m128i_i32[2] > 0x40u && v204.m128i_i64[0] )
              j_j___libc_free_0_0(v204.m128i_i64[0]);
          }
        }
        v132 = sub_B494D0(v5, 52);
        v45 = v192;
        if ( !v132 || *((_QWORD *)v192 + 1) != *(_QWORD *)(v132 + 8) )
        {
LABEL_204:
          LOBYTE(v6) = sub_988010((__int64)v45) && v45 != 0;
          if ( !(_BYTE)v6 )
            goto LABEL_122;
          v133 = sub_987FE0((__int64)v45);
          if ( v133 > 0x192 )
          {
            if ( v133 != 493 )
              goto LABEL_9;
          }
          else
          {
            if ( v133 > 0x136 )
            {
              switch ( v133 )
              {
                case 0x137u:
                  v141 = *((_DWORD *)v45 + 1) & 0x7FFFFFF;
                  v142 = *(unsigned __int8 **)&v45[32 * (1 - v141)];
                  v143 = *(unsigned __int8 **)&v45[-32 * v141];
                  if ( (unsigned __int8)sub_985700(v143, v142) )
                    goto LABEL_94;
                  v44 = sub_9A3DF0(a2, v191, (__int64)a4, v8, v143, v142, 1u, 0);
                  goto LABEL_80;
                case 0x149u:
                  v197 = a4;
                  v198 = &v191;
                  v196 = a2;
                  LOWORD(v194) = 0;
                  LOWORD(v195) = 0;
                  sub_9B0110(&v200, *(_QWORD *)&v45[32 * (1LL - (*((_DWORD *)v45 + 1) & 0x7FFFFFF))], a2, v191, a4);
                  if ( !sub_986C60((__int64 *)&v200, (_DWORD)v201 - 1)
                    || (LODWORD(v6) = sub_9B61E0(
                                        &v196,
                                        *(_QWORD *)&v45[32 * (1LL - (*((_DWORD *)v45 + 1) & 0x7FFFFFF))],
                                        &v195,
                                        &v200),
                        !(_BYTE)v6) )
                  {
                    sub_9B0110(&v204, *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)], a2, v191, a4);
                    if ( !sub_986C60(v204.m128i_i64, v204.m128i_i32[2] - 1)
                      || (LODWORD(v6) = sub_9B61E0(
                                          &v196,
                                          *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)],
                                          &v194,
                                          &v204),
                          !(_BYTE)v6) )
                    {
                      LODWORD(v6) = sub_9B61E0(
                                      &v196,
                                      *(_QWORD *)&v45[32 * (1LL - (*((_DWORD *)v45 + 1) & 0x7FFFFFF))],
                                      &v195,
                                      &v200);
                      if ( (_BYTE)v6 )
                        LODWORD(v6) = sub_9B61E0(
                                        &v196,
                                        *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)],
                                        &v194,
                                        &v204);
                    }
                    sub_969240(v205.m128i_i64);
                    sub_969240(v204.m128i_i64);
                  }
                  sub_969240(&v202);
                  sub_969240((__int64 *)&v200);
                  return (unsigned int)v6;
                case 0x14Au:
                  sub_9B0110(&v200, *(_QWORD *)&v45[32 * (1LL - (*((_DWORD *)v45 + 1) & 0x7FFFFFF))], a2, v191, a4);
                  if ( sub_986C60(&v202, (_DWORD)v203 - 1) )
                    goto LABEL_355;
                  sub_9B0110(&v204, *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)], a2, v191, a4);
                  if ( sub_986C60(v205.m128i_i64, v205.m128i_i32[2] - 1)
                    || !sub_9867B0((__int64)&v202) && !sub_9867B0((__int64)&v205) )
                  {
                    sub_969240(v205.m128i_i64);
                    sub_969240(v204.m128i_i64);
LABEL_355:
                    sub_969240(&v202);
                    sub_969240((__int64 *)&v200);
                    return (unsigned int)v6;
                  }
                  sub_969240(v205.m128i_i64);
                  sub_969240(v204.m128i_i64);
                  sub_969240(&v202);
                  sub_969240((__int64 *)&v200);
LABEL_220:
                  if ( (unsigned __int8)sub_9A6530(
                                          *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)],
                                          a2,
                                          a4,
                                          v191) )
                  {
                    LODWORD(v6) = sub_9A6530(
                                    *(_QWORD *)&v45[32 * (1LL - (*((_DWORD *)v45 + 1) & 0x7FFFFFF))],
                                    a2,
                                    a4,
                                    v191);
                    return (unsigned int)v6;
                  }
                  break;
                case 0x151u:
                case 0x172u:
                  goto LABEL_214;
                case 0x152u:
                  v137 = *((_DWORD *)v45 + 1) & 0x7FFFFFF;
                  v138 = 1 - v137;
                  v139 = -32 * v137;
                  v140 = 32 * v138;
                  goto LABEL_223;
                case 0x167u:
                case 0x16Du:
                  if ( (unsigned __int8)sub_985700(
                                          *(_BYTE **)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)],
                                          *(_BYTE **)&v45[32 * (1LL - (*((_DWORD *)v45 + 1) & 0x7FFFFFF))])
                    || (unsigned __int8)sub_9A6530(
                                          *(_QWORD *)&v45[32 * (1LL - (*((_DWORD *)v45 + 1) & 0x7FFFFFF))],
                                          a2,
                                          a4,
                                          v191) )
                  {
                    goto LABEL_94;
                  }
                  LODWORD(v6) = sub_9A6530(*(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)], a2, a4, v191);
                  return (unsigned int)v6;
                case 0x16Eu:
                  goto LABEL_220;
                case 0x18Cu:
                case 0x18Du:
                case 0x18Eu:
                case 0x18Fu:
                case 0x190u:
                  v136 = *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)];
                  goto LABEL_209;
                case 0x192u:
                  sub_C48440(&v204, a2, v134, v135);
                  LODWORD(v6) = sub_9A6530(*(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)], &v204, a4, v191);
                  sub_969240(v204.m128i_i64);
                  return (unsigned int)v6;
                default:
                  goto LABEL_9;
              }
              goto LABEL_122;
            }
            if ( v133 == 67 )
            {
              sub_9B0110(&v204, *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)], a2, v191, a4);
              v148 = 0;
            }
            else
            {
              if ( v133 > 0x43 )
              {
                if ( v133 == 152 )
                {
                  v84 = v45;
                  goto LABEL_117;
                }
                if ( v133 - 180 > 1 )
                  goto LABEL_9;
                v111 = *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)];
                if ( *(_QWORD *)&v45[32 * (1LL - (*((_DWORD *)v45 + 1) & 0x7FFFFFF))] != v111 )
                  goto LABEL_9;
                goto LABEL_167;
              }
              if ( v133 != 65 )
              {
                if ( v133 > 0x41 || v133 == 1 || v133 - 14 <= 1 )
                {
LABEL_214:
                  v111 = *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)];
                  goto LABEL_167;
                }
                goto LABEL_9;
              }
              sub_9B0110(&v204, *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)], a2, v191, a4);
              v148 = v204.m128i_i32[2] - 1;
            }
            LOBYTE(v149) = sub_986C60(v204.m128i_i64, v148);
            LODWORD(v6) = v149;
            sub_969240(v205.m128i_i64);
            sub_969240(v204.m128i_i64);
          }
          return (unsigned int)v6;
        }
        if ( (unsigned __int8)sub_9B6260(v132, a4, v191) )
          goto LABEL_94;
      }
      v45 = v192;
      goto LABEL_204;
    case 13:
      if ( a4[4].m128i_i8[0] )
      {
        v40 = v5[1];
        v184 = (v40 & 4) != 0;
        v41 = (v40 & 2) != 0;
      }
      else
      {
        v184 = 0;
        v41 = 0;
      }
      v42 = sub_986520((__int64)v5);
      v43 = *(unsigned __int8 **)v42;
      v178 = *(unsigned __int8 **)(v42 + 32);
      if ( (unsigned __int8)sub_985700(*(_BYTE **)v42, v178) )
        goto LABEL_94;
      v44 = sub_9A3DF0(a2, v191, (__int64)a4, v8, v43, v178, v184, v41);
      goto LABEL_80;
    case 15:
      v118 = (__int64 *)sub_986520((__int64)v5);
      v119 = v118[4];
      v120 = *v118;
      goto LABEL_173;
    case 17:
      if ( a4[4].m128i_i8[0] )
      {
        v112 = v5[1];
        v113 = v112 >> 2;
        v114 = (v112 & 2) != 0;
        v115 = v113 & 1;
      }
      else
      {
        v115 = 0;
        v114 = 0;
      }
      v116 = (_QWORD *)sub_986520((__int64)v5);
      v117 = sub_9B4D50(a2, v191, (_DWORD)a4, v8, *v116, v116[4], v115, v114);
      goto LABEL_171;
    case 19:
    case 20:
      v17 = v5[1];
      v15 = (_QWORD *)sub_986520((__int64)v5);
      v16 = v191;
      LODWORD(v6) = (v17 & 2) != 0;
      if ( (_BYTE)v6 )
        goto LABEL_20;
      sub_9B0110(&v196, *v15, a2, v191, a4);
      v18 = (int)v197;
      if ( (unsigned int)v197 <= 0x40 )
        v19 = v196 == 0;
      else
        v19 = v18 == (unsigned int)sub_C444A0(&v196);
      if ( !v19 )
        goto LABEL_27;
      v147 = v199;
      if ( v199 <= 0x40 )
      {
        if ( !v198 )
          goto LABEL_34;
      }
      else if ( v147 == (unsigned int)sub_C444A0(&v198) )
      {
        goto LABEL_34;
      }
LABEL_27:
      v20 = sub_986520((__int64)v192);
      sub_9B0110(&v200, *(_QWORD *)(v20 + 32), a2, v191, a4);
      v21 = *v192;
      if ( (unsigned __int8)v21 <= 0x1Cu )
        v22 = *((unsigned __int16 *)v192 + 1);
      else
        v22 = v21 - 29;
      if ( v22 == 20 )
      {
        sub_C778B0(&v204, &v196, 0);
        sub_984AC0(&v196, v204.m128i_i64);
        sub_969240(v205.m128i_i64);
        sub_969240(v204.m128i_i64);
        sub_C778B0(&v204, &v200, 0);
        sub_984AC0((__int64 *)&v200, v204.m128i_i64);
        sub_969240(v205.m128i_i64);
        sub_969240(v204.m128i_i64);
      }
      v23 = sub_C77420(&v196, &v200);
      LODWORD(v6) = BYTE1(v23);
      if ( BYTE1(v23) )
        LODWORD(v6) = v23;
      sub_969240(&v202);
      sub_969240((__int64 *)&v200);
LABEL_34:
      sub_969240((__int64 *)&v198);
      sub_969240(&v196);
      return (unsigned int)v6;
    case 25:
      if ( a4[4].m128i_i8[0] && ((v5[1] & 2) != 0 || ((v5[1] >> 1) & 2) != 0) )
        goto LABEL_18;
      sub_9878D0((__int64)&v204, v8);
      v47 = (_QWORD *)sub_986520((__int64)v192);
      sub_9AB8E0(*v47, a2, &v204, v191, a4);
      LOBYTE(v48) = sub_986C60(v205.m128i_i64, 0);
      LODWORD(v6) = v48;
      if ( !(_BYTE)v48 )
        goto LABEL_42;
      goto LABEL_43;
    case 26:
    case 27:
      v15 = (_QWORD *)sub_986520((__int64)v5);
      v16 = v191;
      if ( (v5[1] & 2) != 0 )
        goto LABEL_20;
      sub_9B0110(&v204, *v15, a2, v191, a4);
      LOBYTE(v24) = sub_986C60(v205.m128i_i64, v205.m128i_i32[2] - 1);
      LODWORD(v6) = v24;
      if ( !(_BYTE)v24 )
LABEL_42:
        LODWORD(v6) = sub_9B4110(v192, a2, v191, a4, &v204);
LABEL_43:
      sub_969240(v205.m128i_i64);
      sub_969240(v204.m128i_i64);
      return (unsigned int)v6;
    case 29:
      v49 = (_BYTE **)sub_986520((__int64)v5);
      if ( (unsigned __int8)sub_985700(*v49, v49[4]) )
        goto LABEL_94;
      v50 = sub_986520((__int64)v192);
      if ( (unsigned __int8)sub_9A6530(*(_QWORD *)(v50 + 32), a2, a4, v191) )
        goto LABEL_94;
      goto LABEL_120;
    case 30:
      v11 = (_BYTE **)sub_986520((__int64)v5);
      if ( !(unsigned __int8)sub_985700(*v11, v11[4]) )
        goto LABEL_9;
      goto LABEL_94;
    case 31:
      v110 = *((_QWORD *)v5 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v110 + 8) - 17 <= 1 )
        v110 = **(_QWORD **)(v110 + 16);
      LOBYTE(v6) = *(_DWORD *)(v110 + 8) >> 8 == 0;
      return (unsigned int)v6;
    case 32:
      v6 = *((_QWORD *)v5 + 1);
      v107 = a4[4].m128i_i8[0];
      if ( *(_BYTE *)(v6 + 8) != 14 )
      {
        if ( v107 )
        {
          if ( (v5[7] & 0x20) != 0 )
          {
            v144 = sub_B91C10(v5, 4);
            if ( v144 )
            {
              sub_9691E0((__int64)&v204, v8, 0, 0, 0);
              LODWORD(v6) = sub_984D50(v144, (__int64)&v204);
              sub_969240(v204.m128i_i64);
              return (unsigned int)v6;
            }
          }
        }
        goto LABEL_122;
      }
      if ( !v107 || (v5[7] & 0x20) == 0 )
        goto LABEL_122;
      if ( !sub_B91C10(v5, 11) )
      {
        if ( a4[4].m128i_i8[0] && (v5[7] & 0x20) != 0 && sub_B91C10(v5, 12) )
        {
          v108 = *(_DWORD *)(v6 + 8);
          v109 = sub_B43CB0(v5);
          LODWORD(v6) = sub_B2F070(v109, v108 >> 8) ^ 1;
          return (unsigned int)v6;
        }
LABEL_122:
        LODWORD(v6) = 0;
        return (unsigned int)v6;
      }
      goto LABEL_94;
    case 34:
      if ( *(_BYTE *)(*((_QWORD *)v5 + 1) + 8LL) != 14 )
        goto LABEL_9;
      v93 = 0;
      v190 = v191;
      if ( v9 > 0x1Cu )
        v93 = sub_B43CB0(v5);
      if ( ((v5[1] >> 1) & 4) == 0 )
      {
        if ( (v5[1] & 2) == 0 )
          goto LABEL_122;
        v94 = *(_QWORD *)(*(_QWORD *)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)] + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v94 + 8) - 17 <= 1 )
          v94 = **(_QWORD **)(v94 + 16);
        if ( (unsigned __int8)sub_B2F070(v93, *(_DWORD *)(v94 + 8) >> 8) )
          goto LABEL_122;
      }
      LODWORD(v6) = sub_9B6260(*(_QWORD *)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)], a4, v191);
      if ( (_BYTE)v6 )
        goto LABEL_94;
      if ( (v5[7] & 0x40) != 0 )
        v96 = (unsigned __int8 *)*((_QWORD *)v5 - 1);
      else
        v96 = &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
      v97 = v96 + 32;
      v98 = sub_BB5290(v5, a4, v95) & 0xFFFFFFFFFFFFFFF9LL | 4;
      if ( (v5[7] & 0x40) != 0 )
        v5 = (unsigned __int8 *)(*((_QWORD *)v5 - 1) + 32LL * (*((_DWORD *)v5 + 1) & 0x7FFFFFF));
      if ( v97 == v5 )
        return (unsigned int)v6;
      while ( 2 )
      {
        v99 = a4->m128i_i64[0];
        v100 = v98 & 0xFFFFFFFFFFFFFFF8LL;
        v101 = v98 & 0xFFFFFFFFFFFFFFF8LL;
        v185 = (v98 >> 1) & 3;
        if ( v98 )
        {
          if ( ((v98 >> 1) & 3) == 0 )
          {
            if ( v100 )
            {
              v102 = *(_QWORD **)(*(_QWORD *)v97 + 24LL);
              if ( *(_DWORD *)(*(_QWORD *)v97 + 32LL) > 0x40u )
                v102 = (_QWORD *)*v102;
              v186 = v98 & 0xFFFFFFFFFFFFFFF8LL;
              v103 = 16LL * (unsigned int)v102 + sub_AE4AC0(v99, v98 & 0xFFFFFFFFFFFFFFF8LL) + 24;
              v104 = *(_QWORD *)v103;
              LOBYTE(v103) = *(_BYTE *)(v103 + 8);
              v204.m128i_i64[0] = v104;
              v204.m128i_i8[8] = v103;
              if ( sub_CA1930(&v204) )
              {
LABEL_94:
                LODWORD(v6) = 1;
                return (unsigned int)v6;
              }
              v105 = v186;
              goto LABEL_147;
            }
            goto LABEL_332;
          }
          if ( v185 == 2 )
          {
            v153 = v98 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v100 )
            {
LABEL_303:
              v166 = v100;
              v167 = v98;
              v172 = v99;
              v179 = sub_AE5020(v99, v153);
              v154.m128i_i64[0] = sub_9208B0(v172, v153);
              v105 = v166;
              v204 = v154;
              v155 = v167;
              v156 = ((1LL << v179) + ((unsigned __int64)(v154.m128i_i64[0] + 7) >> 3) - 1) >> v179 << v179;
              goto LABEL_304;
            }
LABEL_332:
            v170 = a4->m128i_i64[0];
            v176 = v98;
            v182 = v98 & 0xFFFFFFFFFFFFFFF8LL;
            v162 = sub_BCBAE0(v98 & 0xFFFFFFFFFFFFFFF8LL, *(_QWORD *)v97);
            v100 = v182;
            v98 = v176;
            v99 = v170;
            v153 = v162;
            goto LABEL_303;
          }
          if ( v185 != 1 )
            goto LABEL_332;
          if ( v100 )
          {
            v153 = *(_QWORD *)(v100 + 24);
          }
          else
          {
            v177 = a4->m128i_i64[0];
            v183 = v98;
            v165 = sub_BCBAE0(0, *(_QWORD *)v97);
            v98 = v183;
            v99 = v177;
            v100 = 0;
            v153 = v165;
          }
        }
        else
        {
          v171 = a4->m128i_i64[0];
          v163 = sub_BCBAE0(v100, *(_QWORD *)v97);
          v100 = 0;
          v98 = 0;
          v99 = v171;
          v153 = v163;
          if ( v185 != 1 )
            goto LABEL_303;
        }
        v174 = v100;
        v181 = v98;
        v158.m128i_i64[0] = sub_9208B0(v99, v153);
        v155 = v181;
        v105 = v174;
        v204 = v158;
        v156 = (unsigned __int64)(v158.m128i_i64[0] + 7) >> 3;
LABEL_304:
        if ( v156 )
        {
          v157 = *(_QWORD *)v97;
          if ( **(_BYTE **)v97 == 17 )
          {
            if ( *(_DWORD *)(v157 + 32) <= 0x40u )
            {
              if ( *(_QWORD *)(v157 + 24) )
                goto LABEL_94;
            }
            else
            {
              v173 = *(_DWORD *)(v157 + 32);
              v168 = v105;
              v180 = v155;
              if ( v173 != (unsigned int)sub_C444A0(v157 + 24) )
                goto LABEL_94;
              v155 = v180;
              v105 = v168;
            }
          }
          else
          {
            v169 = v105;
            v175 = v155;
            v159 = (unsigned int *)sub_C94E20(qword_4F862D0);
            v160 = v190 + 1;
            v155 = v175;
            v105 = v169;
            if ( v159 )
              v161 = *v159;
            else
              v161 = qword_4F862D0[2];
            if ( v190 >= v161 )
            {
              ++v190;
            }
            else
            {
              ++v190;
              if ( (unsigned __int8)sub_9B6260(*(_QWORD *)v97, a4, v160) )
                goto LABEL_94;
              v155 = v175;
              v105 = v169;
            }
          }
        }
        if ( !v155 )
          goto LABEL_147;
        if ( v185 == 2 )
        {
          if ( !v105 )
LABEL_147:
            v101 = sub_BCBAE0(v105, *(_QWORD *)v97);
        }
        else
        {
          if ( v185 != 1 || !v105 )
            goto LABEL_147;
          v101 = *(_QWORD *)(v105 + 24);
        }
        v106 = *(_BYTE *)(v101 + 8);
        if ( v106 == 16 )
        {
          v98 = *(_QWORD *)(v101 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
        }
        else if ( (unsigned int)v106 - 17 > 1 )
        {
          if ( v106 != 15 )
            v101 = 0;
          v98 = v101 & 0xFFFFFFFFFFFFFFF9LL;
        }
        else
        {
          v98 = v101 & 0xFFFFFFFFFFFFFFF9LL | 2;
        }
        v97 += 32;
        if ( v5 == v97 )
          return (unsigned int)v6;
        continue;
      }
    case 38:
      if ( v9 != 67 || ((v5[1] >> 1) & 2) == 0 && (v5[1] & 2) == 0 )
        goto LABEL_9;
      v111 = *((_QWORD *)v5 - 4);
LABEL_167:
      LODWORD(v6) = sub_9A6530(v111, a2, a4, v191);
      return (unsigned int)v6;
    case 39:
    case 40:
LABEL_18:
      v14 = v5;
      goto LABEL_19;
    case 47:
      if ( *(_BYTE *)(*((_QWORD *)v5 + 1) + 8LL) != 18 )
      {
        v90 = sub_986520((__int64)v5);
        v200 = (unsigned __int8 **)sub_9208B0(a4->m128i_i64[0], *(_QWORD *)(*(_QWORD *)v90 + 8LL));
        v201 = v91;
        v92.m128i_i64[0] = sub_9208B0(a4->m128i_i64[0], *((_QWORD *)v192 + 1));
        v204 = v92;
        if ( (unsigned __int64)v200 <= v92.m128i_i64[0] )
          goto LABEL_120;
      }
      goto LABEL_9;
    case 48:
      if ( *(_BYTE *)(*((_QWORD *)v5 + 1) + 8LL) == 18 )
        goto LABEL_9;
      v86 = sub_986520((__int64)v5);
      v200 = (unsigned __int8 **)sub_9208B0(a4->m128i_i64[0], *(_QWORD *)(*(_QWORD *)v86 + 8LL));
      v201 = v87;
      v88.m128i_i64[0] = sub_9208B0(a4->m128i_i64[0], *((_QWORD *)v192 + 1));
      v204 = v88;
      if ( v88.m128i_i64[0] < (unsigned __int64)v200 )
        goto LABEL_9;
LABEL_120:
      v14 = v192;
LABEL_19:
      v15 = (_QWORD *)sub_986520((__int64)v14);
      v16 = v191;
LABEL_20:
      LODWORD(v6) = sub_9A6530(*v15, a2, a4, v16);
      return (unsigned int)v6;
    case 49:
      v80 = *(_QWORD *)(*(_QWORD *)sub_986520((__int64)v5) + 8LL);
      v81 = *(unsigned __int8 *)(v80 + 8);
      if ( (unsigned int)(v81 - 17) > 1 )
      {
        if ( (v81 & 0xFD) != 0xC )
          goto LABEL_9;
        v82 = a4->m128i_i64[0];
        v6 = v80;
      }
      else
      {
        v6 = **(_QWORD **)(v80 + 16);
        if ( (*(_BYTE *)(v6 + 8) & 0xFD) != 0xC )
          goto LABEL_9;
        v82 = a4->m128i_i64[0];
        if ( (unsigned __int8)(v81 - 17) >= 2u )
          v6 = v80;
      }
      v83 = sub_BCB060(v6);
      if ( !v83 )
        v83 = sub_AE43A0(v82, v6);
      v84 = v192;
      if ( !(v8 % v83) )
      {
LABEL_117:
        v85 = (_QWORD *)sub_986520((__int64)v84);
        LODWORD(v6) = sub_9B6260(*v85, a4, v191);
        return (unsigned int)v6;
      }
      goto LABEL_9;
    case 55:
      v51 = a4[4].m128i_i8[0] == 0;
      v193 = v5;
      if ( v51 )
        goto LABEL_96;
      v194 = 0;
      v195 = 0;
      v196 = 0;
      LODWORD(v6) = sub_990E50((__int64)v5, &v194, &v195, &v196);
      if ( !(_BYTE)v6 )
        goto LABEL_96;
      v125 = (_BYTE *)v195;
      if ( *(_BYTE *)v195 != 17 )
      {
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v195 + 8) + 8LL) - 17 > 1 )
          goto LABEL_96;
        if ( *(_BYTE *)v195 > 0x15u )
          goto LABEL_96;
        v151 = (_BYTE *)sub_AD7630(v195, 0);
        v125 = v151;
        if ( !v151 || *v151 != 17 )
          goto LABEL_96;
      }
      v126 = *((_DWORD *)v125 + 8);
      if ( v126 <= 0x40 ? *((_QWORD *)v125 + 3) == 0 : v126 == (unsigned int)sub_C444A0(v125 + 24) )
        goto LABEL_96;
      v128 = v194;
      v129 = *v194;
      if ( v129 == 54 )
      {
        if ( (unsigned __int8)sub_B448F0(v194) || (unsigned __int8)sub_B44900(v128) )
          return (unsigned int)v6;
      }
      else if ( (unsigned int)(v129 - 29) > 0x19 )
      {
        if ( (unsigned int)(v129 - 55) <= 1 )
        {
          LODWORD(v6) = sub_B44E60(v194);
          if ( (_BYTE)v6 )
            return (unsigned int)v6;
        }
      }
      else if ( v129 == 42 )
      {
        if ( (unsigned __int8)sub_B448F0(v194) )
          return (unsigned int)v6;
        if ( (unsigned __int8)sub_B44900(v128) )
        {
          v204.m128i_i8[8] = 0;
          v204.m128i_i64[0] = (__int64)&v200;
          if ( (unsigned __int8)sub_991580((__int64)&v204, v196) )
          {
            v164 = sub_986C60((__int64 *)v125 + 3, *((_DWORD *)v125 + 8) - 1);
            if ( v164 == sub_986C60((__int64 *)v200, *((_DWORD *)v200 + 2) - 1) )
              return (unsigned int)v6;
          }
        }
      }
      else if ( v129 == 46 && ((unsigned __int8)sub_B448F0(v194) || (unsigned __int8)sub_B44900(v128)) )
      {
        v204.m128i_i8[8] = 0;
        v204.m128i_i64[0] = (__int64)&v200;
        LODWORD(v6) = sub_991580((__int64)&v204, v196);
        if ( (_BYTE)v6 )
        {
          v130 = *((_DWORD *)v200 + 2);
          if ( !(v130 <= 0x40 ? *v200 == 0 : v130 == (unsigned int)sub_C444A0(v200)) )
            return (unsigned int)v6;
        }
      }
LABEL_96:
      v52 = _mm_loadu_si128(a4 + 1);
      v53 = _mm_loadu_si128(a4 + 2);
      v54 = _mm_loadu_si128(a4 + 3).m128i_u64[0];
      v55 = a4[4].m128i_i64[0];
      v204 = _mm_loadu_si128(a4);
      v207 = v54;
      v208 = v55;
      v205 = v52;
      v206 = v53;
      v56 = (int *)sub_C94E20(qword_4F862D0);
      v58 = &v204;
      if ( v56 )
        v59 = *v56;
      else
        v59 = qword_4F862D0[2];
      v60 = v59 - 1;
      v61 = v193;
      if ( v60 < v191 )
        v60 = v191;
      LODWORD(v196) = v60;
      v62 = 32LL * (*((_DWORD *)v193 + 1) & 0x7FFFFFF);
      if ( (v193[7] & 0x40) != 0 )
      {
        v63 = (unsigned __int8 *)*((_QWORD *)v193 - 1);
        v61 = &v63[v62];
      }
      else
      {
        v63 = &v193[-v62];
      }
      v64 = a2;
      v65 = &v193;
      v66 = v62 >> 7;
      v201 = &v204;
      v200 = &v193;
      v202 = a2;
      v203 = &v196;
      if ( !v66 )
      {
LABEL_267:
        v150 = v61 - v63;
        if ( v61 - v63 != 64 )
        {
          if ( v150 != 96 )
          {
            if ( v150 != 32 )
              goto LABEL_94;
            goto LABEL_270;
          }
          if ( !(unsigned __int8)sub_9B4BB0(&v200, v63, v65, v64, v57, v58) )
          {
LABEL_109:
            LOBYTE(v6) = v63 == v61;
            return (unsigned int)v6;
          }
          v63 += 32;
        }
        if ( !(unsigned __int8)sub_9B4BB0(&v200, v63, v65, v64, v57, v58) )
        {
LABEL_271:
          LOBYTE(v6) = v61 == v63;
          return (unsigned int)v6;
        }
        v63 += 32;
LABEL_270:
        LODWORD(v6) = sub_9B4BB0(&v200, v63, v65, v64, v57, v58);
        if ( (_BYTE)v6 )
          return (unsigned int)v6;
        goto LABEL_271;
      }
      v67 = &v63[128 * v66];
      while ( 1 )
      {
        if ( !(unsigned __int8)sub_9B4BB0(&v200, v63, v65, v64, v57, v58) )
          goto LABEL_109;
        v6 = (__int64)(v63 + 32);
        if ( !(unsigned __int8)sub_9B4BB0(&v200, v63 + 32, v76, v77, v78, v79) )
          goto LABEL_245;
        v6 = (__int64)(v63 + 64);
        if ( !(unsigned __int8)sub_9B4BB0(&v200, v63 + 64, v68, v69, v70, v71) )
          break;
        v6 = (__int64)(v63 + 96);
        if ( !(unsigned __int8)sub_9B4BB0(&v200, v63 + 96, v72, v73, v74, v75) )
        {
LABEL_245:
          LOBYTE(v6) = v61 == (unsigned __int8 *)v6;
          return (unsigned int)v6;
        }
        v63 += 128;
        if ( v67 == v63 )
          goto LABEL_267;
      }
      LOBYTE(v6) = v6 == (_QWORD)v61;
      return (unsigned int)v6;
    case 57:
      v205.m128i_i64[0] = (__int64)a4;
      v204.m128i_i64[0] = (__int64)&v192;
      v204.m128i_i64[1] = a2;
      v205.m128i_i64[1] = (__int64)&v191;
      if ( (unsigned __int8)sub_9B4A70(&v204, 1) )
      {
        LODWORD(v6) = sub_9B4A70(&v204, 0);
        if ( (_BYTE)v6 )
          return (unsigned int)v6;
      }
      goto LABEL_9;
    case 61:
      if ( v9 != 90 )
        goto LABEL_9;
      v6 = *((_QWORD *)v5 - 8);
      v34 = *((_QWORD *)v5 - 4);
      v35 = *(_QWORD *)(v6 + 8);
      v36 = *(_BYTE *)(v35 + 8);
      if ( *(_BYTE *)v34 == 17 )
      {
        if ( v36 != 17 )
          goto LABEL_9;
        v37 = *(_DWORD *)(v35 + 32);
        sub_9691E0((__int64)&v200, v37, -1, 1u, 0);
        if ( !sub_986EE0(v34 + 24, v37) )
          goto LABEL_72;
        v38 = *(_QWORD **)(v34 + 24);
        if ( *(_DWORD *)(v34 + 32) > 0x40u )
          v38 = (_QWORD *)*v38;
        v189 = (unsigned int)v38;
        v204.m128i_i32[2] = v37;
        v39 = 1LL << (char)v38;
        if ( v37 > 0x40 )
        {
          sub_C43690(&v204, 0, 0);
          if ( v204.m128i_i32[2] > 0x40u )
          {
            *(_QWORD *)(v204.m128i_i64[0] + 8LL * (v189 >> 6)) |= v39;
            goto LABEL_71;
          }
        }
        else
        {
          v204.m128i_i64[0] = 0;
        }
        v204.m128i_i64[0] |= v39;
LABEL_71:
        sub_984320((__int64 *)&v200, v204.m128i_i64);
        sub_969240(v204.m128i_i64);
LABEL_72:
        LODWORD(v6) = sub_9A6530(v6, &v200, a4, v191);
        sub_969240((__int64 *)&v200);
        return (unsigned int)v6;
      }
      if ( v36 == 17 )
      {
        sub_9691E0((__int64)&v200, *(_DWORD *)(v35 + 32), -1, 1u, 0);
        goto LABEL_72;
      }
LABEL_9:
      sub_9878D0((__int64)&v204, v8);
      sub_9AB8E0(v192, a2, &v204, v191, a4);
      v12 = v205.m128i_i32[2];
      if ( v205.m128i_i32[2] > 0x40u )
      {
        if ( v12 - (unsigned int)sub_C444A0(&v205) <= 0x40 )
        {
          LOBYTE(v6) = *(_QWORD *)v205.m128i_i64[0] != 0;
        }
        else
        {
          LODWORD(v6) = 1;
          if ( !v205.m128i_i64[0] )
          {
LABEL_12:
            if ( v204.m128i_i32[2] > 0x40u && v204.m128i_i64[0] )
              j_j___libc_free_0_0(v204.m128i_i64[0]);
            return (unsigned int)v6;
          }
        }
        j_j___libc_free_0_0(v205.m128i_i64[0]);
        goto LABEL_12;
      }
      LOBYTE(v6) = v205.m128i_i64[0] != 0;
      goto LABEL_12;
    case 62:
      if ( *(_BYTE *)(*((_QWORD *)v5 + 1) + 8LL) == 18 )
        goto LABEL_9;
      v26 = (__int64 *)sub_986520((__int64)v5);
      v27 = v26[8];
      v28 = *v26;
      v29 = v26[4];
      if ( *(_BYTE *)v27 == 17 )
      {
        v30 = *(_DWORD *)(a2 + 8);
        sub_9865C0((__int64)&v204, a2);
        if ( sub_986EE0(v27 + 24, v30) )
        {
          v31 = *(_QWORD **)(v27 + 24);
          if ( *(_DWORD *)(v27 + 32) > 0x40u )
            v31 = (_QWORD *)*v31;
          sub_987130(v204.m128i_i64, (unsigned int)v31);
          v32 = *(_DWORD *)(v27 + 32) <= 0x40u ? *(_QWORD *)(v27 + 24) : **(_QWORD **)(v27 + 24);
          if ( !sub_986C60((__int64 *)a2, v32) )
            goto LABEL_59;
        }
      }
      else
      {
        sub_9865C0((__int64)&v204, a2);
      }
      LODWORD(v6) = sub_9B6260(v29, a4, v191);
      if ( !(_BYTE)v6 )
        goto LABEL_61;
LABEL_59:
      LOBYTE(v33) = sub_9867B0((__int64)&v204);
      LODWORD(v6) = v33;
      if ( !(_BYTE)v33 )
        LODWORD(v6) = sub_9A6530(v28, &v204, a4, v191);
LABEL_61:
      sub_969240(v204.m128i_i64);
      return (unsigned int)v6;
    case 63:
      if ( v9 != 92 )
        goto LABEL_9;
      LODWORD(v201) = 1;
      v200 = 0;
      v204.m128i_i32[2] = 1;
      v204.m128i_i64[0] = 0;
      if ( !(unsigned __int8)sub_984B30((__int64)v5, a2, (__int64)&v200, (__int64)&v204) )
      {
        sub_969240(v204.m128i_i64);
        sub_969240((__int64 *)&v200);
        goto LABEL_9;
      }
      if ( sub_9867B0((__int64)&v204) || (LODWORD(v6) = sub_9A6530(*((_QWORD *)v5 - 4), &v204, a4, v191), (_BYTE)v6) )
      {
        LOBYTE(v25) = sub_9867B0((__int64)&v200);
        LODWORD(v6) = v25;
        if ( !(_BYTE)v25 )
          LODWORD(v6) = sub_9A6530(*((_QWORD *)v5 - 8), &v200, a4, v191);
      }
      sub_969240(v204.m128i_i64);
      sub_969240((__int64 *)&v200);
      return (unsigned int)v6;
    case 64:
      if ( v9 != 93 )
        goto LABEL_9;
      if ( *((_DWORD *)v5 + 20) != 1 )
        goto LABEL_9;
      if ( **((_DWORD **)v5 + 9) )
        goto LABEL_9;
      v45 = (unsigned __int8 *)*((_QWORD *)v5 - 4);
      if ( !sub_988010((__int64)v45) )
        goto LABEL_9;
      v46 = sub_987FE0((__int64)v45);
      if ( v46 != 312 )
      {
        switch ( v46 )
        {
          case 333:
          case 339:
          case 360:
          case 369:
          case 372:
            break;
          default:
            goto LABEL_9;
        }
      }
      if ( !v45 )
        goto LABEL_9;
      v121 = sub_B5B5E0(v45);
      switch ( v121 )
      {
        case 15:
          v152 = *((_DWORD *)v45 + 1) & 0x7FFFFFF;
          v140 = 32 * (1 - v152);
          v139 = -32 * v152;
LABEL_223:
          v120 = *(_QWORD *)&v45[v139];
          v119 = *(_QWORD *)&v45[v140];
LABEL_173:
          LODWORD(v6) = sub_9B6140(a2, v191, a4, v120, v119);
          break;
        case 17:
          v117 = sub_9B4D50(
                   a2,
                   v191,
                   (_DWORD)a4,
                   v8,
                   *(_QWORD *)&v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)],
                   *(_QWORD *)&v45[32 * (1LL - (*((_DWORD *)v45 + 1) & 0x7FFFFFF))],
                   0,
                   0);
LABEL_171:
          LODWORD(v6) = v117;
          break;
        case 13:
          v122 = *((_DWORD *)v45 + 1) & 0x7FFFFFF;
          v123 = *(unsigned __int8 **)&v45[32 * (1 - v122)];
          v124 = *(unsigned __int8 **)&v45[-32 * v122];
          v187 = v123;
          LODWORD(v6) = sub_985700(v124, v123);
          if ( !(_BYTE)v6 )
          {
            v44 = sub_9A3DF0(a2, v191, (__int64)a4, v8, v124, v187, 0, 0);
LABEL_80:
            LODWORD(v6) = v44;
          }
          break;
        default:
          goto LABEL_9;
      }
      return (unsigned int)v6;
    case 67:
      v89 = (_QWORD *)sub_986520((__int64)v5);
      if ( !(unsigned __int8)sub_9B6260(*v89, a4, v191) )
        goto LABEL_122;
      v145 = (unsigned __int8 **)sub_986520((__int64)v192);
      LOBYTE(v146) = sub_98ED70(*v145, a4[2].m128i_i64[0], a4[2].m128i_i64[1], a4[1].m128i_i64[1], v191);
      LODWORD(v6) = v146;
      return (unsigned int)v6;
    default:
      goto LABEL_9;
  }
}
