// Function: sub_2133C10
// Address: 0x2133c10
//
__int64 __fastcall sub_2133C10(
        __int64 a1,
        unsigned __int64 *a2,
        __int64 a3,
        unsigned int *a4,
        __int64 a5,
        __m128i a6,
        __m128i a7,
        __m128i a8)
{
  unsigned __int64 v13; // rsi
  __int64 v14; // rdx
  unsigned int v15; // eax
  unsigned __int64 v16; // rcx
  int v17; // edx
  unsigned int v18; // ebx
  __int64 v19; // r10
  __int64 v20; // rdi
  __int64 v21; // rax
  unsigned __int8 v22; // cl
  __int64 v23; // r8
  unsigned __int64 v24; // rax
  const void **v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned int v28; // edx
  __int64 v29; // rdx
  unsigned __int8 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rdi
  unsigned int v33; // ebx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // rax
  __int64 *v37; // r10
  __int64 *v38; // rbx
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  const void **v41; // rdx
  unsigned int v42; // edx
  int v43; // eax
  __int64 *v44; // rbx
  int v45; // edx
  __int64 v46; // rdi
  unsigned int v47; // ebx
  int v48; // eax
  __int64 result; // rax
  __int64 v50; // rdi
  unsigned int v51; // ebx
  bool v52; // al
  const void ***v53; // rax
  __int64 *v54; // rax
  int v55; // edx
  __int64 v56; // rdx
  const void ***v57; // rax
  int v58; // edx
  const void ***v59; // rax
  unsigned int v60; // edx
  int v61; // edx
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdi
  unsigned int v65; // ebx
  int v66; // eax
  bool v67; // al
  unsigned __int8 *v68; // rax
  __int64 v69; // rsi
  __int64 v70; // r8
  __int64 v71; // rbx
  __int64 v72; // rax
  unsigned int v73; // r15d
  __int64 v74; // r14
  __int64 v75; // rdx
  _BYTE *v76; // rax
  __int64 v77; // r13
  __int64 v78; // r12
  __int64 v79; // rax
  __int64 v80; // rdi
  unsigned int v81; // eax
  __m128i v82; // xmm0
  __int32 v83; // eax
  unsigned __int8 *v84; // rax
  unsigned int v85; // ebx
  unsigned int v86; // eax
  __int64 v87; // rdx
  const void ***v88; // rax
  int v89; // edx
  __int64 v90; // r9
  __int64 *v91; // rbx
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // r8
  __int128 v95; // rax
  __int64 *v96; // rbx
  __int64 (__fastcall *v97)(__int64 *, __int64, __int64, _QWORD, __int64); // r15
  __int64 v98; // rax
  unsigned int v99; // eax
  const void **v100; // rdx
  unsigned int v101; // edx
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rdx
  unsigned int v105; // edx
  __int64 v106; // rax
  __int64 v107; // rdi
  unsigned int v108; // ebx
  bool v109; // al
  int v110; // edx
  __int64 v111; // rdi
  unsigned int v112; // ebx
  bool v113; // al
  const void ***v114; // rax
  int v115; // edx
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 *v118; // rax
  __int64 v119; // r10
  int v120; // edx
  unsigned int v121; // edx
  unsigned int v122; // esi
  int v123; // eax
  unsigned __int64 v124; // rax
  const void **v125; // rdx
  __int64 *v126; // rax
  int v127; // edx
  __int128 v128; // [rsp-20h] [rbp-220h]
  __int128 v129; // [rsp-20h] [rbp-220h]
  __int64 v130; // [rsp-10h] [rbp-210h]
  unsigned int v131; // [rsp+8h] [rbp-1F8h]
  unsigned int v132; // [rsp+10h] [rbp-1F0h]
  __int64 v133; // [rsp+18h] [rbp-1E8h]
  __int64 v134; // [rsp+20h] [rbp-1E0h]
  unsigned __int64 *v135; // [rsp+28h] [rbp-1D8h]
  __int64 **v136; // [rsp+30h] [rbp-1D0h]
  __int64 *v137; // [rsp+38h] [rbp-1C8h]
  unsigned __int8 v138; // [rsp+40h] [rbp-1C0h]
  __int64 *v139; // [rsp+40h] [rbp-1C0h]
  __int64 v140; // [rsp+48h] [rbp-1B8h]
  __int64 v141; // [rsp+48h] [rbp-1B8h]
  __int64 v142; // [rsp+50h] [rbp-1B0h]
  __int64 *v143; // [rsp+50h] [rbp-1B0h]
  unsigned int v144; // [rsp+50h] [rbp-1B0h]
  __int64 *v145; // [rsp+50h] [rbp-1B0h]
  __int64 v146; // [rsp+50h] [rbp-1B0h]
  __int64 *v147; // [rsp+50h] [rbp-1B0h]
  __int64 v148; // [rsp+58h] [rbp-1A8h]
  __int64 *v149; // [rsp+60h] [rbp-1A0h]
  unsigned __int64 v150; // [rsp+68h] [rbp-198h]
  unsigned int v151; // [rsp+70h] [rbp-190h]
  unsigned int v152; // [rsp+74h] [rbp-18Ch]
  __int64 *v153; // [rsp+78h] [rbp-188h]
  __int64 *v154; // [rsp+78h] [rbp-188h]
  __int64 v155; // [rsp+78h] [rbp-188h]
  __int64 v156; // [rsp+78h] [rbp-188h]
  __int64 v157; // [rsp+80h] [rbp-180h]
  __int64 v158; // [rsp+80h] [rbp-180h]
  __int64 v159; // [rsp+80h] [rbp-180h]
  unsigned __int64 v160; // [rsp+88h] [rbp-178h]
  __int128 v162; // [rsp+90h] [rbp-170h]
  __int64 *v163; // [rsp+90h] [rbp-170h]
  __int64 *v164; // [rsp+110h] [rbp-F0h]
  int v165; // [rsp+138h] [rbp-C8h]
  __m128i v166; // [rsp+150h] [rbp-B0h] BYREF
  __m128i v167; // [rsp+160h] [rbp-A0h] BYREF
  __int128 v168; // [rsp+170h] [rbp-90h] BYREF
  __int128 v169; // [rsp+180h] [rbp-80h] BYREF
  _QWORD v170[4]; // [rsp+190h] [rbp-70h] BYREF
  _BYTE v171[16]; // [rsp+1B0h] [rbp-50h] BYREF
  __int64 v172; // [rsp+1C0h] [rbp-40h]

  v13 = *a2;
  v14 = a2[1];
  v166.m128i_i64[0] = 0;
  v166.m128i_i32[2] = 0;
  v167.m128i_i64[0] = 0;
  v167.m128i_i32[2] = 0;
  *(_QWORD *)&v168 = 0;
  DWORD2(v168) = 0;
  *(_QWORD *)&v169 = 0;
  DWORD2(v169) = 0;
  sub_20174B0(a1, v13, v14, &v166, &v167);
  sub_20174B0(a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), &v168, &v169);
  v15 = *a4;
  if ( *a4 == 17 || v15 == 22 )
  {
    if ( (_QWORD)v169 == (_QWORD)v168
      && DWORD2(v169) == DWORD2(v168)
      && ((v110 = *(unsigned __int16 *)(v168 + 24), v110 == 10) || v110 == 32)
      && ((v111 = *(_QWORD *)(v168 + 88), v112 = *(_DWORD *)(v111 + 32), v112 <= 0x40)
        ? (v113 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v112) == *(_QWORD *)(v111 + 24))
        : (v113 = v112 == (unsigned int)sub_16A58F0(v111 + 24)),
          v113) )
    {
      v114 = (const void ***)(*(_QWORD *)(v166.m128i_i64[0] + 40) + 16LL * v166.m128i_u32[2]);
      *a2 = (unsigned __int64)sub_1D332F0(
                                *(__int64 **)(a1 + 8),
                                118,
                                a5,
                                *(unsigned __int8 *)v114,
                                v114[1],
                                0,
                                *(double *)a6.m128i_i64,
                                *(double *)a7.m128i_i64,
                                a8,
                                v166.m128i_i64[0],
                                v166.m128i_u64[1],
                                *(_OWORD *)&v167);
      *((_DWORD *)a2 + 2) = v115;
      *(_QWORD *)a3 = v168;
      result = DWORD2(v168);
      *(_DWORD *)(a3 + 8) = DWORD2(v168);
    }
    else
    {
      v53 = (const void ***)(*(_QWORD *)(v166.m128i_i64[0] + 40) + 16LL * v166.m128i_u32[2]);
      v54 = sub_1D332F0(
              *(__int64 **)(a1 + 8),
              120,
              a5,
              *(unsigned __int8 *)v53,
              v53[1],
              0,
              *(double *)a6.m128i_i64,
              *(double *)a7.m128i_i64,
              a8,
              v166.m128i_i64[0],
              v166.m128i_u64[1],
              v168);
      v165 = v55;
      v56 = v166.m128i_i64[0];
      *a2 = (unsigned __int64)v54;
      *((_DWORD *)a2 + 2) = v165;
      v57 = (const void ***)(*(_QWORD *)(v56 + 40) + 16LL * v166.m128i_u32[2]);
      *(_QWORD *)a3 = sub_1D332F0(
                        *(__int64 **)(a1 + 8),
                        120,
                        a5,
                        *(unsigned __int8 *)v57,
                        v57[1],
                        0,
                        *(double *)a6.m128i_i64,
                        *(double *)a7.m128i_i64,
                        a8,
                        v167.m128i_i64[0],
                        v167.m128i_u64[1],
                        v169);
      *(_DWORD *)(a3 + 8) = v58;
      v59 = (const void ***)(*(_QWORD *)(*a2 + 40) + 16LL * *((unsigned int *)a2 + 2));
      v164 = sub_1D332F0(
               *(__int64 **)(a1 + 8),
               119,
               a5,
               *(unsigned __int8 *)v59,
               v59[1],
               0,
               *(double *)a6.m128i_i64,
               *(double *)a7.m128i_i64,
               a8,
               *a2,
               a2[1],
               *(_OWORD *)a3);
      *a2 = (unsigned __int64)v164;
      *((_DWORD *)a2 + 2) = v60;
      *(_QWORD *)a3 = sub_1D38BB0(
                        *(_QWORD *)(a1 + 8),
                        0,
                        a5,
                        *(unsigned __int8 *)(v164[5] + 16LL * v60),
                        *(const void ***)(v164[5] + 16LL * v60 + 8),
                        0,
                        a6,
                        *(double *)a7.m128i_i64,
                        a8,
                        0);
      *(_DWORD *)(a3 + 8) = v61;
      return v130;
    }
    return result;
  }
  v16 = *(_QWORD *)a3;
  v17 = *(unsigned __int16 *)(*(_QWORD *)a3 + 24LL);
  if ( v17 != 32 && v17 != 10 )
    goto LABEL_5;
  if ( v15 == 20 )
  {
    v107 = *(_QWORD *)(v16 + 88);
    v108 = *(_DWORD *)(v107 + 32);
    if ( v108 <= 0x40 )
      v109 = *(_QWORD *)(v107 + 24) == 0;
    else
      v109 = v108 == (unsigned int)sub_16A57B0(v107 + 24);
    if ( !v109 )
    {
LABEL_6:
      v18 = 12;
      goto LABEL_7;
    }
LABEL_65:
    *a2 = v167.m128i_i64[0];
    *((_DWORD *)a2 + 2) = v167.m128i_i32[2];
    *(_QWORD *)a3 = v169;
    result = DWORD2(v169);
    *(_DWORD *)(a3 + 8) = DWORD2(v169);
    return result;
  }
  if ( v15 == 18 )
  {
    v50 = *(_QWORD *)(v16 + 88);
    v51 = *(_DWORD *)(v50 + 32);
    if ( v51 <= 0x40 )
      v52 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v51) == *(_QWORD *)(v50 + 24);
    else
      v52 = v51 == (unsigned int)sub_16A58F0(v50 + 24);
    if ( !v52 )
    {
LABEL_30:
      v18 = 10;
      goto LABEL_7;
    }
    goto LABEL_65;
  }
LABEL_5:
  switch ( v15 )
  {
    case 0xAu:
    case 0x12u:
      goto LABEL_30;
    case 0xBu:
    case 0x13u:
      v18 = 11;
      break;
    case 0xCu:
    case 0x14u:
      goto LABEL_6;
    case 0xDu:
    case 0xEu:
    case 0xFu:
    case 0x10u:
    case 0x11u:
    case 0x15u:
      v18 = 13;
      break;
  }
LABEL_7:
  v19 = *(_QWORD *)(a1 + 8);
  v170[0] = 0;
  v170[1] = 0x100000001LL;
  v170[2] = v19;
  v20 = *(_QWORD *)a1;
  v21 = *(_QWORD *)(v166.m128i_i64[0] + 40) + 16LL * v166.m128i_u32[2];
  v22 = *(_BYTE *)v21;
  v23 = *(_QWORD *)(v21 + 8);
  if ( *(_BYTE *)v21 )
  {
    if ( *(_QWORD *)(v20 + 8LL * v22 + 120) )
    {
      v102 = *(unsigned __int8 *)(*(_QWORD *)(v168 + 40) + 16LL * DWORD2(v168));
      if ( (_BYTE)v102 )
      {
        if ( *(_QWORD *)(v20 + 8 * v102 + 120) )
        {
          v156 = *(_QWORD *)a1;
          v103 = sub_21278D0(v20, v19, v22, v23);
          v154 = sub_20ACAE0(
                   v156,
                   v103,
                   v104,
                   v166.m128i_u64[0],
                   v166.m128i_i64[1],
                   v18,
                   a6,
                   a7,
                   a8,
                   v168,
                   0,
                   (__int64)v170,
                   a5);
          v151 = v105;
          v150 = v105;
          if ( v154 )
            goto LABEL_9;
          v19 = *(_QWORD *)(a1 + 8);
          v20 = *(_QWORD *)a1;
          v106 = *(_QWORD *)(v166.m128i_i64[0] + 40) + 16LL * v166.m128i_u32[2];
          v22 = *(_BYTE *)v106;
          v23 = *(_QWORD *)(v106 + 8);
        }
      }
    }
  }
  v153 = (__int64 *)v19;
  v24 = sub_21278D0(v20, v19, v22, v23);
  v154 = sub_1F81070(
           v153,
           a5,
           v24,
           v25,
           v166.m128i_u64[0],
           (__int16 *)v166.m128i_i64[1],
           (__m128)a6,
           *(double *)a7.m128i_i64,
           a8,
           v168,
           v18);
  v151 = v28;
  v150 = v28 | v150 & 0xFFFFFFFF00000000LL;
LABEL_9:
  v29 = v167.m128i_i64[0];
  v30 = (unsigned __int8 *)(*(_QWORD *)(v167.m128i_i64[0] + 40) + 16LL * v167.m128i_u32[2]);
  v31 = *v30;
  if ( !(_BYTE)v31 )
    goto LABEL_14;
  v32 = *(_QWORD *)a1;
  v29 = (unsigned __int8)v31;
  if ( !*(_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned __int8)v31 + 120) )
    goto LABEL_14;
  v29 = *(unsigned __int8 *)(*(_QWORD *)(v169 + 40) + 16LL * DWORD2(v169));
  if ( !(_BYTE)v29 )
    goto LABEL_14;
  if ( !*(_QWORD *)(v32 + 8 * v29 + 120) )
    goto LABEL_14;
  v157 = *(_QWORD *)a1;
  v33 = *a4;
  v34 = sub_21278D0(v32, *(_QWORD *)(a1 + 8), (unsigned __int8)v31, *((_QWORD *)v30 + 1));
  v36 = sub_20ACAE0(v157, v34, v35, v167.m128i_u64[0], v167.m128i_i64[1], v33, a6, a7, a8, v169, 0, (__int64)v170, a5);
  v31 = 0xFFFFFFFF00000000LL;
  v37 = v36;
  v152 = v29;
  v29 = (unsigned int)v29;
  v160 = (unsigned int)v29;
  if ( !v36 )
  {
LABEL_14:
    v38 = *(__int64 **)(a1 + 8);
    v142 = sub_1D28D50(v38, *a4, v29, v31, v26, v27);
    v148 = v39;
    v40 = sub_21278D0(
            *(_QWORD *)a1,
            *(_QWORD *)(a1 + 8),
            *(unsigned __int8 *)(*(_QWORD *)(v167.m128i_i64[0] + 40) + 16LL * v167.m128i_u32[2]),
            *(_QWORD *)(*(_QWORD *)(v167.m128i_i64[0] + 40) + 16LL * v167.m128i_u32[2] + 8));
    v37 = sub_1D3A900(
            v38,
            0x89u,
            a5,
            v40,
            v41,
            0,
            (__m128)a6,
            *(double *)a7.m128i_i64,
            a8,
            v167.m128i_u64[0],
            (__int16 *)v167.m128i_i64[1],
            v169,
            v142,
            v148);
    v152 = v42;
    v160 = v42 | v160 & 0xFFFFFFFF00000000LL;
  }
  v43 = *((unsigned __int16 *)v154 + 12);
  if ( v43 == 32 || (v44 = 0, v43 == 10) )
    v44 = v154;
  v45 = *((unsigned __int16 *)v37 + 12);
  if ( v45 == 32 || v45 == 10 )
  {
    if ( ((*a4 - 11) & 0xFFFFFFFD) != 0 && ((*a4 - 19) & 0xFFFFFFFD) != 0 )
    {
      v62 = v37[11];
      v144 = *(_DWORD *)(v62 + 32);
      if ( v144 > 0x40 )
      {
        v139 = v37;
        v141 = v37[11];
        v123 = sub_16A57B0(v62 + 24);
        v37 = v139;
        if ( v144 - v123 > 0x40 )
          goto LABEL_21;
        v63 = **(_QWORD **)(v141 + 24);
      }
      else
      {
        v63 = *(_QWORD *)(v62 + 24);
      }
      if ( v63 == 1 )
        goto LABEL_24;
LABEL_21:
      if ( v44 )
      {
        v46 = v44[11];
        v47 = *(_DWORD *)(v46 + 32);
        if ( v47 <= 0x40 )
        {
          if ( !*(_QWORD *)(v46 + 24) )
            goto LABEL_24;
        }
        else
        {
          v143 = v37;
          v48 = sub_16A57B0(v46 + 24);
          v37 = v143;
          if ( v47 == v48 )
          {
LABEL_24:
            *a2 = (unsigned __int64)v37;
            *((_DWORD *)a2 + 2) = v152;
            *(_QWORD *)a3 = 0;
            *(_DWORD *)(a3 + 8) = 0;
            return v152;
          }
        }
      }
      goto LABEL_44;
    }
    v64 = v37[11];
    v65 = *(_DWORD *)(v64 + 32);
    if ( v65 <= 0x40 )
    {
      v67 = *(_QWORD *)(v64 + 24) == 0;
    }
    else
    {
      v145 = v37;
      v66 = sub_16A57B0(v64 + 24);
      v37 = v145;
      v67 = v65 == v66;
    }
    if ( v67 )
      goto LABEL_24;
  }
  else if ( ((*a4 - 11) & 0xFFFFFFFD) != 0 && ((*a4 - 19) & 0xFFFFFFFD) != 0 )
  {
    goto LABEL_21;
  }
LABEL_44:
  if ( v167.m128i_i64[0] == (_QWORD)v169 && DWORD2(v169) == v167.m128i_i32[2] )
  {
    *a2 = (unsigned __int64)v154;
    *((_DWORD *)a2 + 2) = v151;
    *(_QWORD *)a3 = 0;
    *(_DWORD *)(a3 + 8) = 0;
    return v151;
  }
  v68 = (unsigned __int8 *)(*(_QWORD *)(v167.m128i_i64[0] + 40) + 16LL * v167.m128i_u32[2]);
  v69 = *(_QWORD *)a1;
  v137 = v37;
  v70 = *((_QWORD *)v68 + 1);
  v71 = *v68;
  v136 = (__int64 **)a1;
  v72 = *(_QWORD *)(a1 + 8);
  v73 = v132;
  v135 = a2;
  v138 = v71;
  v74 = v70;
  v75 = *(_QWORD *)(v72 + 48);
  v140 = v70;
  v76 = v171;
  v134 = a3;
  v133 = a5;
  v77 = v75;
  v78 = v69;
  while ( 1 )
  {
    LOBYTE(v73) = v71;
    v146 = (__int64)v76;
    sub_1F40D10((__int64)v76, v78, v77, v73, v74);
    if ( !v171[0] )
      break;
    v122 = v131;
    LOBYTE(v122) = v71;
    sub_1F40D10(v146, v78, v77, v122, v74);
    v71 = v171[8];
    v74 = v172;
    v76 = (_BYTE *)v146;
  }
  v79 = 1;
  v80 = (__int64)*v136;
  if ( (_BYTE)v71 == 1 || (_BYTE)v71 && (v79 = (unsigned __int8)v71, *(_QWORD *)(v80 + 8 * v71 + 120)) )
  {
    if ( (*(_BYTE *)(v80 + 259 * v79 + 2560) & 0xFB) == 0 )
    {
      v81 = *a4;
      if ( *a4 == 18 )
      {
        *a4 = 20;
      }
      else if ( v81 > 0x12 )
      {
        if ( v81 != 21 )
          goto LABEL_55;
        *a4 = 19;
      }
      else if ( v81 == 10 )
      {
        *a4 = 12;
      }
      else
      {
        if ( v81 != 13 )
        {
LABEL_55:
          v84 = (unsigned __int8 *)(*(_QWORD *)(v166.m128i_i64[0] + 40) + 16LL * v166.m128i_u32[2]);
          v85 = *v84;
          v158 = (__int64)v136[1];
          v155 = *((_QWORD *)v84 + 1);
          v86 = sub_21278D0(v80, v158, v85, v155);
          v88 = (const void ***)sub_1D252B0(v158, v85, v155, v86, v87);
          v91 = sub_1D37440(
                  v136[1],
                  73,
                  v133,
                  v88,
                  v89,
                  v90,
                  *(double *)a6.m128i_i64,
                  *(double *)a7.m128i_i64,
                  a8,
                  *(_OWORD *)&v166,
                  v168);
          v147 = v136[1];
          *(_QWORD *)&v95 = sub_1D28D50(v147, *a4, v92, v93, v94, (__int64)v147);
          v149 = v91;
          v96 = *v136;
          v162 = v95;
          *(_QWORD *)&v95 = v136[1];
          v97 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, _QWORD, __int64))(**v136 + 264);
          v159 = *(_QWORD *)(v95 + 48);
          v98 = sub_1E0A0C0(*(_QWORD *)(v95 + 32));
          v99 = v97(v96, v98, v159, v138, v140);
          *((_QWORD *)&v128 + 1) = 1;
          *(_QWORD *)&v128 = v149;
          *v135 = (unsigned __int64)sub_1D369D0(
                                      v147,
                                      138,
                                      v133,
                                      v99,
                                      v100,
                                      (__int64)v147,
                                      *(_OWORD *)&v167,
                                      v169,
                                      v128,
                                      v162);
          *((_DWORD *)v135 + 2) = v101;
          *(_QWORD *)v134 = 0;
          *(_DWORD *)(v134 + 8) = 0;
          return v101;
        }
        *a4 = 11;
      }
      v82 = _mm_loadu_si128(&v166);
      v166.m128i_i64[0] = v168;
      v166.m128i_i32[2] = DWORD2(v168);
      v83 = v82.m128i_i32[2];
      *(_QWORD *)&v168 = v82.m128i_i64[0];
      a6 = _mm_loadu_si128(&v167);
      DWORD2(v168) = v83;
      v80 = (__int64)*v136;
      v167.m128i_i64[0] = v169;
      *(_QWORD *)&v169 = a6.m128i_i64[0];
      v167.m128i_i32[2] = DWORD2(v169);
      DWORD2(v169) = a6.m128i_i32[2];
      goto LABEL_55;
    }
  }
  v116 = sub_21278D0(v80, (__int64)v136[1], v138, v140);
  v118 = sub_20ACAE0(
           v80,
           v116,
           v117,
           v167.m128i_u64[0],
           v167.m128i_i64[1],
           17,
           a6,
           a7,
           a8,
           v169,
           0,
           (__int64)v170,
           v133);
  v119 = (__int64)v137;
  *v135 = (unsigned __int64)v118;
  *((_DWORD *)v135 + 2) = v120;
  if ( !v118 )
  {
    v163 = v136[1];
    v124 = sub_21278D0((__int64)*v136, (__int64)v163, v138, v140);
    v126 = sub_1F81070(
             v163,
             v133,
             v124,
             v125,
             v167.m128i_u64[0],
             (__int16 *)v167.m128i_i64[1],
             (__m128)a6,
             *(double *)a7.m128i_i64,
             a8,
             v169,
             0x11u);
    v119 = (__int64)v137;
    *v135 = (unsigned __int64)v126;
    *((_DWORD *)v135 + 2) = v127;
  }
  *((_QWORD *)&v129 + 1) = v151 | v150 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v129 = v154;
  *v135 = (unsigned __int64)sub_1F810E0(
                              v136[1],
                              v133,
                              *(unsigned __int8 *)(v154[5] + 16LL * v151),
                              *(const void ***)(v154[5] + 16LL * v151 + 8),
                              *v135,
                              (__int16 *)v135[1],
                              (__m128)a6,
                              *(double *)a7.m128i_i64,
                              a8,
                              v129,
                              v119,
                              v152 | v160 & 0xFFFFFFFF00000000LL);
  *((_DWORD *)v135 + 2) = v121;
  *(_QWORD *)v134 = 0;
  *(_DWORD *)(v134 + 8) = 0;
  return v121;
}
