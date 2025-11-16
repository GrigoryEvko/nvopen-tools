// Function: sub_1954F50
// Address: 0x1954f50
//
__int64 __fastcall sub_1954F50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        double a7,
        double a8,
        double a9)
{
  __m128i *v10; // r12
  unsigned __int8 v12; // r15
  __int64 v13; // r15
  _QWORD *v14; // rax
  int v15; // r8d
  int v16; // r9d
  __m128i *v17; // r13
  __int64 v18; // rax
  __m128i **v19; // rax
  int v20; // r13d
  __int64 v21; // rbx
  char v22; // r8
  __int64 v23; // rax
  bool v26; // al
  __int64 *v27; // rdi
  __int64 v28; // r15
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // rdx
  __int64 v32; // rax
  int v33; // r8d
  int v34; // r9d
  char v35; // dl
  bool v36; // al
  __int64 *v37; // rdi
  __int64 v38; // r15
  int v39; // r14d
  int v40; // r13d
  __int8 *v41; // rdx
  __int64 v42; // r9
  __int8 *v43; // rsi
  char v44; // cl
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 *v47; // rax
  char v48; // al
  __int32 v49; // edx
  __int64 v50; // rsi
  __int64 v51; // rax
  __int64 *v52; // rdi
  __int64 v53; // rax
  __m128i v54; // rax
  int v55; // r8d
  int v56; // r9d
  __int64 v57; // rcx
  __int64 v58; // rsi
  unsigned int v59; // eax
  __int64 ****v60; // rbx
  __int64 ****v61; // r13
  __int64 ***v62; // rsi
  __int64 v63; // rax
  bool v64; // zf
  __int64 v65; // rdx
  __int8 v66; // al
  __int64 v67; // rsi
  __int64 v68; // rcx
  unsigned int v69; // eax
  __int64 v70; // rsi
  unsigned __int8 v71; // dl
  unsigned __int8 v72; // al
  bool v73; // al
  __int64 *v74; // rdi
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rdi
  int v78; // eax
  __int64 v79; // rdx
  __int64 v80; // rcx
  int v81; // r9d
  __int64 v82; // rax
  __int64 v83; // r15
  __int64 v84; // rdx
  __int64 v85; // r14
  __int64 i; // rdi
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rcx
  int v90; // r8d
  int v91; // r9d
  __int64 v92; // r15
  __int64 v93; // rax
  char v94; // al
  __int64 v95; // rdx
  int v96; // r8d
  int v97; // r9d
  __int64 *v98; // rdi
  _QWORD *v99; // r13
  bool v100; // al
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 *v103; // r15
  __int64 v104; // rax
  bool v105; // al
  unsigned int v106; // eax
  __int64 **v107; // rbx
  __int64 **v108; // r12
  __int64 *v109; // rdi
  __int64 v110; // rcx
  __int64 *v111; // rax
  __int64 v112; // rax
  __int64 *v113; // rax
  __int64 v114; // rax
  int v115; // r8d
  int v116; // r9d
  __int64 v117; // r13
  __int64 *v118; // r14
  unsigned int v119; // eax
  __m128i **v120; // rdx
  __int64 *v121; // rdi
  __int64 v122; // r8
  __int64 *v123; // r14
  __int64 v124; // rax
  __int64 *v125; // rbx
  __int64 v126; // r15
  int v127; // r8d
  int v128; // r9d
  __int64 v129; // rax
  __m128i **v130; // rax
  __int64 v131; // r8
  __int64 *v132; // rax
  __int64 v133; // r15
  __int64 v134; // rax
  bool v135; // al
  __int64 *v136; // rdi
  __m128i *v137; // rax
  __int64 v138; // rdx
  __int64 v139; // rax
  __int64 v140; // rbx
  __int64 v141; // r14
  __int64 v142; // r11
  __int64 v143; // rax
  __int64 v144; // rcx
  int v145; // r8d
  int v146; // r9d
  __int64 v147; // rdx
  _QWORD *v148; // rax
  __int64 v149; // rax
  __int64 *v150; // rax
  __int64 v151; // rax
  bool v152; // al
  __int64 *v153; // rdi
  __int64 v154; // rax
  __int64 v155; // rcx
  __int64 v156; // rdx
  __int64 v157; // rax
  __int64 v158; // rdx
  __int64 v159; // rcx
  int v160; // r8d
  int v161; // r9d
  __int32 v162; // eax
  int v163; // eax
  char v164; // r13
  __int64 v165; // [rsp+8h] [rbp-258h]
  __int64 v166; // [rsp+10h] [rbp-250h]
  int v167; // [rsp+10h] [rbp-250h]
  __int64 v168; // [rsp+18h] [rbp-248h]
  __int64 v169; // [rsp+18h] [rbp-248h]
  __int64 v170; // [rsp+20h] [rbp-240h]
  __int64 v171; // [rsp+20h] [rbp-240h]
  __int64 v172; // [rsp+28h] [rbp-238h]
  __int64 v173; // [rsp+28h] [rbp-238h]
  int v174; // [rsp+28h] [rbp-238h]
  __int64 v175; // [rsp+28h] [rbp-238h]
  __int64 v176; // [rsp+30h] [rbp-230h]
  __int64 v177; // [rsp+30h] [rbp-230h]
  __int64 v178; // [rsp+38h] [rbp-228h]
  unsigned int v179; // [rsp+38h] [rbp-228h]
  __int64 v180; // [rsp+38h] [rbp-228h]
  __int64 v181; // [rsp+40h] [rbp-220h]
  int v182; // [rsp+40h] [rbp-220h]
  __int64 v184; // [rsp+48h] [rbp-218h]
  __int64 v185; // [rsp+48h] [rbp-218h]
  __int64 v186; // [rsp+48h] [rbp-218h]
  __int64 v187; // [rsp+48h] [rbp-218h]
  __int64 v188; // [rsp+48h] [rbp-218h]
  __int64 v189; // [rsp+50h] [rbp-210h]
  __int64 v190; // [rsp+50h] [rbp-210h]
  int v191; // [rsp+50h] [rbp-210h]
  __int64 v192; // [rsp+50h] [rbp-210h]
  __int64 v194; // [rsp+58h] [rbp-208h]
  __int64 v195; // [rsp+58h] [rbp-208h]
  __int64 v196; // [rsp+58h] [rbp-208h]
  __int64 v197; // [rsp+68h] [rbp-1F8h] BYREF
  __int64 v198; // [rsp+70h] [rbp-1F0h]
  __int64 v199[3]; // [rsp+78h] [rbp-1E8h] BYREF
  __int64 v200; // [rsp+90h] [rbp-1D0h] BYREF
  unsigned int v201; // [rsp+98h] [rbp-1C8h]
  __int64 v202; // [rsp+C0h] [rbp-1A0h] BYREF
  _BYTE *v203; // [rsp+C8h] [rbp-198h]
  _BYTE *v204; // [rsp+D0h] [rbp-190h] BYREF
  __int64 v205; // [rsp+D8h] [rbp-188h]
  int v206; // [rsp+E0h] [rbp-180h]
  _BYTE v207[40]; // [rsp+E8h] [rbp-178h] BYREF
  __m128i v208; // [rsp+110h] [rbp-150h] BYREF
  __int64 v209[16]; // [rsp+120h] [rbp-140h] BYREF
  __m128i v210; // [rsp+1A0h] [rbp-C0h] BYREF
  _BYTE *v211; // [rsp+1B0h] [rbp-B0h] BYREF
  __int64 v212; // [rsp+1B8h] [rbp-A8h]
  __int64 v213; // [rsp+1C0h] [rbp-A0h]

  v10 = (__m128i *)a2;
  v12 = *(_BYTE *)(a2 + 16);
  v198 = a1 + 224;
  v199[0] = a2;
  v199[1] = a3;
  if ( v12 == 9 )
    goto LABEL_2;
  if ( a5 == 1 )
  {
    v51 = sub_1649C60(a2);
    if ( *(_BYTE *)(v51 + 16) != 4 )
    {
      v12 = *(_BYTE *)(a2 + 16);
      goto LABEL_18;
    }
    v10 = (__m128i *)v51;
LABEL_2:
    v13 = *(_QWORD *)(a3 + 8);
    if ( v13 )
    {
      while ( 1 )
      {
        v14 = sub_1648700(v13);
        if ( (unsigned __int8)(*((_BYTE *)v14 + 16) - 25) <= 9u )
          break;
        v13 = *(_QWORD *)(v13 + 8);
        if ( !v13 )
          goto LABEL_15;
      }
LABEL_6:
      v17 = (__m128i *)v14[5];
      v18 = *(unsigned int *)(a4 + 8);
      if ( (unsigned int)v18 >= *(_DWORD *)(a4 + 12) )
      {
        sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v15, v16);
        v18 = *(unsigned int *)(a4 + 8);
      }
      v19 = (__m128i **)(*(_QWORD *)a4 + 16 * v18);
      v19[1] = v17;
      *v19 = v10;
      v20 = *(_DWORD *)(a4 + 8) + 1;
      *(_DWORD *)(a4 + 8) = v20;
      while ( 1 )
      {
        v13 = *(_QWORD *)(v13 + 8);
        if ( !v13 )
          break;
        v14 = sub_1648700(v13);
        if ( (unsigned __int8)(*((_BYTE *)v14 + 16) - 25) <= 9u )
          goto LABEL_6;
      }
    }
    else
    {
LABEL_15:
      v20 = *(_DWORD *)(a4 + 8);
    }
    LOBYTE(v10) = v20 != 0;
    goto LABEL_11;
  }
  if ( v12 == 13 )
    goto LABEL_2;
LABEL_18:
  if ( v12 <= 0x17u || a3 != *(_QWORD *)(a2 + 40) )
  {
    v26 = sub_15CD740(*(_QWORD *)(a1 + 24));
    v27 = *(__int64 **)(a1 + 8);
    if ( v26 )
      sub_13EBC00(v27);
    else
      sub_13EBC50(v27);
    v28 = *(_QWORD *)(a3 + 8);
    if ( v28 )
    {
      while ( 1 )
      {
        v29 = sub_1648700(v28);
        if ( (unsigned __int8)(*((_BYTE *)v29 + 16) - 25) <= 9u )
          break;
        v28 = *(_QWORD *)(v28 + 8);
        if ( !v28 )
          goto LABEL_37;
      }
LABEL_29:
      v10 = (__m128i *)v29[5];
      v32 = sub_13F39C0(*(__int64 **)(a1 + 8), a2, (__int64)v10, a3, a6);
      if ( !v32 )
        goto LABEL_27;
      v35 = *(_BYTE *)(v32 + 16);
      if ( v35 == 9 )
        goto LABEL_25;
      if ( a5 != 1 )
      {
        if ( v35 != 13 )
          goto LABEL_27;
        v30 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v30 >= *(_DWORD *)(a4 + 12) )
          goto LABEL_34;
        goto LABEL_26;
      }
      v32 = sub_1649C60(v32);
      if ( *(_BYTE *)(v32 + 16) == 4 )
      {
LABEL_25:
        v30 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v30 >= *(_DWORD *)(a4 + 12) )
        {
LABEL_34:
          v178 = v32;
          sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v33, v34);
          v30 = *(unsigned int *)(a4 + 8);
          v32 = v178;
        }
LABEL_26:
        v31 = (__int64 *)(*(_QWORD *)a4 + 16 * v30);
        *v31 = v32;
        v31[1] = (__int64)v10;
        ++*(_DWORD *)(a4 + 8);
      }
LABEL_27:
      while ( 1 )
      {
        v28 = *(_QWORD *)(v28 + 8);
        if ( !v28 )
          break;
        v29 = sub_1648700(v28);
        if ( (unsigned __int8)(*((_BYTE *)v29 + 16) - 25) <= 9u )
          goto LABEL_29;
      }
    }
LABEL_37:
    LOBYTE(v10) = *(_DWORD *)(a4 + 8) != 0;
    goto LABEL_11;
  }
  if ( v12 == 77 )
  {
    v36 = sub_15CD740(*(_QWORD *)(a1 + 24));
    v37 = *(__int64 **)(a1 + 8);
    if ( v36 )
      sub_13EBC00(v37);
    else
      sub_13EBC50(v37);
    v38 = 0;
    if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
    {
      v189 = a1;
      v39 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v181 = a3;
      v40 = a5;
      while ( 1 )
      {
        v48 = v10[1].m128i_i8[7] & 0x40;
        if ( !v48 )
          break;
        v41 = (__int8 *)v10[-1].m128i_i64[1];
        v42 = *(_QWORD *)&v41[24 * v38];
        v43 = v41;
        if ( !v42 )
          goto LABEL_136;
        v44 = *(_BYTE *)(v42 + 16);
        if ( v44 != 9 )
          goto LABEL_59;
LABEL_47:
        v45 = *(_QWORD *)&v41[24 * v10[3].m128i_u32[2] + 8 + 8 * v38];
        v46 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v46 >= *(_DWORD *)(a4 + 12) )
        {
          v172 = v42;
          v185 = *(_QWORD *)&v41[24 * v10[3].m128i_u32[2] + 8 + 8 * v38];
          sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v45, v42);
          v46 = *(unsigned int *)(a4 + 8);
          v42 = v172;
          v45 = v185;
        }
        v47 = (__int64 *)(*(_QWORD *)a4 + 16 * v46);
        *v47 = v42;
        v47[1] = v45;
        ++*(_DWORD *)(a4 + 8);
LABEL_50:
        if ( v39 == (_DWORD)++v38 )
          goto LABEL_70;
      }
      v49 = v10[1].m128i_i32[1];
      v50 = v49 & 0xFFFFFFF;
      v42 = v10->m128i_i64[3 * (v38 - v50)];
      if ( !v42 )
      {
        v43 = &v10->m128i_i8[-24 * v50];
LABEL_136:
        v52 = *(__int64 **)(v189 + 8);
        goto LABEL_66;
      }
      v44 = *(_BYTE *)(v42 + 16);
      if ( v44 != 9 )
      {
LABEL_59:
        if ( v40 == 1 )
        {
          v184 = v42;
          v63 = sub_1649C60(v42);
          v42 = v184;
          v64 = *(_BYTE *)(v63 + 16) == 4;
          v65 = v63;
          v66 = v10[1].m128i_i8[7];
          if ( !v64 )
          {
            v48 = v66 & 0x40;
LABEL_64:
            v52 = *(__int64 **)(v189 + 8);
            if ( v48 )
              v43 = (__int8 *)v10[-1].m128i_i64[1];
            else
              v43 = &v10->m128i_i8[-24 * (v10[1].m128i_i32[1] & 0xFFFFFFF)];
LABEL_66:
            v53 = sub_13F39C0(v52, v42, *(_QWORD *)&v43[24 * v10[3].m128i_u32[2] + 8 + 8 * v38], v181, a6);
            v54.m128i_i64[0] = sub_1951DF0(v53, v40);
            if ( v54.m128i_i64[0] )
            {
              if ( (v10[1].m128i_i8[7] & 0x40) != 0 )
                v57 = v10[-1].m128i_i64[1];
              else
                v57 = (__int64)&v10->m128i_i64[-3 * (v10[1].m128i_i32[1] & 0xFFFFFFF)];
              v54.m128i_i64[1] = *(_QWORD *)(8 * v38 + v57 + 24LL * v10[3].m128i_u32[2] + 8);
              v210 = v54;
              sub_1953A90(a4, &v210, v54.m128i_i64[1], v57, v55, v56);
            }
            goto LABEL_50;
          }
          v48 = v66 & 0x40;
          v42 = v65;
        }
        else if ( v44 != 13 )
        {
          goto LABEL_64;
        }
        if ( v48 )
        {
          v41 = (__int8 *)v10[-1].m128i_i64[1];
          goto LABEL_47;
        }
        v49 = v10[1].m128i_i32[1];
      }
      v41 = &v10->m128i_i8[-24 * (v49 & 0xFFFFFFF)];
      goto LABEL_47;
    }
LABEL_70:
    LOBYTE(v10) = *(_DWORD *)(a4 + 8) != 0;
    goto LABEL_11;
  }
  if ( (unsigned int)v12 - 60 <= 0xC )
  {
    v58 = *(_QWORD *)(a2 - 24);
    if ( (unsigned __int8)(*(_BYTE *)(v58 + 16) - 75) <= 2u )
    {
      sub_1954CE0(a1, v58, a3);
      v59 = *(_DWORD *)(a4 + 8);
      if ( v59 )
      {
        v60 = *(__int64 *****)a4;
        v61 = &v60[2 * v59];
        do
        {
          v62 = *v60;
          v60 += 2;
          *(v60 - 2) = (__int64 ***)sub_15A46C0(
                                      (unsigned int)v10[1].m128i_u8[0] - 24,
                                      v62,
                                      (__int64 **)v10->m128i_i64[0],
                                      0);
        }
        while ( v61 != v60 );
LABEL_77:
        LODWORD(v10) = 1;
        goto LABEL_11;
      }
    }
    goto LABEL_82;
  }
  v176 = *(_QWORD *)a2;
  if ( (unsigned int)sub_1643030(*(_QWORD *)a2) != 1 )
  {
    if ( (unsigned int)v12 - 35 <= 0x11 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) == 13 )
      {
        v67 = *(_QWORD *)(a2 - 48);
        v210.m128i_i64[0] = (__int64)&v211;
        v210.m128i_i64[1] = 0x800000000LL;
        sub_1954CE0(a1, v67, a3);
      }
      goto LABEL_70;
    }
    goto LABEL_87;
  }
  if ( (unsigned __int8)(v12 - 50) <= 1u )
  {
    v208.m128i_i64[0] = (__int64)v209;
    v210.m128i_i64[0] = (__int64)&v211;
    v208.m128i_i64[1] = 0x800000000LL;
    v210.m128i_i64[1] = 0x800000000LL;
    v111 = (__int64 *)sub_13CF970(a2);
    sub_1954CE0(a1, *v111, a3);
    v112 = sub_13CF970(a2);
    sub_1954CE0(a1, *(_QWORD *)(v112 + 24), a3);
    if ( !v210.m128i_i32[2] )
    {
      LODWORD(v10) = 0;
LABEL_173:
      v121 = (__int64 *)v210.m128i_i64[0];
      goto LABEL_168;
    }
    if ( *(_BYTE *)(a2 + 16) == 51 )
    {
      v132 = (__int64 *)sub_16498A0(a2);
      v114 = sub_159C4F0(v132);
    }
    else
    {
      v113 = (__int64 *)sub_16498A0(a2);
      v114 = sub_159C540(v113);
    }
    v10 = (__m128i *)v114;
    v202 = 0;
    v203 = v207;
    v204 = v207;
    v205 = 4;
    v206 = 0;
    v117 = v208.m128i_i64[0] + 16LL * v208.m128i_u32[2];
    if ( v208.m128i_i64[0] == v117 )
    {
      v121 = (__int64 *)v210.m128i_i64[0];
      v131 = 16LL * v210.m128i_u32[2];
      v123 = (__int64 *)(v210.m128i_i64[0] + v131);
      if ( v210.m128i_i64[0] + v131 == v210.m128i_i64[0] )
      {
        LOBYTE(v10) = *(_DWORD *)(a4 + 8) != 0;
LABEL_168:
        if ( v121 != (__int64 *)&v211 )
          _libc_free((unsigned __int64)v121);
        v98 = (__int64 *)v208.m128i_i64[0];
        if ( (__int64 *)v208.m128i_i64[0] == v209 )
          goto LABEL_11;
        goto LABEL_171;
      }
    }
    else
    {
      v118 = (__int64 *)v208.m128i_i64[0];
      do
      {
        if ( (__m128i *)*v118 == v10 || *(_BYTE *)(*v118 + 16) == 9 )
        {
          v119 = *(_DWORD *)(a4 + 8);
          if ( v119 >= *(_DWORD *)(a4 + 12) )
          {
            sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v115, v116);
            v119 = *(_DWORD *)(a4 + 8);
          }
          v120 = (__m128i **)(*(_QWORD *)a4 + 16LL * v119);
          if ( v120 )
          {
            *v120 = v10;
            v120[1] = (__m128i *)v118[1];
            v119 = *(_DWORD *)(a4 + 8);
          }
          *(_DWORD *)(a4 + 8) = v119 + 1;
          sub_1953970((__int64)&v200, (__int64)&v202, v118[1]);
        }
        v118 += 2;
      }
      while ( (__int64 *)v117 != v118 );
      v121 = (__int64 *)v210.m128i_i64[0];
      v122 = 16LL * v210.m128i_u32[2];
      v123 = (__int64 *)(v210.m128i_i64[0] + v122);
      if ( v210.m128i_i64[0] == v210.m128i_i64[0] + v122 )
      {
LABEL_166:
        LOBYTE(v10) = *(_DWORD *)(a4 + 8) != 0;
        if ( v204 != v203 )
        {
          _libc_free((unsigned __int64)v204);
          v121 = (__int64 *)v210.m128i_i64[0];
          goto LABEL_168;
        }
        goto LABEL_173;
      }
    }
    v124 = a4;
    v125 = v121;
    v126 = v124;
    do
    {
      if ( ((__m128i *)*v125 == v10 || *(_BYTE *)(*v125 + 16) == 9) && !sub_183E920((__int64)&v202, v125[1]) )
      {
        v129 = *(unsigned int *)(v126 + 8);
        if ( (unsigned int)v129 >= *(_DWORD *)(v126 + 12) )
        {
          sub_16CD150(v126, (const void *)(v126 + 16), 0, 16, v127, v128);
          v129 = *(unsigned int *)(v126 + 8);
        }
        v130 = (__m128i **)(*(_QWORD *)v126 + 16 * v129);
        if ( v130 )
        {
          *v130 = v10;
          v130[1] = (__m128i *)v125[1];
        }
        ++*(_DWORD *)(v126 + 8);
      }
      v125 += 2;
    }
    while ( v123 != v125 );
    a4 = v126;
    goto LABEL_166;
  }
  if ( v12 == 52 )
  {
    v103 = (__int64 *)sub_13CF970(a2);
    v104 = v103[3];
    if ( *(_BYTE *)(v104 + 16) == 13 )
    {
      if ( *(_DWORD *)(v104 + 32) <= 0x40u )
      {
        v105 = *(_QWORD *)(v104 + 24) == 1;
      }
      else
      {
        v191 = *(_DWORD *)(v104 + 32);
        v105 = v191 - 1 == (unsigned int)sub_16A57B0(v104 + 24);
      }
      if ( v105 )
      {
        sub_1954CE0(a1, *v103, a3);
        v106 = *(_DWORD *)(a4 + 8);
        if ( v106 )
        {
          v107 = *(__int64 ***)a4;
          v108 = &v107[2 * v106];
          while ( v108 != v107 )
          {
            v109 = *v107;
            v107 += 2;
            *(v107 - 2) = (__int64 *)sub_15A2B00(v109, a7, a8, a9);
          }
          goto LABEL_77;
        }
LABEL_82:
        LODWORD(v10) = 0;
        goto LABEL_11;
      }
    }
LABEL_104:
    if ( sub_15CD740(*(_QWORD *)(a1 + 24)) )
      sub_13EBC00(*(__int64 **)(a1 + 8));
    else
      sub_13EBC50(*(__int64 **)(a1 + 8));
    v82 = sub_13F2790(*(__int64 **)(a1 + 8), (__int64)v10, a3, a6);
    v83 = sub_1951DF0(v82, a5);
    if ( v83 )
    {
      v10 = &v208;
      v208.m128i_i64[0] = sub_1952CC0(a3);
      v85 = v84;
      for ( i = v208.m128i_i64[0]; v85 != v208.m128i_i64[0]; i = v208.m128i_i64[0] )
      {
        v87 = sub_1648700(i)[5];
        v210.m128i_i64[0] = v83;
        v210.m128i_i64[1] = v87;
        sub_1953A90(a4, &v210, v88, v89, v90, v91);
        v208.m128i_i64[0] = *(_QWORD *)(v208.m128i_i64[0] + 8);
        sub_15CDD40(v208.m128i_i64);
      }
    }
    goto LABEL_70;
  }
LABEL_87:
  if ( (unsigned __int8)(v12 - 75) > 1u )
  {
    if ( v12 != 79 )
      goto LABEL_104;
    v92 = sub_1951DF0(*(_QWORD *)(a2 - 48), a5);
    v93 = sub_1951DF0(*(_QWORD *)(a2 - 24), a5);
    v210.m128i_i64[0] = (__int64)&v211;
    v210.m128i_i64[1] = 0x800000000LL;
    if ( !(v93 | v92) )
      goto LABEL_104;
    v180 = v93;
    v94 = sub_1954CE0(a1, *(_QWORD *)(a2 - 72), a3);
    v98 = (__int64 *)v210.m128i_i64[0];
    if ( !v94 )
    {
      if ( (_BYTE **)v210.m128i_i64[0] != &v211 )
        _libc_free(v210.m128i_u64[0]);
      goto LABEL_104;
    }
    v10 = &v208;
    if ( v210.m128i_i64[0] == v210.m128i_i64[0] + 16LL * v210.m128i_u32[2] )
      goto LABEL_207;
    v195 = v210.m128i_i64[0] + 16LL * v210.m128i_u32[2];
    v99 = (_QWORD *)v210.m128i_i64[0];
    while ( 1 )
    {
      v102 = *v99;
      if ( *(_BYTE *)(*v99 + 16LL) == 13 )
      {
        if ( *(_DWORD *)(v102 + 32) <= 0x40u )
        {
          v100 = *(_QWORD *)(v102 + 24) == 1;
        }
        else
        {
          v182 = *(_DWORD *)(v102 + 32);
          v100 = v182 - 1 == (unsigned int)sub_16A57B0(v102 + 24);
        }
        v64 = !v100;
        v101 = v92;
        if ( v64 )
          v101 = v180;
      }
      else
      {
        v101 = v92;
        if ( v92 )
        {
LABEL_137:
          v110 = v99[1];
          v208.m128i_i64[0] = v101;
          v208.m128i_i64[1] = v110;
          sub_1953A90(a4, &v208, v95, v110, v96, v97);
          goto LABEL_120;
        }
        v101 = v180;
      }
      if ( v101 )
        goto LABEL_137;
LABEL_120:
      v99 += 2;
      if ( (_QWORD *)v195 == v99 )
      {
        v98 = (__int64 *)v210.m128i_i64[0];
        goto LABEL_207;
      }
    }
  }
  v68 = *(_QWORD *)(a2 - 48);
  v69 = *(unsigned __int16 *)(a2 + 18);
  v70 = *(_QWORD *)(a2 - 24);
  v71 = *(_BYTE *)(v68 + 16);
  BYTE1(v69) &= ~0x80u;
  v190 = v10[-3].m128i_i64[0];
  v179 = v69;
  if ( v71 == 77 )
  {
    v133 = v10[-3].m128i_i64[0];
  }
  else
  {
    v72 = *(_BYTE *)(v70 + 16);
    if ( v72 != 77 )
      goto LABEL_90;
    v133 = v70;
  }
  if ( a3 == *(_QWORD *)(v133 + 40) )
  {
    v134 = sub_15F2050(v133);
    v177 = sub_1632FA0(v134);
    v135 = sub_15CD740(*(_QWORD *)(a1 + 24));
    v136 = *(__int64 **)(a1 + 8);
    if ( v135 )
      sub_13EBC00(v136);
    else
      sub_13EBC50(v136);
    v168 = a1;
    v187 = a4;
    v174 = *(_DWORD *)(v133 + 20) & 0xFFFFFFF;
    v137 = (__m128i *)a6;
    v196 = 0;
    if ( v137 )
      v10 = v137;
    v170 = (__int64)v10;
    v10 = &v210;
    while ( v174 != (_DWORD)v196 )
    {
      if ( (*(_BYTE *)(v133 + 23) & 0x40) != 0 )
        v138 = *(_QWORD *)(v133 - 8);
      else
        v138 = v133 - 24LL * (*(_DWORD *)(v133 + 20) & 0xFFFFFFF);
      v139 = 24LL * *(unsigned int *)(v133 + 56) + 8 * v196;
      v140 = *(_QWORD *)(v138 + v139 + 8);
      if ( v190 == v133 )
      {
        v141 = sub_1455F60(v190, v196);
        v142 = sub_16497E0(v70, a3, v140);
      }
      else
      {
        v141 = sub_16497E0(v190, a3, *(_QWORD *)(v138 + v139 + 8));
        v142 = sub_1455F60(v133, v196);
      }
      v166 = v142;
      v210 = (__m128i)(unsigned __int64)v177;
      v211 = 0;
      v212 = 0;
      v213 = 0;
      v143 = (__int64)sub_13E1240(v179, v141, v142, v210.m128i_i64);
      if ( v143
        || *(_BYTE *)(v166 + 16) <= 0x10u
        && (*(_BYTE *)(v141 + 16) <= 0x17u || a3 != *(_QWORD *)(v141 + 40))
        && (v167 = sub_13F3340(*(__int64 **)(v168 + 8), v179, v141, v166, v140, a3, v170), v167 != -1)
        && (v148 = (_QWORD *)sub_16498A0(v141), v149 = sub_1643320(v148), (v143 = sub_159C470(v149, v167, 0)) != 0) )
      {
        v147 = *(_BYTE *)(v143 + 16) & 0xFB;
        if ( (*(_BYTE *)(v143 + 16) & 0xFB) == 9 )
        {
          v210.m128i_i64[0] = v143;
          v210.m128i_i64[1] = v140;
          sub_1953A90(v187, &v210, v147, v144, v145, v146);
        }
      }
      ++v196;
    }
    LOBYTE(v10) = *(_DWORD *)(v187 + 8) != 0;
    goto LABEL_11;
  }
  v72 = *(_BYTE *)(v70 + 16);
LABEL_90:
  if ( v72 > 0x10u || *(_BYTE *)(v176 + 8) == 16 )
    goto LABEL_104;
  if ( v71 <= 0x17u || a3 != *(_QWORD *)(v190 + 40) )
  {
    v73 = sub_15CD740(*(_QWORD *)(a1 + 24));
    v74 = *(__int64 **)(a1 + 8);
    if ( v73 )
      sub_13EBC00(v74);
    else
      sub_13EBC50(v74);
    v75 = sub_1952CC0(a3);
    v186 = v76;
    v77 = v75;
    v208.m128i_i64[0] = v75;
    if ( v75 != v76 )
    {
      if ( a6 )
        v10 = (__m128i *)a6;
      v194 = (__int64)v10;
      v10 = &v208;
      do
      {
        v173 = sub_1648700(v77)[5];
        v78 = sub_13F3340(*(__int64 **)(a1 + 8), v179, v190, v70, v173, a3, v194);
        if ( v78 != -1 )
        {
          v210.m128i_i64[0] = sub_15A0680(v176, v78, 0);
          v210.m128i_i64[1] = v173;
          sub_1953A90(a4, &v210, v79, v80, v173, v81);
        }
        v208.m128i_i64[0] = *(_QWORD *)(v208.m128i_i64[0] + 8);
        sub_15CDD40(v208.m128i_i64);
        v77 = v208.m128i_i64[0];
      }
      while ( v186 != v208.m128i_i64[0] );
    }
    goto LABEL_70;
  }
  if ( v71 == 35 && v72 == 13 )
  {
    v151 = *(_QWORD *)(v190 - 48);
    v171 = v151;
    if ( v151 )
    {
      v188 = *(_QWORD *)(v190 - 24);
      if ( *(_BYTE *)(v188 + 16) == 13 && (*(_BYTE *)(v151 + 16) <= 0x17u || a3 != *(_QWORD *)(v151 + 40)) )
      {
        v152 = sub_15CD740(*(_QWORD *)(a1 + 24));
        v153 = *(__int64 **)(a1 + 8);
        if ( v152 )
          sub_13EBC00(v153);
        else
          sub_13EBC50(v153);
        v154 = sub_1952CC0(a3);
        v155 = v190;
        v197 = v154;
        v169 = v156;
        if ( a6 )
          v155 = a6;
        v165 = a3;
        v175 = v155;
        while ( 1 )
        {
          if ( v169 == v197 )
          {
            LOBYTE(v10) = *(_DWORD *)(a4 + 8) != 0;
            goto LABEL_11;
          }
          v192 = sub_1648700(v197)[5];
          sub_13F2550((__int64)&v202, *(__int64 **)(a1 + 8), v171, v192, v165, v175);
          v201 = *(_DWORD *)(v188 + 32);
          if ( v201 > 0x40 )
            sub_16A4FD0((__int64)&v200, (const void **)(v188 + 24));
          else
            v200 = *(_QWORD *)(v188 + 24);
          v10 = &v208;
          sub_1589870((__int64)&v208, &v200);
          sub_158E130((__int64)&v210, (__int64)&v202, (__int64)&v208);
          if ( (unsigned int)v203 > 0x40 && v202 )
            j_j___libc_free_0_0(v202);
          v202 = v210.m128i_i64[0];
          v162 = v210.m128i_i32[2];
          v210.m128i_i32[2] = 0;
          LODWORD(v203) = v162;
          if ( (unsigned int)v205 > 0x40 && v204 )
            j_j___libc_free_0_0(v204);
          v204 = v211;
          v163 = v212;
          LODWORD(v212) = 0;
          LODWORD(v205) = v163;
          sub_135E100((__int64 *)&v211);
          sub_135E100(v210.m128i_i64);
          sub_135E100(v209);
          sub_135E100(v208.m128i_i64);
          sub_135E100(&v200);
          sub_158B890((__int64)&v208, v179, v70 + 24);
          if ( (unsigned __int8)sub_158BB40((__int64)&v208, (__int64)&v202) )
            break;
          sub_1590E70((__int64)&v210, (__int64)&v208);
          v164 = sub_158BB40((__int64)&v210, (__int64)&v202);
          sub_135E100((__int64 *)&v211);
          sub_135E100(v210.m128i_i64);
          if ( v164 )
          {
            v157 = sub_15A0640(v176);
            goto LABEL_224;
          }
LABEL_225:
          sub_135E100(v209);
          sub_135E100(v208.m128i_i64);
          sub_135E100((__int64 *)&v204);
          sub_135E100(&v202);
          v197 = *(_QWORD *)(v197 + 8);
          sub_15CDD40(&v197);
        }
        v157 = sub_15A0600(v176);
LABEL_224:
        v210.m128i_i64[0] = v157;
        v210.m128i_i64[1] = v192;
        sub_1953A90(a4, &v210, v158, v159, v160, v161);
        goto LABEL_225;
      }
    }
  }
  v210.m128i_i64[0] = (__int64)&v211;
  v210.m128i_i64[1] = 0x800000000LL;
  v150 = (__int64 *)sub_13CF970((__int64)v10);
  sub_1954CE0(a1, *v150, a3);
  v98 = (__int64 *)&v211;
  v10 = (__m128i *)&v211;
LABEL_207:
  LOBYTE(v10) = *(_DWORD *)(a4 + 8) != 0;
  if ( v98 != (__int64 *)&v211 )
LABEL_171:
    _libc_free((unsigned __int64)v98);
LABEL_11:
  v21 = v198;
  v22 = sub_1954760(v198, v199, (__int64 **)&v210);
  v23 = v210.m128i_i64[0];
  if ( v22 )
  {
    *(_QWORD *)v210.m128i_i64[0] = -16;
    *(_QWORD *)(v23 + 8) = -16;
    --*(_DWORD *)(v21 + 16);
    ++*(_DWORD *)(v21 + 20);
  }
  return (unsigned int)v10;
}
