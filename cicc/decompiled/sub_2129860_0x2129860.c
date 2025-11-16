// Function: sub_2129860
// Address: 0x2129860
//
__int64 __fastcall sub_2129860(
        __int64 a1,
        __int64 a2,
        const void ***a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // rax
  unsigned int v15; // r13d
  __int64 v16; // rax
  char v17; // r8
  __int64 v18; // rax
  char v19; // di
  __int64 v20; // rax
  unsigned int v21; // eax
  char v22; // r8
  unsigned __int64 v23; // r10
  unsigned __int8 *v24; // rax
  const void **v25; // rcx
  unsigned int v26; // r11d
  int v27; // eax
  unsigned __int64 v28; // r15
  const void *v29; // rdx
  __int64 *v30; // r12
  __int128 v31; // rax
  int v32; // edx
  __int64 result; // rax
  unsigned __int64 v34; // r8
  int v35; // eax
  __int64 *v36; // r15
  __int128 v37; // rax
  __int64 *v38; // rax
  bool v39; // cc
  unsigned int v40; // r11d
  int v41; // r10d
  int v42; // edx
  __int64 *v43; // r12
  __int128 v44; // rax
  unsigned int v45; // edx
  int v46; // eax
  __int64 *v47; // r15
  unsigned __int64 v48; // rdx
  unsigned int v49; // eax
  __int128 v50; // rax
  __int128 v51; // rax
  unsigned int v52; // ecx
  __int128 v53; // rax
  __int64 *v54; // rax
  unsigned __int64 v55; // rdx
  __int64 *v56; // rax
  unsigned int v57; // r11d
  int v58; // edx
  __int64 *v59; // r14
  __int128 v60; // rax
  unsigned int v61; // edx
  _QWORD *v62; // rax
  __int64 v63; // rax
  int v64; // edx
  const void *v65; // rax
  __int64 *v66; // r15
  unsigned __int64 v67; // rdx
  unsigned int v68; // eax
  __int128 v69; // rax
  __int128 v70; // rax
  unsigned int v71; // ecx
  __int128 v72; // rax
  __int64 *v73; // rax
  unsigned __int64 v74; // rdx
  __int64 *v75; // rax
  unsigned int v76; // r11d
  int v77; // edx
  __int64 *v78; // r14
  __int128 v79; // rax
  unsigned int v80; // edx
  __int64 *v81; // r12
  __int128 v82; // rax
  unsigned int v83; // edx
  unsigned __int64 v84; // r15
  int v85; // edx
  __int64 *v86; // r14
  unsigned int v87; // r11d
  __int128 v88; // rax
  __int64 *v89; // rax
  unsigned int v90; // edx
  unsigned __int64 v91; // rdi
  __int64 *v92; // r15
  __int128 v93; // rax
  __int64 *v94; // rax
  int v95; // edx
  unsigned int v96; // edx
  __int128 v97; // rax
  __int64 *v98; // rax
  unsigned int v99; // r11d
  int v100; // edx
  char v101; // cl
  __int64 *v102; // r14
  unsigned __int64 v103; // rdx
  unsigned int v104; // eax
  __int128 v105; // rax
  __int128 v106; // rax
  __int64 *v107; // r15
  __int128 v108; // rax
  __int64 *v109; // rax
  unsigned __int64 v110; // rdx
  __int64 *v111; // rax
  unsigned int v112; // edx
  int v113; // edx
  __int64 v114; // rcx
  const void **v115; // r8
  int v116; // edx
  __int64 v117; // [rsp-10h] [rbp-210h]
  const void *v118; // [rsp+8h] [rbp-1F8h]
  unsigned int v119; // [rsp+8h] [rbp-1F8h]
  unsigned int v120; // [rsp+10h] [rbp-1F0h]
  unsigned int v121; // [rsp+10h] [rbp-1F0h]
  unsigned int v122; // [rsp+10h] [rbp-1F0h]
  unsigned int v123; // [rsp+10h] [rbp-1F0h]
  int v124; // [rsp+10h] [rbp-1F0h]
  unsigned int v125; // [rsp+10h] [rbp-1F0h]
  unsigned int v126; // [rsp+10h] [rbp-1F0h]
  int v127; // [rsp+18h] [rbp-1E8h]
  int v128; // [rsp+18h] [rbp-1E8h]
  unsigned int v129; // [rsp+18h] [rbp-1E8h]
  unsigned int v130; // [rsp+18h] [rbp-1E8h]
  unsigned int v131; // [rsp+18h] [rbp-1E8h]
  unsigned int v132; // [rsp+18h] [rbp-1E8h]
  __int64 *v133; // [rsp+18h] [rbp-1E8h]
  unsigned int v134; // [rsp+18h] [rbp-1E8h]
  __int64 *v135; // [rsp+18h] [rbp-1E8h]
  unsigned int v136; // [rsp+18h] [rbp-1E8h]
  unsigned int v137; // [rsp+18h] [rbp-1E8h]
  unsigned int v138; // [rsp+18h] [rbp-1E8h]
  unsigned int v139; // [rsp+18h] [rbp-1E8h]
  unsigned __int64 v140; // [rsp+18h] [rbp-1E8h]
  unsigned __int64 v141; // [rsp+18h] [rbp-1E8h]
  unsigned __int64 v142; // [rsp+18h] [rbp-1E8h]
  unsigned __int64 v143; // [rsp+18h] [rbp-1E8h]
  unsigned int v144; // [rsp+20h] [rbp-1E0h]
  unsigned __int64 v145; // [rsp+20h] [rbp-1E0h]
  unsigned int v146; // [rsp+20h] [rbp-1E0h]
  int v147; // [rsp+20h] [rbp-1E0h]
  int v148; // [rsp+20h] [rbp-1E0h]
  int v149; // [rsp+20h] [rbp-1E0h]
  __int64 v150; // [rsp+20h] [rbp-1E0h]
  __int128 v151; // [rsp+20h] [rbp-1E0h]
  unsigned int v152; // [rsp+20h] [rbp-1E0h]
  unsigned int v153; // [rsp+20h] [rbp-1E0h]
  __int64 v154; // [rsp+20h] [rbp-1E0h]
  __int128 v155; // [rsp+20h] [rbp-1E0h]
  unsigned int v156; // [rsp+20h] [rbp-1E0h]
  unsigned int v157; // [rsp+20h] [rbp-1E0h]
  unsigned int v158; // [rsp+20h] [rbp-1E0h]
  unsigned int v159; // [rsp+20h] [rbp-1E0h]
  unsigned int v160; // [rsp+20h] [rbp-1E0h]
  unsigned int v161; // [rsp+20h] [rbp-1E0h]
  unsigned int v162; // [rsp+20h] [rbp-1E0h]
  __int64 *v163; // [rsp+20h] [rbp-1E0h]
  unsigned int v164; // [rsp+20h] [rbp-1E0h]
  __int128 v165; // [rsp+20h] [rbp-1E0h]
  char v166; // [rsp+30h] [rbp-1D0h]
  const void **v167; // [rsp+30h] [rbp-1D0h]
  __int64 *v169; // [rsp+90h] [rbp-170h]
  __int64 v170; // [rsp+160h] [rbp-A0h] BYREF
  int v171; // [rsp+168h] [rbp-98h]
  __int64 v172; // [rsp+170h] [rbp-90h] BYREF
  unsigned __int64 v173; // [rsp+178h] [rbp-88h]
  __int64 v174; // [rsp+180h] [rbp-80h] BYREF
  unsigned __int64 v175; // [rsp+188h] [rbp-78h]
  unsigned int v176; // [rsp+190h] [rbp-70h] BYREF
  const void **v177; // [rsp+198h] [rbp-68h]
  unsigned __int64 v178; // [rsp+1A0h] [rbp-60h] BYREF
  unsigned int v179; // [rsp+1A8h] [rbp-58h]
  unsigned __int64 v180; // [rsp+1B0h] [rbp-50h] BYREF
  unsigned int v181; // [rsp+1B8h] [rbp-48h]
  unsigned __int64 v182; // [rsp+1C0h] [rbp-40h] BYREF
  __int64 v183; // [rsp+1C8h] [rbp-38h]

  v12 = *(_QWORD *)(a2 + 72);
  v170 = v12;
  if ( v12 )
    sub_1623A60((__int64)&v170, v12, 2);
  v13 = *(_DWORD *)(a2 + 64);
  LODWORD(v173) = 0;
  LODWORD(v175) = 0;
  v171 = v13;
  v14 = *(_QWORD *)(a2 + 32);
  v172 = 0;
  v174 = 0;
  sub_20174B0(a1, *(_QWORD *)v14, *(_QWORD *)(v14 + 8), &v172, &v174);
  v15 = *((_DWORD *)a3 + 2);
  if ( v15 > 0x40 )
  {
    if ( v15 != (unsigned int)sub_16A57B0((__int64)a3) )
      goto LABEL_5;
LABEL_18:
    *(_QWORD *)a4 = v172;
    *(_DWORD *)(a4 + 8) = v173;
    *(_QWORD *)a5 = v174;
    result = (unsigned int)v175;
    *(_DWORD *)(a5 + 8) = v175;
    goto LABEL_19;
  }
  if ( !*a3 )
    goto LABEL_18;
LABEL_5:
  v16 = *(_QWORD *)(v172 + 40) + 16LL * (unsigned int)v173;
  v17 = *(_BYTE *)v16;
  v177 = *(const void ***)(v16 + 8);
  v18 = *(_QWORD *)(a2 + 40);
  LOBYTE(v176) = v17;
  v19 = *(_BYTE *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  LOBYTE(v182) = v19;
  v183 = v20;
  if ( v19 )
  {
    v144 = sub_2127930(v19);
    if ( v22 )
      goto LABEL_7;
  }
  else
  {
    v166 = v17;
    v21 = sub_1F58D40((__int64)&v182);
    v22 = v166;
    v144 = v21;
    if ( v166 )
    {
LABEL_7:
      v23 = (unsigned int)sub_2127930(v22);
      goto LABEL_8;
    }
  }
  v23 = (unsigned int)sub_1F58D40((__int64)&v176);
LABEL_8:
  v24 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
  v25 = (const void **)*((_QWORD *)v24 + 1);
  v26 = *v24;
  v27 = *(unsigned __int16 *)(a2 + 24);
  v28 = v144;
  v167 = v25;
  if ( v27 == 122 )
  {
    if ( v15 > 0x40 )
    {
      v136 = v26;
      v158 = v23;
      if ( v15 - (unsigned int)sub_16A57B0((__int64)a3) > 0x40 )
        goto LABEL_54;
      v26 = v136;
      v62 = **a3;
      if ( (unsigned __int64)v62 > v28 )
        goto LABEL_54;
      v84 = v158;
      if ( (unsigned __int64)v62 <= v158 )
        goto LABEL_98;
    }
    else
    {
      v62 = *a3;
      if ( (unsigned __int64)*a3 > v144 )
        goto LABEL_54;
      v84 = (unsigned int)v23;
      if ( (unsigned __int64)v62 <= (unsigned int)v23 )
      {
LABEL_98:
        if ( v62 == (_QWORD *)v84 )
        {
          *(_QWORD *)a4 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v170, v176, v177, 0, a6, a7, a8, 0);
          *(_DWORD *)(a4 + 8) = v113;
          *(_QWORD *)a5 = v172;
          result = (unsigned int)v173;
          *(_DWORD *)(a5 + 8) = v173;
          goto LABEL_19;
        }
        v138 = v26;
        v163 = *(__int64 **)(a1 + 8);
        *(_QWORD *)&v97 = sub_1D38970((__int64)v163, (__int64)a3, (__int64)&v170, v26, v167, 0, a6, a7, a8, 0);
        v98 = sub_1D332F0(v163, 122, (__int64)&v170, v176, v177, 0, *(double *)a6.m128i_i64, a7, a8, v172, v173, v97);
        v99 = v138;
        *(_QWORD *)a4 = v98;
        *(_DWORD *)(a4 + 8) = v100;
        v102 = *(__int64 **)(a1 + 8);
        v179 = *((_DWORD *)a3 + 2);
        v101 = v179;
        if ( v179 > 0x40 )
        {
          sub_16A4FD0((__int64)&v178, (const void **)a3);
          v101 = v179;
          v99 = v138;
          if ( v179 > 0x40 )
          {
            sub_16A8F40((__int64 *)&v178);
            v99 = v138;
            goto LABEL_102;
          }
          v103 = v178;
        }
        else
        {
          v103 = (unsigned __int64)*a3;
        }
        v178 = ~v103 & (0xFFFFFFFFFFFFFFFFLL >> -v101);
LABEL_102:
        v164 = v99;
        sub_16A7400((__int64)&v178);
        v104 = v179;
        v179 = 0;
        v181 = v104;
        v180 = v178;
        sub_16A7490((__int64)&v180, v84);
        LODWORD(v183) = v181;
        v139 = v164;
        v182 = v180;
        v181 = 0;
        *(_QWORD *)&v105 = sub_1D38970((__int64)v102, (__int64)&v182, (__int64)&v170, v164, v167, 0, a6, a7, a8, 0);
        *(_QWORD *)&v106 = sub_1D332F0(
                             v102,
                             124,
                             (__int64)&v170,
                             v176,
                             v177,
                             0,
                             *(double *)a6.m128i_i64,
                             a7,
                             a8,
                             v172,
                             v173,
                             v105);
        v107 = *(__int64 **)(a1 + 8);
        v165 = v106;
        *(_QWORD *)&v108 = sub_1D38970((__int64)v107, (__int64)a3, (__int64)&v170, v139, v167, 0, a6, a7, a8, 0);
        v109 = sub_1D332F0(v107, 122, (__int64)&v170, v176, v177, 0, *(double *)a6.m128i_i64, a7, a8, v174, v175, v108);
        v111 = sub_1D332F0(
                 v102,
                 119,
                 (__int64)&v170,
                 v176,
                 v177,
                 0,
                 *(double *)a6.m128i_i64,
                 a7,
                 a8,
                 (__int64)v109,
                 v110,
                 v165);
        v39 = (unsigned int)v183 <= 0x40;
        *(_QWORD *)a5 = v111;
        result = v112;
        *(_DWORD *)(a5 + 8) = v112;
        if ( !v39 && v182 )
          result = j_j___libc_free_0_0(v182);
        if ( v181 > 0x40 && v180 )
          result = j_j___libc_free_0_0(v180);
        if ( v179 <= 0x40 )
          goto LABEL_19;
        v91 = v178;
        if ( !v178 )
          goto LABEL_19;
        goto LABEL_83;
      }
    }
    v159 = v26;
    *(_QWORD *)a4 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v170, v176, v177, 0, a6, a7, a8, 0);
    *(_DWORD *)(a4 + 8) = v85;
    v86 = *(__int64 **)(a1 + 8);
    v87 = v159;
    v181 = *((_DWORD *)a3 + 2);
    if ( v181 > 0x40 )
    {
      sub_16A4FD0((__int64)&v180, (const void **)a3);
      v87 = v159;
    }
    else
    {
      v180 = (unsigned __int64)*a3;
    }
    v160 = v87;
    sub_16A7800((__int64)&v180, v84);
    LODWORD(v183) = v181;
    v181 = 0;
    v182 = v180;
    *(_QWORD *)&v88 = sub_1D38970((__int64)v86, (__int64)&v182, (__int64)&v170, v160, v167, 0, a6, a7, a8, 0);
    v89 = sub_1D332F0(v86, 122, (__int64)&v170, v176, v177, 0, *(double *)a6.m128i_i64, a7, a8, v172, v173, v88);
    v39 = (unsigned int)v183 <= 0x40;
    *(_QWORD *)a5 = v89;
    result = v90;
    *(_DWORD *)(a5 + 8) = v90;
    if ( !v39 && v182 )
      result = j_j___libc_free_0_0(v182);
    if ( v181 <= 0x40 )
      goto LABEL_19;
    v91 = v180;
    if ( !v180 )
      goto LABEL_19;
LABEL_83:
    result = j_j___libc_free_0_0(v91);
    goto LABEL_19;
  }
  if ( v27 != 124 )
  {
    if ( v15 > 0x40 )
    {
      v131 = v26;
      v149 = v23;
      v46 = sub_16A57B0((__int64)a3);
      LODWORD(v23) = v149;
      v26 = v131;
      if ( v15 - v46 > 0x40 )
        goto LABEL_12;
      v29 = **a3;
      if ( v28 < (unsigned __int64)v29 )
        goto LABEL_12;
    }
    else
    {
      v29 = *a3;
      if ( v144 < (unsigned __int64)*a3 )
      {
LABEL_12:
        v30 = *(__int64 **)(a1 + 8);
        *(_QWORD *)&v31 = sub_1D38BB0(
                            (__int64)v30,
                            (unsigned int)(v23 - 1),
                            (__int64)&v170,
                            v26,
                            v167,
                            0,
                            a6,
                            a7,
                            a8,
                            0);
        v169 = sub_1D332F0(v30, 123, (__int64)&v170, v176, v177, 0, *(double *)a6.m128i_i64, a7, a8, v174, v175, v31);
        *(_QWORD *)a4 = v169;
        *(_DWORD *)(a4 + 8) = v32;
        *(_QWORD *)a5 = v169;
        result = *(unsigned int *)(a4 + 8);
        *(_DWORD *)(a5 + 8) = result;
        goto LABEL_13;
      }
    }
    v34 = (unsigned int)v23;
    if ( v15 <= 0x40 )
    {
      if ( (unsigned int)v23 < (unsigned __int64)*a3 )
        goto LABEL_25;
    }
    else
    {
      v120 = v26;
      v127 = v23;
      v145 = (unsigned int)v23;
      v118 = v29;
      v35 = sub_16A57B0((__int64)a3);
      v34 = v145;
      LODWORD(v23) = v127;
      v26 = v120;
      if ( v15 - v35 > 0x40 || (v29 = v118, v145 < (unsigned __int64)**a3) )
      {
LABEL_25:
        v181 = v15;
        v36 = *(__int64 **)(a1 + 8);
        if ( v15 > 0x40 )
        {
          v119 = v26;
          v124 = v23;
          v141 = v34;
          sub_16A4FD0((__int64)&v180, (const void **)a3);
          v26 = v119;
          LODWORD(v23) = v124;
          v34 = v141;
        }
        else
        {
          v180 = (unsigned __int64)*a3;
        }
        v128 = v23;
        v146 = v26;
        sub_16A7800((__int64)&v180, v34);
        LODWORD(v183) = v181;
        v181 = 0;
        v182 = v180;
        *(_QWORD *)&v37 = sub_1D38970((__int64)v36, (__int64)&v182, (__int64)&v170, v146, v167, 0, a6, a7, a8, 0);
        v38 = sub_1D332F0(v36, 123, (__int64)&v170, v176, v177, 0, *(double *)a6.m128i_i64, a7, a8, v174, v175, v37);
        v39 = (unsigned int)v183 <= 0x40;
        v40 = v146;
        v41 = v128;
        *(_QWORD *)a4 = v38;
        *(_DWORD *)(a4 + 8) = v42;
        if ( !v39 && v182 )
        {
          v129 = v146;
          v147 = v41;
          j_j___libc_free_0_0(v182);
          v40 = v129;
          v41 = v147;
        }
        if ( v181 > 0x40 && v180 )
        {
          v130 = v40;
          v148 = v41;
          j_j___libc_free_0_0(v180);
          v40 = v130;
          v41 = v148;
        }
        v43 = *(__int64 **)(a1 + 8);
        *(_QWORD *)&v44 = sub_1D38BB0(
                            (__int64)v43,
                            (unsigned int)(v41 - 1),
                            (__int64)&v170,
                            v40,
                            v167,
                            0,
                            a6,
                            a7,
                            a8,
                            0);
        *(_QWORD *)a5 = sub_1D332F0(
                          v43,
                          123,
                          (__int64)&v170,
                          v176,
                          v177,
                          0,
                          *(double *)a6.m128i_i64,
                          a7,
                          a8,
                          v174,
                          v175,
                          v44);
        result = v45;
        *(_DWORD *)(a5 + 8) = v45;
        goto LABEL_13;
      }
    }
    if ( (const void *)v34 == v29 )
    {
      *(_QWORD *)a4 = v174;
      *(_DWORD *)(a4 + 8) = v175;
      v81 = *(__int64 **)(a1 + 8);
      *(_QWORD *)&v82 = sub_1D38BB0((__int64)v81, (unsigned int)(v23 - 1), (__int64)&v170, v26, v167, 0, a6, a7, a8, 0);
      *(_QWORD *)a5 = sub_1D332F0(
                        v81,
                        123,
                        (__int64)&v170,
                        v176,
                        v177,
                        0,
                        *(double *)a6.m128i_i64,
                        a7,
                        a8,
                        v174,
                        v175,
                        v82);
      result = v83;
      *(_DWORD *)(a5 + 8) = v83;
LABEL_13:
      if ( v170 )
        return sub_161E7C0((__int64)&v170, v170);
      return result;
    }
    v179 = v15;
    v47 = *(__int64 **)(a1 + 8);
    if ( v15 > 0x40 )
    {
      v123 = v26;
      v140 = v34;
      sub_16A4FD0((__int64)&v178, (const void **)a3);
      LOBYTE(v15) = v179;
      v34 = v140;
      v26 = v123;
      if ( v179 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v178);
        v26 = v123;
        v34 = v140;
LABEL_42:
        v132 = v26;
        v150 = v34;
        sub_16A7400((__int64)&v178);
        v49 = v179;
        v179 = 0;
        v181 = v49;
        v180 = v178;
        sub_16A7490((__int64)&v180, v150);
        LODWORD(v183) = v181;
        v181 = 0;
        v182 = v180;
        *(_QWORD *)&v50 = sub_1D38970((__int64)v47, (__int64)&v182, (__int64)&v170, v132, v167, 0, a6, a7, a8, 0);
        *(_QWORD *)&v51 = sub_1D332F0(
                            v47,
                            122,
                            (__int64)&v170,
                            v176,
                            v177,
                            0,
                            *(double *)a6.m128i_i64,
                            a7,
                            a8,
                            v174,
                            v175,
                            v50);
        v52 = v132;
        v121 = v132;
        v133 = *(__int64 **)(a1 + 8);
        v151 = v51;
        *(_QWORD *)&v53 = sub_1D38970((__int64)v133, (__int64)a3, (__int64)&v170, v52, v167, 0, a6, a7, a8, 0);
        v54 = sub_1D332F0(v133, 124, (__int64)&v170, v176, v177, 0, *(double *)a6.m128i_i64, a7, a8, v172, v173, v53);
        v56 = sub_1D332F0(
                v47,
                119,
                (__int64)&v170,
                v176,
                v177,
                0,
                *(double *)a6.m128i_i64,
                a7,
                a8,
                (__int64)v54,
                v55,
                v151);
        v39 = (unsigned int)v183 <= 0x40;
        v57 = v121;
        *(_QWORD *)a4 = v56;
        *(_DWORD *)(a4 + 8) = v58;
        if ( !v39 && v182 )
        {
          j_j___libc_free_0_0(v182);
          v57 = v121;
        }
        if ( v181 > 0x40 && v180 )
        {
          v152 = v57;
          j_j___libc_free_0_0(v180);
          v57 = v152;
        }
        if ( v179 > 0x40 && v178 )
        {
          v153 = v57;
          j_j___libc_free_0_0(v178);
          v57 = v153;
        }
        v59 = *(__int64 **)(a1 + 8);
        *(_QWORD *)&v60 = sub_1D38970((__int64)v59, (__int64)a3, (__int64)&v170, v57, v167, 0, a6, a7, a8, 0);
        *(_QWORD *)a5 = sub_1D332F0(
                          v59,
                          123,
                          (__int64)&v170,
                          v176,
                          v177,
                          0,
                          *(double *)a6.m128i_i64,
                          a7,
                          a8,
                          v174,
                          v175,
                          v60);
        result = v61;
        *(_DWORD *)(a5 + 8) = v61;
        goto LABEL_13;
      }
      v48 = v178;
    }
    else
    {
      v48 = (unsigned __int64)*a3;
    }
    v178 = ~v48 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v15);
    goto LABEL_42;
  }
  if ( v15 > 0x40 )
  {
    v137 = v26;
    v161 = v23;
    if ( v15 - (unsigned int)sub_16A57B0((__int64)a3) <= 0x40 )
    {
      v23 = v161;
      v26 = v137;
      v65 = **a3;
      if ( v28 >= (unsigned __int64)v65 )
      {
        if ( (unsigned __int64)v65 <= v161 )
          goto LABEL_58;
LABEL_87:
        v181 = v15;
        v92 = *(__int64 **)(a1 + 8);
        if ( v15 > 0x40 )
        {
          v126 = v26;
          v143 = v23;
          sub_16A4FD0((__int64)&v180, (const void **)a3);
          v26 = v126;
          v23 = v143;
        }
        else
        {
          v180 = (unsigned __int64)*a3;
        }
        v162 = v26;
        sub_16A7800((__int64)&v180, v23);
        LODWORD(v183) = v181;
        v181 = 0;
        v182 = v180;
        *(_QWORD *)&v93 = sub_1D38970((__int64)v92, (__int64)&v182, (__int64)&v170, v162, v167, 0, a6, a7, a8, 0);
        v94 = sub_1D332F0(v92, 124, (__int64)&v170, v176, v177, 0, *(double *)a6.m128i_i64, a7, a8, v174, v175, v93);
        v39 = (unsigned int)v183 <= 0x40;
        *(_QWORD *)a4 = v94;
        *(_DWORD *)(a4 + 8) = v95;
        if ( !v39 && v182 )
          j_j___libc_free_0_0(v182);
        if ( v181 > 0x40 && v180 )
          j_j___libc_free_0_0(v180);
        *(_QWORD *)a5 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v170, v176, v177, 0, a6, a7, a8, 0);
        result = v96;
        *(_DWORD *)(a5 + 8) = v96;
        goto LABEL_19;
      }
    }
LABEL_54:
    v63 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v170, v176, v177, 0, a6, a7, a8, 0);
    *(_QWORD *)a5 = v63;
    *(_DWORD *)(a5 + 8) = v64;
    *(_QWORD *)a4 = v63;
    result = *(unsigned int *)(a5 + 8);
    *(_DWORD *)(a4 + 8) = result;
    goto LABEL_19;
  }
  v65 = *a3;
  if ( v144 < (unsigned __int64)*a3 )
    goto LABEL_54;
  if ( (unsigned __int64)v65 > v23 )
    goto LABEL_87;
LABEL_58:
  if ( (const void *)v23 != v65 )
  {
    v179 = v15;
    v66 = *(__int64 **)(a1 + 8);
    if ( v15 > 0x40 )
    {
      v125 = v26;
      v142 = v23;
      sub_16A4FD0((__int64)&v178, (const void **)a3);
      LOBYTE(v15) = v179;
      v23 = v142;
      v26 = v125;
      if ( v179 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v178);
        v26 = v125;
        v23 = v142;
LABEL_62:
        v134 = v26;
        v154 = v23;
        sub_16A7400((__int64)&v178);
        v68 = v179;
        v179 = 0;
        v181 = v68;
        v180 = v178;
        sub_16A7490((__int64)&v180, v154);
        LODWORD(v183) = v181;
        v181 = 0;
        v182 = v180;
        *(_QWORD *)&v69 = sub_1D38970((__int64)v66, (__int64)&v182, (__int64)&v170, v134, v167, 0, a6, a7, a8, 0);
        *(_QWORD *)&v70 = sub_1D332F0(
                            v66,
                            122,
                            (__int64)&v170,
                            v176,
                            v177,
                            0,
                            *(double *)a6.m128i_i64,
                            a7,
                            a8,
                            v174,
                            v175,
                            v69);
        v71 = v134;
        v122 = v134;
        v135 = *(__int64 **)(a1 + 8);
        v155 = v70;
        *(_QWORD *)&v72 = sub_1D38970((__int64)v135, (__int64)a3, (__int64)&v170, v71, v167, 0, a6, a7, a8, 0);
        v73 = sub_1D332F0(v135, 124, (__int64)&v170, v176, v177, 0, *(double *)a6.m128i_i64, a7, a8, v172, v173, v72);
        v75 = sub_1D332F0(
                v66,
                119,
                (__int64)&v170,
                v176,
                v177,
                0,
                *(double *)a6.m128i_i64,
                a7,
                a8,
                (__int64)v73,
                v74,
                v155);
        v39 = (unsigned int)v183 <= 0x40;
        v76 = v122;
        *(_QWORD *)a4 = v75;
        *(_DWORD *)(a4 + 8) = v77;
        if ( !v39 && v182 )
        {
          j_j___libc_free_0_0(v182);
          v76 = v122;
        }
        if ( v181 > 0x40 && v180 )
        {
          v156 = v76;
          j_j___libc_free_0_0(v180);
          v76 = v156;
        }
        if ( v179 > 0x40 && v178 )
        {
          v157 = v76;
          j_j___libc_free_0_0(v178);
          v76 = v157;
        }
        v78 = *(__int64 **)(a1 + 8);
        *(_QWORD *)&v79 = sub_1D38970((__int64)v78, (__int64)a3, (__int64)&v170, v76, v167, 0, a6, a7, a8, 0);
        *(_QWORD *)a5 = sub_1D332F0(
                          v78,
                          124,
                          (__int64)&v170,
                          v176,
                          v177,
                          0,
                          *(double *)a6.m128i_i64,
                          a7,
                          a8,
                          v174,
                          v175,
                          v79);
        result = v80;
        *(_DWORD *)(a5 + 8) = v80;
        goto LABEL_19;
      }
      v67 = v178;
    }
    else
    {
      v67 = (unsigned __int64)*a3;
    }
    v178 = ~v67 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v15);
    goto LABEL_62;
  }
  v114 = v176;
  v115 = v177;
  *(_QWORD *)a4 = v174;
  *(_DWORD *)(a4 + 8) = v175;
  *(_QWORD *)a5 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v170, v114, v115, 0, a6, a7, a8, 0);
  *(_DWORD *)(a5 + 8) = v116;
  result = v117;
LABEL_19:
  if ( v170 )
    return sub_161E7C0((__int64)&v170, v170);
  return result;
}
