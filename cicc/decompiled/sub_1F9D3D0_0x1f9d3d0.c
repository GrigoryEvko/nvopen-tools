// Function: sub_1F9D3D0
// Address: 0x1f9d3d0
//
__int64 __fastcall sub_1F9D3D0(__int64 **a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // rbx
  unsigned int v11; // r14d
  __int64 v12; // r13
  __int64 v13; // r11
  __int64 v15; // rax
  unsigned int v16; // eax
  int v17; // ecx
  int v18; // r9d
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rax
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // rdi
  const void **v25; // r8
  bool v26; // al
  __int64 v27; // rdx
  __int64 v28; // rcx
  const void **v29; // r8
  __int64 v30; // r9
  __int64 **v31; // r10
  __int64 *v32; // r12
  const __m128i *v33; // rbx
  char *v34; // r14
  __int64 v35; // r12
  unsigned int v36; // r13d
  __int128 v37; // xmm2
  __int64 v38; // rax
  __int128 v39; // xmm3
  char v40; // bl
  const void **v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdi
  unsigned int v44; // edx
  bool v45; // al
  __int64 v46; // rax
  char v47; // dl
  const void **v48; // rax
  int v49; // eax
  const void **v50; // rdx
  __int64 *v51; // rdi
  __int64 v52; // rsi
  char v53; // al
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  unsigned int v58; // r14d
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  unsigned int v63; // eax
  bool v64; // al
  __int64 *v65; // rdi
  __int64 v66; // rax
  __int16 *v67; // rdx
  __int128 v68; // rax
  __int64 *v69; // r15
  __int64 v70; // rax
  unsigned __int8 v71; // al
  __int64 v72; // rdx
  const void **v73; // r15
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 *v78; // rax
  unsigned int v79; // edx
  unsigned int v80; // r8d
  unsigned __int64 v81; // rdx
  unsigned __int8 *v82; // rax
  const void **v83; // rcx
  unsigned int v84; // r15d
  __int16 *v85; // rax
  __int64 v86; // r8
  __int16 *v87; // rdi
  __int64 v88; // rax
  char v89; // si
  __int16 *v90; // rcx
  const void **v91; // rax
  __int64 v92; // r9
  unsigned int v93; // esi
  char v94; // al
  __int64 *v95; // rdi
  __int64 (*v96)(); // rax
  unsigned int v97; // eax
  __int64 v98; // r14
  bool v99; // al
  const void *v100; // r8
  unsigned int v101; // edx
  char v102; // al
  unsigned int v103; // r13d
  const void *v104; // r12
  bool v105; // bl
  char v106; // al
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rbx
  __int64 v110; // r13
  const void *v111; // rsi
  __int64 *v112; // rax
  char v113; // al
  const void **v114; // rdx
  __int64 v115; // r8
  __int64 v116; // r12
  __int64 v117; // r13
  int v118; // ecx
  int v119; // eax
  int v120; // eax
  unsigned int v121; // r9d
  __int64 *v122; // rsi
  __int64 v123; // r10
  int v124; // edx
  bool v125; // al
  __int64 v126; // rsi
  __int64 v127; // rsi
  int v128; // eax
  __int128 v129; // rax
  unsigned __int64 v130; // r15
  __int64 *v131; // r12
  __int64 v132; // rdx
  __int64 v133; // r13
  __int64 *v134; // r14
  unsigned __int64 v135; // rdx
  unsigned __int64 v136; // r15
  __int64 v137; // rsi
  __int64 v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rax
  unsigned int v141; // ebx
  int v142; // eax
  bool v143; // al
  __int64 v144; // rax
  __int128 *v145; // rbx
  __int64 v146; // rax
  unsigned int v147; // r14d
  __int64 v148; // rax
  __int64 v149; // rax
  unsigned __int64 v150; // rdx
  __int128 v151; // [rsp-20h] [rbp-140h]
  __int128 v152; // [rsp-10h] [rbp-130h]
  __int128 v153; // [rsp-10h] [rbp-130h]
  __int64 v154; // [rsp+10h] [rbp-110h]
  __int64 v155; // [rsp+18h] [rbp-108h]
  const void *v156; // [rsp+20h] [rbp-100h]
  __int64 v157; // [rsp+28h] [rbp-F8h]
  unsigned int v158; // [rsp+28h] [rbp-F8h]
  unsigned int v159; // [rsp+28h] [rbp-F8h]
  __int64 v160; // [rsp+30h] [rbp-F0h]
  __int64 (__fastcall *v161)(__int64 *, __int64, __int64, __int64, __int64); // [rsp+38h] [rbp-E8h]
  unsigned int v162; // [rsp+38h] [rbp-E8h]
  __int64 v163; // [rsp+38h] [rbp-E8h]
  __int64 (__fastcall *v164)(__int64 *, __int64, __int64, __int64, const void **); // [rsp+40h] [rbp-E0h]
  bool v165; // [rsp+40h] [rbp-E0h]
  unsigned int v166; // [rsp+50h] [rbp-D0h]
  __int64 v167; // [rsp+50h] [rbp-D0h]
  __int64 v168; // [rsp+50h] [rbp-D0h]
  __int64 v169; // [rsp+58h] [rbp-C8h]
  __int64 v170; // [rsp+58h] [rbp-C8h]
  __int64 v171; // [rsp+58h] [rbp-C8h]
  __int64 v172; // [rsp+58h] [rbp-C8h]
  const void **v173; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v174; // [rsp+58h] [rbp-C8h]
  char v175; // [rsp+58h] [rbp-C8h]
  __int64 v176; // [rsp+60h] [rbp-C0h]
  __int64 v177; // [rsp+60h] [rbp-C0h]
  __int64 *v178; // [rsp+60h] [rbp-C0h]
  unsigned __int128 v179; // [rsp+60h] [rbp-C0h]
  __int128 v180; // [rsp+70h] [rbp-B0h]
  __int128 v181; // [rsp+70h] [rbp-B0h]
  const void **v182; // [rsp+70h] [rbp-B0h]
  unsigned __int8 v183; // [rsp+83h] [rbp-9Dh]
  unsigned int v184; // [rsp+84h] [rbp-9Ch]
  char v185; // [rsp+84h] [rbp-9Ch]
  __int64 *v187; // [rsp+88h] [rbp-98h]
  unsigned int v188; // [rsp+90h] [rbp-90h]
  __int64 v189; // [rsp+90h] [rbp-90h]
  __int64 v190; // [rsp+90h] [rbp-90h]
  __int64 v191; // [rsp+90h] [rbp-90h]
  __int64 v192; // [rsp+A0h] [rbp-80h]
  __int64 v193; // [rsp+A0h] [rbp-80h]
  unsigned __int64 v194; // [rsp+A0h] [rbp-80h]
  __int64 *v195; // [rsp+A0h] [rbp-80h]
  __int16 *v196; // [rsp+A8h] [rbp-78h]
  __int64 *v197; // [rsp+B0h] [rbp-70h] BYREF
  int v198; // [rsp+B8h] [rbp-68h]
  __int64 v199; // [rsp+C0h] [rbp-60h] BYREF
  const void **v200; // [rsp+C8h] [rbp-58h]
  const void *v201; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v202; // [rsp+D8h] [rbp-48h]
  __int64 v203; // [rsp+E0h] [rbp-40h] BYREF
  const void **v204; // [rsp+E8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(__int64 **)(a2 + 72);
  v8 = *(_QWORD *)v6;
  v9 = *(_QWORD *)(v6 + 40);
  v197 = v7;
  v10 = *(_QWORD *)(v6 + 48);
  v11 = *(_DWORD *)(v6 + 48);
  v12 = *(_QWORD *)(v6 + 80);
  v192 = v8;
  v188 = *(_DWORD *)(v6 + 88);
  if ( v7 )
    sub_1623A60((__int64)&v197, (__int64)v7, 2);
  v198 = *(_DWORD *)(a2 + 64);
  if ( v12 == v9 && v188 == v11 )
    goto LABEL_8;
  v13 = v11;
  if ( *(_WORD *)(v192 + 24) == 137 )
  {
    v15 = *(_QWORD *)(v192 + 32);
    a3 = _mm_loadu_si128((const __m128i *)v15);
    a4 = _mm_loadu_si128((const __m128i *)(v15 + 40));
    v176 = *(_QWORD *)v15;
    v169 = *(_QWORD *)(v15 + 40);
    v166 = *(_DWORD *)(v15 + 8);
    *((_QWORD *)&v180 + 1) = a4.m128i_i64[1];
    v184 = *(_DWORD *)(*(_QWORD *)(v15 + 80) + 84LL);
    v16 = sub_1D16620(v169, v7);
    v19 = v169;
    v20 = v16;
    if ( (_BYTE)v16 )
    {
      if ( v184 - 18 > 1 )
      {
        sub_1D16340(v169, (__int64)v7);
        v19 = v169;
        goto LABEL_16;
      }
      if ( v176 != v9 || v166 != v11 )
        goto LABEL_35;
      if ( *(_WORD *)(v12 + 24) != 53 )
        goto LABEL_16;
    }
    else
    {
      v94 = sub_1D16340(v169, (__int64)v7);
      v19 = v169;
      v20 = 0;
      if ( !v94 || v184 != 18 || v166 != v11 || v176 != v9 || *(_WORD *)(v12 + 24) != 53 )
        goto LABEL_35;
    }
    v21 = *(_QWORD *)(v12 + 32);
    if ( *(_QWORD *)(v21 + 40) == v9 && *(_DWORD *)(v21 + 48) == v11 )
    {
LABEL_21:
      v170 = v19;
      v22 = sub_1D16620(*(_QWORD *)v21, v7);
      v19 = v170;
      if ( v22 )
      {
        v23 = *(_QWORD *)(v176 + 40) + 16LL * v166;
        v24 = (__int64)a1[1];
        v25 = *(const void ***)(v23 + 8);
        LOBYTE(v203) = *(_BYTE *)v23;
        v204 = v25;
        v26 = sub_1F6C880(v24, 0x79u, v203);
        v32 = *v31;
        if ( v26 )
        {
          v9 = sub_1D309E0(
                 *v31,
                 121,
                 (__int64)&v197,
                 (unsigned int)v203,
                 v29,
                 0,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 *(double *)a5.m128i_i64,
                 __PAIR128__(v166 | a3.m128i_i64[1] & 0xFFFFFFFF00000000LL, v176));
        }
        else
        {
          v128 = sub_1D159C0((__int64)&v203, 121, v27, v28, (__int64)v29, v30);
          *(_QWORD *)&v129 = sub_1D38BB0(
                               (__int64)v32,
                               (unsigned int)(v128 - 1),
                               (__int64)&v197,
                               (unsigned int)v203,
                               v204,
                               0,
                               a3,
                               *(double *)a4.m128i_i64,
                               a5,
                               0);
          v130 = v166 | a3.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v131 = sub_1D332F0(
                   v32,
                   123,
                   (__int64)&v197,
                   (unsigned int)v203,
                   v204,
                   0,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5,
                   v176,
                   v130,
                   v129);
          v133 = v132;
          *((_QWORD *)&v152 + 1) = v132;
          *(_QWORD *)&v152 = v131;
          v134 = sub_1D332F0(
                   *a1,
                   52,
                   (__int64)&v197,
                   (unsigned int)v203,
                   v204,
                   0,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5,
                   v176,
                   v130,
                   v152);
          v136 = v135;
          sub_1F81BC0((__int64)a1, (__int64)v131);
          sub_1F81BC0((__int64)a1, (__int64)v134);
          *((_QWORD *)&v153 + 1) = v133;
          *(_QWORD *)&v153 = v131;
          v9 = (__int64)sub_1D332F0(
                          *a1,
                          120,
                          (__int64)&v197,
                          (unsigned int)v203,
                          v204,
                          0,
                          *(double *)a3.m128i_i64,
                          *(double *)a4.m128i_i64,
                          a5,
                          (__int64)v134,
                          v136,
                          v153);
        }
        goto LABEL_8;
      }
      goto LABEL_35;
    }
    if ( !(_BYTE)v20 )
    {
LABEL_35:
      *(_QWORD *)&v180 = v19;
      v42 = sub_1D1ADA0(v19, a4.m128i_u32[2], v20, v17, v19, v18);
      if ( v42
        && ((v43 = *(_QWORD *)(v42 + 88), v44 = *(_DWORD *)(v43 + 32), v44 <= 0x40)
          ? (v45 = *(_QWORD *)(v43 + 24) == 0)
          : (v45 = v44 == (unsigned int)sub_16A57B0(v43 + 24)),
            v45) )
      {
        v155 = v11;
        v171 = 16LL * v166;
        v183 = *(_BYTE *)(*(_QWORD *)(v176 + 40) + v171);
        v154 = 16LL * v11;
        v46 = *(_QWORD *)(v9 + 40) + v154;
        v47 = *(_BYTE *)v46;
        v48 = *(const void ***)(v46 + 8);
        LOBYTE(v203) = v47;
        v204 = v48;
        v49 = sub_1F7DF20(&v203);
        v200 = v50;
        LODWORD(v199) = v49;
        v157 = *(_QWORD *)(v171 + *(_QWORD *)(v176 + 40) + 8);
        v160 = *(unsigned __int8 *)(v171 + *(_QWORD *)(v176 + 40));
        v172 = (*a1)[6];
        v161 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(*a1[1] + 264);
        v51 = a1[1];
        v52 = sub_1E0A0C0((*a1)[4]);
        v53 = v161(v51, v52, v172, v160, v157);
        v202 = v54;
        LOBYTE(v201) = v53;
        v58 = sub_1D159C0((__int64)&v201, v52, v54, v55, v56, v57);
        v63 = sub_1D159C0((__int64)&v199, v52, v59, v60, v61, v62);
        v13 = v155;
        if ( *(_WORD *)(v176 + 24) == 185 )
        {
          v158 = v184 - 18;
          v162 = v63;
          v64 = sub_1D18C00(v176, 1, v166);
          v13 = v155;
          if ( v64 && v58 != 1 && v58 < v162 )
          {
            if ( (_BYTE)v199 )
            {
              if ( v183 )
              {
                v65 = a1[1];
                if ( (((int)*((unsigned __int16 *)v65 + 115 * (unsigned __int8)v199 + v183 + 16104) >> (4 * ((v184 - 18 > 3) + 2)))
                    & 0xB) == 0
                  && ((_BYTE)v199 == 1 || v65[(unsigned __int8)v199 + 15])
                  && (*((_BYTE *)v65 + 259 * (unsigned __int8)v199 + 2559) & 0xFB) == 0 )
                {
                  v66 = sub_1D309E0(
                          *a1,
                          (unsigned int)(v158 > 3) + 142,
                          (__int64)&v197,
                          (unsigned int)v199,
                          v200,
                          0,
                          *(double *)a3.m128i_i64,
                          *(double *)a4.m128i_i64,
                          *(double *)a5.m128i_i64,
                          __PAIR128__(v166 | a3.m128i_i64[1] & 0xFFFFFFFF00000000LL, v176));
                  v196 = v67;
                  v194 = v66;
                  *(_QWORD *)&v68 = sub_1D309E0(
                                      *a1,
                                      (unsigned int)(v158 > 3) + 142,
                                      (__int64)&v197,
                                      (unsigned int)v199,
                                      v200,
                                      0,
                                      *(double *)a3.m128i_i64,
                                      *(double *)a4.m128i_i64,
                                      *(double *)a5.m128i_i64,
                                      v180);
                  v69 = a1[1];
                  v181 = v68;
                  v177 = v199;
                  v173 = v200;
                  v164 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, const void **))(*v69 + 264);
                  v167 = (*a1)[6];
                  v70 = sub_1E0A0C0((*a1)[4]);
                  v71 = v164(v69, v70, v167, v177, v173);
                  v73 = (const void **)v72;
                  v174 = v71;
                  v178 = *a1;
                  v76 = sub_1D28D50(*a1, v184, v72, v71, v74, v75);
                  v78 = sub_1D3A900(
                          v178,
                          0x89u,
                          (__int64)&v197,
                          v174,
                          v73,
                          0,
                          (__m128)a3,
                          *(double *)a4.m128i_i64,
                          a5,
                          v194,
                          v196,
                          v181,
                          v76,
                          v77);
                  v80 = v79;
                  v81 = (unsigned __int64)v78;
                  v187 = *a1;
                  v82 = (unsigned __int8 *)(*(_QWORD *)(v9 + 40) + v154);
                  v83 = (const void **)*((_QWORD *)v82 + 1);
                  v84 = *v82;
                  v85 = (__int16 *)v80;
                  v86 = v12;
                  v182 = v83;
                  v87 = v85;
                  v88 = *(_QWORD *)(v81 + 40) + 16LL * (_QWORD)v85;
                  v89 = *(_BYTE *)v88;
                  v90 = v87;
                  v91 = *(const void ***)(v88 + 8);
                  LOBYTE(v203) = v89;
                  v204 = v91;
                  v92 = v188;
                  if ( v89 )
                  {
                    v93 = 135 - ((unsigned __int8)(v89 - 14) >= 0x60u);
                  }
                  else
                  {
                    v179 = __PAIR128__((unsigned __int64)v87, v81);
                    v125 = sub_1F58D20((__int64)&v203);
                    v90 = (__int16 *)*((_QWORD *)&v179 + 1);
                    v81 = v179;
                    v86 = v12;
                    v92 = v188;
                    v93 = 135 - !v125;
                  }
                  *((_QWORD *)&v151 + 1) = v155 | v10 & 0xFFFFFFFF00000000LL;
                  *(_QWORD *)&v151 = v9;
                  v9 = (__int64)sub_1D3A900(
                                  v187,
                                  v93,
                                  (__int64)&v197,
                                  v84,
                                  v182,
                                  0,
                                  (__m128)a3,
                                  *(double *)a4.m128i_i64,
                                  a5,
                                  v81,
                                  v90,
                                  v151,
                                  v86,
                                  v92);
                  goto LABEL_8;
                }
              }
            }
          }
        }
      }
      else
      {
        v13 = v11;
      }
      goto LABEL_6;
    }
LABEL_16:
    v17 = v188;
    LOBYTE(v20) = v166 == v188;
    if ( v176 == v12 && v166 == v188 && v184 - 20 <= 1 && *(_WORD *)(v9 + 24) == 53 )
    {
      v21 = *(_QWORD *)(v9 + 32);
      if ( *(_QWORD *)(v21 + 40) == v12 && *(_DWORD *)(v21 + 48) == v188 )
        goto LABEL_21;
    }
    goto LABEL_35;
  }
LABEL_6:
  if ( (unsigned __int8)sub_1F9C650(
                          a1,
                          a2,
                          v9,
                          v13 | v10 & 0xFFFFFFFF00000000LL,
                          v12,
                          v188,
                          (__m128)a3,
                          *(double *)a4.m128i_i64,
                          a5) )
  {
    v9 = a2;
    goto LABEL_8;
  }
  if ( !(unsigned __int8)sub_1D16340(v192, a2) )
  {
    if ( (unsigned __int8)sub_1D16620(v192, (__int64 *)a2) )
    {
      v9 = v12;
      goto LABEL_8;
    }
    if ( *(_WORD *)(v9 + 24) == 107 && *(_WORD *)(v12 + 24) == 107 && (unsigned __int8)sub_1D168E0(v192) )
    {
      v111 = *(const void **)(a2 + 72);
      v112 = *a1;
      v201 = v111;
      v195 = v112;
      if ( v111 )
        sub_1623A60((__int64)&v201, (__int64)v111, 2);
      v34 = *(char **)(a2 + 40);
      v33 = *(const __m128i **)(a2 + 32);
      LODWORD(v202) = *(_DWORD *)(a2 + 64);
      v113 = *v34;
      v114 = (const void **)*((_QWORD *)v34 + 1);
      v115 = v33->m128i_i64[0];
      v116 = v33[2].m128i_i64[1];
      v117 = v33[5].m128i_i64[0];
      LOBYTE(v203) = v113;
      v204 = v114;
      if ( v113 )
      {
        v118 = word_42FA680[(unsigned __int8)(v113 - 14)];
      }
      else
      {
        v190 = v115;
        v119 = sub_1F58D30((__int64)&v203);
        v115 = v190;
        v118 = v119;
      }
      if ( *(_DWORD *)(v116 + 56) == 2 && *(_DWORD *)(v117 + 56) == 2 )
      {
        v120 = v118 / 2;
        v121 = v118 / 2;
        if ( v118 <= 1 )
        {
          v123 = 0;
        }
        else
        {
          v122 = *(__int64 **)(v115 + 32);
          v123 = 0;
          v124 = 0;
          do
          {
            if ( *(_WORD *)(*v122 + 24) != 48 )
            {
              if ( v123 )
              {
                if ( *v122 != v123 )
                  goto LABEL_104;
              }
              else
              {
                v123 = *v122;
              }
            }
            ++v124;
            v122 += 5;
          }
          while ( v124 < v120 );
        }
        if ( v118 <= v120 )
          BUG();
        v137 = *(_QWORD *)(v115 + 32);
        v138 = 0;
        do
        {
          v139 = *(_QWORD *)(v137 + 40LL * v121);
          if ( *(_WORD *)(v139 + 24) != 48 )
          {
            if ( v138 )
            {
              if ( v139 != v138 )
                goto LABEL_104;
            }
            else
            {
              v138 = *(_QWORD *)(v137 + 40LL * v121);
            }
          }
          ++v121;
        }
        while ( v118 != v121 );
        v140 = *(_QWORD *)(v138 + 88);
        v141 = *(_DWORD *)(v140 + 32);
        if ( v141 <= 0x40 )
        {
          v143 = *(_QWORD *)(v140 + 24) == 0;
        }
        else
        {
          v191 = v123;
          v142 = sub_16A57B0(v140 + 24);
          v123 = v191;
          v143 = v141 == v142;
        }
        if ( v143 )
          v144 = *(_QWORD *)(v117 + 32);
        else
          v144 = *(_QWORD *)(v116 + 32);
        v145 = (__int128 *)(v144 + 40);
        v146 = *(_QWORD *)(v123 + 88);
        v147 = *(_DWORD *)(v146 + 32);
        if ( v147 <= 0x40 )
        {
          if ( !*(_QWORD *)(v146 + 24) )
            goto LABEL_145;
        }
        else if ( v147 == (unsigned int)sub_16A57B0(v146 + 24) )
        {
LABEL_145:
          v148 = *(_QWORD *)(v117 + 32);
          goto LABEL_146;
        }
        v148 = *(_QWORD *)(v116 + 32);
LABEL_146:
        v9 = (__int64)sub_1D332F0(
                        v195,
                        107,
                        (__int64)&v201,
                        (unsigned int)v203,
                        v204,
                        0,
                        *(double *)a3.m128i_i64,
                        *(double *)a4.m128i_i64,
                        a5,
                        *(_QWORD *)v148,
                        *(_QWORD *)(v148 + 8),
                        *v145);
        if ( v201 )
          sub_161E7C0((__int64)&v201, (__int64)v201);
        if ( v9 )
          goto LABEL_8;
        goto LABEL_27;
      }
LABEL_104:
      if ( !v201 )
        goto LABEL_28;
      sub_161E7C0((__int64)&v201, (__int64)v201);
    }
LABEL_27:
    v33 = *(const __m128i **)(a2 + 32);
    v34 = *(char **)(a2 + 40);
LABEL_28:
    v35 = v33->m128i_i64[0];
    v36 = v33->m128i_u32[2];
    v37 = (__int128)_mm_loadu_si128(v33);
    v193 = v33[2].m128i_i64[1];
    v38 = v33[5].m128i_i64[0];
    v39 = (__int128)_mm_loadu_si128(v33 + 5);
    v40 = *v34;
    v189 = v38;
    v41 = (const void **)*((_QWORD *)v34 + 1);
    LOBYTE(v199) = *v34;
    v200 = v41;
    if ( !sub_1D18C00(v35, 1, v36) )
      goto LABEL_29;
    if ( (unsigned int)sub_1F701D0(v35, v36) != 1 )
      goto LABEL_29;
    v95 = a1[1];
    v96 = *(__int64 (**)())(*v95 + 712);
    if ( v96 == sub_1F3CB60 )
      goto LABEL_29;
    if ( !((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, const void **))v96)(v95, (unsigned int)v199, v200) )
      goto LABEL_29;
    if ( !(unsigned __int8)sub_1D168E0(v193) )
      goto LABEL_29;
    v185 = sub_1D168E0(v189);
    if ( !v185 )
      goto LABEL_29;
    if ( v40 )
      v97 = word_42FA680[(unsigned __int8)(v40 - 14)];
    else
      v97 = sub_1F58D30((__int64)&v199);
    if ( v97 )
    {
      v98 = 0;
      v163 = 40LL * v97;
      v175 = v185;
      do
      {
        v107 = *(_QWORD *)(*(_QWORD *)(v193 + 32) + v98);
        if ( *(_WORD *)(v107 + 24) != 48 )
        {
          v108 = *(_QWORD *)(v98 + *(_QWORD *)(v189 + 32));
          if ( *(_WORD *)(v108 + 24) != 48 )
          {
            v109 = *(_QWORD *)(v107 + 88);
            v110 = *(_QWORD *)(v108 + 88);
            v168 = v109 + 24;
            LODWORD(v202) = *(_DWORD *)(v110 + 32);
            if ( (unsigned int)v202 > 0x40 )
              sub_16A4FD0((__int64)&v201, (const void **)(v110 + 24));
            else
              v201 = *(const void **)(v110 + 24);
            sub_16A7490((__int64)&v201, 1);
            v101 = v202;
            v100 = v201;
            LODWORD(v202) = 0;
            LODWORD(v204) = v101;
            v203 = (__int64)v201;
            if ( *(_DWORD *)(v109 + 32) > 0x40u )
            {
              v156 = v201;
              v159 = v101;
              v99 = sub_16A5220(v168, (const void **)&v203);
              v100 = v156;
              v101 = v159;
              v165 = v99;
            }
            else
            {
              v165 = *(_QWORD *)(v109 + 24) == (_QWORD)v201;
            }
            if ( v101 > 0x40 && v100 )
              j_j___libc_free_0_0(v100);
            if ( (unsigned int)v202 > 0x40 && v201 )
              j_j___libc_free_0_0(v201);
            v102 = v185;
            if ( !v165 )
              v102 = 0;
            v185 = v102;
            LODWORD(v202) = *(_DWORD *)(v110 + 32);
            if ( (unsigned int)v202 > 0x40 )
              sub_16A4FD0((__int64)&v201, (const void **)(v110 + 24));
            else
              v201 = *(const void **)(v110 + 24);
            sub_16A7800((__int64)&v201, 1u);
            v103 = v202;
            v104 = v201;
            LODWORD(v202) = 0;
            LODWORD(v204) = v103;
            v203 = (__int64)v201;
            if ( *(_DWORD *)(v109 + 32) <= 0x40u )
              v105 = *(_QWORD *)(v109 + 24) == (_QWORD)v201;
            else
              v105 = sub_16A5220(v168, (const void **)&v203);
            if ( v103 > 0x40 && v104 )
              j_j___libc_free_0_0(v104);
            if ( (unsigned int)v202 > 0x40 && v201 )
              j_j___libc_free_0_0(v201);
            v106 = v175;
            if ( !v105 )
              v106 = 0;
            v175 = v106;
          }
        }
        v98 += 40;
      }
      while ( v98 != v163 );
      v126 = *(_QWORD *)(a2 + 72);
      v203 = v126;
      if ( !v126 )
      {
LABEL_125:
        LODWORD(v204) = *(_DWORD *)(a2 + 64);
        if ( v185 )
        {
          v127 = 143;
        }
        else
        {
          v127 = 142;
          if ( !v175 )
          {
            if ( v203 )
              sub_161E7C0((__int64)&v203, v203);
            goto LABEL_29;
          }
        }
        goto LABEL_151;
      }
    }
    else
    {
      v126 = *(_QWORD *)(a2 + 72);
      v203 = v126;
      v175 = v185;
      if ( !v126 )
      {
        v127 = 143;
        LODWORD(v204) = *(_DWORD *)(a2 + 64);
LABEL_151:
        v149 = sub_1D309E0(
                 *a1,
                 v127,
                 (__int64)&v203,
                 (unsigned int)v199,
                 v200,
                 0,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 *(double *)&v37,
                 v37);
        v9 = (__int64)sub_1D332F0(
                        *a1,
                        52,
                        (__int64)&v203,
                        (unsigned int)v199,
                        v200,
                        0,
                        *(double *)a3.m128i_i64,
                        *(double *)a4.m128i_i64,
                        (__m128i)v37,
                        v149,
                        v150,
                        v39);
        if ( v203 )
          sub_161E7C0((__int64)&v203, v203);
        if ( v9 )
          goto LABEL_8;
LABEL_29:
        v9 = 0;
        goto LABEL_8;
      }
    }
    sub_1623A60((__int64)&v203, v126, 2);
    goto LABEL_125;
  }
LABEL_8:
  if ( v197 )
    sub_161E7C0((__int64)&v197, (__int64)v197);
  return v9;
}
