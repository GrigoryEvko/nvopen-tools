// Function: sub_32C0EA0
// Address: 0x32c0ea0
//
__int64 __fastcall sub_32C0EA0(__int64 **a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  __m128i v6; // xmm2
  __m128i v7; // xmm3
  __int64 v8; // rbx
  int v9; // eax
  int v10; // eax
  __int16 *v11; // rax
  __int64 v12; // rsi
  __int16 v13; // bx
  __int64 v14; // rax
  int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 *v25; // rdi
  int v26; // esi
  unsigned int v27; // ebx
  __m128i v28; // xmm5
  __m128i v29; // xmm6
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rsi
  __int64 v34; // r8
  __int64 v35; // rsi
  unsigned __int16 *v36; // rax
  __int64 v37; // rcx
  unsigned int v38; // r10d
  __int64 v39; // rdx
  __int64 *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 *v49; // rdi
  int v50; // edx
  __int64 v51; // rsi
  int v52; // ebx
  int v53; // eax
  __m128i v54; // xmm7
  __m128i v55; // xmm7
  void *v56; // r8
  __int64 v57; // rax
  void *v58; // r8
  void *v59; // rax
  __int64 v60; // r11
  __int64 v61; // rax
  void *v62; // r8
  __int64 v63; // rcx
  __int64 v64; // r8
  bool v65; // zf
  __int64 v66; // rax
  __int64 v67; // rax
  int v68; // ebx
  int v69; // esi
  int v70; // r9d
  __m128i v71; // xmm0
  __m128i v72; // xmm7
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdi
  bool v76; // al
  __int64 v77; // rcx
  bool v78; // al
  __int64 v79; // rax
  unsigned int v80; // r8d
  __int64 v81; // rax
  char v82; // al
  __int64 *v83; // rdi
  __int64 (*v84)(); // rax
  __int64 *v85; // r10
  __int64 v86; // rax
  __int64 v87; // rax
  unsigned int v88; // edx
  __int64 v89; // r9
  __m128i v90; // xmm0
  __int32 v91; // r12d
  __int64 v92; // rdx
  _QWORD *v93; // rdi
  _QWORD *v94; // rbx
  __int64 v95; // rdx
  _QWORD *v96; // rdi
  _QWORD *v97; // rbx
  __int64 *v98; // rdx
  __int64 v99; // rax
  __int64 v100; // r14
  __int64 v101; // rdx
  __int64 v102; // r15
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rax
  __int64 v106; // rax
  char v107; // al
  __int64 v108; // rax
  bool v109; // al
  __int64 v110; // r9
  __int64 (*v111)(); // rax
  int v112; // r14d
  const __m128i *v113; // r15
  __int128 v114; // rax
  __int64 v115; // r9
  __int64 v116; // r9
  int v117; // r15d
  __int128 v118; // rax
  __int64 v119; // r9
  __int64 v120; // rax
  int v121; // r15d
  __int64 v122; // rdx
  double v123; // rax
  __int64 v124; // rcx
  __int128 v125; // rax
  const __m128i *v126; // r14
  __int64 v127; // r9
  __int128 v128; // rax
  __int64 v129; // r9
  __int64 v130; // r9
  int v131; // r14d
  __int128 v132; // rax
  __int64 v133; // r9
  __int64 v134; // rdx
  __int128 v135; // [rsp-20h] [rbp-260h]
  __int64 v136; // [rsp-10h] [rbp-250h]
  __int128 v137; // [rsp-10h] [rbp-250h]
  __int128 v138; // [rsp-10h] [rbp-250h]
  __int128 v139; // [rsp-10h] [rbp-250h]
  __int128 v140; // [rsp-10h] [rbp-250h]
  unsigned int v141; // [rsp+0h] [rbp-240h]
  void *v142; // [rsp+0h] [rbp-240h]
  __int64 v143; // [rsp+8h] [rbp-238h]
  __int64 v144; // [rsp+8h] [rbp-238h]
  void *v145; // [rsp+8h] [rbp-238h]
  unsigned int v146; // [rsp+8h] [rbp-238h]
  __int64 v147; // [rsp+10h] [rbp-230h]
  __int64 v148; // [rsp+10h] [rbp-230h]
  __int64 v149; // [rsp+10h] [rbp-230h]
  unsigned int v150; // [rsp+10h] [rbp-230h]
  unsigned __int32 v151; // [rsp+18h] [rbp-228h]
  __int32 v152; // [rsp+1Ch] [rbp-224h]
  __int64 v153; // [rsp+20h] [rbp-220h]
  __int64 v154; // [rsp+28h] [rbp-218h]
  __int64 v155; // [rsp+30h] [rbp-210h]
  _DWORD *v156; // [rsp+30h] [rbp-210h]
  bool v157; // [rsp+30h] [rbp-210h]
  __int64 v158; // [rsp+30h] [rbp-210h]
  unsigned int v159; // [rsp+30h] [rbp-210h]
  __int64 v160; // [rsp+30h] [rbp-210h]
  __int64 v161; // [rsp+38h] [rbp-208h]
  __int64 v162; // [rsp+38h] [rbp-208h]
  _DWORD *v163; // [rsp+38h] [rbp-208h]
  bool v164; // [rsp+38h] [rbp-208h]
  char v165; // [rsp+38h] [rbp-208h]
  unsigned int v166; // [rsp+38h] [rbp-208h]
  __int64 v167; // [rsp+40h] [rbp-200h]
  __int64 v168; // [rsp+48h] [rbp-1F8h]
  int v169; // [rsp+48h] [rbp-1F8h]
  __m128i v170; // [rsp+50h] [rbp-1F0h] BYREF
  const __m128i *v171; // [rsp+60h] [rbp-1E0h]
  __int64 v172; // [rsp+68h] [rbp-1D8h]
  __int128 v173; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v174; // [rsp+80h] [rbp-1C0h]
  __int64 v175; // [rsp+88h] [rbp-1B8h]
  __int64 v176; // [rsp+90h] [rbp-1B0h]
  __int64 v177; // [rsp+98h] [rbp-1A8h]
  int v178; // [rsp+A8h] [rbp-198h] BYREF
  int v179; // [rsp+ACh] [rbp-194h] BYREF
  __m128i v180; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v181; // [rsp+C0h] [rbp-180h] BYREF
  int v182; // [rsp+C8h] [rbp-178h]
  __int64 *v183; // [rsp+D0h] [rbp-170h] BYREF
  int v184; // [rsp+D8h] [rbp-168h]
  __int64 v185; // [rsp+E0h] [rbp-160h]
  __m128i v186; // [rsp+F0h] [rbp-150h] BYREF
  __m128i v187; // [rsp+100h] [rbp-140h] BYREF
  __m128i v188; // [rsp+110h] [rbp-130h] BYREF
  __int64 v189; // [rsp+120h] [rbp-120h]
  void *v190; // [rsp+130h] [rbp-110h] BYREF
  _QWORD *v191; // [rsp+138h] [rbp-108h]
  __int64 v192; // [rsp+140h] [rbp-100h]
  int v193; // [rsp+148h] [rbp-F8h]
  __int64 v194; // [rsp+150h] [rbp-F0h]
  __int32 v195; // [rsp+158h] [rbp-E8h]
  __m128i v196; // [rsp+160h] [rbp-E0h]
  __m128i v197; // [rsp+170h] [rbp-D0h]
  __m128i si128; // [rsp+180h] [rbp-C0h] BYREF
  __m128i v199; // [rsp+190h] [rbp-B0h]
  __m128i v200; // [rsp+1A0h] [rbp-A0h]
  __m128i v201; // [rsp+1B0h] [rbp-90h]
  __int64 v202; // [rsp+1C0h] [rbp-80h]
  int v203; // [rsp+1C8h] [rbp-78h]
  __int64 v204; // [rsp+1D0h] [rbp-70h]
  __int64 v205; // [rsp+1D8h] [rbp-68h]
  __int64 v206; // [rsp+1E0h] [rbp-60h] BYREF
  int v207; // [rsp+1E8h] [rbp-58h]
  __m128i *p_si128; // [rsp+1F0h] [rbp-50h]
  __int64 v209; // [rsp+1F8h] [rbp-48h]
  __int64 v210; // [rsp+200h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)v4;
  v6 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v7 = _mm_loadu_si128((const __m128i *)(v4 + 80));
  v151 = *(_DWORD *)(v4 + 8);
  v8 = *(_QWORD *)(v4 + 80);
  LODWORD(v4) = *(_DWORD *)(v4 + 88);
  v168 = v5;
  v173 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v152 = v4;
  v9 = *(_DWORD *)(v5 + 24);
  v154 = v8;
  v180 = v6;
  v170 = v7;
  if ( v9 == 12 || (v155 = 0, v9 == 36) )
    v155 = v5;
  v161 = v180.m128i_i64[0];
  v10 = *(_DWORD *)(v180.m128i_i64[0] + 24);
  if ( v10 == 36 || (v167 = 0, v10 == 12) )
    v167 = v180.m128i_i64[0];
  v11 = *(__int16 **)(a2 + 48);
  v12 = *(_QWORD *)(a2 + 80);
  v13 = *v11;
  v14 = *((_QWORD *)v11 + 1);
  v181 = v12;
  LOWORD(v171) = v13;
  v172 = v14;
  if ( v12 )
    sub_B96E90((__int64)&v181, v12, 1);
  v15 = *(_DWORD *)(a2 + 28);
  v182 = *(_DWORD *)(a2 + 72);
  v16 = *a1;
  v17 = **a1;
  v184 = v15;
  v18 = v16[128];
  v183 = v16;
  v153 = v17;
  v185 = v18;
  v16[128] = (__int64)&v183;
  v19 = (__int64)a1[1];
  v20 = *a1;
  v21 = *(unsigned int *)(a2 + 24);
  v187.m128i_i64[0] = 0;
  v186.m128i_i64[1] = v19;
  v186.m128i_i64[0] = (__int64)v20;
  v187.m128i_i32[2] = 0;
  v188.m128i_i64[0] = 0;
  v188.m128i_i32[2] = 0;
  v189 = a2;
  v190 = (void *)sub_33CB160(v21);
  if ( BYTE4(v190) )
  {
    v22 = *(_QWORD *)(v189 + 40) + 40LL * (unsigned int)v190;
    v187.m128i_i64[0] = *(_QWORD *)v22;
    v187.m128i_i32[2] = *(_DWORD *)(v22 + 8);
    v23 = *(unsigned int *)(v189 + 24);
  }
  else
  {
    v34 = v189;
    v23 = *(unsigned int *)(v189 + 24);
    if ( (_DWORD)v23 == 488 )
    {
      v35 = *(_QWORD *)(v189 + 80);
      v36 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v189 + 40) + 48LL)
                               + 16LL * *(unsigned int *)(*(_QWORD *)(v189 + 40) + 8LL));
      v37 = *((_QWORD *)v36 + 1);
      v38 = *v36;
      si128.m128i_i64[0] = v35;
      if ( v35 )
      {
        v141 = v38;
        v143 = v37;
        v147 = v189;
        sub_B96E90((__int64)&si128, v35, 1);
        v38 = v141;
        v37 = v143;
        v34 = v147;
      }
      si128.m128i_i32[2] = *(_DWORD *)(v34 + 72);
      v176 = sub_34015B0(v20, &si128, v38, v37, 0, 0);
      v177 = v39;
      v187.m128i_i64[0] = v176;
      v187.m128i_i32[2] = v39;
      if ( si128.m128i_i64[0] )
        sub_B91220((__int64)&si128, si128.m128i_i64[0]);
      v23 = *(unsigned int *)(v189 + 24);
    }
  }
  si128.m128i_i64[0] = sub_33CB1F0(v23);
  if ( si128.m128i_i8[4] )
  {
    v24 = *(_QWORD *)(v189 + 40) + 40LL * si128.m128i_u32[0];
    v188.m128i_i64[0] = *(_QWORD *)v24;
    v188.m128i_i32[2] = *(_DWORD *)(v24 + 8);
  }
  v25 = *a1;
  v26 = *(_DWORD *)(a2 + 24);
  v27 = (unsigned __int16)v171;
  v28 = _mm_loadu_si128(&v180);
  v29 = _mm_load_si128(&v170);
  si128 = _mm_load_si128((const __m128i *)&v173);
  v199 = v28;
  v200 = v29;
  v30 = sub_3402EA0((_DWORD)v25, v26, (unsigned int)&v181, (unsigned __int16)v171, v172, 0, (__int64)&si128, 3);
  if ( v30 )
    goto LABEL_12;
  v40 = a1[1];
  v178 = 2;
  v179 = 2;
  v41 = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64 *, _QWORD, _QWORD, int *, _QWORD))(*v40 + 2264))(
          v40,
          v173,
          *((_QWORD *)&v173 + 1),
          *a1,
          *((unsigned __int8 *)a1 + 33),
          *((unsigned __int8 *)a1 + 35),
          &v178,
          0);
  v148 = v42;
  v144 = v41;
  if ( v41 )
  {
    v46 = sub_33ECD10(1, v136, v42, v43, v44, v45);
    v210 = 0;
    v201 = (__m128i)(unsigned __int64)v46;
    v202 = 0x100000000LL;
    si128 = 0u;
    v199.m128i_i64[0] = 0;
    v199.m128i_i64[1] = 328;
    v200.m128i_i64[0] = -65536;
    v203 = 0;
    v204 = 0;
    v205 = 0xFFFFFFFFLL;
    v209 = 0;
    p_si128 = &si128;
    v174 = v144;
    v175 = v148;
    v206 = v144;
    v207 = v148;
    v47 = *(_QWORD *)(v144 + 56);
    v210 = v47;
    if ( v47 )
      *(_QWORD *)(v47 + 24) = &v210;
    v209 = v144 + 56;
    v48 = v180.m128i_i64[0];
    *(_QWORD *)(v144 + 56) = &v206;
    v49 = a1[1];
    LODWORD(v202) = 1;
    v200.m128i_i64[1] = (__int64)&v206;
    v51 = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64 *, _QWORD, _QWORD, int *, _QWORD))(*v49 + 2264))(
            v49,
            v48,
            v180.m128i_i64[1],
            *a1,
            *((unsigned __int8 *)a1 + 33),
            *((unsigned __int8 *)a1 + 35),
            &v179,
            0);
    if ( v51 && (!v178 || !v179) )
    {
      v169 = v50;
      v52 = (unsigned __int16)v171;
      v170.m128i_i64[0] = v51;
      *(_QWORD *)&v173 = v144;
      v53 = sub_33CB7C0(150);
      v54 = _mm_loadu_si128(&v187);
      v193 = v169;
      *((_QWORD *)&v137 + 1) = 5;
      LODWORD(v191) = v148;
      v196 = v54;
      v55 = _mm_loadu_si128(&v188);
      v194 = v154;
      v192 = v170.m128i_i64[0];
      v195 = v152;
      *(_QWORD *)&v137 = &v190;
      v190 = (void *)v173;
      v197 = v55;
      v31 = sub_33FC220(v186.m128i_i32[0], v53, (unsigned int)&v181, v52, v172, v53, v137);
      sub_33CF710(&si128);
      goto LABEL_13;
    }
    sub_33CF710(&si128);
  }
  if ( (*(_BYTE *)(v153 + 864) & 1) != 0 )
  {
    if ( v155 )
    {
      v149 = *(_QWORD *)(v155 + 96);
      v56 = sub_C33340();
      if ( *(void **)(v149 + 24) == v56 )
        v73 = *(_QWORD *)(v149 + 32);
      else
        v73 = v149 + 24;
      if ( (*(_BYTE *)(v73 + 20) & 7) == 3 )
        goto LABEL_86;
      if ( !v167 )
      {
LABEL_35:
        v142 = v56;
        v156 = sub_C33320();
        sub_C3B1B0((__int64)&si128, 1.0);
        sub_C407B0(&v190, si128.m128i_i64, v156);
        sub_C338F0((__int64)&si128);
        sub_C41640((__int64 *)&v190, *(_DWORD **)(v149 + 24), 1, (bool *)si128.m128i_i8);
        v57 = (__int64)v190;
        v157 = 0;
        v58 = v142;
        if ( *(void **)(v149 + 24) == v190 )
        {
          v77 = v149 + 24;
          if ( v190 == v142 )
            v78 = sub_C3E590(v77, (__int64)&v190);
          else
            v78 = sub_C33D00(v77, (__int64)&v190);
          v58 = v142;
          v157 = v78;
          v57 = (__int64)v190;
        }
        if ( (void *)v57 == v58 )
        {
          if ( v191 )
          {
            v95 = *(v191 - 1);
            v96 = &v191[3 * v95];
            if ( v191 != v96 )
            {
              v146 = v27;
              v97 = &v191[3 * v95];
              do
              {
                v97 -= 3;
                sub_91D830(v97);
              }
              while ( v191 != v97 );
              v96 = v97;
              v27 = v146;
            }
            j_j_j___libc_free_0_0((unsigned __int64)(v96 - 1));
          }
        }
        else
        {
          sub_C338F0((__int64)&v190);
        }
        if ( v157 )
        {
          v91 = v180.m128i_i32[2];
          v68 = (unsigned __int16)v171;
          v69 = sub_33CB7C0(96);
          si128.m128i_i32[2] = v91;
          si128.m128i_i64[0] = v161;
          goto LABEL_59;
        }
        goto LABEL_39;
      }
    }
    else
    {
      if ( !v167 )
        goto LABEL_45;
      v56 = sub_C33340();
    }
    v60 = *(_QWORD *)(v167 + 96);
    v74 = v60 + 24;
    if ( *(void **)(v60 + 24) == v56 )
      v74 = *(_QWORD *)(v60 + 32);
    if ( (*(_BYTE *)(v74 + 20) & 7) != 3 )
    {
      if ( !v155 )
        goto LABEL_41;
      v149 = *(_QWORD *)(v155 + 96);
      goto LABEL_35;
    }
LABEL_86:
    v30 = v170.m128i_i64[0];
LABEL_12:
    v31 = v30;
    goto LABEL_13;
  }
  if ( v155 )
  {
    v149 = *(_QWORD *)(v155 + 96);
    v56 = sub_C33340();
    goto LABEL_35;
  }
LABEL_39:
  if ( !v167 )
    goto LABEL_45;
  v162 = *(_QWORD *)(v167 + 96);
  v59 = sub_C33340();
  v60 = v162;
  v56 = v59;
LABEL_41:
  v145 = v56;
  v158 = v60;
  v163 = sub_C33320();
  sub_C3B1B0((__int64)&si128, 1.0);
  sub_C407B0(&v190, si128.m128i_i64, v163);
  sub_C338F0((__int64)&si128);
  sub_C41640((__int64 *)&v190, *(_DWORD **)(v158 + 24), 1, (bool *)si128.m128i_i8);
  v61 = (__int64)v190;
  v164 = 0;
  v62 = v145;
  if ( *(void **)(v158 + 24) == v190 )
  {
    v75 = v158 + 24;
    if ( v190 == v145 )
      v76 = sub_C3E590(v75, (__int64)&v190);
    else
      v76 = sub_C33D00(v75, (__int64)&v190);
    v62 = v145;
    v164 = v76;
    v61 = (__int64)v190;
  }
  if ( v62 == (void *)v61 )
  {
    if ( v191 )
    {
      v92 = *(v191 - 1);
      v93 = &v191[3 * v92];
      if ( v191 != v93 )
      {
        v150 = v27;
        v94 = &v191[3 * v92];
        do
        {
          v94 -= 3;
          sub_91D830(v94);
        }
        while ( v191 != v94 );
        v93 = v94;
        v27 = v150;
      }
      j_j_j___libc_free_0_0((unsigned __int64)(v93 - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v190);
  }
  if ( v164 )
  {
    v68 = (unsigned __int16)v171;
    v69 = sub_33CB7C0(96);
    v200 = _mm_loadu_si128(&v187);
    v90 = _mm_loadu_si128(&v188);
    si128.m128i_i64[0] = v168;
    v201 = v90;
    si128.m128i_i32[2] = v151;
    v199.m128i_i64[0] = v154;
    v199.m128i_i32[2] = v152;
    goto LABEL_60;
  }
LABEL_45:
  if ( (unsigned __int8)sub_33E2470(*a1, v173, *((_QWORD *)&v173 + 1))
    && !(unsigned __int8)sub_33E2470(*a1, v180.m128i_i64[0], v180.m128i_i64[1]) )
  {
    v31 = sub_3290460(&v186, 0x96u, (int)&v181, v27, v172, v89, *(_OWORD *)&v180, v173, *(_OWORD *)&v170);
    goto LABEL_13;
  }
  if ( (*(_BYTE *)(v153 + 864) & 1) == 0 && (*(_BYTE *)(a2 + 29) & 8) == 0 )
  {
    v165 = 0;
    goto LABEL_56;
  }
  v65 = (unsigned __int8)sub_33CB110(*(unsigned int *)(v154 + 24)) == 0;
  v66 = v154;
  if ( v65 )
  {
    if ( *(_DWORD *)(v154 + 24) != 98 )
      goto LABEL_52;
  }
  else
  {
    v190 = (void *)sub_33CB280(*(unsigned int *)(v154 + 24), ((unsigned __int8)(*(_DWORD *)(v154 + 28) >> 12) ^ 1) & 1);
    if ( !BYTE4(v190) || (_DWORD)v190 != 98 )
      goto LABEL_52;
    v166 = *(_DWORD *)(v154 + 24);
    v79 = sub_33CB160(v166);
    v80 = v166;
    si128.m128i_i64[0] = v79;
    if ( BYTE4(v79) )
    {
      v81 = *(_QWORD *)(v154 + 40) + 40LL * si128.m128i_u32[0];
      if ( *(_QWORD *)v81 != v187.m128i_i64[0] || *(_DWORD *)(v81 + 8) != v187.m128i_i32[2] )
      {
        v82 = sub_33D1720(*(_QWORD *)v81, 0);
        v80 = v166;
        if ( !v82 )
          goto LABEL_52;
      }
    }
    si128.m128i_i64[0] = sub_33CB1F0(v80);
    if ( si128.m128i_i8[4] )
    {
      v67 = *(_QWORD *)(v154 + 40);
      v134 = v67 + 40LL * si128.m128i_u32[0];
      if ( v188.m128i_i64[0] != *(_QWORD *)v134 || v188.m128i_i32[2] != *(_DWORD *)(v134 + 8) )
        goto LABEL_52;
      goto LABEL_51;
    }
    v66 = v154;
  }
  v67 = *(_QWORD *)(v66 + 40);
LABEL_51:
  if ( *(_QWORD *)v67 == v168
    && *(_DWORD *)(v67 + 8) == v151
    && (unsigned __int8)sub_33E2470(*a1, v180.m128i_i64[0], v180.m128i_i64[1])
    && (unsigned __int8)sub_33E2470(
                          *a1,
                          *(_QWORD *)(*(_QWORD *)(v154 + 40) + 40LL),
                          *(_QWORD *)(*(_QWORD *)(v154 + 40) + 48LL)) )
  {
    v117 = v172;
    *(_QWORD *)&v118 = sub_328FC10(
                         &v186,
                         0x60u,
                         (int)&v181,
                         v27,
                         v172,
                         v116,
                         *(_OWORD *)&v180,
                         *(_OWORD *)(*(_QWORD *)(v154 + 40) + 40LL));
    v31 = sub_328FC10(&v186, 0x62u, (int)&v181, v27, v117, v119, v173, v118);
    goto LABEL_13;
  }
LABEL_52:
  v165 = sub_33CB110(*(unsigned int *)(v168 + 24));
  if ( v165 )
  {
    v190 = (void *)sub_33CB280(*(unsigned int *)(v168 + 24), ((unsigned __int8)(*(_DWORD *)(v168 + 28) >> 12) ^ 1) & 1);
    if ( !BYTE4(v190) || (_DWORD)v190 != 98 )
      goto LABEL_56;
    v159 = *(_DWORD *)(v168 + 24);
    v105 = sub_33CB160(v159);
    LODWORD(v64) = v159;
    si128.m128i_i64[0] = v105;
    v165 = BYTE4(v105);
    if ( BYTE4(v105) )
    {
      v106 = *(_QWORD *)(v168 + 40) + 40LL * si128.m128i_u32[0];
      if ( *(_QWORD *)v106 != v187.m128i_i64[0] || *(_DWORD *)(v106 + 8) != v187.m128i_i32[2] )
      {
        v107 = sub_33D1720(*(_QWORD *)v106, 0);
        v64 = v159;
        if ( !v107 )
          goto LABEL_56;
      }
    }
    si128.m128i_i64[0] = sub_33CB1F0((unsigned int)v64);
    v165 = si128.m128i_i8[4];
    if ( si128.m128i_i8[4] )
    {
      v108 = *(_QWORD *)(v168 + 40) + 40LL * si128.m128i_u32[0];
      v63 = *(_QWORD *)v108;
      if ( v188.m128i_i64[0] != *(_QWORD *)v108 || v188.m128i_i32[2] != *(_DWORD *)(v108 + 8) )
        goto LABEL_56;
    }
  }
  else if ( *(_DWORD *)(v168 + 24) != 98 )
  {
LABEL_55:
    v165 = 1;
LABEL_56:
    if ( v167 )
    {
      if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v167 + 96) + 24LL), 1.0) )
      {
        v68 = (unsigned __int16)v171;
        v69 = sub_33CB7C0(96);
        si128.m128i_i64[0] = v168;
        si128.m128i_i32[2] = v151;
LABEL_59:
        v71 = _mm_loadu_si128(&v187);
        v72 = _mm_loadu_si128(&v188);
        v199.m128i_i64[0] = v154;
        v200 = v71;
        v199.m128i_i32[2] = v152;
        v201 = v72;
LABEL_60:
        *((_QWORD *)&v138 + 1) = 4;
        *(_QWORD *)&v138 = &si128;
        v31 = sub_33FC220(v186.m128i_i32[0], v69, (unsigned int)&v181, v68, v172, v70, v138);
        goto LABEL_13;
      }
      if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v167 + 96) + 24LL), -1.0) )
      {
        if ( !*((_BYTE *)a1 + 33)
          || ((v98 = a1[1], v99 = 1, (_WORD)v171 == 1)
           || (_WORD)v171 && (v99 = (unsigned __int16)v171, v98[(unsigned __int16)v171 + 14]))
          && !*((_BYTE *)v98 + 500 * v99 + 6658) )
        {
          v139 = v173;
          *(_QWORD *)&v173 = &v186;
          v100 = sub_328FBA0(&v186, 0xF4u, (int)&v181, v27, v172, (__int64)&v186, v139);
          v102 = v101;
          sub_32B3E80((__int64)a1, v100, 1, 0, v103, v104);
          *((_QWORD *)&v135 + 1) = v102;
          *(_QWORD *)&v135 = v100;
          v31 = sub_328FC10((const __m128i *)v173, 0x60u, (int)&v181, v27, v172, v173, *(_OWORD *)&v170, v135);
          goto LABEL_13;
        }
      }
      v171 = &v186;
      if ( (unsigned __int8)sub_325F380((__int64)&v186, v168, 244) )
      {
        v160 = (__int64)a1[1];
        v109 = sub_328D6E0(v160, 0xCu, v27);
        v110 = v160;
        if ( v109
          || (unsigned __int8)sub_3286E00(&v180)
          && ((v110 = v160, v111 = *(__int64 (**)())(*(_QWORD *)v160 + 616LL), v111 == sub_2FE3170)
           || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, __int64, _QWORD))v111)(
                 v160,
                 *(_QWORD *)(v167 + 96) + 24LL,
                 v27,
                 v172,
                 *((unsigned __int8 *)a1 + 35))) )
        {
          v112 = v172;
          v113 = v171;
          *(_QWORD *)&v114 = sub_328FBA0(v171, 0xF4u, (int)&v181, v27, v172, v110, *(_OWORD *)&v180);
          v31 = sub_3290460(
                  v113,
                  0x96u,
                  (int)&v181,
                  v27,
                  v112,
                  v115,
                  *(_OWORD *)*(_QWORD *)(v168 + 40),
                  v114,
                  *(_OWORD *)&v170);
          goto LABEL_13;
        }
      }
      if ( v165 )
      {
        if ( v152 == v151 && v154 == v168 )
        {
          v121 = v172;
          v122 = v27;
          v123 = 1.0;
          v124 = v172;
          goto LABEL_148;
        }
        if ( (unsigned __int8)sub_325F380((__int64)v171, v154, 244) )
        {
          v63 = v168;
          v120 = *(_QWORD *)(v154 + 40);
          if ( *(_QWORD *)v120 == v168 )
          {
            v63 = v151;
            if ( *(_DWORD *)(v120 + 8) == v151 )
            {
              v121 = v172;
              v122 = v27;
              v123 = -1.0;
              v124 = v172;
LABEL_148:
              *(_QWORD *)&v125 = sub_33FE730(*a1, &v181, v122, v124, 0, v123);
              v126 = v171;
              *(_QWORD *)&v128 = sub_328FC10(v171, 0x60u, (int)&v181, v27, v121, v127, *(_OWORD *)&v180, v125);
              v31 = sub_328FC10(v126, 0x62u, (int)&v181, v27, v121, v129, v173, v128);
              goto LABEL_13;
            }
          }
        }
      }
    }
    v83 = a1[1];
    v84 = *(__int64 (**)())(*v83 + 1592);
    if ( v84 != sub_2FE3530 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, __int64, __int64, __int64))v84)(v83, v27, v172, v63, v64) )
        goto LABEL_92;
      v83 = a1[1];
    }
    v85 = *a1;
    si128.m128i_i32[0] = 2;
    v86 = *v83;
    *(_QWORD *)&v173 = v85;
    v87 = (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD, __int64 *, _QWORD, _QWORD, __m128i *, _QWORD))(v86 + 2264))(
            v83,
            a2,
            0,
            v85,
            *((unsigned __int8 *)a1 + 33),
            *((unsigned __int8 *)a1 + 35),
            &si128,
            0);
    if ( v87 )
    {
      if ( si128.m128i_i32[0] <= 0 )
      {
        *((_QWORD *)&v140 + 1) = v88;
        *(_QWORD *)&v140 = v87;
        v31 = sub_328FBA0(&v186, 0xF4u, (int)&v181, v27, v172, (__int64)&v186, v140);
        goto LABEL_13;
      }
      if ( !*(_QWORD *)(v87 + 56) )
        sub_33ECEA0(v173, v87);
    }
LABEL_92:
    v31 = 0;
    goto LABEL_13;
  }
  if ( !(unsigned __int8)sub_33E2470(*a1, v180.m128i_i64[0], v180.m128i_i64[1])
    || !(unsigned __int8)sub_33E2470(
                           *a1,
                           *(_QWORD *)(*(_QWORD *)(v168 + 40) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(v168 + 40) + 48LL)) )
  {
    goto LABEL_55;
  }
  v131 = v172;
  *(_QWORD *)&v132 = sub_328FC10(
                       &v186,
                       0x62u,
                       (int)&v181,
                       v27,
                       v172,
                       v130,
                       *(_OWORD *)&v180,
                       *(_OWORD *)(*(_QWORD *)(v168 + 40) + 40LL));
  v31 = sub_3290460(
          &v186,
          0x96u,
          (int)&v181,
          v27,
          v131,
          v133,
          *(_OWORD *)*(_QWORD *)(v168 + 40),
          v132,
          *(_OWORD *)&v170);
LABEL_13:
  v32 = v181;
  v183[128] = v185;
  if ( v32 )
    sub_B91220((__int64)&v181, v32);
  return v31;
}
