// Function: sub_28BFDB0
// Address: 0x28bfdb0
//
__int64 __fastcall sub_28BFDB0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // rax
  bool v13; // zf
  unsigned __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  int v21; // edx
  __int64 v22; // r12
  __int64 v23; // rax
  __m128i *v24; // rdi
  __m128i v25; // xmm6
  __m128i v26; // xmm7
  __m128i v27; // xmm6
  __m128i v28; // xmm7
  __m128i v29; // xmm6
  __m128i v30; // xmm7
  __m128i v31; // xmm6
  __m128i v32; // xmm7
  void (__fastcall *v33)(__m128i *, __m128i *, __int64); // rax
  __int64 v34; // rsi
  int v35; // r12d
  __int64 v36; // rdx
  _QWORD *v37; // r15
  _QWORD *v38; // rbx
  __int64 v39; // r13
  unsigned __int64 v40; // rax
  __m128i v41; // xmm1
  __m128i v42; // xmm2
  __int64 v43; // r13
  unsigned __int8 *v44; // rdi
  unsigned __int64 v45; // rax
  __int64 v46; // rdx
  __m128i v47; // xmm4
  __m128i v48; // xmm5
  __int64 v49; // r13
  unsigned __int8 *v50; // rdi
  unsigned int v51; // r8d
  __int64 v52; // rax
  char v53; // al
  _BYTE *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int16 v57; // dx
  __int64 v58; // r14
  __int64 v59; // rax
  char v60; // dl
  __int64 v61; // r13
  __int64 v62; // rax
  void (__fastcall *v63)(__m128i *, __m128i *, __int64); // rax
  __int64 v64; // rsi
  __int64 v65; // r9
  __int64 v66; // rax
  __int32 v67; // ecx
  unsigned __int64 *v68; // rdx
  unsigned __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // rax
  int v72; // edi
  char v73; // si
  int v74; // esi
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // r14
  __int64 v78; // r13
  __m128i v79; // rax
  __int64 v80; // rax
  __int64 v81; // rsi
  __int64 v82; // r14
  _QWORD *v83; // r10
  __int64 v84; // rax
  __int64 v85; // r12
  __int64 v86; // rax
  __int64 v87; // r12
  int v88; // eax
  int v89; // eax
  unsigned int v90; // edx
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rdx
  __int64 v94; // r12
  int v95; // eax
  int v96; // eax
  unsigned int v97; // edx
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rdx
  __int64 v101; // rax
  __int64 v102; // rax
  _QWORD *v103; // r13
  __int64 v104; // rax
  __int64 v105; // r12
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rax
  int v109; // r9d
  unsigned __int64 v110; // r8
  unsigned __int64 v111; // r10
  __int64 v112; // [rsp+0h] [rbp-2F0h]
  __int64 v115; // [rsp+18h] [rbp-2D8h]
  __int64 v116; // [rsp+20h] [rbp-2D0h]
  __int64 v117; // [rsp+30h] [rbp-2C0h]
  __int64 v118; // [rsp+38h] [rbp-2B8h]
  char v119; // [rsp+46h] [rbp-2AAh]
  char v120; // [rsp+47h] [rbp-2A9h]
  __int64 v121; // [rsp+50h] [rbp-2A0h]
  __int64 v122; // [rsp+68h] [rbp-288h]
  bool v123; // [rsp+78h] [rbp-278h]
  __int64 v124; // [rsp+80h] [rbp-270h]
  __int64 v125; // [rsp+88h] [rbp-268h]
  bool v126; // [rsp+88h] [rbp-268h]
  int v127; // [rsp+90h] [rbp-260h]
  int v128; // [rsp+94h] [rbp-25Ch]
  __int64 v129; // [rsp+A0h] [rbp-250h]
  _QWORD *v130; // [rsp+A0h] [rbp-250h]
  __int64 v131; // [rsp+A0h] [rbp-250h]
  __int64 v132; // [rsp+B0h] [rbp-240h]
  unsigned __int64 v133; // [rsp+B8h] [rbp-238h]
  __int64 v134; // [rsp+B8h] [rbp-238h]
  unsigned __int64 v135; // [rsp+B8h] [rbp-238h]
  __int64 v136; // [rsp+C0h] [rbp-230h]
  __int64 v137; // [rsp+C0h] [rbp-230h]
  _QWORD *v138; // [rsp+C0h] [rbp-230h]
  __int64 v139; // [rsp+C8h] [rbp-228h]
  __int64 v140; // [rsp+C8h] [rbp-228h]
  __int64 v141; // [rsp+C8h] [rbp-228h]
  __int64 v142; // [rsp+D0h] [rbp-220h]
  _QWORD *v143; // [rsp+D0h] [rbp-220h]
  __int64 v144; // [rsp+D0h] [rbp-220h]
  __int64 v145; // [rsp+D8h] [rbp-218h]
  unsigned __int64 v146; // [rsp+D8h] [rbp-218h]
  unsigned __int64 v147; // [rsp+D8h] [rbp-218h]
  _QWORD *v148; // [rsp+D8h] [rbp-218h]
  _QWORD *v149; // [rsp+E0h] [rbp-210h]
  unsigned __int8 *v150; // [rsp+E8h] [rbp-208h]
  __m128i v151; // [rsp+F0h] [rbp-200h] BYREF
  __m128i v152; // [rsp+100h] [rbp-1F0h] BYREF
  __m128i v153; // [rsp+110h] [rbp-1E0h] BYREF
  void (__fastcall *v154)(__m128i *, __m128i *, __int64); // [rsp+120h] [rbp-1D0h]
  unsigned __int8 (__fastcall *v155)(__m128i *, __int64); // [rsp+128h] [rbp-1C8h]
  __m128i v156; // [rsp+130h] [rbp-1C0h] BYREF
  __m128i v157; // [rsp+140h] [rbp-1B0h] BYREF
  __m128i v158; // [rsp+150h] [rbp-1A0h] BYREF
  void (__fastcall *v159)(__m128i *, __m128i *, __int64); // [rsp+160h] [rbp-190h]
  unsigned __int64 v160; // [rsp+168h] [rbp-188h]
  __m128i v161; // [rsp+170h] [rbp-180h] BYREF
  __m128i v162; // [rsp+180h] [rbp-170h] BYREF
  __m128i v163; // [rsp+190h] [rbp-160h] BYREF
  void (__fastcall *v164)(__m128i *, __m128i *, __int64); // [rsp+1A0h] [rbp-150h]
  unsigned __int8 (__fastcall *v165)(__m128i *, __int64); // [rsp+1A8h] [rbp-148h]
  __m128i v166; // [rsp+1B0h] [rbp-140h] BYREF
  __m128i v167; // [rsp+1C0h] [rbp-130h] BYREF
  __m128i v168; // [rsp+1D0h] [rbp-120h] BYREF
  void (__fastcall *v169)(__m128i *, __m128i *, __int64); // [rsp+1E0h] [rbp-110h]
  unsigned __int8 (__fastcall *v170)(__m128i *, __int64); // [rsp+1E8h] [rbp-108h]
  __m128i v171; // [rsp+1F0h] [rbp-100h] BYREF
  __m128i v172; // [rsp+200h] [rbp-F0h] BYREF
  __m128i v173; // [rsp+210h] [rbp-E0h] BYREF
  void (__fastcall *v174)(__m128i *, __m128i *, __int64); // [rsp+220h] [rbp-D0h]
  unsigned __int64 v175; // [rsp+228h] [rbp-C8h]
  __m128i v176; // [rsp+230h] [rbp-C0h] BYREF
  __m128i v177; // [rsp+240h] [rbp-B0h] BYREF
  __m128i i; // [rsp+250h] [rbp-A0h] BYREF
  void (__fastcall *v179)(__m128i *, __m128i *, __int64); // [rsp+260h] [rbp-90h] BYREF
  _QWORD *v180; // [rsp+268h] [rbp-88h]
  __int64 v181; // [rsp+270h] [rbp-80h]
  __int64 v182; // [rsp+278h] [rbp-78h]
  _QWORD v183[3]; // [rsp+280h] [rbp-70h] BYREF
  int v184; // [rsp+298h] [rbp-58h]
  __int16 v185; // [rsp+29Ch] [rbp-54h]
  char v186; // [rsp+29Eh] [rbp-52h]
  __int64 v187; // [rsp+2A0h] [rbp-50h]
  __int64 v188; // [rsp+2A8h] [rbp-48h]
  void *v189; // [rsp+2B0h] [rbp-40h] BYREF
  void *v190; // [rsp+2B8h] [rbp-38h] BYREF

  v119 = *a2;
  v120 = 0;
  v149 = (_QWORD *)(sub_BC1CD0(a4, &unk_4F86540, a3) + 8);
  v5 = a3 + 72;
  v6 = *(_QWORD *)(a3 + 80);
  v115 = v5;
  if ( v5 == v6 )
    goto LABEL_207;
  do
  {
    v7 = v6 + 24;
    v8 = *(_QWORD *)(v6 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v6 + 24 == v8 || !v8 || (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
LABEL_213:
      BUG();
    v117 = *(_QWORD *)(v6 + 8);
    if ( *(_BYTE *)(v8 - 24) != 31 )
      goto LABEL_10;
    if ( (*(_DWORD *)(v8 - 20) & 0x7FFFFFF) != 3 )
      goto LABEL_10;
    v9 = *(_QWORD *)(v8 - 56);
    v10 = *(_QWORD *)(v8 - 88);
    if ( !sub_AA54C0(v9) )
      goto LABEL_10;
    if ( !sub_AA54C0(v10) )
      goto LABEL_10;
    v11 = sub_AA56F0(v9);
    v12 = sub_AA56F0(v10);
    v123 = v11 != v12 || v12 == 0 || v11 == 0;
    if ( v123 )
      goto LABEL_10;
    v15 = *(_QWORD *)(v6 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v7 == v15 )
    {
      v17 = 0;
    }
    else
    {
      if ( !v15 )
        goto LABEL_213;
      v16 = *(unsigned __int8 *)(v15 - 24);
      v17 = 0;
      v18 = v15 - 24;
      if ( (unsigned int)(v16 - 30) < 0xB )
        v17 = v18;
    }
    v19 = sub_B46EC0(v17, 0);
    v122 = sub_AA56F0(v19);
    v20 = *(_QWORD *)(v6 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v7 == v20 )
    {
      v22 = 0;
    }
    else
    {
      if ( !v20 )
        goto LABEL_213;
      v21 = *(unsigned __int8 *)(v20 - 24);
      v22 = 0;
      v23 = v20 - 24;
      if ( (unsigned int)(v21 - 30) < 0xB )
        v22 = v23;
    }
    v118 = sub_B46EC0(v22, 0);
    v132 = sub_B46EC0(v22, 1u);
    if ( v118 == v132 || !v119 && sub_AA5650(v122, 3) )
      goto LABEL_10;
    v24 = &v166;
    sub_AA72C0(&v166, v132, 1);
    v25 = _mm_load_si128(&v171);
    v26 = _mm_load_si128(&v172);
    v159 = 0;
    v156 = v25;
    v157 = v26;
    if ( v174 )
    {
      v24 = &v158;
      v174(&v158, &v173, 2);
      v160 = v175;
      v159 = v174;
    }
    v27 = _mm_load_si128(&v166);
    v28 = _mm_load_si128(&v167);
    v154 = 0;
    v151 = v27;
    v152 = v28;
    if ( v169 )
    {
      v24 = &v153;
      v169(&v153, &v168, 2);
      v155 = v170;
      v154 = v169;
    }
    v29 = _mm_load_si128(&v156);
    v30 = _mm_load_si128(&v157);
    v179 = 0;
    v176 = v29;
    v177 = v30;
    if ( v159 )
    {
      v24 = &i;
      v159(&i, &v158, 2);
      v180 = (_QWORD *)v160;
      v179 = v159;
    }
    v31 = _mm_load_si128(&v151);
    v32 = _mm_load_si128(&v152);
    v164 = 0;
    v33 = v154;
    v161 = v31;
    v162 = v32;
    if ( v154 )
    {
      v24 = &v163;
      v154(&v163, &v153, 2);
      v34 = v161.m128i_i64[0];
      v165 = v155;
      v33 = v154;
      v164 = v154;
      if ( v161.m128i_i64[0] == v176.m128i_i64[0] )
      {
        v127 = 0;
      }
      else
      {
LABEL_36:
        v35 = 0;
        do
        {
          v34 = *(_QWORD *)(v34 + 8);
          v161.m128i_i16[4] = 0;
          v161.m128i_i64[0] = v34;
          if ( v34 != v162.m128i_i64[0] )
          {
            while ( 1 )
            {
              v36 = v34 - 24;
              if ( v34 )
                v34 -= 24;
              if ( !v33 )
                sub_4263D6(v24, v34, v36);
              v24 = &v163;
              if ( v165(&v163, v34) )
                break;
              v34 = *(_QWORD *)(v161.m128i_i64[0] + 8);
              v161.m128i_i16[4] = 0;
              v33 = v164;
              v161.m128i_i64[0] = v34;
              if ( v162.m128i_i64[0] == v34 )
                goto LABEL_45;
            }
            v34 = v161.m128i_i64[0];
            v33 = v164;
          }
LABEL_45:
          ++v35;
        }
        while ( v176.m128i_i64[0] != v34 );
        v127 = v35;
      }
      if ( v33 )
        v33(&v163, &v163, 3);
      goto LABEL_49;
    }
    v34 = v161.m128i_i64[0];
    if ( v161.m128i_i64[0] != v176.m128i_i64[0] )
      goto LABEL_36;
    v127 = 0;
LABEL_49:
    if ( v179 )
      v179(&i, &i, 3);
    if ( v154 )
      v154(&v153, &v153, 3);
    if ( v159 )
      v159(&v158, &v158, 3);
    v128 = 0;
    v37 = (_QWORD *)(*(_QWORD *)(v118 + 48) & 0xFFFFFFFFFFFFFFF8LL);
    v124 = v122;
    if ( v37 != (_QWORD *)(v118 + 48) )
    {
      while ( 1 )
      {
        if ( !v37 )
          goto LABEL_213;
        v133 = *v37 & 0xFFFFFFFFFFFFFFF8LL;
        if ( *((_BYTE *)v37 - 24) != 62 )
          goto LABEL_98;
        v150 = (unsigned __int8 *)(v37 - 3);
        if ( sub_B46500((unsigned __int8 *)v37 - 24) || (*((_BYTE *)v37 - 22) & 1) != 0 )
          goto LABEL_98;
        ++v128;
        if ( v127 * v128 > 249 )
          goto LABEL_93;
        v125 = v37[2];
        v38 = (_QWORD *)(*(_QWORD *)(v132 + 48) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (_QWORD *)(v132 + 48) == v38 )
        {
LABEL_98:
          v37 = (_QWORD *)v133;
          goto LABEL_99;
        }
        while ( 1 )
        {
          if ( !v38 )
            goto LABEL_213;
          if ( *((_BYTE *)v38 - 24) == 62 )
          {
            sub_D66630(&v156, (__int64)v150);
            sub_D66630(&v161, (__int64)(v38 - 3));
            if ( (unsigned __int8)sub_CF4E00((__int64)v149, (__int64)&v156, (__int64)&v161) == 3 )
            {
              v39 = v38[1];
              v145 = *(_QWORD *)(v132 + 48);
              v40 = 0;
              if ( (v145 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                v40 = (v145 & 0xFFFFFFFFFFFFFFF8LL) - 24;
              v139 = v40;
              if ( v39 == v38[2] + 48LL || !v39 )
                v136 = 0;
              else
                v136 = v39 - 24;
              v41 = _mm_load_si128(&v162);
              v42 = _mm_load_si128(&v163);
              v43 = v136 + 24;
              v146 = v40 + 24;
              v176 = _mm_load_si128(&v161);
              v177 = v41;
              for ( i = v42; v146 != v43; v43 = *(_QWORD *)(v43 + 8) )
              {
                v44 = (unsigned __int8 *)(v43 - 24);
                if ( !v43 )
                  v44 = 0;
                if ( (unsigned __int8)sub_B46790(v44, 0) )
                  goto LABEL_63;
              }
              if ( !(unsigned __int8)sub_CF66C0(v149, v136, v139, &v176, 3u) )
              {
                v45 = 0;
                if ( (*(_QWORD *)(v125 + 48) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                  v45 = (*(_QWORD *)(v125 + 48) & 0xFFFFFFFFFFFFFFF8LL) - 24;
                v140 = v45;
                v46 = v37[1];
                if ( v46 == v37[2] + 48LL || !v46 )
                  v137 = 0;
                else
                  v137 = v46 - 24;
                v47 = _mm_load_si128(&v157);
                v48 = _mm_load_si128(&v158);
                v49 = v137 + 24;
                v147 = v45 + 24;
                v176 = _mm_load_si128(&v156);
                v177 = v47;
                for ( i = v48; v147 != v49; v49 = *(_QWORD *)(v49 + 8) )
                {
                  v50 = (unsigned __int8 *)(v49 - 24);
                  if ( !v49 )
                    v50 = 0;
                  if ( (unsigned __int8)sub_B46790(v50, 0) )
                    goto LABEL_63;
                }
                if ( !(unsigned __int8)sub_CF66C0(v149, v137, v140, &v176, 3u) )
                {
                  if ( (unsigned __int8)sub_B45D20((__int64)v150, (__int64)(v38 - 3), 0, 0, v51) )
                  {
                    v52 = sub_B43CC0((__int64)v150);
                    v53 = sub_B50C50(*(_QWORD *)(*(v37 - 11) + 8LL), *(_QWORD *)(*(v38 - 11) + 8LL), v52);
                    if ( v53 )
                      break;
                  }
                }
              }
            }
          }
LABEL_63:
          v38 = (_QWORD *)(*v38 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (_QWORD *)(v132 + 48) == v38 )
            goto LABEL_98;
        }
        v148 = v38 - 3;
        v126 = v53;
        v54 = (_BYTE *)*(v37 - 7);
        v138 = (_QWORD *)*(v38 - 7);
        v141 = (__int64)v54;
        if ( v54 != (_BYTE *)v138 )
        {
          if ( *v54 != 63 )
            goto LABEL_93;
          if ( *(_BYTE *)v138 != 63 )
            goto LABEL_93;
          if ( !sub_B46220((__int64)v54, (__int64)v138) )
            goto LABEL_93;
          v55 = *(_QWORD *)(v141 + 16);
          if ( !v55 )
            goto LABEL_93;
          if ( *(_QWORD *)(v55 + 8) )
            goto LABEL_93;
          if ( *(_QWORD *)(v141 + 40) != v37[2] )
            goto LABEL_93;
          v56 = v138[2];
          if ( !v56 || *(_QWORD *)(v56 + 8) || v138[5] != v38[2] )
            goto LABEL_93;
        }
        if ( v122 != v124 )
          goto LABEL_111;
        if ( sub_AA5650(v122, 3) )
        {
          v176.m128i_i64[0] = v118;
          v176.m128i_i64[1] = v132;
          v124 = sub_F41DE0(v122, (__int64 **)&v176, 2, ".sink.split", 0, 0, 0, 0);
          if ( !v124 )
          {
LABEL_93:
            v120 |= v123;
            break;
          }
        }
        v141 = *(v37 - 7);
        v138 = (_QWORD *)*(v38 - 7);
LABEL_111:
        v58 = sub_AA5190(v124);
        if ( v58 )
        {
          LOBYTE(v59) = v57;
          v60 = HIBYTE(v57);
        }
        else
        {
          v60 = 0;
          LOBYTE(v59) = 0;
        }
        v59 = (unsigned __int8)v59;
        BYTE1(v59) = v60;
        v61 = v59;
        sub_B45560(v150, (unsigned __int64)v148);
        sub_F57030(v150, (__int64)v148, 1);
        v142 = sub_B10CD0((__int64)(v38 + 3));
        v62 = sub_B10CD0((__int64)(v37 + 3));
        sub_AE8F10((__int64)v150, v62, v142);
        v176.m128i_i64[0] = (__int64)(v38 - 3);
        sub_AE9860((__int64)v150, (__int64)&v176, 1);
        v182 = sub_BD5C60((__int64)v150);
        v183[0] = &v189;
        v183[1] = &v190;
        v176.m128i_i64[0] = (__int64)&v177;
        v185 = 512;
        v189 = &unk_49DA100;
        v176.m128i_i64[1] = 0x200000000LL;
        v179 = 0;
        v180 = 0;
        v183[2] = 0;
        v184 = 0;
        v186 = 7;
        v187 = 0;
        v188 = 0;
        LOWORD(v181) = 0;
        v190 = &unk_49DA0B0;
        v63 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v37[2];
        v180 = v37;
        v179 = v63;
        v64 = *(_QWORD *)sub_B46C60((__int64)v150);
        v161.m128i_i64[0] = v64;
        if ( !v64 || (sub_B96E90((__int64)&v161, v64, 1), (v65 = v161.m128i_i64[0]) == 0) )
        {
          sub_93FB40((__int64)&v176, 0);
          v65 = v161.m128i_i64[0];
          goto LABEL_183;
        }
        v66 = v176.m128i_i64[0];
        v67 = v176.m128i_i32[2];
        v68 = (unsigned __int64 *)(v176.m128i_i64[0] + 16LL * v176.m128i_u32[2]);
        if ( (unsigned __int64 *)v176.m128i_i64[0] != v68 )
        {
          while ( *(_DWORD *)v66 )
          {
            v66 += 16;
            if ( v68 == (unsigned __int64 *)v66 )
              goto LABEL_179;
          }
          *(_QWORD *)(v66 + 8) = v161.m128i_i64[0];
          goto LABEL_120;
        }
LABEL_179:
        if ( v176.m128i_u32[2] >= (unsigned __int64)v176.m128i_u32[3] )
        {
          v110 = v176.m128i_u32[2] + 1LL;
          v111 = v112 & 0xFFFFFFFF00000000LL;
          v112 &= 0xFFFFFFFF00000000LL;
          if ( v176.m128i_u32[3] < v110 )
          {
            v135 = v111;
            v144 = v161.m128i_i64[0];
            sub_C8D5F0((__int64)&v176, &v177, v176.m128i_u32[2] + 1LL, 0x10u, v110, v161.m128i_i64[0]);
            v111 = v135;
            v65 = v144;
            v68 = (unsigned __int64 *)(v176.m128i_i64[0] + 16LL * v176.m128i_u32[2]);
          }
          *v68 = v111;
          v68[1] = v65;
          v65 = v161.m128i_i64[0];
          ++v176.m128i_i32[2];
        }
        else
        {
          if ( v68 )
          {
            *(_DWORD *)v68 = 0;
            v68[1] = v65;
            v67 = v176.m128i_i32[2];
            v65 = v161.m128i_i64[0];
          }
          v176.m128i_i32[2] = v67 + 1;
        }
LABEL_183:
        if ( v65 )
LABEL_120:
          sub_B91220((__int64)&v161, v65);
        v163.m128i_i16[0] = 257;
        v69 = *(v37 - 11);
        v70 = *(_QWORD *)(*(v38 - 11) + 8LL);
        v71 = *(_QWORD *)(v69 + 8);
        if ( v70 == v71 )
        {
          if ( *(v37 - 11) )
          {
            v75 = *(v37 - 10);
            *(_QWORD *)*(v37 - 9) = v75;
            if ( v75 )
              goto LABEL_133;
            *(v37 - 11) = v69;
          }
LABEL_135:
          v76 = *(_QWORD *)(v69 + 16);
          *(v37 - 10) = v76;
          if ( v76 )
            *(_QWORD *)(v76 + 16) = v37 - 10;
          *(v37 - 9) = v69 + 16;
          *(_QWORD *)(v69 + 16) = v37 - 11;
        }
        else
        {
          v72 = *(unsigned __int8 *)(v71 + 8);
          v73 = *(_BYTE *)(v71 + 8);
          if ( (unsigned int)(v72 - 17) > 1 )
          {
            if ( (_BYTE)v72 == 14 )
              goto LABEL_186;
          }
          else
          {
            if ( *(_BYTE *)(**(_QWORD **)(v71 + 16) + 8LL) != 14 )
              goto LABEL_124;
LABEL_186:
            v109 = *(unsigned __int8 *)(v70 + 8);
            if ( (unsigned int)(v109 - 17) <= 1 )
              LOBYTE(v109) = *(_BYTE *)(**(_QWORD **)(v70 + 16) + 8LL);
            if ( (_BYTE)v109 == 12 )
            {
              v69 = sub_28BF970(v176.m128i_i64, 0x2Fu, v69, (__int64 **)v70, (__int64)&v161, 0, v156.m128i_i32[0], 0);
              goto LABEL_131;
            }
LABEL_124:
            if ( v72 == 18 )
            {
LABEL_125:
              v73 = *(_BYTE *)(**(_QWORD **)(v71 + 16) + 8LL);
              goto LABEL_126;
            }
          }
          if ( v72 == 17 )
            goto LABEL_125;
LABEL_126:
          if ( v73 != 12 )
            goto LABEL_130;
          v74 = *(unsigned __int8 *)(v70 + 8);
          if ( (unsigned int)(v74 - 17) <= 1 )
            LOBYTE(v74) = *(_BYTE *)(**(_QWORD **)(v70 + 16) + 8LL);
          if ( (_BYTE)v74 == 14 )
            v69 = sub_28BF970(v176.m128i_i64, 0x30u, v69, (__int64 **)v70, (__int64)&v161, 0, v156.m128i_i32[0], 0);
          else
LABEL_130:
            v69 = sub_28BF970(v176.m128i_i64, 0x31u, v69, (__int64 **)v70, (__int64)&v161, 0, v156.m128i_i32[0], 0);
LABEL_131:
          if ( *(v37 - 11) )
          {
            v75 = *(v37 - 10);
            *(_QWORD *)*(v37 - 9) = v75;
            if ( v75 )
LABEL_133:
              *(_QWORD *)(v75 + 16) = *(v37 - 9);
          }
          *(v37 - 11) = v69;
          if ( v69 )
            goto LABEL_135;
        }
        v143 = (_QWORD *)sub_B47F80(v150);
        sub_B44220(v143, v58, v61);
        v77 = *(v37 - 11);
        v78 = *(v38 - 11);
        v134 = v77;
        if ( v77 != v78 )
        {
          v79.m128i_i64[0] = (__int64)sub_BD5D20(v78);
          v161 = v79;
          v163.m128i_i16[0] = 773;
          v162.m128i_i64[0] = (__int64)".sink";
          v129 = *(_QWORD *)(v77 + 8);
          v80 = sub_BD2DA0(80);
          v81 = v129;
          v82 = v80;
          if ( v80 )
          {
            v130 = (_QWORD *)v80;
            sub_B44260(v80, v81, 55, 0x8000000u, 0, 0);
            *(_DWORD *)(v82 + 72) = 2;
            sub_BD6B50((unsigned __int8 *)v82, (const char **)&v161);
            sub_BD2A10(v82, *(_DWORD *)(v82 + 72), 1);
            v83 = v130;
          }
          else
          {
            v83 = 0;
          }
          v131 = (__int64)v83;
          v84 = v121;
          LOWORD(v84) = 1;
          v121 = v84;
          sub_B44220(v83, *(_QWORD *)(v124 + 56), v84);
          v85 = sub_B10CD0((__int64)(v38 + 3));
          v86 = sub_B10CD0((__int64)(v37 + 3));
          sub_AE8F10(v131, v86, v85);
          v87 = v37[2];
          v88 = *(_DWORD *)(v82 + 4) & 0x7FFFFFF;
          if ( v88 == *(_DWORD *)(v82 + 72) )
          {
            sub_B48D90(v82);
            v88 = *(_DWORD *)(v82 + 4) & 0x7FFFFFF;
          }
          v89 = (v88 + 1) & 0x7FFFFFF;
          v90 = v89 | *(_DWORD *)(v82 + 4) & 0xF8000000;
          v91 = *(_QWORD *)(v82 - 8) + 32LL * (unsigned int)(v89 - 1);
          *(_DWORD *)(v82 + 4) = v90;
          if ( *(_QWORD *)v91 )
          {
            v92 = *(_QWORD *)(v91 + 8);
            **(_QWORD **)(v91 + 16) = v92;
            if ( v92 )
              *(_QWORD *)(v92 + 16) = *(_QWORD *)(v91 + 16);
          }
          *(_QWORD *)v91 = v134;
          v93 = *(_QWORD *)(v134 + 16);
          *(_QWORD *)(v91 + 8) = v93;
          if ( v93 )
            *(_QWORD *)(v93 + 16) = v91 + 8;
          *(_QWORD *)(v91 + 16) = v134 + 16;
          *(_QWORD *)(v134 + 16) = v91;
          *(_QWORD *)(*(_QWORD *)(v82 - 8)
                    + 32LL * *(unsigned int *)(v82 + 72)
                    + 8LL * ((*(_DWORD *)(v82 + 4) & 0x7FFFFFFu) - 1)) = v87;
          v94 = v38[2];
          v95 = *(_DWORD *)(v82 + 4) & 0x7FFFFFF;
          if ( v95 == *(_DWORD *)(v82 + 72) )
          {
            sub_B48D90(v82);
            v95 = *(_DWORD *)(v82 + 4) & 0x7FFFFFF;
          }
          v96 = (v95 + 1) & 0x7FFFFFF;
          v97 = v96 | *(_DWORD *)(v82 + 4) & 0xF8000000;
          v98 = *(_QWORD *)(v82 - 8) + 32LL * (unsigned int)(v96 - 1);
          *(_DWORD *)(v82 + 4) = v97;
          if ( *(_QWORD *)v98 )
          {
            v99 = *(_QWORD *)(v98 + 8);
            **(_QWORD **)(v98 + 16) = v99;
            if ( v99 )
              *(_QWORD *)(v99 + 16) = *(_QWORD *)(v98 + 16);
          }
          *(_QWORD *)v98 = v78;
          if ( v78 )
          {
            v100 = *(_QWORD *)(v78 + 16);
            *(_QWORD *)(v98 + 8) = v100;
            if ( v100 )
              *(_QWORD *)(v100 + 16) = v98 + 8;
            *(_QWORD *)(v98 + 16) = v78 + 16;
            *(_QWORD *)(v78 + 16) = v98;
          }
          *(_QWORD *)(*(_QWORD *)(v82 - 8)
                    + 32LL * *(unsigned int *)(v82 + 72)
                    + 8LL * ((*(_DWORD *)(v82 + 4) & 0x7FFFFFFu) - 1)) = v94;
          if ( *(v143 - 8) )
          {
            v101 = *(v143 - 7);
            *(_QWORD *)*(v143 - 6) = v101;
            if ( v101 )
              *(_QWORD *)(v101 + 16) = *(v143 - 6);
          }
          *(v143 - 8) = v82;
          v102 = *(_QWORD *)(v82 + 16);
          *(v143 - 7) = v102;
          if ( v102 )
            *(_QWORD *)(v102 + 16) = v143 - 7;
          *(v143 - 6) = v82 + 16;
          *(_QWORD *)(v82 + 16) = v143 - 8;
        }
        sub_B43D60(v150);
        sub_B43D60(v148);
        if ( v138 != (_QWORD *)v141 )
        {
          v103 = (_QWORD *)sub_B47F80((_BYTE *)v141);
          v104 = v116;
          LOWORD(v104) = 0;
          v116 = v104;
          sub_B44220(v103, (__int64)(v143 + 3), v104);
          v105 = sub_B10CD0((__int64)(v138 + 6));
          v106 = sub_B10CD0(v141 + 48);
          sub_AE8F10((__int64)v103, v106, v105);
          if ( *(v143 - 4) )
          {
            v107 = *(v143 - 3);
            *(_QWORD *)*(v143 - 2) = v107;
            if ( v107 )
              *(_QWORD *)(v107 + 16) = *(v143 - 2);
          }
          *(v143 - 4) = v103;
          if ( v103 )
          {
            v108 = v103[2];
            *(v143 - 3) = v108;
            if ( v108 )
              *(_QWORD *)(v108 + 16) = v143 - 3;
            *(v143 - 2) = v103 + 2;
            v103[2] = v143 - 4;
          }
          sub_BD84D0(v141, (__int64)v103);
          sub_B43D60((_QWORD *)v141);
          sub_BD84D0((__int64)v138, (__int64)v103);
          sub_B43D60(v138);
        }
        nullsub_61();
        v189 = &unk_49DA100;
        nullsub_63();
        if ( (__m128i *)v176.m128i_i64[0] != &v177 )
          _libc_free(v176.m128i_u64[0]);
        v37 = (_QWORD *)(*(_QWORD *)(v118 + 48) & 0xFFFFFFFFFFFFFFF8LL);
        v123 = v126;
LABEL_99:
        if ( (_QWORD *)(v118 + 48) == v37 )
          goto LABEL_93;
      }
    }
    if ( v174 )
      v174(&v173, &v173, 3);
    if ( v169 )
      v169(&v168, &v168, 3);
LABEL_10:
    v6 = v117;
  }
  while ( v115 != v117 );
  if ( !v120 )
  {
LABEL_207:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v176.m128i_i64[0] = 0;
  v176.m128i_i64[1] = (__int64)&i;
  v13 = *a2 == 0;
  v177.m128i_i8[12] = 1;
  v177.m128i_i64[0] = 2;
  v177.m128i_i32[2] = 0;
  v179 = 0;
  v180 = v183;
  v181 = 2;
  LODWORD(v182) = 0;
  BYTE4(v182) = 1;
  if ( v13 )
  {
    v177.m128i_i32[1] = 1;
    v176.m128i_i64[0] = 1;
    i.m128i_i64[0] = (__int64)&unk_4F82408;
  }
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&i, (__int64)&v176);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v183, (__int64)&v179);
  if ( !BYTE4(v182) )
  {
    _libc_free((unsigned __int64)v180);
    if ( v177.m128i_i8[12] )
      return a1;
    goto LABEL_201;
  }
  if ( !v177.m128i_i8[12] )
LABEL_201:
    _libc_free(v176.m128i_u64[1]);
  return a1;
}
