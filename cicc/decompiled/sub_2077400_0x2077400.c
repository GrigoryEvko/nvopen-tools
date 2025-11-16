// Function: sub_2077400
// Address: 0x2077400
//
void __fastcall sub_2077400(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, double a5, __m128i a6, __m128i a7)
{
  __int64 v10; // rdi
  __int64 v11; // r14
  const __m128i *v12; // r15
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v15; // rax
  unsigned int v16; // eax
  unsigned __int8 v17; // r8
  int v18; // edx
  __int64 v19; // rax
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  unsigned int v22; // edx
  __int64 v23; // r9
  __int64 v24; // r15
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 *m128i_i64; // rax
  int v29; // r14d
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r15
  int v33; // r15d
  __int64 v34; // rax
  __int64 v35; // rdx
  int v36; // eax
  int v37; // r14d
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // rsi
  unsigned int v41; // edx
  __int64 v42; // rax
  __int64 *v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rax
  int v46; // r8d
  int v47; // r9d
  __int64 v48; // rdx
  __int64 v49; // r15
  __int64 v50; // r14
  __int64 v51; // rdx
  __int64 *v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r15
  __int64 v55; // rax
  int v56; // r8d
  int v57; // r9d
  const void ***v58; // rcx
  int v59; // edx
  int v60; // r8d
  __int64 v61; // r9
  _QWORD *v62; // rax
  unsigned __int64 v63; // r14
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  int v66; // edx
  __int64 v67; // rax
  __int64 *v68; // r10
  __int64 v69; // r11
  __int64 v70; // rsi
  unsigned int v71; // edx
  int v72; // r8d
  int v73; // r9d
  __int64 v74; // r14
  __int64 v75; // r11
  __int64 v76; // r15
  char v77; // al
  __int64 v78; // rsi
  __int64 v79; // rax
  __int64 v80; // rax
  const void **v81; // rdx
  __int64 *v82; // r15
  __int64 v83; // rcx
  __int64 v84; // rax
  const void **v85; // r8
  __int64 v86; // r11
  bool v87; // zf
  __int64 v88; // rsi
  __int64 v89; // r15
  int v90; // edx
  int v91; // r14d
  __int64 v92; // rax
  unsigned __int8 *v93; // rdx
  unsigned int v94; // edx
  unsigned int v95; // edx
  unsigned int v96; // eax
  __int64 v97; // rax
  unsigned __int8 *v98; // rax
  __int64 v99; // rdi
  int v100; // edx
  __int64 v101; // r9
  __int64 v102; // rax
  __int64 *v103; // r14
  unsigned __int64 v104; // r10
  __int64 v105; // r11
  __int64 v106; // rsi
  __int64 *v107; // rax
  unsigned int v108; // edx
  __int64 v109; // r15
  __int64 v110; // rax
  __int64 *v111; // rax
  __int64 v112; // rax
  unsigned int v113; // eax
  unsigned __int8 v114; // r8
  int v115; // edx
  __int64 v116; // rax
  unsigned __int8 *v117; // rsi
  int v118; // edx
  __int64 *v119; // rax
  __int64 v120; // rdx
  __int64 v121; // rsi
  int v122; // edx
  __int64 v123; // rax
  __int64 *v124; // r9
  unsigned __int64 v125; // r14
  __int64 v126; // r15
  __int64 v127; // rsi
  unsigned int v128; // edx
  __int128 v129; // [rsp-10h] [rbp-290h]
  __int128 v130; // [rsp-10h] [rbp-290h]
  __int128 v131; // [rsp-10h] [rbp-290h]
  __int128 v132; // [rsp-10h] [rbp-290h]
  __int64 v133; // [rsp-8h] [rbp-288h]
  int v134; // [rsp+8h] [rbp-278h]
  unsigned __int64 v135; // [rsp+10h] [rbp-270h]
  __int64 v136; // [rsp+18h] [rbp-268h]
  const void ***v137; // [rsp+20h] [rbp-260h]
  unsigned int v138; // [rsp+28h] [rbp-258h]
  __int64 v139; // [rsp+28h] [rbp-258h]
  unsigned int v141; // [rsp+30h] [rbp-250h]
  int v142; // [rsp+30h] [rbp-250h]
  int v143; // [rsp+30h] [rbp-250h]
  int v144; // [rsp+30h] [rbp-250h]
  __int64 v145; // [rsp+38h] [rbp-248h]
  unsigned int v146; // [rsp+40h] [rbp-240h]
  __int64 v147; // [rsp+40h] [rbp-240h]
  unsigned __int8 v148; // [rsp+48h] [rbp-238h]
  unsigned __int8 v149; // [rsp+48h] [rbp-238h]
  __int64 v150; // [rsp+48h] [rbp-238h]
  char v151; // [rsp+50h] [rbp-230h]
  __int64 v152; // [rsp+50h] [rbp-230h]
  int v153; // [rsp+50h] [rbp-230h]
  __int64 v154; // [rsp+58h] [rbp-228h]
  __int64 v155; // [rsp+60h] [rbp-220h]
  const void **v156; // [rsp+60h] [rbp-220h]
  char v157; // [rsp+70h] [rbp-210h]
  _QWORD *v158; // [rsp+70h] [rbp-210h]
  unsigned int v159; // [rsp+70h] [rbp-210h]
  const void ***v160; // [rsp+70h] [rbp-210h]
  const void ***v161; // [rsp+70h] [rbp-210h]
  const void ***v162; // [rsp+70h] [rbp-210h]
  const void ***v163; // [rsp+70h] [rbp-210h]
  char v164; // [rsp+80h] [rbp-200h]
  unsigned __int64 v165; // [rsp+80h] [rbp-200h]
  __int64 v166; // [rsp+80h] [rbp-200h]
  unsigned __int64 v167; // [rsp+80h] [rbp-200h]
  __int64 *v168; // [rsp+80h] [rbp-200h]
  __int64 v169; // [rsp+88h] [rbp-1F8h]
  __int64 v170; // [rsp+88h] [rbp-1F8h]
  __int64 v171; // [rsp+88h] [rbp-1F8h]
  __int64 v172; // [rsp+88h] [rbp-1F8h]
  __int64 v173; // [rsp+88h] [rbp-1F8h]
  __int64 v174; // [rsp+88h] [rbp-1F8h]
  __int64 v175; // [rsp+F0h] [rbp-190h] BYREF
  int v176; // [rsp+F8h] [rbp-188h]
  unsigned __int64 v177; // [rsp+100h] [rbp-180h]
  unsigned __int64 v178; // [rsp+108h] [rbp-178h]
  __int64 v179; // [rsp+110h] [rbp-170h]
  __int64 v180; // [rsp+120h] [rbp-160h] BYREF
  __int64 v181; // [rsp+128h] [rbp-158h]
  __int64 v182; // [rsp+130h] [rbp-150h]
  __int128 v183; // [rsp+140h] [rbp-140h] BYREF
  __int128 v184; // [rsp+150h] [rbp-130h]
  __int128 v185; // [rsp+160h] [rbp-120h]
  unsigned __int8 *v186; // [rsp+170h] [rbp-110h] BYREF
  __int64 v187; // [rsp+178h] [rbp-108h]
  _BYTE v188[64]; // [rsp+180h] [rbp-100h] BYREF
  __m128i *v189; // [rsp+1C0h] [rbp-C0h] BYREF
  __int64 v190; // [rsp+1C8h] [rbp-B8h]
  __m128i v191; // [rsp+1D0h] [rbp-B0h] BYREF

  v10 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v10 + 16) )
    v10 = 0;
  v11 = v10 + 112;
  v164 = sub_1560180(v10 + 112, 36) ^ 1;
  if ( v164 )
  {
    if ( (unsigned __int8)sub_1560180(v11, 36) || (v151 = sub_1560180(v11, 37)) != 0 )
    {
      v12 = *(const __m128i **)(a1 + 552);
      v189 = &v191;
      a6 = _mm_loadu_si128(v12 + 11);
      v190 = 0x800000001LL;
      v191 = a6;
      v151 = v164;
    }
    else
    {
      v189 = &v191;
      v190 = 0x800000000LL;
      v191.m128i_i64[0] = (__int64)sub_2051C20((__int64 *)a1, a5, *(double *)a6.m128i_i64, a7);
      v191.m128i_i64[1] = v120;
      v12 = *(const __m128i **)(a1 + 552);
      LODWORD(v190) = 1;
    }
  }
  else
  {
    v151 = 0;
    v12 = *(const __m128i **)(a1 + 552);
    v189 = &v191;
    v190 = 0x800000000LL;
  }
  v157 = 0;
  v185 = 0;
  v183 = 0;
  DWORD2(v185) = 1;
  v184 = 0;
  v13 = v12[2].m128i_i64[0];
  v155 = v12[1].m128i_i64[0];
  v14 = *(__int64 (**)())(*(_QWORD *)v155 + 320LL);
  if ( v14 != sub_1F3CA60 )
  {
    v157 = ((__int64 (__fastcall *)(__int64, __int128 *, __int64, __int64, _QWORD))v14)(v155, &v183, a2, v13, a3);
    if ( v157 && (unsigned int)(v183 - 44) > 1 )
      goto LABEL_20;
    v12 = *(const __m128i **)(a1 + 552);
    v13 = v12[2].m128i_i64[0];
  }
  if ( a3 == 5232 )
  {
    v112 = sub_1E0A0C0(v13);
    v113 = 8 * sub_15A9520(v112, 0);
    if ( v113 == 32 )
    {
      v114 = 5;
    }
    else if ( v113 > 0x20 )
    {
      v114 = 6;
      if ( v113 != 64 )
      {
        v114 = 0;
        if ( v113 == 128 )
          v114 = 7;
      }
    }
    else
    {
      v114 = 3;
      if ( v113 != 8 )
        v114 = 4 * (v113 == 16);
    }
    v115 = *(_DWORD *)(a1 + 536);
    v116 = *(_QWORD *)a1;
    v186 = 0;
    LODWORD(v187) = v115;
    if ( v116 )
    {
      if ( &v186 != (unsigned __int8 **)(v116 + 48) )
      {
        v117 = *(unsigned __int8 **)(v116 + 48);
        v186 = v117;
        if ( v117 )
        {
          v149 = v114;
          sub_1623A60((__int64)&v186, (__int64)v117, 2);
          v114 = v149;
        }
      }
    }
    v21 = sub_1D38BB0((__int64)v12, 5233, (__int64)&v186, v114, 0, 1, (__m128i)0LL, *(double *)a6.m128i_i64, a7, 0);
  }
  else
  {
    v15 = sub_1E0A0C0(v13);
    v16 = 8 * sub_15A9520(v15, 0);
    if ( v16 == 32 )
    {
      v17 = 5;
    }
    else if ( v16 > 0x20 )
    {
      v17 = 6;
      if ( v16 != 64 )
      {
        v17 = 0;
        if ( v16 == 128 )
          v17 = 7;
      }
    }
    else
    {
      v17 = 3;
      if ( v16 != 8 )
        v17 = 4 * (v16 == 16);
    }
    v18 = *(_DWORD *)(a1 + 536);
    v19 = *(_QWORD *)a1;
    v186 = 0;
    LODWORD(v187) = v18;
    if ( v19 )
    {
      if ( &v186 != (unsigned __int8 **)(v19 + 48) )
      {
        v20 = *(unsigned __int8 **)(v19 + 48);
        v186 = v20;
        if ( v20 )
        {
          v148 = v17;
          sub_1623A60((__int64)&v186, (__int64)v20, 2);
          v17 = v148;
        }
      }
    }
    v21 = sub_1D38BB0((__int64)v12, a3, (__int64)&v186, v17, 0, 1, (__m128i)0LL, *(double *)a6.m128i_i64, a7, 0);
  }
  v24 = v21;
  v25 = v22;
  v26 = (unsigned int)v190;
  v27 = v133;
  if ( (unsigned int)v190 >= HIDWORD(v190) )
  {
    v150 = v22;
    sub_16CD150((__int64)&v189, &v191, 0, 16, v22, v23);
    v26 = (unsigned int)v190;
    v25 = v150;
  }
  m128i_i64 = v189[v26].m128i_i64;
  *m128i_i64 = v24;
  m128i_i64[1] = v25;
  LODWORD(v190) = v190 + 1;
  if ( v186 )
    sub_161E7C0((__int64)&v186, (__int64)v186);
LABEL_20:
  v29 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_31;
  v30 = sub_1648A40(a2);
  v32 = v30 + v31;
  if ( *(char *)(a2 + 23) >= 0 )
  {
    if ( (unsigned int)(v32 >> 4) )
LABEL_143:
      BUG();
LABEL_31:
    v36 = 0;
    goto LABEL_32;
  }
  if ( !(unsigned int)((v32 - sub_1648A40(a2)) >> 4) )
    goto LABEL_31;
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_143;
  v33 = *(_DWORD *)(sub_1648A40(a2) + 8);
  if ( *(char *)(a2 + 23) >= 0 )
    BUG();
  v34 = sub_1648A40(a2);
  v36 = *(_DWORD *)(v34 + v35 - 4) - v33;
LABEL_32:
  v37 = v29 - 1 - v36;
  if ( v37 )
  {
    v38 = 0;
    while ( 1 )
    {
      v39 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      if ( a3 == 5232 )
        break;
      v40 = *(_QWORD *)(a2 + 24 * (v38 - v39));
      if ( a3 != 3956 )
      {
        if ( a3 == 3760 && (_DWORD)v38 == 1 )
          goto LABEL_38;
        goto LABEL_75;
      }
      if ( v38 == 1 )
      {
        if ( *(_BYTE *)(v40 + 16) != 19 )
          BUG();
LABEL_38:
        v25 = (__int64)sub_1D2AF60(*(_QWORD **)(a1 + 552), *(_QWORD *)(v40 + 24), v38 - v39, v27, v25, v23);
        v23 = v41;
LABEL_39:
        v42 = (unsigned int)v190;
        if ( (unsigned int)v190 >= HIDWORD(v190) )
          goto LABEL_78;
        goto LABEL_40;
      }
      v25 = (__int64)sub_20685E0(a1, (__int64 *)v40, (__m128i)0LL, a6, a7);
      v42 = (unsigned int)v190;
      v23 = v95;
      if ( (unsigned int)v190 >= HIDWORD(v190) )
      {
LABEL_78:
        v139 = v23;
        v147 = v25;
        sub_16CD150((__int64)&v189, &v191, 0, 16, v25, v23);
        v42 = (unsigned int)v190;
        v23 = v139;
        v25 = v147;
      }
LABEL_40:
      ++v38;
      v43 = v189[v42].m128i_i64;
      *v43 = v25;
      v43[1] = v23;
      LODWORD(v190) = v190 + 1;
      if ( v38 == v37 )
        goto LABEL_41;
    }
    if ( v38 == 1 )
      goto LABEL_45;
    v92 = *(_QWORD *)(a2 - 24 * v39);
    if ( *(_BYTE *)(v92 + 16) != 19
      || (v93 = *(unsigned __int8 **)(v92 + 24), (unsigned int)*v93 - 1 > 1)
      || (v40 = *((_QWORD *)v93 + 17), *(_BYTE *)(v40 + 16) != 3) )
    {
      sub_16BD130("nvvm_texsurf_handle op0 must be metadata wrapping a GlobalVariable", 1u);
    }
LABEL_75:
    v25 = (__int64)sub_20685E0(a1, (__int64 *)v40, (__m128i)0LL, a6, a7);
    v23 = v94;
    goto LABEL_39;
  }
LABEL_41:
  v44 = *(_QWORD *)(a1 + 552);
  if ( a3 == 3785 )
  {
    v45 = sub_1D38BB0(v44, *(unsigned int *)(v44 + 352), a4, 5, 0, 0, (__m128i)0LL, *(double *)a6.m128i_i64, a7, 0);
    v49 = v48;
    v50 = v45;
    v51 = (unsigned int)v190;
    if ( (unsigned int)v190 >= HIDWORD(v190) )
    {
      sub_16CD150((__int64)&v189, &v191, 0, 16, v46, v47);
      v51 = (unsigned int)v190;
    }
    v52 = v189[v51].m128i_i64;
    *v52 = v50;
    v52[1] = v49;
    v53 = *(_QWORD *)(a1 + 552);
    LODWORD(v190) = v190 + 1;
    ++*(_DWORD *)(v53 + 352);
    *(_BYTE *)(*(_QWORD *)(a1 + 552) + 356LL) = 1;
LABEL_45:
    v44 = *(_QWORD *)(a1 + 552);
  }
  v54 = *(_QWORD *)a2;
  v186 = v188;
  v187 = 0x400000000LL;
  v55 = sub_1E0A0C0(*(_QWORD *)(v44 + 32));
  sub_20C7CE0(v155, v55, v54, &v186, 0, 0);
  if ( v164 )
  {
    v97 = (unsigned int)v187;
    if ( (unsigned int)v187 >= HIDWORD(v187) )
    {
      sub_16CD150((__int64)&v186, v188, 0, 16, v56, v57);
      v97 = (unsigned int)v187;
    }
    v98 = &v186[16 * v97];
    *(_QWORD *)v98 = 1;
    *((_QWORD *)v98 + 1) = 0;
    v99 = *(_QWORD *)(a1 + 552);
    LODWORD(v187) = v187 + 1;
    v58 = (const void ***)sub_1D25C30(v99, v186, (unsigned int)v187);
    v60 = v100;
    if ( !v157 )
    {
      v101 = *(_QWORD *)a2;
      v87 = *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 0;
      v102 = *(_QWORD *)a1;
      LODWORD(v181) = *(_DWORD *)(a1 + 536);
      v103 = *(__int64 **)(a1 + 552);
      v104 = (unsigned __int64)v189;
      v105 = (unsigned int)v190;
      v180 = 0;
      if ( v87 )
      {
        if ( v102 )
        {
          if ( &v180 != (__int64 *)(v102 + 48) )
          {
            v121 = *(_QWORD *)(v102 + 48);
            v180 = v121;
            if ( v121 )
            {
              v167 = (unsigned __int64)v189;
              v144 = v100;
              v162 = v58;
              v172 = (unsigned int)v190;
              sub_1623A60((__int64)&v180, v121, 2);
              v60 = v144;
              v58 = v162;
              v104 = v167;
              v105 = v172;
            }
          }
        }
        *((_QWORD *)&v131 + 1) = v105;
        *(_QWORD *)&v131 = v104;
        v107 = sub_1D36D80(v103, 45, (__int64)&v180, v58, v60, 0.0, *(double *)a6.m128i_i64, a7, v101, v131);
      }
      else
      {
        if ( v102 )
        {
          if ( &v180 != (__int64 *)(v102 + 48) )
          {
            v106 = *(_QWORD *)(v102 + 48);
            v180 = v106;
            if ( v106 )
            {
              v165 = (unsigned __int64)v189;
              v143 = v100;
              v161 = v58;
              v170 = (unsigned int)v190;
              sub_1623A60((__int64)&v180, v106, 2);
              v60 = v143;
              v58 = v161;
              v104 = v165;
              v105 = v170;
            }
          }
        }
        *((_QWORD *)&v130 + 1) = v105;
        *(_QWORD *)&v130 = v104;
        v107 = sub_1D36D80(v103, 44, (__int64)&v180, v58, v60, 0.0, *(double *)a6.m128i_i64, a7, v101, v130);
      }
      v74 = (__int64)v107;
      v159 = v108;
      v75 = v108;
      if ( v180 )
      {
        v171 = v108;
        sub_161E7C0((__int64)&v180, v180);
        v75 = v171;
      }
      goto LABEL_97;
    }
  }
  else
  {
    v58 = (const void ***)sub_1D25C30(*(_QWORD *)(a1 + 552), v186, (unsigned int)v187);
    v60 = v59;
    if ( !v157 )
    {
      v122 = *(_DWORD *)(a1 + 536);
      v123 = *(_QWORD *)a1;
      v180 = 0;
      v124 = *(__int64 **)(a1 + 552);
      v125 = (unsigned __int64)v189;
      LODWORD(v181) = v122;
      v126 = (unsigned int)v190;
      if ( v123 )
      {
        if ( &v180 != (__int64 *)(v123 + 48) )
        {
          v127 = *(_QWORD *)(v123 + 48);
          v180 = v127;
          if ( v127 )
          {
            v153 = v60;
            v163 = v58;
            v168 = v124;
            sub_1623A60((__int64)&v180, v127, 2);
            v60 = v153;
            v58 = v163;
            v124 = v168;
          }
        }
      }
      *((_QWORD *)&v132 + 1) = v126;
      *(_QWORD *)&v132 = v125;
      v74 = (__int64)sub_1D36D80(
                       v124,
                       43,
                       (__int64)&v180,
                       v58,
                       v60,
                       0.0,
                       *(double *)a6.m128i_i64,
                       a7,
                       (__int64)v124,
                       v132);
      v159 = v128;
      v75 = v128;
      if ( v180 )
      {
        v173 = v128;
        sub_161E7C0((__int64)&v180, v180);
        v75 = v173;
      }
LABEL_63:
      v76 = *(_QWORD *)a2;
      v77 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
      if ( !v77 )
        goto LABEL_101;
      goto LABEL_64;
    }
  }
  if ( a3 > 0xFDE )
  {
    if ( a3 - 4069 > 2 )
      goto LABEL_50;
  }
  else if ( a3 <= 0xFDB )
  {
LABEL_50:
    v61 = DWORD2(v185);
    goto LABEL_51;
  }
  v142 = v60;
  v160 = v58;
  v96 = sub_1D1FC50(*(_QWORD *)(a1 + 552), v189[2].m128i_i64[0]);
  v61 = DWORD2(v185);
  v58 = v160;
  v60 = v142;
  if ( DWORD2(v185) < v96 )
  {
    DWORD2(v185) = v96;
    v61 = v96;
  }
LABEL_51:
  v62 = *(_QWORD **)(a1 + 552);
  v63 = *((_QWORD *)&v184 + 1);
  v180 = 0;
  v181 = 0;
  v158 = v62;
  v182 = 0;
  v141 = DWORD1(v185);
  LOBYTE(v179) = 0;
  v146 = WORD6(v185);
  HIDWORD(v179) = 0;
  v178 = (int)v185;
  v64 = *((_QWORD *)&v184 + 1) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*((_QWORD *)&v184 + 1) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (BYTE8(v184) & 4) != 0 )
    {
      HIDWORD(v179) = *(_DWORD *)(v64 + 12);
    }
    else
    {
      v65 = *(_QWORD *)v64;
      if ( *(_BYTE *)(v65 + 8) == 16 )
        v65 = **(_QWORD **)(v65 + 16);
      HIDWORD(v179) = *(_DWORD *)(v65 + 8) >> 8;
    }
  }
  v66 = *(_DWORD *)(a1 + 536);
  v67 = *(_QWORD *)a1;
  v175 = 0;
  v68 = (__int64 *)v189;
  v69 = (unsigned int)v190;
  v176 = v66;
  if ( v67 )
  {
    if ( &v175 != (__int64 *)(v67 + 48) )
    {
      v70 = *(_QWORD *)(v67 + 48);
      v175 = v70;
      if ( v70 )
      {
        v134 = v60;
        v137 = v58;
        v135 = (unsigned __int64)v189;
        v136 = (unsigned int)v190;
        v138 = v61;
        sub_1623A60((__int64)&v175, v70, 2);
        v60 = v134;
        v58 = v137;
        v68 = (__int64 *)v135;
        v69 = v136;
        v61 = v138;
      }
    }
  }
  v177 = v63;
  v74 = sub_1D251C0(
          v158,
          (unsigned int)v183,
          (__int64)&v175,
          (__int64)v58,
          v60,
          v61,
          v68,
          v69,
          *((__int64 *)&v183 + 1),
          v184,
          __PAIR128__(v178, v63),
          v179,
          v146,
          v141,
          (__int64)&v180);
  v159 = v71;
  v75 = v71;
  if ( v175 )
  {
    v145 = v71;
    sub_161E7C0((__int64)&v175, v175);
    v75 = v145;
  }
  if ( !v164 )
    goto LABEL_63;
LABEL_97:
  v109 = (unsigned int)(*(_DWORD *)(v74 + 60) - 1);
  if ( !v151 )
  {
    v154 = v75;
    v166 = *(_QWORD *)(a1 + 552);
    nullsub_686();
    *(_QWORD *)(v166 + 176) = v74;
    *(_DWORD *)(v166 + 184) = v109;
    sub_1D23870();
    v75 = v154;
    goto LABEL_63;
  }
  v110 = *(unsigned int *)(a1 + 112);
  if ( (unsigned int)v110 >= *(_DWORD *)(a1 + 116) )
  {
    v174 = v75;
    sub_16CD150(a1 + 104, (const void *)(a1 + 120), 0, 16, v72, v73);
    v110 = *(unsigned int *)(a1 + 112);
    v75 = v174;
  }
  v111 = (__int64 *)(*(_QWORD *)(a1 + 104) + 16 * v110);
  *v111 = v74;
  v111[1] = v109;
  ++*(_DWORD *)(a1 + 112);
  v76 = *(_QWORD *)a2;
  v77 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v77 )
  {
LABEL_64:
    v78 = *(_QWORD *)(a1 + 552);
    if ( v77 == 16 )
    {
      v169 = v75;
      v79 = sub_1E0A0C0(*(_QWORD *)(v78 + 32));
      LOBYTE(v80) = sub_204D4D0(v155, v79, v76);
      v82 = *(__int64 **)(a1 + 552);
      v180 = 0;
      v83 = v80;
      v84 = *(_QWORD *)a1;
      v85 = v81;
      v86 = v169;
      v87 = *(_QWORD *)a1 == 0;
      LODWORD(v181) = *(_DWORD *)(a1 + 536);
      if ( !v87 && &v180 != (__int64 *)(v84 + 48) )
      {
        v88 = *(_QWORD *)(v84 + 48);
        v180 = v88;
        if ( v88 )
        {
          v152 = v83;
          v156 = v81;
          sub_1623A60((__int64)&v180, v88, 2);
          v83 = v152;
          v85 = v156;
          v86 = v169;
        }
      }
      *((_QWORD *)&v129 + 1) = v159 | v86 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v129 = v74;
      v89 = sub_1D309E0(
              v82,
              158,
              (__int64)&v180,
              v83,
              v85,
              0,
              0.0,
              *(double *)a6.m128i_i64,
              *(double *)a7.m128i_i64,
              v129);
      v91 = v90;
      if ( v180 )
        sub_161E7C0((__int64)&v180, v180);
    }
    else
    {
      v89 = (__int64)sub_2055040(a1, v78, a2, v74, v75 & 0xFFFFFFFF00000000LL | v159, 0.0, *(double *)a6.m128i_i64, a7);
      v91 = v118;
    }
    v180 = a2;
    v119 = sub_205F5C0(a1 + 8, &v180);
    v119[1] = v89;
    *((_DWORD *)v119 + 4) = v91;
  }
LABEL_101:
  if ( v186 != v188 )
    _libc_free((unsigned __int64)v186);
  if ( v189 != &v191 )
    _libc_free((unsigned __int64)v189);
}
