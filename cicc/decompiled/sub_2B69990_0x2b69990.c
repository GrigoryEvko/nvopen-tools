// Function: sub_2B69990
// Address: 0x2b69990
//
__int64 __fastcall sub_2B69990(
        _QWORD *a1,
        __int64 a2,
        char a3,
        unsigned int *a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7,
        _DWORD *a8,
        _BYTE *a9,
        char a10)
{
  __int64 v11; // r12
  __int64 v12; // rdx
  _BYTE **v13; // rsi
  _BYTE **v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  _BYTE **v17; // rax
  _BYTE **v18; // rdx
  unsigned int v19; // r13d
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rdi
  int *v23; // r8
  int v24; // esi
  int v25; // eax
  unsigned int v26; // edx
  int *v27; // rcx
  int v28; // r9d
  __int64 v30; // rdx
  __int64 *v31; // r14
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rbx
  __int64 *v35; // rbx
  _BYTE *v36; // rdi
  _BYTE *v37; // rdi
  _BYTE *v38; // rdi
  unsigned __int64 v39; // rax
  _BYTE *v40; // rdi
  bool v41; // bl
  bool v42; // zf
  __int64 v43; // rax
  int v44; // eax
  int v45; // edx
  __int64 v46; // rsi
  int v47; // r9d
  _QWORD *v48; // rdi
  unsigned int v49; // eax
  _QWORD *v50; // r8
  __int64 v51; // rcx
  _QWORD *v52; // rax
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 *v56; // rbx
  __int64 v57; // rax
  char *v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 *v64; // r12
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // rdx
  __int64 v70; // rcx
  _BYTE *v71; // r14
  __int64 v72; // rax
  unsigned int v73; // edx
  __int64 v74; // rax
  __int64 v75; // rdi
  unsigned __int64 v76; // rax
  int v77; // ecx
  int v78; // r10d
  __int64 v79; // rdi
  unsigned __int64 v80; // rax
  __int64 v81; // rdi
  signed __int64 v82; // rax
  __int64 *v83; // rax
  __int64 *v84; // rax
  __int64 *v85; // rax
  unsigned int *v86; // rdx
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 *v89; // rsi
  int v90; // eax
  __int64 v91; // rax
  char (__fastcall *v92)(__int64 *, int, int); // rbx
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // r9
  _QWORD *v96; // r9
  __int64 v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // r8
  __int64 v101; // r9
  unsigned int v102; // eax
  unsigned int *v103; // rax
  unsigned int v104; // edx
  unsigned int v105; // esi
  unsigned int v106; // r15d
  _QWORD *v107; // r13
  unsigned int *v108; // rax
  void *v109; // r8
  unsigned __int64 v110; // r13
  _QWORD *v111; // rcx
  _QWORD *v112; // rdx
  _QWORD *v113; // rax
  __int64 v114; // r14
  _QWORD *v115; // rbx
  unsigned int v116; // eax
  unsigned __int64 v117; // rdi
  __int64 v118; // rcx
  _BYTE *v119; // rdx
  unsigned int v120; // eax
  __int64 v121; // [rsp+0h] [rbp-280h]
  __int64 *v122; // [rsp+10h] [rbp-270h]
  __int64 v123; // [rsp+38h] [rbp-248h]
  __int64 *v124; // [rsp+38h] [rbp-248h]
  __int64 v125; // [rsp+38h] [rbp-248h]
  _BYTE v129[4]; // [rsp+58h] [rbp-228h] BYREF
  _BYTE v130[11]; // [rsp+5Ch] [rbp-224h] BYREF
  bool v131; // [rsp+67h] [rbp-219h] BYREF
  int v132; // [rsp+68h] [rbp-218h] BYREF
  int v133; // [rsp+6Ch] [rbp-214h] BYREF
  unsigned int v134; // [rsp+70h] [rbp-210h] BYREF
  int v135; // [rsp+74h] [rbp-20Ch] BYREF
  _BYTE *v136; // [rsp+78h] [rbp-208h] BYREF
  _QWORD v137[2]; // [rsp+80h] [rbp-200h] BYREF
  __int64 v138; // [rsp+90h] [rbp-1F0h] BYREF
  int v139; // [rsp+98h] [rbp-1E8h]
  _QWORD v140[4]; // [rsp+A0h] [rbp-1E0h] BYREF
  _QWORD v141[4]; // [rsp+C0h] [rbp-1C0h] BYREF
  unsigned int *v142; // [rsp+E0h] [rbp-1A0h] BYREF
  unsigned int *v143; // [rsp+E8h] [rbp-198h]
  __m128i *v144; // [rsp+F0h] [rbp-190h]
  _DWORD *v145; // [rsp+F8h] [rbp-188h]
  _BYTE *v146; // [rsp+100h] [rbp-180h] BYREF
  __int64 v147; // [rsp+108h] [rbp-178h]
  _BYTE v148[16]; // [rsp+110h] [rbp-170h] BYREF
  _QWORD v149[8]; // [rsp+120h] [rbp-160h] BYREF
  _QWORD *v150; // [rsp+160h] [rbp-120h] BYREF
  __int64 v151; // [rsp+168h] [rbp-118h]
  _QWORD v152[6]; // [rsp+170h] [rbp-110h] BYREF
  _DWORD *v153; // [rsp+1A0h] [rbp-E0h] BYREF
  _BYTE *v154; // [rsp+1A8h] [rbp-D8h]
  unsigned int *v155; // [rsp+1B0h] [rbp-D0h]
  _QWORD *v156; // [rsp+1B8h] [rbp-C8h]
  __int64 v157; // [rsp+1C0h] [rbp-C0h]
  __int64 v158; // [rsp+1C8h] [rbp-B8h]
  _BYTE *v159; // [rsp+1D0h] [rbp-B0h]
  _BYTE *v160; // [rsp+1D8h] [rbp-A8h]
  _QWORD *v161; // [rsp+1E0h] [rbp-A0h]
  __m128i *v162; // [rsp+1E8h] [rbp-98h]
  __m128i v163; // [rsp+1F0h] [rbp-90h] BYREF
  __int64 v164; // [rsp+200h] [rbp-80h]
  _QWORD *v165; // [rsp+208h] [rbp-78h]
  unsigned int *v166; // [rsp+210h] [rbp-70h]
  _QWORD *v167; // [rsp+218h] [rbp-68h]
  _BYTE *v168; // [rsp+220h] [rbp-60h]
  _QWORD *v169; // [rsp+228h] [rbp-58h]
  __int64 v170; // [rsp+230h] [rbp-50h]
  __int64 v171; // [rsp+238h] [rbp-48h]
  _DWORD *v172; // [rsp+240h] [rbp-40h]
  _BYTE *v173; // [rsp+248h] [rbp-38h]

  v11 = a2;
  v130[0] = a3;
  v12 = *(unsigned int *)(a2 + 8);
  v13 = *(_BYTE ***)a2;
  v12 *= 8;
  v14 = (_BYTE **)((char *)v13 + v12);
  v15 = v12 >> 3;
  v16 = v12 >> 5;
  v129[0] = a10;
  if ( v16 )
  {
    v17 = v13;
    v18 = &v13[4 * v16];
    while ( **v17 <= 0x15u )
    {
      if ( *v17[1] > 0x15u )
      {
        ++v17;
        goto LABEL_8;
      }
      if ( *v17[2] > 0x15u )
      {
        v17 += 2;
        goto LABEL_8;
      }
      if ( *v17[3] > 0x15u )
      {
        v17 += 3;
        goto LABEL_8;
      }
      v17 += 4;
      if ( v18 == v17 )
      {
        v15 = v14 - v17;
        goto LABEL_42;
      }
    }
    goto LABEL_8;
  }
  v17 = v13;
LABEL_42:
  if ( v15 == 2 )
    goto LABEL_49;
  if ( v15 == 3 )
  {
    if ( **v17 > 0x15u )
      goto LABEL_8;
    ++v17;
LABEL_49:
    if ( **v17 > 0x15u )
      goto LABEL_8;
    ++v17;
    goto LABEL_45;
  }
  v19 = 1;
  if ( v15 != 1 )
    return v19;
LABEL_45:
  v19 = 1;
  if ( **v17 <= 0x15u )
    return v19;
LABEL_8:
  v19 = 1;
  if ( v14 == v17 )
    return v19;
  v20 = *((_QWORD *)*v13 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
    v20 = **(_QWORD **)(v20 + 16);
  v163.m128i_i64[0] = sub_9208B0(a1[418], v20);
  v163.m128i_i64[1] = v21;
  v132 = sub_CA1930(&v163);
  if ( v132 == *a4 )
  {
    v19 = 1;
    *a8 = 1;
    return v19;
  }
  if ( (*(_BYTE *)(a7 + 8) & 1) != 0 )
  {
    v22 = a7 + 16;
    v23 = (int *)(a7 + 48);
    v24 = 7;
  }
  else
  {
    v22 = *(_QWORD *)(a7 + 16);
    v30 = *(unsigned int *)(a7 + 24);
    v23 = (int *)(v22 + 4 * v30);
    if ( !(_DWORD)v30 )
      goto LABEL_21;
    v24 = v30 - 1;
  }
  v25 = *(_DWORD *)(v11 + 200);
  v26 = v24 & (37 * v25);
  v27 = (int *)(v22 + 4LL * v26);
  v28 = *v27;
  if ( *v27 == v25 )
  {
LABEL_15:
    if ( v27 != v23 )
      return 0;
  }
  else
  {
    v77 = 1;
    while ( v28 != -1 )
    {
      v78 = v77 + 1;
      v26 = v24 & (v77 + v26);
      v27 = (int *)(v22 + 4LL * v26);
      v28 = *v27;
      if ( v25 == *v27 )
        goto LABEL_15;
      v77 = v78;
    }
  }
LABEL_21:
  v31 = *(__int64 **)v11;
  v32 = 8LL * *(unsigned int *)(v11 + 8);
  v123 = *(_QWORD *)v11 + v32;
  v33 = v32 >> 3;
  v34 = v32 >> 5;
  if ( v34 )
  {
    v35 = &v31[4 * v34];
    while ( 1 )
    {
      v40 = (_BYTE *)*v31;
      if ( *(_BYTE *)*v31 != 13 )
      {
        v163 = (__m128i)(unsigned __int64)a1[418];
        v164 = 0;
        v165 = 0;
        v166 = 0;
        v167 = 0;
        v168 = 0;
        v169 = 0;
        LOWORD(v170) = 257;
        if ( !(unsigned __int8)sub_9AC470((__int64)v40, &v163, 0) )
          goto LABEL_32;
      }
      v36 = (_BYTE *)v31[1];
      if ( *v36 != 13 )
      {
        v163 = (__m128i)(unsigned __int64)a1[418];
        v164 = 0;
        v165 = 0;
        v166 = 0;
        v167 = 0;
        v168 = 0;
        v169 = 0;
        LOWORD(v170) = 257;
        if ( !(unsigned __int8)sub_9AC470((__int64)v36, &v163, 0) )
        {
          v41 = v123 != (_QWORD)(v31 + 1);
          goto LABEL_33;
        }
      }
      v37 = (_BYTE *)v31[2];
      if ( *v37 != 13 )
      {
        v163 = (__m128i)(unsigned __int64)a1[418];
        v164 = 0;
        v165 = 0;
        v166 = 0;
        v167 = 0;
        v168 = 0;
        v169 = 0;
        LOWORD(v170) = 257;
        if ( !(unsigned __int8)sub_9AC470((__int64)v37, &v163, 0) )
        {
          v41 = v123 != (_QWORD)(v31 + 2);
          goto LABEL_33;
        }
      }
      v38 = (_BYTE *)v31[3];
      if ( *v38 != 13 )
      {
        v39 = a1[418];
        LOWORD(v170) = 257;
        v163 = (__m128i)v39;
        v164 = 0;
        v165 = 0;
        v166 = 0;
        v167 = 0;
        v168 = 0;
        v169 = 0;
        if ( !(unsigned __int8)sub_9AC470((__int64)v38, &v163, 0) )
        {
          v41 = v123 != (_QWORD)(v31 + 3);
          goto LABEL_33;
        }
      }
      v31 += 4;
      if ( v35 == v31 )
      {
        v33 = (v123 - (__int64)v31) >> 3;
        break;
      }
    }
  }
  if ( v33 != 2 )
  {
    if ( v33 != 3 )
    {
      v41 = 0;
      if ( v33 != 1 )
        goto LABEL_33;
      goto LABEL_95;
    }
    v81 = *v31;
    if ( *(_BYTE *)*v31 != 13 )
    {
      v163 = (__m128i)(unsigned __int64)a1[418];
      v164 = 0;
      v165 = 0;
      v166 = 0;
      v167 = 0;
      v168 = 0;
      v169 = 0;
      LOWORD(v170) = 257;
      if ( !(unsigned __int8)sub_9AC470(v81, &v163, 0) )
      {
        v41 = v31 != (__int64 *)v123;
        goto LABEL_33;
      }
    }
    ++v31;
  }
  v79 = *v31;
  if ( *(_BYTE *)*v31 != 13 )
  {
    v80 = a1[418];
    LOWORD(v170) = 257;
    v163 = (__m128i)v80;
    v164 = 0;
    v165 = 0;
    v166 = 0;
    v167 = 0;
    v168 = 0;
    v169 = 0;
    if ( !(unsigned __int8)sub_9AC470(v79, &v163, 0) )
      goto LABEL_32;
  }
  ++v31;
LABEL_95:
  v75 = *v31;
  v41 = 0;
  if ( *(_BYTE *)*v31 != 13 )
  {
    v76 = a1[418];
    v164 = 0;
    v163 = (__m128i)v76;
    v165 = 0;
    v166 = 0;
    v167 = 0;
    v168 = 0;
    v169 = 0;
    LOWORD(v170) = 257;
    if ( !(unsigned __int8)sub_9AC470(v75, &v163, 0) )
LABEL_32:
      v41 = v123 != (_QWORD)v31;
  }
LABEL_33:
  v131 = v41;
  v140[1] = &v131;
  v42 = *(_DWORD *)(v11 + 104) == 3;
  v140[2] = &v132;
  v43 = a1[412];
  v171 = a7;
  v163.m128i_i64[0] = v43;
  v172 = a8;
  v163.m128i_i64[1] = (__int64)a9;
  v165 = v140;
  v140[0] = a1;
  v166 = a4;
  v168 = v130;
  v164 = v11;
  v169 = a5;
  v167 = a1;
  v170 = a6;
  v173 = v129;
  if ( v42 )
    goto LABEL_36;
  v44 = *(_DWORD *)(a6 + 24);
  v153 = (_DWORD *)v11;
  if ( v44 )
  {
    v45 = v44 - 1;
    v46 = *(_QWORD *)(a6 + 8);
    v47 = 1;
    v48 = 0;
    v49 = (v44 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v50 = (_QWORD *)(v46 + 8LL * v49);
    v51 = *v50;
    if ( v11 == *v50 )
      goto LABEL_36;
    while ( v51 != -4096 )
    {
      if ( !v48 && v51 == -8192 )
        v48 = v50;
      v49 = v45 & (v47 + v49);
      v50 = (_QWORD *)(v46 + 8LL * v49);
      v51 = *v50;
      if ( v11 == *v50 )
        goto LABEL_36;
      ++v47;
    }
    if ( v48 )
      v50 = v48;
  }
  else
  {
    v50 = 0;
  }
  v52 = sub_2B5CC20(a6, &v153, v50);
  *v52 = v153;
  v56 = *(__int64 **)v11;
  v57 = 8LL * *(unsigned int *)(v11 + 8);
  v58 = (char *)(*(_QWORD *)v11 + v57);
  v122 = *(__int64 **)v11;
  v59 = v57 >> 3;
  v60 = v57 >> 5;
  v121 = v60;
  if ( v60 )
  {
    v124 = &v56[4 * v60];
    while ( *(_BYTE *)*v56 == 13 || !sub_2B15980(*(_QWORD *)(*v56 + 16), 0, (__int64)a1) )
    {
      v61 = v56[1];
      if ( *(_BYTE *)v61 != 13 && sub_2B15980(*(_QWORD *)(v61 + 16), 0, (__int64)a1) )
      {
        ++v56;
        break;
      }
      v62 = v56[2];
      if ( *(_BYTE *)v62 != 13 && sub_2B15980(*(_QWORD *)(v62 + 16), 0, (__int64)a1) )
      {
        v56 += 2;
        break;
      }
      v63 = v56[3];
      if ( *(_BYTE *)v63 != 13 && sub_2B15980(*(_QWORD *)(v63 + 16), 0, (__int64)a1) )
      {
        v56 += 3;
        break;
      }
      v56 += 4;
      if ( v124 == v56 )
      {
        v59 = (v58 - (char *)v56) >> 3;
        goto LABEL_72;
      }
    }
    if ( v58 != (char *)v56 )
      goto LABEL_60;
    v153 = a1;
    v154 = (_BYTE *)v11;
    v155 = a4;
    v156 = v140;
    goto LABEL_76;
  }
  v56 = *(__int64 **)v11;
LABEL_72:
  if ( v59 == 2 )
  {
LABEL_123:
    if ( *(_BYTE *)*v56 == 13 || !sub_2B15980(*(_QWORD *)(*v56 + 16), 0, (__int64)a1) )
    {
      ++v56;
      goto LABEL_126;
    }
    goto LABEL_128;
  }
  if ( v59 != 3 )
  {
    if ( v59 != 1 )
      goto LABEL_75;
LABEL_126:
    if ( *(_BYTE *)*v56 == 13 || !sub_2B15980(*(_QWORD *)(*v56 + 16), 0, (__int64)a1) )
      goto LABEL_75;
    goto LABEL_128;
  }
  if ( *(_BYTE *)*v56 == 13 || !sub_2B15980(*(_QWORD *)(*v56 + 16), 0, (__int64)a1) )
  {
    ++v56;
    goto LABEL_123;
  }
LABEL_128:
  if ( v58 != (char *)v56 )
  {
LABEL_60:
    a9 = (_BYTE *)v163.m128i_i64[1];
LABEL_36:
    if ( *a9 )
      return (unsigned int)sub_2B6ACB0(&v163);
    return 0;
  }
LABEL_75:
  v153 = a1;
  v154 = (_BYTE *)v11;
  v155 = a4;
  v156 = v140;
  if ( v121 )
  {
LABEL_76:
    v125 = v11;
    v64 = v122;
    while ( 1 )
    {
      if ( (unsigned __int8)sub_2B43390((__int64 *)&v153, *v64, v59, v53, v54, v55) )
      {
        v122 = v64;
        v11 = v125;
        goto LABEL_83;
      }
      if ( (unsigned __int8)sub_2B43390((__int64 *)&v153, v64[1], v69, v70, v54, v55) )
        break;
      if ( (unsigned __int8)sub_2B43390((__int64 *)&v153, v64[2], v65, v66, v54, v55) )
      {
        v84 = v64;
        v11 = v125;
        v122 = v84 + 2;
        goto LABEL_83;
      }
      if ( (unsigned __int8)sub_2B43390((__int64 *)&v153, v64[3], v67, v68, v54, v55) )
      {
        v83 = v64;
        v11 = v125;
        v122 = v83 + 3;
        goto LABEL_83;
      }
      v64 += 4;
      if ( &v122[4 * v121] == v64 )
      {
        v122 = v64;
        v11 = v125;
        goto LABEL_131;
      }
    }
    v85 = v64;
    v11 = v125;
    v122 = v85 + 1;
LABEL_83:
    if ( v58 != (char *)v122 )
      return 0;
    goto LABEL_84;
  }
LABEL_131:
  v82 = v58 - (char *)v122;
  if ( v58 - (char *)v122 != 16 )
  {
    if ( v82 != 24 )
    {
      if ( v82 != 8 )
        goto LABEL_84;
      goto LABEL_134;
    }
    if ( (unsigned __int8)sub_2B43390((__int64 *)&v153, *v122, v59, v53, v54, v55) )
      goto LABEL_83;
    ++v122;
  }
  if ( (unsigned __int8)sub_2B43390((__int64 *)&v153, *v122, v59, v53, v54, v55) )
    goto LABEL_83;
  ++v122;
LABEL_134:
  if ( (unsigned __int8)sub_2B43390((__int64 *)&v153, *v122, v59, v53, v54, v55) )
    goto LABEL_83;
LABEL_84:
  v161 = a1;
  v162 = &v163;
  v153 = a8;
  v157 = a6;
  v154 = v130;
  v158 = a7;
  v155 = a4;
  v142 = a4;
  v159 = a9;
  v143 = (unsigned int *)&v132;
  v156 = a5;
  v145 = a8;
  v149[1] = a8;
  v160 = v129;
  v149[3] = v140;
  v144 = &v163;
  v149[0] = v129;
  v149[2] = v11;
  v149[4] = &v142;
  v149[5] = &v153;
  v149[6] = a5;
  v42 = *(_DWORD *)(v11 + 104) == 5;
  v149[7] = a9;
  if ( v42 )
  {
    v86 = *(unsigned int **)(v11 + 208);
    v87 = *a1;
    v150 = *(_QWORD **)(*a1 + 8LL * *v86);
    v74 = *(_QWORD *)(v87 + 8LL * v86[2 * *(unsigned int *)(v11 + 216) - 2]);
LABEL_90:
    v151 = v74;
    return (unsigned int)sub_2B6BA70(v149, a4, &v150, 2, 0, 0);
  }
  else
  {
    v71 = *(_BYTE **)(v11 + 416);
    switch ( *v71 )
    {
      case '*':
      case ',':
      case '.':
      case '9':
      case ':':
      case ';':
        v72 = sub_2B68AE0((__int64)a1, v11, 0);
        v73 = 1;
        v150 = (_QWORD *)v72;
        goto LABEL_89;
      case '0':
      case '3':
        v146 = (_BYTE *)v11;
        v147 = (__int64)a1;
        v150 = (_QWORD *)sub_2B68AE0((__int64)a1, v11, 0);
        v151 = sub_2B68AE0((__int64)a1, v11, 1u);
        v109 = sub_2B19330;
        return (unsigned int)sub_2B6BA70(v149, a4, &v150, 2, v109, &v146);
      case '6':
        v146 = (_BYTE *)v11;
        v147 = (__int64)a1;
        v150 = (_QWORD *)sub_2B68AE0((__int64)a1, v11, 0);
        v151 = sub_2B68AE0((__int64)a1, v11, 1u);
        v109 = sub_2B16210;
        return (unsigned int)sub_2B6BA70(v149, a4, &v150, 2, v109, &v146);
      case '7':
        v146 = (_BYTE *)v11;
        v147 = (__int64)a1;
        v150 = (_QWORD *)sub_2B68AE0((__int64)a1, v11, 0);
        v151 = sub_2B68AE0((__int64)a1, v11, 1u);
        v109 = sub_2B1B2F0;
        return (unsigned int)sub_2B6BA70(v149, a4, &v150, 2, v109, &v146);
      case '8':
        v146 = (_BYTE *)v11;
        v147 = (__int64)a1;
        v150 = (_QWORD *)sub_2B68AE0((__int64)a1, v11, 0);
        v151 = sub_2B68AE0((__int64)a1, v11, 1u);
        v109 = sub_2B1AB20;
        return (unsigned int)sub_2B6BA70(v149, a4, &v150, 2, v109, &v146);
      case 'C':
        if ( v130[0] )
          goto LABEL_178;
        return (unsigned int)sub_2B6BA70(v149, a4, 0, 0, 0, 0);
      case 'D':
      case 'E':
LABEL_178:
        *a9 = 1;
        return (unsigned int)sub_2B6BA70(v149, a4, 0, 0, 0, 0);
      case 'T':
        v110 = *(unsigned int *)(v11 + 248);
        v150 = v152;
        v151 = 0x600000000LL;
        if ( v110 )
        {
          if ( v110 > 6 )
            sub_C8D5F0((__int64)&v150, v152, v110, 8u, v54, v55);
          v111 = v150;
          v112 = &v150[v110];
          v113 = &v150[(unsigned int)v151];
          if ( v113 != v112 )
          {
            do
            {
              if ( v113 )
                *v113 = 0;
              ++v113;
            }
            while ( v112 != v113 );
            v111 = v150;
          }
          LODWORD(v151) = v110;
          v114 = 0;
          v115 = v111;
          do
          {
            v115[v114] = sub_2B68AE0((__int64)a1, v11, v114);
            ++v114;
          }
          while ( v114 != v110 );
        }
        v116 = sub_2B6BA70(v149, a4, v150, (unsigned int)v151, 0, 0);
        v117 = (unsigned __int64)v150;
        v19 = v116;
        if ( v150 != v152 )
          goto LABEL_176;
        return v19;
      case 'U':
        if ( !sub_988010(*(_QWORD *)(v11 + 416)) )
          goto LABEL_86;
        v89 = (__int64 *)a1[413];
        v136 = v71;
        v90 = sub_9B78C0((__int64)v71, v89);
        v133 = v90;
        if ( v90 != 1 && (unsigned int)(v90 - 329) > 1 && (unsigned int)(v90 - 365) > 1 )
          goto LABEL_86;
        v91 = sub_2B68AE0((__int64)a1, v11, 0);
        v146 = v148;
        v92 = sub_2B19780;
        v147 = 0x200000000LL;
        sub_2B3FC00((__int64)&v146, 1u, v91, v93, v94, v95);
        v141[0] = v11;
        v141[1] = &v133;
        v96 = v137;
        v141[2] = a1;
        v137[0] = v11;
        v137[1] = a1;
        if ( v133 != 1 )
        {
          v97 = sub_2B68AE0((__int64)a1, v11, 1u);
          v92 = sub_2B20DA0;
          sub_2B3BAC0((__int64)&v146, v97, v98, v99, v100, v101);
          v96 = v141;
        }
        v151 = (__int64)&v133;
        v138 = 0x7FFFFFFFFFFFFFFFLL;
        v139 = 0;
        v102 = *a4;
        v152[1] = a1;
        v134 = v102;
        v135 = *(_DWORD *)(v11 + 8);
        v150 = &v136;
        v152[0] = &v135;
        v152[2] = &v138;
        v152[3] = &v134;
        v103 = v142;
        v104 = *v143;
        v105 = *v142;
        if ( *v143 <= *v142 )
          goto LABEL_184;
        v106 = 0;
        v107 = v96;
        break;
      case 'V':
        v88 = sub_2B68AE0((__int64)a1, v11, 1u);
        v73 = 2;
        v150 = (_QWORD *)v88;
LABEL_89:
        v74 = sub_2B68AE0((__int64)a1, v11, v73);
        goto LABEL_90;
      case '`':
        v150 = (_QWORD *)sub_2B68AE0((__int64)a1, v11, 0);
        return (unsigned int)sub_2B6BA70(v149, a4, &v150, 1, 0, 0);
      default:
LABEL_86:
        *a8 = 1;
        if ( *(_BYTE *)v163.m128i_i64[1] )
          return (unsigned int)sub_2B6ACB0(&v163);
        return 0;
    }
    do
    {
      if ( (unsigned __int8)sub_2B45AF0((__int64)&v150, v105) )
      {
        v96 = v107;
        goto LABEL_182;
      }
      v108 = v142;
      if ( !v106 )
      {
        if ( *(_BYTE *)v144->m128i_i64[1] )
        {
          v42 = (unsigned __int8)((__int64 (*)(void))sub_2B6ACB0)() == 0;
          v108 = v142;
          if ( !v42 )
            v106 = *v142;
        }
      }
      *v108 *= 2;
      v103 = v142;
      v105 = *v142;
      v104 = *v143;
    }
    while ( *v142 < *v143 );
    v96 = v107;
    if ( v106 )
    {
      *v145 = 1;
      *v142 = v106;
    }
    else
    {
LABEL_184:
      *v103 = v104;
    }
LABEL_182:
    v118 = (unsigned int)v147;
    v119 = v146;
    *a4 = v134;
    v120 = sub_2B6BA70(v149, a4, v119, v118, v92, v96);
    v117 = (unsigned __int64)v146;
    v19 = v120;
    if ( v146 != v148 )
LABEL_176:
      _libc_free(v117);
  }
  return v19;
}
