// Function: sub_32DEAD0
// Address: 0x32dead0
//
__int64 __fastcall sub_32DEAD0(__int64 *a1, __int64 a2)
{
  const __m128i *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r12
  __m128i v6; // xmm0
  __int128 *v7; // rcx
  __int128 *v8; // r14
  __int64 v9; // r15
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // r12
  __m128i v16; // xmm1
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // r8
  int v22; // eax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int128 *v27; // rdx
  __int64 v28; // rdi
  __int128 v29; // rax
  int v30; // r9d
  __int64 v31; // rax
  int v32; // r9d
  __int64 v33; // r12
  unsigned int v34; // eax
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rax
  __int128 *v39; // rdx
  int v40; // ecx
  __int64 v41; // rdi
  __int64 v42; // rdi
  __int128 v43; // rax
  int v44; // r9d
  const __m128i *v45; // roff
  unsigned __int16 *v46; // rax
  __int128 v47; // rax
  int v48; // r9d
  int v49; // r9d
  int v50; // r9d
  int v51; // r9d
  int v52; // r9d
  int v53; // r9d
  __int64 *v54; // rax
  __int64 v55; // rdx
  int v56; // r9d
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rbx
  int v61; // r9d
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // r15
  __int64 v65; // r14
  __int64 v66; // r12
  int v67; // r9d
  __int128 v68; // rax
  int v69; // r9d
  char v70; // bl
  __int64 v71; // rsi
  int v72; // eax
  __int64 v73; // rax
  __int64 v74; // rsi
  int v75; // eax
  __int128 v76; // rax
  int v77; // r9d
  __int64 v78; // rax
  __int64 *v79; // rcx
  __int64 v80; // rdi
  __int64 v81; // rcx
  __int128 *v82; // rbx
  __int64 v83; // r13
  __int128 v84; // rax
  int v85; // r9d
  __int64 v86; // rbx
  unsigned int v87; // esi
  __int64 v88; // rax
  __int64 *v89; // rax
  int v90; // esi
  int v91; // ebx
  int v92; // ecx
  __int64 v93; // rax
  int v94; // edx
  int v95; // eax
  __int64 v96; // r12
  __int128 v97; // rax
  __int64 v98; // rdx
  __int64 v99; // r12
  __int64 v100; // r14
  __int64 v101; // rdx
  __int64 v102; // r15
  __int64 v103; // r12
  __int64 v104; // r13
  __int32 v105; // eax
  __int128 v106; // rax
  __int64 v107; // rbx
  unsigned int v108; // esi
  __int64 v109; // rax
  int v110; // ecx
  int v111; // edx
  __int64 v112; // r12
  __int128 v113; // rax
  __int64 v114; // rdx
  __int128 v115; // rax
  int v116; // r9d
  __int64 v117; // rax
  __int64 v118; // r10
  __int64 v119; // rcx
  __int64 v120; // rdi
  __int128 v121; // [rsp-30h] [rbp-230h]
  __int128 v122; // [rsp-20h] [rbp-220h]
  __int128 v123; // [rsp-20h] [rbp-220h]
  __int128 v124; // [rsp-10h] [rbp-210h]
  __int128 v125; // [rsp-10h] [rbp-210h]
  __int128 v126; // [rsp-10h] [rbp-210h]
  int v127; // [rsp-10h] [rbp-210h]
  __int64 (__fastcall *v128)(__int64, __int64); // [rsp+0h] [rbp-200h]
  __int64 *v129; // [rsp+8h] [rbp-1F8h]
  __int64 v130; // [rsp+10h] [rbp-1F0h]
  __int64 *v131; // [rsp+10h] [rbp-1F0h]
  __int128 v132; // [rsp+10h] [rbp-1F0h]
  __m128i v133; // [rsp+20h] [rbp-1E0h]
  char v134; // [rsp+20h] [rbp-1E0h]
  __int128 *v135; // [rsp+30h] [rbp-1D0h]
  __int64 v136; // [rsp+30h] [rbp-1D0h]
  __int128 v137; // [rsp+30h] [rbp-1D0h]
  char v138; // [rsp+40h] [rbp-1C0h]
  __int64 (__fastcall *v139)(__int64, __int64); // [rsp+40h] [rbp-1C0h]
  int v140; // [rsp+48h] [rbp-1B8h]
  unsigned int v141; // [rsp+48h] [rbp-1B8h]
  __int128 v142; // [rsp+50h] [rbp-1B0h]
  unsigned int v143; // [rsp+68h] [rbp-198h]
  __int64 v145; // [rsp+78h] [rbp-188h]
  __m128i v146; // [rsp+80h] [rbp-180h]
  __m128i v147; // [rsp+90h] [rbp-170h] BYREF
  __int64 v148; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v149; // [rsp+A8h] [rbp-158h]
  __int64 v150; // [rsp+B0h] [rbp-150h] BYREF
  int v151; // [rsp+B8h] [rbp-148h]
  __int128 v152; // [rsp+C0h] [rbp-140h] BYREF
  __int128 v153; // [rsp+D0h] [rbp-130h] BYREF
  __int128 v154; // [rsp+E0h] [rbp-120h] BYREF
  __int128 v155; // [rsp+F0h] [rbp-110h] BYREF
  __int64 v156; // [rsp+100h] [rbp-100h] BYREF
  int v157; // [rsp+108h] [rbp-F8h]
  __int64 v158; // [rsp+110h] [rbp-F0h] BYREF
  int v159; // [rsp+118h] [rbp-E8h]
  __int128 *v160; // [rsp+120h] [rbp-E0h] BYREF
  unsigned int v161; // [rsp+128h] [rbp-D8h]
  __int128 *v162; // [rsp+130h] [rbp-D0h] BYREF
  __int128 *v163; // [rsp+138h] [rbp-C8h]
  __int128 *v164; // [rsp+140h] [rbp-C0h]
  char v165; // [rsp+14Ch] [rbp-B4h]
  __m128i v166; // [rsp+150h] [rbp-B0h] BYREF
  __int128 *v167; // [rsp+160h] [rbp-A0h]
  bool (__fastcall *v168)(__int64, __int64 *, __int64 *); // [rsp+168h] [rbp-98h]
  __int64 *v169; // [rsp+170h] [rbp-90h]
  __int64 *v170; // [rsp+178h] [rbp-88h]
  __int64 *v171; // [rsp+180h] [rbp-80h]
  char v172; // [rsp+18Ch] [rbp-74h]
  __int128 *v173; // [rsp+190h] [rbp-70h]
  int v174; // [rsp+198h] [rbp-68h]
  char v175; // [rsp+19Ch] [rbp-64h]
  __int128 *v176; // [rsp+1A0h] [rbp-60h]
  __int64 v177; // [rsp+1A8h] [rbp-58h]
  __int32 v178; // [rsp+1B0h] [rbp-50h]
  char v179; // [rsp+1BCh] [rbp-44h]
  __int128 *v180; // [rsp+1C0h] [rbp-40h]
  char v181; // [rsp+1CCh] [rbp-34h]

  v3 = *(const __m128i **)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = v3->m128i_i64[0];
  v6 = _mm_loadu_si128(v3);
  v7 = (__int128 *)v3[2].m128i_i64[1];
  v8 = v7;
  v9 = v3[3].m128i_i64[0];
  LODWORD(v3) = v3[3].m128i_i32[0];
  v147 = v6;
  v145 = (__int64)v7;
  v143 = (unsigned int)v3;
  v10 = *(_QWORD *)(v5 + 48) + 16LL * v6.m128i_u32[2];
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v150 = v4;
  LOWORD(v148) = v11;
  v149 = v12;
  if ( v4 )
    sub_B96E90((__int64)&v150, v4, 1);
  v13 = *(_DWORD *)(v5 + 24) == 51;
  v151 = *(_DWORD *)(a2 + 72);
  if ( v13 )
    goto LABEL_11;
  if ( *(_DWORD *)(v145 + 24) == 51 )
  {
    v14 = (__int64)v8;
    goto LABEL_6;
  }
  v16 = _mm_loadu_si128(&v147);
  v17 = *a1;
  v167 = v8;
  v168 = (bool (__fastcall *)(__int64, __int64 *, __int64 *))v9;
  v166 = v16;
  v18 = sub_3402EA0(v17, 56, (unsigned int)&v150, v148, v149, 0, (__int64)&v166, 2);
  if ( v18 )
    goto LABEL_10;
  if ( (unsigned __int8)sub_33E2390(*a1, v147.m128i_i64[0], v147.m128i_i64[1], 1)
    && !(unsigned __int8)sub_33E2390(*a1, v8, v9, 1) )
  {
    *((_QWORD *)&v122 + 1) = v9;
    *(_QWORD *)&v122 = v8;
    v31 = sub_3406EB0(*a1, 56, (unsigned int)&v150, v148, v149, v32, v122, *(_OWORD *)&v147);
    goto LABEL_25;
  }
  v19 = v147.m128i_i64[1];
  if ( sub_325D600(v147.m128i_i64[0], v147.m128i_i64[1], (__int64)v8, v9) )
  {
    v33 = *a1;
    v34 = sub_32844A0((unsigned __int16 *)&v148, v19);
    v166.m128i_i32[2] = v34;
    if ( v34 > 0x40 )
    {
      sub_C43690((__int64)&v166, -1, 1);
    }
    else
    {
      v35 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v34;
      v13 = v34 == 0;
      v36 = 0;
      if ( !v13 )
        v36 = v35;
      v166.m128i_i64[0] = v36;
    }
    v14 = sub_34007B0(v33, (unsigned int)&v166, (unsigned int)&v150, v148, v149, 0, 0);
    if ( v166.m128i_i32[2] > 0x40u && v166.m128i_i64[0] )
      j_j___libc_free_0_0(v166.m128i_u64[0]);
    goto LABEL_6;
  }
  if ( (_WORD)v148 )
  {
    if ( (unsigned __int16)(v148 - 17) > 0xD3u )
      goto LABEL_18;
  }
  else if ( !sub_30070B0((__int64)&v148) )
  {
    goto LABEL_18;
  }
  v18 = sub_3295970(a1, a2, (__int64)&v150, v20, v21);
  if ( v18 )
    goto LABEL_10;
  if ( (unsigned __int8)sub_33D1AE0(v145, 0) )
  {
LABEL_11:
    v14 = v147.m128i_i64[0];
    goto LABEL_6;
  }
LABEL_18:
  if ( (unsigned __int8)sub_33CF170(v8, v9) )
    goto LABEL_11;
  v22 = *(_DWORD *)(v5 + 24);
  if ( v22 == 57 )
  {
    v38 = *(_QWORD *)(v5 + 40);
    v39 = *(__int128 **)(v38 + 40);
    v135 = *(__int128 **)v38;
    v40 = *(_DWORD *)(v38 + 8);
    LODWORD(v38) = *(_DWORD *)(v38 + 48);
    v167 = v39;
    v140 = v40;
    LODWORD(v168) = v38;
    v41 = *a1;
    v166.m128i_i64[0] = (__int64)v8;
    v166.m128i_i64[1] = v9;
    *(_QWORD *)&v29 = sub_3402EA0(v41, 57, (unsigned int)&v150, v148, v149, 0, (__int64)&v166, 2);
    if ( (_QWORD)v29 )
      goto LABEL_24;
    v166.m128i_i64[0] = (__int64)v8;
    v42 = *a1;
    v167 = v135;
    LODWORD(v168) = v140;
    v166.m128i_i64[1] = v9;
    *(_QWORD *)&v43 = sub_3402EA0(v42, 56, (unsigned int)&v150, v148, v149, 0, (__int64)&v166, 2);
    if ( (_QWORD)v43 )
    {
      v125 = *(_OWORD *)(*(_QWORD *)(v5 + 40) + 40LL);
      goto LABEL_45;
    }
    v22 = *(_DWORD *)(v5 + 24);
  }
  if ( v22 == 213 )
  {
    if ( (unsigned __int8)sub_3286E00(&v147) )
    {
      if ( (unsigned __int8)sub_33E0780(v8, v9, 0, v23, v24, v25) )
      {
        v45 = *(const __m128i **)(v5 + 40);
        v136 = v45->m128i_i64[0];
        v141 = v45->m128i_u32[2];
        v133 = _mm_loadu_si128(v45);
        if ( !*((_BYTE *)a1 + 33)
          || (v130 = a1[1], sub_328D6E0(v130, 0xBCu, *(_WORD *)(*(_QWORD *)(v136 + 48) + 16LL * v141)))
          && sub_328D6E0(v130, 0xD6u, v148) )
        {
          if ( sub_3263630(v136, v141) == 1 )
          {
            v46 = (unsigned __int16 *)(*(_QWORD *)(v136 + 48) + 16LL * v141);
            *(_QWORD *)&v47 = sub_34074A0(*a1, &v150, v133.m128i_i64[0], v133.m128i_i64[1], *v46, *((_QWORD *)v46 + 1));
            v14 = sub_33FAF80(*a1, 214, (unsigned int)&v150, v148, v149, v48, v47);
            goto LABEL_6;
          }
        }
      }
    }
  }
  if ( (unsigned __int8)sub_33E03A0(*a1, v147.m128i_i64[0], v147.m128i_i64[1], 0) )
  {
    v26 = *(_QWORD *)(v5 + 40);
    v27 = *(__int128 **)(v26 + 40);
    LODWORD(v26) = *(_DWORD *)(v26 + 48);
    v166.m128i_i64[0] = (__int64)v8;
    v28 = *a1;
    v166.m128i_i64[1] = v9;
    LODWORD(v168) = v26;
    v167 = v27;
    *(_QWORD *)&v29 = sub_3402EA0(v28, 56, (unsigned int)&v150, v148, v149, 0, (__int64)&v166, 2);
    if ( (_QWORD)v29 )
    {
LABEL_24:
      v31 = sub_3406EB0(*a1, 56, (unsigned int)&v150, v148, v149, v30, *(_OWORD *)*(_QWORD *)(v5 + 40), v29);
LABEL_25:
      v14 = v31;
      goto LABEL_6;
    }
  }
  v18 = sub_329BF20(a1, a2);
  if ( v18 )
  {
LABEL_10:
    v14 = v18;
    goto LABEL_6;
  }
  if ( !(unsigned __int8)sub_32724F0(a1, 56, a2, v147.m128i_i64[0], v147.m128i_i64[1], v145, v143) )
  {
    *((_QWORD *)&v124 + 1) = v9;
    *(_QWORD *)&v124 = v8;
    v37 = sub_3286810(a1, 0x38u, (__int64)&v150, v147.m128i_i64[0], v147.m128i_i64[1], *(_DWORD *)(a2 + 28), v124);
    if ( v37
      || (v166.m128i_i64[0] = (__int64)a1,
          v166.m128i_i64[1] = (__int64)&v150,
          v167 = (__int128 *)&v148,
          (v37 = sub_326AAE0((__int64 **)&v166, v147.m128i_i64[0], v147.m128i_i64[1], (__int64)v8, v9)) != 0)
      || (v37 = sub_326AAE0((__int64 **)&v166, v8, v9, v147.m128i_i64[0], v147.m128i_i64[1])) != 0
      || (v37 = sub_328C120(a1, 0x17Eu, 0x38u, (int)&v150, v148, v149, v5, v145, 0)) != 0 )
    {
      v14 = v37;
      goto LABEL_6;
    }
  }
  *(_QWORD *)&v152 = 0;
  v162 = &v152;
  DWORD2(v152) = 0;
  *(_QWORD *)&v153 = 0;
  DWORD2(v153) = 0;
  *(_QWORD *)&v154 = 0;
  DWORD2(v154) = 0;
  *(_QWORD *)&v155 = 0;
  DWORD2(v155) = 0;
  sub_3298290((__int64)&v166, (__int64 *)&v162);
  v134 = sub_329D290(v147.m128i_i64[0], v147.m128i_i64[1], 0, (__int64)&v166);
  sub_969240(&v166.m128i_i64[1]);
  if ( v134 )
  {
    *((_QWORD *)&v123 + 1) = v9;
    *(_QWORD *)&v123 = v8;
    v31 = sub_3406EB0(*a1, 57, (unsigned int)&v150, v148, v149, v49, v123, v152);
    goto LABEL_25;
  }
  v162 = &v153;
  sub_3298290((__int64)&v166, (__int64 *)&v162);
  v138 = sub_329D290((__int64)v8, v9, 0, (__int64)&v166);
  sub_969240(&v166.m128i_i64[1]);
  if ( v138 )
  {
    v31 = sub_3406EB0(*a1, 57, (unsigned int)&v150, v148, v149, v50, *(_OWORD *)&v147, v153);
    goto LABEL_25;
  }
  v167 = (__int128 *)v5;
  v166.m128i_i32[0] = 57;
  v166.m128i_i64[1] = (__int64)&v153;
  LODWORD(v168) = v6.m128i_i32[2];
  BYTE4(v169) = 0;
  if ( sub_329D320((__int64)v8, v9, 0, (__int64)&v166)
    || (v166.m128i_i32[0] = 57,
        BYTE4(v169) = 0,
        v166.m128i_i64[1] = (__int64)&v153,
        v167 = (__int128 *)v145,
        LODWORD(v168) = v143,
        sub_329D320(v147.m128i_i64[0], v147.m128i_i64[1], 0, (__int64)&v166)) )
  {
    v14 = v153;
    goto LABEL_6;
  }
  LODWORD(v162) = 57;
  v165 = 0;
  v163 = &v152;
  v164 = &v153;
  if ( sub_329D3B0(v147.m128i_i64[0], v147.m128i_i64[1], 0, (__int64)&v162) )
  {
    v166.m128i_i64[1] = (__int64)&v154;
    BYTE4(v169) = 0;
    v167 = (__int128 *)v152;
    v166.m128i_i32[0] = 57;
    LODWORD(v168) = DWORD2(v152);
    if ( sub_329D320((__int64)v8, v9, 0, (__int64)&v166) )
    {
      v31 = sub_3406EB0(*a1, 57, (unsigned int)&v150, v148, v149, v51, v154, v153);
      goto LABEL_25;
    }
  }
  v166.m128i_i32[0] = 57;
  BYTE4(v168) = 0;
  v166.m128i_i64[1] = (__int64)&v152;
  v167 = &v153;
  if ( !sub_329D3B0(v147.m128i_i64[0], v147.m128i_i64[1], 0, (__int64)&v166) || *(_DWORD *)(v145 + 24) != 57 )
    goto LABEL_67;
  v54 = *(__int64 **)(v145 + 40);
  v55 = *v54;
  if ( (_QWORD)v153 )
  {
    if ( (_QWORD)v153 == v55 && DWORD2(v153) == *((_DWORD *)v54 + 2) )
      goto LABEL_73;
  }
  else if ( v55 )
  {
LABEL_73:
    v146 = _mm_loadu_si128((const __m128i *)(v54 + 5));
    *(_QWORD *)&v154 = v146.m128i_i64[0];
    DWORD2(v154) = v146.m128i_i32[2];
    v31 = sub_3406EB0(
            *a1,
            57,
            (unsigned int)&v150,
            v148,
            v149,
            v52,
            v152,
            __PAIR128__(*((unsigned __int64 *)&v154 + 1), v146.m128i_u64[0]));
    goto LABEL_25;
  }
LABEL_67:
  v168 = (bool (__fastcall *)(__int64, __int64 *, __int64 *))v5;
  v166.m128i_i64[1] = (__int64)&v153;
  v166.m128i_i32[0] = 57;
  LODWORD(v167) = 56;
  LODWORD(v169) = v6.m128i_i32[2];
  v170 = (__int64 *)&v154;
  BYTE4(v171) = 0;
  v172 = 0;
  if ( sub_329D420((__int64)v8, v9, 0, (__int64)&v166) )
  {
    v31 = sub_3406EB0(*a1, 57, (unsigned int)&v150, v148, v149, v53, v153, v154);
    goto LABEL_25;
  }
  LODWORD(v169) = v6.m128i_i32[2];
  v171 = (__int64 *)&v154;
  v180 = &v154;
  v166.m128i_i32[0] = 57;
  v166.m128i_i32[2] = 57;
  v167 = &v153;
  v168 = (bool (__fastcall *)(__int64, __int64 *, __int64 *))v5;
  BYTE4(v170) = 0;
  v172 = 0;
  LODWORD(v173) = 56;
  v174 = 57;
  v176 = &v153;
  v177 = v5;
  v178 = v6.m128i_i32[2];
  v179 = 0;
  v181 = 0;
  if ( (unsigned __int8)sub_329D520((__int64)v8, v9, 0, (__int64)&v166) )
  {
    v31 = sub_3406EB0(*a1, *(_DWORD *)(v145 + 24), (unsigned int)&v150, v148, v149, v56, v153, v154);
    goto LABEL_25;
  }
  LODWORD(v162) = 57;
  v163 = &v152;
  v165 = 0;
  v164 = &v153;
  if ( (unsigned __int8)sub_329D770(v147.m128i_i64[0], v147.m128i_i32[2], 0, (__int64)&v162) )
  {
    BYTE4(v168) = 0;
    v166.m128i_i64[1] = (__int64)&v154;
    v166.m128i_i32[0] = 57;
    v167 = &v155;
    if ( (unsigned __int8)sub_329D770((__int64)v8, v9, 0, (__int64)&v166) )
    {
      if ( (unsigned __int8)sub_326A930(v152, DWORD2(v152), 0) || (unsigned __int8)sub_326A930(v154, DWORD2(v154), 0) )
      {
        v60 = *a1;
        sub_3285E70((__int64)&v166, (__int64)v8);
        v62 = sub_3406EB0(v60, 56, (unsigned int)&v166, v148, v149, v61, v153, v155);
        v64 = v63;
        v65 = v62;
        v66 = *a1;
        sub_3285E70((__int64)&v162, v147.m128i_i64[0]);
        *(_QWORD *)&v68 = sub_3406EB0(v66, 56, (unsigned int)&v162, v148, v149, v67, v152, v154);
        *((_QWORD *)&v126 + 1) = v64;
        *(_QWORD *)&v126 = v65;
        v14 = sub_3406EB0(v60, 57, (unsigned int)&v150, v148, v149, v69, v68, v126);
        sub_9C6650(&v162);
        sub_9C6650(&v166);
        goto LABEL_6;
      }
    }
  }
  if ( *(_DWORD *)(v5 + 24) == 183 )
  {
    if ( (unsigned __int8)sub_328A020(a1[1], 0x55u, v148, v149, *((unsigned __int8 *)a1 + 33)) )
    {
      v168 = sub_3264760;
      v167 = (__int128 *)sub_325D4C0;
      v70 = sub_33CACD0(
              *(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL),
              *(_QWORD *)(*(_QWORD *)(v5 + 40) + 48LL),
              (_DWORD)v8,
              v9,
              (unsigned int)&v166,
              1,
              0);
      sub_A17130((__int64)&v166);
      if ( v70 )
      {
        v31 = sub_3406EB0(
                *a1,
                85,
                (unsigned int)&v150,
                v148,
                v149,
                v127,
                *(_OWORD *)*(_QWORD *)(v5 + 40),
                *(_OWORD *)(*(_QWORD *)(v5 + 40) + 40LL));
        goto LABEL_25;
      }
    }
  }
  if ( (unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) )
  {
    v14 = a2;
    goto LABEL_6;
  }
  v71 = v9;
  if ( !(unsigned __int8)sub_33E0780(v8, v9, 0, v57, v58, v59) )
    goto LABEL_90;
  v74 = v147.m128i_i64[1];
  if ( !(unsigned __int8)sub_33DFCF0(v147.m128i_i64[0], v147.m128i_i64[1], 0) )
  {
    if ( *(_DWORD *)(v5 + 24) != 56 )
      goto LABEL_98;
    if ( (unsigned __int8)sub_33DFCF0(**(_QWORD **)(v5 + 40), *(_QWORD *)(*(_QWORD *)(v5 + 40) + 8LL), 0) )
    {
      v79 = *(__int64 **)(v5 + 40);
      v80 = *((unsigned int *)v79 + 12);
      *(_QWORD *)&v43 = v79[5];
      v81 = *v79;
      *((_QWORD *)&v43 + 1) = v80;
    }
    else
    {
      if ( !(unsigned __int8)sub_33DFCF0(
                               *(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL),
                               *(_QWORD *)(*(_QWORD *)(v5 + 40) + 48LL),
                               0) )
        goto LABEL_98;
      v119 = *(_QWORD *)(v5 + 40);
      v120 = *(unsigned int *)(v119 + 8);
      *(_QWORD *)&v43 = *(_QWORD *)v119;
      v81 = *(_QWORD *)(v119 + 40);
      *((_QWORD *)&v43 + 1) = v120;
    }
    if ( !v81 )
    {
LABEL_98:
      v71 = (unsigned int)v148;
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a1[1] + 456LL))(
              a1[1],
              (unsigned int)v148,
              v149)
        && *(_DWORD *)(v5 + 24) == 56 )
      {
        if ( (unsigned __int8)sub_3286E00(&v147) )
        {
          if ( *((int *)a1 + 6) > 2 || (v75 = *(_DWORD *)(a2 + 28), (v75 & 1) == 0) && (v75 & 2) == 0 )
          {
            *(_QWORD *)&v76 = sub_34074A0(
                                *a1,
                                &v150,
                                **(_QWORD **)(v5 + 40),
                                *(_QWORD *)(*(_QWORD *)(v5 + 40) + 8LL),
                                (unsigned int)v148,
                                v149);
            v78 = sub_3406EB0(
                    *a1,
                    57,
                    (unsigned int)&v150,
                    v148,
                    v149,
                    v77,
                    *(_OWORD *)(*(_QWORD *)(v5 + 40) + 40LL),
                    v76);
LABEL_105:
            v14 = v78;
            goto LABEL_6;
          }
        }
LABEL_91:
        v157 = 1;
        v156 = 0;
        v159 = 1;
        v72 = *(_DWORD *)(v145 + 24);
        v158 = 0;
        if ( (v72 == 35 || v72 == 11) && (unsigned __int64)sub_32844A0((unsigned __int16 *)&v148, v71) <= 0x40 )
        {
          BYTE4(v171) = 0;
          v166.m128i_i32[0] = 58;
          v167 = &v152;
          v168 = (bool (__fastcall *)(__int64, __int64 *, __int64 *))&v158;
          v166.m128i_i32[2] = 56;
          BYTE4(v169) = 0;
          v170 = &v156;
          if ( (unsigned __int8)sub_329D830(v147.m128i_i64[0], v147.m128i_i32[2], 0, (__int64)&v166) )
          {
            v107 = a1[1];
            v128 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v107 + 1320LL);
            v129 = (__int64 *)(*(_QWORD *)(v145 + 96) + 24LL);
            sub_C472A0((__int64)&v160, (__int64)&v158, &v156);
            sub_C45EE0((__int64)&v160, v129);
            v108 = v161;
            v161 = 0;
            LODWORD(v163) = v108;
            v162 = v160;
            v109 = sub_325F4E0((__int64)v160, v108);
            LOBYTE(v107) = v128(v107, v109);
            sub_969240((__int64 *)&v162);
            sub_969240((__int64 *)&v160);
            if ( (_BYTE)v107 )
            {
              v110 = *(_DWORD *)(a2 + 28);
              v91 = 0;
              if ( (v110 & 1) != 0 )
              {
                v111 = *(_DWORD *)(v5 + 28);
                if ( (v111 & 1) != 0 && (*(_DWORD *)(**(_QWORD **)(v5 + 40) + 28LL) & 1) != 0 )
                {
                  v91 = 1;
                  if ( (v110 & 2) != 0 && (v111 & 2) != 0 )
                    v91 = (*(_DWORD *)(**(_QWORD **)(v5 + 40) + 28LL) & 2) == 0 ? 1 : 3;
                }
              }
              v112 = *a1;
              *(_QWORD *)&v113 = sub_34007B0(*a1, (unsigned int)&v156, (unsigned int)&v150, v148, v149, 0, 0);
              v137 = v113;
              sub_3285E70((__int64)&v166, (__int64)v8);
              v100 = sub_3405C90(v112, 58, (unsigned int)&v166, v148, v149, v91, v152, v137);
              v102 = v114;
LABEL_125:
              sub_9C6650(&v166);
              v103 = *a1;
              v104 = *(_QWORD *)(v145 + 96);
              sub_C472A0((__int64)&v162, (__int64)&v158, &v156);
              sub_C45EE0((__int64)&v162, (__int64 *)(v104 + 24));
              v105 = (int)v163;
              LODWORD(v163) = 0;
              v166.m128i_i32[2] = v105;
              v166.m128i_i64[0] = (__int64)v162;
              *(_QWORD *)&v106 = sub_34007B0(v103, (unsigned int)&v166, (unsigned int)&v150, v148, v149, 0, 0);
              *((_QWORD *)&v121 + 1) = v102;
              *(_QWORD *)&v121 = v100;
              v14 = sub_3405C90(v103, 56, (unsigned int)&v150, v148, v149, v91, v121, v106);
              sub_969240(v166.m128i_i64);
              sub_969240((__int64 *)&v162);
LABEL_95:
              sub_969240(&v158);
              sub_969240(&v156);
              goto LABEL_6;
            }
          }
          v172 = 0;
          v175 = 0;
          v168 = (bool (__fastcall *)(__int64, __int64 *, __int64 *))&v152;
          v166.m128i_i32[0] = 56;
          v169 = &v158;
          v166.m128i_i32[2] = 58;
          v171 = &v156;
          LODWORD(v167) = 56;
          BYTE4(v170) = 0;
          v173 = &v153;
          if ( (unsigned __int8)sub_329DDA0(v147.m128i_i64[0], v147.m128i_i32[2], 0, (__int64)&v166) )
          {
            v86 = a1[1];
            v139 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v86 + 1320LL);
            v131 = (__int64 *)(*(_QWORD *)(v145 + 96) + 24LL);
            sub_C472A0((__int64)&v160, (__int64)&v158, &v156);
            sub_C45EE0((__int64)&v160, v131);
            v87 = v161;
            v161 = 0;
            LODWORD(v163) = v87;
            v162 = v160;
            v88 = sub_325F4E0((__int64)v160, v87);
            LOBYTE(v86) = v139(v86, v88);
            sub_969240((__int64 *)&v162);
            sub_969240((__int64 *)&v160);
            if ( (_BYTE)v86 )
            {
              v89 = *(__int64 **)(v5 + 40);
              if ( *v89 == (_QWORD)v153 && *((_DWORD *)v89 + 2) == DWORD2(v153) )
                v89 += 5;
              v90 = *(_DWORD *)(a2 + 28);
              v91 = 0;
              if ( (v90 & 1) != 0 )
              {
                v92 = *(_DWORD *)(v5 + 28);
                if ( (v92 & 1) != 0 )
                {
                  v93 = *v89;
                  v94 = *(_DWORD *)(v93 + 28);
                  if ( (v94 & 1) != 0 )
                  {
                    v95 = *(_DWORD *)(**(_QWORD **)(v93 + 40) + 28LL);
                    if ( (v95 & 1) != 0 )
                    {
                      v91 = 1;
                      if ( (v90 & 2) != 0 && (v92 & 2) != 0 && (v94 & 2) != 0 )
                        v91 = (v95 & 2) == 0 ? 1 : 3;
                    }
                  }
                }
              }
              v96 = *a1;
              *(_QWORD *)&v97 = sub_34007B0(*a1, (unsigned int)&v156, (unsigned int)&v150, v148, v149, 0, 0);
              v132 = v97;
              sub_3285E70((__int64)&v166, (__int64)v8);
              *(_QWORD *)&v142 = sub_3405C90(v96, 58, (unsigned int)&v166, v148, v149, v91, v152, v132);
              *((_QWORD *)&v142 + 1) = v98;
              sub_9C6650(&v166);
              v99 = *a1;
              sub_3285E70((__int64)&v166, (__int64)v8);
              v100 = sub_3405C90(v99, 56, (unsigned int)&v166, v148, v149, v91, v142, v153);
              v102 = v101;
              goto LABEL_125;
            }
          }
        }
        v73 = sub_3286E70(a1, v147.m128i_i64[0], v147.m128i_i64[1], (__int64)v8, v9, a2);
        if ( v73 )
        {
          v14 = v73;
        }
        else
        {
          v117 = sub_3286E70(a1, (__int64)v8, v9, v147.m128i_i64[0], v147.m128i_i64[1], a2);
          v118 = 0;
          if ( v117 )
            v118 = v117;
          v14 = v118;
        }
        goto LABEL_95;
      }
LABEL_90:
      if ( *(_DWORD *)(v5 + 24) == 57 )
      {
        if ( (unsigned __int8)sub_3286E00(&v147) )
        {
          v71 = v9;
          if ( (unsigned __int8)sub_33E07E0(v8, v9, 1) )
          {
            *(_QWORD *)&v115 = sub_34074A0(
                                 *a1,
                                 &v150,
                                 *(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL),
                                 *(_QWORD *)(*(_QWORD *)(v5 + 40) + 48LL),
                                 (unsigned int)v148,
                                 v149);
            v78 = sub_3406EB0(*a1, 56, (unsigned int)&v150, v148, v149, v116, v115, *(_OWORD *)*(_QWORD *)(v5 + 40));
            goto LABEL_105;
          }
        }
      }
      goto LABEL_91;
    }
    v125 = *(_OWORD *)*(_QWORD *)(v81 + 40);
LABEL_45:
    v31 = sub_3406EB0(*a1, 57, (unsigned int)&v150, v148, v149, v44, v43, v125);
    goto LABEL_25;
  }
  v82 = *(__int128 **)(v5 + 40);
  v83 = *a1;
  *(_QWORD *)&v84 = sub_3400BD0(*a1, 0, (unsigned int)&v150, v148, v149, 0, 0, v74);
  v14 = sub_3406EB0(v83, 57, (unsigned int)&v150, v148, v149, v85, v84, *v82);
LABEL_6:
  if ( v150 )
    sub_B91220((__int64)&v150, v150);
  return v14;
}
