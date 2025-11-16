// Function: sub_3450B60
// Address: 0x3450b60
//
__int64 __fastcall sub_3450B60(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4, _QWORD *a5, __m128i a6)
{
  __int64 v9; // rsi
  int v10; // edx
  __int64 v11; // rax
  __int64 *v12; // rdi
  const __m128i *v13; // roff
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 (__fastcall *v20)(__int64 *, __int64, __int64, _QWORD, __int64); // r13
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // rdi
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 (__fastcall *v27)(__int64 *, __int64, __int64, _QWORD, __int64); // r13
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdx
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned __int8 v34; // al
  unsigned int v35; // r12d
  _DWORD *v37; // rax
  unsigned __int16 v38; // r13
  _DWORD *v39; // r15
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // rdx
  __int64 v44; // rdx
  _DWORD *v45; // rax
  _DWORD *v46; // r13
  __int64 v47; // rsi
  unsigned __int16 v48; // r15
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned int v52; // r15d
  __int64 v53; // rcx
  char v54; // dl
  int v55; // r9d
  int v56; // eax
  __int64 v57; // rdx
  __int64 v58; // rax
  __m128i v59; // rax
  int v60; // eax
  __int128 v61; // rax
  __int64 v62; // r9
  __int64 v63; // rax
  int v64; // r9d
  __int32 v65; // edx
  __int64 v66; // r15
  int v67; // eax
  __int64 (*v68)(); // rax
  __int128 v69; // rax
  __int64 v70; // r9
  int v71; // r9d
  __int128 v72; // rax
  __int128 v73; // rax
  __int64 v74; // r9
  __int64 v75; // rdx
  unsigned __int8 *v76; // rax
  int v77; // edi
  unsigned int v78; // edx
  __int64 v79; // rax
  _QWORD *v80; // rbx
  __int64 v81; // rdx
  __int64 v82; // rdx
  __int64 v83; // rax
  __int32 v84; // edx
  __m128i v85; // xmm2
  __int64 v86; // rdx
  __int64 v87; // r9
  __int64 v88; // rax
  __int32 v89; // edx
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // rdi
  _QWORD *v95; // rbx
  __int64 v96; // rdx
  __m128i v97; // xmm3
  __m128i v98; // xmm4
  _QWORD *v99; // rdi
  __m128i v100; // xmm2
  unsigned __int8 *v101; // rax
  _QWORD *v102; // rbx
  __int64 v103; // rdx
  __int64 v104; // rbx
  __int128 v105; // rax
  unsigned int v106; // r14d
  unsigned int v107; // ecx
  __int64 v108; // r8
  __int16 v109; // dx
  unsigned __int128 v110; // kr10_16
  unsigned int v111; // esi
  __m128i v112; // rax
  unsigned __int8 *v113; // r15
  unsigned int v114; // edx
  unsigned int v115; // r14d
  __int128 v116; // rax
  __int128 v117; // rax
  __int64 v118; // r10
  unsigned int v119; // ecx
  __int64 v120; // r8
  __int64 v121; // r14
  __int64 v122; // r11
  __int16 v123; // ax
  bool v124; // al
  __int64 v125; // r9
  __int64 v126; // r14
  __int64 v127; // rdx
  __int64 v128; // r15
  int v129; // eax
  int v130; // r9d
  unsigned __int8 *v131; // rax
  __int64 v132; // r9
  __int64 v133; // rdx
  unsigned __int64 v134; // r11
  _QWORD *v135; // rdi
  unsigned __int8 *v136; // rax
  _QWORD *v137; // rbx
  __int64 v138; // rdx
  __m128i v139; // xmm7
  __m128i v140; // xmm6
  __m128i v141; // xmm5
  __m128i si128; // xmm5
  __m128i v143; // rax
  __m128i v144; // xmm7
  __int64 v145; // r9
  __int64 v146; // rbx
  __int64 v147; // rdx
  unsigned __int64 v148; // rcx
  bool v149; // al
  __int128 v150; // rax
  __int64 v151; // r9
  __int128 v152; // [rsp-10h] [rbp-250h]
  __int128 v153; // [rsp-10h] [rbp-250h]
  __int128 v154; // [rsp+0h] [rbp-240h]
  __int128 v155; // [rsp+0h] [rbp-240h]
  __int128 v156; // [rsp+0h] [rbp-240h]
  __int128 v157; // [rsp+0h] [rbp-240h]
  __int64 v158; // [rsp+10h] [rbp-230h]
  __int32 v159; // [rsp+18h] [rbp-228h]
  __m128i v160; // [rsp+20h] [rbp-220h] BYREF
  __int128 v161; // [rsp+30h] [rbp-210h]
  unsigned __int128 v162; // [rsp+40h] [rbp-200h] BYREF
  __int64 v163; // [rsp+50h] [rbp-1F0h]
  __int64 v164; // [rsp+58h] [rbp-1E8h]
  __int128 v165; // [rsp+60h] [rbp-1E0h]
  __int128 v166; // [rsp+70h] [rbp-1D0h]
  __int128 v167; // [rsp+80h] [rbp-1C0h]
  __m128i v168; // [rsp+90h] [rbp-1B0h] BYREF
  __int128 v169; // [rsp+A0h] [rbp-1A0h]
  _QWORD *v170; // [rsp+B0h] [rbp-190h]
  __m128i *v171; // [rsp+B8h] [rbp-188h]
  unsigned __int8 *v172; // [rsp+C0h] [rbp-180h]
  __int64 v173; // [rsp+C8h] [rbp-178h]
  unsigned __int8 *v174; // [rsp+D0h] [rbp-170h]
  __int64 v175; // [rsp+D8h] [rbp-168h]
  unsigned __int8 *v176; // [rsp+E0h] [rbp-160h]
  __int64 v177; // [rsp+E8h] [rbp-158h]
  __int64 v178; // [rsp+F0h] [rbp-150h]
  __int64 v179; // [rsp+F8h] [rbp-148h]
  unsigned __int8 *v180; // [rsp+100h] [rbp-140h]
  __int64 v181; // [rsp+108h] [rbp-138h]
  unsigned __int8 *v182; // [rsp+110h] [rbp-130h]
  __int64 v183; // [rsp+118h] [rbp-128h]
  unsigned __int8 *v184; // [rsp+120h] [rbp-120h]
  __int64 v185; // [rsp+128h] [rbp-118h]
  __int64 v186; // [rsp+130h] [rbp-110h] BYREF
  int v187; // [rsp+138h] [rbp-108h]
  __m128i v188; // [rsp+140h] [rbp-100h] BYREF
  __m128i v189; // [rsp+150h] [rbp-F0h] BYREF
  unsigned __int64 v190; // [rsp+160h] [rbp-E0h] BYREF
  unsigned int v191; // [rsp+168h] [rbp-D8h]
  __int64 v192; // [rsp+170h] [rbp-D0h]
  __int64 v193; // [rsp+178h] [rbp-C8h]
  __int64 v194; // [rsp+180h] [rbp-C0h]
  __int64 v195; // [rsp+188h] [rbp-B8h]
  _QWORD v196[4]; // [rsp+190h] [rbp-B0h] BYREF
  __m128i v197; // [rsp+1B0h] [rbp-90h] BYREF
  __int16 v198; // [rsp+1C0h] [rbp-80h]
  __int64 v199; // [rsp+1C8h] [rbp-78h]
  __m128i v200; // [rsp+1D0h] [rbp-70h] BYREF
  __m128i v201; // [rsp+1E0h] [rbp-60h]
  __m128i v202; // [rsp+1F0h] [rbp-50h]
  __int64 v203; // [rsp+200h] [rbp-40h]
  __int64 v204; // [rsp+208h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 80);
  v170 = a3;
  *(_QWORD *)&v166 = a4;
  v186 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v186, v9, 1);
  v10 = *(_DWORD *)(a2 + 24);
  v187 = *(_DWORD *)(a2 + 72);
  if ( v10 > 239 )
  {
    v11 = (unsigned int)(v10 - 242) < 2 ? 0x28 : 0;
  }
  else
  {
    v11 = 40;
    if ( v10 <= 237 )
      v11 = (unsigned int)(v10 - 101) < 0x30 ? 0x28 : 0;
  }
  v12 = (__int64 *)a5[5];
  v13 = (const __m128i *)(*(_QWORD *)(a2 + 40) + v11);
  v14 = v13->m128i_i64[0];
  v15 = v13->m128i_u32[2];
  v16 = a5[8];
  v168 = _mm_loadu_si128(v13);
  v17 = *(_QWORD *)(v14 + 48) + 16 * v15;
  LOWORD(v14) = *(_WORD *)v17;
  v188.m128i_i64[1] = *(_QWORD *)(v17 + 8);
  v18 = *(_QWORD *)(a2 + 48);
  v188.m128i_i16[0] = v14;
  LOWORD(v14) = *(_WORD *)v18;
  v189.m128i_i64[1] = *(_QWORD *)(v18 + 8);
  v19 = *a1;
  v189.m128i_i16[0] = v14;
  v20 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, _QWORD, __int64))(v19 + 528);
  v21 = sub_2E79000(v12);
  v22 = v20(a1, v21, v16, v188.m128i_u32[0], v188.m128i_i64[1]);
  v23 = (__int64 *)a5[5];
  v24 = a5[8];
  *(_QWORD *)&v161 = v22;
  v25 = *a1;
  *(_QWORD *)&v162 = v26;
  v27 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, _QWORD, __int64))(v25 + 528);
  v28 = sub_2E79000(v23);
  v29 = v27(a1, v28, v24, v189.m128i_u32[0], v189.m128i_i64[1]);
  *(_QWORD *)&v165 = v30;
  v31 = *(_DWORD *)(a2 + 24);
  v163 = v29;
  if ( v31 > 239 )
  {
    v32 = (unsigned int)(v31 - 242) < 2 ? 141 : 226;
  }
  else
  {
    v32 = 141;
    if ( v31 <= 237 )
      v32 = (unsigned int)(v31 - 101) < 0x30 ? 141 : 226;
  }
  if ( v189.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v189.m128i_i16[0] - 17) <= 0xD3u )
    {
      if ( !a1[v189.m128i_u16[0] + 14] )
        goto LABEL_18;
      if ( (*((_BYTE *)a1 + 500 * v189.m128i_u16[0] + v32 + 6414) & 0xFB) != 0 )
        goto LABEL_18;
      v33 = 1;
      if ( v188.m128i_i16[0] != 1 )
      {
        if ( !v188.m128i_i16[0] )
          goto LABEL_18;
        v33 = v188.m128i_u16[0];
        if ( !a1[v188.m128i_u16[0] + 14] )
          goto LABEL_18;
      }
      v34 = *((_BYTE *)a1 + 500 * v33 + 6602);
      if ( v34 > 1u && v34 != 4 )
        goto LABEL_18;
    }
  }
  else if ( sub_30070B0((__int64)&v189) )
  {
LABEL_18:
    v35 = 0;
    goto LABEL_19;
  }
  v171 = &v188;
  v37 = sub_300AC80((unsigned __int16 *)&v188, v28);
  v38 = v188.m128i_i16[0];
  v39 = v37;
  if ( !v188.m128i_i16[0] )
  {
    if ( sub_30070B0((__int64)&v188) )
    {
      v38 = sub_3009970((__int64)v171, v28, v40, v41, v42);
LABEL_26:
      v200.m128i_i16[0] = v38;
      v200.m128i_i64[1] = v43;
      if ( !v38 )
        goto LABEL_27;
LABEL_59:
      if ( v38 == 1 || (unsigned __int16)(v38 - 504) <= 7u )
        goto LABEL_114;
      *(_QWORD *)&v169 = &v200;
      v200.m128i_i32[2] = *(_QWORD *)&byte_444C4A0[16 * v38 - 16];
      if ( v200.m128i_i32[2] <= 0x40u )
        goto LABEL_28;
      goto LABEL_62;
    }
LABEL_25:
    v43 = v188.m128i_i64[1];
    goto LABEL_26;
  }
  if ( (unsigned __int16)(v188.m128i_i16[0] - 17) > 0xD3u )
    goto LABEL_25;
  v200.m128i_i64[1] = 0;
  v38 = word_4456580[v188.m128i_u16[0] - 1];
  v200.m128i_i16[0] = v38;
  if ( v38 )
    goto LABEL_59;
LABEL_27:
  *(_QWORD *)&v169 = &v200;
  v192 = sub_3007260((__int64)&v200);
  v193 = v44;
  v200.m128i_i32[2] = v192;
  if ( (unsigned int)v192 <= 0x40 )
  {
LABEL_28:
    v200.m128i_i64[0] = 0;
    goto LABEL_29;
  }
LABEL_62:
  sub_C43690(v169, 0, 0);
LABEL_29:
  v45 = sub_C33340();
  v46 = v45;
  v171 = (__m128i *)v196;
  if ( v39 == v45 )
  {
    v47 = (__int64)v45;
    sub_C3C640(v196, (__int64)v45, (_QWORD *)v169);
  }
  else
  {
    v47 = (__int64)v39;
    sub_C3B160((__int64)v196, v39, (__int64 *)v169);
  }
  if ( v200.m128i_i32[2] > 0x40u && v200.m128i_i64[0] )
    j_j___libc_free_0_0(v200.m128i_u64[0]);
  v48 = v189.m128i_i16[0];
  if ( v189.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v189.m128i_i16[0] - 17) > 0xD3u )
    {
LABEL_36:
      v49 = v189.m128i_i64[1];
      goto LABEL_37;
    }
    v49 = 0;
    v48 = word_4456580[v189.m128i_u16[0] - 1];
  }
  else
  {
    *(_QWORD *)&v167 = &v189;
    if ( !sub_30070B0((__int64)&v189) )
      goto LABEL_36;
    v48 = sub_3009970((__int64)&v189, v47, v91, v92, v93);
  }
LABEL_37:
  v197.m128i_i16[0] = v48;
  v197.m128i_i64[1] = v49;
  if ( !v48 )
  {
    v50 = sub_3007260((__int64)&v197);
    v194 = v50;
    v195 = v51;
    goto LABEL_39;
  }
  if ( v48 == 1 || (unsigned __int16)(v48 - 504) <= 7u )
LABEL_114:
    BUG();
  v50 = *(_QWORD *)&byte_444C4A0[16 * v48 - 16];
LABEL_39:
  v52 = v50 - 1;
  v191 = v50;
  v53 = 1LL << ((unsigned __int8)v50 - 1);
  if ( (unsigned int)v50 <= 0x40 )
  {
    v190 = 0;
    *(_QWORD *)&v167 = &v190;
LABEL_41:
    v190 |= v53;
    goto LABEL_42;
  }
  v160.m128i_i64[0] = 1LL << ((unsigned __int8)v50 - 1);
  *(_QWORD *)&v167 = &v190;
  sub_C43690((__int64)&v190, 0, 0);
  v53 = v160.m128i_i64[0];
  if ( v191 <= 0x40 )
    goto LABEL_41;
  *(_QWORD *)(v190 + 8LL * (v52 >> 6)) |= v160.m128i_i64[0];
LABEL_42:
  if ( (_DWORD *)v196[0] == v46 )
    v54 = sub_C400C0(v171, v167, 0, 1u);
  else
    v54 = sub_C36910((__int64)v171, v167, 0, 1);
  v56 = *(_DWORD *)(a2 + 24);
  if ( (v54 & 4) != 0 )
  {
    if ( v56 > 239 )
    {
      if ( (unsigned int)(v56 - 242) > 1 )
      {
LABEL_84:
        v94 = (__int64)a5;
        v35 = 1;
        v95 = v170;
        v182 = sub_33FAF80(v94, 226, (__int64)&v186, v189.m128i_u32[0], v189.m128i_i64[1], v55, a6);
        v183 = v96;
        *v170 = v182;
        *((_DWORD *)v95 + 2) = v183;
        goto LABEL_51;
      }
    }
    else if ( v56 <= 237 && (unsigned int)(v56 - 101) > 0x2F )
    {
      goto LABEL_84;
    }
    v97 = _mm_load_si128(&v168);
    v98 = _mm_loadu_si128(&v189);
    v99 = a5;
    v100 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    *((_QWORD *)&v155 + 1) = 2;
    v198 = 1;
    v35 = 1;
    *(_QWORD *)&v155 = v169;
    v199 = 0;
    v200 = v100;
    v201 = v97;
    v197 = v98;
    v101 = sub_3411BE0(v99, 0x8Du, (__int64)&v186, (unsigned __int16 *)&v197, 2, (__int64)&v186, v155);
    v102 = v170;
    v185 = v103;
    v184 = v101;
    *v170 = v101;
    *((_DWORD *)v102 + 2) = v185;
    v104 = v166;
    *(_QWORD *)v166 = v101;
    *(_DWORD *)(v104 + 8) = 1;
    goto LABEL_51;
  }
  if ( v56 > 239 )
  {
    v57 = (unsigned int)(v56 - 242) < 2 ? 102 : 97;
  }
  else
  {
    v57 = 102;
    if ( v56 <= 237 )
      v57 = (unsigned int)(v56 - 101) < 0x30 ? 102 : 97;
  }
  v58 = 1;
  if ( v188.m128i_i16[0] != 1 && (!v188.m128i_i16[0] || (v58 = v188.m128i_u16[0], !a1[v188.m128i_u16[0] + 14]))
    || (*((_BYTE *)a1 + 500 * v58 + v57 + 6414) & 0xFB) != 0 )
  {
    v35 = 0;
    goto LABEL_51;
  }
  v59.m128i_i64[0] = sub_33FE6E0(
                       (__int64)a5,
                       v171->m128i_i64,
                       (__int64)&v186,
                       v188.m128i_u32[0],
                       v188.m128i_i64[1],
                       0,
                       a6);
  v160 = v59;
  v60 = *(_DWORD *)(a2 + 24);
  if ( v60 > 239 )
  {
    if ( (unsigned int)(v60 - 242) > 1 )
    {
LABEL_66:
      *(_QWORD *)&v61 = sub_33ED040(a5, 0x14u);
      v63 = sub_340F900(a5, 0xD0u, (__int64)&v186, v161, v162, v62, *(_OWORD *)&v168, *(_OWORD *)&v160, v61);
      LODWORD(v162) = v65;
      v66 = v63;
      goto LABEL_67;
    }
  }
  else if ( v60 <= 237 && (unsigned int)(v60 - 101) > 0x2F )
  {
    goto LABEL_66;
  }
  v82 = *(_QWORD *)(a2 + 40);
  v83 = *(_QWORD *)v82;
  if ( *(_QWORD *)v82 )
  {
    v84 = *(_DWORD *)(v82 + 8);
    a6 = _mm_load_si128(&v168);
    v85 = _mm_load_si128(&v160);
    v200.m128i_i64[0] = v83;
    v201 = a6;
    v202 = v85;
    v200.m128i_i32[2] = v84;
    v203 = sub_33ED040(a5, 0x14u);
    v204 = v86;
    *((_QWORD *)&v154 + 1) = 4;
    *(_QWORD *)&v154 = v169;
    v197.m128i_i64[0] = v161;
    v198 = 1;
    v197.m128i_i64[1] = v162;
    v199 = 0;
    v88 = (__int64)sub_3411BE0(a5, 0x94u, (__int64)&v186, (unsigned __int16 *)&v197, 2, v87, v154);
  }
  else
  {
    *(_QWORD *)&v150 = sub_33ED040(a5, 0x14u);
    v88 = sub_340F900(a5, 0xD0u, (__int64)&v186, v161, v162, v151, *(_OWORD *)&v168, *(_OWORD *)&v160, v150);
  }
  v66 = v88;
  v90 = v166;
  LODWORD(v162) = v89;
  *(_QWORD *)v166 = v66;
  *(_DWORD *)(v90 + 8) = 1;
LABEL_67:
  v67 = *(_DWORD *)(a2 + 24);
  if ( v67 > 239 )
  {
    if ( (unsigned int)(v67 - 242) <= 1 )
      goto LABEL_95;
  }
  else if ( v67 > 237 || (unsigned int)(v67 - 101) <= 0x2F )
  {
    goto LABEL_95;
  }
  v68 = *(__int64 (**)())(*a1 + 1264);
  if ( v68 == sub_2FE33B0
    || !((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64, _QWORD))v68)(
          a1,
          v188.m128i_u32[0],
          v188.m128i_i64[1],
          v189.m128i_u32[0],
          v189.m128i_i64[1],
          0) )
  {
    *(_QWORD *)&v69 = sub_33FAF80((__int64)a5, 226, (__int64)&v186, v189.m128i_u32[0], v189.m128i_i64[1], v64, a6);
    v166 = v69;
    sub_3406EB0(
      a5,
      0x61u,
      (__int64)&v186,
      v188.m128i_u32[0],
      v188.m128i_i64[1],
      v70,
      *(_OWORD *)&v168,
      *(_OWORD *)&v160);
    *(_QWORD *)&v72 = sub_33FAF80((__int64)a5, 226, (__int64)&v186, v189.m128i_u32[0], v189.m128i_i64[1], v71, a6);
    v169 = v72;
    *(_QWORD *)&v73 = sub_34007B0((__int64)a5, v167, (__int64)&v186, v189.m128i_u32[0], v189.m128i_i64[1], 0, a6, 0);
    v180 = sub_3406EB0(a5, 0xBCu, (__int64)&v186, v189.m128i_u32[0], v189.m128i_i64[1], v74, v169, v73);
    *(_QWORD *)&v169 = v180;
    v181 = v75;
    *((_QWORD *)&v169 + 1) = (unsigned int)v75 | *((_QWORD *)&v169 + 1) & 0xFFFFFFFF00000000LL;
    v76 = sub_33FB620((__int64)a5, v66, v162, (__int64)&v186, (unsigned int)v163, v165, a6, *(_OWORD *)&v189);
    v77 = (int)a5;
    v35 = 1;
    v79 = sub_3288B20(v77, (int)&v186, v189.m128i_i32[0], v189.m128i_i32[2], (__int64)v76, v78, v166, v169, 0);
    v80 = v170;
    v178 = v79;
    v179 = v81;
    *v170 = v79;
    *((_DWORD *)v80 + 2) = v179;
    goto LABEL_51;
  }
LABEL_95:
  *(_QWORD *)&v105 = sub_33FE730((__int64)a5, (__int64)&v186, v188.m128i_u32[0], v188.m128i_i64[1], 0, (__m128i)0LL);
  v106 = v162;
  v161 = v105;
  v107 = v188.m128i_i32[0];
  v108 = v188.m128i_i64[1];
  *(_QWORD *)&v105 = *(_QWORD *)(v66 + 48) + 16LL * (unsigned int)v162;
  v109 = *(_WORD *)v105;
  *(_QWORD *)&v105 = *(_QWORD *)(v105 + 8);
  v200.m128i_i16[0] = v109;
  v200.m128i_i64[1] = v105;
  v110 = __PAIR128__((unsigned int)v162, v66);
  if ( v109 )
  {
    v111 = ((unsigned __int16)(v109 - 17) < 0xD4u) + 205;
  }
  else
  {
    v158 = v188.m128i_i64[1];
    v159 = v188.m128i_i32[0];
    *((_QWORD *)&v162 + 1) = (unsigned int)v162;
    *(_QWORD *)&v162 = v66;
    v149 = sub_30070B0(v169);
    v108 = v158;
    v107 = v159;
    v110 = v162;
    v111 = 205 - (!v149 - 1);
  }
  v112.m128i_i64[0] = sub_340EC60(
                        a5,
                        v111,
                        (__int64)&v186,
                        v107,
                        v108,
                        0,
                        v110,
                        *((__int64 *)&v110 + 1),
                        v161,
                        *(_OWORD *)&v160);
  v162 = (unsigned __int128)v112;
  v113 = sub_33FB620((__int64)a5, v66, v106, (__int64)&v186, (unsigned int)v163, v165, (__m128i)0LL, *(_OWORD *)&v189);
  v115 = v114;
  *(_QWORD *)&v116 = sub_34007B0(
                       (__int64)a5,
                       v167,
                       (__int64)&v186,
                       v189.m128i_u32[0],
                       v189.m128i_i64[1],
                       0,
                       (__m128i)0LL,
                       0);
  v165 = v116;
  *(_QWORD *)&v117 = sub_3400BD0(
                       (__int64)a5,
                       0,
                       (__int64)&v186,
                       v189.m128i_u32[0],
                       v189.m128i_i64[1],
                       0,
                       (__m128i)0LL,
                       0);
  v118 = (__int64)v113;
  v167 = v117;
  v120 = v189.m128i_i64[1];
  v119 = v189.m128i_i32[0];
  *(_QWORD *)&v117 = v115;
  v121 = *((_QWORD *)v113 + 6) + 16LL * v115;
  *((_QWORD *)&v117 + 1) = *(_QWORD *)(v121 + 8);
  v122 = v117;
  v123 = *(_WORD *)v121;
  v200.m128i_i16[0] = v123;
  v200.m128i_i64[1] = *((_QWORD *)&v117 + 1);
  if ( v123 )
  {
    v124 = (unsigned __int16)(v123 - 17) <= 0xD3u;
  }
  else
  {
    v160.m128i_i64[0] = v189.m128i_i64[1];
    *(_QWORD *)&v161 = v189.m128i_i64[0];
    v164 = v122;
    v163 = (__int64)v113;
    v124 = sub_30070B0(v169);
    v120 = v160.m128i_i64[0];
    v119 = v161;
    v118 = v163;
    v122 = v164;
  }
  v126 = sub_340EC60(a5, 205 - ((unsigned int)!v124 - 1), (__int64)&v186, v119, v120, 0, v118, v122, v167, v165);
  v128 = v127;
  v129 = *(_DWORD *)(a2 + 24);
  if ( v129 <= 239 )
  {
    if ( v129 <= 237 && (unsigned int)(v129 - 101) > 0x2F )
      goto LABEL_102;
LABEL_107:
    v139 = _mm_load_si128(&v168);
    v140 = _mm_loadu_si128(&v188);
    v141 = _mm_loadu_si128((const __m128i *)v166);
    *((_QWORD *)&v157 + 1) = 3;
    v198 = 1;
    *(_QWORD *)&v157 = v169;
    v167 = (__int128)v141;
    v200 = v141;
    si128 = _mm_load_si128((const __m128i *)&v162);
    v165 = 0u;
    v201 = v139;
    v202 = si128;
    v197 = v140;
    v199 = 0;
    v143.m128i_i64[0] = (__int64)sub_3411BE0(a5, 0x66u, (__int64)&v186, (unsigned __int16 *)&v197, 2, v125, v157);
    v144 = _mm_loadu_si128(&v189);
    v200.m128i_i64[0] = v143.m128i_i64[0];
    v201 = v143;
    *((_QWORD *)&v153 + 1) = 2;
    *(_QWORD *)&v153 = v169;
    v198 = 1;
    v200.m128i_i32[2] = 1;
    v199 = 0;
    v197 = v144;
    v131 = sub_3411BE0(a5, 0x8Du, (__int64)&v186, (unsigned __int16 *)&v197, 2, v145, v153);
    v146 = v166;
    v177 = v147;
    v176 = v131;
    v148 = *((_QWORD *)&v165 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)v166 = v131;
    *(_DWORD *)(v146 + 8) = 1;
    v134 = (unsigned int)v147 | v148;
    goto LABEL_103;
  }
  if ( (unsigned int)(v129 - 242) <= 1 )
    goto LABEL_107;
LABEL_102:
  v169 = 0u;
  sub_3406EB0(a5, 0x61u, (__int64)&v186, v188.m128i_u32[0], v188.m128i_i64[1], v125, *(_OWORD *)&v168, v162);
  v131 = sub_33FAF80((__int64)a5, 226, (__int64)&v186, v189.m128i_u32[0], v189.m128i_i64[1], v130, (__m128i)0LL);
  v175 = v133;
  v174 = v131;
  v134 = (unsigned int)v133 | *((_QWORD *)&v169 + 1) & 0xFFFFFFFF00000000LL;
LABEL_103:
  *((_QWORD *)&v156 + 1) = v128;
  v135 = a5;
  *(_QWORD *)&v156 = v126;
  v35 = 1;
  *((_QWORD *)&v152 + 1) = v134;
  *(_QWORD *)&v152 = v131;
  v136 = sub_3406EB0(v135, 0xBCu, (__int64)&v186, v189.m128i_u32[0], v189.m128i_i64[1], v132, v152, v156);
  v137 = v170;
  v172 = v136;
  v173 = v138;
  *v170 = v136;
  *((_DWORD *)v137 + 2) = v173;
LABEL_51:
  if ( v191 > 0x40 && v190 )
    j_j___libc_free_0_0(v190);
  sub_91D830(v171);
LABEL_19:
  if ( v186 )
    sub_B91220((__int64)&v186, v186);
  return v35;
}
