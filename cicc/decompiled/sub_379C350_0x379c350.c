// Function: sub_379C350
// Address: 0x379c350
//
unsigned __int8 *__fastcall sub_379C350(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v4; // eax
  __int64 v5; // rsi
  __int64 v6; // r9
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v8; // rax
  unsigned __int16 v9; // si
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int32 v16; // eax
  __int64 v17; // rdx
  __m128i v18; // xmm0
  __int64 v19; // rbx
  unsigned int v20; // r15d
  int v21; // eax
  __int64 v22; // r13
  unsigned int v23; // r12d
  unsigned int v24; // r15d
  __int64 v25; // r12
  unsigned int v26; // ebx
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rdx
  __int128 v32; // rax
  __int128 v33; // rax
  __int64 v34; // r13
  __int64 (*v35)(void); // rdx
  unsigned __int16 v36; // ax
  __int128 v37; // rax
  unsigned __int8 *v38; // r12
  __int64 *v39; // r15
  unsigned __int16 v40; // ax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int16 *v49; // rdx
  __int16 v50; // ax
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int128 v55; // rax
  unsigned int *v57; // rax
  unsigned int *v58; // rdx
  unsigned int *i; // rdx
  unsigned int v60; // r10d
  int v61; // r11d
  _QWORD *v62; // r12
  __int128 v63; // rax
  __int64 v64; // r9
  unsigned __int8 *v65; // rax
  __int64 v66; // r9
  __int64 v67; // rdx
  __int64 v68; // r13
  unsigned __int8 *v69; // r12
  __int128 v70; // rax
  __int64 v71; // r9
  __int128 v72; // rax
  unsigned int *v73; // rax
  unsigned int v74; // edx
  unsigned int v75; // r13d
  __int64 v76; // r15
  __int64 *v77; // r12
  unsigned __int16 v78; // ax
  __int64 v79; // rdx
  __int64 v80; // r10
  unsigned int v81; // r12d
  __int64 v82; // r15
  __int128 v83; // rax
  __int64 v84; // r9
  __int128 v85; // rax
  __int64 v86; // r10
  __int128 v87; // rax
  __int64 v88; // r9
  __int128 v89; // rax
  unsigned __int8 *v90; // rax
  unsigned int *v91; // r15
  unsigned int v92; // edx
  _QWORD *v93; // r13
  unsigned int v94; // eax
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rdx
  __int128 v99; // [rsp-20h] [rbp-270h]
  __m128i v100; // [rsp+0h] [rbp-250h]
  unsigned __int64 v101; // [rsp+20h] [rbp-230h]
  __int64 v102; // [rsp+28h] [rbp-228h]
  __int64 v103; // [rsp+30h] [rbp-220h]
  __int64 v104; // [rsp+30h] [rbp-220h]
  __int64 v105; // [rsp+30h] [rbp-220h]
  int v106; // [rsp+38h] [rbp-218h]
  __int16 v107; // [rsp+3Eh] [rbp-212h]
  __int128 v108; // [rsp+40h] [rbp-210h]
  __int128 v109; // [rsp+50h] [rbp-200h]
  int v110; // [rsp+60h] [rbp-1F0h]
  unsigned int v111; // [rsp+64h] [rbp-1ECh]
  unsigned int v112; // [rsp+68h] [rbp-1E8h]
  __int64 v113; // [rsp+68h] [rbp-1E8h]
  _QWORD *v114; // [rsp+68h] [rbp-1E8h]
  unsigned int v115; // [rsp+70h] [rbp-1E0h]
  __int128 v116; // [rsp+70h] [rbp-1E0h]
  int v117; // [rsp+70h] [rbp-1E0h]
  int v118; // [rsp+70h] [rbp-1E0h]
  _QWORD *v119; // [rsp+70h] [rbp-1E0h]
  __int128 v120; // [rsp+70h] [rbp-1E0h]
  __int128 v121; // [rsp+80h] [rbp-1D0h]
  unsigned int v122; // [rsp+80h] [rbp-1D0h]
  __int64 v123; // [rsp+80h] [rbp-1D0h]
  _QWORD *v124; // [rsp+80h] [rbp-1D0h]
  __int64 v125; // [rsp+90h] [rbp-1C0h]
  __int128 v126; // [rsp+90h] [rbp-1C0h]
  __int128 v127; // [rsp+90h] [rbp-1C0h]
  unsigned int v128; // [rsp+90h] [rbp-1C0h]
  unsigned __int8 *v129; // [rsp+B0h] [rbp-1A0h]
  __int64 v130; // [rsp+C0h] [rbp-190h]
  __int64 v131; // [rsp+C0h] [rbp-190h]
  __int64 v132; // [rsp+C8h] [rbp-188h]
  __int64 v133; // [rsp+D0h] [rbp-180h]
  __int64 v134; // [rsp+E0h] [rbp-170h] BYREF
  int v135; // [rsp+E8h] [rbp-168h]
  __m128i v136; // [rsp+F0h] [rbp-160h] BYREF
  __m128i v137; // [rsp+100h] [rbp-150h] BYREF
  unsigned int *v138; // [rsp+110h] [rbp-140h] BYREF
  __int64 v139; // [rsp+118h] [rbp-138h]
  _QWORD v140[38]; // [rsp+120h] [rbp-130h] BYREF

  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 80);
  v111 = v4;
  v134 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v134, v5, 1);
  v6 = *a1;
  v135 = *(_DWORD *)(a2 + 72);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v11 = a1[1];
  if ( v7 == sub_2D56A50 )
  {
    v12 = v9;
    v13 = v6;
    sub_2FE6CC0((__int64)&v138, v6, *(_QWORD *)(v11 + 64), v12, v10);
    LOWORD(v16) = v139;
    v17 = v140[0];
    v136.m128i_i16[0] = v139;
    v136.m128i_i64[1] = v140[0];
  }
  else
  {
    v98 = v9;
    v13 = *(_QWORD *)(v11 + 64);
    v16 = v7(v6, v13, v98, v10);
    v136.m128i_i32[0] = v16;
    v136.m128i_i64[1] = v17;
  }
  if ( (_WORD)v16 )
  {
    v102 = 0;
    v107 = word_4456580[(unsigned __int16)v16 - 1];
  }
  else
  {
    v52 = sub_3009970((__int64)&v136, v13, v17, v14, v15);
    v107 = v52;
    v2 = v52;
    v102 = v53;
  }
  LOWORD(v2) = v107;
  v18 = _mm_loadu_si128(&v136);
  v103 = v2;
  v19 = v136.m128i_u16[0];
  v137 = v18;
  if ( v136.m128i_i16[0] )
    v20 = word_4456340[v136.m128i_u16[0] - 1];
  else
    v20 = sub_3007240((__int64)&v137);
  v21 = *(_DWORD *)(a2 + 28);
  v22 = v103;
  v125 = a2;
  v23 = v20;
  v110 = v21;
  while ( 1 )
  {
    if ( (_WORD)v19 && *(_QWORD *)(*a1 + 8 * v19 + 112) )
    {
      v24 = v23;
      v104 = v22;
      v25 = v125;
      if ( v24 == 1 )
        goto LABEL_72;
      v13 = *(unsigned int *)(v125 + 24);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)*a1 + 632LL))(
              *a1,
              v13,
              v137.m128i_u32[0],
              v137.m128i_i64[1]) )
      {
        *(_QWORD *)&v127 = sub_379AB60((__int64)a1, **(_QWORD **)(v125 + 40), *(_QWORD *)(*(_QWORD *)(v125 + 40) + 8LL));
        *((_QWORD *)&v127 + 1) = v54;
        *(_QWORD *)&v55 = sub_379AB60(
                            (__int64)a1,
                            *(_QWORD *)(*(_QWORD *)(v25 + 40) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(v25 + 40) + 48LL));
        v38 = sub_3405C90(
                (_QWORD *)a1[1],
                *(_DWORD *)(v25 + 24),
                (__int64)&v134,
                v136.m128i_u32[0],
                v136.m128i_i64[1],
                v110,
                v18,
                v127,
                v55);
        goto LABEL_37;
      }
      v130 = sub_33CB7C0(v111);
      if ( BYTE4(v130) )
      {
        v26 = v130;
        goto LABEL_16;
      }
      goto LABEL_32;
    }
    if ( v23 == 1 )
      break;
    LOWORD(v22) = v107;
    v23 >>= 1;
    v13 = v23;
    v39 = *(__int64 **)(a1[1] + 64);
    v40 = sub_2D43050(v107, v23);
    v41 = 0;
    v19 = v40;
    if ( !v40 )
    {
      v13 = (unsigned int)v22;
      v19 = (unsigned __int16)sub_3009400(v39, (unsigned int)v22, v102, v23, 0);
    }
    v137.m128i_i16[0] = v19;
    v137.m128i_i64[1] = v41;
  }
  v104 = v22;
  v25 = v125;
LABEL_72:
  v131 = sub_33CB7C0(v111);
  if ( !BYTE4(v131) )
    goto LABEL_74;
  v26 = v131;
  v24 = 1;
LABEL_16:
  v27 = *a1;
  if ( (v136.m128i_i16[0] == 1 || v136.m128i_i16[0] && *(_QWORD *)(v27 + 8LL * v136.m128i_u16[0] + 112))
    && (v26 > 0x1F3 || (*(_BYTE *)(v26 + v27 + 500LL * v136.m128i_u16[0] + 6414) & 0xFB) == 0) )
  {
    v13 = 2;
    v132 = sub_3281590((__int64)&v136);
    v28 = sub_327FD70(*(__int64 **)(a1[1] + 64), 2u, 0, v132);
    v30 = v29;
    if ( (_WORD)v28 )
    {
      v115 = v28;
      if ( *(_QWORD *)(*a1 + 8LL * (unsigned __int16)v28 + 112) )
      {
        *(_QWORD *)&v126 = sub_379AB60((__int64)a1, **(_QWORD **)(v25 + 40), *(_QWORD *)(*(_QWORD *)(v25 + 40) + 8LL));
        *((_QWORD *)&v126 + 1) = v31;
        *(_QWORD *)&v32 = sub_379AB60(
                            (__int64)a1,
                            *(_QWORD *)(*(_QWORD *)(v25 + 40) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(v25 + 40) + 48LL));
        v121 = v32;
        *(_QWORD *)&v33 = sub_34015B0(a1[1], (__int64)&v134, v115, v30, 0, 0, v18);
        v34 = a1[1];
        v116 = v33;
        *(_QWORD *)&v33 = *(_QWORD *)(v25 + 48);
        WORD4(v33) = *(_WORD *)v33;
        *(_QWORD *)&v33 = *(_QWORD *)(v33 + 8);
        LOWORD(v138) = WORD4(v33);
        v139 = v33;
        v133 = sub_3281590((__int64)&v138);
        v35 = *(__int64 (**)(void))(*(_QWORD *)*a1 + 80LL);
        v36 = 7;
        if ( v35 != sub_2FE2E20 )
          v36 = v35();
        *(_QWORD *)&v37 = sub_3401C20(v34, (__int64)&v134, v36, 0, v133, v18);
        v38 = sub_33FC0E0(
                (_QWORD *)a1[1],
                v26,
                (__int64)&v134,
                v136.m128i_u32[0],
                v136.m128i_i64[1],
                v110,
                v126,
                v121,
                v116,
                v37);
        goto LABEL_37;
      }
    }
  }
  if ( v24 == 1 )
  {
LABEL_74:
    v93 = (_QWORD *)a1[1];
    v94 = sub_3281500(&v136, v13);
    v38 = sub_3412A00(v93, v25, v94, v95, v96, v97, v18);
    goto LABEL_37;
  }
LABEL_32:
  v100 = _mm_loadu_si128(&v137);
  *(_QWORD *)&v109 = sub_379AB60((__int64)a1, **(_QWORD **)(v25 + 40), *(_QWORD *)(*(_QWORD *)(v25 + 40) + 8LL));
  v42 = *(_QWORD *)(v25 + 40);
  *((_QWORD *)&v109 + 1) = v43;
  v44 = *(_QWORD *)(v42 + 40);
  v45 = sub_379AB60((__int64)a1, v44, *(_QWORD *)(v42 + 48));
  *((_QWORD *)&v108 + 1) = v48;
  v49 = *(__int16 **)(v25 + 48);
  *(_QWORD *)&v108 = v45;
  v50 = *v49;
  v51 = *((_QWORD *)v49 + 1);
  LOWORD(v138) = v50;
  v139 = v51;
  if ( v50 )
  {
    if ( (unsigned __int16)(v50 - 176) > 0x34u )
    {
LABEL_34:
      v122 = word_4456340[(unsigned __int16)v138 - 1];
      goto LABEL_42;
    }
  }
  else if ( !sub_3007100((__int64)&v138) )
  {
    goto LABEL_41;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v138 )
  {
    if ( (unsigned __int16)((_WORD)v138 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_34;
  }
LABEL_41:
  v122 = sub_3007130((__int64)&v138, v44);
LABEL_42:
  v57 = (unsigned int *)v140;
  v58 = (unsigned int *)v140;
  v138 = (unsigned int *)v140;
  v139 = 0x1000000000LL;
  if ( v122 )
  {
    if ( v122 > 0x10uLL )
    {
      sub_C8D5F0((__int64)&v138, v140, v122, 0x10u, v46, v47);
      v58 = v138;
      v57 = &v138[4 * (unsigned int)v139];
    }
    for ( i = &v58[4 * v122]; i != v57; v57 += 4 )
    {
      if ( v57 )
      {
        *(_QWORD *)v57 = 0;
        v57[2] = 0;
      }
    }
    LODWORD(v139) = v122;
  }
  else
  {
    v122 = 0;
  }
  v117 = 0;
  v128 = 0;
  while ( 2 )
  {
    v60 = v122;
    if ( !v122 )
      goto LABEL_68;
    if ( v24 > v122 )
    {
      v75 = v24;
      v76 = v104;
    }
    else
    {
      v61 = v117;
      do
      {
        v62 = (_QWORD *)a1[1];
        v112 = v60;
        v118 = v61;
        v123 = v61;
        *(_QWORD *)&v63 = sub_3400EE0((__int64)v62, v61, (__int64)&v134, 0, v18);
        v65 = sub_3406EB0(v62, 0xA1u, (__int64)&v134, v137.m128i_u32[0], v137.m128i_i64[1], v64, v109, v63);
        v66 = v123;
        v68 = v67;
        v69 = v65;
        v124 = (_QWORD *)a1[1];
        *(_QWORD *)&v70 = sub_3400EE0((__int64)v124, v66, (__int64)&v134, 0, v18);
        *(_QWORD *)&v72 = sub_3406EB0(v124, 0xA1u, (__int64)&v134, v137.m128i_u32[0], v137.m128i_i64[1], v71, v108, v70);
        *((_QWORD *)&v99 + 1) = v68;
        *(_QWORD *)&v99 = v69;
        v129 = sub_3405C90(
                 (_QWORD *)a1[1],
                 v111,
                 (__int64)&v134,
                 v137.m128i_u32[0],
                 v137.m128i_i64[1],
                 v110,
                 v18,
                 v99,
                 v72);
        v60 = v112 - v24;
        v73 = &v138[4 * v128];
        v61 = v24 + v118;
        *(_QWORD *)v73 = v129;
        ++v128;
        v73[2] = v74;
      }
      while ( v24 <= v112 - v24 );
      v122 = v112 - v24;
      v75 = v24;
      v76 = v104;
      v117 = v61;
    }
    while ( 1 )
    {
      LOWORD(v76) = v107;
      v75 >>= 1;
      v77 = *(__int64 **)(a1[1] + 64);
      v78 = sub_2D43050(v107, v75);
      if ( v78 )
      {
        v137.m128i_i16[0] = v78;
        v137.m128i_i64[1] = 0;
      }
      else
      {
        v78 = sub_3009400(v77, (unsigned int)v76, v102, v75, 0);
        v137.m128i_i16[0] = v78;
        v137.m128i_i64[1] = v79;
        if ( !v78 )
          goto LABEL_62;
      }
      if ( *(_QWORD *)(*a1 + 8LL * v78 + 112) )
        break;
LABEL_62:
      if ( v75 == 1 )
      {
        WORD1(v104) = WORD1(v76);
        goto LABEL_64;
      }
    }
    v104 = v76;
    v24 = v75;
    if ( v75 != 1 )
      continue;
    break;
  }
LABEL_64:
  if ( v122 )
  {
    v80 = v117;
    HIWORD(v81) = WORD1(v104);
    v101 = v117 + (unsigned __int64)(v122 - 1) + 1;
    v106 = v128 - v117;
    do
    {
      v113 = v80;
      v82 = (unsigned int)(v106 + v80);
      v119 = (_QWORD *)a1[1];
      *(_QWORD *)&v83 = sub_3400EE0((__int64)v119, v80, (__int64)&v134, 0, v18);
      LOWORD(v81) = v107;
      *(_QWORD *)&v85 = sub_3406EB0(v119, 0x9Eu, (__int64)&v134, v81, v102, v84, v109, v83);
      v86 = v113;
      v114 = (_QWORD *)a1[1];
      v105 = v86;
      v120 = v85;
      *(_QWORD *)&v87 = sub_3400EE0((__int64)v114, v86, (__int64)&v134, 0, v18);
      *(_QWORD *)&v89 = sub_3406EB0(v114, 0x9Eu, (__int64)&v134, v81, v102, v88, v108, v87);
      v90 = sub_3405C90((_QWORD *)a1[1], v111, (__int64)&v134, v81, v102, v110, v18, v120, v89);
      v91 = &v138[4 * v82];
      v80 = v105 + 1;
      *(_QWORD *)v91 = v90;
      v91[2] = v92;
    }
    while ( v101 != v105 + 1 );
    v128 += v122;
  }
LABEL_68:
  v38 = sub_37753B0(
          (_QWORD *)a1[1],
          *a1,
          &v138,
          v128,
          v137.m128i_u32[0],
          v137.m128i_i64[1],
          v18,
          v100.m128i_i64[0],
          v100.m128i_i64[1],
          v136.m128i_u32[0],
          v136.m128i_i64[1]);
  if ( v138 != (unsigned int *)v140 )
    _libc_free((unsigned __int64)v138);
LABEL_37:
  if ( v134 )
    sub_B91220((__int64)&v134, v134);
  return v38;
}
