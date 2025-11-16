// Function: sub_383FAD0
// Address: 0x383fad0
//
__int64 __fastcall sub_383FAD0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // r14
  int v6; // edi
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // edi
  __int64 v12; // rdx
  __int64 v13; // rsi
  int v14; // r13d
  unsigned __int16 *v15; // rax
  int v16; // r14d
  __int64 v17; // r15
  unsigned int v18; // r15d
  __int64 v19; // rdx
  unsigned __int8 *v20; // rax
  __int64 v21; // rsi
  __int32 v22; // edx
  __int32 v23; // edx
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // r12
  __int64 v28; // r12
  unsigned int v29; // r13d
  __int64 v30; // r12
  unsigned __int64 v31; // rax
  __int128 v32; // rax
  __int128 v33; // rax
  __int64 v34; // r9
  __int128 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rax
  unsigned int v38; // edx
  __int64 v39; // r9
  __int64 result; // rax
  __int64 v41; // rax
  __int64 v42; // rsi
  __int32 v43; // edx
  __int32 v44; // edx
  __int64 v45; // rax
  __int64 v46; // rax
  int v47; // eax
  __int128 v48; // rax
  _QWORD *v49; // rdi
  int v50; // r15d
  __int64 v51; // rbx
  __int64 v52; // r9
  unsigned __int8 *v53; // rax
  __int32 v54; // edx
  int v55; // r9d
  __int32 v56; // r14d
  unsigned int v57; // eax
  __m128i v58; // xmm2
  __m128i v59; // xmm3
  __int64 v60; // r9
  unsigned __int8 *v61; // rax
  __int64 v62; // r14
  unsigned __int64 v63; // r13
  int v64; // edx
  unsigned int v65; // eax
  __m128i v66; // xmm4
  __m128i v67; // xmm5
  __int64 v68; // r9
  __int64 v69; // r13
  unsigned __int16 *v70; // rax
  __int64 v71; // r12
  unsigned int v72; // r15d
  __int32 v73; // edx
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // rdx
  __int64 v78; // rax
  __int64 v79; // rdx
  unsigned __int16 v80; // r13
  __int64 v81; // r14
  __int64 (__fastcall *v82)(__int64, __int64, unsigned int, __int64); // rax
  __int64 (*v83)(); // rax
  unsigned __int8 *v84; // rax
  __int64 v85; // rsi
  __int32 v86; // edx
  __int32 v87; // edx
  unsigned __int64 v88; // rax
  unsigned __int8 *v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r15
  unsigned __int8 *v92; // r14
  __int64 v93; // r9
  __int128 v94; // rax
  __int64 v95; // r9
  unsigned __int64 v96; // r13
  __int32 v97; // r15d
  __int32 v98; // r12d
  __int64 v99; // rax
  __int64 v100; // r14
  unsigned __int16 v101; // bx
  unsigned int v102; // eax
  __m128i v103; // xmm6
  __m128i v104; // xmm7
  __int64 v105; // r9
  __int64 v106; // rdx
  __int64 v107; // rcx
  __int32 v108; // edx
  __int32 v109; // edx
  __int64 v110; // r9
  __int128 v111; // rax
  __int64 v112; // r9
  unsigned __int8 *v113; // rax
  __int64 v114; // r14
  __int32 v115; // edx
  unsigned int v116; // eax
  __m128i v117; // xmm6
  __m128i v118; // xmm7
  __int64 v119; // r9
  unsigned __int8 *v120; // rax
  __int32 v121; // edx
  __int128 v122; // [rsp-10h] [rbp-270h]
  __int128 v123; // [rsp-10h] [rbp-270h]
  __int128 v124; // [rsp+0h] [rbp-260h]
  __int128 v125; // [rsp+0h] [rbp-260h]
  __int128 v126; // [rsp+0h] [rbp-260h]
  __int128 v127; // [rsp+0h] [rbp-260h]
  int v128; // [rsp+1Ch] [rbp-244h]
  unsigned __int64 v129; // [rsp+20h] [rbp-240h]
  __int32 v130; // [rsp+28h] [rbp-238h]
  unsigned __int64 v131; // [rsp+28h] [rbp-238h]
  __int64 v132; // [rsp+30h] [rbp-230h]
  __int32 v133; // [rsp+30h] [rbp-230h]
  __int64 v134; // [rsp+38h] [rbp-228h]
  __int64 v135; // [rsp+40h] [rbp-220h]
  __int128 v136; // [rsp+40h] [rbp-220h]
  __int64 v137; // [rsp+40h] [rbp-220h]
  __int64 v138; // [rsp+40h] [rbp-220h]
  unsigned int v139; // [rsp+50h] [rbp-210h]
  __int128 v140; // [rsp+50h] [rbp-210h]
  __int64 v141; // [rsp+50h] [rbp-210h]
  __int64 v142; // [rsp+50h] [rbp-210h]
  __int64 v143; // [rsp+50h] [rbp-210h]
  __int64 v144; // [rsp+50h] [rbp-210h]
  __int64 v145; // [rsp+50h] [rbp-210h]
  __int64 v146; // [rsp+50h] [rbp-210h]
  int v147; // [rsp+58h] [rbp-208h]
  __int64 v148; // [rsp+A0h] [rbp-1C0h]
  __int64 v149; // [rsp+150h] [rbp-110h] BYREF
  int v150; // [rsp+158h] [rbp-108h]
  __m128i v151; // [rsp+160h] [rbp-100h] BYREF
  __m128i v152; // [rsp+170h] [rbp-F0h] BYREF
  __int64 v153; // [rsp+180h] [rbp-E0h] BYREF
  __int64 v154; // [rsp+188h] [rbp-D8h]
  __int64 v155[2]; // [rsp+190h] [rbp-D0h] BYREF
  __int64 v156; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v157; // [rsp+1A8h] [rbp-B8h]
  __m128i v158; // [rsp+1B0h] [rbp-B0h] BYREF
  __m128i v159; // [rsp+1C0h] [rbp-A0h] BYREF
  __m128i v160; // [rsp+1D0h] [rbp-90h] BYREF
  __int64 v161; // [rsp+1E0h] [rbp-80h]
  unsigned __int64 v162; // [rsp+1F0h] [rbp-70h] BYREF
  __int64 v163; // [rsp+1F8h] [rbp-68h]
  __int64 v164; // [rsp+200h] [rbp-60h]
  __int32 v165; // [rsp+208h] [rbp-58h]
  __m128i v166; // [rsp+210h] [rbp-50h]
  __m128i v167; // [rsp+220h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 80);
  v149 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v149, v4, 1);
  v5 = a1[1];
  v6 = *(_DWORD *)(a2 + 24);
  v150 = *(_DWORD *)(a2 + 72);
  v7 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v8 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
  v158.m128i_i64[0] = v5;
  v9 = *a1;
  v161 = a2;
  v159.m128i_i64[0] = 0;
  v158.m128i_i64[1] = v9;
  v159.m128i_i32[2] = 0;
  v160.m128i_i64[0] = 0;
  v160.m128i_i32[2] = 0;
  v151 = v7;
  v152 = v8;
  v156 = sub_33CB160(v6);
  if ( BYTE4(v156) )
  {
    v10 = *(_QWORD *)(v161 + 40) + 40LL * (unsigned int)v156;
    v159.m128i_i64[0] = *(_QWORD *)v10;
    v159.m128i_i32[2] = *(_DWORD *)(v10 + 8);
    v11 = *(_DWORD *)(v161 + 24);
  }
  else
  {
    v69 = v161;
    v11 = *(_DWORD *)(v161 + 24);
    if ( v11 == 488 )
    {
      v70 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v161 + 40) + 48LL)
                               + 16LL * *(unsigned int *)(*(_QWORD *)(v161 + 40) + 8LL));
      v71 = *((_QWORD *)v70 + 1);
      v72 = *v70;
      v162 = *(_QWORD *)(v161 + 80);
      if ( v162 )
        sub_3813810((__int64 *)&v162);
      LODWORD(v163) = *(_DWORD *)(v69 + 72);
      v159.m128i_i64[0] = (__int64)sub_34015B0(v5, (__int64)&v162, v72, v71, 0, 0, v7);
      v159.m128i_i32[2] = v73;
      sub_9C6650(&v162);
      v11 = *(_DWORD *)(v161 + 24);
    }
  }
  v162 = sub_33CB1F0(v11);
  if ( BYTE4(v162) )
  {
    v12 = *(_QWORD *)(v161 + 40) + 40LL * (unsigned int)v162;
    v160.m128i_i64[0] = *(_QWORD *)v12;
    v160.m128i_i32[2] = *(_DWORD *)(v12 + 8);
  }
  v13 = ((unsigned __int8)(*(_DWORD *)(v161 + 28) >> 12) ^ 1) & 1;
  v162 = sub_33CB280(*(_DWORD *)(v161 + 24));
  v14 = v162;
  v15 = (unsigned __int16 *)(*(_QWORD *)(v151.m128i_i64[0] + 48) + 16LL * v151.m128i_u32[2]);
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  LOWORD(v156) = v16;
  v157 = v17;
  if ( (_WORD)v16 )
  {
    if ( (unsigned __int16)(v16 - 17) <= 0xD3u )
    {
      v17 = 0;
      LOWORD(v16) = word_4456580[v16 - 1];
    }
  }
  else if ( sub_30070B0((__int64)&v156) )
  {
    LOWORD(v16) = sub_3009970((__int64)&v156, v13, v74, v75, v76);
    v17 = v77;
  }
  LOWORD(v155[0]) = v16;
  v155[1] = v17;
  v162 = sub_2D5B750((unsigned __int16 *)v155);
  v18 = v162;
  v163 = v19;
  if ( v14 == 85 )
  {
    sub_383E4F0(a1, (__int64)&v151, (__int64)&v152, v7);
    v96 = v151.m128i_i64[0];
    v97 = v152.m128i_i32[2];
    v98 = v151.m128i_i32[2];
    v99 = *(_QWORD *)(v151.m128i_i64[0] + 48) + 16LL * v151.m128i_u32[2];
    v100 = *(_QWORD *)(v99 + 8);
    v101 = *(_WORD *)v99;
    v145 = v152.m128i_i64[0];
    v102 = sub_33CB7C0(85);
    LODWORD(v163) = v98;
    v103 = _mm_loadu_si128(&v159);
    v164 = v145;
    *((_QWORD *)&v126 + 1) = 4;
    v104 = _mm_loadu_si128(&v160);
    *(_QWORD *)&v126 = &v162;
    v162 = v96;
    v165 = v97;
    v166 = v103;
    v167 = v104;
    result = (__int64)sub_33FC220(v158.m128i_i64[0], v102, (__int64)&v149, v101, v100, v105, v126);
  }
  else if ( v14 == 83 )
  {
    v78 = *(_QWORD *)(v151.m128i_i64[0] + 48) + 16LL * v151.m128i_u32[2];
    v79 = a1[1];
    v80 = *(_WORD *)v78;
    v81 = *(_QWORD *)(v78 + 8);
    v82 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
    if ( v82 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v162, *a1, *(_QWORD *)(v79 + 64), v80, v81);
      LOWORD(v156) = v163;
      v157 = v164;
    }
    else
    {
      v135 = v80;
      LODWORD(v156) = v82(*a1, *(_QWORD *)(v79 + 64), v80, v81);
      v157 = v106;
    }
    v83 = *(__int64 (**)())(*(_QWORD *)*a1 + 1456LL);
    if ( v83 == sub_2D56680
      || (v107 = v135,
          LOWORD(v107) = v80,
          !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, _QWORD, __int64))v83)(
             *a1,
             v107,
             v81,
             (unsigned int)v156,
             v157)) )
    {
      v84 = sub_37AF270((__int64)a1, v151.m128i_u64[0], v151.m128i_i64[1], v7);
      v85 = v152.m128i_i64[0];
      v151.m128i_i64[0] = (__int64)v84;
      v151.m128i_i32[2] = v86;
      v152.m128i_i64[0] = (__int64)sub_37AF270((__int64)a1, v152.m128i_u64[0], v152.m128i_i64[1], v7);
      v152.m128i_i32[2] = v87;
      LODWORD(v163) = sub_32844A0((unsigned __int16 *)&v156, v85);
      if ( (unsigned int)v163 > 0x40 )
        sub_C43690((__int64)&v162, 0, 0);
      else
        v162 = 0;
      if ( v18 )
      {
        if ( v18 > 0x40 )
        {
          sub_C43C90(&v162, 0, v18);
        }
        else
        {
          v88 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18);
          if ( (unsigned int)v163 > 0x40 )
            *(_QWORD *)v162 |= v88;
          else
            v162 |= v88;
        }
      }
      v89 = sub_34007B0(a1[1], (__int64)&v162, (__int64)&v149, v156, v157, 0, v7, 0);
      v91 = v90;
      v92 = v89;
      *(_QWORD *)&v94 = sub_328FC10(&v158, 0x38u, (int)&v149, v156, v157, v93, *(_OWORD *)&v151, *(_OWORD *)&v152);
      *((_QWORD *)&v125 + 1) = v91;
      *(_QWORD *)&v125 = v92;
      result = sub_328FC10(&v158, 0xB6u, (int)&v149, v156, v157, v95, v94, v125);
      if ( (unsigned int)v163 > 0x40 && v162 )
      {
        v144 = result;
        j_j___libc_free_0_0(v162);
        result = v144;
      }
    }
    else
    {
      v151.m128i_i64[0] = (__int64)sub_383B380((__int64)a1, v151.m128i_u64[0], v151.m128i_i64[1]);
      v151.m128i_i32[2] = v108;
      v152.m128i_i64[0] = (__int64)sub_383B380((__int64)a1, v152.m128i_u64[0], v152.m128i_i64[1]);
      v152.m128i_i32[2] = v109;
      result = sub_328FC10(&v158, 0x53u, (int)&v149, v156, v157, v110, *(_OWORD *)&v151, *(_OWORD *)&v152);
    }
  }
  else
  {
    if ( (unsigned int)(v14 - 86) <= 1 )
    {
      v41 = sub_37AE0F0((__int64)a1, v151.m128i_u64[0], v151.m128i_i64[1]);
      v42 = v152.m128i_i64[0];
      v151.m128i_i64[0] = v41;
      v151.m128i_i32[2] = v43;
      v152.m128i_i64[0] = (__int64)sub_37AF270((__int64)a1, v152.m128i_u64[0], v152.m128i_i64[1], v7);
      v152.m128i_i32[2] = v44;
      v45 = *(_QWORD *)(v151.m128i_i64[0] + 48) + 16LL * v151.m128i_u32[2];
      LOWORD(v44) = *(_WORD *)v45;
      v46 = *(_QWORD *)(v45 + 8);
      LOWORD(v153) = v44;
      v154 = v46;
      v47 = sub_32844A0((unsigned __int16 *)&v153, v42);
      *(_QWORD *)&v48 = sub_3400E40(a1[1], v47 - v18, v153, v154, (__int64)&v149, v7);
      v49 = (_QWORD *)a1[1];
      v50 = DWORD2(v48);
      v51 = v48;
      v53 = sub_3406EB0(v49, 0xBEu, (__int64)&v149, (unsigned int)v153, v154, v52, *(_OWORD *)&v151, v48);
      if ( v14 == 87 )
        v55 = 192;
      else
        v55 = 191;
      v151.m128i_i64[0] = (__int64)v53;
      v151.m128i_i32[2] = v54;
    }
    else
    {
      v20 = sub_383B380((__int64)a1, v151.m128i_u64[0], v151.m128i_i64[1]);
      v21 = v152.m128i_i64[0];
      v151.m128i_i64[0] = (__int64)v20;
      v151.m128i_i32[2] = v22;
      v152.m128i_i64[0] = (__int64)sub_383B380((__int64)a1, v152.m128i_u64[0], v152.m128i_i64[1]);
      v152.m128i_i32[2] = v23;
      v24 = *(_QWORD *)(v151.m128i_i64[0] + 48) + 16LL * v151.m128i_u32[2];
      LOWORD(v23) = *(_WORD *)v24;
      v25 = *(_QWORD *)(v24 + 8);
      LOWORD(v153) = v23;
      v154 = v25;
      v26 = sub_32844A0((unsigned __int16 *)&v153, v21);
      v27 = (unsigned __int16)v153;
      v139 = v26;
      v162 = sub_33CB7C0(v14);
      if ( (_WORD)v27 != 1 && (!(_WORD)v27 || !*(_QWORD *)(v158.m128i_i64[1] + 8LL * (unsigned __int16)v27 + 112))
        || (unsigned int)v162 > 0x1F3
        || *(_BYTE *)((unsigned int)v162 + v158.m128i_i64[1] + 500 * v27 + 6414) )
      {
        LODWORD(v163) = v18;
        v28 = 1LL << ((unsigned __int8)v18 - 1);
        v29 = (v14 != 82) + 56;
        if ( v18 > 0x40 )
        {
          sub_C43690((__int64)&v162, 0, 0);
          if ( (unsigned int)v163 <= 0x40 )
            v162 |= v28;
          else
            *(_QWORD *)(v162 + 8LL * ((v18 - 1) >> 6)) |= v28;
          v30 = ~v28;
          sub_C44830((__int64)v155, &v162, v139);
          sub_969240((__int64 *)&v162);
          LODWORD(v163) = v18;
          sub_C43690((__int64)&v162, -1, 1);
          if ( (unsigned int)v163 > 0x40 )
          {
            *(_QWORD *)(v162 + 8LL * ((v18 - 1) >> 6)) &= v30;
            goto LABEL_21;
          }
        }
        else
        {
          v162 = 1LL << ((unsigned __int8)v18 - 1);
          v30 = ~v28;
          sub_C44830((__int64)v155, &v162, v139);
          sub_969240((__int64 *)&v162);
          LODWORD(v163) = v18;
          v31 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v18 - 1) & 0x3F));
          if ( !v18 )
            v31 = 0;
          v162 = v31;
        }
        v162 &= v30;
LABEL_21:
        sub_C44830((__int64)&v156, &v162, v139);
        sub_969240((__int64 *)&v162);
        *(_QWORD *)&v32 = sub_34007B0(a1[1], (__int64)v155, (__int64)&v149, v153, v154, 0, v7, 0);
        v140 = v32;
        *(_QWORD *)&v33 = sub_34007B0(a1[1], (__int64)&v156, (__int64)&v149, v153, v154, 0, v7, 0);
        v136 = v33;
        *(_QWORD *)&v35 = sub_328FC10(&v158, v29, (int)&v149, v153, v154, v34, *(_OWORD *)&v151, *(_OWORD *)&v152);
        v134 = *((_QWORD *)&v35 + 1);
        v37 = sub_328FC10(&v158, 0xB4u, (int)&v149, v153, v154, v36, v35, v136);
        *((_QWORD *)&v122 + 1) = v38 | v134 & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v122 = v37;
        v148 = sub_328FC10(&v158, 0xB5u, (int)&v149, v153, v154, v39, v122, v140);
        sub_969240(&v156);
        sub_969240(v155);
        result = v148;
        goto LABEL_26;
      }
      if ( v14 != 84 && v14 != 82 )
        BUG();
      *(_QWORD *)&v111 = sub_3400E40(a1[1], v139 - v18, v153, v154, (__int64)&v149, v7);
      v147 = DWORD2(v111);
      v146 = v111;
      v113 = sub_3406EB0((_QWORD *)a1[1], 0xBEu, (__int64)&v149, (unsigned int)v153, v154, v112, *(_OWORD *)&v151, v111);
      v114 = v153;
      v151.m128i_i64[0] = (__int64)v113;
      v151.m128i_i32[2] = v115;
      v133 = v152.m128i_i32[2];
      v138 = v154;
      v50 = v147;
      v131 = v152.m128i_i64[0];
      v51 = v146;
      v116 = sub_33CB7C0(190);
      v117 = _mm_loadu_si128(&v159);
      v118 = _mm_loadu_si128(&v160);
      v165 = v147;
      LODWORD(v163) = v133;
      *((_QWORD *)&v127 + 1) = 4;
      *(_QWORD *)&v127 = &v162;
      v162 = v131;
      v164 = v146;
      v166 = v117;
      v167 = v118;
      v120 = sub_33FC220(v158.m128i_i64[0], v116, (__int64)&v149, v114, v138, v119, v127);
      v55 = 191;
      v152.m128i_i64[0] = (__int64)v120;
      v152.m128i_i32[2] = v121;
    }
    v128 = v55;
    v56 = v152.m128i_i32[2];
    v137 = v154;
    v129 = v151.m128i_i64[0];
    v130 = v151.m128i_i32[2];
    v132 = v152.m128i_i64[0];
    v141 = v153;
    v57 = sub_33CB7C0(v14);
    *((_QWORD *)&v124 + 1) = 4;
    *(_QWORD *)&v124 = &v162;
    v58 = _mm_loadu_si128(&v159);
    v59 = _mm_loadu_si128(&v160);
    LODWORD(v163) = v130;
    v164 = v132;
    v162 = v129;
    v166 = v58;
    v167 = v59;
    v165 = v56;
    v61 = sub_33FC220(v158.m128i_i64[0], v57, (__int64)&v149, v141, v137, v60, v124);
    v62 = v153;
    v63 = (unsigned __int64)v61;
    LODWORD(v137) = v64;
    v142 = v154;
    v65 = sub_33CB7C0(v128);
    v66 = _mm_loadu_si128(&v159);
    v162 = v63;
    LODWORD(v163) = v137;
    v67 = _mm_loadu_si128(&v160);
    *((_QWORD *)&v123 + 1) = 4;
    *(_QWORD *)&v123 = &v162;
    v164 = v51;
    v165 = v50;
    v166 = v66;
    v167 = v67;
    result = (__int64)sub_33FC220(v158.m128i_i64[0], v65, (__int64)&v149, v62, v142, v68, v123);
  }
LABEL_26:
  if ( v149 )
  {
    v143 = result;
    sub_B91220((__int64)&v149, v149);
    return v143;
  }
  return result;
}
