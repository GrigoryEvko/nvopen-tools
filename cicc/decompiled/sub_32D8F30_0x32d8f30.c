// Function: sub_32D8F30
// Address: 0x32d8f30
//
__int64 __fastcall sub_32D8F30(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  const __m128i *v5; // roff
  __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 result; // rax
  __int64 v11; // rsi
  __int64 v12; // r15
  unsigned __int16 *v13; // rdx
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __m128i v20; // xmm1
  __int64 v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rdx
  __m128i v25; // xmm2
  _DWORD *v26; // rsi
  __int64 v27; // rdx
  __int128 v28; // rax
  int v29; // r9d
  int v30; // r9d
  __int64 v31; // rcx
  __int64 v32; // rax
  __int16 v33; // dx
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rdx
  __int64 v37; // rax
  int v38; // r9d
  __int64 v39; // rdi
  int v40; // edx
  __int64 v41; // r12
  __int64 v42; // r13
  __int64 v43; // rdx
  _QWORD *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  _QWORD *v47; // rax
  __int64 v48; // rcx
  _QWORD *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  _QWORD *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdi
  __int128 v55; // rax
  __int64 v56; // r12
  unsigned int v57; // eax
  __int64 v58; // rdx
  unsigned int v59; // edx
  __int64 v60; // rsi
  __int64 v61; // rsi
  __int64 v62; // rax
  __int16 v63; // cx
  __int64 v64; // rax
  int v65; // eax
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // r14
  __int64 v69; // rdx
  __int64 v70; // r13
  __int64 v71; // r12
  __int128 v72; // rax
  int v73; // r9d
  unsigned int v74; // edx
  int v75; // r9d
  __int128 v76; // rax
  int v77; // r9d
  unsigned int v78; // edx
  unsigned int v79; // edx
  __int64 v80; // rdx
  __int64 v81; // rdx
  __int128 v82; // rax
  int v83; // r9d
  __int128 v84; // rax
  int v85; // r9d
  __int128 v86; // rax
  int v87; // r9d
  __int128 v88; // [rsp-30h] [rbp-2C0h]
  __int128 v89; // [rsp-10h] [rbp-2A0h]
  __int128 v90; // [rsp-10h] [rbp-2A0h]
  __int128 v91; // [rsp-10h] [rbp-2A0h]
  __int128 v92; // [rsp-10h] [rbp-2A0h]
  __int64 *v93; // [rsp+8h] [rbp-288h]
  __int64 v94; // [rsp+8h] [rbp-288h]
  __int64 *v95; // [rsp+8h] [rbp-288h]
  __int64 v96; // [rsp+10h] [rbp-280h]
  __int64 v97; // [rsp+18h] [rbp-278h]
  int v98; // [rsp+18h] [rbp-278h]
  unsigned int v99; // [rsp+18h] [rbp-278h]
  __int64 v100; // [rsp+20h] [rbp-270h]
  unsigned int v101; // [rsp+28h] [rbp-268h]
  char v102; // [rsp+2Fh] [rbp-261h]
  __int64 v103; // [rsp+30h] [rbp-260h]
  int v104; // [rsp+38h] [rbp-258h]
  __int64 v105; // [rsp+38h] [rbp-258h]
  unsigned int v106; // [rsp+38h] [rbp-258h]
  __int64 v107; // [rsp+40h] [rbp-250h]
  __int128 v108; // [rsp+40h] [rbp-250h]
  __int64 v109; // [rsp+40h] [rbp-250h]
  int v110; // [rsp+40h] [rbp-250h]
  __int64 v111; // [rsp+50h] [rbp-240h]
  int v112; // [rsp+50h] [rbp-240h]
  unsigned __int32 v113; // [rsp+58h] [rbp-238h]
  __int64 v114; // [rsp+58h] [rbp-238h]
  __int64 v115; // [rsp+58h] [rbp-238h]
  __int64 v116; // [rsp+58h] [rbp-238h]
  unsigned int v117; // [rsp+58h] [rbp-238h]
  int v118; // [rsp+58h] [rbp-238h]
  unsigned int v119; // [rsp+58h] [rbp-238h]
  __int64 v120; // [rsp+60h] [rbp-230h]
  __int64 v121; // [rsp+60h] [rbp-230h]
  __int64 v122; // [rsp+60h] [rbp-230h]
  __int64 v123; // [rsp+60h] [rbp-230h]
  __int128 v124; // [rsp+60h] [rbp-230h]
  __int64 v125; // [rsp+70h] [rbp-220h]
  int v126; // [rsp+BCh] [rbp-1D4h] BYREF
  __int64 v127; // [rsp+C0h] [rbp-1D0h]
  __int64 v128; // [rsp+C8h] [rbp-1C8h]
  __m128i v129; // [rsp+D0h] [rbp-1C0h] BYREF
  __int64 v130; // [rsp+E0h] [rbp-1B0h] BYREF
  int v131; // [rsp+E8h] [rbp-1A8h]
  unsigned int v132; // [rsp+F0h] [rbp-1A0h] BYREF
  __int64 v133; // [rsp+F8h] [rbp-198h]
  __m128i v134; // [rsp+100h] [rbp-190h] BYREF
  unsigned int v135; // [rsp+110h] [rbp-180h] BYREF
  __int64 v136; // [rsp+118h] [rbp-178h]
  __int64 v137; // [rsp+120h] [rbp-170h]
  __int64 v138; // [rsp+128h] [rbp-168h]
  char *v139[2]; // [rsp+130h] [rbp-160h] BYREF
  __int64 (__fastcall *v140)(unsigned __int64 *, const __m128i **, int); // [rsp+140h] [rbp-150h]
  __int64 (__fastcall *v141)(unsigned int ***, __int64, __int64 *); // [rsp+148h] [rbp-148h]
  __m128i v142; // [rsp+150h] [rbp-140h] BYREF
  _QWORD v143[38]; // [rsp+160h] [rbp-130h] BYREF

  v4 = *a1;
  v5 = *(const __m128i **)(a2 + 40);
  v6 = v5[2].m128i_i64[1];
  v7 = v6;
  v8 = v5[3].m128i_i64[0];
  v9 = v5->m128i_i64[0];
  v129 = _mm_loadu_si128(v5);
  v120 = v6;
  v113 = v5[3].m128i_u32[0];
  result = sub_3401190(v4, v9, v129.m128i_i64[1], v6, v8);
  if ( !result )
  {
    v11 = *(_QWORD *)(a2 + 80);
    v12 = 0;
    v130 = v11;
    if ( v11 )
      sub_B96E90((__int64)&v130, v11, 1);
    v131 = *(_DWORD *)(a2 + 72);
    v13 = (unsigned __int16 *)(*(_QWORD *)(v129.m128i_i64[0] + 48) + 16LL * v129.m128i_u32[2]);
    v107 = v129.m128i_i64[0];
    v14 = *v13;
    v15 = *((_QWORD *)v13 + 1);
    LOWORD(v132) = v14;
    v133 = v15;
    if ( (_WORD)v14 )
    {
      if ( (unsigned __int16)(v14 - 17) > 0xD3u )
      {
        v142.m128i_i16[0] = v14;
        v142.m128i_i64[1] = v15;
        goto LABEL_14;
      }
      LOWORD(v14) = word_4456580[v14 - 1];
    }
    else
    {
      v111 = v15;
      if ( !sub_30070B0((__int64)&v132) )
      {
        v142.m128i_i64[1] = v111;
        v142.m128i_i16[0] = 0;
LABEL_8:
        v18 = sub_3007260((__int64)&v142);
        v137 = v18;
        v138 = v19;
        goto LABEL_9;
      }
      LOWORD(v14) = sub_3009970((__int64)&v132, v11, v111, v16, v17);
      v12 = v27;
    }
    v142.m128i_i16[0] = v14;
    v142.m128i_i64[1] = v12;
    if ( !(_WORD)v14 )
      goto LABEL_8;
LABEL_14:
    if ( (_WORD)v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
      BUG();
    v18 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v14 - 16];
LABEL_9:
    v126 = v18;
    v20 = _mm_loadu_si128(&v129);
    v21 = *a1;
    v143[0] = v7;
    v143[1] = v8;
    v142 = v20;
    result = sub_3402EA0(v21, 191, (unsigned int)&v130, v132, v133, 0, (__int64)&v142, 2);
    if ( result )
      goto LABEL_10;
    if ( (unsigned int)sub_33D4D80(*a1, v129.m128i_i64[0], v129.m128i_i64[1], 0) == v126 )
    {
      result = v129.m128i_i64[0];
      goto LABEL_10;
    }
    if ( (_WORD)v132 )
    {
      if ( (unsigned __int16)(v132 - 17) > 0xD3u )
        goto LABEL_24;
    }
    else if ( !sub_30070B0((__int64)&v132) )
    {
      goto LABEL_24;
    }
    result = sub_3295970(a1, a2, (__int64)&v130, v22, v23);
    if ( result )
      goto LABEL_10;
LABEL_24:
    result = sub_329BF20(a1, a2);
    if ( result )
      goto LABEL_10;
    v103 = sub_33DFBC0(v7, v8, 0, 0);
    v104 = *(_DWORD *)(v107 + 24);
    if ( v104 == 191 )
    {
      v32 = *(_QWORD *)(v120 + 48) + 16LL * v113;
      v33 = *(_WORD *)v32;
      v34 = *(_QWORD *)(v32 + 8);
      v134.m128i_i16[0] = v33;
      v134.m128i_i64[1] = v34;
      LOWORD(v35) = sub_3281100((unsigned __int16 *)&v134, v8);
      v140 = 0;
      v135 = v35;
      v142.m128i_i64[0] = (__int64)v143;
      v136 = v36;
      v142.m128i_i64[1] = 0x1000000000LL;
      v37 = sub_22077B0(0x28u);
      if ( v37 )
      {
        *(_QWORD *)(v37 + 16) = a1;
        *(_QWORD *)v37 = &v126;
        *(_QWORD *)(v37 + 8) = &v142;
        *(_QWORD *)(v37 + 24) = &v130;
        *(_QWORD *)(v37 + 32) = &v135;
      }
      v139[0] = (char *)v37;
      v141 = sub_3262F10;
      v140 = sub_325EEA0;
      v102 = sub_33CACD0(
               v7,
               v8,
               *(_QWORD *)(*(_QWORD *)(v107 + 40) + 40LL),
               *(_QWORD *)(*(_QWORD *)(v107 + 40) + 48LL),
               (unsigned int)v139,
               0,
               0);
      sub_A17130((__int64)v139);
      if ( v102 )
      {
        v39 = *a1;
        v40 = *(_DWORD *)(v120 + 24);
        if ( v40 == 156 )
        {
          *((_QWORD *)&v92 + 1) = v142.m128i_u32[2];
          *(_QWORD *)&v92 = v142.m128i_i64[0];
          v41 = sub_33FC220(v39, 156, (unsigned int)&v130, v134.m128i_i32[0], v134.m128i_i32[2], v38, v92);
          v42 = v79;
        }
        else if ( v40 == 168 )
        {
          v41 = sub_3288900(
                  v39,
                  v134.m128i_u32[0],
                  v134.m128i_i64[1],
                  (int)&v130,
                  *(_QWORD *)v142.m128i_i64[0],
                  *(_QWORD *)(v142.m128i_i64[0] + 8));
          v42 = v78;
        }
        else
        {
          v41 = *(_QWORD *)v142.m128i_i64[0];
          v42 = *(unsigned int *)(v142.m128i_i64[0] + 8);
        }
        *((_QWORD *)&v89 + 1) = v42;
        *(_QWORD *)&v89 = v41;
        result = sub_3406EB0(*a1, 191, (unsigned int)&v130, v132, v133, v38, *(_OWORD *)*(_QWORD *)(v107 + 40), v89);
        if ( (_QWORD *)v142.m128i_i64[0] != v143 )
        {
          v123 = result;
          _libc_free(v142.m128i_u64[0]);
          result = v123;
        }
        goto LABEL_10;
      }
      if ( (_QWORD *)v142.m128i_i64[0] != v143 )
        _libc_free(v142.m128i_u64[0]);
      v104 = *(_DWORD *)(v107 + 24);
    }
    if ( v104 == 190 && v103 )
    {
      v97 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(v107 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v107 + 40) + 48LL), 0, 0);
      if ( v97 )
      {
        v43 = *(_QWORD *)(v103 + 96);
        v44 = *(_QWORD **)(v43 + 24);
        if ( *(_DWORD *)(v43 + 32) > 0x40u )
          v44 = (_QWORD *)*v44;
        v93 = *(__int64 **)(*a1 + 64);
        v106 = sub_327FC40(v93, v126 - (int)v44);
        v100 = v45;
        if ( sub_32801E0((__int64)&v132) )
        {
          v127 = sub_3281590((__int64)&v132);
          v106 = sub_327FD70(v93, v106, v100, v127);
          v100 = v80;
        }
        v46 = *(_QWORD *)(v103 + 96);
        v47 = *(_QWORD **)(v46 + 24);
        if ( *(_DWORD *)(v46 + 32) > 0x40u )
          v47 = (_QWORD *)*v47;
        v48 = *(_QWORD *)(v97 + 96);
        v49 = *(_QWORD **)(v48 + 24);
        if ( *(_DWORD *)(v48 + 32) > 0x40u )
          v49 = (_QWORD *)*v49;
        v98 = (_DWORD)v47 - (_DWORD)v49;
        if ( (int)v47 - (int)v49 > 0 )
        {
          v96 = a1[1];
          if ( (unsigned __int8)sub_328A020(v96, 0xD5u, v106, v100, 0) )
          {
            v101 = v132;
            v94 = v133;
            if ( (unsigned __int8)sub_328A020(v96, 0xD8u, v132, v133, 0) )
            {
              if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64, _QWORD, __int64))(*(_QWORD *)v96 + 1392LL))(
                     v96,
                     v101,
                     v94,
                     v106,
                     v100) )
              {
                *(_QWORD *)&v82 = sub_3400E40(*a1, v98, v132, v133, &v130);
                *(_QWORD *)&v84 = sub_3406EB0(
                                    *a1,
                                    192,
                                    (unsigned int)&v130,
                                    v132,
                                    v133,
                                    v83,
                                    *(_OWORD *)*(_QWORD *)(v107 + 40),
                                    v82);
                *(_QWORD *)&v86 = sub_33FAF80(*a1, 216, (unsigned int)&v130, v106, v100, v85, v84);
                result = sub_33FAF80(
                           *a1,
                           213,
                           (unsigned int)&v130,
                           **(unsigned __int16 **)(a2 + 48),
                           *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                           v87,
                           v86);
LABEL_10:
                if ( v130 )
                {
                  v121 = result;
                  sub_B91220((__int64)&v130, v130);
                  return v121;
                }
                return result;
              }
            }
          }
        }
      }
      v104 = *(_DWORD *)(v107 + 24);
      if ( (unsigned int)(v104 - 56) > 1 )
      {
LABEL_32:
        if ( *(_DWORD *)(v120 + 24) != 216 )
          goto LABEL_33;
        goto LABEL_45;
      }
    }
    else
    {
      if ( (unsigned int)(v104 - 56) > 1 )
        goto LABEL_32;
      if ( !v103 )
        goto LABEL_44;
    }
    if ( (unsigned __int8)sub_3286E00(&v129) )
    {
      v24 = *(_QWORD *)(v107 + 40);
      v25 = _mm_loadu_si128((const __m128i *)(v24 + 40LL * (v104 != 56)));
      v134 = v25;
      if ( *(_DWORD *)(v25.m128i_i64[0] + 24) == 190 )
      {
        v50 = *(_QWORD *)(v25.m128i_i64[0] + 40);
        if ( v120 == *(_QWORD *)(v50 + 40) && v113 == *(_DWORD *)(v50 + 48) )
        {
          v115 = v24;
          if ( (unsigned __int8)sub_3286E00(&v134) )
          {
            v116 = sub_33DFBC0(
                     *(_QWORD *)(40LL * (v104 == 56) + v115),
                     *(_QWORD *)(40LL * (v104 == 56) + v115 + 8),
                     0,
                     0);
            if ( v116 )
            {
              v51 = *(_QWORD *)(v103 + 96);
              v52 = *(_QWORD **)(v51 + 24);
              if ( *(_DWORD *)(v51 + 32) > 0x40u )
                v52 = (_QWORD *)*v52;
              v95 = *(__int64 **)(*a1 + 64);
              v99 = (unsigned int)v52;
              v135 = sub_327FC40(v95, v126 - (int)v52);
              v136 = v53;
              if ( sub_32801E0((__int64)&v132) )
              {
                v128 = sub_3281590((__int64)&v132);
                v135 = sub_327FD70(v95, v135, v136, v128);
                v136 = v81;
              }
              if ( (_WORD)v135 )
              {
                v54 = a1[1];
                if ( !*((_BYTE *)a1 + 34) || (unsigned __int8)sub_325E6A0(v54, (unsigned __int16 *)&v135) )
                {
                  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64, _QWORD, __int64))(*(_QWORD *)v54 + 1392LL))(
                         v54,
                         v132,
                         v133,
                         v135,
                         v136) )
                  {
                    *(_QWORD *)&v55 = sub_33FB310(
                                        *a1,
                                        **(_QWORD **)(v25.m128i_i64[0] + 40),
                                        *(_QWORD *)(*(_QWORD *)(v25.m128i_i64[0] + 40) + 8LL),
                                        &v130,
                                        v135,
                                        v136);
                    v56 = *a1;
                    v108 = v55;
                    sub_9865C0((__int64)v139, *(_QWORD *)(v116 + 96) + 24LL);
                    sub_986C30((__int64)v139, v99);
                    v57 = sub_32844A0((unsigned __int16 *)&v135, v99);
                    sub_C44740((__int64)&v142, v139, v57);
                    *(_QWORD *)&v124 = sub_34007B0(v56, (unsigned int)&v142, (unsigned int)&v130, v135, v136, 0, 0);
                    *((_QWORD *)&v124 + 1) = v58;
                    sub_969240(v142.m128i_i64);
                    sub_969240((__int64 *)v139);
                    if ( v104 == 56 )
                      v60 = sub_3406EB0(*a1, 56, (unsigned int)&v130, v135, v136, (unsigned int)&v142, v108, v124);
                    else
                      v60 = sub_3406EB0(*a1, 57, (unsigned int)&v130, v135, v136, (unsigned int)&v142, v124, v108);
                    result = sub_33FB160(*a1, v60, v59, &v130, v132, v133);
                    goto LABEL_10;
                  }
                }
              }
            }
          }
        }
      }
      goto LABEL_32;
    }
LABEL_44:
    if ( *(_DWORD *)(v120 + 24) != 216 )
      goto LABEL_38;
LABEL_45:
    if ( *(_DWORD *)(**(_QWORD **)(v120 + 40) + 24LL) == 186 )
    {
      *(_QWORD *)&v28 = sub_32CB9C0((__int64)a1, (_QWORD *)v120);
      if ( (_QWORD)v28 )
      {
        result = sub_3406EB0(*a1, 191, (unsigned int)&v130, v132, v133, v29, *(_OWORD *)&v129, v28);
        goto LABEL_10;
      }
    }
LABEL_33:
    if ( *(_DWORD *)(v107 + 24) == 216 )
    {
      v26 = *(_DWORD **)(v107 + 40);
      v122 = *(_QWORD *)v26;
      if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)v26 + 24LL) - 191) <= 1 )
      {
        v114 = *(_QWORD *)(v107 + 40);
        if ( (unsigned __int8)sub_3286E00(v26) )
        {
          v105 = *(_QWORD *)(v122 + 40);
          if ( (unsigned __int8)sub_3286E00((_DWORD *)(v105 + 40)) )
          {
            if ( v103 )
            {
              v61 = *(_QWORD *)(v105 + 48);
              v117 = *(_DWORD *)(v114 + 8);
              v109 = sub_33DFBC0(*(_QWORD *)(v105 + 40), v61, 0, 0);
              if ( v109 )
              {
                v62 = *(_QWORD *)(v122 + 48) + 16LL * v117;
                v63 = *(_WORD *)v62;
                v64 = *(_QWORD *)(v62 + 8);
                v142.m128i_i16[0] = v63;
                v142.m128i_i64[1] = v64;
                v65 = sub_32844A0((unsigned __int16 *)&v142, v61);
                v118 = v65 - v126;
                if ( sub_D94970(*(_QWORD *)(v109 + 96) + 24LL, (_QWORD *)(unsigned int)(v65 - v126)) )
                {
                  v110 = v118;
                  v119 = sub_325F340(*a1, a1[1], v142.m128i_u32[0], v142.m128i_i64[1]);
                  v112 = v66;
                  v67 = sub_33FB310(*a1, v7, v8, &v130, v119, v66);
                  v68 = *a1;
                  v70 = v69;
                  v71 = v67;
                  *(_QWORD *)&v72 = sub_3400BD0(*a1, v110, (unsigned int)&v130, v119, v112, 0, 0, v69);
                  *((_QWORD *)&v88 + 1) = v70;
                  *(_QWORD *)&v88 = v71;
                  v125 = sub_3406EB0(v68, 56, (unsigned int)&v130, v119, v112, v73, v88, v72);
                  *((_QWORD *)&v91 + 1) = v74 | v70 & 0xFFFFFFFF00000000LL;
                  *(_QWORD *)&v91 = v125;
                  *(_QWORD *)&v76 = sub_3406EB0(
                                      *a1,
                                      191,
                                      (unsigned int)&v130,
                                      v142.m128i_i32[0],
                                      v142.m128i_i32[2],
                                      v75,
                                      *(_OWORD *)*(_QWORD *)(v122 + 40),
                                      v91);
                  result = sub_33FAF80(*a1, 216, (unsigned int)&v130, v132, v133, v77, v76);
                  goto LABEL_10;
                }
              }
            }
          }
        }
      }
    }
LABEL_38:
    if ( (unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) )
    {
      result = a2;
    }
    else if ( (unsigned __int8)sub_33DD2A0(*a1, v129.m128i_i64[0], v129.m128i_i64[1], 0) )
    {
      *((_QWORD *)&v90 + 1) = v8;
      *(_QWORD *)&v90 = v7;
      result = sub_3406EB0(*a1, 192, (unsigned int)&v130, v132, v133, v30, *(_OWORD *)&v129, v90);
    }
    else if ( !v103 || (*(_BYTE *)(v103 + 32) & 8) != 0 || (result = sub_327E0B0(a1, a2)) == 0 )
    {
      result = sub_328A0F0(a2, (__int64)&v130, *a1, a1[1]);
      if ( !result )
      {
        result = sub_32B3F40(a1, a2);
        if ( !result )
        {
          v31 = sub_326C8E0(a1, a2);
          result = 0;
          if ( v31 )
            result = v31;
        }
      }
    }
    goto LABEL_10;
  }
  return result;
}
