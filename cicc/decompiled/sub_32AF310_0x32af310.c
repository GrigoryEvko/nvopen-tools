// Function: sub_32AF310
// Address: 0x32af310
//
__int64 __fastcall sub_32AF310(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int16 *v5; // rdx
  int v6; // eax
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 *v21; // rax
  __int64 v22; // r13
  __int64 v23; // rdi
  unsigned int v24; // r15d
  __int64 v25; // r13
  __int64 v27; // r14
  __int64 v28; // rsi
  unsigned __int16 *v29; // rax
  __int64 v30; // r10
  unsigned int v31; // ecx
  __int32 v32; // edx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  unsigned int v36; // r13d
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // r14d
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r14
  __int64 v43; // rdi
  int v44; // eax
  int v45; // r9d
  unsigned __int16 *v46; // rdx
  int v47; // eax
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // rdi
  unsigned __int64 *v51; // r8
  __int64 v52; // rdi
  int v53; // eax
  char v54; // r15
  __int64 v55; // rax
  unsigned int v56; // r14d
  unsigned int v57; // eax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // rdx
  _QWORD *v64; // rbx
  _QWORD *v65; // rax
  int v66; // edx
  __int64 v67; // rbx
  int v68; // r15d
  __int32 v69; // edx
  int v70; // eax
  __m128i v71; // xmm1
  __int64 v72; // rax
  int v73; // edx
  int v74; // r15d
  __int64 v75; // rax
  int v76; // r12d
  __int64 v77; // rbx
  __int32 v78; // edx
  int v79; // eax
  __m128i v80; // xmm2
  __m128i v81; // xmm3
  int v82; // r9d
  __int64 v83; // rax
  int v84; // r12d
  __int64 v85; // rbx
  int v86; // edx
  int v87; // r15d
  int v88; // eax
  __m128i v89; // xmm4
  __m128i v90; // xmm5
  int v91; // r9d
  __int64 v92; // rax
  __int64 v93; // rax
  unsigned int v94; // r8d
  __int64 v95; // rax
  char v96; // al
  __int64 v97; // rax
  __int64 v98; // rdx
  unsigned __int64 *v99; // r9
  __int64 v100; // rdi
  int v101; // eax
  char v102; // al
  __int64 v103; // rax
  unsigned int v104; // edx
  __int64 v105; // rax
  __int64 v106; // rsi
  __int64 v107; // rax
  _QWORD *v108; // rax
  __int128 v109; // [rsp-20h] [rbp-1A0h]
  __int128 v110; // [rsp-10h] [rbp-190h]
  __int128 v111; // [rsp-10h] [rbp-190h]
  __int32 v112; // [rsp+0h] [rbp-180h]
  unsigned int v113; // [rsp+8h] [rbp-178h]
  __int64 v114; // [rsp+8h] [rbp-178h]
  __int64 v115; // [rsp+8h] [rbp-178h]
  unsigned int v116; // [rsp+8h] [rbp-178h]
  char v117; // [rsp+8h] [rbp-178h]
  unsigned int v118; // [rsp+10h] [rbp-170h]
  int v119; // [rsp+10h] [rbp-170h]
  __int32 v120; // [rsp+10h] [rbp-170h]
  __int64 v121; // [rsp+18h] [rbp-168h]
  int v122; // [rsp+18h] [rbp-168h]
  int v123; // [rsp+18h] [rbp-168h]
  int v124; // [rsp+18h] [rbp-168h]
  __int64 v125; // [rsp+40h] [rbp-140h] BYREF
  int v126; // [rsp+48h] [rbp-138h]
  __int64 v127; // [rsp+50h] [rbp-130h] BYREF
  __int64 v128; // [rsp+58h] [rbp-128h]
  unsigned __int64 v129; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v130; // [rsp+68h] [rbp-118h]
  unsigned __int64 v131; // [rsp+70h] [rbp-110h] BYREF
  unsigned int v132; // [rsp+78h] [rbp-108h]
  _QWORD *v133; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v134; // [rsp+88h] [rbp-F8h]
  __int128 v135; // [rsp+90h] [rbp-F0h] BYREF
  _QWORD v136[4]; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned __int64 v137; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v138; // [rsp+C8h] [rbp-B8h]
  _QWORD v139[2]; // [rsp+D0h] [rbp-B0h] BYREF
  __m128i v140; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i v141; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v142; // [rsp+100h] [rbp-80h]
  __int64 v143; // [rsp+110h] [rbp-70h] BYREF
  __int64 v144; // [rsp+118h] [rbp-68h] BYREF
  __m128i v145; // [rsp+120h] [rbp-60h]
  __m128i v146; // [rsp+130h] [rbp-50h]
  __m128i v147; // [rsp+140h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 80);
  v125 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v125, v4, 1);
  v126 = *(_DWORD *)(a1 + 72);
  v5 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a1 + 40) + 48LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 40) + 8LL));
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  LOWORD(v127) = v6;
  v128 = v7;
  if ( (_WORD)v6 )
  {
    if ( (unsigned __int16)(v6 - 17) > 0xD3u )
    {
      LOWORD(v137) = v6;
      v138 = v7;
      goto LABEL_6;
    }
    LOWORD(v6) = word_4456580[v6 - 1];
    v9 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v127) )
    {
      v138 = v7;
      LOWORD(v137) = 0;
      goto LABEL_11;
    }
    LOWORD(v6) = sub_3009970((__int64)&v127, v4, v33, v34, v35);
  }
  LOWORD(v137) = v6;
  v138 = v9;
  if ( !(_WORD)v6 )
  {
LABEL_11:
    v10 = sub_3007260((__int64)&v137);
    v8 = v11;
    v12 = v10;
    v13 = v8;
    v136[2] = v12;
    LODWORD(v8) = v12;
    v136[3] = v13;
    goto LABEL_12;
  }
LABEL_6:
  if ( (_WORD)v6 == 1 || (unsigned __int16)(v6 - 504) <= 7u )
    goto LABEL_153;
  v8 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v6 - 16];
LABEL_12:
  v14 = *(_QWORD *)(a2 + 16);
  v15 = *(unsigned int *)(a1 + 24);
  v139[0] = a2;
  v140.m128i_i64[0] = 0;
  v139[1] = v14;
  v140.m128i_i32[2] = 0;
  v141.m128i_i64[0] = 0;
  v141.m128i_i32[2] = 0;
  v142 = a1;
  v137 = sub_33CB160(v15);
  if ( BYTE4(v137) )
  {
    v16 = *(_QWORD *)(v142 + 40) + 40LL * (unsigned int)v137;
    v140.m128i_i64[0] = *(_QWORD *)v16;
    v140.m128i_i32[2] = *(_DWORD *)(v16 + 8);
    v17 = *(unsigned int *)(v142 + 24);
  }
  else
  {
    v27 = v142;
    v17 = *(unsigned int *)(v142 + 24);
    if ( (_DWORD)v17 == 488 )
    {
      v28 = *(_QWORD *)(v142 + 80);
      v29 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v142 + 40) + 48LL)
                               + 16LL * *(unsigned int *)(*(_QWORD *)(v142 + 40) + 8LL));
      v30 = *((_QWORD *)v29 + 1);
      v31 = *v29;
      v143 = v28;
      if ( v28 )
      {
        v118 = v31;
        v121 = v30;
        sub_B96E90((__int64)&v143, v28, 1);
        v31 = v118;
        v30 = v121;
      }
      LODWORD(v144) = *(_DWORD *)(v27 + 72);
      v140.m128i_i64[0] = sub_34015B0(a2, &v143, v31, v30, 0, 0);
      v140.m128i_i32[2] = v32;
      if ( v143 )
        sub_B91220((__int64)&v143, v143);
      v17 = *(unsigned int *)(v142 + 24);
    }
  }
  v143 = sub_33CB1F0(v17);
  if ( BYTE4(v143) )
  {
    v18 = *(_QWORD *)(v142 + 40) + 40LL * (unsigned int)v143;
    v141.m128i_i64[0] = *(_QWORD *)v18;
    v141.m128i_i32[2] = *(_DWORD *)(v18 + 8);
  }
  v130 = 1;
  v19 = *(unsigned int *)(a1 + 24);
  v129 = 0;
  v132 = 1;
  v131 = 0;
  v134 = 1;
  v133 = 0;
  *(_QWORD *)&v135 = 0;
  DWORD2(v135) = 0;
  if ( (unsigned __int8)sub_33CB110(v19) )
  {
    v137 = sub_33CB280(*(unsigned int *)(a1 + 24), ((unsigned __int8)(*(_DWORD *)(a1 + 28) >> 12) ^ 1) & 1);
    if ( !BYTE4(v137) || (_DWORD)v137 != 57 )
      goto LABEL_20;
    v36 = *(_DWORD *)(a1 + 24);
    v143 = sub_33CB160(v36);
    if ( BYTE4(v143) )
    {
      v37 = *(_QWORD *)(a1 + 40) + 40LL * (unsigned int)v143;
      if ( (*(_QWORD *)v37 != v140.m128i_i64[0] || *(_DWORD *)(v37 + 8) != v140.m128i_i32[2])
        && !(unsigned __int8)sub_33D1720(*(_QWORD *)v37, 0) )
      {
        goto LABEL_20;
      }
    }
    v143 = sub_33CB1F0(v36);
    if ( BYTE4(v143) )
    {
      v38 = *(_QWORD *)(a1 + 40) + 40LL * (unsigned int)v143;
      if ( v141.m128i_i64[0] != *(_QWORD *)v38 || v141.m128i_i32[2] != *(_DWORD *)(v38 + 8) )
        goto LABEL_20;
    }
    v20 = *(unsigned int *)(a1 + 24);
LABEL_18:
    sub_33CB110(v20);
    v21 = *(__int64 **)(a1 + 40);
    v22 = *v21;
    if ( (unsigned __int8)sub_33CB110(*(unsigned int *)(*v21 + 24)) )
    {
      v137 = sub_33CB280(*(unsigned int *)(v22 + 24), ((unsigned __int8)(*(_DWORD *)(v22 + 28) >> 12) ^ 1) & 1);
      if ( !BYTE4(v137) || (_DWORD)v137 != 199 )
        goto LABEL_20;
      v39 = *(_DWORD *)(v22 + 24);
      v143 = sub_33CB160(v39);
      if ( BYTE4(v143) )
      {
        v40 = *(_QWORD *)(v22 + 40) + 40LL * (unsigned int)v143;
        if ( (*(_QWORD *)v40 != v140.m128i_i64[0] || *(_DWORD *)(v40 + 8) != v140.m128i_i32[2])
          && !(unsigned __int8)sub_33D1720(*(_QWORD *)v40, 0) )
        {
          goto LABEL_20;
        }
      }
      v143 = sub_33CB1F0(v39);
      if ( BYTE4(v143) )
      {
        v41 = *(_QWORD *)(v22 + 40) + 40LL * (unsigned int)v143;
        if ( v141.m128i_i64[0] != *(_QWORD *)v41 || v141.m128i_i32[2] != *(_DWORD *)(v41 + 8) )
          goto LABEL_20;
      }
      v23 = *(unsigned int *)(v22 + 24);
    }
    else
    {
      v23 = *(unsigned int *)(v22 + 24);
      if ( (_DWORD)v23 != 199 )
        goto LABEL_20;
    }
    sub_33CB110(v23);
    v42 = **(_QWORD **)(v22 + 40);
    v43 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL);
    if ( v43 && ((v44 = *(_DWORD *)(v43 + 24), v44 == 11) || v44 == 35) )
    {
      v103 = *(_QWORD *)(v43 + 96);
      if ( v134 <= 0x40 && (v104 = *(_DWORD *)(v103 + 32), v104 <= 0x40) )
      {
        v108 = *(_QWORD **)(v103 + 24);
        v134 = v104;
        v133 = v108;
      }
      else
      {
        sub_C43990((__int64)&v133, v103 + 24);
      }
    }
    else if ( !(unsigned __int8)sub_33D1410(v43, &v133) )
    {
      goto LABEL_20;
    }
    v146.m128i_i8[0] = 0;
    LODWORD(v143) = 214;
    LODWORD(v144) = 188;
    v145.m128i_i64[0] = (__int64)&v135;
    v146.m128i_i8[12] = 0;
    if ( sub_32AE740((int *)&v143, (__int64)v139, v42) )
    {
      v46 = (unsigned __int16 *)(*(_QWORD *)(v135 + 48) + 16LL * DWORD2(v135));
      v47 = *v46;
      v48 = *((_QWORD *)v46 + 1);
      LOWORD(v136[0]) = v47;
      v136[1] = v48;
      if ( (_WORD)v47 )
      {
        if ( (unsigned __int16)(v47 - 17) > 0xD3u )
        {
          LOWORD(v143) = v47;
          v144 = v48;
LABEL_67:
          if ( (_WORD)v47 != 1 && (unsigned __int16)(v47 - 504) > 7u )
          {
            v49 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v47 - 16];
LABEL_99:
            v24 = v134;
            v64 = (_QWORD *)((unsigned int)v8 - v49);
            if ( v134 > 0x40 )
            {
              if ( v24 - (unsigned int)sub_C444A0((__int64)&v133) > 0x40 )
                goto LABEL_102;
              v65 = (_QWORD *)*v133;
            }
            else
            {
              v65 = v133;
            }
            if ( v64 != v65 )
            {
LABEL_102:
              v25 = 0;
              goto LABEL_21;
            }
            *(_QWORD *)&v135 = sub_33FAF80(a2, 215, (unsigned int)&v125, v127, v128, v45, v135);
            DWORD2(v135) = v66;
LABEL_106:
            v67 = sub_3400EC0(a2, &v133, (unsigned int)v127, v128, &v125);
            v68 = DWORD2(v135);
            v112 = v69;
            v119 = v128;
            v114 = v135;
            v122 = v127;
            v70 = sub_33CB7C0(190);
            LODWORD(v144) = v68;
            v145.m128i_i64[0] = v67;
            v145.m128i_i32[2] = v112;
            v71 = _mm_loadu_si128(&v141);
            *((_QWORD *)&v110 + 1) = 4;
            *(_QWORD *)&v110 = &v143;
            v146 = _mm_loadu_si128(&v140);
            v147 = v71;
            v143 = v114;
            v72 = sub_33FC220(v139[0], v70, (unsigned int)&v125, v122, v119, v114, v110);
            v74 = v73;
            v115 = v72;
            v75 = sub_34015B0(a2, &v125, (unsigned int)v127, v128, 0, 0);
            v76 = v127;
            v77 = v75;
            v120 = v78;
            v123 = v128;
            v79 = sub_33CB7C0(188);
            v80 = _mm_loadu_si128(&v140);
            LODWORD(v144) = v74;
            v145.m128i_i32[2] = v120;
            v81 = _mm_loadu_si128(&v141);
            *((_QWORD *)&v109 + 1) = 4;
            *(_QWORD *)&v109 = &v143;
            v143 = v115;
            v146 = v80;
            v147 = v81;
            v145.m128i_i64[0] = v77;
            v83 = sub_33FC220(v139[0], v79, (unsigned int)&v125, v76, v123, v82, v109);
            v84 = v127;
            v85 = v83;
            v87 = v86;
            v124 = v128;
            v88 = sub_33CB7C0(204);
            v89 = _mm_loadu_si128(&v140);
            *((_QWORD *)&v111 + 1) = 3;
            *(_QWORD *)&v111 = &v143;
            v90 = _mm_loadu_si128(&v141);
            LODWORD(v144) = v87;
            v143 = v85;
            v145 = v89;
            v146 = v90;
            v92 = sub_33FC220(v139[0], v88, (unsigned int)&v125, v84, v124, v91, v111);
            v24 = v134;
            v25 = v92;
            goto LABEL_21;
          }
LABEL_153:
          BUG();
        }
        LOWORD(v47) = word_4456580[v47 - 1];
        v98 = 0;
      }
      else
      {
        if ( !sub_30070B0((__int64)v136) )
        {
          v144 = v48;
          LOWORD(v143) = 0;
          goto LABEL_98;
        }
        LOWORD(v47) = sub_3009970((__int64)v136, (__int64)v139, v60, v61, v62);
      }
      LOWORD(v143) = v47;
      v144 = v98;
      if ( (_WORD)v47 )
        goto LABEL_67;
LABEL_98:
      v49 = sub_3007260((__int64)&v143);
      v137 = v49;
      v138 = v63;
      goto LABEL_99;
    }
    v146.m128i_i8[4] = 0;
    v145.m128i_i64[1] = (__int64)&v131;
    LODWORD(v143) = 186;
    LODWORD(v144) = 188;
    v145.m128i_i64[0] = (__int64)&v135;
    v146.m128i_i64[1] = (__int64)&v129;
    v147.m128i_i8[4] = 0;
    if ( (unsigned __int8)sub_33CB110(*(unsigned int *)(v42 + 24)) )
    {
      v136[0] = sub_33CB280(*(unsigned int *)(v42 + 24), ((unsigned __int8)(*(_DWORD *)(v42 + 28) >> 12) ^ 1) & 1);
      if ( !BYTE4(v136[0]) || LODWORD(v136[0]) != 186 )
        goto LABEL_20;
      v116 = *(_DWORD *)(v42 + 24);
      v93 = sub_33CB160(v116);
      v94 = v116;
      v137 = v93;
      if ( BYTE4(v93) )
      {
        v95 = *(_QWORD *)(v42 + 40) + 40LL * (unsigned int)v137;
        if ( *(_QWORD *)v95 != v140.m128i_i64[0] || *(_DWORD *)(v95 + 8) != v140.m128i_i32[2] )
        {
          v96 = sub_33D1720(*(_QWORD *)v95, 0);
          v94 = v116;
          if ( !v96 )
            goto LABEL_20;
        }
      }
      v137 = sub_33CB1F0(v94);
      if ( BYTE4(v137) )
      {
        v97 = *(_QWORD *)(v42 + 40) + 40LL * (unsigned int)v137;
        if ( v141.m128i_i64[0] != *(_QWORD *)v97 || v141.m128i_i32[2] != *(_DWORD *)(v97 + 8) )
          goto LABEL_20;
      }
      v50 = *(unsigned int *)(v42 + 24);
    }
    else
    {
      v50 = *(unsigned int *)(v42 + 24);
      if ( (_DWORD)v50 != 186 )
        goto LABEL_20;
    }
    sub_33CB110(v50);
    if ( sub_32AF080((int *)&v144, (__int64)v139, **(_QWORD **)(v42 + 40)) )
    {
      v99 = (unsigned __int64 *)v146.m128i_i64[1];
      v100 = *(_QWORD *)(*(_QWORD *)(v42 + 40) + 40LL);
      if ( v100 )
      {
        v101 = *(_DWORD *)(v100 + 24);
        if ( v101 == 11 || v101 == 35 )
        {
          if ( !v146.m128i_i64[1] )
            goto LABEL_83;
          v105 = *(_QWORD *)(v100 + 96);
          v106 = v105 + 24;
          if ( *(_DWORD *)(v146.m128i_i64[1] + 8) > 0x40u || *(_DWORD *)(v105 + 32) > 0x40u )
            goto LABEL_139;
          *(_QWORD *)v146.m128i_i64[1] = *(_QWORD *)(v105 + 24);
          *((_DWORD *)v99 + 2) = *(_DWORD *)(v105 + 32);
          goto LABEL_83;
        }
      }
      LODWORD(v138) = 1;
      if ( !v146.m128i_i64[1] )
        v99 = &v137;
      v137 = 0;
      v102 = sub_33D1410(v100, v99);
      if ( (unsigned int)v138 > 0x40 && v137 )
      {
        v117 = v102;
        j_j___libc_free_0_0(v137);
        v102 = v117;
      }
      if ( v102 )
        goto LABEL_83;
    }
    if ( !sub_32AF080((int *)&v144, (__int64)v139, *(_QWORD *)(*(_QWORD *)(v42 + 40) + 40LL)) )
      goto LABEL_20;
    v51 = (unsigned __int64 *)v146.m128i_i64[1];
    v52 = **(_QWORD **)(v42 + 40);
    if ( !v52 || (v53 = *(_DWORD *)(v52 + 24), v53 != 35) && v53 != 11 )
    {
      LODWORD(v138) = 1;
      v137 = 0;
      if ( !v146.m128i_i64[1] )
        v51 = &v137;
      v54 = sub_33D1410(v52, v51);
      if ( (unsigned int)v138 > 0x40 && v137 )
        j_j___libc_free_0_0(v137);
      if ( !v54 )
        goto LABEL_20;
      goto LABEL_83;
    }
    if ( v146.m128i_i64[1] )
    {
      v107 = *(_QWORD *)(v52 + 96);
      v106 = v107 + 24;
      if ( *(_DWORD *)(v146.m128i_i64[1] + 8) > 0x40u || *(_DWORD *)(v107 + 32) > 0x40u )
      {
LABEL_139:
        sub_C43990(v146.m128i_i64[1], v106);
        goto LABEL_83;
      }
      *(_QWORD *)v146.m128i_i64[1] = *(_QWORD *)(v107 + 24);
      *((_DWORD *)v51 + 2) = *(_DWORD *)(v107 + 32);
    }
LABEL_83:
    if ( v147.m128i_i8[4] && v147.m128i_i32[0] != (v147.m128i_i32[0] & *(_DWORD *)(v42 + 28)) )
      goto LABEL_20;
    v24 = v134;
    LODWORD(v55) = (_DWORD)v133;
    if ( v134 > 0x40 )
      v55 = *v133;
    v56 = v8 - v55;
    if ( v130 <= 0x40 )
    {
      if ( v129 != 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v55 - (unsigned __int8)v8 + 64) )
        goto LABEL_102;
    }
    else
    {
      v113 = v130;
      if ( v56 != (unsigned int)sub_C445E0((__int64)&v129) || v113 != v56 + (unsigned int)sub_C444A0((__int64)&v129) )
        goto LABEL_102;
    }
    if ( v132 > 0x40 )
    {
      v57 = sub_C445E0((__int64)&v131);
    }
    else
    {
      v57 = 64;
      _RDX = ~v131;
      __asm { tzcnt   rcx, rdx }
      if ( v131 != -1 )
        v57 = _RCX;
    }
    if ( v56 > v57 )
      goto LABEL_102;
    goto LABEL_106;
  }
  v20 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v20 == 57 )
    goto LABEL_18;
LABEL_20:
  v24 = v134;
  v25 = 0;
LABEL_21:
  if ( v24 > 0x40 && v133 )
    j_j___libc_free_0_0((unsigned __int64)v133);
  if ( v132 > 0x40 && v131 )
    j_j___libc_free_0_0(v131);
  if ( v130 > 0x40 && v129 )
    j_j___libc_free_0_0(v129);
  if ( v125 )
    sub_B91220((__int64)&v125, v125);
  return v25;
}
