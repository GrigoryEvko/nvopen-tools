// Function: sub_8AF6E0
// Address: 0x8af6e0
//
__int64 __fastcall sub_8AF6E0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // r14
  __m128i v6; // xmm2
  __m128i v7; // xmm3
  unsigned int *v8; // rsi
  char v9; // dl
  __int64 v10; // rdx
  _BYTE *v11; // rsi
  unsigned int *v12; // rax
  __int64 *v13; // rbx
  unsigned int v14; // r14d
  __m128i *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 *v19; // rcx
  unsigned __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // eax
  __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned __int16 v32; // ax
  char v33; // bl
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  _BOOL4 v38; // r10d
  int v39; // ebx
  int v40; // eax
  int v41; // edi
  int v42; // ecx
  __int64 v43; // rdx
  int v44; // esi
  __m128i *v45; // rax
  unsigned int *v46; // rsi
  __int64 v47; // r14
  __m128i *v48; // rbx
  __m128i *v49; // rdi
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __m128i **v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  int v67; // eax
  _BYTE *v68; // rdi
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  unsigned __int16 v73; // bx
  unsigned int *v74; // rsi
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  _BYTE *v79; // rdi
  __int64 v80; // rdx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rdx
  char *v84; // r8
  unsigned __int16 v85; // ax
  __int64 v86; // rcx
  __m128i *v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r9
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // r9
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // rax
  int v102; // edx
  unsigned __int64 v103; // rdi
  __int64 v104; // rdx
  __int64 v105; // rcx
  __int64 v106; // r8
  __int64 v107; // r9
  unsigned __int64 v108; // rdi
  unsigned int *v109; // rsi
  char v110; // dl
  __int8 v111; // al
  __int64 v112; // r9
  int v113; // eax
  __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  __int64 v118; // rdx
  __int64 v119; // rcx
  __int64 v120; // r8
  __int64 v121; // r9
  __int64 v122; // rdx
  __int64 v123; // rcx
  __int64 v124; // r8
  __int64 v125; // r9
  __int64 v126; // rax
  __int64 result; // rax
  __m128i *v128; // [rsp-10h] [rbp-180h]
  __int64 v129; // [rsp-8h] [rbp-178h]
  __m128i *v130; // [rsp+18h] [rbp-158h]
  int v131; // [rsp+20h] [rbp-150h]
  unsigned int v132; // [rsp+20h] [rbp-150h]
  unsigned int v133; // [rsp+20h] [rbp-150h]
  char *v134; // [rsp+20h] [rbp-150h]
  char *v135; // [rsp+20h] [rbp-150h]
  __int64 v136; // [rsp+20h] [rbp-150h]
  __int64 v137; // [rsp+20h] [rbp-150h]
  int v138; // [rsp+28h] [rbp-148h]
  __m128i *v139; // [rsp+30h] [rbp-140h]
  unsigned int v140; // [rsp+38h] [rbp-138h]
  unsigned int v141; // [rsp+3Ch] [rbp-134h]
  unsigned int v142; // [rsp+44h] [rbp-12Ch] BYREF
  int v143; // [rsp+48h] [rbp-128h] BYREF
  int v144; // [rsp+4Ch] [rbp-124h] BYREF
  int v145; // [rsp+50h] [rbp-120h] BYREF
  int v146; // [rsp+54h] [rbp-11Ch] BYREF
  unsigned __int64 v147; // [rsp+58h] [rbp-118h] BYREF
  __int64 v148; // [rsp+60h] [rbp-110h] BYREF
  __m128i *v149; // [rsp+68h] [rbp-108h] BYREF
  __int64 v150; // [rsp+70h] [rbp-100h] BYREF
  unsigned __int64 v151; // [rsp+78h] [rbp-F8h]
  __m128i v152[2]; // [rsp+80h] [rbp-F0h] BYREF
  __m128i v153[4]; // [rsp+A0h] [rbp-D0h] BYREF
  _BYTE v154[88]; // [rsp+E0h] [rbp-90h] BYREF
  _BYTE v155[56]; // [rsp+138h] [rbp-38h] BYREF

  v150 = 0;
  v151 = 0;
  v138 = 0;
  v2 = qword_4F061C8;
  v139 = 0;
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  ++*(_BYTE *)(v2 + 81);
  ++*(_BYTE *)(v2 + 52);
  v130 = 0;
  while ( word_4F06418[0] != 44 && word_4F06418[0] != 9 )
  {
    v8 = (unsigned int *)&v148;
    ++HIDWORD(v150);
    ++*(_BYTE *)(qword_4F061C8 + 75LL);
    v25 = sub_868D90(&v147, &v148, 0, 1, 0);
    v26 = *(_QWORD *)(a1 + 496);
    if ( v26 )
      v27 = *(_QWORD *)(v26 + 8);
    else
      v27 = v147;
    v151 = v27;
    if ( v25 )
    {
      v140 = 0;
      while ( 1 )
      {
        LODWORD(v150) = v150 + 1;
        sub_7BDB60(1);
        v68 = v154;
        v141 = dword_4F06650[0];
        sub_7ADF70((__int64)v154, 0);
        v73 = word_4F06418[0];
        if ( dword_4D04494 )
        {
          if ( dword_4F077C4 == 2 )
          {
            if ( word_4F06418[0] == 1 && (word_4D04A10 & 0x200) != 0
              || (v68 = aMquoffmafNvFma + 2, (unsigned int)sub_7C0F00(0x4000001u, 0, v69, v70, v71, v72)) )
            {
LABEL_62:
              v153[0].m128i_i32[0] = 0;
              v96 = sub_7BF130(0x4000001u, 0, v153);
              if ( !v96 || *(_BYTE *)(v96 + 80) != 22 || (v136 = v96, sub_651150(1)) )
              {
                sub_7AE700((__int64)(qword_4F061C0 + 3), v141, dword_4F06650[0], 0, (__int64)v154);
                sub_7BC000((unsigned __int64)v154, v141, v97, v98, v99, v100);
                if ( *(_DWORD *)(a1 + 92) )
                {
                  sub_6851C0(0x83Bu, &dword_4F063F8);
                  v140 = 1;
                }
                goto LABEL_19;
              }
              sub_7AE700((__int64)(qword_4F061C0 + 3), v141, dword_4F06650[0], 0, (__int64)v154);
              sub_7BC000((unsigned __int64)v154, v141, v122, v123, v124, v125);
              memset(v154, 0, sizeof(v154));
              v74 = 0;
              v79 = (_BYTE *)v136;
              *(_QWORD *)&v154[32] = *(_QWORD *)&dword_4F063F8;
              *(_QWORD *)&v154[40] = qword_4F063F0;
              v84 = sub_8988D0(v136, 0);
              goto LABEL_42;
            }
            v73 = word_4F06418[0];
          }
          else if ( word_4F06418[0] == 1 )
          {
            goto LABEL_62;
          }
        }
        if ( v73 != 9 )
        {
          sub_7B8B50((unsigned __int64)v68, 0, v69, v70, v71, v72);
          v32 = word_4F06418[0];
          if ( word_4F06418[0] == 76 )
          {
            if ( !dword_4D04408 )
              goto LABEL_17;
            sub_7B8B50((unsigned __int64)v68, 0, v28, v29, v30, v31);
            v32 = word_4F06418[0];
          }
          if ( v32 == 1 )
          {
            sub_7B8B50((unsigned __int64)v68, 0, v28, v29, v30, v31);
            v32 = word_4F06418[0];
          }
          if ( v32 != 67 && v32 != 44 && v32 != 56 && v32 != 9 )
            goto LABEL_17;
        }
        if ( (v73 & 0xFFDF) != 0x97 )
        {
LABEL_17:
          if ( v73 == 160 )
          {
            sub_7AE700((__int64)(qword_4F061C0 + 3), v141, dword_4F06650[0], 0, (__int64)v154);
            sub_7BC000((unsigned __int64)v154, v141, v92, v93, v94, v95);
            if ( !*(_DWORD *)(a1 + 92) )
              goto LABEL_49;
            v33 = 19;
          }
          else
          {
            v33 = 2;
            sub_7AE700((__int64)(qword_4F061C0 + 3), v141, dword_4F06650[0], 0, (__int64)v154);
            sub_7BC000((unsigned __int64)v154, v141, v34, v35, v36, v37);
            if ( !*(_DWORD *)(a1 + 92) )
              goto LABEL_19;
          }
          sub_6851C0(0x83Bu, &dword_4F063F8);
          v140 = 1;
          if ( v33 != 2 )
          {
LABEL_49:
            v48 = (__m128i *)sub_8B06F0(a1, &v150, 0);
            v48[3].m128i_i32[3] = HIDWORD(v150);
LABEL_50:
            sub_7AE700((__int64)(qword_4F061C0 + 3), v141, dword_4F06650[0], 0, (__int64)v48[1].m128i_i64);
            sub_7AE210((__int64)v48[1].m128i_i64);
            sub_7AE340((__int64)v48[1].m128i_i64);
            v138 = 1;
            goto LABEL_26;
          }
LABEL_19:
          v38 = 0;
          v143 = 0;
          v144 = 0;
          v145 = 0;
          if ( v151 )
            v38 = *(_QWORD *)(v151 + 16) != 0;
          v131 = v38;
          v39 = sub_866580();
          memset(v154, 0, sizeof(v154));
          sub_8AE280((__int64)v153, &v149, (int *)&v142, &v143, &v144, &v146, &v145, *(_DWORD *)(a1 + 168), v154);
          v40 = v146;
          v41 = *(_DWORD *)(a1 + 168);
          v128 = v149;
          v42 = v144;
          v43 = v142;
          *(_DWORD *)(a1 + 132) = v131;
          v44 = v150;
          *(_DWORD *)(a1 + 136) = v40;
          v45 = (__m128i *)sub_898A60(v41, v44, v43, v42, v131, v39, v40, (__int64)v153, (__int64)v128, a1);
          v46 = (unsigned int *)v154;
          v47 = v45->m128i_i64[1];
          v48 = v45;
          v49 = *(__m128i **)(v47 + 88);
          sub_729470((__int64)v49, (const __m128i *)v154);
          if ( v145
            && (v48[3].m128i_i8[9] |= 8u, v49 = v149, v101 = sub_8D4940(v149), *(_BYTE *)(v101 + 140) == 14)
            && (v51 = *(_QWORD *)(v101 + 168), *(_DWORD *)(v51 + 28) == -1) )
          {
            v102 = *(unsigned __int8 *)(v101 + 161) | 4;
            *(_BYTE *)(v101 + 161) = v102;
            LOBYTE(v51) = *(_DWORD *)(v51 + 24) == 2;
            v51 = (unsigned int)(8 * v51);
            v50 = (unsigned int)v51 | v102 & 0xFFFFFFF7;
            *(_BYTE *)(v101 + 161) = v50;
            if ( v143 )
            {
LABEL_85:
              v48[4].m128i_i8[8] |= 1u;
              v49 = v48 + 1;
              v46 = 0;
              sub_879080(v48 + 1, 0, *(_QWORD *)(a1 + 192));
              v48[3].m128i_i8[9] |= 2u;
              v138 = 1;
            }
          }
          else if ( v143 )
          {
            goto LABEL_85;
          }
          if ( word_4F06418[0] != 56 )
          {
LABEL_24:
            *(_BYTE *)(v47 + 83) &= ~0x40u;
            goto LABEL_25;
          }
          if ( v144 )
          {
            sub_6851C0(0x77Au, &dword_4F063F8);
            LOBYTE(v114) = v144 == 0;
            v48[3].m128i_i8[8] = (v144 == 0) | v48[3].m128i_i8[8] & 0xFE;
            sub_7B8B50(0x77Au, &dword_4F063F8, v114, v115, v116, v117);
            sub_6790F0((__int64)v152, 1, 0, 0, 1);
            if ( v143 )
              goto LABEL_74;
            v48[3].m128i_i8[9] |= 1u;
            if ( !*(_DWORD *)(a1 + 76) )
              goto LABEL_78;
          }
          else
          {
            v48[3].m128i_i8[8] |= 1u;
            sub_7B8B50((unsigned __int64)v49, v46, v50, v51, v52, v53);
            sub_6790F0((__int64)v152, 1, 0, 0, 1);
            if ( v143 )
            {
              v48[3].m128i_i8[8] |= 2u;
LABEL_74:
              if ( !(dword_4D047B0 | *(_DWORD *)(a1 + 76))
                && (dword_4F04C64 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) == 0) )
              {
                v48[3].m128i_i8[8] |= 4u;
                goto LABEL_78;
              }
            }
            v48[3].m128i_i8[9] |= 1u;
            if ( !*(_DWORD *)(a1 + 76) )
            {
LABEL_78:
              if ( (v48[3].m128i_i8[8] & 2) != 0 )
                v48[3].m128i_i8[9] |= 2u;
              sub_879080(v48 + 6, v152, *(_QWORD *)(a1 + 192));
              goto LABEL_24;
            }
          }
          sub_7BC270(v152);
          sub_88EE40((__int64)v48);
          goto LABEL_78;
        }
        v74 = (unsigned int *)v141;
        sub_7AE700((__int64)(qword_4F061C0 + 3), v141, dword_4F06650[0], 0, (__int64)v154);
        sub_7BC000((unsigned __int64)v154, v141, v75, v76, v77, v78);
        memset(v154, 0, sizeof(v154));
        v79 = v155;
        *(_QWORD *)&v154[32] = *(_QWORD *)&dword_4F063F8;
        *(_QWORD *)&v154[40] = qword_4F063F0;
        sub_7B8B50((unsigned __int64)v155, (unsigned int *)v141, v80, 0, v81, v82);
        v84 = 0;
LABEL_42:
        v85 = word_4F06418[0];
        LODWORD(v86) = 0;
        if ( word_4F06418[0] == 76 )
        {
          v86 = dword_4D04408;
          if ( !dword_4D04408 )
            goto LABEL_44;
          v112 = *(unsigned int *)(a1 + 92);
          if ( (_DWORD)v112 )
          {
            LODWORD(v86) = 0;
LABEL_44:
            v132 = v86;
            v87 = (__m128i *)sub_897A40(v150, 0, 0, v86, (__int64)v84, a1, (const __m128i *)v154);
            v90 = v132;
            v48 = v87;
            v91 = v129;
            if ( word_4F06418[0] == 56 )
              goto LABEL_87;
            goto LABEL_45;
          }
          if ( v84 && *((_QWORD *)v84 + 8) && (v134 = v84, v113 = sub_8670F0(), v84 = v134, v113) )
          {
            sub_867610((__int64)v79, v74);
            v85 = word_4F06418[0];
            LODWORD(v86) = 1;
            v84 = v134;
          }
          else
          {
            v135 = v84;
            sub_7B8B50((unsigned __int64)v79, v74, v83, v86, (__int64)v84, v112);
            v85 = word_4F06418[0];
            LODWORD(v86) = 1;
            v84 = v135;
          }
        }
        if ( v85 != 1 )
          goto LABEL_44;
        v103 = (unsigned int)v150;
        v133 = v86;
        *(_QWORD *)&v154[16] = *(_QWORD *)&dword_4F063F8;
        *(_QWORD *)&v154[24] = qword_4F063F0;
        v48 = (__m128i *)sub_897A40(v150, (__int64)&qword_4D04A00, 1, v86, (__int64)v84, a1, (const __m128i *)v154);
        sub_7B8B50(v103, (unsigned int *)&qword_4D04A00, v104, v105, v106, v107);
        v90 = v133;
        v91 = v129;
        if ( word_4F06418[0] == 56 )
        {
LABEL_87:
          v108 = 1914;
          if ( !(_DWORD)v90 )
          {
            v109 = (unsigned int *)*(unsigned int *)(a1 + 92);
            if ( !(_DWORD)v109 )
            {
              sub_7B8B50(0x77Au, v109, v88, v90, v91, v89);
              sub_6790F0((__int64)v153, 1, 0, 0, 0);
              v48[3].m128i_i8[9] |= 1u;
              v110 = 1;
              if ( *(_DWORD *)(a1 + 76) )
              {
                sub_7BC270(v153);
                v137 = sub_65CFF0(0, 1);
                if ( (unsigned int)sub_8DC060(v137) )
                  v48[3].m128i_i16[4] |= 0x202u;
                v48[5].m128i_i64[0] = v137;
                v110 = 1;
              }
              goto LABEL_90;
            }
            v108 = 2106;
          }
          sub_6851C0(v108, &dword_4F063F8);
          sub_7B8B50(v108, &dword_4F063F8, v118, v119, v120, v121);
          sub_6790F0((__int64)v153, 1, 0, 0, 0);
          v110 = 0;
LABEL_90:
          v111 = v110 | v48[3].m128i_i8[8] & 0xFE;
          v48[3].m128i_i8[8] = v111;
          if ( (v111 & 2) != 0 )
            v48[3].m128i_i8[9] |= 2u;
          sub_879080(v48 + 6, v153, *(_QWORD *)(a1 + 192));
        }
LABEL_45:
        *(_BYTE *)(v48->m128i_i64[1] + 83) &= ~0x40u;
LABEL_25:
        v48[3].m128i_i32[3] = HIDWORD(v150);
        if ( v138 )
          goto LABEL_50;
LABEL_26:
        sub_7BDC00();
        v57 = v140;
        if ( !v140 )
        {
          if ( v130 )
          {
            v58 = (__m128i **)v139;
            v139 = v48;
          }
          else
          {
            v58 = *(__m128i ***)(a1 + 192);
            v139 = v48;
            v130 = v48;
          }
          *v58 = v48;
        }
        if ( word_4F06418[0] != 44 && word_4F06418[0] != 67 )
        {
          sub_6851C0(0x2C2u, &dword_4F063F8);
          sub_7BE180(706, (__int64)&dword_4F063F8, v59, v60, v61, v62);
        }
        v8 = 0;
        sub_867630(v147, 0, v54, v55, v56, v57);
        v67 = sub_866C00(v147, 0, v63, v64, v65, v66);
        v3 = v147;
        if ( v147 )
        {
          if ( !*(_QWORD *)(v147 + 16) && (v48[3].m128i_i8[8] & 0x20) != 0 )
          {
            v3 = v148;
            *(_QWORD *)(v148 + 40) = *(_QWORD *)v48->m128i_i64[1];
            v19 = (__int64 *)v48->m128i_i64[1];
            if ( *((_BYTE *)v19 + 80) == 3 )
            {
              v19 = (__int64 *)v19[11];
              *(_QWORD *)(v3 + 48) = v19;
            }
          }
        }
        if ( !v67 )
          goto LABEL_9;
      }
    }
    v3 = v148;
    v4 = *(_QWORD *)(v148 + 40);
    if ( v4 )
    {
      v5 = *(_QWORD *)(v148 + 48);
      *(_QWORD *)v154 = *(_QWORD *)(v148 + 40);
      v6 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v7 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v8 = (unsigned int *)(unsigned int)(v150 + 1);
      *(__m128i *)&v154[16] = _mm_loadu_si128(&xmmword_4F06660[1]);
      *(__m128i *)&v154[32] = v6;
      *(__m128i *)&v154[48] = v7;
      *(_QWORD *)&v154[8] = *(_QWORD *)&dword_4F077C8;
      v9 = *(_BYTE *)(v4 + 73);
      LODWORD(v150) = v150 + 1;
      v10 = v9 & 1;
      if ( v5 )
      {
        v11 = 0;
        if ( !(_DWORD)v10 )
          v11 = v154;
        v12 = (unsigned int *)sub_897810(3u, (__int64)v11, v10, 0);
        *((_QWORD *)v12 + 11) = v5;
        v8 = v12;
        v13 = (__int64 *)v12;
        v14 = dword_4F04C3C;
        dword_4F04C3C = 1;
        sub_8756F0(3, (__int64)v12, (_QWORD *)v12 + 6, 0);
        dword_4F04C3C = v14;
      }
      else
      {
        v13 = sub_8978E0(*(_DWORD *)(a1 + 168), (int)v8, v10, 1, (__int64)v154, dword_4D03B80);
      }
      *((_BYTE *)v13 + 84) |= 0x40u;
      v15 = (__m128i *)sub_880AD0((__int64)v13);
      v15[3].m128i_i8[8] |= 0xB0u;
      *((_BYTE *)v13 + 84) |= 0x20u;
      v18 = dword_4F04C64;
      v15[3].m128i_i8[8] |= 0x40u;
      *(_QWORD *)(a1 + 84) = 0x100000001LL;
      v19 = qword_4F04C68;
      v3 = qword_4F04C68[0] + 776 * v18;
      *(_BYTE *)(v3 + 7) |= 1u;
      if ( v130 )
      {
        v19 = (__int64 *)v139;
        v139 = v15;
        *v19 = (__int64)v15;
      }
      else
      {
        v3 = *(_QWORD *)(a1 + 192);
        v139 = v15;
        v130 = v15;
        *(_QWORD *)v3 = v15;
      }
    }
LABEL_9:
    v20 = 67;
    --*(_BYTE *)(qword_4F061C8 + 75LL);
    if ( !(unsigned int)sub_7BE800(0x43u, v8, v3, (__int64)v19, v16, v17) )
      goto LABEL_109;
  }
  v8 = dword_4F07508;
  v20 = 440;
  sub_6851C0(0x1B8u, dword_4F07508);
LABEL_109:
  if ( word_4F06418[0] == 44 )
    sub_7B8B50(v20, v8, v21, v22, v23, v24);
  v126 = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 52LL);
  --*(_BYTE *)(v126 + 81);
  --*(_BYTE *)(v126 + 83);
  result = (unsigned int)v150;
  if ( (unsigned int)v150 > 0xFFFF )
    sub_685200(0xA4Du, (_DWORD *)(*(_QWORD *)a1 + 24LL), 0);
  *(_WORD *)(*(_QWORD *)(a1 + 192) + 42LL) = v150;
  return result;
}
