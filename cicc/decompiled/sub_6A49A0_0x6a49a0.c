// Function: sub_6A49A0
// Address: 0x6a49a0
//
__int64 __fastcall sub_6A49A0(__int64 a1, unsigned __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  bool v11; // r14
  __int32 v12; // r14d
  __int16 v13; // bx
  int v14; // eax
  char v16; // al
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int8 v21; // r11
  __m128i v22; // xmm1
  __m128i v23; // xmm2
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __m128i v26; // xmm5
  __m128i v27; // xmm6
  __m128i v28; // xmm7
  __m128i v29; // xmm0
  __int64 v30; // rsi
  __int64 v31; // r8
  __int64 v32; // r9
  __int8 v33; // al
  __int64 v34; // r15
  __m128i v35; // xmm2
  __m128i v36; // xmm3
  __m128i v37; // xmm4
  __m128i v38; // xmm5
  __m128i v39; // xmm6
  __m128i v40; // xmm7
  __m128i v41; // xmm1
  __m128i v42; // xmm2
  __m128i v43; // xmm3
  __m128i v44; // xmm4
  __m128i v45; // xmm5
  __m128i v46; // xmm6
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 j; // rax
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rax
  char i; // dl
  __m128i v55; // xmm6
  __m128i v56; // xmm7
  __m128i v57; // xmm5
  __m128i v58; // xmm6
  __m128i v59; // xmm7
  __m128i v60; // xmm6
  __m128i v61; // xmm7
  __int8 v62; // al
  __m128i v63; // xmm1
  __m128i v64; // xmm2
  __m128i v65; // xmm3
  __m128i v66; // xmm4
  __m128i v67; // xmm7
  __m128i v68; // xmm2
  __m128i v69; // xmm3
  __m128i v70; // xmm4
  __m128i v71; // xmm5
  __m128i v72; // xmm6
  __m128i v73; // xmm7
  __m128i v74; // xmm1
  __m128i v75; // xmm2
  __m128i v76; // xmm4
  __m128i v77; // xmm5
  __m128i v78; // xmm6
  __m128i v79; // xmm7
  __m128i v80; // xmm1
  __m128i v81; // xmm2
  __m128i v82; // xmm3
  __m128i v83; // xmm0
  __m128i v84; // xmm4
  __m128i v85; // xmm5
  __m128i v86; // xmm6
  __m128i v87; // xmm7
  int v88; // eax
  __int64 v89; // rax
  char v90; // dl
  __int64 v91; // rax
  const char *v92; // rdx
  int v93; // eax
  int v94; // eax
  int v95; // eax
  int v96; // eax
  __int64 v97; // rax
  int v98; // eax
  char *v99; // rdx
  __int64 v100; // rdx
  int v101; // eax
  char *v102; // rdx
  __int64 v103; // rdx
  __m128i v104; // xmm4
  __m128i v105; // xmm5
  __m128i v106; // xmm6
  __m128i v107; // xmm1
  __m128i v108; // xmm2
  __m128i v109; // xmm0
  __m128i v110; // xmm3
  __m128i v111; // xmm4
  __int64 v112; // rax
  __int64 v113; // rbx
  __int64 *v114; // rax
  int v115; // eax
  int v116; // eax
  __int64 v117; // [rsp-10h] [rbp-330h]
  __int64 v118; // [rsp-8h] [rbp-328h]
  __int8 v119; // [rsp+0h] [rbp-320h]
  char *s1; // [rsp+8h] [rbp-318h]
  char *s1a; // [rsp+8h] [rbp-318h]
  char *s1b; // [rsp+8h] [rbp-318h]
  unsigned int v123; // [rsp+18h] [rbp-308h] BYREF
  int v124; // [rsp+1Ch] [rbp-304h] BYREF
  __int64 v125; // [rsp+20h] [rbp-300h] BYREF
  __int64 v126; // [rsp+28h] [rbp-2F8h] BYREF
  __m128i v127; // [rsp+30h] [rbp-2F0h] BYREF
  __m128i v128; // [rsp+40h] [rbp-2E0h] BYREF
  __m128i v129; // [rsp+50h] [rbp-2D0h] BYREF
  __m128i v130; // [rsp+60h] [rbp-2C0h] BYREF
  __m128i v131; // [rsp+70h] [rbp-2B0h] BYREF
  __m128i v132; // [rsp+80h] [rbp-2A0h] BYREF
  __m128i v133; // [rsp+90h] [rbp-290h] BYREF
  __m128i v134; // [rsp+A0h] [rbp-280h] BYREF
  __m128i v135; // [rsp+B0h] [rbp-270h] BYREF
  __m128i v136; // [rsp+C0h] [rbp-260h] BYREF
  __m128i v137; // [rsp+D0h] [rbp-250h] BYREF
  __m128i v138; // [rsp+E0h] [rbp-240h] BYREF
  __m128i v139; // [rsp+F0h] [rbp-230h] BYREF
  __m128i v140; // [rsp+100h] [rbp-220h] BYREF
  __m128i v141; // [rsp+110h] [rbp-210h] BYREF
  __m128i v142; // [rsp+120h] [rbp-200h] BYREF
  __m128i v143; // [rsp+130h] [rbp-1F0h] BYREF
  __m128i v144; // [rsp+140h] [rbp-1E0h] BYREF
  __m128i v145; // [rsp+150h] [rbp-1D0h] BYREF
  __m128i v146; // [rsp+160h] [rbp-1C0h] BYREF
  __m128i v147; // [rsp+170h] [rbp-1B0h] BYREF
  __m128i v148; // [rsp+180h] [rbp-1A0h] BYREF
  __m128i v149; // [rsp+190h] [rbp-190h] BYREF
  __m128i v150; // [rsp+1A0h] [rbp-180h]
  __m128i v151; // [rsp+1B0h] [rbp-170h]
  __m128i v152; // [rsp+1C0h] [rbp-160h]
  __m128i v153; // [rsp+1D0h] [rbp-150h]
  __m128i v154; // [rsp+1E0h] [rbp-140h]
  __m128i v155; // [rsp+1F0h] [rbp-130h]
  __m128i v156; // [rsp+200h] [rbp-120h]
  __m128i v157; // [rsp+210h] [rbp-110h]
  __m128i v158; // [rsp+220h] [rbp-100h]
  __m128i v159; // [rsp+230h] [rbp-F0h]
  __m128i v160; // [rsp+240h] [rbp-E0h]
  __m128i v161; // [rsp+250h] [rbp-D0h]
  __m128i v162; // [rsp+260h] [rbp-C0h]
  __m128i v163; // [rsp+270h] [rbp-B0h]
  __m128i v164; // [rsp+280h] [rbp-A0h]
  __m128i v165; // [rsp+290h] [rbp-90h]
  __m128i v166; // [rsp+2A0h] [rbp-80h]
  __m128i v167; // [rsp+2B0h] [rbp-70h]
  __m128i v168; // [rsp+2C0h] [rbp-60h]
  __m128i v169; // [rsp+2D0h] [rbp-50h]
  __m128i v170; // [rsp+2E0h] [rbp-40h]

  v7 = a2;
  v8 = a1;
  v124 = 0;
  if ( a1 )
  {
    a2 = (unsigned __int64)&v127;
    sub_6F8AB0(a1, (unsigned int)&v127, 0, 0, (unsigned int)&v126, (unsigned int)&v123, 0);
    a1 = v117;
    a5 = v118;
    v125 = v126;
    v10 = qword_4D03C50;
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
    {
LABEL_3:
      v11 = (v128.m128i_i8[2] & 8) != 0;
LABEL_4:
      if ( dword_4F077C4 == 2 && (unsigned int)sub_68FE10(&v127, 1, 1) && v128.m128i_i8[0] != 4 )
        sub_84EC30(11, 1, 0, 0, 1, (unsigned int)&v127, 0, (__int64)&v125, v123, 0, 0, v7, 0, 0, (__int64)&v124);
      if ( v124 )
        goto LABEL_6;
      v22 = _mm_loadu_si128(&v128);
      v23 = _mm_loadu_si128(&v129);
      v119 = v128.m128i_i8[1];
      v24 = _mm_loadu_si128(&v130);
      v25 = _mm_loadu_si128(&v131);
      v26 = _mm_loadu_si128(&v132);
      v149 = _mm_loadu_si128(&v127);
      v27 = _mm_loadu_si128(&v133);
      v28 = _mm_loadu_si128(&v134);
      v150 = v22;
      v29 = _mm_loadu_si128(&v135);
      v151 = v23;
      v152 = v24;
      v153 = v25;
      v154 = v26;
      v155 = v27;
      v156 = v28;
      v157 = v29;
      if ( v128.m128i_i8[0] == 2 )
      {
        v35 = _mm_loadu_si128(&v137);
        v36 = _mm_loadu_si128(&v138);
        v37 = _mm_loadu_si128(&v139);
        v38 = _mm_loadu_si128(&v140);
        v39 = _mm_loadu_si128(&v141);
        v158 = _mm_loadu_si128(&v136);
        v40 = _mm_loadu_si128(&v142);
        v41 = _mm_loadu_si128(&v143);
        v159 = v35;
        v160 = v36;
        v42 = _mm_loadu_si128(&v144);
        v43 = _mm_loadu_si128(&v145);
        v161 = v37;
        v44 = _mm_loadu_si128(&v146);
        v162 = v38;
        v45 = _mm_loadu_si128(&v147);
        v163 = v39;
        v46 = _mm_loadu_si128(&v148);
        v164 = v40;
        v165 = v41;
        v166 = v42;
        v167 = v43;
        v168 = v44;
        v169 = v45;
        v170 = v46;
      }
      else if ( v128.m128i_i8[0] == 5 || v128.m128i_i8[0] == 1 )
      {
        v158.m128i_i64[0] = v136.m128i_i64[0];
      }
      v30 = 47;
      sub_6F69D0(&v127, 47);
      if ( HIDWORD(qword_4F077B4) )
      {
        v30 = 0;
        sub_6FA330(&v127, 0);
      }
      if ( v128.m128i_i8[0] == 1 && *(_BYTE *)(v136.m128i_i64[0] + 24) == 1 && *(_BYTE *)(v136.m128i_i64[0] + 56) == 3 )
      {
        v34 = *(_QWORD *)(v136.m128i_i64[0] + 72);
        sub_6E70E0(v34, v7);
        v126 = *(_QWORD *)(v34 + 28);
        goto LABEL_30;
      }
      v33 = v128.m128i_i8[1];
      if ( v128.m128i_i8[1] != 1 )
      {
LABEL_25:
        if ( v33 == 3 )
        {
          sub_6F5FA0(&v127, &v126, 0, 0, v31, v32);
          v63 = _mm_loadu_si128(&v129);
          v64 = _mm_loadu_si128(&v130);
          v65 = _mm_loadu_si128(&v131);
          v66 = _mm_loadu_si128(&v132);
          *(__m128i *)v7 = _mm_loadu_si128(&v127);
          v67 = _mm_loadu_si128(&v128);
          *(__m128i *)(v7 + 32) = v63;
          *(__m128i *)(v7 + 16) = v67;
          *(__m128i *)(v7 + 48) = v64;
          *(__m128i *)(v7 + 64) = v65;
          *(__m128i *)(v7 + 80) = v66;
        }
        else
        {
          if ( v128.m128i_i8[0] == 4 )
          {
            sub_6EE880(&v127, &v126);
            v68 = _mm_loadu_si128(&v128);
            v69 = _mm_loadu_si128(&v129);
            v70 = _mm_loadu_si128(&v130);
            v71 = _mm_loadu_si128(&v131);
            v72 = _mm_loadu_si128(&v132);
            *(__m128i *)v7 = _mm_loadu_si128(&v127);
            v73 = _mm_loadu_si128(&v133);
            v74 = _mm_loadu_si128(&v134);
            *(__m128i *)(v7 + 16) = v68;
            v62 = v128.m128i_i8[0];
            v75 = _mm_loadu_si128(&v135);
            *(__m128i *)(v7 + 32) = v69;
            *(__m128i *)(v7 + 48) = v70;
            *(__m128i *)(v7 + 64) = v71;
            *(__m128i *)(v7 + 80) = v72;
            *(__m128i *)(v7 + 96) = v73;
            *(__m128i *)(v7 + 112) = v74;
            *(__m128i *)(v7 + 128) = v75;
            if ( v62 != 2 )
              goto LABEL_75;
            goto LABEL_81;
          }
          if ( v128.m128i_i8[0] != 2 )
          {
            if ( !v128.m128i_i8[0] )
            {
LABEL_29:
              sub_6E6260(v7);
LABEL_30:
              if ( *(_BYTE *)(v7 + 16) == 2
                && *(_BYTE *)(v7 + 317) == 7
                && !v8
                && dword_4F04C64 != -1
                && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0
                && (*(_BYTE *)(v7 + 20) & 4) != 0 )
              {
                *(_QWORD *)(v7 + 328) = sub_7CADA0(v7 + 144, v7 + 24);
              }
LABEL_6:
              if ( v11 )
                *(_BYTE *)(v7 + 18) |= 0x10u;
              goto LABEL_8;
            }
LABEL_65:
            v53 = v127.m128i_i64[0];
            for ( i = *(_BYTE *)(v127.m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v53 + 140) )
              v53 = *(_QWORD *)(v53 + 160);
            if ( i )
              sub_6E68E0(158, &v127);
            goto LABEL_29;
          }
          if ( v146.m128i_i8[13] != 12 || v147.m128i_i8[0] != 1 )
            goto LABEL_65;
          sub_6F3DD0(&v127, 1, 0);
          sub_6FF5E0(&v127, &v126);
          v55 = _mm_loadu_si128(&v128);
          v56 = _mm_loadu_si128(&v129);
          *(__m128i *)v7 = _mm_loadu_si128(&v127);
          v57 = _mm_loadu_si128(&v130);
          *(__m128i *)(v7 + 16) = v55;
          v58 = _mm_loadu_si128(&v131);
          *(__m128i *)(v7 + 32) = v56;
          v59 = _mm_loadu_si128(&v132);
          *(__m128i *)(v7 + 48) = v57;
          *(__m128i *)(v7 + 64) = v58;
          *(__m128i *)(v7 + 80) = v59;
        }
        v60 = _mm_loadu_si128(&v134);
        v61 = _mm_loadu_si128(&v135);
        v62 = v128.m128i_i8[0];
        *(__m128i *)(v7 + 96) = _mm_loadu_si128(&v133);
        *(__m128i *)(v7 + 112) = v60;
        *(__m128i *)(v7 + 128) = v61;
        if ( v62 != 2 )
        {
LABEL_75:
          if ( v62 == 5 || v62 == 1 )
            *(_QWORD *)(v7 + 144) = v136.m128i_i64[0];
          goto LABEL_77;
        }
LABEL_81:
        v76 = _mm_loadu_si128(&v137);
        v77 = _mm_loadu_si128(&v138);
        v78 = _mm_loadu_si128(&v139);
        v79 = _mm_loadu_si128(&v140);
        v80 = _mm_loadu_si128(&v145);
        *(__m128i *)(v7 + 144) = _mm_loadu_si128(&v136);
        v81 = _mm_loadu_si128(&v146);
        v82 = _mm_loadu_si128(&v141);
        *(__m128i *)(v7 + 160) = v76;
        v83 = _mm_loadu_si128(&v147);
        v84 = _mm_loadu_si128(&v142);
        *(__m128i *)(v7 + 176) = v77;
        *(__m128i *)(v7 + 192) = v78;
        v85 = _mm_loadu_si128(&v143);
        v86 = _mm_loadu_si128(&v144);
        *(__m128i *)(v7 + 208) = v79;
        v87 = _mm_loadu_si128(&v148);
        *(__m128i *)(v7 + 224) = v82;
        *(__m128i *)(v7 + 240) = v84;
        *(__m128i *)(v7 + 256) = v85;
        *(__m128i *)(v7 + 272) = v86;
        *(__m128i *)(v7 + 288) = v80;
        *(__m128i *)(v7 + 304) = v81;
        *(__m128i *)(v7 + 320) = v83;
        *(__m128i *)(v7 + 336) = v87;
LABEL_77:
        sub_6E5070(v7, &v149);
        goto LABEL_30;
      }
      if ( (unsigned int)sub_6ED0A0(&v127) )
      {
        v33 = v128.m128i_i8[1];
        goto LABEL_25;
      }
      v88 = dword_4F077C4;
      if ( dword_4F077C4 == 1 )
      {
        if ( (unsigned int)sub_8D3410(v127.m128i_i64[0]) )
        {
          if ( (unsigned int)sub_6E53E0(5, 178, &v125) )
            sub_684B30(0xB2u, &v125);
          sub_6FB570(&v127);
LABEL_114:
          v104 = _mm_loadu_si128(&v128);
          v105 = _mm_loadu_si128(&v129);
          v106 = _mm_loadu_si128(&v130);
          v107 = _mm_loadu_si128(&v131);
          v108 = _mm_loadu_si128(&v132);
          *(__m128i *)v7 = _mm_loadu_si128(&v127);
          v109 = _mm_loadu_si128(&v133);
          v110 = _mm_loadu_si128(&v134);
          *(__m128i *)(v7 + 16) = v104;
          v62 = v128.m128i_i8[0];
          v111 = _mm_loadu_si128(&v135);
          *(__m128i *)(v7 + 32) = v105;
          *(__m128i *)(v7 + 48) = v106;
          *(__m128i *)(v7 + 64) = v107;
          *(__m128i *)(v7 + 80) = v108;
          *(__m128i *)(v7 + 96) = v109;
          *(__m128i *)(v7 + 112) = v110;
          *(__m128i *)(v7 + 128) = v111;
          if ( v62 != 2 )
            goto LABEL_75;
          goto LABEL_81;
        }
        v88 = dword_4F077C4;
      }
      if ( v88 == 2 && v119 == 2 && (unsigned int)sub_8D3A70(v127.m128i_i64[0]) )
      {
        v30 = (__int64)&v125;
        sub_69D070(0x51Cu, &v125);
      }
      if ( (!qword_4D03C50 || *(char *)(qword_4D03C50 + 18LL) >= 0) && v128.m128i_i8[0] == 1 )
      {
        v89 = v136.m128i_i64[0];
        if ( v136.m128i_i64[0] )
        {
          v90 = *(_BYTE *)(v136.m128i_i64[0] + 24);
          if ( v90 != 1 )
          {
LABEL_90:
            if ( v90 == 3 )
            {
              v91 = *(_QWORD *)(v89 + 56);
              if ( v91 )
              {
                if ( (*(_BYTE *)(v91 - 8) & 0x10) != 0 )
                {
                  v92 = *(const char **)(v91 + 8);
                  if ( v92 )
                  {
                    if ( !strcmp(*(const char **)(v91 + 8), "threadIdx") )
                      goto LABEL_99;
                    s1 = *(char **)(v91 + 8);
                    v93 = strcmp(v92, "blockIdx");
                    v92 = s1;
                    if ( !v93
                      || (v94 = strcmp(s1, "blockDim"), v92 = s1, !v94)
                      || (v95 = strcmp(s1, "gridDim"), v92 = s1, !v95)
                      || (v30 = (__int64)"warpSize", v96 = strcmp(s1, "warpSize"), v92 = s1, !v96) )
                    {
LABEL_99:
                      v30 = (__int64)&v125;
                      sub_6851A0(0xDD0u, &v125, (__int64)v92);
                    }
                  }
                }
              }
            }
            goto LABEL_100;
          }
          if ( *(_BYTE *)(v136.m128i_i64[0] + 56) == 94 )
          {
            v89 = *(_QWORD *)(v136.m128i_i64[0] + 72);
            if ( v89 )
            {
              v90 = *(_BYTE *)(v89 + 24);
              goto LABEL_90;
            }
          }
        }
      }
LABEL_100:
      if ( dword_4F04C58 != -1 )
      {
        v97 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
        if ( v97 )
        {
          if ( (*(_BYTE *)(v97 + 198) & 0x10) != 0
            && (!qword_4D03C50 || *(char *)(qword_4D03C50 + 18LL) >= 0)
            && v128.m128i_i8[0] == 1 )
          {
            if ( v136.m128i_i64[0] )
            {
              s1a = (char *)v136.m128i_i64[0];
              v98 = sub_8D2FF0(*(_QWORD *)v136.m128i_i64[0], v30);
              v99 = s1a;
              if ( v98 || (v116 = sub_8D3030(*(_QWORD *)s1a), v99 = s1a, v116) )
              {
                if ( v99[24] == 3
                  && (v100 = *((_QWORD *)v99 + 7)) != 0
                  && ((s1b = (char *)v100, v101 = sub_8D2FF0(*(_QWORD *)(v100 + 120), v30), v102 = s1b, v101)
                   || (v115 = sub_8D3030(*((_QWORD *)s1b + 15)), v102 = s1b, v115))
                  && (v103 = *((_QWORD *)v102 + 1)) != 0 )
                {
                  sub_6851A0(0xDE2u, &v125, v103);
                }
                else
                {
                  sub_6851C0(0xDE3u, &v125);
                }
              }
            }
          }
        }
      }
      sub_6FF5E0(&v127, &v126);
      goto LABEL_114;
    }
  }
  else
  {
    v9 = dword_4F06650[0];
    v126 = *(_QWORD *)&dword_4F063F8;
    v125 = *(_QWORD *)&dword_4F063F8;
    v10 = qword_4D03C50;
    v123 = dword_4F06650[0];
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
    {
      sub_7B8B50(0, a2, dword_4F06650[0], a4);
      a2 = dword_4D04368;
      if ( !dword_4D04368 || word_4F06418[0] != 76 )
      {
        v150.m128i_i8[2] &= ~2u;
        sub_69ED20((__int64)&v127, 0, 18, a3 | 0x44u);
        if ( (v128.m128i_i8[2] & 1) == 0 )
        {
          v11 = (v128.m128i_i8[2] & 8) != 0;
          goto LABEL_4;
        }
        sub_82F8F0(&v149, (v150.m128i_i8[2] & 2) != 0, &v127);
        v21 = v128.m128i_i8[2];
        goto LABEL_41;
      }
      goto LABEL_57;
    }
  }
  v16 = *(_BYTE *)(v10 + 16);
  if ( v16 )
  {
    if ( v16 != 1 )
    {
      if ( v8 )
        goto LABEL_3;
      sub_7B8B50(a1, a2, v9, a4);
      if ( !dword_4D04368 || word_4F06418[0] != 76 )
      {
        v150.m128i_i8[2] &= ~2u;
        sub_69ED20((__int64)&v127, 0, 18, a3 | 0x44u);
        v21 = v128.m128i_i8[2];
        if ( (v128.m128i_i8[2] & 1) != 0 )
        {
          sub_82F8F0(&v149, (v150.m128i_i8[2] & 2) != 0, &v127);
          v21 = v128.m128i_i8[2];
        }
LABEL_41:
        v11 = (v21 & 8) != 0;
        goto LABEL_4;
      }
LABEL_57:
      if ( dword_4F04C58 != -1 )
      {
        for ( j = *(_QWORD *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        if ( (*(_BYTE *)(*(_QWORD *)(j + 168) + 16LL) & 1) != 0 )
        {
          v112 = sub_72CBE0(a1, a2, v17, v18, v19, v20);
          v113 = sub_72D2E0(v112, 0);
          v114 = (__int64 *)sub_726700(16);
          a2 = v7;
          *v114 = v113;
          v50 = (__int64)v114;
          sub_6E70E0(v114, v7);
          v52 = (unsigned int)dword_4D04964;
          if ( dword_4D04964 )
          {
            a2 = 669;
            v50 = byte_4F07472[0];
            sub_6E5C80(byte_4F07472[0], 669, &dword_4F063F8);
          }
          goto LABEL_64;
        }
      }
      if ( (unsigned int)sub_6E5430(a1, a2, v17, v18, v19, v20) )
      {
        a2 = (unsigned __int64)&dword_4F063F8;
        sub_6851C0(0x29Eu, &dword_4F063F8);
      }
LABEL_63:
      v50 = v7;
      sub_6E6260(v7);
LABEL_64:
      v12 = qword_4F063F0;
      v13 = WORD2(qword_4F063F0);
      sub_7B8B50(v50, a2, v51, v52);
      goto LABEL_9;
    }
    if ( (unsigned int)sub_6E5430(a1, a2, v9, a4, a5, a6) )
    {
      a2 = (unsigned __int64)&v125;
      a1 = 60;
      sub_6851C0(0x3Cu, &v125);
    }
  }
  else if ( (unsigned int)sub_6E5430(a1, a2, v9, a4, a5, a6) )
  {
    a2 = (unsigned __int64)&v125;
    a1 = 58;
    sub_6851C0(0x3Au, &v125);
  }
  if ( !v8 )
  {
    sub_7B8B50(a1, a2, v47, v48);
    if ( dword_4D04368 && word_4F06418[0] == 76 )
      goto LABEL_63;
    v150.m128i_i8[2] &= ~2u;
    sub_69ED20((__int64)&v127, 0, 18, a3 | 0x44u);
    if ( (v128.m128i_i8[2] & 1) != 0 )
      sub_82F8F0(&v149, (v150.m128i_i8[2] & 2) != 0, &v127);
  }
  sub_6E6260(v7);
  sub_6E6450(&v127);
LABEL_8:
  v12 = v131.m128i_i32[3];
  v13 = v132.m128i_i16[0];
LABEL_9:
  v14 = v125;
  *(_DWORD *)(v7 + 76) = v12;
  *(_WORD *)(v7 + 80) = v13;
  *(_DWORD *)(v7 + 68) = v14;
  *(_WORD *)(v7 + 72) = WORD2(v125);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v7 + 68);
  unk_4F061D8 = *(_QWORD *)(v7 + 76);
  sub_6E3280(v7, &v126);
  sub_6E3BA0(v7, &v126, v123, 0);
  return sub_6E26D0(1, v7);
}
