// Function: sub_66F9E0
// Address: 0x66f9e0
//
__int64 __fastcall sub_66F9E0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        unsigned int *a7,
        unsigned int *a8,
        __int64 a9)
{
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 *v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rax
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // r8
  __int64 v26; // rax
  unsigned __int16 v27; // ax
  unsigned int v28; // r12d
  char v29; // al
  unsigned int *v30; // rdx
  __int64 v31; // r12
  __int64 v32; // rax
  __m128i v33; // xmm5
  __m128i v34; // xmm6
  __m128i v35; // xmm7
  __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // rcx
  char v39; // dl
  char v40; // al
  char v41; // al
  __int64 v42; // r14
  char v43; // al
  __int64 v44; // r13
  _BOOL4 v45; // r9d
  _BOOL4 v46; // r8d
  int v47; // r15d
  __m128i *v48; // rdi
  __int64 v49; // rsi
  _QWORD *v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // r8
  unsigned __int8 v56; // al
  __m128i v58; // xmm5
  __m128i v59; // xmm2
  __m128i v60; // xmm1
  char v61; // al
  __int64 v62; // rcx
  int v63; // eax
  __int64 v64; // rdi
  unsigned __int8 v65; // cl
  __int64 v66; // rax
  __int64 v67; // rax
  __m128i v68; // xmm2
  __m128i v69; // xmm3
  __m128i v70; // xmm4
  __m128i *v71; // rax
  __int64 v72; // rax
  __int64 v73; // r15
  __m128i *v74; // rax
  __m128i v75; // xmm7
  __m128i v76; // xmm4
  __m128i v77; // xmm0
  __int64 v78; // r12
  __int64 v79; // r9
  char v80; // dl
  __int64 v81; // rsi
  __int64 v82; // r8
  __int64 v83; // rax
  unsigned int v84; // r8d
  __int64 v85; // rdi
  __int64 v86; // rdx
  int v87; // r12d
  _QWORD *v88; // r15
  int v89; // r9d
  __int64 v90; // rax
  __m128i v91; // xmm6
  __m128i v92; // xmm7
  __int64 v93; // rax
  __m128i v94; // xmm5
  int v95; // edx
  __int64 v96; // rdi
  __m128i v97; // xmm3
  __m128i v98; // xmm6
  __m128i v99; // xmm7
  __int64 v100; // rdi
  _BOOL4 v101; // eax
  __m128i v102; // xmm6
  __m128i v103; // xmm7
  __m128i v104; // xmm4
  char v105; // al
  __int64 v106; // rdi
  int v107; // eax
  __int64 v108; // r13
  __int64 v109; // rax
  __int64 v110; // rsi
  __int64 v111; // rax
  int v112; // eax
  __int64 v113; // rax
  __int64 v114; // rdi
  __int64 v115; // rdi
  __m128i v116; // xmm3
  __m128i v117; // xmm2
  __m128i v118; // xmm1
  __int64 v119; // rax
  char v120; // dl
  __int64 v121; // rax
  int v122; // eax
  __int64 v123; // [rsp+10h] [rbp-1A0h]
  __int64 v124; // [rsp+10h] [rbp-1A0h]
  __int64 v125; // [rsp+10h] [rbp-1A0h]
  __int64 v126; // [rsp+10h] [rbp-1A0h]
  unsigned int v127; // [rsp+18h] [rbp-198h]
  int v128; // [rsp+1Ch] [rbp-194h]
  int v131; // [rsp+48h] [rbp-168h]
  int v132; // [rsp+4Ch] [rbp-164h]
  __int64 v133; // [rsp+50h] [rbp-160h]
  __int64 v135; // [rsp+68h] [rbp-148h]
  char v136; // [rsp+72h] [rbp-13Eh]
  unsigned __int8 v137; // [rsp+73h] [rbp-13Dh]
  unsigned int v138; // [rsp+74h] [rbp-13Ch]
  char *v139; // [rsp+78h] [rbp-138h]
  unsigned int v140; // [rsp+78h] [rbp-138h]
  __int64 v141; // [rsp+78h] [rbp-138h]
  __int64 v142; // [rsp+78h] [rbp-138h]
  __int64 v143; // [rsp+78h] [rbp-138h]
  __int64 v144; // [rsp+78h] [rbp-138h]
  __int64 v145; // [rsp+78h] [rbp-138h]
  unsigned int v146; // [rsp+80h] [rbp-130h]
  __int64 v147; // [rsp+80h] [rbp-130h]
  int v148; // [rsp+90h] [rbp-120h]
  __int64 v149; // [rsp+90h] [rbp-120h]
  __int64 v150; // [rsp+90h] [rbp-120h]
  char v151; // [rsp+90h] [rbp-120h]
  __int64 v152; // [rsp+90h] [rbp-120h]
  __int64 v153; // [rsp+90h] [rbp-120h]
  __int64 v154; // [rsp+90h] [rbp-120h]
  unsigned int v155; // [rsp+98h] [rbp-118h]
  int v156; // [rsp+98h] [rbp-118h]
  int v157; // [rsp+9Ch] [rbp-114h] BYREF
  int v158; // [rsp+A4h] [rbp-10Ch] BYREF
  unsigned int v159; // [rsp+A8h] [rbp-108h] BYREF
  int v160; // [rsp+ACh] [rbp-104h] BYREF
  __int64 v161; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v162; // [rsp+B8h] [rbp-F8h] BYREF
  __int64 v163; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v164; // [rsp+C8h] [rbp-E8h] BYREF
  __int64 v165; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v166; // [rsp+D8h] [rbp-D8h] BYREF
  __m128i v167; // [rsp+E0h] [rbp-D0h] BYREF
  __m128i v168; // [rsp+F0h] [rbp-C0h]
  __m128i v169; // [rsp+100h] [rbp-B0h]
  __m128i v170; // [rsp+110h] [rbp-A0h]
  _BYTE v171[88]; // [rsp+120h] [rbp-90h] BYREF
  _BYTE v172[56]; // [rsp+178h] [rbp-38h] BYREF

  v155 = a4;
  v157 = a3;
  v158 = 0;
  v161 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v160 = 0;
  v162 = sub_724DC0(a1, a2, v10, v11, v12, v13);
  v159 = dword_4F04C5C;
  *a7 = 0;
  v14 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  if ( *(_BYTE *)(v14 + 4) == 6 )
  {
    v128 = 1;
    v135 = *(_QWORD *)(v14 + 208);
    v137 = *(_BYTE *)(v14 + 5) & 3;
  }
  else
  {
    v128 = 0;
    v137 = 0;
    v135 = 0;
  }
  memset(v171, 0, sizeof(v171));
  v15 = (__int64)v172;
  v16 = (__int64 *)dword_4F06650[0];
  v127 = dword_4F06650[0];
  v164 = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)&v171[32] = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(v172, dword_4F06650[0], dword_4F06650, 0);
  if ( dword_4F077C4 == 2 )
  {
    v30 = (unsigned int *)&unk_4F07778;
    if ( unk_4F07778 <= 201102 )
    {
      v30 = &dword_4F07774;
      if ( !dword_4F07774 )
        goto LABEL_4;
      if ( word_4F06418[0] == 151 )
        goto LABEL_34;
      if ( word_4F06418[0] == 101 )
        goto LABEL_150;
    }
    else if ( word_4F06418[0] == 151 || word_4F06418[0] == 101 )
    {
      goto LABEL_34;
    }
    v146 = 0;
    goto LABEL_5;
  }
LABEL_4:
  v16 = (__int64 *)dword_4F077BC;
  v146 = dword_4F077BC;
  if ( !dword_4F077BC )
    goto LABEL_5;
  if ( qword_4F077A8 <= 0x9DCFu )
  {
    v146 = 0;
  }
  else
  {
    v30 = (unsigned int *)word_4F06418[0];
    if ( word_4F06418[0] == 151 || (v146 = 0, word_4F06418[0] == 101) )
    {
      if ( dword_4F077C4 != 2 )
      {
LABEL_33:
        v16 = &v164;
        v15 = 2368;
        sub_684B30(2368, &v164);
LABEL_34:
        sub_7B8B50(v15, v16, v30, v17);
        v146 = 1;
        goto LABEL_5;
      }
      if ( unk_4F07778 > 201102 )
        goto LABEL_34;
      v30 = &dword_4F07774;
LABEL_150:
      if ( dword_4F07774 )
        goto LABEL_34;
      goto LABEL_33;
    }
  }
LABEL_5:
  *(_QWORD *)(a1 + 216) = sub_5CC190(2);
  if ( dword_4F077C4 == 2 )
  {
    if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 )
    {
      v138 = 1;
      v165 = *(_QWORD *)&dword_4F063F8;
      *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
      goto LABEL_8;
    }
    v138 = sub_7C0F00(v155 == 0 ? 0x4000 : 17408, 0);
  }
  else
  {
    v138 = word_4F06418[0] == 1;
  }
  v165 = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  if ( v138 )
  {
LABEL_8:
    LODWORD(v18) = unk_4F0776C;
    if ( unk_4F0776C )
      v18 = (*(_QWORD *)(a1 + 8) >> 3) & 1LL;
    LODWORD(v166) = v18;
    *a7 = 1;
    v19 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v20 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v167 = _mm_loadu_si128(xmmword_4F06660);
    v21 = _mm_loadu_si128(&xmmword_4F06660[3]);
    *(_QWORD *)&v171[16] = *(_QWORD *)&dword_4F063F8;
    v168 = v19;
    v169 = v20;
    v167.m128i_i64[1] = unk_4F077C8;
    v170 = v21;
    v139 = (char *)sub_668EE0(
                     6u,
                     &v167,
                     &v166,
                     0,
                     &v157,
                     (a2 & 0x40) != 0,
                     (a2 >> 24) & 1,
                     (int *)&v159,
                     &v163,
                     &v160,
                     (__int64)v171);
    v22 = 6;
    v23 = word_4F06418[0];
    v24 = sub_667FD0(word_4F06418[0], 6, (a2 & 0x40) != 0, (a2 >> 24) & 1);
    v25 = (__int64)v139;
    v148 = v24;
    if ( v139 )
    {
      v23 = v155;
      if ( v155 )
      {
        if ( *((_QWORD *)v139 + 9) && v139[81] < 0 )
          v25 = *((_QWORD *)v139 + 9);
      }
    }
    v131 = v163;
    if ( (_DWORD)v163 )
    {
      v22 = v137;
      if ( v159 != dword_4F04C5C )
        v22 = 0;
      v26 = 0;
      if ( v159 == dword_4F04C5C )
        v26 = v135;
      v137 = v22;
      v135 = v26;
LABEL_21:
      if ( !v148 )
      {
        v132 = dword_4D0449C;
        if ( !dword_4D0449C )
        {
          v140 = 0;
          v131 = 0;
          v133 = 0;
          v27 = word_4F06418[0];
          v136 = 13;
LABEL_73:
          v29 = v27 == 75;
          v28 = 0;
LABEL_74:
          v39 = v29 << 7;
          v40 = (v29 << 7) | *(_BYTE *)(a1 + 127) & 0x7F;
          *(_BYTE *)(a1 + 127) = v39 | *(_BYTE *)(a1 + 127) & 0x7F;
          if ( !v25 )
          {
            v148 = v28;
LABEL_213:
            if ( !v155 || (v168.m128i_i8[1] & 0x20) != 0 )
            {
LABEL_215:
              v78 = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)v159 + 184);
              v42 = sub_7259C0(2);
              v80 = *(_BYTE *)(v42 + 141) | 0x20;
              *(_BYTE *)(v42 + 141) = v80;
              v81 = qword_4F04C68[0];
              v82 = (int)v159;
              v83 = qword_4F04C68[0] + 776LL * (int)v159;
              if ( (*(_BYTE *)(v83 + 6) & 6) != 0 )
                *(_BYTE *)(v42 + 162) |= 0x40u;
              if ( v78 && (*(_BYTE *)(v78 - 8) & 1) != 0 )
                *(_QWORD *)(v42 + 40) = v78;
              if ( (dword_4F077C4 == 2 || dword_4D04964) && !(v146 | v148 | v140) && (v168.m128i_i8[1] & 0x20) == 0 )
              {
                if ( dword_4F077C4 == 2 )
                {
                  v85 = 7;
                  v84 = 102;
                  if ( word_4F06418[0] == 55 && dword_4D044A8 && *(_BYTE *)(v81 + 776LL * dword_4F04C64 + 4) == 6 )
                  {
                    v85 = 8;
                    v84 = 3265;
                    *(_BYTE *)(v42 + 141) = v80 & 0xDF;
                  }
                }
                else
                {
                  v84 = 102;
                  v85 = unk_4F07471;
                }
                sub_684AA0(v85, v84, &v167.m128i_u64[1]);
                v82 = (int)v159;
                v83 = qword_4F04C68[0] + 776LL * (int)v159;
              }
              v86 = *(unsigned __int16 *)(v42 + 160);
              *(_QWORD *)(v42 + 168) = 0;
              LOWORD(v86) = v86 & 0xF700 | 0x805;
              *(_WORD *)(v42 + 160) = v86;
              if ( *(_BYTE *)(v83 + 4) == 1 )
                *(_BYTE *)(v42 + 141) |= 0x80u;
              if ( v138 )
              {
                v87 = 0;
                v44 = sub_647630(6u, (__int64)&v167, (unsigned int)v82, 0);
                *a7 = 1;
                sub_877D80(v42, v44);
                *(_QWORD *)(v44 + 88) = v42;
              }
              else
              {
                v87 = 1;
                v44 = sub_87F680(6, &dword_4F063F8, v86, 0, v82, v79);
                sub_877D80(v42, v44);
                sub_877D70(v42);
                v90 = v167.m128i_i64[1];
                *(_QWORD *)(v44 + 88) = v42;
                *(_BYTE *)(v42 + 88) &= ~4u;
                *(_BYTE *)(v42 + 162) |= 8u;
                *(_QWORD *)(v42 + 64) = v90;
              }
              v88 = *(_QWORD **)(v44 + 96);
              sub_85E280(v44, v159);
              if ( dword_4F077C4 == 2 )
              {
                if ( v135 )
                {
                  sub_877E20(v44, v42, v135);
                  sub_878FA0(v44);
                  if ( dword_4F04C58 != -1 )
                    goto LABEL_232;
                }
                else
                {
                  if ( (unsigned __int8)(*(_BYTE *)(qword_4F04C68[0] + 776LL * (int)v159 + 4) - 3) <= 1u )
                    sub_877E90(v44, v42);
                  if ( dword_4F04C58 != -1 )
                    goto LABEL_235;
                }
                if ( !unk_4F04C38 )
                  sub_66A6A0(v42);
              }
LABEL_232:
              if ( (*(_BYTE *)(v44 + 81) & 0x20) == 0 && v135 && dword_4D0449C )
              {
                if ( sub_5F2660() )
                {
                  v119 = sub_87E420(6);
                  v88[4] = v119;
                  *(_BYTE *)(v42 + 162) |= 0x20u;
                  *(_DWORD *)(v119 + 64) = v127;
                  *(_BYTE *)(v42 + 162) |= 0x10u;
                  *(_QWORD *)(v119 + 176) = v44;
                }
                else if ( !v87 )
                {
                  if ( (unsigned int)sub_5F26C0() )
                  {
                    *(_BYTE *)(v42 + 162) |= 0x10u;
                    sub_896740(v44, *(_QWORD *)(v44 + 64), v127);
                    v113 = v88[3];
                    if ( v113 )
                    {
                      v114 = *(_QWORD *)(*(_QWORD *)(v113 + 96) + 32LL);
                      if ( v114 )
                      {
                        if ( (v146 & 1) == 0 )
                        {
                          v115 = sub_892400(v114);
                          if ( *(_QWORD *)(v115 + 8) )
                          {
                            sub_7BC160(v115);
                            v148 = 1;
                            v88[5] = *(_QWORD *)&dword_4F063F8;
                            *(_BYTE *)(v42 + 88) = v137 & 3 | *(_BYTE *)(v42 + 88) & 0xFC;
                            if ( v138 )
                              goto LABEL_237;
                            goto LABEL_241;
                          }
                        }
                      }
                    }
                  }
                }
              }
LABEL_235:
              *(_BYTE *)(v42 + 88) = v137 & 3 | *(_BYTE *)(v42 + 88) & 0xFC;
              if ( v138 )
              {
                if ( !v148 )
                {
                  v28 = 0;
                  sub_8756F0(1, v44, &v167.m128i_u64[1], 0);
                  v89 = 1;
                  goto LABEL_238;
                }
LABEL_237:
                sub_8756F0(3, v44, &v167.m128i_u64[1], 0);
                v28 = v148;
                v89 = 0;
LABEL_238:
                v156 = v89;
                sub_86F690(v44);
                sub_667260(v42, (__int64)&v167, v28, 0);
                v45 = v156;
                v148 = 0;
LABEL_93:
                if ( v28 )
                {
                  v46 = 0;
                  v47 = v28 | v140;
                  goto LABEL_95;
                }
                goto LABEL_191;
              }
              v45 = 1;
              if ( !v148 )
              {
LABEL_191:
                if ( v140 )
                {
                  v46 = v140;
                  v28 = 0;
                  v47 = v140;
                }
                else
                {
                  if ( word_4F06418[0] == 75 )
                  {
                    v46 = dword_4D04964 == 0;
                    v47 = 0;
                  }
                  else
                  {
                    v47 = 0;
                    v46 = 0;
                  }
                  v28 = 0;
                }
                goto LABEL_95;
              }
LABEL_241:
              sub_86F690(v44);
              if ( !dword_4F04C3C )
                sub_8699D0(v42, 6, 0);
              v28 = v148;
              v46 = 0;
              v45 = 0;
              v148 = 0;
              v47 = v28 | v140;
LABEL_95:
              v48 = *(__m128i **)(a1 + 216);
              v49 = v42;
              sub_66A990(v48, v42, a1, v28, v46, v45);
              if ( *(_QWORD *)(v42 + 104) )
              {
                v48 = (__m128i *)a1;
                v49 = 6;
                sub_656C00(a1, 6, v42, v148, v28);
              }
              v50 = qword_4F04C68;
              v51 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) & 0x80;
              *(_BYTE *)(v42 + 143) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) & 0x80
                                    | *(_BYTE *)(v42 + 143) & 0x7F;
              if ( word_4F06418[0] == 17 )
                sub_7B8B50(v48, v49, v51, qword_4F04C68);
              if ( !v47 )
              {
                v55 = v146;
                if ( !v146 )
                  goto LABEL_117;
LABEL_111:
                *(_BYTE *)(v42 + 161) |= 0x10u;
LABEL_112:
                if ( v28 )
                {
                  v49 = a1;
                  v48 = (__m128i *)v42;
                  sub_66DF40(v42, a1, a2, a6, v135, a7, (__int64)v171);
                  if ( !a9 )
                  {
                    v71 = *(__m128i **)(v42 + 72);
                    if ( !v71 )
                    {
LABEL_122:
                      if ( (*(_BYTE *)(v44 + 81) & 0x40) == 0 )
                      {
                        v49 = v159;
                        v48 = (__m128i *)v42;
                        if ( (unsigned int)sub_736420(v42, v159) )
                        {
                          if ( v148 )
                          {
                            if ( v28 )
                            {
                              v49 = v159;
                              v48 = (__m128i *)v42;
                              sub_736870(v42, v159);
                            }
                          }
                          else
                          {
                            v49 = v159;
                            v48 = (__m128i *)v42;
                            sub_7365B0(v42, v159);
                          }
                        }
                      }
                      if ( v132 )
                      {
                        sub_8642D0();
                      }
                      else if ( v131 )
                      {
                        sub_866010(v48, v49, v51, v50, v55);
                      }
                      *a5 = v42;
                      if ( a8 )
                        *a8 = v28;
                      goto LABEL_132;
                    }
                    goto LABEL_196;
                  }
                  goto LABEL_118;
                }
                v55 = v140;
                if ( v140 )
                {
                  v56 = v136;
                  if ( v136 == 13 )
                  {
                    v105 = *(_BYTE *)(v42 + 161);
                    if ( (v105 & 0x10) != 0
                      || !unk_4D042D8 && ((v105 & 0x20) == 0 || qword_4F077A8 <= 0x9C3Fu) && !unk_4F072F0 )
                    {
                      goto LABEL_116;
                    }
                    v56 = byte_4F068B0[0];
                  }
                  *(_BYTE *)(v42 + 160) = v56;
LABEL_116:
                  *(_BYTE *)(v42 + 141) &= ~0x20u;
                  v48 = (__m128i *)v42;
                  sub_8D6090(v42);
                }
LABEL_117:
                if ( !a9 )
                {
LABEL_119:
                  if ( !v28 && (v148 & 1) != 0 )
                  {
                    if ( !*a7 )
                      goto LABEL_122;
LABEL_200:
                    v48 = (__m128i *)v42;
                    v72 = sub_86A2A0(v42);
                    if ( v72 && *(_BYTE *)(v72 + 16) == 53 )
                    {
                      v73 = *(_QWORD *)(v72 + 24);
                      v48 = (__m128i *)(*(_BYTE *)(v73 - 8) & 1);
                      v74 = (__m128i *)sub_7274B0(v48);
                      v51 = v138;
                      v74[1] = _mm_loadu_si128((const __m128i *)&v171[32]);
                      if ( v138 )
                        *v74 = _mm_loadu_si128((const __m128i *)&v171[16]);
                      *(_QWORD *)(v73 + 8) = v74;
                    }
                    goto LABEL_122;
                  }
                  v71 = *(__m128i **)(v42 + 72);
                  if ( !v71 )
                  {
LABEL_198:
                    v50 = (_QWORD *)*a7;
                    if ( !(_DWORD)v50 || v28 )
                      goto LABEL_122;
                    goto LABEL_200;
                  }
LABEL_196:
                  v49 = v138;
                  v71[1] = _mm_loadu_si128((const __m128i *)&v171[32]);
                  if ( v138 )
                    *v71 = _mm_loadu_si128((const __m128i *)&v171[16]);
                  goto LABEL_198;
                }
LABEL_118:
                v51 = a9;
                *(_QWORD *)(a9 + 40) = *(_QWORD *)&v171[40];
                goto LABEL_119;
              }
              v50 = (_QWORD *)v133;
              if ( !v133 )
              {
LABEL_110:
                if ( !v146 )
                  goto LABEL_112;
                goto LABEL_111;
              }
              *(_QWORD *)(*(_QWORD *)(v42 + 176) + 8LL) = v133;
              if ( v28 )
              {
                v52 = *(_QWORD *)(v42 + 176);
              }
              else if ( (*(_BYTE *)(v42 + 141) & 0x20) == 0 || (v52 = *(_QWORD *)(v42 + 176), (*(_BYTE *)v52 & 1) != 0) )
              {
LABEL_104:
                v53 = v133;
                *(_BYTE *)(v42 + 161) |= 4u;
                *(_BYTE *)(v42 + 141) &= ~0x20u;
                if ( *(_BYTE *)(v133 + 140) == 12 )
                {
                  do
                    v53 = *(_QWORD *)(v53 + 160);
                  while ( *(_BYTE *)(v53 + 140) == 12 );
                }
                else
                {
                  v53 = v133;
                }
                *(_QWORD *)(v42 + 128) = *(_QWORD *)(v53 + 128);
                v54 = v133;
                if ( *(char *)(v133 + 142) >= 0 && (v54 = v133, *(_BYTE *)(v133 + 140) == 12) )
                  v49 = (unsigned int)sub_8D4AB0(v133, v49, v51);
                else
                  v49 = *(unsigned int *)(v54 + 136);
                v48 = (__m128i *)v42;
                *(_DWORD *)(v42 + 136) = sub_8D6010(v42, v49);
                goto LABEL_110;
              }
              v51 = v166;
              *(_QWORD *)(v52 + 16) = v166;
              goto LABEL_104;
            }
            v28 = v148;
LABEL_260:
            sub_6851C0(2518, &v167.m128i_u64[1]);
            v148 = v28;
            v97 = _mm_loadu_si128(&xmmword_4F06660[1]);
            v98 = _mm_loadu_si128(&xmmword_4F06660[2]);
            v99 = _mm_loadu_si128(&xmmword_4F06660[3]);
            v167.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
            v168 = v97;
            v167.m128i_i64[1] = *(_QWORD *)dword_4F07508;
            v168.m128i_i8[1] = v97.m128i_i8[1] | 0x20;
            v169 = v98;
            v170 = v99;
            goto LABEL_215;
          }
LABEL_75:
          v41 = v40 | 0x40;
          v42 = *(_QWORD *)(v25 + 88);
          *(_BYTE *)(a1 + 127) = v41;
          if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 6 && (v168.m128i_i8[0] & 1) != 0 )
          {
            if ( dword_4F077BC )
            {
              if ( !(v28 | v140) )
                goto LABEL_76;
            }
            else if ( v41 >= 0 && !v140 )
            {
              goto LABEL_76;
            }
            if ( (*(_BYTE *)(a1 + 8) & 8) == 0 )
            {
              sub_6851C0(283, &v167.m128i_u64[1]);
              v58 = _mm_loadu_si128(&xmmword_4F06660[1]);
              v59 = _mm_loadu_si128(&xmmword_4F06660[3]);
              v167 = _mm_loadu_si128(xmmword_4F06660);
              v60 = _mm_loadu_si128(&xmmword_4F06660[2]);
              v168 = v58;
              v169 = v60;
              v170 = v59;
LABEL_212:
              v148 = v28;
              v168.m128i_i8[1] |= 0x20u;
              v167.m128i_i64[1] = *(_QWORD *)dword_4F07508;
              goto LABEL_213;
            }
          }
LABEL_76:
          if ( *(_BYTE *)(v42 + 140) != 2 || (v61 = *(_BYTE *)(v42 + 161), (v61 & 8) == 0) )
          {
            v43 = *(_BYTE *)(v25 + 81);
            if ( ((v43 & 0x40) != 0
               || (*(_BYTE *)(v25 + 84) & 2) != 0
               && *(_BYTE *)(v25 + 80) == 3
               && *(_BYTE *)(*(_QWORD *)(v25 + 88) + 140LL) == 14)
              && !(v28 | v140) )
            {
              v149 = v25;
              sub_8767A0(4, v25, &v167.m128i_u64[1], 1);
              *a7 = 0;
              v44 = v149;
              *a5 = v42;
              goto LABEL_132;
            }
            if ( v43 >= 0 )
            {
              sub_6854C0(2345, &v167.m128i_u64[1], v25);
              v75 = _mm_loadu_si128(&xmmword_4F06660[1]);
              v76 = _mm_loadu_si128(&xmmword_4F06660[2]);
              v77 = _mm_loadu_si128(&xmmword_4F06660[3]);
              v167 = _mm_loadu_si128(xmmword_4F06660);
              v168 = v75;
              v169 = v76;
              v170 = v77;
              goto LABEL_212;
            }
            v44 = *(_QWORD *)(v25 + 72);
            if ( !v44 )
              v44 = v25;
            v42 = *(_QWORD *)(v44 + 88);
LABEL_83:
            if ( !v28 )
              goto LABEL_84;
LABEL_253:
            sub_8756F0(3, v44, &v167.m128i_u64[1], 0);
            sub_86F690(v44);
            if ( dword_4F077C4 == 2 && v128 && (*(_BYTE *)(v42 + 88) & 3) != v137 )
            {
              v96 = 5;
              if ( dword_4D04964 )
                v96 = unk_4F07471;
              sub_6853B0(v96, 936, &v167.m128i_u64[1], v44);
              *(_BYTE *)(v42 + 88) = v137 & 3 | *(_BYTE *)(v42 + 88) & 0xFC;
            }
LABEL_88:
            if ( (*(_BYTE *)(v44 + 81) & 0x10) != 0 && *(char *)(*(_QWORD *)(v44 + 64) + 177LL) < 0
              || !v155
              || (v168.m128i_i8[1] & 0x20) != 0 )
            {
              v148 = 1;
              v45 = v28 == 0;
              goto LABEL_93;
            }
            goto LABEL_260;
          }
          v62 = dword_4D0449C;
          if ( dword_4D0449C )
          {
            if ( !(v28 | v140) )
            {
              if ( v146 )
              {
                v106 = 8;
                if ( (v61 & 0x10) != 0 )
                {
                  v106 = 7;
                  if ( dword_4F077BC )
                    v106 = (_DWORD)qword_4F077B4 == 0 ? 5 : 7;
                }
                v147 = v25;
                sub_684AA0(v106, 2343, &v167.m128i_u64[1]);
                v25 = v147;
              }
              v44 = v25;
              v146 = 0;
              if ( (*(_BYTE *)(v42 + 141) & 0x20) != 0 )
                goto LABEL_85;
              goto LABEL_298;
            }
            if ( ((v61 & 0x10) != 0) != v146 )
            {
              v123 = v25;
              sub_6854C0(1956, &v167.m128i_u64[1], v25);
              v25 = v123;
              if ( v28 )
              {
                v44 = v123;
                v95 = v146 & 1;
                v62 = (unsigned int)(16 * v95);
                *(_BYTE *)(v42 + 161) = (16 * v95) | *(_BYTE *)(v42 + 161) & 0xEF;
                if ( (*(_BYTE *)(v42 + 141) & 0x20) != 0 )
                  goto LABEL_253;
                goto LABEL_301;
              }
              v44 = v123;
              v146 = (*(_BYTE *)(v42 + 161) & 0x10) != 0;
              v107 = v140;
              if ( (*(_BYTE *)(v42 + 141) & 0x20) != 0 )
              {
LABEL_84:
                if ( v140 )
                {
LABEL_166:
                  sub_8756F0(1, v44, &v167.m128i_u64[1], 0);
                  sub_86F690(v44);
                  goto LABEL_88;
                }
LABEL_85:
                if ( dword_4D04964 || word_4F06418[0] != 75 )
                {
                  sub_8767A0(4, v44, &v167.m128i_u64[1], 1);
                  *a7 = 0;
                  goto LABEL_88;
                }
                goto LABEL_166;
              }
              goto LABEL_299;
            }
            if ( (*(_BYTE *)(v42 + 141) & 0x20) == 0 )
            {
              LOBYTE(v95) = v146 & 1;
LABEL_301:
              v108 = *(_QWORD *)(*(_QWORD *)(v42 + 176) + 8LL);
              if ( v133 || !(_BYTE)v95 )
              {
                v110 = v133;
              }
              else
              {
                v124 = v25;
                v151 = v95;
                v109 = sub_72BA30(5);
                LOBYTE(v95) = v151;
                v25 = v124;
                v110 = v109;
              }
              if ( !v108 && (_BYTE)v95 )
              {
                v125 = v25;
                v111 = sub_72BA30(5);
                v25 = v125;
                v108 = v111;
              }
              if ( v110 | v108 )
              {
                if ( !v108
                  || !v110
                  || v108 != v110 && (v152 = v25, v112 = sub_8D97D0(v108, v110, 32, v62, v25), v25 = v152, !v112) )
                {
                  v126 = v25;
                  sub_6854C0(2342, &v167.m128i_u64[1], v25);
                  v136 = 13;
                  v133 = 0;
                  v25 = v126;
                }
              }
              v44 = v25;
              if ( unk_4F06C60 == v42 )
                *(_BYTE *)(v25 + 83) &= ~0x40u;
              goto LABEL_83;
            }
          }
          else if ( (*(_BYTE *)(v42 + 141) & 0x20) == 0 )
          {
LABEL_298:
            v107 = v28 | v140;
LABEL_299:
            v44 = v25;
            if ( !v107 )
              goto LABEL_85;
            LOBYTE(v95) = v146 & 1;
            goto LABEL_301;
          }
          v44 = v25;
          goto LABEL_83;
        }
        v131 = 0;
        v132 = 0;
        v133 = 0;
        if ( !v146 )
        {
          v136 = 13;
LABEL_362:
          v140 = 0;
          v27 = word_4F06418[0];
          goto LABEL_73;
        }
LABEL_24:
        v27 = word_4F06418[0];
        if ( word_4F06418[0] != 17 )
        {
          v136 = 13;
          if ( word_4F06418[0] != 75 )
          {
LABEL_26:
            if ( v148 )
            {
              if ( !v25 )
              {
LABEL_28:
                v140 = 0;
                v28 = v148;
                v29 = 1;
                goto LABEL_74;
              }
LABEL_67:
              if ( (*(_BYTE *)(v25 + 81) & 2) != 0 )
              {
                if ( (a2 & 0x20000) != 0 && *(char *)(*(_QWORD *)(v25 + 88) + 162LL) >= 0 )
                {
                  if ( v146 )
                    sub_686A30(8, 1449, &v167.m128i_u64[1], *(_QWORD *)(v25 + 96) + 40LL);
                  else
                    sub_684AA0(8, 2517, &v167.m128i_u64[1]);
                }
                else
                {
                  sub_685920(&v167.m128i_u64[1], v25, 8);
                }
                v140 = 0;
                v91 = _mm_loadu_si128(&xmmword_4F06660[1]);
                v92 = _mm_loadu_si128(&xmmword_4F06660[2]);
                v93 = *(_QWORD *)dword_4F07508;
                v167 = _mm_loadu_si128(xmmword_4F06660);
                v94 = _mm_loadu_si128(&xmmword_4F06660[3]);
                v168 = v91;
                *(_BYTE *)(a1 + 127) |= 0x80u;
                v168.m128i_i8[1] |= 0x20u;
                v167.m128i_i64[1] = v93;
                v169 = v92;
                v170 = v94;
                goto LABEL_213;
              }
              v28 = v148;
              v140 = 0;
              v40 = *(_BYTE *)(a1 + 127) | 0x80;
              *(_BYTE *)(a1 + 127) = v40;
              goto LABEL_75;
            }
            goto LABEL_362;
          }
        }
        v136 = 13;
LABEL_184:
        if ( v27 == 17 || v27 == 75 || (_DWORD)qword_4F077B4 )
        {
          v140 = 1;
          if ( (*(_BYTE *)(a1 + 122) & 0x20) != 0 )
          {
            v150 = v25;
            sub_684AA0(unk_4F07471, 2626, &v165);
            v25 = v150;
            v27 = word_4F06418[0];
          }
        }
        else
        {
          v153 = v25;
          sub_6851C0(2625, &v165);
          v140 = 1;
          v25 = v153;
          v27 = word_4F06418[0];
        }
        goto LABEL_73;
      }
      v131 = 0;
      goto LABEL_291;
    }
    if ( !v25 || (v22 = v24) == 0 )
    {
      if ( (v168.m128i_i8[1] & 0x20) != 0 && !v24 )
      {
        v154 = v25;
        *a5 = sub_72C930(v23);
        v44 = v154;
        goto LABEL_132;
      }
      if ( !v25 )
      {
        v22 = v135;
        if ( v135 )
        {
          v65 = v137;
          if ( v159 != dword_4F04C5C )
            v65 = 0;
          v66 = 0;
          if ( v159 == dword_4F04C5C )
            v66 = v135;
          v137 = v65;
          v135 = v66;
        }
      }
      goto LABEL_21;
    }
    v31 = *(_QWORD *)(v25 + 64);
    if ( (*(_BYTE *)(v25 + 81) & 0x10) != 0 )
    {
      v22 = (__int64)&dword_4D0449C;
      if ( !dword_4D0449C || v135 )
      {
        if ( v135 == v31
          || v31 && v135 && dword_4F07588 && (v67 = *(_QWORD *)(v31 + 32), *(_QWORD *)(v135 + 32) == v67) && v67 )
        {
          v132 = 0;
        }
        else
        {
          v22 = (__int64)&v165;
          sub_6854C0(551, &v165, v25);
          v132 = 0;
          v68 = _mm_loadu_si128(&xmmword_4F06660[1]);
          v69 = _mm_loadu_si128(&xmmword_4F06660[2]);
          v25 = 0;
          v70 = _mm_loadu_si128(&xmmword_4F06660[3]);
          v167.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
          v168 = v68;
          v167.m128i_i64[1] = *(_QWORD *)dword_4F07508;
          v168.m128i_i8[1] = v68.m128i_i8[1] | 0x20;
          v169 = v69;
          v170 = v70;
        }
        goto LABEL_63;
      }
      if ( (v168.m128i_i8[2] & 2) != 0 )
      {
        if ( v169.m128i_i64[0] == v31 )
          goto LABEL_48;
        if ( v169.m128i_i64[0] )
        {
          if ( v31 )
          {
            if ( dword_4F07588 )
            {
              v32 = *(_QWORD *)(v169.m128i_i64[0] + 32);
              if ( *(_QWORD *)(v31 + 32) == v32 )
              {
                if ( v32 )
                  goto LABEL_48;
              }
            }
          }
        }
      }
      else if ( !v31 )
      {
        goto LABEL_48;
      }
      v141 = v25;
      sub_685360(2472, &v167.m128i_u64[1]);
      v25 = v141;
LABEL_48:
      v22 = 0;
      v142 = v25;
      sub_8646E0(v31, 0);
      v132 = 0;
      v135 = v31;
      v25 = v142;
      v131 = 1;
      v159 = dword_4F04C64;
      goto LABEL_63;
    }
    if ( v31 )
    {
      v145 = v25;
      v22 = (__int64)&v165;
      v158 = 0;
      v101 = sub_668160(v25, (__int64)&v165, &v158, 0);
      v25 = v145;
      if ( v101 )
      {
        v22 = 0;
        sub_864230(*(_QWORD *)(v145 + 64), 0);
        v132 = 1;
        v25 = v145;
        v159 = dword_4F04C64;
LABEL_63:
        v36 = a1;
        v143 = v25;
        sub_643D30(a1);
        v25 = v143;
        if ( dword_4D044A8 && word_4F06418[0] == 55 )
        {
          v163 = 0;
          if ( unk_4D044A4 )
          {
            v22 = (__int64)&dword_4F063F8;
            v36 = 2372;
            sub_684B30(2372, &dword_4F063F8);
            v25 = v143;
          }
          v144 = v25;
          sub_7B8B50(v36, v22, v37, v38);
          v166 = *(_QWORD *)&dword_4F063F8;
          ++*(_BYTE *)(qword_4F061C8 + 81LL);
          sub_65CD60(&v163);
          v25 = v144;
          --*(_BYTE *)(qword_4F061C8 + 81LL);
          v133 = v163;
          if ( v163 )
          {
            v63 = sub_8DBE70(v163);
            v25 = v144;
            if ( v63 )
            {
              v64 = v163;
              v136 = byte_4CFDE80;
              if ( !v163 )
              {
                v133 = 0;
                goto LABEL_162;
              }
              goto LABEL_347;
            }
            v64 = v163;
            v120 = *(_BYTE *)(v163 + 140);
            if ( v120 == 12 )
            {
              v121 = v163;
              do
              {
                v121 = *(_QWORD *)(v121 + 160);
                v120 = *(_BYTE *)(v121 + 140);
              }
              while ( v120 == 12 );
            }
            if ( !v120 )
            {
              v136 = byte_4CFDE80;
LABEL_347:
              if ( v136 != 13 )
              {
                if ( dword_4D0449C )
                {
LABEL_164:
                  v27 = word_4F06418[0];
                  if ( word_4F06418[0] == 73 )
                    goto LABEL_26;
                  goto LABEL_184;
                }
                goto LABEL_66;
              }
              goto LABEL_351;
            }
            v122 = sub_8D2780(v163);
            v25 = v144;
            if ( v122 )
            {
              v64 = v163;
              if ( v163 )
              {
LABEL_351:
                while ( *(_BYTE *)(v64 + 140) == 12 )
                  v64 = *(_QWORD *)(v64 + 160);
                v136 = *(_BYTE *)(v64 + 160);
LABEL_162:
                if ( dword_4D0449C )
                {
                  if ( v136 != 13 )
                    goto LABEL_164;
LABEL_153:
                  if ( !v146 )
                  {
                    v136 = 13;
                    goto LABEL_26;
                  }
                  goto LABEL_24;
                }
                goto LABEL_66;
              }
            }
            else
            {
              sub_6851C0(1541, &v166);
              v25 = v144;
            }
          }
        }
        if ( dword_4D0449C )
        {
          v133 = 0;
          goto LABEL_153;
        }
        v133 = 0;
        v136 = 13;
LABEL_66:
        if ( !v25 )
          goto LABEL_28;
        goto LABEL_67;
      }
      v132 = v158;
      if ( !v158 )
        goto LABEL_63;
      v102 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v103 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v104 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v167.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v168 = v102;
      v167.m128i_i64[1] = *(_QWORD *)dword_4F07508;
      v168.m128i_i8[1] = v102.m128i_i8[1] | 0x20;
      v169 = v103;
      v170 = v104;
    }
    else
    {
      if ( *(_DWORD *)(v25 + 40) != unk_4F066A8 || !*(_BYTE *)(qword_4F04C68[0] + 776LL * (int)v159 + 4) )
      {
LABEL_291:
        v132 = 0;
        goto LABEL_63;
      }
      v22 = (__int64)&v165;
      sub_6854C0(551, &v165, v25);
      v116 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v117 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v167 = _mm_loadu_si128(xmmword_4F06660);
      v118 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v168 = v116;
      v167.m128i_i64[1] = *(_QWORD *)dword_4F07508;
      v168.m128i_i8[1] = v116.m128i_i8[1] | 0x20;
      v169 = v118;
      v170 = v117;
    }
LABEL_62:
    v132 = 0;
    v25 = 0;
    goto LABEL_63;
  }
  v33 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v34 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v35 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v167.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v167.m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
  v22 = 6;
  v168 = v33;
  v168.m128i_i8[1] = v33.m128i_i8[1] | 0x20;
  v169 = v34;
  v165 = *(_QWORD *)&dword_4F063F8;
  v170 = v35;
  v148 = sub_667FD0(word_4F06418[0], 6, (a2 & 0x40) != 0, (a2 >> 24) & 1);
  if ( v148 )
  {
    if ( v146 )
    {
      v22 = (__int64)&dword_4F063F8;
      sub_6851C0(2176, &dword_4F063F8);
    }
    v131 = 0;
    goto LABEL_62;
  }
  ++*(_BYTE *)(qword_4F061C8 + 81LL);
  if ( (v146 & 1) != 0 || (v100 = 110, (a2 & 0x1000000) != 0) )
    v100 = 40;
  sub_6851D0(v100);
  v44 = 0;
  --*(_BYTE *)(qword_4F061C8 + 81LL);
  *a7 = 1;
  *a5 = sub_72C930(v100);
LABEL_132:
  *(_QWORD *)a1 = v44;
  sub_724E30(&v161);
  return sub_724E30(&v162);
}
