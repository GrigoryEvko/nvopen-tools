// Function: sub_8A09D0
// Address: 0x8a09d0
//
__m128i *__fastcall sub_8A09D0(__int64 a1, __int64 a2, const __m128i *a3, __m128i *a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  _QWORD *v10; // rsi
  int v11; // eax
  __int64 v12; // r11
  __int64 v13; // r10
  __m128i *v14; // r12
  _BOOL4 v15; // ecx
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __m128i v18; // xmm3
  __m128i v19; // xmm4
  __m128i v20; // xmm5
  __m128i v21; // xmm6
  __m128i v22; // xmm7
  __m128i v23; // xmm0
  __m128i v24; // xmm1
  __m128i v25; // xmm2
  __m128i v26; // xmm3
  __m128i v27; // xmm4
  __m128i v28; // xmm3
  __m128i v29; // xmm4
  __m128i v30; // xmm5
  __m128i v31; // xmm6
  __m128i v32; // xmm7
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  __m128i v35; // xmm4
  __m128i v36; // xmm5
  __m128i v37; // xmm6
  __m128i v38; // xmm7
  __m128i v39; // xmm2
  unsigned int v40; // edx
  unsigned __int64 v41; // rsi
  unsigned __int64 v42; // rdi
  __m128i *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r8
  __int64 *v52; // r9
  unsigned __int16 v53; // dx
  __int64 v55; // rax
  __m128i v56; // xmm6
  __m128i v57; // xmm7
  __m128i v58; // xmm0
  __m128i v59; // xmm1
  __m128i v60; // xmm2
  __m128i v61; // xmm3
  __m128i v62; // xmm4
  __m128i v63; // xmm5
  __m128i v64; // xmm6
  __m128i v65; // xmm7
  __m128i v66; // xmm0
  __m128i v67; // xmm1
  __m128i v68; // xmm4
  __m128i v69; // xmm5
  __m128i v70; // xmm6
  __m128i v71; // xmm7
  __m128i v72; // xmm0
  __m128i v73; // xmm1
  __m128i v74; // xmm2
  __m128i v75; // xmm3
  __m128i v76; // xmm4
  __m128i v77; // xmm5
  __m128i v78; // xmm6
  __m128i v79; // xmm7
  __int64 v80; // rbx
  __int64 v81; // rax
  __int64 i; // r14
  __int64 v83; // r15
  __int64 v84; // rax
  _QWORD *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rax
  char v93; // dl
  int v94; // ecx
  __int64 v95; // r10
  _BOOL4 v96; // ecx
  __int64 *v97; // rdx
  __m128i *v98; // rax
  __m128i *v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rax
  __int64 *v105; // rcx
  __int64 v106; // rbx
  __int64 *v107; // r12
  __int64 v108; // r13
  __int64 v109; // rsi
  _BYTE *v110; // r15
  __int64 v111; // rax
  unsigned int *v112; // [rsp-8h] [rbp-448h]
  __int64 v113; // [rsp+0h] [rbp-440h]
  __int64 v114; // [rsp+8h] [rbp-438h]
  const char *v115; // [rsp+10h] [rbp-430h]
  __int64 v116; // [rsp+18h] [rbp-428h]
  unsigned int v117; // [rsp+24h] [rbp-41Ch]
  unsigned __int16 v118; // [rsp+42h] [rbp-3FEh]
  int v119; // [rsp+44h] [rbp-3FCh]
  __int16 v120; // [rsp+48h] [rbp-3F8h]
  __int64 v121; // [rsp+48h] [rbp-3F8h]
  unsigned __int16 v122; // [rsp+50h] [rbp-3F0h]
  __int64 v123; // [rsp+50h] [rbp-3F0h]
  __int64 v124; // [rsp+50h] [rbp-3F0h]
  __m128i *v125; // [rsp+50h] [rbp-3F0h]
  int v126; // [rsp+58h] [rbp-3E8h]
  unsigned int v127; // [rsp+58h] [rbp-3E8h]
  __int64 *v128; // [rsp+58h] [rbp-3E8h]
  __int64 v129; // [rsp+58h] [rbp-3E8h]
  __m128i *v130; // [rsp+58h] [rbp-3E8h]
  int v131; // [rsp+60h] [rbp-3E0h]
  _BOOL4 v132; // [rsp+60h] [rbp-3E0h]
  int v133; // [rsp+64h] [rbp-3DCh]
  __int64 v134; // [rsp+68h] [rbp-3D8h]
  __int64 v136; // [rsp+70h] [rbp-3D0h]
  __int64 v137; // [rsp+70h] [rbp-3D0h]
  __int64 v138; // [rsp+70h] [rbp-3D0h]
  __int64 v139; // [rsp+78h] [rbp-3C8h]
  __int64 v140; // [rsp+78h] [rbp-3C8h]
  __int64 v141; // [rsp+78h] [rbp-3C8h]
  int v142; // [rsp+84h] [rbp-3BCh] BYREF
  unsigned int *v143; // [rsp+88h] [rbp-3B8h] BYREF
  __m128i v144; // [rsp+90h] [rbp-3B0h] BYREF
  __m128i v145; // [rsp+A0h] [rbp-3A0h] BYREF
  __m128i v146; // [rsp+B0h] [rbp-390h] BYREF
  __m128i v147; // [rsp+C0h] [rbp-380h] BYREF
  __m128i v148; // [rsp+D0h] [rbp-370h] BYREF
  __m128i v149; // [rsp+E0h] [rbp-360h] BYREF
  __m128i v150; // [rsp+F0h] [rbp-350h] BYREF
  __m128i v151; // [rsp+100h] [rbp-340h] BYREF
  __m128i v152; // [rsp+110h] [rbp-330h] BYREF
  __m128i v153; // [rsp+120h] [rbp-320h] BYREF
  __m128i v154; // [rsp+130h] [rbp-310h] BYREF
  __m128i v155; // [rsp+140h] [rbp-300h] BYREF
  __m128i v156; // [rsp+150h] [rbp-2F0h] BYREF
  __m128i v157; // [rsp+160h] [rbp-2E0h] BYREF
  __m128i v158; // [rsp+170h] [rbp-2D0h] BYREF
  __m128i v159; // [rsp+180h] [rbp-2C0h] BYREF
  __m128i v160; // [rsp+190h] [rbp-2B0h] BYREF
  __m128i v161; // [rsp+1A0h] [rbp-2A0h] BYREF
  __m128i v162; // [rsp+1B0h] [rbp-290h] BYREF
  __m128i v163; // [rsp+1C0h] [rbp-280h] BYREF
  __m128i v164; // [rsp+1D0h] [rbp-270h] BYREF
  __m128i v165; // [rsp+1E0h] [rbp-260h] BYREF
  __m128i v166; // [rsp+1F0h] [rbp-250h] BYREF
  __m128i v167; // [rsp+200h] [rbp-240h] BYREF
  __m128i v168; // [rsp+210h] [rbp-230h] BYREF
  __m128i v169; // [rsp+220h] [rbp-220h] BYREF
  __m128i *v170[66]; // [rsp+230h] [rbp-210h] BYREF

  v7 = *(_QWORD *)(a1 + 88);
  v142 = 0;
  v8 = *(_QWORD *)(v7 + 88);
  if ( v8 )
  {
    if ( (*(_BYTE *)(v7 + 160) & 1) != 0 )
      v8 = a1;
  }
  else
  {
    v8 = a1;
  }
  switch ( *(_BYTE *)(v8 + 80) )
  {
    case 4:
    case 5:
      v9 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 80LL);
      break;
    case 6:
      v9 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v9 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v9 = *(_QWORD *)(v8 + 88);
      break;
    default:
      v9 = 0;
      break;
  }
  v139 = sub_892400(v9);
  v10 = **(_QWORD ***)(v139 + 32);
  v126 = sub_89A200((__int64 *)a2, v10, &v142);
  v134 = *(_QWORD *)(a1 + 88);
  v11 = sub_8D0B70(a1);
  v12 = v139;
  v133 = v11;
  if ( !a4 )
  {
    v123 = v139;
    v14 = (__m128i *)sub_87F3D0(a1);
    v85 = sub_7259C0(12);
    *((_BYTE *)v85 + 184) = 10;
    v14[5].m128i_i64[1] = (__int64)v85;
    v141 = (__int64)v85;
    sub_877D80((__int64)v85, v14->m128i_i64);
    sub_877F10(v141, (__int64)v14, v86, v87, v88, v89);
    v140 = v14[5].m128i_i64[1];
    sub_890140(a1, (_QWORD *)v134, (__int64)v14, (__int64)a3, v90, v91);
    v92 = v14[5].m128i_i64[1];
    v93 = *(_BYTE *)(v92 + 186);
    if ( (v93 & 0x30) == 0 )
    {
      v94 = v142;
      if ( v126 )
        *(_BYTE *)(v92 + 186) = v93 | 0x10;
      if ( v94 )
        *(_BYTE *)(v92 + 186) |= 0x20u;
    }
    if ( (v14[5].m128i_i8[1] & 0x10) != 0 )
    {
      v95 = v14[4].m128i_i64[0];
      v96 = (*(_BYTE *)(v95 + 177) & 0x20) != 0;
    }
    else
    {
      v95 = 0;
      v96 = 0;
    }
    v10 = (_QWORD *)v140;
    v121 = v123;
    v132 = v96;
    v97 = *(__int64 **)(v140 + 168);
    v124 = v95;
    *v97 = a2;
    v128 = v97;
    v98 = sub_72F240(a3);
    v13 = v124;
    v15 = v132;
    v12 = v121;
    v128[1] = (__int64)v98;
    v128[2] = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 104LL);
    if ( (v14[5].m128i_i8[1] & 0x10) != 0 )
    {
      v10 = (_QWORD *)v140;
      *(_BYTE *)(v140 + 88) = (*(_BYTE *)(v134 + 265) >> 6) | *(_BYTE *)(v140 + 88) & 0xFC;
    }
LABEL_9:
    if ( unk_4D049D0 != a1 )
      goto LABEL_10;
LABEL_44:
    v157.m128i_i64[0] = a2;
    if ( a2 && *(_BYTE *)(a2 + 8) == 3 )
    {
      sub_72F220((__int64 **)&v157);
      a2 = v157.m128i_i64[0];
    }
    v80 = *(_QWORD *)(a2 + 32);
    v81 = *(_QWORD *)a2;
    v157.m128i_i64[0] = v81;
    if ( v81 && *(_BYTE *)(v81 + 8) == 3 )
    {
      sub_72F220((__int64 **)&v157);
      v81 = v157.m128i_i64[0];
    }
    for ( i = *(_QWORD *)(v81 + 32); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v83 = *(_QWORD *)v81;
    v157.m128i_i64[0] = v83;
    if ( v83 && *(_BYTE *)(v83 + 8) == 3 )
    {
      sub_72F220((__int64 **)&v157);
      v83 = v157.m128i_i64[0];
    }
    if ( *(_BYTE *)(v80 + 120) == 1 )
    {
      v136 = *(_QWORD *)(v83 + 32);
      if ( ((unsigned int)sub_8D3D40(i) || (unsigned int)sub_8D2780(i))
        && ((unsigned int)sub_8D3D40(*(_QWORD *)(v136 + 128)) || (unsigned int)sub_8D2780(*(_QWORD *)(v136 + 128))) )
      {
        v99 = (__m128i *)sub_725090(0);
        v157.m128i_i64[0] = (__int64)v99;
        v99[2].m128i_i64[0] = i;
        v170[0] = v99;
        if ( (unsigned int)sub_88D7A0(v83, (__int64)v10, v100, v101, v102, v103) )
        {
          v157.m128i_i64[0] = (__int64)sub_725090(1u);
          *(_QWORD *)(v157.m128i_i64[0] + 32) = sub_740900((const __m128i *)v136, 1);
          *(_QWORD *)v170[0] = v157.m128i_i64[0];
          goto LABEL_88;
        }
        v104 = sub_620FA0(v136, &v143);
        if ( v104 >= 0 && !(_DWORD)v143 )
        {
          v129 = v104;
          if ( ***(_QWORD ***)(sub_8794A0((_QWORD *)v80) + 32) )
          {
            v157.m128i_i64[0] = (__int64)sub_725090(3u);
            v137 = v157.m128i_i64[0];
            *(_QWORD *)v170[0] = v157.m128i_i64[0];
            sub_7296C0(&v144);
            v105 = (__int64 *)v137;
            if ( v129 )
            {
              v138 = v80;
              v106 = v129;
              v130 = v14;
              v107 = v105;
              v125 = a4;
              v108 = 0;
              do
              {
                v157.m128i_i64[0] = (__int64)sub_725090(1u);
                v109 = v108++;
                v110 = sub_724D50(1);
                sub_72BAF0((__int64)v110, v109, *(_BYTE *)(i + 160));
                v111 = v157.m128i_i64[0];
                *(_BYTE *)(v157.m128i_i64[0] + 24) |= 8u;
                *(_QWORD *)(v111 + 32) = v110;
                *v107 = v111;
                v107 = (__int64 *)v157.m128i_i64[0];
              }
              while ( v106 > v108 );
              v80 = v138;
              v14 = v130;
              a4 = v125;
            }
            sub_729730(v144.m128i_i32[0]);
            goto LABEL_88;
          }
          if ( !v129 )
          {
LABEL_88:
            v84 = sub_8A0370(*(_QWORD *)v80, v170, 0, 0, 0, 0, 0)[11];
LABEL_57:
            *(_QWORD *)(v140 + 160) = v84;
            if ( a4 )
              goto LABEL_35;
            goto LABEL_58;
          }
          sub_6854C0(0xAD5u, (FILE *)(v136 + 64), *(_QWORD *)v80);
        }
      }
    }
    v84 = sub_72C930();
    goto LABEL_57;
  }
  v140 = a4[5].m128i_i64[1];
  if ( (a4[5].m128i_i8[1] & 0x10) != 0 )
  {
    v13 = a4[4].m128i_i64[0];
    v14 = a4;
    v15 = 0;
    goto LABEL_9;
  }
  v14 = a4;
  v15 = 0;
  v13 = 0;
  if ( unk_4D049D0 == a1 )
    goto LABEL_44;
LABEL_10:
  if ( unk_4D049C0 == a1 )
  {
    v84 = sub_88E770((__int64 *)a2, (__int64)v10);
    goto LABEL_57;
  }
  v127 = dword_4F063F8;
  v122 = word_4F063FC[0];
  v131 = dword_4F07508[0];
  v120 = dword_4F07508[1];
  v119 = dword_4F061D8;
  v118 = word_4F061DC[0];
  if ( (unsigned __int16)(word_4F06418[0] - 2) <= 6u )
  {
    v16 = _mm_loadu_si128(&xmmword_4F06300[1]);
    v17 = _mm_loadu_si128(&xmmword_4F06300[2]);
    v18 = _mm_loadu_si128(&xmmword_4F06300[3]);
    v19 = _mm_loadu_si128(&xmmword_4F06300[4]);
    v20 = _mm_loadu_si128(&xmmword_4F06300[5]);
    v144 = _mm_loadu_si128(xmmword_4F06300);
    v21 = _mm_loadu_si128(&xmmword_4F06300[6]);
    v22 = _mm_loadu_si128(&xmmword_4F06300[7]);
    v145 = v16;
    v23 = _mm_loadu_si128(xmmword_4F06380);
    v24 = _mm_loadu_si128(&xmmword_4F06380[1]);
    v146 = v17;
    v147 = v18;
    v25 = _mm_loadu_si128((const __m128i *)&unk_4F063A0);
    v26 = _mm_loadu_si128((const __m128i *)word_4F063B0);
    v148 = v19;
    v27 = _mm_loadu_si128(&xmmword_4F063C0);
    v149 = v20;
    v150 = v21;
    v151 = v22;
    v152 = v23;
    v153 = v24;
    v154 = v25;
    v155 = v26;
    v156 = v27;
    if ( word_4F06418[0] == 8 )
    {
      v28 = _mm_loadu_si128(&xmmword_4F06220[1]);
      v29 = _mm_loadu_si128(&xmmword_4F06220[2]);
      v30 = _mm_loadu_si128(&xmmword_4F06220[3]);
      v31 = _mm_loadu_si128(&xmmword_4F06220[4]);
      v32 = _mm_loadu_si128(&xmmword_4F06220[5]);
      v157 = _mm_loadu_si128(xmmword_4F06220);
      v33 = _mm_loadu_si128(&xmmword_4F06220[6]);
      v158 = v28;
      v34 = _mm_loadu_si128(&xmmword_4F06220[7]);
      v159 = v29;
      v35 = _mm_loadu_si128(xmmword_4F062A0);
      v160 = v30;
      v36 = _mm_loadu_si128(&xmmword_4F062A0[1]);
      v161 = v31;
      v37 = _mm_loadu_si128((const __m128i *)&unk_4F062C0);
      v162 = v32;
      v38 = _mm_loadu_si128((const __m128i *)&unk_4F062D0);
      v163 = v33;
      v39 = _mm_loadu_si128(xmmword_4F062E0);
      v164 = v34;
      v165 = v35;
      v166 = v36;
      v167 = v37;
      v168 = v38;
      v169 = v39;
    }
  }
  if ( !*(_QWORD *)(v12 + 8) || (*(_BYTE *)(v9 + 266) & 0x10) != 0 )
  {
LABEL_40:
    v55 = sub_72C930();
    v53 = word_4F06418[0];
    *(_QWORD *)(v140 + 160) = v55;
    if ( (unsigned __int16)(v53 - 2) > 6u )
      goto LABEL_34;
    goto LABEL_41;
  }
  if ( (unsigned __int64)*(unsigned int *)(v9 + 40) >= unk_4D042F0 )
  {
    sub_6854E0(0x1C8u, (__int64)v14);
    goto LABEL_40;
  }
  if ( (*(_BYTE *)(v140 + 186) & 0x30) != 0 || v15 )
  {
    if ( dword_4F04C44 == -1 )
    {
      v40 = 1050628;
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
      {
        v40 = 2052;
        if ( unk_4F04C48 != -1 )
          v40 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 6) & 0x10) == 0 ? 2052 : 1050628;
      }
    }
    else
    {
      v40 = 1050628;
    }
  }
  else
  {
    v40 = 2048;
  }
  memset(v170, 0, 0x1D8u);
  v170[19] = (__m128i *)v170;
  v170[3] = *(__m128i **)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v170[22]) |= 1u;
  BYTE3(v170[16]) |= 0x20u;
  BYTE2(v170[15]) |= 0x80u;
  ++*(_DWORD *)(v9 + 40);
  v113 = v13;
  v114 = v12;
  sub_864700(*(_QWORD *)(v12 + 32), 0, 0, (__int64)v14, a1, a2, 1, v40);
  v117 = dword_4F04C3C;
  dword_4F04C3C = 1;
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) |= 0x10u;
  sub_854C10(*(const __m128i **)(v134 + 56));
  v115 = qword_4F06410;
  v116 = qword_4F06408;
  sub_7BC160(v114);
  sub_8756F0(32770, (__int64)v14, (__m128i *)v14[3].m128i_i64, 0);
  LOBYTE(v114) = *(_BYTE *)(a1 + 83) >> 7;
  *(_BYTE *)(a1 + 83) |= 0x80u;
  sub_65C7C0((__int64)v170);
  sub_64EC60((__int64)v170);
  v41 = *(_BYTE *)(a1 + 83) & 0x7F;
  *(_BYTE *)(a1 + 83) = ((_BYTE)v114 << 7) | *(_BYTE *)(a1 + 83) & 0x7F;
  v42 = *(_QWORD *)(v134 + 128);
  if ( v42 )
  {
    v43 = (__m128i *)sub_5CF220((const __m128i *)v42, 0, a1, **(_QWORD **)(v134 + 32), a2, v113, 0, 0);
    v41 = (unsigned __int64)v112;
    v170[25] = v43;
    if ( *(_QWORD *)(v140 + 160) )
      goto LABEL_26;
    if ( *(_QWORD *)(v134 + 128) )
    {
      v143 = 0;
      sub_5CF140((__int64 **)&v170[25], &v143);
      v41 = (unsigned __int64)v143;
      v42 = (unsigned __int64)&v170[36];
      sub_5CF030((__int64 *)&v170[36], v143, (__int64)v170);
    }
  }
  else if ( *(_QWORD *)(v140 + 160) )
  {
    goto LABEL_26;
  }
  *(__m128i **)(v140 + 160) = v170[36];
LABEL_26:
  v170[0] = v14;
  if ( (*(_BYTE *)(v140 + 186) & 0x10) != 0 )
  {
    sub_854B40();
  }
  else
  {
    v41 = 0;
    v42 = (unsigned __int64)v14;
    sub_854980((__int64)v14, 0);
  }
  if ( *(_QWORD *)(v134 + 128) )
  {
    v42 = (unsigned __int64)v170;
    v41 = 1;
    sub_644920(v170, 1);
  }
  if ( word_4F06418[0] != 9 )
  {
    v41 = (unsigned __int64)&dword_4F063F8;
    v42 = 65;
    sub_6851C0(0x41u, &dword_4F063F8);
    while ( word_4F06418[0] != 9 )
      sub_7B8B50(0x41u, &dword_4F063F8, v44, v45, v46, v47);
  }
  sub_7B8B50(v42, (unsigned int *)v41, v44, v45, v46, v47);
  --*(_DWORD *)(v9 + 40);
  dword_4F04C3C = v117;
  v48 = 16 * (v117 & 1);
  v49 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v50 = (unsigned int)v48 | *(_BYTE *)(v49 + 7) & 0xEF;
  *(_BYTE *)(v49 + 7) = (16 * (v117 & 1)) | *(_BYTE *)(v49 + 7) & 0xEF;
  sub_863FE0(v42, (__int64)qword_4F04C68, v50, v48, v51, v52);
  v53 = word_4F06418[0];
  qword_4F06408 = v116;
  qword_4F06410 = v115;
  if ( (unsigned __int16)(word_4F06418[0] - 2) <= 6u )
  {
LABEL_41:
    v56 = _mm_loadu_si128(&v145);
    v57 = _mm_loadu_si128(&v146);
    v58 = _mm_loadu_si128(&v147);
    v59 = _mm_loadu_si128(&v148);
    xmmword_4F06300[0] = _mm_loadu_si128(&v144);
    v60 = _mm_loadu_si128(&v149);
    v61 = _mm_loadu_si128(&v150);
    xmmword_4F06300[1] = v56;
    v62 = _mm_loadu_si128(&v151);
    v63 = _mm_loadu_si128(&v152);
    xmmword_4F06300[2] = v57;
    v64 = _mm_loadu_si128(&v153);
    v65 = _mm_loadu_si128(&v154);
    xmmword_4F06300[3] = v58;
    xmmword_4F06300[4] = v59;
    v66 = _mm_loadu_si128(&v155);
    v67 = _mm_loadu_si128(&v156);
    xmmword_4F06300[5] = v60;
    xmmword_4F06300[6] = v61;
    xmmword_4F06300[7] = v62;
    xmmword_4F06380[0] = v63;
    xmmword_4F06380[1] = v64;
    unk_4F063A0 = v65;
    *(__m128i *)word_4F063B0 = v66;
    xmmword_4F063C0 = v67;
    if ( v53 == 8 )
    {
      v68 = _mm_loadu_si128(&v158);
      v69 = _mm_loadu_si128(&v159);
      v70 = _mm_loadu_si128(&v160);
      v71 = _mm_loadu_si128(&v161);
      xmmword_4F06220[0] = _mm_loadu_si128(&v157);
      v72 = _mm_loadu_si128(&v162);
      v73 = _mm_loadu_si128(&v163);
      xmmword_4F06220[1] = v68;
      v74 = _mm_loadu_si128(&v164);
      v75 = _mm_loadu_si128(&v165);
      xmmword_4F06220[2] = v69;
      v76 = _mm_loadu_si128(&v166);
      v77 = _mm_loadu_si128(&v167);
      xmmword_4F06220[3] = v70;
      xmmword_4F06220[4] = v71;
      v78 = _mm_loadu_si128(&v168);
      v79 = _mm_loadu_si128(&v169);
      xmmword_4F06220[5] = v72;
      xmmword_4F06220[6] = v73;
      xmmword_4F06220[7] = v74;
      xmmword_4F062A0[0] = v75;
      xmmword_4F062A0[1] = v76;
      unk_4F062C0 = v77;
      unk_4F062D0 = v78;
      xmmword_4F062E0[0] = v79;
    }
  }
LABEL_34:
  dword_4F07508[0] = v131;
  LOWORD(dword_4F07508[1]) = v120;
  dword_4F063F8 = v127;
  word_4F063FC[0] = v122;
  dword_4F061D8 = v119;
  word_4F061DC[0] = v118;
  if ( a4 )
    goto LABEL_35;
LABEL_58:
  if ( (*(_BYTE *)(v140 + 186) & 0x10) == 0 || dword_4F07590 )
    sub_7365B0(v140, -1);
LABEL_35:
  sub_8CCE20(v14, v134);
  if ( v133 )
    sub_8D0B10();
  return v14;
}
