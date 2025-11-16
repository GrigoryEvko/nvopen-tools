// Function: sub_8A2270
// Address: 0x8a2270
//
__int64 **__fastcall sub_8A2270(__int64 a1, __m128i *a2, __int64 a3, __int64 *a4, __int64 a5, int *a6, __m128i *a7)
{
  int v7; // r11d
  __int64 **v10; // rbx
  char v11; // cl
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rsi
  int v15; // edi
  __int64 v16; // rdx
  __int64 *v17; // r14
  __int64 v18; // rsi
  char v19; // al
  char *v20; // r14
  size_t v21; // rax
  __int64 j; // rax
  int v23; // eax
  __int64 **result; // rax
  __int64 **v25; // r14
  unsigned __int64 v26; // rcx
  char v27; // al
  int v28; // r14d
  __int64 v29; // rdx
  char v30; // al
  unsigned __int64 v31; // rax
  const __m128i *v32; // r13
  _QWORD *v33; // r14
  __int64 v34; // rdi
  int v35; // ecx
  __int64 **v36; // rax
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v39; // r14
  __int64 **v40; // rax
  __int64 *v41; // r13
  __int64 v42; // rsi
  __int64 v43; // rcx
  __int64 v44; // r10
  __int64 v45; // r14
  _QWORD *v46; // rbx
  __int64 v47; // rdx
  __int64 v48; // rdi
  __int64 v49; // r14
  unsigned __int64 v50; // r15
  __m128i *v51; // r11
  __int64 **v52; // rdx
  __m128i *v53; // r14
  __int64 v54; // r15
  __int32 v55; // eax
  char v56; // al
  const __m128i *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  char v60; // al
  __int64 **v61; // rax
  __m128i *v62; // r10
  __m128i v63; // xmm6
  __int64 v64; // rax
  __int64 *v65; // r10
  int v66; // r11d
  __int64 **v67; // rdi
  __m128i *v68; // r14
  int v69; // ecx
  __int64 v70; // rax
  char v71; // dl
  __int64 *v72; // rdx
  __int64 v73; // rdi
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rsi
  __int64 v76; // rdi
  __int64 *v77; // rax
  char v78; // dl
  __int64 *v79; // rax
  int v80; // eax
  __int64 v81; // rax
  __int64 v82; // rax
  char v83; // dl
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // r14
  __int64 v88; // rdx
  __m128i *v89; // r8
  __int64 v90; // r13
  __int64 v91; // rax
  int v92; // eax
  __int64 v93; // rax
  char v94; // cl
  __int64 **v95; // rdi
  __int64 v96; // rdx
  __int64 v97; // rax
  __int64 *v98; // r13
  __int64 *v99; // rdx
  __int64 **v100; // rdx
  __int64 v101; // rax
  const __m128i *v102; // r14
  __m128i *v103; // r13
  __m128i v104; // xmm4
  __m128i *v105; // rax
  const __m128i *v106; // rdx
  _QWORD *v107; // rax
  __m128i *v108; // rax
  __m128i *v109; // r11
  __int8 v110; // al
  __m128i *v111; // rdi
  __m128i v112; // xmm4
  __int64 *v113; // rax
  int v114; // eax
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r9
  int v118; // eax
  int v119; // eax
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rdi
  char v123; // al
  __int64 v124; // rax
  int v125; // eax
  _QWORD *v126; // rbx
  __int64 v127; // rax
  __int64 v128; // rcx
  __int64 v129; // r8
  int v130; // eax
  __int64 v131; // [rsp-10h] [rbp-100h]
  __int64 v132; // [rsp-8h] [rbp-F8h]
  __int64 v133; // [rsp-8h] [rbp-F8h]
  int v134; // [rsp+Ch] [rbp-E4h]
  int v135; // [rsp+Ch] [rbp-E4h]
  __int64 v136; // [rsp+10h] [rbp-E0h]
  int v137; // [rsp+10h] [rbp-E0h]
  __int64 *v138; // [rsp+10h] [rbp-E0h]
  int v139; // [rsp+10h] [rbp-E0h]
  int v140; // [rsp+10h] [rbp-E0h]
  int v141; // [rsp+10h] [rbp-E0h]
  __int64 **v142; // [rsp+10h] [rbp-E0h]
  __int64 **v143; // [rsp+20h] [rbp-D0h]
  __int64 v144; // [rsp+20h] [rbp-D0h]
  int v145; // [rsp+20h] [rbp-D0h]
  __int64 v146; // [rsp+28h] [rbp-C8h]
  int v147; // [rsp+30h] [rbp-C0h]
  __m128i *v148; // [rsp+30h] [rbp-C0h]
  __int64 *v149; // [rsp+38h] [rbp-B8h]
  __int64 *v150; // [rsp+40h] [rbp-B0h]
  __m128i *v151; // [rsp+40h] [rbp-B0h]
  int v152; // [rsp+48h] [rbp-A8h]
  int v153; // [rsp+48h] [rbp-A8h]
  __int64 v154; // [rsp+48h] [rbp-A8h]
  __int64 *v155; // [rsp+48h] [rbp-A8h]
  __m128i *v156; // [rsp+48h] [rbp-A8h]
  __int64 v157; // [rsp+48h] [rbp-A8h]
  __int64 *v158; // [rsp+48h] [rbp-A8h]
  __int64 *v159; // [rsp+48h] [rbp-A8h]
  __int64 v160; // [rsp+50h] [rbp-A0h]
  const __m128i *v161; // [rsp+50h] [rbp-A0h]
  __m128i *v162; // [rsp+50h] [rbp-A0h]
  __int16 v163; // [rsp+50h] [rbp-A0h]
  int v164; // [rsp+50h] [rbp-A0h]
  __int64 *v165; // [rsp+50h] [rbp-A0h]
  __m128i *v166; // [rsp+50h] [rbp-A0h]
  unsigned int v167; // [rsp+58h] [rbp-98h]
  int v168; // [rsp+58h] [rbp-98h]
  int v169; // [rsp+58h] [rbp-98h]
  int v170; // [rsp+58h] [rbp-98h]
  __int16 v172; // [rsp+60h] [rbp-90h]
  __m128i *v173; // [rsp+60h] [rbp-90h]
  __int64 **v174; // [rsp+60h] [rbp-90h]
  __m128i *v175; // [rsp+60h] [rbp-90h]
  __m128i *v176; // [rsp+60h] [rbp-90h]
  __m128i *v177; // [rsp+68h] [rbp-88h] BYREF
  __int64 **i; // [rsp+70h] [rbp-80h] BYREF
  __m128i *v179; // [rsp+78h] [rbp-78h] BYREF
  __m128i *v180[2]; // [rsp+80h] [rbp-70h] BYREF
  __m128i v181; // [rsp+90h] [rbp-60h]
  __m128i v182; // [rsp+A0h] [rbp-50h]
  __m128i v183; // [rsp+B0h] [rbp-40h]

  v7 = a5;
  v10 = (__int64 **)a1;
  v11 = *(_BYTE *)(a1 + 89);
  v177 = a2;
  i = 0;
  if ( (v11 & 4) == 0 )
  {
    v13 = a1;
    goto LABEL_9;
  }
  LOBYTE(v12) = *(_BYTE *)(a1 + 140);
  if ( (_BYTE)v12 == 12 )
  {
    if ( (a5 & 0x82000) == 0 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
      if ( (unsigned __int8)(*(_BYTE *)(v14 + 140) - 9) > 2u )
      {
        v13 = a1;
        v167 = a5 & 0xFFFDFFFE;
        goto LABEL_37;
      }
      v17 = *(__int64 **)a1;
      v13 = a1;
LABEL_109:
      v56 = *(_BYTE *)(v14 + 177);
      if ( *(_QWORD *)(v13 + 8) && v56 < 0 )
      {
        v164 = v7;
        v80 = sub_8DBE70(v13);
        v7 = v164;
        if ( !v80 )
        {
          i = *(__int64 ***)(v13 + 160);
          v23 = *a6;
          goto LABEL_67;
        }
        goto LABEL_5;
      }
      if ( (v56 & 0x20) != 0 )
        goto LABEL_112;
      v167 = v7 & 0xFFFDFFFE;
LABEL_37:
      v26 = *(unsigned __int8 *)(v13 + 184);
      if ( (unsigned __int8)v26 > 0xCu )
      {
        v27 = *(_BYTE *)(v13 + 186) & 8;
LABEL_39:
        if ( !v27 )
        {
LABEL_40:
          if ( *a6 )
            goto LABEL_25;
          if ( *(_BYTE *)(v13 + 140) != 12 )
            goto LABEL_68;
          v28 = 0;
          v29 = 797;
          do
          {
            v30 = *(_BYTE *)(v13 + 185);
            v13 = *(_QWORD *)(v13 + 160);
            v28 |= v30 & 0x7F;
            if ( *(_BYTE *)(v13 + 140) != 12 )
              break;
            v31 = *(unsigned __int8 *)(v13 + 184);
            if ( (unsigned __int8)v31 >= 0xAu )
              break;
          }
          while ( _bittest64(&v29, v31) );
          i = (__int64 **)sub_8A2270(v13, (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
          v32 = (const __m128i *)i;
          if ( !v28 || (unsigned int)sub_8D2310(i) )
            goto LABEL_33;
          if ( !(unsigned int)sub_8D4070(v32) )
          {
            i = (__int64 **)sub_73C570(v32, v28);
            v23 = *a6;
            goto LABEL_67;
          }
          goto LABEL_24;
        }
        goto LABEL_188;
      }
      v70 = 6338;
      if ( _bittest64(&v70, v26) )
      {
        v71 = *(_BYTE *)(v13 + 186);
        if ( (v71 & 8) == 0 )
        {
          if ( (unsigned __int8)(v26 - 11) <= 1u )
            goto LABEL_24;
          goto LABEL_150;
        }
        v163 = v7;
        v79 = sub_746BE0(v13);
        LOWORD(v7) = v163;
        if ( v79 && v79[10] )
        {
          i = (__int64 **)sub_730C00(v13, (__int64)v79, (__int64)v177, a3, v167 | 4, a6, (__int64)a7);
          v23 = *a6;
          goto LABEL_67;
        }
        LOBYTE(v26) = *(_BYTE *)(v13 + 184);
      }
      if ( (unsigned __int8)(v26 - 11) <= 1u )
        goto LABEL_24;
      v71 = *(_BYTE *)(v13 + 186);
      v27 = v71 & 8;
      if ( (unsigned __int8)v26 > 0xAu )
        goto LABEL_39;
      if ( ((unsigned __int8)~(0x71DuLL >> v26) & (v27 != 0)) != 0 && (_BYTE)v26 != 1 && (unsigned __int8)(v26 - 6) > 1u )
      {
LABEL_188:
        v81 = sub_8A2270(*(_QWORD *)(v13 + 160), (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
        v82 = sub_8D5290(v81, *(unsigned __int8 *)(v13 + 184), 0, 0);
        v83 = *(_BYTE *)(v82 + 140);
        for ( i = (__int64 **)v82; v83 == 12; v83 = *(_BYTE *)(v82 + 140) )
          v82 = *(_QWORD *)(v82 + 160);
        if ( v83 )
          goto LABEL_21;
        goto LABEL_24;
      }
LABEL_150:
      if ( (_BYTE)v26 != 10 || (v71 & 0x20) == 0 )
        goto LABEL_40;
      v72 = *(__int64 **)(*(_QWORD *)(v13 + 168) + 16LL);
      v73 = *v72;
      if ( *v72 )
      {
        switch ( *(_BYTE *)(v73 + 80) )
        {
          case 4:
          case 5:
            v84 = *(_QWORD *)(*(_QWORD *)(v73 + 96) + 80LL);
            break;
          case 6:
            v84 = *(_QWORD *)(*(_QWORD *)(v73 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            v84 = *(_QWORD *)(*(_QWORD *)(v73 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v84 = *(_QWORD *)(v73 + 88);
            break;
          default:
            goto LABEL_199;
        }
        if ( v84 )
        {
          if ( (*(_BYTE *)(v84 + 267) & 4) == 0 && (v7 & 0x408) == 0 )
          {
            v85 = v72[22];
            if ( !v85 || !*(_QWORD *)(v85 + 16) && (*(_BYTE *)(*(_QWORD *)(v73 + 88) + 160LL) & 0x20) == 0 )
              goto LABEL_40;
          }
        }
      }
LABEL_199:
      v86 = sub_892920(v73);
      v87 = v86;
      v88 = **(_QWORD **)(*(_QWORD *)(v86 + 88) + 32LL);
      if ( a3 == v88 )
        v89 = sub_72F240(v177);
      else
        v89 = (__m128i *)sub_8A55D0(
                           v86,
                           *(_QWORD *)(*(_QWORD *)(v13 + 168) + 8LL),
                           v88,
                           0,
                           (_DWORD)v177,
                           a3,
                           (__int64)a4,
                           v167,
                           (__int64)a6,
                           (__int64)a7);
      if ( !*a6 )
      {
        v165 = (__int64 *)v89;
        v114 = sub_89A370(v89->m128i_i64);
        v89 = (__m128i *)v165;
        if ( !v114 )
        {
          if ( v87 == unk_4D049C8 )
          {
            v125 = sub_88E560(v165, 0, v115, v116, (__int64)v165, v117);
            v89 = (__m128i *)v165;
            if ( v125 )
            {
              v13 = sub_88E770(v165, 0);
LABEL_204:
              i = (__int64 **)v13;
              goto LABEL_40;
            }
          }
          v166 = v89;
          v118 = sub_8A00C0(v87, v89->m128i_i64, 0);
          v89 = v166;
          if ( !v118 )
            *a6 = 1;
        }
      }
      if ( v89 )
        sub_725130(v89->m128i_i64);
      goto LABEL_204;
    }
    v170 = a5;
    v12 = sub_8D2220(a1);
    v7 = v170;
    v11 = *(_BYTE *)(v12 + 89);
    v13 = v12;
    if ( (v11 & 4) != 0 )
    {
      LOBYTE(v12) = *(_BYTE *)(v12 + 140);
      v14 = *(_QWORD *)(*(_QWORD *)(v13 + 40) + 32LL);
      v69 = *(unsigned __int8 *)(v14 + 140);
      v16 = (unsigned int)(v69 - 9);
      if ( (unsigned __int8)(v69 - 9) > 2u )
        goto LABEL_7;
      v17 = *(__int64 **)v13;
      if ( (_BYTE)v12 != 12 )
        goto LABEL_5;
      goto LABEL_109;
    }
LABEL_9:
    LODWORD(v12) = *(unsigned __int8 *)(v13 + 140);
    v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v16 + 12) & 0x10) == 0 )
      goto LABEL_7;
    v16 = (unsigned int)(v12 - 9);
    if ( (unsigned __int8)(v12 - 9) > 2u )
      goto LABEL_7;
    v19 = *(_BYTE *)(v13 + 177) & 0x20;
    if ( v19 && (v11 & 1) != 0 && *(_QWORD *)(v13 + 48) && *(_QWORD *)(v13 + 8) )
    {
      v168 = v7;
      v180[0] = (__m128i *)_mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v181 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v182 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v183 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v180[1] = *(__m128i **)&dword_4F077C8;
      v20 = *(char **)(v13 + 8);
      v21 = strlen(v20);
      sub_878540(v20, v21, (__int64 *)v180);
      v7 = v168;
      for ( j = v180[0][1].m128i_i64[1]; j; j = *(_QWORD *)(j + 8) )
      {
        if ( *(_BYTE *)(j + 80) == 4 )
        {
          i = *(__int64 ***)(j + 88);
          v23 = *a6;
          goto LABEL_67;
        }
      }
      goto LABEL_6;
    }
    v167 = v7 & 0xFFFDFFFE;
LABEL_30:
    if ( v19 )
    {
      v25 = *(__int64 ***)(*(_QWORD *)(v13 + 168) + 256LL);
      if ( v25 )
      {
        i = (__int64 **)sub_8A2270((_DWORD)v25, (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
        if ( v25 != i )
          goto LABEL_33;
      }
      else
      {
        v76 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v13 + 96LL) + 72LL);
        if ( v76 )
        {
          v77 = sub_8A1930(v76, (__int64 **)v13, (int)v177, a3, (__int64)a4, v167, a6, (__int64)a7, &i);
          if ( i )
            goto LABEL_21;
          if ( v77 )
          {
            v78 = *((_BYTE *)v77 + 80);
            if ( v78 == 3 || dword_4F077C4 == 2 && (unsigned __int8)(v78 - 4) <= 2u )
            {
              v36 = (__int64 **)v77[11];
              goto LABEL_121;
            }
          }
LABEL_24:
          *a6 = 1;
          goto LABEL_25;
        }
      }
    }
LABEL_20:
    i = (__int64 **)v13;
    goto LABEL_21;
  }
  v13 = a1;
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  v15 = *(unsigned __int8 *)(v14 + 140);
  v16 = (unsigned int)(v15 - 9);
  if ( (unsigned __int8)(v15 - 9) > 2u )
  {
LABEL_7:
    v18 = v7 & 0xFFFDFFFE;
    v167 = v7 & 0xFFFDFFFE;
    switch ( (char)v12 )
    {
      case 6:
        v33 = (_QWORD *)sub_8A2270(*(_QWORD *)(v13 + 160), (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
        if ( (unsigned int)sub_8D3150(v33) )
          goto LABEL_24;
        if ( (*(_BYTE *)(v13 + 168) & 1) == 0 )
        {
          if ( (unsigned int)sub_8D32E0(v33) )
            goto LABEL_24;
          i = (__int64 **)sub_72D2E0(v33);
          v23 = *a6;
          goto LABEL_67;
        }
        if ( (unsigned int)sub_8D2600(v33) )
          goto LABEL_24;
        if ( !(unsigned int)sub_8D32E0(v33) )
        {
          if ( (*(_BYTE *)(v13 + 168) & 2) != 0 )
            i = (__int64 **)sub_72D6A0(v33);
          else
            i = (__int64 **)sub_72D600(v33);
          v23 = *a6;
          goto LABEL_67;
        }
        v34 = *(_QWORD *)(v13 + 160);
        v35 = 0;
        if ( (*(_BYTE *)(v34 + 140) & 0xFB) == 8 )
          v35 = sub_8D4C10(v34, dword_4F077C4 != 2);
        v36 = (__int64 **)sub_72D790((__int64)v33, (*(_BYTE *)(v13 + 168) & 2) != 0, 0, v35, 0);
        goto LABEL_121;
      case 7:
        ++a7[4].m128i_i32[2];
        v43 = *(_QWORD *)(v13 + 168);
        v146 = a7->m128i_i64[0];
        v161 = (const __m128i *)v43;
        v149 = (__int64 *)a7->m128i_i64[1];
        v150 = *(__int64 **)(v13 + 160);
        v147 = v7 & 8;
        if ( (v7 & 8) != 0 )
          goto LABEL_123;
        if ( (*(_BYTE *)(v43 + 16) & 8) != 0 )
        {
          v44 = *(_QWORD *)(v43 + 40);
          if ( !v44 )
          {
            v45 = 0;
            goto LABEL_73;
          }
        }
        else
        {
          v153 = v7;
          v18 = (__int64)v177;
          v58 = sub_8A4DE0((_DWORD)v150, (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
          v7 = v153;
          v150 = (__int64 *)v58;
          a5 = v132;
LABEL_123:
          v44 = v161[2].m128i_i64[1];
          if ( !v44 )
          {
            v45 = 0;
            goto LABEL_175;
          }
        }
        if ( a7[2].m128i_i64[1] == v44 )
        {
          v45 = a7[3].m128i_i64[0];
        }
        else
        {
          v137 = v7;
          v154 = v44;
          v59 = sub_8A2270(v44, (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
          v43 = v131;
          v18 = v132;
          v7 = v137;
          v44 = v154;
          v45 = v59;
        }
        while ( 1 )
        {
          v60 = *(_BYTE *)(v45 + 140);
          if ( v60 != 12 )
            break;
          v45 = *(_QWORD *)(v45 + 160);
        }
        if ( v60 == 14 )
        {
          v140 = v7;
          v157 = v44;
          v101 = sub_7CFE40(v45, v18, v16, v43, a5, (__int64)a6);
          v7 = v140;
          v44 = v157;
          v45 = v101;
        }
        if ( (unsigned __int8)(*(_BYTE *)(v45 + 140) - 9) > 2u )
          *a6 = 1;
        v152 = 0;
        v47 = 0;
        if ( v45 != v44 )
          goto LABEL_133;
LABEL_175:
        if ( (v161[1].m128i_i8[0] & 8) == 0 )
        {
          v152 = 0;
          v47 = 0;
          if ( *(__int64 **)(v13 + 160) != v150 )
            goto LABEL_133;
        }
LABEL_73:
        v152 = 0;
        if ( !v161->m128i_i64[0] )
          goto LABEL_212;
        v136 = v13;
        v134 = v7;
        v143 = v10;
        v46 = (_QWORD *)v161->m128i_i64[0];
        while ( !v46[10] )
        {
          v90 = sub_8D72A0(v46);
          v91 = sub_8A2270(v90, (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
          if ( v90 != v91 || !v91 )
          {
            v13 = v136;
            v10 = v143;
            v47 = v91;
            v7 = v134;
            goto LABEL_133;
          }
          v46 = (_QWORD *)*v46;
          ++v152;
          if ( !v46 )
          {
            v13 = v136;
            v10 = v143;
            v7 = v134;
LABEL_212:
            v139 = v7;
            v150 = *(__int64 **)(v13 + 160);
            v92 = sub_8DC100(v150);
            v7 = v139;
            if ( !v92 )
            {
              if ( !dword_4F06978 )
                goto LABEL_216;
              v93 = v161[3].m128i_i64[1];
              if ( !v93 )
                goto LABEL_216;
              v94 = *(_BYTE *)v93;
              if ( (*(_BYTE *)v93 & 0x20) != 0 )
                goto LABEL_216;
              if ( (v94 & 0x40) == 0 )
              {
                v121 = *(_QWORD *)(v93 + 8);
                if ( (v94 & 1) != 0 )
                {
                  if ( !v121 || *(_BYTE *)(v121 + 173) != 12 )
                    goto LABEL_216;
                  v47 = 0;
                }
                else
                {
                  if ( v121 )
                  {
                    v142 = v10;
                    v126 = *(_QWORD **)(v93 + 8);
                    v145 = v7;
                    do
                    {
                      if ( (unsigned int)sub_8DC100(v126[1]) )
                      {
                        v10 = v142;
                        v7 = v145;
                        v47 = 0;
                        goto LABEL_133;
                      }
                      v126 = (_QWORD *)*v126;
                    }
                    while ( v126 );
                    v10 = v142;
                    v7 = v145;
                  }
LABEL_216:
                  if ( v161[1].m128i_i8[4] >= 0
                    || (v141 = v7,
                        v113 = sub_736C60(84, *(__int64 **)(v13 + 104)),
                        v7 = v141,
                        *(_BYTE *)(*(_QWORD *)(v113[4] + 40) + 173LL) != 12) )
                  {
                    i = (__int64 **)v13;
                    --a7[4].m128i_i32[2];
                    goto LABEL_21;
                  }
                  v47 = 0;
                }
LABEL_133:
                v135 = v7;
                v144 = v47;
                v61 = (__int64 **)sub_7259C0(7);
                v62 = (__m128i *)v61[21];
                i = v61;
                *v62 = _mm_loadu_si128(v161);
                v62[1] = _mm_loadu_si128(v161 + 1);
                v62[2] = _mm_loadu_si128(v161 + 2);
                v63 = _mm_loadu_si128(v161 + 3);
                v62[2].m128i_i64[1] = v45;
                LOBYTE(v61) = v62[1].m128i_i8[5];
                v62[3] = v63;
                v62->m128i_i64[1] = 0;
                v62[3].m128i_i64[0] = 0;
                v62[1].m128i_i8[5] = (v45 != 0) | (unsigned __int8)v61 & 0xFE;
                v138 = (__int64 *)v62;
                v64 = sub_8A46D0(
                        v161->m128i_i64[0],
                        (_DWORD)v177,
                        a3,
                        (_DWORD)a4,
                        v167,
                        (_DWORD)a6,
                        (__int64)a7,
                        v152,
                        v144);
                v65 = v138;
                *v138 = v64;
                if ( v147 )
                {
                  v95 = i;
                  i[20] = v150;
                  sub_7325D0((__int64)v95, &dword_4F077C8);
                }
                else
                {
                  v66 = v135;
                  if ( (v161[1].m128i_i8[0] & 8) != 0 )
                  {
                    v150 = (__int64 *)sub_8A4DE0(
                                        (_DWORD)v150,
                                        (_DWORD)v177,
                                        a3,
                                        (_DWORD)a4,
                                        v167,
                                        (_DWORD)a6,
                                        (__int64)a7);
                    v65 = v138;
                    v66 = v135;
                  }
                  v67 = i;
                  i[20] = v150;
                  v68 = (__m128i *)v161[3].m128i_i64[1];
                  if ( v68 && dword_4F06978 && (v66 & 0x20000) == 0 )
                  {
                    if ( (v68->m128i_i8[0] & 0x20) == 0 )
                    {
                      v151 = v177;
                      if ( (v68->m128i_i8[0] & 0x40) != 0 && (v158 = v65, sub_8955E0(v68, a6), v65 = v158, *a6) )
                      {
                        v67 = i;
                        v68 = 0;
                      }
                      else
                      {
                        v159 = v65;
                        v108 = sub_725E60();
                        v65 = v159;
                        v109 = v108;
                        v110 = v68->m128i_i8[0];
                        if ( (v68->m128i_i8[0] & 2) != 0 )
                        {
                          v109->m128i_i8[0] |= 2u;
                          v110 = v68->m128i_i8[0];
                        }
                        if ( (v110 & 4) != 0 )
                        {
                          v109->m128i_i8[0] |= 4u;
                          v110 = v68->m128i_i8[0];
                        }
                        if ( (v110 & 8) != 0 )
                        {
                          v109->m128i_i8[0] |= 8u;
                          v110 = v68->m128i_i8[0];
                        }
                        if ( (v110 & 1) != 0 )
                        {
                          v111 = (__m128i *)v68->m128i_i64[1];
                          v109->m128i_i8[0] |= 1u;
                          if ( v111 )
                          {
                            if ( v111[10].m128i_i8[13] == 12 )
                            {
                              v148 = v109;
                              v127 = sub_744A50(v111, v151, a3, 0, a4, v167, a6, a7);
                              v65 = v159;
                              v109 = v148;
                              v111 = (__m128i *)v127;
                              if ( *(_BYTE *)(v127 + 173) != 12 )
                              {
                                v130 = sub_711520(v127, (__int64)v151, v133, v128, v129);
                                v65 = v159;
                                v109 = v148;
                                if ( !v130 )
                                  v148->m128i_i8[0] &= ~4u;
                              }
                            }
                            v109->m128i_i64[1] = (__int64)v111;
                          }
                        }
                        v112 = _mm_loadu_si128(v68 + 1);
                        v67 = i;
                        v68 = v109;
                        v109[1] = v112;
                      }
                    }
                    v65[7] = (__int64)v68;
                  }
                  v155 = v65;
                  sub_7325D0((__int64)v67, &dword_4F077C8);
                  if ( v161[1].m128i_i8[4] < 0 )
                  {
                    v102 = (const __m128i *)sub_736C60(84, *(__int64 **)(v13 + 104));
                    v103 = (__m128i *)sub_727670();
                    *v103 = _mm_loadu_si128(v102);
                    v103[1] = _mm_loadu_si128(v102 + 1);
                    v103[2] = _mm_loadu_si128(v102 + 2);
                    v103[3] = _mm_loadu_si128(v102 + 3);
                    v104 = _mm_loadu_si128(v102 + 4);
                    v103->m128i_i64[0] = 0;
                    v103[4] = v104;
                    v105 = (__m128i *)sub_7276D0();
                    v103[2].m128i_i64[0] = (__int64)v105;
                    v106 = (const __m128i *)v102[2].m128i_i64[0];
                    *v105 = _mm_loadu_si128(v106);
                    v105[1] = _mm_loadu_si128(v106 + 1);
                    v105[2] = _mm_loadu_si128(v106 + 2);
                    v107 = (_QWORD *)sub_72C390();
                    *(_QWORD *)(v103[2].m128i_i64[0] + 40) = sub_744A50(
                                                               *(__m128i **)(v102[2].m128i_i64[0] + 40),
                                                               v177,
                                                               a3,
                                                               v107,
                                                               a4,
                                                               v167,
                                                               a6,
                                                               a7);
                    *((_BYTE *)v155 + 20) |= 0x80u;
                    v103->m128i_i64[0] = (__int64)i[13];
                    i[13] = (__int64 *)v103;
                  }
                }
                --a7[4].m128i_i32[2];
                a7->m128i_i64[0] = v146;
                a7->m128i_i64[1] = (__int64)v149;
                if ( v149 )
                {
                  sub_8921C0(*v149);
                  *v149 = 0;
                  v23 = *a6;
                  goto LABEL_67;
                }
LABEL_33:
                v23 = *a6;
LABEL_67:
                if ( !v23 )
                {
LABEL_68:
                  result = i;
                  goto LABEL_26;
                }
LABEL_25:
                result = (__int64 **)sub_72C930();
LABEL_26:
                if ( !result )
                  return v10;
                return result;
              }
            }
LABEL_77:
            v47 = 0;
            goto LABEL_133;
          }
        }
        v13 = v136;
        v10 = v143;
        v7 = v134;
        goto LABEL_77;
      case 8:
        v48 = *(_QWORD *)(v13 + 160);
        v179 = 0;
        v49 = sub_8A2270(v48, (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
        if ( *(char *)(v13 + 168) < 0 )
        {
          v162 = *(__m128i **)(v13 + 176);
          if ( !v162 )
          {
            v179 = 0;
            v50 = 0;
            if ( v49 != *(_QWORD *)(v13 + 160) )
              goto LABEL_84;
            goto LABEL_228;
          }
          v96 = a3;
          v50 = 0;
          v97 = sub_744A50(v162, v177, v96, 0, a4, v167, a6, a7);
          v51 = 0;
          v179 = (__m128i *)v97;
        }
        else if ( (*(_BYTE *)(v13 + 169) & 1) != 0 )
        {
          v51 = *(__m128i **)(v13 + 176);
          if ( v51 )
          {
            v156 = *(__m128i **)(v13 + 176);
            v180[0] = (__m128i *)sub_724DC0();
            v74 = sub_7410C0(v156, v177, a3, 0, a4, v167, a6, a7->m128i_i64, v180[0], &v179);
            v75 = (unsigned int)*a6;
            v51 = v156;
            v50 = v74;
            if ( !(_DWORD)v75 && !v179 )
            {
              if ( !v74
                || (v75 = (unsigned __int64)v180[0],
                    v119 = sub_719770(v74, (__int64)v180[0]->m128i_i64, 1u, 1u),
                    v51 = v156,
                    v119) )
              {
                v176 = v51;
                v50 = 0;
                v120 = sub_724E50((__int64 *)v180, (_BYTE *)v75);
                v51 = v176;
                v179 = (__m128i *)v120;
              }
            }
            if ( v180[0] )
            {
              v175 = v51;
              sub_724E30((__int64)v180);
              v51 = v175;
            }
          }
          else
          {
            v50 = 0;
          }
          v162 = 0;
        }
        else
        {
          v162 = 0;
          v50 = 0;
          v51 = 0;
        }
        if ( v49 != *(_QWORD *)(v13 + 160) || v179 != v162 || v51 != (__m128i *)v50 )
        {
LABEL_84:
          if ( (unsigned int)sub_8D2310(v49)
            || (unsigned int)sub_8D2600(v49)
            || (unsigned int)sub_8D32E0(v49)
            || (unsigned int)sub_8D5830(v49)
            || (unsigned int)sub_8D23E0(v49) )
          {
LABEL_97:
            *a6 = 1;
            i = 0;
            goto LABEL_25;
          }
          v173 = (__m128i *)sub_7259C0(8);
          sub_73C230((const __m128i *)v13, v173);
          v52 = (__int64 **)v173;
          v173[8].m128i_i8[14] &= 0xFCu;
          v173[10].m128i_i64[0] = v49;
          if ( v50 )
          {
            v173[11].m128i_i64[0] = v50;
          }
          else
          {
            v122 = (__int64)v179;
            if ( v179 != v162 )
            {
              v173[10].m128i_i8[9] &= ~1u;
              v123 = *(_BYTE *)(v122 + 173);
              if ( v123 == 1 )
              {
                v173[10].m128i_i8[8] &= ~0x80u;
                v124 = sub_620FD0(v122, v180);
                v52 = (__int64 **)v173;
                v173[11].m128i_i64[0] = v124;
                if ( !LODWORD(v180[0]) && v124 )
                  goto LABEL_91;
              }
              else if ( v123 == 12 )
              {
                v173[11].m128i_i64[0] = v122;
                goto LABEL_91;
              }
              *a6 = 1;
            }
          }
LABEL_91:
          v174 = v52;
          if ( !(unsigned int)sub_8D62B0(v52, 1) )
          {
            *a6 = 1;
            i = v174;
            goto LABEL_25;
          }
          v23 = *a6;
          v13 = (__int64)v174;
LABEL_93:
          i = (__int64 **)v13;
          goto LABEL_67;
        }
LABEL_228:
        v23 = *a6;
        goto LABEL_93;
      case 9:
      case 10:
      case 11:
        v19 = *(_BYTE *)(v13 + 177) & 0x20;
        goto LABEL_30;
      case 12:
        goto LABEL_37;
      case 13:
        v53 = (__m128i *)sub_8A2270(*(_QWORD *)(v13 + 168), (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
        v54 = sub_8A2270(*(_QWORD *)(v13 + 160), (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
        if ( *(_OWORD *)(v13 + 160) == __PAIR128__((unsigned __int64)v53, v54) )
          goto LABEL_20;
        if ( !(unsigned int)sub_8D3A70(v54) && !(unsigned int)sub_8D3D40(v54)
          || (unsigned int)sub_8D2600(v53)
          || (unsigned int)sub_8D32E0(v53) )
        {
          goto LABEL_97;
        }
        if ( !dword_4F077BC )
        {
          while ( *(_BYTE *)(v54 + 140) == 12 )
            v54 = *(_QWORD *)(v54 + 160);
        }
        i = (__int64 **)sub_73F0A0(v53, v54);
        v23 = *a6;
        goto LABEL_67;
      case 14:
        v37 = *(_QWORD *)(v13 + 168);
        if ( *(_DWORD *)(v37 + 28) == -2 )
        {
          v98 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v37 + 32) + 88LL) + 104LL);
          v99 = (__int64 *)sub_8A4520((_DWORD)v98, (_DWORD)v177, a3, (_DWORD)a4, v167, (_DWORD)a6, (__int64)a7);
          v23 = *a6;
          if ( v98 == v99 || v23 )
          {
            v100 = 0;
          }
          else
          {
            v100 = (__int64 **)sub_72EF10(*v99);
            v23 = *a6;
          }
          i = v100;
          goto LABEL_67;
        }
        v172 = v7;
        v160 = *(_QWORD *)(v13 + 168);
        v38 = sub_8A4460(v37 + 24, v167, &v177, a3);
        v39 = v38;
        if ( !v38 || *(_BYTE *)(v38 + 8) || (v40 = *(__int64 ***)(v38 + 32)) == 0 )
        {
          i = (__int64 **)v13;
          if ( (v172 & 0x2000) != 0 )
          {
            v55 = a7[5].m128i_i32[2];
            if ( *(_BYTE *)(v13 + 140) == 14 && (*(_BYTE *)(v13 + 161) & 1) != 0 )
              v55 |= 1u;
            a7[5].m128i_i32[2] = v55;
          }
          goto LABEL_21;
        }
        i = v40;
        if ( (*(_BYTE *)(v39 + 25) & 2) != 0 )
          goto LABEL_24;
        v41 = (__int64 *)a7[4].m128i_i64[0];
        if ( v41 )
        {
          LODWORD(v180[0]) = 0;
          v42 = v41[2];
          if ( *(unsigned int *)(v160 + 24) >= v42 )
            v42 = *(unsigned int *)(v160 + 24);
          sub_89F5F0(v41, v42, v180);
          *(_DWORD *)(*v41 + 4LL * (unsigned int)(*(_DWORD *)(v160 + 24) - 1)) = 1;
        }
        if ( (*(_BYTE *)(v39 + 24) & 0x10) != 0 )
        {
          a7[5].m128i_i32[2] = 1;
          v23 = *a6;
          goto LABEL_67;
        }
        goto LABEL_33;
      case 15:
        if ( (unsigned int)sub_8DBF30(v13) )
          goto LABEL_24;
        goto LABEL_20;
      default:
        goto LABEL_20;
    }
  }
  v17 = *v10;
  v13 = (__int64)v10;
LABEL_5:
  if ( (*(_BYTE *)(v14 + 177) & 0x20) == 0 )
  {
LABEL_6:
    LOBYTE(v12) = *(_BYTE *)(v13 + 140);
    goto LABEL_7;
  }
LABEL_112:
  v169 = v7;
  v57 = sub_8A1CE0((__int64)v17, v14, (int)v177, a3, a4, 1, &i, v7 & 0xFFFDFFFF, a6, (__int64)a7);
  if ( !i )
  {
    v7 = v169;
    if ( v57 )
    {
      v16 = v57[5].m128i_u8[0];
      if ( (_BYTE)v16 == 16 )
      {
        v57 = *(const __m128i **)v57[5].m128i_i64[1];
        v16 = v57[5].m128i_u8[0];
      }
      if ( (_BYTE)v16 == 24 )
      {
        v57 = (const __m128i *)v57[5].m128i_i64[1];
        if ( !v57 )
          goto LABEL_119;
        v16 = v57[5].m128i_u8[0];
      }
      if ( (_BYTE)v16 == 3 || dword_4F077C4 == 2 && (v16 = (unsigned int)(v16 - 4), (unsigned __int8)v16 <= 2u) )
      {
        v36 = (__int64 **)v57[5].m128i_i64[1];
LABEL_120:
        if ( v36 == (__int64 **)v13 )
          goto LABEL_6;
LABEL_121:
        i = v36;
        v23 = *a6;
        goto LABEL_67;
      }
    }
LABEL_119:
    *a6 = 1;
    v36 = (__int64 **)sub_72C930();
    v7 = v169;
    goto LABEL_120;
  }
LABEL_21:
  if ( *a6 )
    goto LABEL_25;
  return i;
}
