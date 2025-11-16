// Function: sub_344E280
// Address: 0x344e280
//
__int64 __fastcall sub_344E280(__int64 a1, __int64 a2, _QWORD *a3, __m128i a4)
{
  unsigned __int16 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 result; // rax
  int v10; // r13d
  __int64 v11; // rdx
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __m128i v14; // xmm7
  __int16 *v15; // rax
  unsigned __int16 v16; // cx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int128 v19; // xmm0
  unsigned int v20; // edi
  __int64 v21; // r14
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned __int16 *v27; // rax
  __int64 v28; // r14
  unsigned int v29; // r15d
  __int128 v30; // rax
  __int64 v31; // r9
  int v32; // edx
  __int64 v33; // r9
  __int64 v34; // r9
  int v35; // edx
  __int64 v36; // r9
  unsigned int v37; // edx
  __int128 *v38; // rax
  __int64 v39; // r9
  unsigned int v40; // edx
  unsigned __int8 *v41; // r12
  __int128 v42; // rax
  __int64 v43; // r9
  __int128 v44; // rax
  __int64 v45; // r9
  int v46; // edx
  __int64 v47; // r9
  int v48; // edx
  __int64 v49; // rdx
  __int64 v50; // r9
  unsigned __int8 *v51; // r14
  __int64 v52; // r15
  __int128 v53; // rax
  __int64 v54; // r9
  unsigned int v55; // edx
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // rdx
  __int64 v63; // rsi
  int v64; // r10d
  __int64 v65; // rdx
  __int64 v66; // rbx
  unsigned int v67; // r15d
  __int64 v68; // rdx
  __int128 v69; // rax
  __int64 v70; // r9
  int v71; // edx
  __int64 v72; // r9
  __int64 v73; // r9
  int v74; // edx
  unsigned int v75; // edx
  __int64 v76; // r9
  __int128 *v77; // rax
  __int64 v78; // r9
  unsigned int v79; // edx
  __int128 v80; // rax
  __int64 v81; // r9
  unsigned __int8 *v82; // rax
  unsigned int v83; // edx
  __int64 v84; // r9
  unsigned int v85; // r11d
  int v86; // r14d
  __int128 v87; // rax
  __int64 v88; // r9
  int v89; // r10d
  __int128 v90; // rax
  __int64 v91; // r9
  int v92; // edx
  __int64 v93; // r9
  int v94; // edx
  __int128 v95; // rax
  __int64 v96; // r9
  __int128 v97; // kr00_16
  __int128 v98; // rax
  __int64 v99; // r9
  unsigned int v100; // edx
  __int64 v101; // r9
  __int64 v102; // rdx
  __int64 v103; // rdx
  unsigned int v104; // ecx
  __int64 v105; // rdx
  unsigned int v106; // edx
  __int64 v107; // r9
  __int128 v108; // rax
  __int64 v109; // r9
  unsigned __int8 *v110; // rax
  unsigned int v111; // edx
  int v112; // edx
  __int128 v113; // rax
  __int64 v114; // r9
  __int128 v115; // rax
  __int64 v116; // r9
  int v117; // edx
  int v118; // edx
  __int128 v119; // rax
  __int64 v120; // r9
  int v121; // edx
  unsigned int v122; // edx
  __int64 v123; // r9
  __int128 v124; // rax
  __int64 v125; // r9
  unsigned __int8 *v126; // rax
  __int128 v127; // rax
  __int64 v128; // r9
  unsigned int v129; // edx
  unsigned __int8 *v130; // rax
  int v131; // r11d
  unsigned int v132; // edx
  unsigned int v133; // edx
  unsigned __int8 *v134; // rax
  unsigned int v135; // edx
  __int128 v136; // [rsp-30h] [rbp-340h]
  __int128 v137; // [rsp-30h] [rbp-340h]
  __int128 v138; // [rsp+10h] [rbp-300h]
  int v139; // [rsp+20h] [rbp-2F0h]
  int v140; // [rsp+20h] [rbp-2F0h]
  __int128 v141; // [rsp+20h] [rbp-2F0h]
  __int128 v142; // [rsp+30h] [rbp-2E0h]
  __int128 v143; // [rsp+30h] [rbp-2E0h]
  __int128 v144; // [rsp+40h] [rbp-2D0h]
  __int64 v145; // [rsp+40h] [rbp-2D0h]
  __int128 v146; // [rsp+40h] [rbp-2D0h]
  __int128 v147; // [rsp+40h] [rbp-2D0h]
  __m128i v148; // [rsp+50h] [rbp-2C0h]
  __int64 v149; // [rsp+50h] [rbp-2C0h]
  __int128 v150; // [rsp+50h] [rbp-2C0h]
  __int128 v151; // [rsp+60h] [rbp-2B0h]
  unsigned int v152; // [rsp+60h] [rbp-2B0h]
  __m128i v153; // [rsp+70h] [rbp-2A0h]
  __int128 v154; // [rsp+70h] [rbp-2A0h]
  unsigned int v155; // [rsp+80h] [rbp-290h]
  __int128 v156; // [rsp+80h] [rbp-290h]
  __int128 v157; // [rsp+80h] [rbp-290h]
  __int128 v158; // [rsp+80h] [rbp-290h]
  __int64 v159; // [rsp+90h] [rbp-280h]
  __int128 v160; // [rsp+90h] [rbp-280h]
  unsigned int v161; // [rsp+90h] [rbp-280h]
  __int128 v162; // [rsp+90h] [rbp-280h]
  __int64 v163; // [rsp+90h] [rbp-280h]
  unsigned int v164; // [rsp+90h] [rbp-280h]
  unsigned __int8 *v165; // [rsp+A0h] [rbp-270h]
  unsigned __int8 *v166; // [rsp+B0h] [rbp-260h]
  unsigned int v167; // [rsp+280h] [rbp-90h] BYREF
  __int64 v168; // [rsp+288h] [rbp-88h]
  unsigned int v169; // [rsp+290h] [rbp-80h] BYREF
  __int64 v170; // [rsp+298h] [rbp-78h]
  __int128 v171; // [rsp+2A0h] [rbp-70h] BYREF
  __int128 v172; // [rsp+2B0h] [rbp-60h] BYREF
  __int128 v173; // [rsp+2C0h] [rbp-50h] BYREF
  __int64 v174; // [rsp+2D0h] [rbp-40h] BYREF
  __int64 v175; // [rsp+2D8h] [rbp-38h]

  if ( !sub_33CB110(*(_DWORD *)(a2 + 24)) )
  {
    v6 = *(unsigned __int16 **)(a2 + 48);
    v7 = *v6;
    v8 = *((_QWORD *)v6 + 1);
    LOWORD(v167) = v7;
    v168 = v8;
    if ( (_WORD)v7 )
    {
      if ( (unsigned __int16)(v7 - 17) > 0xD3u )
      {
        v103 = *(_QWORD *)(a2 + 40);
        v159 = *(_QWORD *)(v103 + 80);
        v104 = *(_DWORD *)(v103 + 88);
        LOWORD(v173) = v7;
        v155 = v104;
        *((_QWORD *)&v173 + 1) = v8;
        v142 = (__int128)_mm_loadu_si128((const __m128i *)v103);
        v144 = (__int128)_mm_loadu_si128((const __m128i *)(v103 + 40));
        v153 = _mm_loadu_si128((const __m128i *)(v103 + 80));
LABEL_73:
        if ( (_WORD)v7 != 1 && (unsigned __int16)(v7 - 504) > 7u )
        {
          v149 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v7 - 16];
LABEL_44:
          v63 = *(_QWORD *)(a2 + 80);
          v64 = v149;
          v152 = *(_DWORD *)(a2 + 24);
          *(_QWORD *)&v171 = v63;
          if ( v63 )
          {
            sub_B96E90((__int64)&v171, v63, 1);
            v65 = *(unsigned int *)(a2 + 24);
            v64 = v149;
          }
          else
          {
            v65 = v152;
          }
          DWORD2(v171) = *(_DWORD *)(a2 + 72);
          v66 = *(_QWORD *)(*(_QWORD *)(v159 + 48) + 16LL * v155 + 8);
          v67 = *(unsigned __int16 *)(*(_QWORD *)(v159 + 48) + 16LL * v155);
          if ( (_WORD)v167 != 1 && (!(_WORD)v167 || !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v167 + 112))
            || (unsigned int)v65 > 0x1F3
            || (*(_BYTE *)(v65 + a1 + 500LL * (unsigned __int16)v167 + 6414) & 0xFB) == 0 )
          {
            goto LABEL_53;
          }
          v68 = 1;
          if ( (_WORD)v167 != 1 )
          {
            if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v167 + 112) )
              goto LABEL_53;
            v68 = (unsigned __int16)v167;
          }
          v161 = (v152 == 195) + 195;
          if ( (*(_BYTE *)(v161 + 500 * v68 + a1 + 6414) & 0xFB) == 0 && v64 && (v64 & (v64 - 1)) == 0 )
          {
            if ( (unsigned __int8)sub_3441200(v153.m128i_i64[0], v153.m128i_i64[1], v64) )
            {
              *(_QWORD *)&v80 = sub_3400BD0((__int64)a3, 0, (__int64)&v171, v67, v66, 0, a4, 0);
              v82 = sub_3406EB0(a3, 0x39u, (__int64)&v171, v167, v168, v81, v80, *(_OWORD *)&v153);
              v85 = (v152 == 195) + 195;
            }
            else
            {
              *(_QWORD *)&v127 = sub_3400BD0((__int64)a3, 1, (__int64)&v171, v67, v66, 0, a4, 0);
              v158 = v127;
              if ( v152 == 195 )
              {
                *(_QWORD *)&v144 = sub_340F900(a3, v161, (__int64)&v171, v167, v168, v128, v142, v144, v127);
                *((_QWORD *)&v144 + 1) = v133 | *((_QWORD *)&v144 + 1) & 0xFFFFFFFF00000000LL;
                v134 = sub_3406EB0(a3, 0xC0u, (__int64)&v171, v167, v168, v158, v142, v158);
                v131 = v161;
                *(_QWORD *)&v142 = v134;
                *((_QWORD *)&v142 + 1) = v135 | *((_QWORD *)&v142 + 1) & 0xFFFFFFFF00000000LL;
              }
              else
              {
                *(_QWORD *)&v142 = sub_340F900(a3, v161, (__int64)&v171, v167, v168, v128, v142, v144, v127);
                *((_QWORD *)&v142 + 1) = v129 | *((_QWORD *)&v142 + 1) & 0xFFFFFFFF00000000LL;
                v130 = sub_3406EB0(a3, 0xBEu, (__int64)&v171, v167, v168, v158, v144, v158);
                v131 = (v152 == 195) + 195;
                *(_QWORD *)&v144 = v130;
                *((_QWORD *)&v144 + 1) = v132 | *((_QWORD *)&v144 + 1) & 0xFFFFFFFF00000000LL;
              }
              v164 = v131;
              v82 = sub_34074A0(a3, (__int64)&v171, v153.m128i_i64[0], v153.m128i_i64[1], v67, v66, a4);
              v85 = v164;
            }
            v153.m128i_i64[0] = (__int64)v82;
            v153.m128i_i64[1] = v83 | v153.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            result = sub_340F900(a3, v85, (__int64)&v171, v167, v168, v84, v142, v144, *(_OWORD *)&v153);
LABEL_58:
            if ( (_QWORD)v171 )
            {
              v163 = result;
              sub_B91220((__int64)&v171, v171);
              return v163;
            }
            return result;
          }
LABEL_53:
          *(_QWORD *)&v172 = 0;
          DWORD2(v172) = 0;
          *(_QWORD *)&v173 = 0;
          DWORD2(v173) = 0;
          v140 = v64;
          if ( (unsigned __int8)sub_3441200(v153.m128i_i64[0], v153.m128i_i64[1], v64) )
          {
            *(_QWORD *)&v69 = sub_3400BD0((__int64)a3, (unsigned int)v149, (__int64)&v171, v67, v66, 0, a4, 0);
            v150 = v69;
            *(_QWORD *)&v172 = sub_3406EB0(a3, 0x3Eu, (__int64)&v171, v67, v66, v70, *(_OWORD *)&v153, v69);
            DWORD2(v172) = v71;
            *(_QWORD *)&v173 = sub_3406EB0(a3, 0x39u, (__int64)&v171, v67, v66, v72, v150, v172);
            DWORD2(v173) = v74;
            if ( v152 == 195 )
            {
              v166 = sub_3406EB0(a3, 0xBEu, (__int64)&v171, v167, v168, v73, v142, v172);
              v77 = &v173;
              *(_QWORD *)&v157 = v166;
            }
            else
            {
              v165 = sub_3406EB0(a3, 0xBEu, (__int64)&v171, v167, v168, v73, v142, v173);
              v77 = &v172;
              *(_QWORD *)&v157 = v165;
            }
            *((_QWORD *)&v157 + 1) = v75;
            *(_QWORD *)&v162 = sub_3406EB0(a3, 0xC0u, (__int64)&v171, v167, v168, v76, v144, *v77);
            *((_QWORD *)&v162 + 1) = v79;
          }
          else
          {
            v86 = v140 - 1;
            *(_QWORD *)&v87 = sub_3400BD0((__int64)a3, (unsigned int)(v140 - 1), (__int64)&v171, v67, v66, 0, a4, 0);
            v89 = v140;
            v141 = v87;
            if ( v89 && (v89 & v86) == 0 )
            {
              *(_QWORD *)&v172 = sub_3406EB0(a3, 0xBAu, (__int64)&v171, v67, v66, v88, *(_OWORD *)&v153, v87);
              DWORD2(v172) = v118;
              *(_QWORD *)&v119 = sub_34074A0(a3, (__int64)&v171, v153.m128i_i64[0], v153.m128i_i64[1], v67, v66, a4);
              *(_QWORD *)&v173 = sub_3406EB0(a3, 0xBAu, (__int64)&v171, v67, v66, v120, v119, v141);
              DWORD2(v173) = v121;
            }
            else
            {
              *(_QWORD *)&v90 = sub_3400BD0((__int64)a3, (unsigned int)v149, (__int64)&v171, v67, v66, 0, a4, 0);
              *(_QWORD *)&v172 = sub_3406EB0(a3, 0x3Eu, (__int64)&v171, v67, v66, v91, *(_OWORD *)&v153, v90);
              DWORD2(v172) = v92;
              *(_QWORD *)&v173 = sub_3406EB0(a3, 0x39u, (__int64)&v171, v67, v66, v93, v141, v172);
              DWORD2(v173) = v94;
            }
            *(_QWORD *)&v95 = sub_3400BD0((__int64)a3, 1, (__int64)&v171, v67, v66, 0, a4, 0);
            v97 = v95;
            if ( v152 == 195 )
            {
              *(_QWORD *)&v157 = sub_3406EB0(a3, 0xBEu, (__int64)&v171, v167, v168, v96, v142, v172);
              *((_QWORD *)&v157 + 1) = v122;
              *(_QWORD *)&v124 = sub_3406EB0(a3, 0xC0u, (__int64)&v171, v167, v168, v123, v144, v97);
              v126 = sub_3406EB0(a3, 0xC0u, (__int64)&v171, v167, v168, v125, v124, v173);
              v102 = (unsigned int)v102;
              *(_QWORD *)&v162 = v126;
            }
            else
            {
              *(_QWORD *)&v98 = sub_3406EB0(a3, 0xBEu, (__int64)&v171, v167, v168, v96, v142, v95);
              *(_QWORD *)&v157 = sub_3406EB0(a3, 0xBEu, (__int64)&v171, v167, v168, v99, v98, v173);
              *((_QWORD *)&v157 + 1) = v100;
              *(_QWORD *)&v162 = sub_3406EB0(a3, 0xC0u, (__int64)&v171, v167, v168, v101, v144, v172);
              v102 = (unsigned int)v102;
            }
            *((_QWORD *)&v162 + 1) = v102;
          }
          result = (__int64)sub_3406EB0(a3, 0xBBu, (__int64)&v171, v167, v168, v78, v157, v162);
          goto LABEL_58;
        }
LABEL_93:
        BUG();
      }
      v10 = (unsigned __int16)v7;
      v11 = v7 + 14;
      if ( !*(_QWORD *)(a1 + 8 * v7 + 112) )
        return 0;
      v12 = (_BYTE *)(a1 + 500LL * (unsigned __int16)v7);
      if ( (v12[6604] & 0xFB) != 0
        || !*(_QWORD *)(a1 + 8 * v11)
        || (v12[6606] & 0xFB) != 0
        || !*(_QWORD *)(a1 + 8 * v11)
        || (v12[6471] & 0xFB) != 0
        || !(unsigned __int8)sub_328C7F0(a1, 0xBBu, v167, v8, 0) )
      {
        return 0;
      }
      v7 = *(_QWORD *)(a2 + 40);
      v13 = 0;
      v14 = _mm_loadu_si128((const __m128i *)(v7 + 80));
      v142 = (__int128)_mm_loadu_si128((const __m128i *)v7);
      v159 = *(_QWORD *)(v7 + 80);
      v155 = *(_DWORD *)(v7 + 88);
      v144 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
      LOWORD(v7) = word_4456580[v10 - 1];
      v153 = v14;
    }
    else
    {
      if ( sub_30070B0((__int64)&v167) )
        return 0;
      v58 = *(_QWORD *)(a2 + 40);
      v142 = (__int128)_mm_loadu_si128((const __m128i *)v58);
      v159 = *(_QWORD *)(v58 + 80);
      v155 = *(_DWORD *)(v58 + 88);
      v144 = (__int128)_mm_loadu_si128((const __m128i *)(v58 + 40));
      v153 = _mm_loadu_si128((const __m128i *)(v58 + 80));
      if ( !sub_30070B0((__int64)&v167) )
      {
        *((_QWORD *)&v173 + 1) = v8;
        LOWORD(v173) = 0;
        goto LABEL_43;
      }
      LOWORD(v7) = sub_3009970((__int64)&v167, a2, v59, v60, v61);
    }
    LOWORD(v173) = v7;
    *((_QWORD *)&v173 + 1) = v13;
    if ( !(_WORD)v7 )
    {
LABEL_43:
      v174 = sub_3007260((__int64)&v173);
      v175 = v62;
      LODWORD(v149) = v174;
      goto LABEL_44;
    }
    goto LABEL_73;
  }
  v15 = *(__int16 **)(a2 + 48);
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  *(_QWORD *)&v171 = 0;
  DWORD2(v171) = 0;
  v18 = *(_QWORD *)(a2 + 40);
  LOWORD(v169) = v16;
  v170 = v17;
  v19 = (__int128)_mm_loadu_si128((const __m128i *)v18);
  v20 = *(_DWORD *)(v18 + 88);
  *(_QWORD *)&v172 = 0;
  v21 = *(_QWORD *)(v18 + 80);
  DWORD2(v172) = 0;
  v138 = (__int128)_mm_loadu_si128((const __m128i *)(v18 + 40));
  v148 = _mm_loadu_si128((const __m128i *)(v18 + 80));
  v156 = (__int128)_mm_loadu_si128((const __m128i *)(v18 + 120));
  v160 = (__int128)_mm_loadu_si128((const __m128i *)(v18 + 160));
  if ( v16 )
  {
    if ( (unsigned __int16)(v16 - 17) > 0xD3u )
    {
      LOWORD(v174) = v16;
      v175 = v17;
      goto LABEL_32;
    }
    v16 = word_4456580[v16 - 1];
    v105 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v169) )
    {
      v175 = v17;
      LOWORD(v174) = 0;
LABEL_20:
      *(_QWORD *)&v173 = sub_3007260((__int64)&v174);
      *((_QWORD *)&v173 + 1) = v25;
      LODWORD(v145) = v173;
      goto LABEL_21;
    }
    v16 = sub_3009970((__int64)&v169, 0, v22, v23, v24);
  }
  LOWORD(v174) = v16;
  v175 = v105;
  if ( !v16 )
    goto LABEL_20;
LABEL_32:
  if ( v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
    goto LABEL_93;
  v145 = *(_QWORD *)&byte_444C4A0[16 * v16 - 16];
LABEL_21:
  v26 = *(_QWORD *)(a2 + 80);
  v139 = *(_DWORD *)(a2 + 24);
  v174 = v26;
  if ( v26 )
    sub_B96E90((__int64)&v174, v26, 1);
  LODWORD(v175) = *(_DWORD *)(a2 + 72);
  v27 = (unsigned __int16 *)(*(_QWORD *)(v21 + 48) + 16LL * v20);
  v28 = *((_QWORD *)v27 + 1);
  v29 = *v27;
  if ( (unsigned __int8)sub_3441200(v148.m128i_i64[0], v148.m128i_i64[1], v145) )
  {
    *(_QWORD *)&v30 = sub_3400BD0((__int64)a3, (unsigned int)v145, (__int64)&v174, v29, v28, 0, (__m128i)v19, 0);
    v146 = v30;
    *(_QWORD *)&v171 = sub_33FC130(a3, 406, (__int64)&v174, v29, v28, v31, *(_OWORD *)&v148, v30, v156, v160);
    DWORD2(v171) = v32;
    *(_QWORD *)&v172 = sub_33FC130(a3, 404, (__int64)&v174, v29, v28, v33, v146, v171, v156, v160);
    DWORD2(v172) = v35;
    if ( v139 == 422 )
    {
      *(_QWORD *)&v151 = sub_33FC130(a3, 402, (__int64)&v174, v169, v170, v34, v19, v171, v156, v160);
      *((_QWORD *)&v151 + 1) = v111;
      v38 = &v172;
    }
    else
    {
      *(_QWORD *)&v151 = sub_33FC130(a3, 402, (__int64)&v174, v169, v170, v34, v19, v172, v156, v160);
      *((_QWORD *)&v151 + 1) = v37;
      v38 = &v171;
    }
    *(_QWORD *)&v154 = sub_33FC130(a3, 398, (__int64)&v174, v169, v170, v36, v138, *v38, v156, v160);
    *((_QWORD *)&v154 + 1) = v40;
  }
  else
  {
    *(_QWORD *)&v42 = sub_3400BD0((__int64)a3, (unsigned int)(v145 - 1), (__int64)&v174, v29, v28, 0, (__m128i)v19, 0);
    if ( (_DWORD)v145 && ((unsigned int)v145 & ((_DWORD)v145 - 1)) == 0 )
    {
      v147 = v42;
      *(_QWORD *)&v171 = sub_33FC130(a3, 396, (__int64)&v174, v29, v28, v43, *(_OWORD *)&v148, v42, v156, v160);
      DWORD2(v171) = v112;
      *(_QWORD *)&v113 = sub_34015B0((__int64)a3, (__int64)&v174, v29, v28, 0, 0, (__m128i)v19);
      *(_QWORD *)&v115 = sub_33FC130(a3, 407, (__int64)&v174, v29, v28, v114, *(_OWORD *)&v148, v113, v156, v160);
      *(_QWORD *)&v172 = sub_33FC130(a3, 396, (__int64)&v174, v29, v28, v116, v115, v147, v156, v160);
      DWORD2(v172) = v117;
    }
    else
    {
      v143 = v42;
      *(_QWORD *)&v44 = sub_3400BD0((__int64)a3, (unsigned int)v145, (__int64)&v174, v29, v28, 0, (__m128i)v19, 0);
      *(_QWORD *)&v171 = sub_33FC130(a3, 406, (__int64)&v174, v29, v28, v45, *(_OWORD *)&v148, v44, v156, v160);
      DWORD2(v171) = v46;
      *(_QWORD *)&v172 = sub_33FC130(a3, 404, (__int64)&v174, v29, v28, v47, v143, v171, v156, v160);
      DWORD2(v172) = v48;
    }
    v51 = sub_3400BD0((__int64)a3, 1, (__int64)&v174, v29, v28, 0, (__m128i)v19, 0);
    v52 = v49;
    if ( v139 == 422 )
    {
      *(_QWORD *)&v151 = sub_33FC130(a3, 402, (__int64)&v174, v169, v170, v50, v19, v171, v156, v160);
      *((_QWORD *)&v137 + 1) = v52;
      *(_QWORD *)&v137 = v51;
      *((_QWORD *)&v151 + 1) = v106;
      *(_QWORD *)&v108 = sub_33FC130(a3, 398, (__int64)&v174, v169, v170, v107, v138, v137, v156, v160);
      v110 = sub_33FC130(a3, 398, (__int64)&v174, v169, v170, v109, v108, v172, v156, v160);
      v57 = (unsigned int)v57;
      *(_QWORD *)&v154 = v110;
    }
    else
    {
      *((_QWORD *)&v136 + 1) = v49;
      *(_QWORD *)&v136 = v51;
      *(_QWORD *)&v53 = sub_33FC130(a3, 402, (__int64)&v174, v169, v170, v50, v19, v136, v156, v160);
      *(_QWORD *)&v151 = sub_33FC130(a3, 402, (__int64)&v174, v169, v170, v54, v53, v172, v156, v160);
      *((_QWORD *)&v151 + 1) = v55;
      *(_QWORD *)&v154 = sub_33FC130(a3, 398, (__int64)&v174, v169, v170, v56, v138, v171, v156, v160);
      v57 = (unsigned int)v57;
    }
    *((_QWORD *)&v154 + 1) = v57;
  }
  v41 = sub_33FC130(a3, 400, (__int64)&v174, v169, v170, v39, v151, v154, v156, v160);
  if ( v174 )
    sub_B91220((__int64)&v174, v174);
  return (__int64)v41;
}
