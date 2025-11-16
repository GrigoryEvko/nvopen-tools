// Function: sub_77D750
// Address: 0x77d750
//
_BOOL8 __fastcall sub_77D750(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r14
  __m128i *v8; // r12
  char i; // al
  unsigned __int64 v11; // rbx
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdi
  __m128i *v15; // rsi
  _BYTE *v16; // r14
  unsigned __int64 v17; // r11
  char i3; // al
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rbx
  _BOOL4 v21; // r10d
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rbx
  __m128i si128; // xmm6
  unsigned __int64 v27; // r11
  char kk; // al
  __int64 v29; // rax
  unsigned __int64 v30; // r12
  __int64 v31; // r10
  _BYTE *v32; // rbx
  unsigned int v33; // edx
  __int64 v34; // rax
  _BYTE *v35; // r13
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // r15
  int v39; // r9d
  __int64 v40; // r8
  __int32 v41; // r13d
  unsigned __int64 i2; // rdx
  unsigned int v43; // edx
  __int64 v44; // rax
  int v45; // r11d
  __int64 v46; // rax
  __int64 v47; // rdi
  _QWORD *v48; // rcx
  unsigned __int64 v49; // r11
  __int64 v50; // rsi
  unsigned __int64 v51; // rdx
  unsigned int v52; // edx
  __int64 v53; // rax
  _QWORD *v54; // r14
  _QWORD *v55; // rbx
  _QWORD *v56; // rdi
  unsigned __int64 v57; // rcx
  unsigned __int64 i1; // rdx
  unsigned int v59; // edx
  __int64 v60; // rax
  bool v61; // al
  __int8 v62; // al
  __int64 v63; // r15
  char v64; // al
  __int64 v65; // rax
  __m128i *v66; // rax
  __int64 v67; // r15
  __int64 v68; // rsi
  __int64 v69; // rdx
  _QWORD *v70; // rcx
  _QWORD *jj; // rax
  __int64 v72; // rax
  unsigned int v73; // eax
  unsigned int v74; // eax
  unsigned __int64 v75; // rbx
  unsigned int mm; // edx
  __int64 v77; // rax
  __int64 v78; // rsi
  __int64 v79; // r12
  char nn; // al
  _BYTE *v81; // r14
  unsigned __int64 v82; // r15
  char v83; // al
  __int64 v84; // rsi
  int v85; // r8d
  __int64 v86; // rdi
  unsigned int j; // edx
  _QWORD *v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rax
  __int64 v91; // rbx
  unsigned __int64 m; // rcx
  int v93; // eax
  __int64 v94; // r15
  __int64 v95; // r12
  _BYTE *v96; // rbx
  char v97; // dl
  unsigned __int64 v98; // rax
  __int64 v99; // rsi
  __int64 **v100; // rdx
  unsigned int v101; // esi
  __int64 v102; // rdx
  __int64 v103; // r9
  __int64 v104; // r13
  __int64 v105; // rcx
  __int64 v106; // r8
  __int64 v107; // rsi
  __int64 v108; // rdx
  bool v109; // al
  char v110; // al
  char v111; // dl
  __int64 v112; // rsi
  unsigned int v113; // esi
  _BYTE *v114; // rax
  _BYTE *v115; // rdx
  _BYTE *v116; // rax
  _BYTE *v117; // r8
  char v118; // al
  const __m128i *v119; // rdi
  int v120; // r8d
  unsigned __int64 v121; // rcx
  unsigned int k; // edx
  _QWORD *v123; // rax
  int v124; // esi
  __int64 v125; // rcx
  __int64 v126; // rax
  __int64 v127; // rdx
  unsigned int v128; // r8d
  _QWORD *v129; // rdx
  int v130; // ebx
  unsigned int v131; // esi
  unsigned int v132; // eax
  __int64 v133; // rax
  _BYTE *v134; // rax
  int v135; // r8d
  unsigned int n; // edx
  __int64 v137; // rax
  unsigned __int64 v138; // rcx
  unsigned int v139; // edx
  __int64 *v140; // rax
  __int64 v141; // r8
  unsigned __int64 v142; // rbx
  const __m128i *v143; // r15
  _QWORD *v144; // rax
  __int64 ii; // rbx
  unsigned __int64 v146; // rdi
  unsigned int v147; // ecx
  __int64 v148; // rsi
  unsigned int v149; // edx
  __m128i *v150; // rax
  __m128i v151; // xmm0
  __m128i *v152; // rax
  _BYTE *v153; // [rsp+0h] [rbp-80h]
  unsigned __int64 v154; // [rsp+8h] [rbp-78h]
  unsigned __int64 v155; // [rsp+8h] [rbp-78h]
  __int64 v156; // [rsp+10h] [rbp-70h]
  __int64 v157; // [rsp+10h] [rbp-70h]
  __int64 v158; // [rsp+10h] [rbp-70h]
  char v159; // [rsp+10h] [rbp-70h]
  __m128i *v160; // [rsp+10h] [rbp-70h]
  __int64 v161; // [rsp+10h] [rbp-70h]
  __int64 v162; // [rsp+18h] [rbp-68h]
  unsigned __int64 v163; // [rsp+18h] [rbp-68h]
  unsigned __int64 v164; // [rsp+18h] [rbp-68h]
  __int64 v165; // [rsp+18h] [rbp-68h]
  unsigned __int64 v166; // [rsp+18h] [rbp-68h]
  __int64 v167; // [rsp+18h] [rbp-68h]
  unsigned int v168; // [rsp+18h] [rbp-68h]
  unsigned int v169; // [rsp+18h] [rbp-68h]
  unsigned int v170; // [rsp+18h] [rbp-68h]
  unsigned int v172; // [rsp+20h] [rbp-60h]
  unsigned int v173; // [rsp+20h] [rbp-60h]
  unsigned __int64 v174; // [rsp+20h] [rbp-60h]
  __int64 v175; // [rsp+20h] [rbp-60h]
  char v176; // [rsp+20h] [rbp-60h]
  __int64 v177; // [rsp+20h] [rbp-60h]
  _QWORD *v178; // [rsp+20h] [rbp-60h]
  unsigned __int64 v179; // [rsp+28h] [rbp-58h]
  _BYTE *v180; // [rsp+28h] [rbp-58h]
  unsigned __int64 v181; // [rsp+28h] [rbp-58h]
  _BYTE *v182; // [rsp+28h] [rbp-58h]
  __int64 v183; // [rsp+28h] [rbp-58h]
  unsigned __int64 v184; // [rsp+28h] [rbp-58h]
  unsigned __int64 v185; // [rsp+28h] [rbp-58h]
  bool v186; // [rsp+28h] [rbp-58h]
  _BYTE *v187; // [rsp+28h] [rbp-58h]
  __int64 v188; // [rsp+28h] [rbp-58h]
  unsigned __int64 v189; // [rsp+28h] [rbp-58h]
  unsigned __int64 v190; // [rsp+28h] [rbp-58h]
  unsigned int v191; // [rsp+30h] [rbp-50h] BYREF
  int v192; // [rsp+34h] [rbp-4Ch] BYREF
  const __m128i *v193; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 v194; // [rsp+40h] [rbp-40h] BYREF
  __int64 v195[7]; // [rsp+48h] [rbp-38h] BYREF

  v6 = (__int64)a2;
  v8 = (__m128i *)a5;
  v191 = 1;
  sub_724C70(a5, 0);
  v8[8].m128i_i64[0] = a4;
  for ( i = *(_BYTE *)(a4 + 140); i == 12; i = *(_BYTE *)(a4 + 140) )
    a4 = *(_QWORD *)(a4 + 160);
  if ( *(char *)(a1 + 132) < 0 || (unsigned __int8)(i - 9) <= 2u && (*(_BYTE *)(a4 + 176) & 0x40) != 0 )
  {
    v8[10].m128i_i8[11] |= 2u;
    if ( *(_BYTE *)(a4 + 140) > 0x14u )
LABEL_8:
      sub_721090();
  }
  switch ( *(_BYTE *)(a4 + 140) )
  {
    case 0:
      v191 = 0;
      v21 = 0;
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_6855B0(0xAA1u, (FILE *)(a1 + 112), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        v21 = v191;
      }
      *(_BYTE *)(a1 + 132) |= 0x40u;
      return v21;
    case 1:
      sub_724A80((__int64)v8, 14);
      return (_BOOL4)v191;
    case 2:
      sub_724A80((__int64)v8, 1);
      v61 = 1;
      v8[11] = _mm_loadu_si128(a2);
      if ( *(char *)(a1 + 132) >= 0 )
        v61 = dword_4F077C4 == 2;
      v8[10].m128i_i8[9] = (4 * v61) | v8[10].m128i_i8[9] & 0xFB;
      return (_BOOL4)v191;
    case 3:
      sub_724A80((__int64)v8, 3);
      v21 = v191;
      v8[11] = _mm_loadu_si128(a2);
      return v21;
    case 4:
      sub_724A80((__int64)v8, 5);
      v21 = v191;
      v8[11] = _mm_loadu_si128(a2);
      return v21;
    case 5:
      sub_724A80((__int64)v8, 4);
      v66 = (__m128i *)v8[11].m128i_i64[0];
      *v66 = _mm_loadu_si128(a2);
      v21 = v191;
      v66[1] = _mm_loadu_si128(a2 + 1);
      return v21;
    case 6:
      v62 = a2->m128i_i8[8];
      if ( (v62 & 1) == 0 )
      {
        if ( (v62 & 0x20) != 0 )
        {
          v67 = a2[1].m128i_i64[0];
          if ( (*(_BYTE *)(v67 + 193) & 4) == 0 )
          {
            v104 = *(_QWORD *)(a4 + 160);
            sub_72D3B0(a2[1].m128i_i64[0], (__int64)v8, 1);
            v8[8].m128i_i64[0] = a4;
            v107 = *(_QWORD *)(v67 + 152);
            if ( v107 == v104 || (unsigned int)sub_8D97D0(v104, v107, 0, v105, v106) )
              goto LABEL_102;
            goto LABEL_101;
          }
          if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
          {
            sub_686E10(0xB76u, (FILE *)(a1 + 112), *(_QWORD *)v67, (_QWORD *)(a1 + 96));
            sub_770D30(a1);
            v62 = a2->m128i_i8[8];
          }
          v191 = 0;
          goto LABEL_115;
        }
        if ( !a2->m128i_i64[0] )
        {
          sub_724A80((__int64)v8, 1);
          v109 = 1;
          v8[10].m128i_i8[8] |= 8u;
          if ( *(char *)(a1 + 132) >= 0 )
            v109 = dword_4F077C4 == 2;
          v8[10].m128i_i8[9] = (4 * v109) | v8[10].m128i_i8[9] & 0xFB;
          v62 = a2->m128i_i8[8];
          goto LABEL_115;
        }
        if ( a2->m128i_i32[3] > 1u )
        {
          v191 = 0;
          if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
          {
            sub_6855B0(0xAA3u, (FILE *)(a1 + 112), (_QWORD *)(a1 + 96));
            sub_770D30(a1);
            v62 = a2->m128i_i8[8];
          }
          goto LABEL_115;
        }
        v186 = 1;
        v82 = *(_QWORD *)(a4 + 160);
        if ( a2->m128i_i64[0] == a2[1].m128i_i64[1] )
          v186 = (v62 & 8) != 0;
        v83 = *(_BYTE *)(a1 + 133);
        v176 = v83 & 1;
        if ( (v83 & 1) != 0 )
          *(_BYTE *)(a1 + 133) = v83 & 0xFE;
        sub_724A80((__int64)v8, 6);
        v84 = a2[1].m128i_i64[1];
        v85 = *(_DWORD *)(a1 + 8);
        v86 = *(_QWORD *)a1;
        for ( j = v85 & (*(_QWORD *)(v6 + 24) >> 3); ; j = v85 & (j + 1) )
        {
          v88 = (_QWORD *)(v86 + 16LL * j);
          if ( v84 == *v88 )
            break;
          if ( !*v88 )
            goto LABEL_223;
        }
        v89 = v88[1];
        if ( v89 )
        {
          if ( *(_BYTE *)(v89 + 173) == 6 )
          {
            v90 = *(_QWORD *)(v89 + 184);
            if ( *(_BYTE *)(v89 + 176) == 1 )
            {
              if ( !v90 )
                return 0;
              v193 = 0;
            }
            else
            {
              v193 = *(const __m128i **)(v89 + 184);
              v90 = 0;
            }
          }
          else
          {
            v193 = (const __m128i *)v88[1];
            v90 = 0;
          }
        }
        else
        {
LABEL_223:
          if ( (*(_BYTE *)(v82 + 140) & 0xFB) == 8 && (sub_8D4C10(v82, dword_4F077C4 != 2) & 1) != 0 )
            *(_BYTE *)(v6 + 8) |= 0x40u;
          if ( (*(_DWORD *)(v6 + 8) & 0xFFFFFF00) != 0 )
          {
            v110 = *(_BYTE *)(v6 + 8);
            if ( (v110 & 0x10) == 0 )
            {
              while ( 1 )
              {
                v111 = *(_BYTE *)(v82 + 140);
                if ( v111 != 12 )
                  break;
                v82 = *(_QWORD *)(v82 + 160);
              }
              if ( (v110 & 8) != 0 )
              {
                v112 = *(_QWORD *)(v6 + 16);
                if ( (v110 & 4) != 0 )
                  v112 = *(_QWORD *)(v112 + 24);
                v169 = *(_DWORD *)v6 - v112;
                if ( v169 )
                {
                  if ( (*(_BYTE *)(v82 + 141) & 0x20) != 0 && v111 == 8 )
                  {
                    do
                      v82 = *(_QWORD *)(v82 + 160);
                    while ( *(_BYTE *)(v82 + 140) == 12 );
                  }
                  v113 = 16;
                  if ( (unsigned __int8)(*(_BYTE *)(v82 + 140) - 2) > 1u )
                    v113 = sub_7764B0(a1, v82, &v191);
                  v8[12].m128i_i64[0] = *(_QWORD *)(v82 + 128) * (v169 / v113);
                }
              }
            }
          }
          if ( !(unsigned int)sub_77F070(a1, v6, &v193) )
            return 0;
          v90 = 0;
        }
        if ( (*(_BYTE *)(v6 + 8) & 8) != 0 || (*(_BYTE *)(a4 + 168) & 1) != 0 )
          v8[10].m128i_i8[8] |= 8u;
        if ( v90 )
        {
          if ( (*(_BYTE *)(v90 + 176) & 8) != 0 || *(_BYTE *)(v90 + 136) > 2u )
          {
            v191 = 0;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_6855B0(0xA8Du, (FILE *)(a1 + 112), (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
          }
          else
          {
            v8[11].m128i_i8[0] = 1;
            v8[11].m128i_i64[1] = v90;
          }
          goto LABEL_175;
        }
        v119 = v193;
        if ( v193[10].m128i_i8[13] != 2 )
        {
          if ( (*(_BYTE *)(a1 + 132) & 1) == 0 )
          {
            v191 = 0;
            sub_770DD0(0xA8Du, (FILE *)(a1 + 112), a1);
            v119 = v193;
            goto LABEL_282;
          }
          if ( (*(_BYTE *)(a1 + 133) & 4) == 0 || (v193[10].m128i_i8[11] & 1) == 0 )
          {
            v8[11].m128i_i8[0] = 3;
            if ( *(char *)(v6 + 8) >= 0 && *(_DWORD *)(v6 + 12) || (*(_BYTE *)(a1 + 133) & 4) == 0 && !v176 )
            {
              v120 = *(_DWORD *)(a1 + 8);
              v121 = *(_QWORD *)(v6 + 24) - 1LL;
              for ( k = v120 & (v121 >> 3); ; k = v120 & (k + 1) )
              {
                v123 = (_QWORD *)(*(_QWORD *)a1 + 16LL * k);
                if ( v121 == *v123 )
                  break;
                if ( !*v123 )
                  goto LABEL_315;
              }
              if ( v123[1] == a1 + 144 )
              {
                v8[11].m128i_i8[0] = 2;
                if ( dword_4F07270[0] != unk_4F073B8 )
                {
                  sub_7296C0(v195);
                  v124 = *(_DWORD *)(a1 + 8);
                  v125 = *(_QWORD *)a1;
                  LODWORD(v126) = v124 & (*(_QWORD *)(v6 + 24) >> 3);
                  do
                  {
                    v127 = (unsigned int)v126;
                    v128 = v126;
                    v126 = v124 & (unsigned int)(v126 + 1);
                    v129 = (_QWORD *)(v125 + 16 * v127);
                  }
                  while ( *(_QWORD *)(v6 + 24) != *v129 );
                  *v129 = 0;
                  if ( *(_QWORD *)(v125 + 16 * v126) )
                    sub_771200(*(_QWORD *)a1, *(_DWORD *)(a1 + 8), v128);
                  --*(_DWORD *)(a1 + 12);
                  v130 = sub_77F070(a1, v6, &v193);
                  sub_729730(v195[0]);
                  if ( !v130 )
                    return 0;
                  v119 = v193;
                }
                v119[10].m128i_i8[12] |= 0x10u;
                v119 = v193;
              }
              else
              {
LABEL_315:
                v191 = 0;
                sub_770DD0(0xAFFu, (FILE *)(a1 + 112), a1);
                v119 = v193;
              }
            }
            goto LABEL_282;
          }
LABEL_281:
          v8[11].m128i_i8[0] = 2;
LABEL_282:
          v8[11].m128i_i64[1] = (__int64)v119;
LABEL_175:
          if ( !v186 )
            goto LABEL_102;
          v91 = *(_QWORD *)(v6 + 24);
          v177 = *(_QWORD *)v6;
          if ( *(_QWORD *)v6 == v91 )
            goto LABEL_102;
          v194 = 0;
          v195[0] = 0;
          m = *(_QWORD *)(v91 - 8);
          if ( (*(_BYTE *)(v6 + 8) & 2) != 0 )
          {
            v192 = 1;
            v93 = 16;
            if ( (unsigned __int8)(*(_BYTE *)(m + 140) - 2) > 1u )
            {
              v190 = m;
              v93 = sub_7764B0(a1, m, &v192);
              m = v190;
            }
            if ( v93 == (_DWORD)v177 - (_DWORD)v91 )
            {
              v189 = m;
              v133 = sub_8D46C0(v8[8].m128i_i64[0]);
              for ( m = v189; *(_BYTE *)(v133 + 140) == 12; v133 = *(_QWORD *)(v133 + 160) )
                ;
              if ( v189 == v133 )
              {
                v8[12].m128i_i64[0] = *(_QWORD *)(v189 + 128);
                v134 = sub_724980();
                v134[8] |= 1u;
                *((_QWORD *)v134 + 2) = 1;
                v8[12].m128i_i64[1] = (__int64)v134;
                goto LABEL_102;
              }
            }
          }
          v160 = v8;
          v94 = 0;
          v95 = v91;
          v187 = 0;
          v96 = 0;
          do
          {
            if ( (unsigned __int8)(*(_BYTE *)(m + 140) - 9) > 2u )
            {
              v168 = v177 - v95;
              if ( (_DWORD)v177 != (_DWORD)v95 )
              {
                v154 = m;
                v192 = 1;
                v116 = sub_724980();
                m = v154;
                v117 = v116;
                if ( v187 )
                  *(_QWORD *)v96 = v116;
                else
                  v187 = v116;
                v116[8] |= 1u;
                v118 = *(_BYTE *)(v154 + 140);
                while ( v118 == 8 )
                {
                  do
                  {
                    m = *(_QWORD *)(m + 160);
                    v118 = *(_BYTE *)(m + 140);
                  }
                  while ( v118 == 12 );
                }
                v131 = 16;
                if ( (unsigned __int8)(v118 - 2) > 1u )
                {
                  v153 = v117;
                  v155 = m;
                  v132 = sub_7764B0(a1, m, &v192);
                  v117 = v153;
                  m = v155;
                  v131 = v132;
                }
                *((_QWORD *)v117 + 2) = v168 / v131;
                v96 = v117;
                v95 += v168 / v131 * v131;
                v94 += *(_QWORD *)(m + 128) * (v168 / v131);
              }
            }
            else
            {
              sub_777100(a1, v6, v95, m, &v194, (unsigned __int64 *)v195);
              if ( !v194 && v96 && (v97 = v96[8], (v97 & 2) != 0) )
              {
                v98 = v195[0];
                v99 = *((_QWORD *)v96 + 2);
                m = *(_QWORD *)(v195[0] + 40);
                v94 += *(_QWORD *)(v195[0] + 104);
                v96[8] = v97 | 2;
                v100 = **(__int64 ****)(*(_QWORD *)(v99 + 56) + 168LL);
                if ( (*(_BYTE *)(v98 + 96) & 2) != 0 )
                {
                  for ( ; v100; v100 = (__int64 **)*v100 )
                  {
                    if ( v100[5] == *(__int64 **)(v98 + 40) && ((_BYTE)v100[12] & 2) != 0 )
                      break;
                  }
                }
                else
                {
                  for ( ; v100; v100 = (__int64 **)*v100 )
                  {
                    if ( v100[5] == *(__int64 **)(v98 + 40)
                      && ((_BYTE)v100[12] & 2) == 0
                      && v99 == *(_QWORD *)(*(_QWORD *)(v100[14][2] + 8) + 16LL) )
                    {
                      break;
                    }
                  }
                }
                *((_QWORD *)v96 + 2) = v100;
              }
              else
              {
                v114 = sub_724980();
                v115 = v114;
                if ( v187 )
                  *(_QWORD *)v96 = v114;
                else
                  v187 = v114;
                v98 = v194;
                if ( v194 )
                {
                  m = *(_QWORD *)(v194 + 120);
                  for ( v94 += *(_QWORD *)(v194 + 128); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                    ;
                }
                else
                {
                  v98 = v195[0];
                  m = *(_QWORD *)(v195[0] + 40);
                  v94 += *(_QWORD *)(v195[0] + 104);
                  v115[8] |= 2u;
                }
                *((_QWORD *)v115 + 2) = v98;
                v96 = v115;
              }
              v101 = qword_4F08388 & (v98 >> 3);
              v102 = qword_4F08380 + 16LL * v101;
              v103 = *(_QWORD *)v102;
              if ( *(_QWORD *)v102 == v98 )
              {
LABEL_198:
                v95 += *(unsigned int *)(v102 + 8);
              }
              else
              {
                while ( v103 )
                {
                  v101 = qword_4F08388 & (v101 + 1);
                  v102 = qword_4F08380 + 16LL * v101;
                  v103 = *(_QWORD *)v102;
                  if ( v98 == *(_QWORD *)v102 )
                    goto LABEL_198;
                }
              }
            }
          }
          while ( v177 != v95 );
          v8 = v160;
          v160[12].m128i_i64[0] = v94;
          v160[12].m128i_i64[1] = (__int64)v187;
LABEL_101:
          v8[10].m128i_i8[8] |= 8u;
LABEL_102:
          v62 = *(_BYTE *)(v6 + 8);
          goto LABEL_115;
        }
        if ( v193 != (const __m128i *)qword_4F08070 )
        {
          if ( *(char *)(a1 + 132) < 0 )
          {
            v135 = *(_DWORD *)(a1 + 8);
            for ( n = v135 & ((unsigned __int64)&v193[11].m128i_u64[1] >> 3); ; n = v135 & (n + 1) )
            {
              v137 = *(_QWORD *)a1 + 16LL * n;
              if ( &v193[11].m128i_u64[1] == (unsigned __int64 *)*(const __m128i **)v137 )
                break;
              if ( !*(_QWORD *)v137 )
                goto LABEL_281;
            }
            if ( *(_QWORD *)(v137 + 8) )
            {
              v193 = sub_740630(v193);
              v193[10].m128i_i8[11] |= 2u;
              v119 = v193;
            }
          }
          goto LABEL_281;
        }
        v138 = *(_QWORD *)(v6 + 24);
        v139 = qword_4F08388 & (v138 >> 3);
        v140 = (__int64 *)(qword_4F08380 + 16LL * v139);
        v141 = *v140;
        if ( *v140 == v138 )
        {
LABEL_312:
          v143 = (const __m128i *)v140[1];
          if ( v143 )
          {
LABEL_313:
            v193 = v143;
            v119 = v143;
            goto LABEL_281;
          }
        }
        else
        {
          while ( v141 )
          {
            v139 = qword_4F08388 & (v139 + 1);
            v140 = (__int64 *)(qword_4F08380 + 16LL * v139);
            v141 = *v140;
            if ( v138 == *v140 )
              goto LABEL_312;
          }
        }
        v142 = *(_DWORD *)(v6 + 8) >> 8;
        v170 = *(_DWORD *)(v6 + 8) >> 8;
        v178 = sub_724830(v142);
        v161 = *(_QWORD *)(v6 + 24);
        v143 = (const __m128i *)sub_724D80(2);
        v144 = sub_73CA60(v142);
        v143[11].m128i_i64[0] = v142;
        v143[8].m128i_i64[0] = (__int64)v144;
        v143[11].m128i_i64[1] = (__int64)v178;
        for ( ii = 0; v170 > (unsigned int)ii; ++ii )
        {
          sub_620E00((_WORD *)(v161 + 16 * ii), 0, v195, (int *)&v194);
          *((_BYTE *)v178 + ii) = v195[0];
        }
        v146 = *(_QWORD *)(v6 + 24);
        v147 = qword_4F08388;
        v148 = qword_4F08380;
        v149 = qword_4F08388 & (v146 >> 3);
        v150 = (__m128i *)(qword_4F08380 + 16LL * v149);
        if ( v150->m128i_i64[0] )
        {
          v151 = _mm_loadu_si128(v150);
          v150->m128i_i64[0] = v146;
          v150->m128i_i64[1] = (__int64)v143;
          do
          {
            v149 = v147 & (v149 + 1);
            v152 = (__m128i *)(v148 + 16LL * v149);
          }
          while ( v152->m128i_i64[0] );
          *v152 = v151;
        }
        else
        {
          v150->m128i_i64[0] = v146;
          v150->m128i_i64[1] = (__int64)v143;
        }
        ++HIDWORD(qword_4F08388);
        if ( v147 < 2 * HIDWORD(qword_4F08388) )
          sub_7704A0((__int64)&qword_4F08380);
        goto LABEL_313;
      }
      v63 = a2[1].m128i_i64[0];
      if ( *(_BYTE *)(v63 + 173) == 6 )
      {
        v64 = *(_BYTE *)(v63 + 176);
        if ( v64 != 1 )
        {
          if ( v64 )
          {
            if ( v64 != 6 )
              goto LABEL_99;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_6855B0(0xA8Du, (FILE *)(*(_QWORD *)(v63 + 184) + 64LL), (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
          }
          else
          {
            v65 = *(_QWORD *)(v63 + 184);
            if ( (*(_BYTE *)(v65 + 193) & 4) == 0 )
              goto LABEL_99;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_686E10(0xB76u, (FILE *)(a1 + 112), *(_QWORD *)v65, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
          }
          v191 = 0;
          goto LABEL_99;
        }
        v188 = *(_QWORD *)(v63 + 184);
        v21 = sub_6EA100((_BYTE *)v188);
        if ( !v21 )
        {
          v108 = *(_QWORD *)v188;
          v191 = 0;
          if ( v108 )
          {
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_686E10(0xA82u, (FILE *)(a1 + 112), v108, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
              return (_BOOL4)v191;
            }
          }
          else
          {
            sub_770DD0(0xA8Du, (FILE *)(v188 + 64), a1);
            return (_BOOL4)v191;
          }
          return v21;
        }
      }
LABEL_99:
      sub_740190(v63, v8, 0x800u);
      v8[8].m128i_i64[0] = a4;
      if ( (*(_BYTE *)(a4 + 168) & 1) != 0 )
        goto LABEL_101;
      v62 = a2->m128i_i8[8];
      if ( (v62 & 8) != 0 )
        goto LABEL_101;
LABEL_115:
      if ( (v62 & 4) == 0 )
        return (_BOOL4)v191;
      v68 = *(_QWORD *)(v6 + 16);
      v69 = 2;
      v70 = *(_QWORD **)v68;
      for ( jj = **(_QWORD ***)v68; jj; ++v69 )
      {
        v70 = jj;
        jj = (_QWORD *)*jj;
      }
      *v70 = qword_4F08088;
      v21 = v191;
      *(_BYTE *)(v6 + 8) &= ~4u;
      v72 = *(_QWORD *)(v68 + 24);
      qword_4F08080 += v69;
      qword_4F08088 = v68;
      *(_QWORD *)(v6 + 16) = v72;
      return v21;
    case 7:
      sub_72D3B0(a2[1].m128i_i64[0], (__int64)v8, 1);
      return (_BOOL4)v191;
    case 8:
      v27 = *(_QWORD *)(a4 + 160);
      for ( kk = *(_BYTE *)(v27 + 140); kk == 12; kk = *(_BYTE *)(v27 + 140) )
        v27 = *(_QWORD *)(v27 + 160);
      v173 = 16;
      if ( (unsigned __int8)(kk - 2) > 1u )
      {
        v184 = v27;
        v73 = sub_7764B0(a1, v27, &v191);
        v27 = v184;
        v173 = v73;
      }
      v21 = v191;
      if ( !v191 )
        return v21;
      v29 = *(_QWORD *)(a4 + 176);
      v181 = v27;
      v24 = 0;
      v165 = v29;
      sub_724A80((__int64)v8, 10);
      if ( !v165 )
        return (_BOOL4)v191;
      v158 = (__int64)v8;
      v30 = v181;
      while ( ((unsigned __int8)(1 << ((v6 - a3) & 7)) & *(_BYTE *)(a3 + -(((unsigned int)(v6 - a3) >> 3) + 10))) != 0
           || (unsigned __int8)(*(_BYTE *)(v30 + 140) - 8) <= 2u )
      {
        v182 = sub_724D50(0);
        if ( !(unsigned int)sub_77D750(a1, v6, a3, v30, v182) )
          return 0;
        ++v24;
        sub_72A690((__int64)v182, v158, 0, 0);
        v6 += v173;
        if ( v165 == v24 )
          return (_BOOL4)v191;
      }
      goto LABEL_32;
    case 9:
    case 0xA:
      if ( (*(_BYTE *)(a4 + 177) & 0x20) != 0 )
      {
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          sub_686CA0(0xAA6u, a1 + 112, a4, (_QWORD *)(a1 + 96));
          sub_770D30(a1);
        }
        return 0;
      }
      v162 = *(_QWORD *)(a4 + 160);
      sub_724A80((__int64)v8, 10);
      if ( !**(_QWORD **)(a4 + 168) )
        goto LABEL_127;
      v156 = a4;
      v11 = **(_QWORD **)(a4 + 168);
      do
      {
        if ( (*(_BYTE *)(v11 + 96) & 3) == 1 )
        {
          v12 = qword_4F08388 & (v11 >> 3);
          v13 = qword_4F08380 + 16LL * v12;
          v14 = *(_QWORD *)v13;
          if ( v11 == *(_QWORD *)v13 )
          {
LABEL_125:
            v15 = (__m128i *)((char *)a2 + *(unsigned int *)(v13 + 8));
          }
          else
          {
            while ( v14 )
            {
              v12 = qword_4F08388 & (v12 + 1);
              v13 = qword_4F08380 + 16LL * v12;
              v14 = *(_QWORD *)v13;
              if ( *(_QWORD *)v13 == v11 )
                goto LABEL_125;
            }
            v15 = a2;
          }
          v16 = sub_724D50(0);
          if ( !(unsigned int)sub_77D750(a1, v15, a3, *(_QWORD *)(v11 + 40), v16) )
            return 0;
          *(_WORD *)(v16 + 171) |= 0x180u;
          sub_72A690((__int64)v16, (__int64)v8, v11, 0);
        }
        v11 = *(_QWORD *)v11;
      }
      while ( v11 );
      v6 = (__int64)a2;
      a4 = v156;
LABEL_127:
      v21 = v191;
      if ( v191 )
      {
        v159 = 0;
        if ( (*(_BYTE *)(*(_QWORD *)(a4 + 168) + 110LL) & 1) != 0 && (*(_BYTE *)(a1 + 133) & 5) != 0 )
          v159 = (unsigned __int8)~*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 11) >> 7;
        v75 = sub_76FF70(v162);
        if ( v75 )
        {
          v175 = v6;
          v167 = (__int64)v8;
          while ( 1 )
          {
            if ( (*(_BYTE *)(v75 + 144) & 0x50) != 0x40 )
            {
              for ( mm = qword_4F08388 & (v75 >> 3); ; mm = qword_4F08388 & (mm + 1) )
              {
                v77 = qword_4F08380 + 16LL * mm;
                if ( *(_QWORD *)v77 == v75 )
                {
                  v78 = v175 + *(unsigned int *)(v77 + 8);
                  goto LABEL_139;
                }
                if ( !*(_QWORD *)v77 )
                  break;
              }
              v78 = v175;
LABEL_139:
              v79 = *(_QWORD *)(v75 + 120);
              for ( nn = *(_BYTE *)(v79 + 140); nn == 12; nn = *(_BYTE *)(v79 + 140) )
                v79 = *(_QWORD *)(v79 + 160);
              if ( ((unsigned __int8)(1 << ((v78 - a3) & 7)) & *(_BYTE *)(a3 + -(((unsigned int)(v78 - a3) >> 3) + 10))) != 0 )
              {
                if ( nn == 6 && v159 )
                {
                  *(_BYTE *)(v78 + 8) |= 0x80u;
                  *(_BYTE *)(a1 + 133) |= 1u;
                }
              }
              else if ( (unsigned __int8)(nn - 8) > 2u && (nn != 11 || (*(_BYTE *)(v79 + 179) & 1) == 0) )
              {
                if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
                {
                  sub_686E10(0xAC6u, (FILE *)(a1 + 112), *(_QWORD *)v75, (_QWORD *)(a1 + 96));
                  sub_770D30(a1);
                }
                return 0;
              }
              v81 = sub_724D50(0);
              if ( !(unsigned int)sub_77D750(a1, v78, a3, v79, v81) )
                return 0;
              sub_72A690((__int64)v81, v167, 0, v75);
            }
            v75 = sub_76FF70(*(_QWORD *)(v75 + 112));
            if ( !v75 )
              return (_BOOL4)v191;
          }
        }
      }
      return v21;
    case 0xB:
      sub_724A80((__int64)v8, 10);
      v183 = sub_76FF70(*(_QWORD *)(a4 + 160));
      if ( !v183 )
        return (_BOOL4)v191;
      v31 = a2->m128i_i64[0];
      if ( !a2->m128i_i64[0] )
      {
        sub_7790A0(a1, a2, a4, a3);
        v31 = a2->m128i_i64[0];
      }
      v174 = v31;
      v32 = sub_724D50(0);
      v33 = qword_4F08388 & (v174 >> 3);
      while ( 2 )
      {
        v34 = qword_4F08380 + 16LL * v33;
        if ( v174 == *(_QWORD *)v34 )
        {
          v6 = (__int64)a2->m128i_i64 + *(unsigned int *)(v34 + 8);
        }
        else if ( *(_QWORD *)v34 )
        {
          v33 = qword_4F08388 & (v33 + 1);
          continue;
        }
        break;
      }
      if ( !(unsigned int)sub_77D750(a1, v6, a3, *(_QWORD *)(v174 + 120), v32) )
        return 0;
      if ( v174 != v183 )
      {
        v35 = sub_724D50(13);
        v36 = sub_72CBE0();
        v35[176] |= 1u;
        *((_QWORD *)v35 + 16) = v36;
        *((_QWORD *)v35 + 23) = v174;
        sub_72A690((__int64)v35, (__int64)v8, 0, 0);
      }
      sub_72A690((__int64)v32, (__int64)v8, 0, 0);
      return (_BOOL4)v191;
    case 0xD:
      sub_724A80((__int64)v8, 7);
      v37 = a2->m128i_i64[1];
      if ( (a2->m128i_i8[0] & 1) != 0 )
      {
        if ( v37 && (*(_BYTE *)(v37 + 193) & 4) != 0 )
        {
          if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
          {
            sub_686E10(0xB76u, (FILE *)(a1 + 112), *(_QWORD *)v37, (_QWORD *)(a1 + 96));
            sub_770D30(a1);
          }
          v191 = 0;
          v21 = 0;
        }
        else
        {
          v8[12].m128i_i8[0] |= 2u;
          v8[12].m128i_i64[1] = v37;
          v21 = v191;
        }
      }
      else
      {
        v8[12].m128i_i64[1] = v37;
        v21 = v191;
      }
      if ( !a2->m128i_i32[1] )
        return v21;
      v38 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a2->m128i_i64[1] + 40) + 32LL) + 168LL);
      v8[12].m128i_i8[0] = ((a2->m128i_i8[0] & 2) != 0) | v8[12].m128i_i8[0] & 0xFE;
      if ( !v38 )
        return v21;
      v39 = qword_4F08388;
      v40 = qword_4F08380;
      v41 = a2->m128i_i32[1];
      if ( (*(_BYTE *)(v38 + 96) & 3) != 0 )
        goto LABEL_70;
LABEL_78:
      v46 = *(_QWORD *)(v38 + 112);
      v47 = *(_QWORD *)(v46 + 8);
      v48 = *(_QWORD **)(v46 + 16);
      v49 = *(_QWORD *)(v47 + 16);
      v50 = *(_QWORD *)(v49 + 40);
      v51 = v49 >> 3;
      while ( 2 )
      {
        v52 = v39 & v51;
        v53 = v40 + 16LL * v52;
        if ( v49 == *(_QWORD *)v53 )
        {
          v45 = *(_DWORD *)(v53 + 8);
        }
        else
        {
          if ( *(_QWORD *)v53 )
          {
            LODWORD(v51) = v52 + 1;
            continue;
          }
          v45 = 0;
        }
        break;
      }
      v54 = *(_QWORD **)v47;
      v55 = (_QWORD *)*v48;
      if ( *(_QWORD *)v47 == *v48 )
        goto LABEL_75;
      do
      {
        v56 = *(_QWORD **)(v50 + 168);
        v50 = *(_QWORD *)(v54[2] + 40LL);
        v57 = (unsigned __int64)sub_771030(v56, v50);
        for ( i1 = v57 >> 3; ; LODWORD(i1) = v59 + 1 )
        {
          v59 = v39 & i1;
          v60 = v40 + 16LL * v59;
          if ( v57 == *(_QWORD *)v60 )
            break;
          if ( !*(_QWORD *)v60 )
            goto LABEL_89;
        }
        v45 += *(_DWORD *)(v60 + 8);
LABEL_89:
        v54 = (_QWORD *)*v54;
      }
      while ( v54 != v55 );
LABEL_75:
      while ( v45 != v41 )
      {
        v38 = *(_QWORD *)v38;
        if ( !v38 )
          return v21;
        if ( (*(_BYTE *)(v38 + 96) & 3) == 0 )
          goto LABEL_78;
LABEL_70:
        for ( i2 = v38 >> 3; ; LODWORD(i2) = v43 + 1 )
        {
          v43 = v39 & i2;
          v44 = v40 + 16LL * v43;
          if ( *(_QWORD *)v44 == v38 )
          {
            v45 = *(_DWORD *)(v44 + 8);
            goto LABEL_75;
          }
          if ( !*(_QWORD *)v44 )
            break;
        }
        v45 = 0;
      }
      v8[10].m128i_i8[8] |= 8u;
      v8[11].m128i_i64[0] = v38;
      return v21;
    case 0xF:
      v17 = *(_QWORD *)(a4 + 160);
      for ( i3 = *(_BYTE *)(v17 + 140); i3 == 12; i3 = *(_BYTE *)(v17 + 140) )
        v17 = *(_QWORD *)(v17 + 160);
      v19 = *(_QWORD *)(a4 + 128);
      v172 = 16;
      v20 = *(_QWORD *)(v17 + 128);
      if ( (unsigned __int8)(i3 - 2) > 1u )
      {
        v166 = v19;
        v185 = v17;
        v74 = sub_7764B0(a1, v17, &v191);
        v19 = v166;
        v17 = v185;
        v172 = v74;
      }
      v21 = v191;
      if ( !v191 )
        return v21;
      v163 = v17;
      v179 = v19;
      sub_724A80((__int64)v8, 10);
      if ( v179 < v20 )
        return (_BOOL4)v191;
      v157 = (__int64)v8;
      v22 = v163;
      v23 = v179 / v20;
      v24 = 0;
      v164 = v23;
      while ( 2 )
      {
        if ( ((unsigned __int8)(1 << ((v6 - a3) & 7)) & *(_BYTE *)(a3 + -(((unsigned int)(v6 - a3) >> 3) + 10))) != 0
          || (unsigned __int8)(*(_BYTE *)(v22 + 140) - 8) <= 2u )
        {
          v180 = sub_724D50(0);
          if ( (unsigned int)sub_77D750(a1, v6, a3, v22, v180) )
          {
            ++v24;
            sub_72A690((__int64)v180, v157, 0, 0);
            v6 += v172;
            if ( v164 <= v24 )
              return (_BOOL4)v191;
            continue;
          }
        }
        else
        {
LABEL_32:
          if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
          {
            sub_67E440(0xCF5u, (_DWORD *)(a1 + 112), v24, (_QWORD *)(a1 + 96));
            sub_770D30(a1);
          }
        }
        break;
      }
      return 0;
    case 0x13:
      sub_724A80((__int64)v8, 1);
      si128 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
      v21 = v191;
      v8[10].m128i_i8[8] |= 8u;
      v8[11] = si128;
      return v21;
    case 0x14:
      sub_724A80((__int64)v8, 15);
      v21 = v191;
      v8[11] = _mm_loadu_si128(a2);
      v8[12].m128i_i64[0] = a2[1].m128i_i64[0];
      return v21;
    default:
      goto LABEL_8;
  }
}
