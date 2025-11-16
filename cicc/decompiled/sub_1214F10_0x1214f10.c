// Function: sub_1214F10
// Address: 0x1214f10
//
__int64 __fastcall sub_1214F10(_QWORD *a1, __int64 a2)
{
  _QWORD **v2; // r15
  _QWORD *j; // r13
  unsigned int *v4; // r12
  unsigned int *k; // r14
  _QWORD *v6; // rax
  unsigned int v7; // esi
  _QWORD **v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  char v11; // al
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  size_t v16; // rdi
  __int64 v17; // r13
  _QWORD *v18; // r12
  __int64 v20; // r14
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // r14
  _QWORD *n; // r13
  __int64 v26; // r13
  _QWORD *ii; // r12
  __int64 v28; // rdi
  unsigned __int64 *v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 *v32; // rbx
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // rdx
  _QWORD *v37; // rdi
  _QWORD *v38; // rbx
  _QWORD *v39; // r13
  _QWORD *v40; // rdi
  _QWORD *v41; // rbx
  char v42; // al
  _QWORD *v43; // rbx
  _QWORD *v44; // rdx
  _QWORD *v45; // rax
  _QWORD *v46; // rbx
  _QWORD *v47; // rdx
  int v48; // edx
  size_t **v49; // rax
  size_t *v50; // rcx
  size_t **v51; // rbx
  size_t v52; // r15
  _QWORD *v53; // r14
  const char *v54; // r12
  size_t v55; // rax
  int v56; // eax
  size_t *v57; // rdx
  size_t *v58; // rax
  __int64 nn; // r12
  int v60; // edx
  _QWORD *v61; // rax
  __int64 v62; // rdi
  __int64 v63; // rdi
  __int32 v64; // eax
  __int64 v65; // rdx
  unsigned __int64 *v66; // r12
  _QWORD *v67; // r13
  unsigned int v68; // r12d
  _QWORD *v69; // r14
  __int64 v70; // rdi
  int v71; // ecx
  __int64 v72; // r12
  _QWORD *v73; // r9
  __int64 v74; // rax
  __int64 v75; // r13
  _QWORD *v76; // rdi
  _QWORD *v77; // rbx
  const char *v78; // r14
  __int64 v79; // rax
  _BYTE *v80; // rdi
  __int64 v81; // rax
  __int64 i1; // rsi
  unsigned __int8 *v83; // rdx
  int v84; // ecx
  unsigned __int64 v85; // rcx
  __int64 v86; // rdi
  __int64 v87; // rdi
  unsigned __int64 *v88; // rax
  unsigned __int64 *v89; // rcx
  __int64 *v90; // rdx
  unsigned __int64 v91; // rax
  unsigned int v92; // r12d
  __int64 v93; // r13
  unsigned __int64 v94; // rax
  __int64 v95; // rcx
  __int64 v96; // rsi
  __int64 v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rdx
  __int64 v100; // rbx
  unsigned __int8 *v101; // rdx
  int v102; // eax
  __int64 v103; // rcx
  _QWORD *v104; // rax
  _QWORD *v105; // rax
  const char *v106; // rbx
  __int32 v107; // edx
  __int64 v108; // rax
  _QWORD *v109; // rcx
  unsigned __int64 v110; // rsi
  __int64 v111; // rcx
  __m128i *v112; // rax
  _QWORD *v113; // rax
  __int64 v114; // rcx
  __m128i *v115; // rax
  _QWORD *v116; // rax
  _QWORD *mm; // r14
  __int64 v118; // rdi
  _QWORD *kk; // r14
  __int64 v120; // rdi
  _QWORD *v121; // [rsp+8h] [rbp-178h]
  char v122; // [rsp+10h] [rbp-170h]
  _QWORD *v123; // [rsp+10h] [rbp-170h]
  unsigned int v124; // [rsp+18h] [rbp-168h]
  size_t **v125; // [rsp+20h] [rbp-160h]
  size_t v126; // [rsp+28h] [rbp-158h]
  _QWORD *v127; // [rsp+28h] [rbp-158h]
  _QWORD **v128; // [rsp+30h] [rbp-150h]
  __int64 v129; // [rsp+38h] [rbp-148h]
  __int64 *jj; // [rsp+38h] [rbp-148h]
  _QWORD **v131; // [rsp+38h] [rbp-148h]
  char i; // [rsp+38h] [rbp-148h]
  __int64 v133; // [rsp+38h] [rbp-148h]
  _QWORD *v134; // [rsp+38h] [rbp-148h]
  _QWORD **m; // [rsp+48h] [rbp-138h] BYREF
  __int64 v136[2]; // [rsp+50h] [rbp-130h] BYREF
  __int64 v137; // [rsp+60h] [rbp-120h] BYREF
  __int64 *v138; // [rsp+70h] [rbp-110h] BYREF
  __int64 v139; // [rsp+78h] [rbp-108h]
  __int64 v140; // [rsp+80h] [rbp-100h] BYREF
  __m128i *v141; // [rsp+90h] [rbp-F0h] BYREF
  unsigned __int64 *v142; // [rsp+98h] [rbp-E8h]
  __m128i v143; // [rsp+A0h] [rbp-E0h] BYREF
  __int16 v144; // [rsp+B0h] [rbp-D0h]
  const char *v145; // [rsp+F0h] [rbp-90h] BYREF
  size_t v146; // [rsp+F8h] [rbp-88h]
  char *v147; // [rsp+100h] [rbp-80h] BYREF
  char v148; // [rsp+108h] [rbp-78h] BYREF
  __int16 v149; // [rsp+110h] [rbp-70h]

  v122 = a2;
  if ( !a1[43] )
    return 0;
  v2 = (_QWORD **)a1;
  if ( LODWORD(qword_4F80E68[8]) == 1 )
  {
    v65 = *((unsigned __int8 *)a1 + 1745);
    v66 = (_QWORD *)((char *)a1 + 1745);
    LOBYTE(qword_4F80F48[8]) = *((_BYTE *)a1 + 1745);
    if ( !qword_4F80F48[14]
      || (a1 = &qword_4F80F48[12],
          a2 = (__int64)v66,
          ((void (__fastcall *)(_QWORD *, unsigned __int64 *))qword_4F80F48[15])(&qword_4F80F48[12], v66),
          v65 = *((unsigned __int8 *)v2 + 1745),
          unk_4F80E08 = v65,
          unk_4F81788 = v65,
          !unk_4F817B8) )
    {
      sub_4263D6(a1, a2, v65);
    }
    a2 = (__int64)v66;
    unk_4F817C0(&unk_4F817A8, v66);
    v67 = v2[43];
    v68 = *((unsigned __int8 *)v2 + 1745);
    v69 = (_QWORD *)v67[4];
    for ( i = *((_BYTE *)v2 + 1745); v67 + 3 != v69; v69 = (_QWORD *)v69[1] )
    {
      v70 = (__int64)(v69 - 7);
      a2 = v68;
      if ( !v69 )
        v70 = 0;
      sub_B2BA20(v70, v68);
    }
    *((_BYTE *)v67 + 872) = i;
  }
  for ( j = v2[182]; v2 + 180 != j; j = (_QWORD *)sub_220EEE0(j) )
  {
    v129 = j[4];
    v141 = (__m128i *)*v2;
    v142 = &v143.m128i_u64[1];
    v143.m128i_i64[0] = 0x800000000LL;
    v4 = (unsigned int *)j[6];
    for ( k = (unsigned int *)j[5]; v4 != k; ++k )
    {
      v6 = v2[187];
      if ( v6 )
      {
        v7 = *k;
        v8 = v2 + 186;
        do
        {
          while ( 1 )
          {
            v9 = v6[2];
            v10 = v6[3];
            if ( *((_DWORD *)v6 + 8) >= v7 )
              break;
            v6 = (_QWORD *)v6[3];
            if ( !v10 )
              goto LABEL_10;
          }
          v8 = (_QWORD **)v6;
          v6 = (_QWORD *)v6[2];
        }
        while ( v9 );
LABEL_10:
        if ( v2 + 186 != v8 && v7 >= *((_DWORD *)v8 + 8) )
          sub_A776F0((__int64)&v141, (__int64)(v8 + 5));
      }
    }
    v11 = *(_BYTE *)v129;
    if ( *(_BYTE *)v129 )
    {
      if ( v11 == 85 || v11 == 34 || v11 == 40 )
      {
        v20 = v129;
        v138 = *(__int64 **)(v129 + 72);
        v22 = sub_A74680(&v138);
        sub_A74940((__int64)&v145, *v2[43], v22);
        v138 = (__int64 *)sub_A786C0((__int64 *)&v138, *v2, -1);
        sub_A776F0((__int64)&v145, (__int64)&v141);
        a2 = (__int64)*v2;
        v21 = sub_A7B2C0((__int64 *)&v138, *v2, -1, (__int64)&v145);
      }
      else
      {
        if ( v11 != 3 )
          BUG();
        v20 = v129;
        sub_A74940((__int64)&v145, *v2[43], *(_QWORD *)(v129 + 72));
        sub_A776F0((__int64)&v145, (__int64)&v141);
        a2 = (__int64)&v145;
        v21 = sub_A7A280(*v2, (__int64)&v145);
      }
      *(_QWORD *)(v20 + 72) = v21;
      v16 = v146;
      if ( (char *)v146 == &v148 )
        goto LABEL_20;
    }
    else
    {
      v136[0] = *(_QWORD *)(v129 + 120);
      v12 = sub_A74680(v136);
      sub_A74940((__int64)&v145, *v2[43], v12);
      v136[0] = sub_A786C0(v136, *v2, -1);
      sub_A776F0((__int64)&v145, (__int64)&v141);
      v13 = (__int64 *)sub_A74DF0((__int64)&v145, 86);
      v139 = v14;
      v138 = v13;
      if ( (_BYTE)v14 && v138 )
      {
        _BitScanReverse64(&v15, (unsigned __int64)v138);
        sub_B2F770(v129, 63 - (v15 ^ 0x3F));
        sub_A77390((__int64)&v145, 86);
      }
      a2 = (__int64)*v2;
      *(_QWORD *)(v129 + 120) = sub_A7B2C0(v136, *v2, -1, (__int64)&v145);
      v16 = v146;
      if ( (char *)v146 == &v148 )
        goto LABEL_20;
    }
    _libc_free(v16, a2);
LABEL_20:
    if ( v142 != &v143.m128i_u64[1] )
      _libc_free(v142, a2);
  }
  if ( v2[165] )
  {
    v145 = "expected function name in blockaddress";
    v23 = v2[163];
    v149 = 259;
    a2 = v23[5];
    goto LABEL_38;
  }
  v17 = (__int64)v2[176];
  v18 = v2 + 174;
  for ( m = v2; (_QWORD *)v17 != v18; v17 = sub_220EEE0(v17) )
  {
    a2 = v17 + 32;
    if ( (unsigned __int8)sub_1212A30((__int64 *)&m, v17 + 32, *(_QWORD **)(v17 + 192)) )
      return 1;
  }
  v24 = (__int64)v2[170];
  for ( n = v2 + 168; n != (_QWORD *)v24; v24 = sub_220EEE0(v24) )
  {
    a2 = v24 + 32;
    if ( (unsigned __int8)sub_1212A30((__int64 *)&m, v24 + 32, *(_QWORD **)(v24 + 192)) )
      return 1;
  }
  sub_1209010((__int64)v2[175]);
  v2[176] = v18;
  v62 = (__int64)v2[169];
  v2[175] = 0;
  v2[177] = v18;
  v2[178] = 0;
  sub_1209010(v62);
  v63 = (__int64)v2[122];
  v2[169] = 0;
  v2[170] = n;
  v2[171] = n;
  v2[172] = 0;
  v128 = v2 + 120;
  if ( v2 + 120 != (_QWORD **)v63 )
  {
    while ( 1 )
    {
      a2 = *(_QWORD *)(v63 + 48);
      if ( a2 )
        break;
      v63 = sub_220EEE0(v63);
      if ( v128 == (_QWORD **)v63 )
        goto LABEL_100;
    }
    v64 = *(_DWORD *)(v63 + 32);
    v141 = (__m128i *)"use of undefined type '%";
    v143.m128i_i32[0] = v64;
    v145 = (const char *)&v141;
    v144 = 2307;
    v147 = "'";
    v149 = 770;
    goto LABEL_38;
  }
LABEL_100:
  v71 = *((_DWORD *)v2 + 234);
  if ( v71 )
  {
    v29 = v2[116];
    if ( *v29 != -8 && *v29 )
    {
      v88 = v2[116];
    }
    else
    {
      a2 = (__int64)(v29 + 1);
      do
      {
        do
        {
          v87 = *(_QWORD *)a2;
          v88 = (unsigned __int64 *)a2;
          a2 += 8;
        }
        while ( v87 == -8 );
      }
      while ( !v87 );
    }
    v89 = &v29[v71];
    while ( 1 )
    {
      if ( v88 == v89 )
        goto LABEL_101;
      v90 = (__int64 *)*v88;
      if ( *(_QWORD *)(*v88 + 16) )
        break;
      v29 = v88 + 1;
      v91 = v88[1];
      if ( v91 && v91 != -8 )
      {
        v88 = v29;
      }
      else
      {
        v88 = v29;
        do
        {
          do
          {
            v29 = (unsigned __int64 *)v88[1];
            ++v88;
          }
          while ( v29 == (unsigned __int64 *)-8LL );
        }
        while ( !v29 );
      }
    }
    v103 = *v90;
    v141 = (__m128i *)"use of undefined type named '";
    v143.m128i_i64[0] = (__int64)(v90 + 3);
    v144 = 1283;
    v143.m128i_i64[1] = v103;
    v145 = (const char *)&v141;
    v147 = "'";
    v149 = 770;
    a2 = *(_QWORD *)(*v88 + 16);
    goto LABEL_38;
  }
LABEL_101:
  if ( v2[159] )
  {
    sub_8FD6D0((__int64)v136, "use of undefined comdat '$", v2[157] + 4);
    if ( v136[1] != 0x3FFFFFFFFFFFFFFFLL )
    {
      v115 = (__m128i *)sub_2241490(v136, "'", 1, v114);
      v141 = &v143;
      if ( (__m128i *)v115->m128i_i64[0] == &v115[1] )
      {
        v143 = _mm_loadu_si128(v115 + 1);
      }
      else
      {
        v141 = (__m128i *)v115->m128i_i64[0];
        v143.m128i_i64[0] = v115[1].m128i_i64[0];
      }
      v142 = (unsigned __int64 *)v115->m128i_i64[1];
      v115->m128i_i64[0] = (__int64)v115[1].m128i_i64;
      v115->m128i_i64[1] = 0;
      v115[1].m128i_i8[0] = 0;
      v145 = (const char *)&v141;
      v116 = v2[157];
      v149 = 260;
      sub_11FD800((__int64)(v2 + 22), v116[8], (__int64)&v145, 1);
      if ( v141 != &v143 )
        j_j___libc_free_0(v141, v143.m128i_i64[0] + 1);
      if ( (__int64 *)v136[0] != &v137 )
        j_j___libc_free_0(v136[0], v137 + 1);
      return 1;
    }
    goto LABEL_186;
  }
  v30 = (__int64)(v2 + 137);
  v127 = v2[140];
  if ( v2 + 138 == v127 )
  {
LABEL_44:
    if ( !v2[142] )
    {
      if ( v2[148] )
      {
        v105 = v2[146];
        v106 = "use of undefined value '@";
        v107 = *((_DWORD *)v105 + 8);
      }
      else
      {
        if ( (_BYTE)qword_4F92428 )
        {
          if ( !v2[136] )
            goto LABEL_48;
          sub_1209C30(v2);
        }
        if ( !v2[136] )
        {
LABEL_48:
          v26 = (__int64)v2[128];
          for ( ii = v2 + 126; ii != (_QWORD *)v26; v26 = sub_220EEE0(v26) )
          {
            v28 = *(_QWORD *)(v26 + 40);
            if ( v28 && ((*(_BYTE *)(v28 + 1) & 0x7F) == 2 || *(_DWORD *)(v28 - 8)) )
              sub_B931A0(v28, a2, (__int64)v29, v30, v31);
          }
          v32 = v2[46];
          for ( jj = &v32[*((unsigned int *)v2 + 94)]; jj != v32; ++v32 )
          {
            v33 = *v32;
            if ( (*(_BYTE *)(*v32 + 7) & 0x20) != 0 )
            {
              v34 = sub_B91C10(*v32, 1);
              v35 = v34;
              if ( v34 )
              {
                v36 = sub_A849A0(v34);
                if ( v36 != v35 )
                  sub_B99FD0(v33, 1u, v36);
              }
            }
          }
          v37 = v2[43];
          v38 = (_QWORD *)v37[4];
          v39 = v37 + 3;
          if ( v37 + 3 != v38 )
          {
            do
            {
              v40 = v38;
              v38 = (_QWORD *)v38[1];
              sub_AA3950((__int64)(v40 - 7));
            }
            while ( v39 != v38 );
            v37 = v2[43];
          }
          if ( v122 )
          {
            sub_A84D20(v37);
            v37 = v2[43];
          }
          sub_A85B60(v37);
          sub_A84F90(v2[43]);
          sub_A86A60((__int64)v2[43]);
          if ( LODWORD(qword_4F80E68[8]) != 1 )
          {
            v41 = v2[43];
            v42 = *((_BYTE *)v41 + 872);
            if ( LOBYTE(qword_4F80F48[8]) )
            {
              if ( !v42 )
              {
                for ( kk = (_QWORD *)v41[4]; v41 + 3 != kk; kk = (_QWORD *)kk[1] )
                {
                  v120 = (__int64)(kk - 7);
                  if ( !kk )
                    v120 = 0;
                  sub_B2B950(v120);
                }
                *((_BYTE *)v41 + 872) = 1;
              }
            }
            else if ( v42 )
            {
              for ( mm = (_QWORD *)v41[4]; v41 + 3 != mm; mm = (_QWORD *)mm[1] )
              {
                v118 = (__int64)(mm - 7);
                if ( !mm )
                  v118 = 0;
                sub_B2B9A0(v118);
              }
              *((_BYTE *)v41 + 872) = 0;
            }
          }
          v43 = v2[45];
          if ( v43 )
          {
            sub_C7D6A0(v43[1], 16LL * *((unsigned int *)v43 + 6), 8);
            v43[2] = 0;
            v43[1] = 0;
            *((_DWORD *)v43 + 6) = 0;
            ++*v43;
            v44 = v2[150];
            v2[149] = (_QWORD *)((char *)v2[149] + 1);
            v45 = (_QWORD *)v43[1];
            v43[1] = v44;
            LODWORD(v44) = *((_DWORD *)v2 + 302);
            v2[150] = v45;
            LODWORD(v45) = *((_DWORD *)v43 + 4);
            *((_DWORD *)v43 + 4) = (_DWORD)v44;
            LODWORD(v44) = *((_DWORD *)v2 + 303);
            *((_DWORD *)v2 + 302) = (_DWORD)v45;
            LODWORD(v45) = *((_DWORD *)v43 + 5);
            *((_DWORD *)v43 + 5) = (_DWORD)v44;
            LODWORD(v44) = *((_DWORD *)v2 + 304);
            *((_DWORD *)v2 + 303) = (_DWORD)v45;
            LODWORD(v45) = *((_DWORD *)v43 + 6);
            *((_DWORD *)v43 + 6) = (_DWORD)v44;
            *((_DWORD *)v2 + 304) = (_DWORD)v45;
            *((_DWORD *)v43 + 8) = *((_DWORD *)v2 + 306);
            v46 = v2[45];
            sub_1206DC0((_QWORD *)v46[7]);
            v46[7] = 0;
            v46[8] = v46 + 6;
            v46[9] = v46 + 6;
            v46[10] = 0;
            if ( v2[127] )
            {
              *((_DWORD *)v46 + 12) = *((_DWORD *)v2 + 252);
              v47 = v2[127];
              v46[7] = v47;
              v46[8] = v2[128];
              v46[9] = v2[129];
              v47[1] = v46 + 6;
              v46[10] = v2[130];
              v2[127] = 0;
              v2[128] = ii;
              v2[129] = ii;
              v2[130] = 0;
            }
            v48 = *((_DWORD *)v2 + 234);
            if ( v48 )
            {
              v49 = (size_t **)v2[116];
              v50 = *v49;
              v51 = v49;
              if ( *v49 != (size_t *)-8LL )
                goto LABEL_74;
              do
              {
                do
                {
                  v50 = v51[1];
                  ++v51;
                }
                while ( v50 == (size_t *)-8LL );
LABEL_74:
                ;
              }
              while ( !v50 );
              v125 = &v49[v48];
              if ( v51 != v125 )
              {
                v131 = v2;
                while ( 1 )
                {
                  v52 = **v51;
                  v53 = v131[45];
                  v54 = (const char *)(*v51 + 3);
                  v55 = (*v51)[1];
                  v145 = v54;
                  v146 = v52;
                  v126 = v55;
                  v56 = sub_C92610();
                  v124 = sub_C92740((__int64)(v53 + 11), v145, v146, v56);
                  v123 = (_QWORD *)(v53[11] + 8LL * v124);
                  if ( !*v123 )
                    goto LABEL_171;
                  if ( *v123 == -8 )
                    break;
LABEL_79:
                  v57 = v51[1];
                  ++v51;
                  if ( v57 == (size_t *)-8LL || !v57 )
                  {
                    do
                    {
                      do
                      {
                        v58 = v51[1];
                        ++v51;
                      }
                      while ( v58 == (size_t *)-8LL );
                    }
                    while ( !v58 );
                  }
                  if ( v51 == v125 )
                  {
                    v2 = v131;
                    goto LABEL_85;
                  }
                }
                --*((_DWORD *)v53 + 26);
LABEL_171:
                v108 = sub_C7D670(v52 + 17, 8);
                v109 = (_QWORD *)v108;
                if ( v52 )
                {
                  v121 = (_QWORD *)v108;
                  memcpy((void *)(v108 + 16), v54, v52);
                  v109 = v121;
                }
                *((_BYTE *)v109 + v52 + 16) = 0;
                *v109 = v52;
                v109[1] = v126;
                *v123 = v109;
                ++*((_DWORD *)v53 + 25);
                sub_C929D0(v53 + 11, v124);
                goto LABEL_79;
              }
            }
LABEL_85:
            for ( nn = (__int64)v2[122]; v128 != (_QWORD **)nn; nn = sub_220EEE0(nn) )
            {
              v60 = *(_DWORD *)(nn + 32);
              v146 = *(_QWORD *)(nn + 40);
              v61 = v2[45];
              LODWORD(v145) = v60;
              sub_1212DD0(v61 + 14, (unsigned int *)&v145);
            }
          }
          return 0;
        }
        v105 = v2[134];
        v106 = "use of undefined metadata '!";
        v107 = *((_DWORD *)v105 + 8);
      }
      v143.m128i_i32[0] = v107;
      v141 = (__m128i *)v106;
      v144 = 2307;
      v145 = (const char *)&v141;
      v147 = "'";
      v149 = 770;
      a2 = v105[6];
LABEL_38:
      sub_11FD800((__int64)(v2 + 22), a2, (__int64)&v145, 1);
      return 1;
    }
    sub_8FD6D0((__int64)&v138, "use of undefined value '@", v2[140] + 4);
    if ( v139 != 0x3FFFFFFFFFFFFFFFLL )
    {
      v112 = (__m128i *)sub_2241490(&v138, "'", 1, v111);
      v141 = &v143;
      if ( (__m128i *)v112->m128i_i64[0] == &v112[1] )
      {
        v143 = _mm_loadu_si128(v112 + 1);
      }
      else
      {
        v141 = (__m128i *)v112->m128i_i64[0];
        v143.m128i_i64[0] = v112[1].m128i_i64[0];
      }
      v142 = (unsigned __int64 *)v112->m128i_i64[1];
      v112->m128i_i64[0] = (__int64)v112[1].m128i_i64;
      v112->m128i_i64[1] = 0;
      v112[1].m128i_i8[0] = 0;
      v145 = (const char *)&v141;
      v113 = v2[140];
      v149 = 260;
      sub_11FD800((__int64)(v2 + 22), v113[9], (__int64)&v145, 1);
      if ( v141 != &v143 )
        j_j___libc_free_0(v141, v143.m128i_i64[0] + 1);
      if ( v138 != &v140 )
        j_j___libc_free_0(v138, v140 + 1);
      return 1;
    }
LABEL_186:
    sub_4262D8((__int64)"basic_string::append");
  }
  while ( 1 )
  {
    v77 = v127;
    v78 = (const char *)(v127 + 4);
    v79 = sub_220EEE0(v127);
    a2 = v127[5];
    v80 = (_BYTE *)v127[4];
    v127 = (_QWORD *)v79;
    if ( (unsigned __int64)a2 <= 4 || *(_DWORD *)v80 != 1836477548 || v80[4] != 46 )
    {
      if ( (_BYTE)qword_4F92428 )
      {
        v81 = *(_QWORD *)(v77[8] + 16LL);
        if ( v81 )
        {
          for ( i1 = 0; ; i1 = *((_QWORD *)v83 + 10) )
          {
            v83 = *(unsigned __int8 **)(v81 + 24);
            v84 = *v83;
            if ( (unsigned __int8)v84 <= 0x1Cu )
              break;
            v85 = (unsigned int)(v84 - 34);
            if ( (unsigned __int8)v85 > 0x33u )
              break;
            v86 = 0x8000000000041LL;
            if ( !_bittest64(&v86, v85) )
              break;
            if ( (unsigned __int8 *)v81 != v83 - 32 )
              break;
            v72 = *((_QWORD *)v83 + 10);
            if ( i1 )
            {
              if ( i1 != v72 )
                break;
            }
            v81 = *(_QWORD *)(v81 + 8);
            if ( !v81 )
            {
              if ( v72 )
                goto LABEL_105;
              break;
            }
          }
        }
        v72 = sub_BCB2B0(*v2);
LABEL_105:
        if ( *(_BYTE *)(v72 + 8) == 13 )
        {
          v73 = v2[43];
          v145 = v78;
          v149 = 260;
          v133 = (__int64)v73;
          v74 = sub_BD2DA0(136);
          v75 = v74;
          if ( v74 )
            sub_B2C3B0(v74, v72, 0, 0xFFFFFFFF, (__int64)&v145, v133);
        }
        else
        {
          v145 = v78;
          v149 = 260;
          BYTE4(v141) = 0;
          v104 = sub_BD2C40(88, unk_3F0FAE8);
          v75 = (__int64)v104;
          if ( v104 )
            sub_B30000((__int64)v104, (__int64)v2[43], (_QWORD *)v72, 0, 0, 0, (__int64)&v145, 0, 0, (__int64)v141, 0);
        }
        sub_BD84D0(v77[8], v75);
        v76 = (_QWORD *)v77[8];
LABEL_109:
        sub_B30810(v76);
        a2 = (__int64)v78;
        sub_1214E50((__int64)(v2 + 137), (__int64)v78);
      }
      goto LABEL_110;
    }
    v92 = sub_B60C50(v80, a2);
    if ( v92 )
      break;
LABEL_110:
    if ( v2 + 138 == v127 )
      goto LABEL_44;
  }
  v76 = (_QWORD *)v77[8];
  v93 = v76[2];
  if ( !v93 )
    goto LABEL_109;
  v134 = v77;
  while ( 1 )
  {
    v100 = v93;
    v93 = *(_QWORD *)(v93 + 8);
    v101 = *(unsigned __int8 **)(v100 + 24);
    v102 = *v101;
    if ( (unsigned __int8)v102 <= 0x1Cu
      || (v94 = (unsigned int)(v102 - 34), (unsigned __int8)v94 > 0x33u)
      || (v95 = 0x8000000000041LL, !_bittest64(&v95, v94))
      || (unsigned __int8 *)v100 != v101 - 32 )
    {
      v149 = 259;
      v145 = "intrinsic can only be used as callee";
      sub_11FD800((__int64)(v2 + 22), v134[9], (__int64)&v145, 1);
      return 1;
    }
    v145 = (const char *)&v147;
    v146 = 0x600000000LL;
    if ( !(unsigned __int8)sub_B6E220(v92, *((_QWORD *)v101 + 10), (__int64)&v145) )
      break;
    v96 = v92;
    v97 = sub_B6E160(v2[43], v92, (__int64)v145, (unsigned int)v146);
    if ( *(_QWORD *)v100 )
    {
      v98 = *(_QWORD *)(v100 + 8);
      **(_QWORD **)(v100 + 16) = v98;
      if ( v98 )
        *(_QWORD *)(v98 + 16) = *(_QWORD *)(v100 + 16);
    }
    *(_QWORD *)v100 = v97;
    if ( v97 )
    {
      v99 = *(_QWORD *)(v97 + 16);
      *(_QWORD *)(v100 + 8) = v99;
      if ( v99 )
      {
        v96 = v100 + 8;
        *(_QWORD *)(v99 + 16) = v100 + 8;
      }
      *(_QWORD *)(v100 + 16) = v97 + 16;
      *(_QWORD *)(v97 + 16) = v100;
    }
    if ( v145 != (const char *)&v147 )
      _libc_free(v145, v96);
    if ( !v93 )
    {
      v76 = (_QWORD *)v134[8];
      goto LABEL_109;
    }
  }
  v141 = (__m128i *)"invalid intrinsic signature";
  v144 = 259;
  v110 = v134[9];
  sub_11FD800((__int64)(v2 + 22), v110, (__int64)&v141, 1);
  if ( v145 != (const char *)&v147 )
    _libc_free(v145, v110);
  return 1;
}
