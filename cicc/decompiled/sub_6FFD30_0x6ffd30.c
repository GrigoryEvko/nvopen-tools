// Function: sub_6FFD30
// Address: 0x6ffd30
//
__int64 __fastcall sub_6FFD30(
        const __m128i *a1,
        _BYTE *a2,
        __m128i *a3,
        __int64 a4,
        int a5,
        int a6,
        unsigned int a7,
        int a8,
        unsigned int a9,
        __int64 *a10,
        _QWORD *a11,
        _QWORD *a12)
{
  int v14; // edx
  __int8 v15; // al
  bool v16; // r10
  int v17; // esi
  __int64 v18; // rax
  char i; // dl
  char v20; // di
  char v21; // al
  unsigned int v23; // eax
  __int64 v24; // rax
  char j; // dl
  __int8 v26; // r8
  __int64 v27; // rax
  char k; // dl
  _QWORD *v29; // rsi
  __int64 ii; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  bool v34; // r10
  int v35; // eax
  int v36; // eax
  __m128i *v37; // r9
  const __m128i *v38; // r8
  __int64 v39; // rax
  int v40; // eax
  __int8 v41; // al
  char v42; // al
  int v43; // esi
  char v44; // dl
  bool v45; // zf
  __int64 *v46; // rsi
  int v47; // eax
  bool v48; // r10
  __int64 v49; // rcx
  int v50; // r8d
  int v51; // eax
  __int64 m; // rax
  __int64 v53; // rdx
  char n; // al
  __int64 v55; // rax
  int v56; // eax
  int v57; // eax
  __int64 *v58; // r9
  bool v59; // r10
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rax
  __int64 v65; // rax
  int v66; // eax
  __int64 v67; // rdx
  __int64 v68; // r8
  __int64 v69; // r9
  bool v70; // r10
  __int64 v71; // rcx
  int v72; // eax
  __int64 v73; // rdx
  __int64 v74; // r8
  __int64 v75; // r9
  int v76; // eax
  int v77; // eax
  __int64 v78; // [rsp-10h] [rbp-220h]
  _QWORD *v79; // [rsp-8h] [rbp-218h]
  __int64 v80; // [rsp+10h] [rbp-200h]
  __int64 v81; // [rsp+18h] [rbp-1F8h]
  bool v82; // [rsp+18h] [rbp-1F8h]
  __int64 v83; // [rsp+20h] [rbp-1F0h]
  bool v84; // [rsp+20h] [rbp-1F0h]
  int v85; // [rsp+20h] [rbp-1F0h]
  __int64 v86; // [rsp+20h] [rbp-1F0h]
  int v87; // [rsp+28h] [rbp-1E8h]
  bool v88; // [rsp+28h] [rbp-1E8h]
  bool v89; // [rsp+28h] [rbp-1E8h]
  bool v90; // [rsp+28h] [rbp-1E8h]
  bool v91; // [rsp+28h] [rbp-1E8h]
  __int64 v92; // [rsp+30h] [rbp-1E0h]
  __m128i *v93; // [rsp+30h] [rbp-1E0h]
  __int64 v94; // [rsp+30h] [rbp-1E0h]
  __int64 v95; // [rsp+30h] [rbp-1E0h]
  __int64 *v96; // [rsp+30h] [rbp-1E0h]
  __int64 v97; // [rsp+30h] [rbp-1E0h]
  __int64 v98; // [rsp+30h] [rbp-1E0h]
  const __m128i *v99; // [rsp+30h] [rbp-1E0h]
  bool v100; // [rsp+38h] [rbp-1D8h]
  __int64 v101; // [rsp+38h] [rbp-1D8h]
  const __m128i *v102; // [rsp+38h] [rbp-1D8h]
  bool v103; // [rsp+38h] [rbp-1D8h]
  bool v104; // [rsp+38h] [rbp-1D8h]
  int v105; // [rsp+38h] [rbp-1D8h]
  bool v106; // [rsp+38h] [rbp-1D8h]
  int v107; // [rsp+38h] [rbp-1D8h]
  __int64 v108; // [rsp+38h] [rbp-1D8h]
  bool v109; // [rsp+38h] [rbp-1D8h]
  bool v110; // [rsp+38h] [rbp-1D8h]
  __m128i *v111; // [rsp+38h] [rbp-1D8h]
  _BOOL4 v112; // [rsp+48h] [rbp-1C8h]
  __int64 v113; // [rsp+50h] [rbp-1C0h]
  bool v114; // [rsp+50h] [rbp-1C0h]
  __int64 v115; // [rsp+50h] [rbp-1C0h]
  __int64 v116; // [rsp+58h] [rbp-1B8h]
  __int64 v119; // [rsp+78h] [rbp-198h] BYREF
  __int64 v120[50]; // [rsp+80h] [rbp-190h] BYREF

  v14 = a8;
  LODWORD(v113) = a6;
  if ( a5 )
  {
    v112 = 0;
  }
  else
  {
    if ( dword_4F077C4 != 2 )
    {
      v112 = 0;
      v15 = a1[1].m128i_i8[0];
      goto LABEL_4;
    }
    v101 = a4;
    v35 = sub_8D3A70(a4);
    a4 = v101;
    v14 = a8;
    v112 = v35 != 0;
  }
  v15 = a1[1].m128i_i8[0];
  if ( dword_4F077C4 != 2 )
    goto LABEL_4;
  if ( !v14 )
  {
    if ( v15 == 2 && a1[19].m128i_i8[13] == 12 )
      goto LABEL_21;
    if ( a2[16] == 2 && a2[317] == 12 )
    {
      if ( v15 != 2 )
      {
        v17 = 0;
        v16 = 0;
        goto LABEL_5;
      }
      goto LABEL_54;
    }
    if ( a3[1].m128i_i8[0] != 2 || a3[19].m128i_i8[13] != 12 )
    {
LABEL_4:
      v16 = 0;
      v17 = 0;
      if ( v15 != 2 )
      {
LABEL_5:
        if ( !v15 )
          goto LABEL_10;
        v18 = a1->m128i_i64[0];
        for ( i = *(_BYTE *)(a1->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v18 + 140) )
          v18 = *(_QWORD *)(v18 + 160);
        if ( !i )
          goto LABEL_10;
        v20 = a2[16];
        if ( !v20 )
          goto LABEL_10;
        v24 = *(_QWORD *)a2;
        for ( j = *(_BYTE *)(*(_QWORD *)a2 + 140LL); j == 12; j = *(_BYTE *)(v24 + 140) )
          v24 = *(_QWORD *)(v24 + 160);
        if ( !j )
          goto LABEL_10;
        v26 = a3[1].m128i_i8[0];
        if ( !v26 )
          goto LABEL_10;
        v27 = a3->m128i_i64[0];
        for ( k = *(_BYTE *)(a3->m128i_i64[0] + 140); k == 12; k = *(_BYTE *)(v27 + 140) )
          v27 = *(_QWORD *)(v27 + 160);
        if ( !k )
          goto LABEL_10;
        if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u
          && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0
          && !(word_4D04898 | v17)
          && dword_4F04C44 == -1 )
        {
          v65 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          if ( (*(_BYTE *)(v65 + 6) & 6) == 0 && *(_BYTE *)(v65 + 4) != 12 )
          {
            sub_6E68E0(0x103u, (__int64)a1);
LABEL_10:
            sub_6E6260(a12);
LABEL_11:
            v21 = a3[4].m128i_i8[0] | a1[4].m128i_i8[0];
            *((_BYTE *)a12 + 64) = v21;
            *((_BYTE *)a12 + 64) = a2[64] | v21;
            return sub_6E3BA0((__int64)a12, a10, 0, a11);
          }
        }
        if ( (_DWORD)v113 || !v112 )
        {
          v29 = a2;
          v114 = v16;
          sub_6F7CC0(a1, (const __m128i *)a2, a3, a4, a5 != 0, a9, (__int64)a12);
          v33 = v78;
          v34 = v114;
          if ( dword_4F077C4 != 2 )
            goto LABEL_11;
LABEL_38:
          if ( v34 )
            sub_6F4B70((__m128i *)a12, (__int64)v29, ii, v31, v32, v33);
          if ( a5 )
            sub_6E6A20((__int64)a12);
          goto LABEL_11;
        }
        v119 = 0;
        v120[0] = 0;
        if ( v20 == 1 )
        {
          v94 = a4;
          v103 = v16;
          sub_6EC9C0((__int64 *)a2);
          a4 = v94;
          v16 = v103;
          v26 = a3[1].m128i_i8[0];
        }
        if ( v26 == 1 )
        {
          v95 = a4;
          v104 = v16;
          sub_6EC9C0(a3->m128i_i64);
          a4 = v95;
          v16 = v104;
        }
        v81 = a4;
        v84 = v16;
        v46 = v120;
        v105 = sub_8319F0(a2, &v119);
        v47 = sub_8319F0(a3, v120);
        v48 = v84;
        v49 = v81;
        v50 = v47;
        if ( v105 )
        {
          if ( v47 )
          {
LABEL_123:
            if ( v50 )
            {
              v29 = a2;
              v89 = v48;
              v113 = *(_QWORD *)(v119 + 56);
              v108 = *(_QWORD *)(v120[0] + 56);
              sub_6F7CC0(a1, (const __m128i *)a2, a3, v49, a5 != 0, a9, (__int64)a12);
              v34 = v89;
              if ( dword_4F077C4 != 2 )
                goto LABEL_11;
              v31 = a7;
              *(_QWORD *)((char *)a12 + 68) = *(_QWORD *)(a2 + 68);
              if ( a7 )
              {
                *(_BYTE *)(v113 + 49) |= 4u;
                *(_BYTE *)(v108 + 49) |= 4u;
                LOBYTE(v113) = v112;
              }
              else
              {
                sub_733B20(v113);
                sub_733B20(v108);
                v58 = (_QWORD *)((char *)a12 + 68);
                v59 = v89;
                *(_QWORD *)(v113 + 16) = 0;
                *(_QWORD *)(v108 + 16) = 0;
                *(_BYTE *)(v113 + 49) &= ~4u;
                *(_BYTE *)(v108 + 49) &= ~4u;
                if ( dword_4D04964 )
                {
                  v29 = 0;
                  sub_83EB20(*a12, 0, (char *)a12 + 68);
                  v59 = v89;
                  v58 = (_QWORD *)((char *)a12 + 68);
                }
                v79 = v29;
                v90 = v59;
                v96 = (__int64 *)sub_6ECAE0(*a12, 0, 0, 1, 3u, v58, v120);
                *(_BYTE *)(v120[0] + 51) |= 2u;
                v116 = sub_6F6F40((const __m128i *)a12, 0, v60, v61, v62, v63);
                sub_7304E0(v116);
                v29 = a12;
                *(_QWORD *)(v120[0] + 56) = v116;
                sub_6E70E0(v96, (__int64)a12);
                v64 = v120[0];
                v34 = v90;
                *(_QWORD *)(v113 + 112) = v120[0];
                *(_QWORD *)(v108 + 112) = v64;
                v32 = (__int64)v79;
                LOBYTE(v113) = v112;
              }
              goto LABEL_111;
            }
LABEL_101:
            for ( m = v49; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
              ;
            if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)m + 96LL) + 177LL) & 0x40) != 0 )
            {
              v112 = 0;
            }
            else
            {
              v97 = v49;
              v109 = v48;
              v66 = sub_8D3A70(*(_QWORD *)a2);
              v70 = v109;
              v71 = v97;
              if ( v66 )
              {
                sub_6FA340((__int64)a2, (__int64)v46, v67, v97, v68, v69);
                v70 = v109;
                v71 = *(_QWORD *)a2;
              }
              v98 = v71;
              v110 = v70;
              v72 = sub_8D3A70(a3->m128i_i64[0]);
              v48 = v110;
              v49 = v98;
              if ( v72 )
              {
                sub_6FA340((__int64)a3, (__int64)v46, v73, v98, v74, v75);
                v48 = v110;
                v49 = a3->m128i_i64[0];
              }
            }
            v106 = v48;
            sub_6F7CC0(a1, (const __m128i *)a2, a3, v49, (a5 | v112) != 0, a9, (__int64)a12);
            if ( dword_4F077C4 != 2 )
              goto LABEL_11;
            v53 = *a12;
            v29 = (_QWORD *)v53;
            *(_QWORD *)((char *)a12 + 68) = *(_QWORD *)(a2 + 68);
            for ( n = *(_BYTE *)(v53 + 140); n == 12; n = *((_BYTE *)v29 + 140) )
              v29 = (_QWORD *)v29[20];
            if ( (unsigned __int8)(n - 9) <= 2u )
              v29 = (_QWORD *)v53;
            sub_8443E0(a12, v29, 0);
            v34 = v106;
LABEL_111:
            if ( *((_BYTE *)a12 + 16) )
            {
              v55 = *a12;
              for ( ii = *(unsigned __int8 *)(*a12 + 140LL); (_BYTE)ii == 12; ii = *(unsigned __int8 *)(v55 + 140) )
                v55 = *(_QWORD *)(v55 + 160);
              if ( (_BYTE)ii && ((v113 & 1) == 0 || !a7) )
                *(_BYTE *)(*(_QWORD *)(a12[18] + 56LL) + 51LL) |= 1u;
            }
            goto LABEL_38;
          }
          v51 = sub_8D3A70(a3->m128i_i64[0]);
          v48 = v84;
          v49 = v81;
          if ( !v51 )
            goto LABEL_101;
          sub_8443E0(a3, a3->m128i_i64[0], 0);
          v46 = v120;
          v76 = sub_8319F0(a3, v120);
          v48 = v84;
          v49 = v81;
          v50 = v76;
        }
        else if ( v47 )
        {
          v107 = v47;
          v56 = sub_8D3A70(*(_QWORD *)a2);
          v48 = v84;
          v49 = v81;
          if ( !v56 )
            goto LABEL_101;
          v80 = v81;
          v82 = v84;
          v85 = v107;
          sub_8443E0(a2, *(_QWORD *)a2, 0);
          v46 = &v119;
          v57 = sub_8319F0(a2, &v119);
          v49 = v80;
          v105 = v57;
          v48 = v82;
          v50 = v85;
        }
        if ( !v105 )
          goto LABEL_101;
        goto LABEL_123;
      }
      goto LABEL_22;
    }
  }
  v16 = 0;
  v17 = 0;
  if ( v15 != 2 )
    goto LABEL_5;
LABEL_21:
  v16 = 0;
  if ( a2[16] == 2 )
  {
LABEL_54:
    if ( a3[1].m128i_i8[0] == 2 )
      v14 = 1;
    v16 = a3[1].m128i_i8[0] == 2;
  }
LABEL_22:
  v87 = v14;
  v92 = a4;
  v100 = v16;
  v23 = sub_70FCE0(&a1[9]);
  v16 = v100;
  a4 = v92;
  v17 = v23;
  if ( !v23 )
  {
LABEL_23:
    v15 = a1[1].m128i_i8[0];
    goto LABEL_5;
  }
  v36 = sub_6E9820((__int64)a1, v23);
  v16 = v100;
  a4 = v92;
  if ( v36 )
  {
    v37 = (__m128i *)a2;
    v38 = a3;
  }
  else
  {
    v37 = a3;
    v38 = (const __m128i *)a2;
  }
  if ( v38[1].m128i_i8[0] != 2 )
    goto LABEL_47;
  if ( v87 )
    goto LABEL_47;
  v39 = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x20) != 0 )
    goto LABEL_47;
  if ( *(_QWORD *)a2 != a3->m128i_i64[0] )
  {
    v83 = v92;
    v88 = v100;
    v93 = v37;
    v102 = v38;
    v40 = sub_8D97D0(*(_QWORD *)a2, a3->m128i_i64[0], 0, a4, v38);
    v38 = v102;
    v37 = v93;
    v16 = v88;
    a4 = v83;
    if ( !v40 )
      goto LABEL_47;
    v39 = qword_4D03C50;
  }
  if ( *(_BYTE *)(v39 + 16) <= 3u || a2[16] == 2 && a3[1].m128i_i8[0] == 2 )
    goto LABEL_72;
  v17 = v112;
  if ( v112 )
    goto LABEL_23;
  if ( HIDWORD(qword_4F077B4) )
  {
    if ( v37[1].m128i_i8[0] == 1 )
    {
      v86 = a4;
      v91 = v16;
      v99 = v38;
      v111 = v37;
      v77 = sub_731D60(v37[9].m128i_i64[0]);
      v37 = v111;
      v38 = v99;
      v16 = v91;
      a4 = v86;
      if ( v77 )
      {
LABEL_47:
        v15 = a1[1].m128i_i8[0];
        v17 = 1;
        goto LABEL_5;
      }
    }
  }
  if ( (a2[64] & 2) != 0 || (a3[4].m128i_i8[0] & 2) != 0 )
  {
    if ( dword_4D04964 )
      goto LABEL_47;
    if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) == 0 )
    {
      v112 = 0;
      v15 = a1[1].m128i_i8[0];
      v17 = 1;
      goto LABEL_5;
    }
  }
LABEL_72:
  *(__m128i *)a12 = _mm_loadu_si128(v38);
  *((__m128i *)a12 + 1) = _mm_loadu_si128(v38 + 1);
  *((__m128i *)a12 + 2) = _mm_loadu_si128(v38 + 2);
  *((__m128i *)a12 + 3) = _mm_loadu_si128(v38 + 3);
  *((__m128i *)a12 + 4) = _mm_loadu_si128(v38 + 4);
  *((__m128i *)a12 + 5) = _mm_loadu_si128(v38 + 5);
  *((__m128i *)a12 + 6) = _mm_loadu_si128(v38 + 6);
  *((__m128i *)a12 + 7) = _mm_loadu_si128(v38 + 7);
  *((__m128i *)a12 + 8) = _mm_loadu_si128(v38 + 8);
  v41 = v38[1].m128i_i8[0];
  if ( v41 == 2 )
  {
    *((__m128i *)a12 + 9) = _mm_loadu_si128(v38 + 9);
    *((__m128i *)a12 + 10) = _mm_loadu_si128(v38 + 10);
    *((__m128i *)a12 + 11) = _mm_loadu_si128(v38 + 11);
    *((__m128i *)a12 + 12) = _mm_loadu_si128(v38 + 12);
    *((__m128i *)a12 + 13) = _mm_loadu_si128(v38 + 13);
    *((__m128i *)a12 + 14) = _mm_loadu_si128(v38 + 14);
    *((__m128i *)a12 + 15) = _mm_loadu_si128(v38 + 15);
    *((__m128i *)a12 + 16) = _mm_loadu_si128(v38 + 16);
    *((__m128i *)a12 + 17) = _mm_loadu_si128(v38 + 17);
    *((__m128i *)a12 + 18) = _mm_loadu_si128(v38 + 18);
    *((__m128i *)a12 + 19) = _mm_loadu_si128(v38 + 19);
    *((__m128i *)a12 + 20) = _mm_loadu_si128(v38 + 20);
    *((__m128i *)a12 + 21) = _mm_loadu_si128(v38 + 21);
  }
  else if ( v41 == 5 || v41 == 1 )
  {
    a12[18] = v38[9].m128i_i64[0];
  }
  *((_WORD *)a12 + 9) &= 0xAFD7u;
  v42 = *((_BYTE *)a12 + 64);
  v43 = dword_4D04964;
  v44 = v42 | v37[4].m128i_i8[0];
  v45 = *((_BYTE *)a12 + 16) == 2;
  *((_BYTE *)a12 + 64) = v44;
  if ( v45 )
  {
    if ( v37[1].m128i_i8[0] != 2 || (v37[19].m128i_i8[9] & 4) != 0 )
      *((_BYTE *)a12 + 313) |= 4u;
    if ( v43 )
    {
      *((_BYTE *)a12 + 64) = a1[4].m128i_i8[0] | v44;
    }
    else
    {
      *((_BYTE *)a12 + 64) = v42;
      *((_BYTE *)a12 + 64) = a1[4].m128i_i8[0] | v42;
    }
    if ( a1[1].m128i_i8[0] != 2 || (a1[19].m128i_i8[9] & 4) != 0 )
      *((_BYTE *)a12 + 313) |= 4u;
    v115 = a4;
    sub_72A160(a12 + 18);
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
    {
      sub_6F7CC0(a1, (const __m128i *)a2, a3, v115, a5, a9, (__int64)v120);
      a12[36] = v120[18];
    }
  }
  else if ( v43 )
  {
    *((_BYTE *)a12 + 64) = a1[4].m128i_i8[0] | v44;
  }
  else
  {
    *((_BYTE *)a12 + 64) = v42;
    *((_BYTE *)a12 + 64) = a1[4].m128i_i8[0] | v42;
  }
  return sub_6E3BA0((__int64)a12, a10, 0, a11);
}
