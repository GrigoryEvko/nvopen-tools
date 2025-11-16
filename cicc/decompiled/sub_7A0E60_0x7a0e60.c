// Function: sub_7A0E60
// Address: 0x7a0e60
//
__int64 __fastcall sub_7A0E60(__m128i *a1, _QWORD *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v6; // r12
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // r8
  int v14; // edi
  int v15; // eax
  __int64 v16; // r15
  unsigned __int64 v17; // r14
  char i; // al
  int v19; // r13d
  size_t v20; // r8
  __int64 v21; // rbx
  __int64 v22; // r13
  void *v23; // rcx
  _QWORD *v24; // rbx
  unsigned __int64 *v25; // rax
  unsigned __int64 v26; // rbx
  char m; // al
  int v28; // r13d
  unsigned int v29; // edx
  unsigned int v30; // eax
  size_t v31; // r8
  unsigned int v32; // r13d
  void *v33; // rcx
  _QWORD *v34; // rcx
  __m128i *v35; // rax
  __int64 v36; // r12
  __int64 v37; // r13
  __int64 v38; // r14
  unsigned int v39; // eax
  __int64 v40; // rax
  _QWORD *v41; // rcx
  bool v42; // r11
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  __int32 v45; // ecx
  __int32 v46; // edi
  __int64 v47; // rsi
  __int64 v48; // r13
  unsigned int v49; // edx
  _DWORD *kk; // rax
  __int64 v51; // rcx
  __int64 v52; // rsi
  unsigned int v53; // eax
  __int64 v54; // rdx
  __int64 v55; // rbx
  unsigned __int64 jj; // rbx
  unsigned int v57; // edx
  unsigned int v58; // eax
  unsigned int v59; // r13d
  __int64 v60; // rax
  __int64 v61; // rdx
  unsigned int v62; // eax
  char v63; // al
  unsigned int v64; // ecx
  __int64 v65; // rsi
  int v66; // edx
  unsigned int v67; // eax
  int *v68; // r8
  int v69; // edi
  int v70; // eax
  unsigned int v71; // eax
  bool v72; // zf
  unsigned int v73; // edi
  unsigned int v74; // esi
  __int64 v75; // rcx
  __int64 *v76; // r8
  _DWORD *j; // rax
  __int64 v78; // rcx
  __int64 v79; // rsi
  unsigned int v80; // eax
  unsigned int v82; // ecx
  __int64 v83; // rsi
  __int64 v84; // r13
  int v85; // edx
  unsigned int v86; // eax
  int *v87; // r8
  int v88; // edi
  int v89; // eax
  unsigned int v90; // eax
  __int32 v91; // edi
  __int32 v92; // esi
  __int64 v93; // rcx
  _DWORD *n; // rax
  __int64 v95; // rcx
  __int64 v96; // rsi
  unsigned int v97; // eax
  __int64 v98; // r13
  int *v99; // rdx
  __int64 v100; // rsi
  _QWORD *v101; // rcx
  _QWORD *ii; // rax
  __int64 v103; // rax
  unsigned int v104; // r13d
  __int64 v105; // rax
  __int64 v106; // rdx
  unsigned int v107; // edx
  int *v108; // rdx
  __int64 v109; // rsi
  _QWORD *v110; // rcx
  _QWORD *k; // rax
  __int64 v112; // rax
  __int64 *v113; // [rsp+8h] [rbp-F8h]
  __int64 v114; // [rsp+10h] [rbp-F0h]
  __int64 *v115; // [rsp+18h] [rbp-E8h]
  __int64 v116; // [rsp+18h] [rbp-E8h]
  int v117; // [rsp+24h] [rbp-DCh]
  __int32 v118; // [rsp+24h] [rbp-DCh]
  __int64 v119; // [rsp+28h] [rbp-D8h]
  __int64 v120; // [rsp+28h] [rbp-D8h]
  __int64 v121; // [rsp+30h] [rbp-D0h]
  __int64 *v122; // [rsp+30h] [rbp-D0h]
  __int64 v123; // [rsp+30h] [rbp-D0h]
  size_t v124; // [rsp+30h] [rbp-D0h]
  size_t v125; // [rsp+30h] [rbp-D0h]
  __int64 v126; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v127; // [rsp+38h] [rbp-C8h]
  unsigned __int64 *v128; // [rsp+40h] [rbp-C0h]
  __int64 v129; // [rsp+48h] [rbp-B8h]
  __int64 v130; // [rsp+50h] [rbp-B0h]
  __int64 v131; // [rsp+58h] [rbp-A8h]
  __int64 v132; // [rsp+60h] [rbp-A0h]
  __int64 v134; // [rsp+68h] [rbp-98h]
  __int32 v136; // [rsp+78h] [rbp-88h]
  unsigned int v137; // [rsp+7Ch] [rbp-84h]
  __int64 *v138; // [rsp+80h] [rbp-80h]
  _WORD *v139; // [rsp+88h] [rbp-78h]
  size_t v140; // [rsp+88h] [rbp-78h]
  size_t v141; // [rsp+88h] [rbp-78h]
  unsigned int v142; // [rsp+98h] [rbp-68h] BYREF
  int v143; // [rsp+9Ch] [rbp-64h] BYREF
  __m128i v144[6]; // [rsp+A0h] [rbp-60h] BYREF

  v6 = a1;
  v7 = (__int64 *)a2[10];
  v8 = a1[4].m128i_u32[0];
  v142 = 1;
  v9 = *v7;
  v138 = v7;
  v10 = a1[3].m128i_i64[1];
  v129 = a1[1].m128i_i64[0];
  v130 = a1[1].m128i_i64[1];
  v131 = a1[2].m128i_i64[0];
  v136 = a1[2].m128i_i32[2];
  v132 = a1[3].m128i_i64[0];
  v11 = (unsigned int)(a1[8].m128i_i32[0] + 1);
  a1[8].m128i_i32[0] = v11;
  v12 = v8 & v11;
  a1[2].m128i_i32[2] = v11;
  v13 = v10 + 4LL * ((unsigned int)v8 & (unsigned int)v11);
  v14 = *(_DWORD *)v13;
  *(_DWORD *)v13 = v11;
  if ( v14 )
  {
    do
    {
      v12 = v8 & (v12 + 1);
      v11 = v10 + 4LL * v12;
    }
    while ( *(_DWORD *)v11 );
    *(_DWORD *)v11 = v14;
  }
  v15 = v6[4].m128i_i32[1] + 1;
  v6[4].m128i_i32[1] = v15;
  if ( 2 * v15 > (unsigned int)v8 )
    sub_7702C0((__int64)&v6[3].m128i_i64[1]);
  v6[3].m128i_i64[0] = 0;
  if ( v9 && (a3 & 1) == 0 && !(unsigned int)sub_795660((__int64)v6, v9, v11, v8, v13, a6) )
    goto LABEL_75;
  v16 = a2[6];
  if ( !v16 )
  {
    v139 = 0;
    v17 = 0;
    v137 = 0;
    goto LABEL_21;
  }
  v17 = *(_QWORD *)v16;
  for ( i = *(_BYTE *)(*(_QWORD *)v16 + 140LL); i == 12; i = *(_BYTE *)(v17 + 140) )
    v17 = *(_QWORD *)(v17 + 160);
  if ( (*(_BYTE *)(v16 + 25) & 3) != 0 )
  {
    v19 = 32;
    goto LABEL_11;
  }
  v19 = 16;
  if ( (unsigned __int8)(i - 2) <= 1u )
  {
LABEL_11:
    if ( (unsigned __int8)(*(_BYTE *)(v17 + 140) - 8) <= 3u )
    {
      v107 = ((unsigned int)(v19 + 7) >> 3) + 17 - (((unsigned __int8)((unsigned int)(v19 + 7) >> 3) + 9) & 7);
      v21 = v107;
      v22 = v107 + v19;
      v20 = v107 - 8LL;
      goto LABEL_13;
    }
    goto LABEL_12;
  }
  v19 = sub_7764B0((__int64)v6, v17, &v142);
  if ( (unsigned __int8)(*(_BYTE *)(v17 + 140) - 8) <= 3u )
  {
    v57 = (unsigned int)(v19 + 7) >> 3;
    v58 = v57 + 9;
    if ( (((_BYTE)v57 + 9) & 7) != 0 )
      v58 = v57 + 17 - (((_BYTE)v57 + 9) & 7);
    v21 = v58;
    v22 = v58 + v19;
    v20 = v58 - 8LL;
    if ( (unsigned int)v22 <= 0x400 )
      goto LABEL_13;
LABEL_85:
    v140 = v20;
    v59 = v22 + 16;
    v60 = sub_822B10(v59);
    v61 = v6[2].m128i_i64[0];
    v20 = v140;
    *(_DWORD *)(v60 + 8) = v59;
    v23 = (void *)(v60 + 16);
    *(_QWORD *)v60 = v61;
    *(_DWORD *)(v60 + 12) = v6[2].m128i_i32[2];
    v6[2].m128i_i64[0] = v60;
    goto LABEL_18;
  }
LABEL_12:
  v20 = 8;
  v21 = 16;
  v22 = (unsigned int)(v19 + 16);
  if ( (unsigned int)v22 > 0x400 )
    goto LABEL_85;
LABEL_13:
  v23 = (void *)v6[1].m128i_i64[0];
  if ( (v22 & 7) != 0 )
    v22 = (_DWORD)v22 + 8 - (unsigned int)(v22 & 7);
  if ( 0x10000 - (v6[1].m128i_i32[0] - v6[1].m128i_i32[2]) < (unsigned int)v22 )
  {
    v141 = v20;
    sub_772E70((__m128i *)v6[1].m128i_i64);
    v23 = (void *)v6[1].m128i_i64[0];
    v20 = v141;
  }
  v6[1].m128i_i64[0] = (__int64)v23 + v22;
LABEL_18:
  v24 = (char *)memset(v23, 0, v20) + v21;
  *(v24 - 1) = v17;
  v139 = v24;
  if ( (unsigned __int8)(*(_BYTE *)(v17 + 140) - 9) <= 2u )
    *v24 = 0;
  v137 = 0;
  if ( *(_BYTE *)(v16 + 24) == 9 )
  {
    v137 = 1;
    if ( !(unsigned int)sub_77A4E0(v6, *(_QWORD *)(v16 + 56), v144) )
    {
LABEL_75:
      v142 = 0;
      goto LABEL_76;
    }
  }
LABEL_21:
  v25 = (unsigned __int64 *)v138[1];
  v128 = v25;
  if ( !v25 )
  {
    v26 = 0;
    v127 = (unsigned __int64)v139;
LABEL_41:
    v35 = v6;
    v36 = v17;
    v37 = 0;
    v38 = (__int64)v35;
    while ( 1 )
    {
      v41 = &qword_4D042E0;
      v43 = *(_QWORD *)(v38 + 120) + 1LL;
      *(_QWORD *)(v38 + 120) = v43;
      if ( v43 <= qword_4D042E0 )
        break;
      sub_6855B0(0x97Fu, (FILE *)(v38 + 112), (_QWORD *)(v38 + 96));
      v142 = 0;
      v42 = v37 == 0;
      if ( !v137 )
      {
LABEL_48:
        if ( !v142 )
          goto LABEL_53;
        goto LABEL_49;
      }
LABEL_52:
      sub_7762B0(v38, *(_QWORD *)(v16 + 56));
      if ( !v142 )
      {
LABEL_53:
        v6 = (__m128i *)v38;
        goto LABEL_54;
      }
LABEL_49:
      if ( v42 )
        goto LABEL_53;
    }
    if ( v16 )
    {
      v39 = sub_7A0A10(v137, (const __m128i *)v38, v16, v36, (unsigned __int64)v139, a6);
      ++*(_QWORD *)(v38 + 120);
      v142 = v39;
      if ( v39 )
      {
        v40 = sub_620EE0(v139, byte_4B6DF90[*(unsigned __int8 *)(v36 + 160)], &v143);
        v37 = v40;
        if ( !v143 )
        {
          if ( !v40 )
          {
LABEL_46:
            v42 = 1;
            v37 = 0;
            goto LABEL_47;
          }
          goto LABEL_89;
        }
      }
    }
    else if ( v142 )
    {
      v143 = 0;
      v37 = 1;
LABEL_89:
      v62 = sub_795660(v38, a2[9], v11, (__int64)v41, v13, a6);
      v42 = 0;
      v142 = v62;
      if ( v62 )
      {
        v11 = *(_QWORD *)(v38 + 72);
        v63 = *(_BYTE *)(v11 + 48);
        if ( (v63 & 1) != 0 )
          goto LABEL_46;
        if ( (v63 & 2) != 0 )
        {
          *(_BYTE *)(v11 + 48) = v63 & 0xFD;
          goto LABEL_46;
        }
        if ( (v63 & 4) != 0 )
          *(_BYTE *)(v11 + 48) = v63 & 0xFB;
        v42 = 0;
        if ( v128 )
        {
          v64 = *(_DWORD *)(v38 + 64);
          v65 = *(_QWORD *)(v38 + 56);
          v121 = *(_QWORD *)(v38 + 16);
          v114 = *(_QWORD *)(v38 + 24);
          v115 = *(__int64 **)(v38 + 32);
          v117 = *(_DWORD *)(v38 + 40);
          v119 = *(_QWORD *)(v38 + 48);
          v66 = *(_DWORD *)(v38 + 128) + 1;
          *(_DWORD *)(v38 + 128) = v66;
          v67 = v64 & v66;
          *(_DWORD *)(v38 + 40) = v66;
          v68 = (int *)(v65 + 4LL * (v64 & v66));
          v69 = *v68;
          *v68 = v66;
          if ( v69 )
          {
            do
            {
              v67 = v64 & (v67 + 1);
              v108 = (int *)(v65 + 4LL * v67);
            }
            while ( *v108 );
            *v108 = v69;
          }
          v70 = *(_DWORD *)(v38 + 68) + 1;
          *(_DWORD *)(v38 + 68) = v70;
          if ( 2 * v70 > v64 )
            sub_7702C0(v38 + 56);
          *(_QWORD *)(v38 + 48) = 0;
          v71 = sub_786210(v38, (_QWORD **)v128, v127, (char *)v127);
          v72 = *(_QWORD *)(v38 + 48) == 0;
          v142 = v71;
          if ( !v72 && v71 )
            v142 = sub_799890(v38);
          v73 = *(_DWORD *)(v38 + 40);
          v74 = *(_DWORD *)(v38 + 64);
          v75 = *(_QWORD *)(v38 + 56);
          v76 = *(__int64 **)(v38 + 32);
          v11 = v74 & v73;
          for ( j = (_DWORD *)(v75 + 4 * v11); v73 != *j; j = (_DWORD *)(v75 + 4LL * (unsigned int)v11) )
            v11 = v74 & ((_DWORD)v11 + 1);
          *j = 0;
          a6 = *(unsigned int *)(v75 + 4LL * (((_DWORD)v11 + 1) & v74));
          if ( (_DWORD)a6 )
          {
            v113 = v76;
            sub_771390(*(_QWORD *)(v38 + 56), *(_DWORD *)(v38 + 64), v11);
            v76 = v113;
          }
          --*(_DWORD *)(v38 + 68);
          *(_QWORD *)(v38 + 16) = v121;
          *(_DWORD *)(v38 + 40) = v117;
          *(_QWORD *)(v38 + 24) = v114;
          *(_QWORD *)(v38 + 48) = v119;
          *(_QWORD *)(v38 + 32) = v115;
          if ( v76 && v115 != v76 )
          {
            while ( 1 )
            {
              v78 = *((unsigned int *)v76 + 3);
              v79 = *(_QWORD *)(v38 + 56);
              v80 = v78 & *(_DWORD *)(v38 + 64);
              v11 = *(unsigned int *)(v79 + 4LL * v80);
              if ( !(_DWORD)v78 || (_DWORD)v11 == (_DWORD)v78 )
                break;
              while ( (_DWORD)v11 )
              {
                v80 = *(_DWORD *)(v38 + 64) & (v80 + 1);
                v11 = *(unsigned int *)(v79 + 4LL * v80);
                if ( (_DWORD)v78 == (_DWORD)v11 )
                  goto LABEL_159;
              }
              v122 = (__int64 *)*v76;
              sub_822B90(v76, *((unsigned int *)v76 + 2), v11, v78);
              if ( !v122 )
              {
                v76 = 0;
                break;
              }
              v76 = v122;
            }
LABEL_159:
            *(_QWORD *)(v38 + 32) = v76;
          }
          v13 = v142;
          v42 = 0;
          if ( v142 && ((*((_BYTE *)v128 + 25) & 3) != 0 || *(_BYTE *)(v26 + 140) == 6) )
          {
            v42 = 0;
            if ( (*(_BYTE *)(v127 + 8) & 4) != 0 )
            {
              v109 = *(_QWORD *)(v127 + 16);
              v11 = 2;
              v110 = *(_QWORD **)v109;
              for ( k = **(_QWORD ***)v109; k; ++v11 )
              {
                v110 = k;
                k = (_QWORD *)*k;
              }
              v42 = 0;
              *v110 = qword_4F08088;
              *(_BYTE *)(v127 + 8) &= ~4u;
              v112 = *(_QWORD *)(v109 + 24);
              qword_4F08080 += v11;
              qword_4F08088 = v109;
              *(_QWORD *)(v127 + 16) = v112;
            }
          }
        }
      }
LABEL_47:
      if ( !v137 )
        goto LABEL_48;
      goto LABEL_52;
    }
    v42 = v37 == 0;
    goto LABEL_47;
  }
  v26 = *v25;
  for ( m = *(_BYTE *)(*v25 + 140); m == 12; m = *(_BYTE *)(v26 + 140) )
    v26 = *(_QWORD *)(v26 + 160);
  v28 = 32;
  if ( (*((_BYTE *)v128 + 25) & 3) == 0 )
  {
    v28 = 16;
    if ( (unsigned __int8)(m - 2) > 1u )
      v28 = sub_7764B0((__int64)v6, v26, &v142);
  }
  if ( v142 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(v26 + 140) - 8) > 3u )
    {
      v126 = 16;
      v31 = 8;
      v30 = 16;
    }
    else
    {
      v29 = (unsigned int)(v28 + 7) >> 3;
      v30 = v29 + 9;
      if ( (((_BYTE)v29 + 9) & 7) != 0 )
        v30 = v29 + 17 - (((_BYTE)v29 + 9) & 7);
      v126 = v30;
      v31 = v30 - 8LL;
    }
    v32 = v30 + v28;
    if ( v32 > 0x400 )
    {
      v125 = v31;
      v104 = v32 + 16;
      v105 = sub_822B10(v104);
      v106 = v6[2].m128i_i64[0];
      v31 = v125;
      *(_DWORD *)(v105 + 8) = v104;
      v33 = (void *)(v105 + 16);
      *(_QWORD *)v105 = v106;
      *(_DWORD *)(v105 + 12) = v6[2].m128i_i32[2];
      v6[2].m128i_i64[0] = v105;
    }
    else
    {
      v33 = (void *)v6[1].m128i_i64[0];
      if ( (v32 & 7) != 0 )
        v32 = v32 + 8 - (v32 & 7);
      if ( 0x10000 - (v6[1].m128i_i32[0] - v6[1].m128i_i32[2]) < v32 )
      {
        v124 = v31;
        sub_772E70((__m128i *)v6[1].m128i_i64);
        v33 = (void *)v6[1].m128i_i64[0];
        v31 = v124;
      }
      v6[1].m128i_i64[0] = (__int64)v33 + v32;
    }
    v34 = (char *)memset(v33, 0, v31) + v126;
    *(v34 - 1) = v26;
    v127 = (unsigned __int64)v34;
    if ( (unsigned __int8)(*(_BYTE *)(v26 + 140) - 9) <= 2u )
      *v34 = 0;
    a6 = a3;
    if ( a3 )
    {
      v82 = v6[4].m128i_u32[0];
      v83 = v6[3].m128i_i64[1];
      v84 = v6[2].m128i_i64[0];
      v134 = v6[1].m128i_i64[0];
      v123 = v6[1].m128i_i64[1];
      v118 = v6[2].m128i_i32[2];
      v120 = v6[3].m128i_i64[0];
      v85 = v6[8].m128i_i32[0] + 1;
      v6[8].m128i_i32[0] = v85;
      v86 = v82 & v85;
      v6[2].m128i_i32[2] = v85;
      v87 = (int *)(v83 + 4LL * (v82 & v85));
      v88 = *v87;
      *v87 = v85;
      if ( v88 )
      {
        do
        {
          v86 = v82 & (v86 + 1);
          v99 = (int *)(v83 + 4LL * v86);
        }
        while ( *v99 );
        *v99 = v88;
      }
      v89 = v6[4].m128i_i32[1] + 1;
      v6[4].m128i_i32[1] = v89;
      if ( 2 * v89 > v82 )
        sub_7702C0((__int64)&v6[3].m128i_i64[1]);
      v6[3].m128i_i64[0] = 0;
      v90 = sub_786210((__int64)v6, (_QWORD **)v128, v127, (char *)v127);
      v72 = v6[3].m128i_i64[0] == 0;
      v142 = v90;
      if ( !v72 && v90 )
        v142 = sub_799890((__int64)v6);
      v91 = v6[2].m128i_i32[2];
      v92 = v6[4].m128i_i32[0];
      v93 = v6[3].m128i_i64[1];
      v13 = v6[2].m128i_i64[0];
      v11 = v92 & (unsigned int)v91;
      for ( n = (_DWORD *)(v93 + 4 * v11); v91 != *n; n = (_DWORD *)(v93 + 4LL * (unsigned int)v11) )
        v11 = v92 & (unsigned int)(v11 + 1);
      *n = 0;
      if ( *(_DWORD *)(v93 + 4LL * (((_DWORD)v11 + 1) & (unsigned int)v92)) )
      {
        v116 = v13;
        sub_771390(v6[3].m128i_i64[1], v6[4].m128i_i32[0], v11);
        v13 = v116;
      }
      --v6[4].m128i_i32[1];
      v6[2].m128i_i64[0] = v84;
      v6[1].m128i_i64[0] = v134;
      v6[1].m128i_i64[1] = v123;
      v6[2].m128i_i32[2] = v118;
      v6[3].m128i_i64[0] = v120;
      if ( v13 && v84 != v13 )
      {
        while ( 1 )
        {
          v95 = *(unsigned int *)(v13 + 12);
          v96 = v6[3].m128i_i64[1];
          v97 = v95 & v6[4].m128i_i32[0];
          v11 = *(unsigned int *)(v96 + 4LL * v97);
          if ( (_DWORD)v11 == (_DWORD)v95 || !(_DWORD)v95 )
            goto LABEL_141;
          while ( (_DWORD)v11 )
          {
            v97 = v6[4].m128i_i32[0] & (v97 + 1);
            v11 = *(unsigned int *)(v96 + 4LL * v97);
            if ( (_DWORD)v95 == (_DWORD)v11 )
              goto LABEL_141;
          }
          v98 = *(_QWORD *)v13;
          sub_822B90(v13, *(unsigned int *)(v13 + 8), v11, v95);
          if ( !v98 )
            break;
          v13 = v98;
        }
        v13 = 0;
LABEL_141:
        v6[2].m128i_i64[0] = v13;
      }
      if ( v142 && ((*((_BYTE *)v128 + 25) & 3) != 0 || *(_BYTE *)(v26 + 140) == 6) && (*(_BYTE *)(v127 + 8) & 4) != 0 )
      {
        v100 = *(_QWORD *)(v127 + 16);
        v11 = 2;
        v101 = *(_QWORD **)v100;
        for ( ii = **(_QWORD ***)v100; ii; ++v11 )
        {
          v101 = ii;
          ii = (_QWORD *)*ii;
        }
        *v101 = qword_4F08088;
        *(_BYTE *)(v127 + 8) &= ~4u;
        v103 = *(_QWORD *)(v100 + 24);
        qword_4F08080 += v11;
        qword_4F08088 = v100;
        *(_QWORD *)(v127 + 16) = v103;
      }
    }
    goto LABEL_41;
  }
LABEL_54:
  if ( v137 )
  {
    sub_7999E0(v6, *(_QWORD *)(v16 + 56), v144, &v142);
    v44 = v138[2];
    if ( !v44 )
      goto LABEL_56;
    goto LABEL_77;
  }
LABEL_76:
  v44 = v138[2];
  if ( !v44 )
    goto LABEL_56;
LABEL_77:
  for ( jj = *(_QWORD *)(v44 + 120); jj; jj = *(_QWORD *)(jj + 112) )
    sub_77A750((__int64)v6, jj);
LABEL_56:
  if ( v6[3].m128i_i64[0] && v142 )
    v142 = sub_799890((__int64)v6);
  v45 = v6[2].m128i_i32[2];
  v46 = v6[4].m128i_i32[0];
  v47 = v6[3].m128i_i64[1];
  v48 = v6[2].m128i_i64[0];
  v49 = v46 & v45;
  for ( kk = (_DWORD *)(v47 + 4LL * (v46 & (unsigned int)v45)); v45 != *kk; kk = (_DWORD *)(v47 + 4LL * v49) )
    v49 = v46 & (v49 + 1);
  *kk = 0;
  if ( *(_DWORD *)(v47 + 4LL * ((v49 + 1) & v46)) )
    sub_771390(v6[3].m128i_i64[1], v6[4].m128i_i32[0], v49);
  --v6[4].m128i_i32[1];
  v6[1].m128i_i64[0] = v129;
  v6[2].m128i_i32[2] = v136;
  v6[1].m128i_i64[1] = v130;
  v6[3].m128i_i64[0] = v132;
  v6[2].m128i_i64[0] = v131;
  if ( v131 != v48 && v48 )
  {
    while ( 1 )
    {
      v51 = *(unsigned int *)(v48 + 12);
      v52 = v6[3].m128i_i64[1];
      v53 = v51 & v6[4].m128i_i32[0];
      v54 = *(unsigned int *)(v52 + 4LL * v53);
      if ( !(_DWORD)v51 || (_DWORD)v54 == (_DWORD)v51 )
        break;
      while ( (_DWORD)v54 )
      {
        v53 = v6[4].m128i_i32[0] & (v53 + 1);
        v54 = *(unsigned int *)(v52 + 4LL * v53);
        if ( (_DWORD)v51 == (_DWORD)v54 )
          goto LABEL_115;
      }
      v55 = *(_QWORD *)v48;
      sub_822B90(v48, *(unsigned int *)(v48 + 8), v54, v51);
      if ( !v55 )
      {
        v48 = 0;
        break;
      }
      v48 = v55;
    }
LABEL_115:
    v6[2].m128i_i64[0] = v48;
  }
  return v142;
}
