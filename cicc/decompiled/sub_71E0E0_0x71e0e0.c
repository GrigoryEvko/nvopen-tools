// Function: sub_71E0E0
// Address: 0x71e0e0
//
unsigned int *__fastcall sub_71E0E0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 i; // rdi
  __int64 j; // r15
  __int64 v8; // rsi
  char v9; // r13
  unsigned __int64 v10; // rax
  _QWORD *v11; // r14
  unsigned int v12; // edx
  __int64 v13; // rdi
  unsigned __int64 *v14; // r9
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 *v17; // rdi
  unsigned __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rax
  char v21; // al
  _QWORD *v22; // r14
  __int64 v23; // rbx
  _QWORD *v24; // rdi
  bool v25; // zf
  _QWORD *v26; // r12
  int v27; // edi
  const __m128i *v28; // r14
  __int64 v29; // r15
  __int64 **v30; // rax
  const __m128i *v31; // rax
  __int8 v32; // al
  __int64 v33; // rdi
  unsigned int v34; // r8d
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // rcx
  int v39; // ebx
  char v40; // al
  __int64 v41; // rdi
  __int64 v42; // rax
  _QWORD *v43; // rdx
  __int64 v44; // r8
  char v45; // al
  __int64 *v46; // rcx
  __int64 mm; // rax
  char v48; // dl
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  unsigned int *result; // rax
  __int64 v53; // rdx
  int v54; // eax
  __int64 jj; // rax
  unsigned __int64 v56; // rax
  __int64 kk; // r13
  unsigned int v58; // edx
  int v59; // eax
  unsigned int v60; // ebx
  _QWORD *v61; // rax
  unsigned int v62; // r11d
  _QWORD *v63; // rcx
  _QWORD *v64; // rsi
  __int64 v65; // r10
  unsigned __int64 *v66; // rsi
  unsigned __int64 v67; // rdi
  unsigned __int64 m; // rdx
  unsigned int v69; // edx
  unsigned __int64 *v70; // rax
  _QWORD *v71; // r13
  __int64 *v72; // r13
  _QWORD *n; // rbx
  __int64 v74; // rax
  __int64 v75; // r13
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // r12
  __int64 v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rcx
  _BYTE *v82; // rsi
  _QWORD *v83; // r14
  __int64 v84; // rdi
  __int64 v85; // rax
  __int64 v86; // rdx
  unsigned __int8 *v87; // r10
  unsigned __int8 *v88; // rax
  unsigned __int8 *v89; // r10
  __int64 v90; // r14
  __int64 v91; // rax
  __int64 v92; // r14
  _QWORD *v93; // r15
  __int64 v94; // r12
  _QWORD *v95; // r13
  __int64 v96; // rsi
  __int64 ii; // r14
  __int64 v98; // rax
  __int64 v99; // r13
  __int64 v100; // rax
  __int64 v101; // rcx
  __int64 v102; // rax
  __int64 v103; // r13
  __int64 v104; // rax
  int v105; // r13d
  _QWORD *v106; // rax
  _QWORD *v107; // rsi
  unsigned __int64 *v108; // rsi
  unsigned __int64 v109; // rdi
  unsigned __int64 k; // rdx
  unsigned int v111; // edx
  unsigned __int64 *v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rax
  char v116; // dl
  __int64 v117; // rax
  __int64 v118; // [rsp+0h] [rbp-1A0h]
  unsigned __int8 *v119; // [rsp+10h] [rbp-190h]
  void *v120; // [rsp+10h] [rbp-190h]
  _QWORD *v121; // [rsp+18h] [rbp-188h]
  __int64 *v122; // [rsp+20h] [rbp-180h]
  __int64 v123; // [rsp+28h] [rbp-178h]
  __int64 v124; // [rsp+30h] [rbp-170h]
  _QWORD *v125; // [rsp+38h] [rbp-168h]
  __int64 v126; // [rsp+38h] [rbp-168h]
  __int64 v127; // [rsp+40h] [rbp-160h]
  __int64 v128; // [rsp+40h] [rbp-160h]
  __int64 v129; // [rsp+40h] [rbp-160h]
  __int64 *v130; // [rsp+48h] [rbp-158h]
  unsigned int v131; // [rsp+48h] [rbp-158h]
  __int64 v132; // [rsp+48h] [rbp-158h]
  unsigned int v133; // [rsp+48h] [rbp-158h]
  __int64 v134; // [rsp+50h] [rbp-150h]
  char v135; // [rsp+58h] [rbp-148h]
  __int64 **v137; // [rsp+68h] [rbp-138h]
  unsigned int v138; // [rsp+68h] [rbp-138h]
  __int64 v139; // [rsp+68h] [rbp-138h]
  _QWORD *v140; // [rsp+68h] [rbp-138h]
  unsigned int v141; // [rsp+68h] [rbp-138h]
  unsigned int v143; // [rsp+78h] [rbp-128h]
  unsigned int v144; // [rsp+78h] [rbp-128h]
  __int64 v145; // [rsp+78h] [rbp-128h]
  unsigned int v146; // [rsp+78h] [rbp-128h]
  __int64 v147; // [rsp+80h] [rbp-120h] BYREF
  __int64 v148; // [rsp+88h] [rbp-118h] BYREF
  _BYTE v149[16]; // [rsp+90h] [rbp-110h] BYREF
  _BYTE v150[48]; // [rsp+A0h] [rbp-100h] BYREF
  _BYTE v151[208]; // [rsp+D0h] [rbp-D0h] BYREF

  v5 = *(_QWORD *)&dword_4F063F8;
  v135 = a3;
  v147 = *(_QWORD *)&dword_4F063F8;
  if ( word_4F06418[0] != 73 )
    v5 = *(_QWORD *)&dword_4F077C8;
  v148 = v5;
  v134 = 0;
  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    v134 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  if ( *(_BYTE *)(a1 + 172) == 1 )
    *(_BYTE *)(a1 + 172) = 0;
  if ( (*(_BYTE *)(a1 + 195) & 1) != 0 )
    sub_894C00(*(_QWORD *)a1, a2, a3, word_4F06418, a5);
  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v122 = (__int64 *)(a1 + 64);
  sub_71CA50(i, (_DWORD *)(a1 + 64), 0, 0, 0, a1);
  for ( j = *(_QWORD *)(a1 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v127 = *(_QWORD *)(j + 168);
  v124 = v135 & 8;
  if ( (v135 & 8) != 0 )
  {
    v8 = 0xFFFFFFFFLL;
    v123 = 0;
    v9 = ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) >> 1) ^ 1) & 1;
    goto LABEL_15;
  }
  if ( dword_4F077C4 != 2 )
  {
LABEL_138:
    v123 = 0;
    goto LABEL_139;
  }
  if ( v134 )
  {
    v123 = 0;
    if ( (v135 & 2) == 0 )
    {
      v105 = unk_4D04238;
      sub_865D70(v134, 0, (*(_BYTE *)(a1 + 195) & 2) != 0, 1, 1, 0);
      unk_4D04238 = v105;
    }
    goto LABEL_139;
  }
  v114 = *(_QWORD *)(a1 + 40);
  v123 = v114;
  if ( v114 )
  {
    if ( *(_BYTE *)(v114 + 28) == 3 )
    {
      v123 = *(_QWORD *)(v114 + 32);
      if ( !v123 )
        goto LABEL_139;
      v115 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      v116 = *(_BYTE *)(v115 + 4);
      if ( v116 == 7 )
        v116 = *(_BYTE *)(v115 - 772);
      if ( v116 != 9 && (!(_DWORD)qword_4F077B4 || !dword_4F077BC || (v135 & 0x20) == 0) )
      {
        v117 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
        if ( (unsigned __int8)(*(_BYTE *)(v117 + 4) - 3) > 1u || *(_QWORD *)(*(_QWORD *)(v117 + 184) + 32LL) != v123 )
        {
          sub_864230(v123, 0);
          goto LABEL_139;
        }
      }
    }
    goto LABEL_138;
  }
LABEL_139:
  v9 = 0;
  v8 = *(unsigned int *)(a2 + 40);
LABEL_15:
  v125 = (_QWORD *)sub_8600D0(17, v8, 0, a1);
  if ( (*(_BYTE *)(a1 + 193) & 4) != 0 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) |= 0x20u;
  v10 = *(_QWORD *)(a2 + 56);
  if ( v10 )
  {
    v11 = qword_4F04C10;
    v8 = *((unsigned int *)qword_4F04C10 + 2);
    v12 = v8 & (a1 >> 3);
    v13 = 16LL * v12;
    v14 = (unsigned __int64 *)(*qword_4F04C10 + v13);
    if ( *v14 )
    {
      do
      {
        v12 = v8 & (v12 + 1);
        v15 = (unsigned __int64 *)(*qword_4F04C10 + 16LL * v12);
      }
      while ( *v15 );
      v16 = v14[1];
      *v15 = *v14;
      v15[1] = v16;
      *v14 = 0;
      v17 = (unsigned __int64 *)(*v11 + v13);
      v18 = *(_QWORD *)(a2 + 56);
      *v17 = a1;
      v17[1] = v18;
      v19 = *((_DWORD *)v11 + 2);
      LODWORD(v18) = *((_DWORD *)v11 + 3) + 1;
      *((_DWORD *)v11 + 3) = v18;
      if ( 2 * (int)v18 <= v19 )
        goto LABEL_21;
      v141 = v19;
      v133 = v19 + 1;
      v60 = 2 * v19 + 1;
      v146 = 2 * v19 + 2;
      v106 = (_QWORD *)sub_823970(16LL * v146);
      v62 = v133;
      v63 = v106;
      if ( v146 )
      {
        v107 = &v106[2 * v60 + 2];
        do
        {
          if ( v106 )
            *v106 = 0;
          v106 += 2;
        }
        while ( v107 != v106 );
      }
      v65 = *v11;
      if ( v133 )
      {
        v108 = (unsigned __int64 *)*v11;
        do
        {
          v109 = *v108;
          if ( *v108 )
          {
            for ( k = v109 >> 3; ; LODWORD(k) = v111 + 1 )
            {
              v111 = v60 & k;
              v112 = &v63[2 * v111];
              if ( !*v112 )
                break;
            }
            *v112 = v109;
            v112[1] = v108[1];
          }
          v108 += 2;
        }
        while ( (unsigned __int64 *)(v65 + 16LL * v141 + 16) != v108 );
      }
    }
    else
    {
      v14[1] = v10;
      *v14 = a1;
      v58 = *((_DWORD *)v11 + 2);
      v59 = *((_DWORD *)v11 + 3) + 1;
      *((_DWORD *)v11 + 3) = v59;
      if ( 2 * v59 <= v58 )
      {
LABEL_21:
        if ( (*(_BYTE *)(a1 + 195) & 1) != 0 )
          v20 = **(_QWORD **)(a1 + 248);
        else
          v20 = *(_QWORD *)a1;
        *(_WORD *)(v20 + 82) |= 0x4010u;
        goto LABEL_24;
      }
      v138 = v58;
      v131 = v58 + 1;
      v60 = 2 * v58 + 1;
      v144 = 2 * v58 + 2;
      v61 = (_QWORD *)sub_823970(16LL * v144);
      v62 = v131;
      v63 = v61;
      if ( v144 )
      {
        v64 = &v61[2 * v60 + 2];
        do
        {
          if ( v61 )
            *v61 = 0;
          v61 += 2;
        }
        while ( v64 != v61 );
      }
      v65 = *v11;
      if ( v131 )
      {
        v66 = (unsigned __int64 *)*v11;
        do
        {
          v67 = *v66;
          if ( *v66 )
          {
            for ( m = v67 >> 3; ; LODWORD(m) = v69 + 1 )
            {
              v69 = v60 & m;
              v70 = &v63[2 * v69];
              if ( !*v70 )
                break;
            }
            *v70 = v67;
            v70[1] = v66[1];
          }
          v66 += 2;
        }
        while ( v66 != (unsigned __int64 *)(v65 + 16LL * v138 + 16) );
      }
    }
    *v11 = v63;
    *((_DWORD *)v11 + 2) = v60;
    v8 = 16LL * v62;
    sub_823A00(v65, v8);
    goto LABEL_21;
  }
LABEL_24:
  if ( (*(_BYTE *)(a1 + 195) & 8) != 0 && !unk_4D04828 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) &= ~2u;
  *(_QWORD *)(v127 + 8) = a1;
  if ( (*(_BYTE *)(v127 + 16) & 0x20) != 0 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) |= 1u;
  if ( v134 && *(_QWORD *)(v127 + 40) )
    v125[8] = sub_71B620(j);
  v21 = *(_BYTE *)(a2 + 64);
  if ( v21 < 0 )
    goto LABEL_82;
  v22 = *(_QWORD **)(a2 + 88);
  if ( v22 )
  {
    v23 = qword_4F04C68[0];
    while ( 1 )
    {
      while ( 1 )
      {
        v24 = v22;
        v22 = (_QWORD *)*v22;
        v25 = *((_BYTE *)v24 + 16) == 0;
        *v24 = 0;
        v24[1] = 0;
        if ( v25 )
          break;
        sub_869970();
LABEL_36:
        if ( !v22 )
          goto LABEL_44;
      }
      v26 = *(_QWORD **)(a2 + 8);
      if ( !v26 )
        goto LABEL_36;
      while ( (_QWORD *)v26[6] != v24 )
      {
        v26 = (_QWORD *)*v26;
        if ( !v26 )
          goto LABEL_36;
      }
      *v24 = *(_QWORD *)(v23 + 320);
      *(_QWORD *)(v23 + 320) = v24;
      v26[6] = 0;
      if ( v9 )
        goto LABEL_36;
      v26[6] = sub_869D30();
      if ( !v22 )
      {
LABEL_44:
        *(_QWORD *)(a2 + 88) = 0;
        v21 = *(_BYTE *)(a2 + 64);
        break;
      }
    }
  }
  if ( (v21 & 1) != 0 && dword_4F077C4 != 2 && unk_4F07778 <= 202310 )
  {
    if ( dword_4F077C0 )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        v27 = qword_4F077A8 < 0x1ADB0u ? 7 : 5;
LABEL_52:
        v8 = 141;
        sub_684AC0(v27, 0x8Du);
        goto LABEL_53;
      }
    }
    else
    {
      LOBYTE(v27) = 7;
      if ( !(_DWORD)qword_4F077B4 )
        goto LABEL_52;
    }
    v27 = qword_4F077A0 < 0x1ADB0u ? 7 : 5;
    goto LABEL_52;
  }
LABEL_53:
  if ( qword_4D04900 )
  {
    v28 = *(const __m128i **)(a2 + 8);
    if ( !v28 )
      goto LABEL_76;
    v8 = v28[7].m128i_u32[0];
    if ( !(_DWORD)v8 )
    {
      v29 = *(_QWORD *)v127;
LABEL_60:
      if ( v9 )
      {
        v130 = (__int64 *)v28;
        v30 = *(__int64 ***)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 112LL);
        v137 = v30;
        if ( !v30 )
          goto LABEL_68;
        if ( *((_BYTE *)v30 + 140) != 12 )
        {
LABEL_65:
          v137 = (__int64 **)*v137[21];
          v31 = 0;
          if ( !v137 )
            v31 = v28;
          v130 = (__int64 *)v31;
LABEL_68:
          v32 = v28[2].m128i_i8[10];
          if ( v29 || (v32 & 1) != 0 )
          {
            while ( 1 )
            {
              while ( (v32 & 1) == 0 )
              {
                if ( !v29 )
                  goto LABEL_76;
LABEL_125:
                if ( v137 )
                {
                  v8 = (__int64)v137[2];
                  v137 = (__int64 **)*v137;
                }
                else
                {
                  v8 = v130[3];
                  v130 = (__int64 *)*v130;
                }
                sub_71B760(v28, v8, v29, v124 != 0);
                v53 = *(_QWORD *)v29;
                if ( (*(_BYTE *)(a1 + 195) & 3) != 1
                  || !v53
                  || (v54 = *(_DWORD *)(v53 + 36), *(_DWORD *)(v29 + 36) != v54)
                  || !v54
                  || (v32 = v28[2].m128i_i8[10], (v32 & 2) != 0) )
                {
                  v28 = (const __m128i *)v28->m128i_i64[0];
                  if ( !v28 )
                    goto LABEL_76;
                  v32 = v28[2].m128i_i8[10];
                }
                v29 = *(_QWORD *)v29;
              }
              if ( v29 && v28[7].m128i_i32[2] >= *(_DWORD *)(v29 + 36) )
                goto LABEL_125;
              v33 = v28[1].m128i_i64[0];
              v34 = dword_4F04C3C;
              dword_4F04C3C = 1;
              v143 = v34;
              v35 = sub_724EF0(v33);
              v36 = dword_4D03B80;
              *(_QWORD *)(v35 + 8) = dword_4D03B80;
              *(_QWORD *)(v35 + 16) = v36;
              v8 = v28[1].m128i_i64[0];
              sub_71B760(v28, v8, v35, v124 != 0);
              sub_724F80(v35);
              v28 = (const __m128i *)v28->m128i_i64[0];
              dword_4F04C3C = v143;
              if ( !v28 )
                goto LABEL_76;
              v32 = v28[2].m128i_i8[10];
            }
          }
          goto LABEL_76;
        }
        do
          v30 = (__int64 **)v30[20];
        while ( *((_BYTE *)v30 + 140) == 12 );
      }
      else
      {
        v130 = (__int64 *)v28;
        v30 = *(__int64 ***)(a2 + 80);
        v137 = v30;
        if ( !v30 )
          goto LABEL_68;
        if ( *((_BYTE *)v30 + 140) != 12 )
          goto LABEL_65;
        do
          v30 = (__int64 **)v30[20];
        while ( *((_BYTE *)v30 + 140) == 12 );
      }
      v137 = v30;
      goto LABEL_65;
    }
    do
    {
      if ( v28[2].m128i_i8[9] >= 0 )
      {
        v8 = v28->m128i_i64[1];
        sub_8756F0(1, v8, &v28[7], 0);
      }
      v28 = (const __m128i *)v28->m128i_i64[0];
    }
    while ( v28 );
  }
  v28 = *(const __m128i **)(a2 + 8);
  v29 = *(_QWORD *)v127;
  if ( v28 )
    goto LABEL_60;
LABEL_76:
  if ( dword_4D047EC && dword_4F077C4 != 2 )
  {
    v71 = *(_QWORD **)(a2 + 48);
    if ( v71 )
    {
      do
      {
        while ( v71[1] )
        {
          v71 = (_QWORD *)*v71;
          if ( !v71 )
            goto LABEL_188;
        }
        *(_QWORD *)(v71[2] + 56LL) = *(_QWORD *)(v71[3] + 88LL);
        v71 = (_QWORD *)*v71;
      }
      while ( v71 );
LABEL_188:
      v71 = *(_QWORD **)(a2 + 48);
      if ( v71 )
      {
        do
        {
          if ( v71[1] )
          {
            v8 = sub_73B8B0(v71[2], 0);
            sub_733470(v71[1], v8, 1, v71 + 4);
          }
          v71 = (_QWORD *)*v71;
        }
        while ( v71 );
        v71 = *(_QWORD **)(a2 + 48);
      }
    }
    sub_87E1F0(v71);
    *(_QWORD *)(a2 + 48) = 0;
    v72 = *(__int64 **)(a2 + 8);
    for ( n = *(_QWORD **)v127; v72; n = (_QWORD *)*n )
    {
      if ( (unsigned int)sub_8DCFE0(v72[3]) )
      {
        v8 = (__int64)(v72 + 4);
        sub_6851C0(0x37Au, (_DWORD *)v72 + 8);
        v74 = sub_72C930(890);
        n[1] = v74;
        v72[2] = v74;
      }
      else if ( (unsigned int)sub_8DD010(n[1]) )
      {
        n[1] = sub_8E3390(n[1]);
      }
      v72 = (__int64 *)*v72;
    }
  }
  if ( (*(_BYTE *)(a1 + 193) & 2) != 0 )
  {
    v8 = a1 + 64;
    if ( !(unsigned int)sub_64A440(a1, (__int64)v122) )
      *(_BYTE *)(a1 + 193) &= ~2u;
  }
  if ( (v135 & 8) == 0 && *(_QWORD *)a2 )
    sub_886000(*(_QWORD *)a2);
LABEL_82:
  if ( (v135 & 0x10) != 0 )
  {
    v8 = (__int64)v149;
    sub_7A74B0(*(unsigned int *)(a2 + 96), v149);
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 10) |= 2u;
  }
  v37 = v135 & 4;
  if ( (v135 & 4) != 0 )
    sub_86F800(v150);
  sub_7B80F0();
  v39 = 0;
  if ( word_4F06418[0] == 163 && !*(_QWORD *)(a2 + 56) )
  {
    sub_875110();
    v39 = 1;
  }
  v40 = *(_BYTE *)(a1 + 174);
  if ( v40 == 1 )
  {
    v8 = 1;
    v125[6] = sub_63CAE0(a1, 1, 0, v38);
  }
  else if ( v40 == 2 )
  {
    v125[6] = sub_63FE70(a1);
  }
  if ( v39 )
  {
    v41 = (v135 & 1) == 0;
    v42 = sub_875270(v41);
  }
  else
  {
    v8 = (v135 & 1) == 0;
    v41 = 1;
    v42 = sub_86FD00(1, v8, 0, 0, 0, 0);
  }
  v125[10] = v42;
  if ( (*(_BYTE *)(a1 + 193) & 2) != 0
    && (v43 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 13) & 0x10) == 0) )
  {
    v41 = (__int64)v125;
    sub_71B520((__int64)v125);
    if ( (v135 & 0x10) == 0 )
      goto LABEL_95;
  }
  else if ( (v135 & 0x10) == 0 )
  {
    goto LABEL_95;
  }
  v41 = (__int64)v149;
  sub_7A7500(v149);
LABEL_95:
  v45 = *(_BYTE *)(a1 + 207);
  if ( v45 < 0 )
  {
    v41 = a1;
    if ( (*(_BYTE *)(sub_71DF80(a1) + 120) & 1) == 0 && (*(_BYTE *)(a1 + 195) & 8) == 0 )
    {
      v75 = sub_72B840(a1);
      v76 = *(_QWORD *)(v75 + 80);
      if ( *(_BYTE *)(v76 + 40) == 19 )
        v76 = *(_QWORD *)(*(_QWORD *)(v76 + 72) + 8LL);
      v77 = *(_QWORD *)(v76 + 72);
      v78 = *(_QWORD *)(v77 + 72);
      v79 = *(_QWORD *)(v77 + 16);
      *(_QWORD *)(v77 + 16) = 0;
      v139 = v77;
      v145 = sub_86F990(a1, v79, v78, *(_QWORD *)(v78 + 56));
      v80 = *(_QWORD *)(v75 + 88);
      v81 = *(_QWORD *)(v80 + 48);
      *(_QWORD *)(v80 + 48) = 0;
      v132 = v81;
      v82 = v151;
      v83 = (_QWORD *)sub_72B840(a1);
      v84 = 4;
      v121 = v83 + 5;
      v128 = qword_4F06BC0;
      v126 = qword_4D03C50;
      sub_6E1E00(4u, (__int64)v151, 0, 0);
      v85 = qword_4D03C50;
      *(_BYTE *)(qword_4D03C50 + 21LL) |= 8u;
      v86 = v83[11];
      *(_QWORD *)(v85 + 48) = v86;
      v87 = (unsigned __int8 *)v83[8];
      qword_4F06BC0 = v86;
      *(_QWORD *)(v78 + 32) = v87;
      *(_QWORD *)(v78 + 40) = v83[5];
      if ( v87 )
      {
        v119 = v87;
        v88 = (unsigned __int8 *)sub_725D10(v87[136]);
        v89 = v119;
        v120 = v88;
        qmemcpy(v88, v89, 0x110u);
        v82 = v89;
        v84 = (__int64)v88;
        sub_71B080((__int64)v88, (__int64)v89, v122);
        v83[8] = v120;
      }
      v90 = v83[5];
      if ( v90 )
      {
        v118 = v75;
        v91 = v90;
        v92 = v78;
        v93 = v121;
        v94 = v91;
        do
        {
          v95 = (_QWORD *)sub_725D10(*(unsigned __int8 *)(v94 + 136));
          qmemcpy(v95, (const void *)v94, 0x110u);
          v82 = (_BYTE *)v94;
          v84 = (__int64)v95;
          sub_71B080((__int64)v95, v94, v122);
          *v93 = v95;
          v94 = *(_QWORD *)(v94 + 112);
          v93 = v95 + 14;
        }
        while ( v94 );
        v37 = v135 & 4;
        v78 = v92;
        v75 = v118;
      }
      sub_6E2B30(v84, (__int64)v82);
      v96 = *(_QWORD *)(v78 + 32);
      qword_4D03C50 = v126;
      qword_4F06BC0 = v128;
      v129 = *(_QWORD *)(v78 + 40);
      v41 = sub_71B300(v139, v96, v75);
      for ( ii = v129; ii; v41 = v98 )
      {
        v98 = sub_71B300(v41, ii, v75);
        ii = *(_QWORD *)(ii + 112);
      }
      v8 = *(_QWORD *)(v78 + 16);
      *(_QWORD *)(sub_71B300(v41, v8, v75) + 16) = v145;
      if ( v132 )
      {
        *(_QWORD *)(v132 + 56) = *(_QWORD *)(*(_QWORD *)(v75 + 88) + 48LL);
        *(_QWORD *)(v132 + 40) = *(_QWORD *)(*(_QWORD *)(v75 + 88) + 24LL);
        *(_QWORD *)(*(_QWORD *)(v75 + 88) + 48LL) = v132;
      }
      v43 = *(_QWORD **)(v78 + 48);
      v140 = v43;
      if ( v43 )
      {
        v41 = 7;
        v99 = *(_QWORD *)(v75 + 88);
        v100 = sub_726B30(7);
        v101 = *(_QWORD *)(v145 + 24);
        *(_QWORD *)(v145 + 16) = v100;
        *(_QWORD *)(v100 + 24) = v101;
        v102 = *(_QWORD *)(v145 + 16);
        v43 = v140;
        *(_QWORD *)(v102 + 80) = v99;
        *(_QWORD *)(v102 + 72) = v140;
        v145 = v102;
        v140[16] = v102;
      }
      v103 = *(_QWORD *)(v78 + 64);
      if ( v103 )
      {
        v41 = 0;
        v104 = sub_726B30(0);
        v43 = *(_QWORD **)(v145 + 24);
        *(_QWORD *)(v145 + 16) = v104;
        *(_QWORD *)(v104 + 24) = v43;
        *(_QWORD *)(*(_QWORD *)(v145 + 16) + 48LL) = v103;
      }
      *(_BYTE *)(v78 + 120) |= 4u;
    }
    for ( jj = *(_QWORD *)(a1 + 152); *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
      ;
    if ( (*(_BYTE *)(*(_QWORD *)(jj + 168) + 16LL) & 1) != 0 )
    {
      v8 = a1 + 64;
      v41 = 2681;
      sub_6851C0(0xA79u, v122);
    }
    v56 = a1;
    if ( (*(_BYTE *)(a1 + 193) & 5) != 0 )
    {
      v8 = a1 + 64;
      v41 = 2978;
      sub_6851C0(0xBA2u, v122);
      v56 = a1;
    }
    v45 = *(_BYTE *)(v56 + 207);
  }
  if ( (v45 & 0x10) != 0 )
  {
    if ( (v45 & 0x20) != 0 )
    {
      for ( kk = *(_QWORD *)(a1 + 152); *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
        ;
      v41 = *(_QWORD *)(kk + 160);
      if ( !(unsigned int)sub_8D4290(v41) )
      {
        if ( (*(_BYTE *)(a1 + 195) & 0xB) != 1 && (*(_BYTE *)(a1 + 193) & 1) != 0 )
        {
          v8 = (__int64)&v147;
          v41 = 2391;
          sub_685360(0x957u, &v147, *(_QWORD *)(kk + 160));
        }
        *(_BYTE *)(a1 + 193) &= ~2u;
      }
    }
    else if ( dword_4F04C44 != -1
           || (v43 = qword_4F04C68, v113 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v113 + 6) & 6) != 0)
           || *(_BYTE *)(v113 + 4) == 12 )
    {
      *(_BYTE *)(a1 + 207) |= 0x20u;
    }
    else
    {
      v41 = a1;
      v8 = ((*(_BYTE *)(a1 + 206) >> 1) ^ 1) & 1;
      sub_695540(a1, v8, &v147);
    }
  }
  v46 = *(__int64 **)(a2 + 56);
  if ( !v46 )
  {
LABEL_105:
    sub_863FC0(v41, v8, v43, v46, v44);
    if ( !v37 )
      goto LABEL_106;
    goto LABEL_117;
  }
  for ( mm = *v46; mm; v46 = (__int64 *)v8 )
  {
    while ( 1 )
    {
      v48 = *(_BYTE *)(mm + 33);
      v8 = mm;
      mm = *(_QWORD *)mm;
      v43 = (_QWORD *)(v48 & 2);
      if ( !(_DWORD)v43 )
        break;
      *v46 = mm;
      if ( !mm )
        goto LABEL_103;
    }
  }
LABEL_103:
  if ( (*(_BYTE *)(a1 + 195) & 1) != 0 )
  {
    *(_BYTE *)(**(_QWORD **)(a1 + 248) + 83LL) &= ~0x40u;
    goto LABEL_105;
  }
  *(_BYTE *)(*(_QWORD *)a1 + 83LL) &= ~0x40u;
  sub_863FC0(v41, v8, v43, v46, v44);
  if ( !v37 )
  {
LABEL_106:
    if ( v134 )
      goto LABEL_107;
    goto LABEL_118;
  }
LABEL_117:
  sub_86F8D0(v150);
  if ( v134 )
  {
LABEL_107:
    sub_71BD50(a1);
    if ( (v135 & 0xA) == 0 )
      sub_866010(a1, v8, v49, v50, v51);
    goto LABEL_109;
  }
LABEL_118:
  if ( (v135 & 8) == 0 && v123 )
  {
    if ( dword_4F077BC )
      sub_8645D0();
    else
      sub_8642D0();
  }
LABEL_109:
  if ( !v39 && word_4F06418[0] != 74 )
  {
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    sub_7BE200(67, 3196, &v148);
  }
  sub_7B8160();
  result = &dword_4F068EC;
  if ( dword_4F068EC )
  {
    result = (unsigned int *)(*(_DWORD *)(a1 + 192) & 0x8000480);
    if ( (_DWORD)result == 128 )
    {
      result = (unsigned int *)a1;
      if ( *(char *)(a1 + 203) >= 0 )
        return (unsigned int *)sub_89A080(a1);
    }
  }
  return result;
}
