// Function: sub_34BF740
// Address: 0x34bf740
//
void __fastcall sub_34BF740(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rbx
  _QWORD *v12; // rax
  _QWORD *v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // r13
  int v18; // edx
  __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // r12
  __int64 i; // r12
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rbx
  __int64 j; // rbx
  unsigned __int64 v26; // rax
  int v27; // r12d
  unsigned __int64 v28; // r15
  __int16 v29; // ax
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  __int64 v32; // rax
  __int64 k; // r15
  bool v34; // cf
  __int64 v35; // r15
  int v36; // eax
  __int64 v37; // rsi
  unsigned int v38; // ebx
  __int64 v39; // rax
  unsigned __int64 v40; // r13
  __int64 v41; // rdi
  __int16 v42; // ax
  __int64 v43; // r15
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rax
  unsigned __int8 **v47; // rbx
  __int64 v48; // rsi
  int v49; // eax
  _QWORD *v50; // rax
  _QWORD *v51; // rdx
  __int64 v52; // rax
  unsigned __int64 v53; // rax
  _QWORD *v54; // r13
  int v55; // eax
  __int64 v56; // rax
  __int64 v57; // rsi
  _BYTE *v58; // rdx
  char v59; // cl
  _QWORD *v60; // rax
  _QWORD *v61; // rdx
  __int64 v62; // rax
  __int64 m; // r15
  unsigned __int64 v64; // rdx
  __int64 n; // rbx
  int v66; // eax
  __int64 v67; // rax
  __int64 v68; // rsi
  unsigned __int8 *v69; // rsi
  __int64 v70; // rax
  unsigned int v71; // ebx
  unsigned __int8 *v72; // rax
  __int64 v73; // r14
  __int64 v74; // rax
  unsigned int v75; // ebx
  __int64 v76; // rax
  unsigned __int64 v77; // rdi
  _QWORD *v78; // r15
  __int64 *v79; // rax
  __int64 v80; // r13
  unsigned __int64 v81; // rax
  unsigned __int16 *v82; // rbx
  __int64 *v83; // rsi
  _QWORD *v84; // r13
  __int64 v85; // rax
  __int64 v86; // rsi
  _QWORD *v87; // r15
  _QWORD *v88; // rax
  __int64 v89; // r9
  __int64 v90; // r9
  __int64 v91; // rax
  __int64 v92; // r12
  __int16 *v93; // rax
  __int16 *v94; // rsi
  int v95; // eax
  int v96; // r9d
  unsigned __int16 ii; // cx
  unsigned int v98; // edi
  unsigned __int16 *v99; // r8
  int v100; // ecx
  __int64 *v101; // [rsp+10h] [rbp-130h]
  __int64 *v102; // [rsp+18h] [rbp-128h]
  __int64 *v103; // [rsp+20h] [rbp-120h]
  _QWORD *v104; // [rsp+28h] [rbp-118h]
  __int64 v105; // [rsp+30h] [rbp-110h]
  __int64 v107; // [rsp+38h] [rbp-108h]
  __int64 *v108; // [rsp+38h] [rbp-108h]
  __int64 v109; // [rsp+40h] [rbp-100h]
  __int64 v110; // [rsp+48h] [rbp-F8h]
  __int64 v111; // [rsp+48h] [rbp-F8h]
  __int64 v112; // [rsp+50h] [rbp-F0h]
  __int64 v113; // [rsp+50h] [rbp-F0h]
  __int64 v114; // [rsp+50h] [rbp-F0h]
  _QWORD *v115; // [rsp+50h] [rbp-F0h]
  __int64 v116; // [rsp+50h] [rbp-F0h]
  unsigned __int64 v117; // [rsp+58h] [rbp-E8h]
  int v118; // [rsp+60h] [rbp-E0h]
  unsigned __int64 v119; // [rsp+60h] [rbp-E0h]
  int v120; // [rsp+68h] [rbp-D8h]
  unsigned __int16 *v121; // [rsp+68h] [rbp-D8h]
  __int64 v122; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v123; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v124; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v125; // [rsp+88h] [rbp-B8h]
  __int64 v126; // [rsp+90h] [rbp-B0h]
  __m128i v127; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v128; // [rsp+B0h] [rbp-90h]
  __int64 v129; // [rsp+B8h] [rbp-88h]
  __int64 v130; // [rsp+C0h] [rbp-80h]
  unsigned __int8 *v131; // [rsp+D0h] [rbp-70h] BYREF
  unsigned __int16 *v132; // [rsp+D8h] [rbp-68h]
  __int64 v133; // [rsp+E0h] [rbp-60h]
  __int64 v134; // [rsp+E8h] [rbp-58h]
  _BYTE v135[16]; // [rsp+F0h] [rbp-50h] BYREF
  unsigned __int8 *v136; // [rsp+100h] [rbp-40h]
  unsigned int v137; // [rsp+108h] [rbp-38h]

  v6 = a2;
  v7 = *(_QWORD *)(a1 + 104);
  v8 = *(_QWORD *)(a1 + 112);
  v120 = v6;
  v109 = *(_QWORD *)(*(_QWORD *)(v7 + 16 * v6) + 8LL);
  v9 = (v8 - v7) >> 4;
  if ( v8 - v7 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v10 = a1;
  v11 = v9;
  if ( v9 )
  {
    v12 = (_QWORD *)sub_22077B0(8 * v9);
    v13 = &v12[v11];
    v117 = (unsigned __int64)v12;
    v104 = v13;
    do
    {
      if ( v12 )
        *v12 = 0;
      ++v12;
    }
    while ( v13 != v12 );
    v7 = *(_QWORD *)(a1 + 104);
    v8 = *(_QWORD *)(a1 + 112);
  }
  else
  {
    v117 = 0;
    v104 = 0;
  }
  v14 = 0;
  v118 = 0;
  v15 = v109 + 48;
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      if ( v120 == v118 )
        goto LABEL_40;
      v16 = *(_QWORD *)(v7 + 16 * v14 + 8);
      *(_QWORD *)(v117 + 8 * v14) = v16;
      v17 = *(_QWORD *)(v16 + 24);
      v18 = 0;
      v19 = v17 + 48;
      if ( v16 != v17 + 48 )
      {
        while ( 1 )
        {
          ++v18;
          if ( !v16 )
            break;
          if ( (*(_BYTE *)v16 & 4) != 0 )
          {
            v16 = *(_QWORD *)(v16 + 8);
            if ( v19 == v16 )
              goto LABEL_16;
          }
          else
          {
            while ( (*(_BYTE *)(v16 + 44) & 8) != 0 )
              v16 = *(_QWORD *)(v16 + 8);
            v16 = *(_QWORD *)(v16 + 8);
            if ( v19 == v16 )
              goto LABEL_16;
          }
        }
LABEL_197:
        BUG();
      }
LABEL_16:
      v20 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v20 )
        goto LABEL_197;
      v21 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v20 & 4) == 0 && (*(_BYTE *)(v20 + 44) & 4) != 0 )
      {
        for ( i = *(_QWORD *)v20; ; i = *(_QWORD *)v21 )
        {
          v21 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v21 + 44) & 4) == 0 )
            break;
        }
      }
      v112 = *(_QWORD *)(v109 + 48);
      v23 = v112 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v112 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_197;
      v24 = v112 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v23 & 4) == 0 && (*(_BYTE *)(v23 + 44) & 4) != 0 )
      {
        for ( j = *(_QWORD *)v23; ; j = *(_QWORD *)v24 )
        {
          v24 = j & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v24 + 44) & 4) == 0 )
            break;
        }
      }
      if ( v18 )
        break;
LABEL_40:
      v14 = (unsigned int)++v118;
      if ( v118 == (v8 - v7) >> 4 )
      {
        v10 = a1;
        goto LABEL_42;
      }
    }
    v113 = v17;
    v26 = v21;
    v27 = v18 - 1;
    v28 = v26;
    while ( 1 )
    {
      v29 = *(_WORD *)(v28 + 68);
      if ( (unsigned __int16)(v29 - 14) <= 4u || v29 == 3 )
      {
        v30 = (_QWORD *)(*(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL);
        v31 = v30;
        if ( !v30 )
          goto LABEL_197;
        v28 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
        v32 = *v30;
        if ( (v32 & 4) == 0 && (*((_BYTE *)v31 + 44) & 4) != 0 )
        {
          for ( k = v32; ; k = *(_QWORD *)v28 )
          {
            v28 = k & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v28 + 44) & 4) == 0 )
              break;
          }
        }
        goto LABEL_38;
      }
      if ( v24 == v15 )
      {
        v49 = *(unsigned __int16 *)(v24 + 68);
        v54 = (_QWORD *)v24;
      }
      else
      {
        while ( 1 )
        {
          v49 = *(unsigned __int16 *)(v24 + 68);
          if ( (unsigned __int16)(v49 - 14) > 4u && (_WORD)v49 != 3 )
            break;
          v50 = (_QWORD *)(*(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL);
          v51 = v50;
          if ( !v50 )
            goto LABEL_197;
          v24 = *(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL;
          v52 = *v50;
          if ( (v52 & 4) == 0 && (*((_BYTE *)v51 + 44) & 4) != 0 )
          {
            while ( 1 )
            {
              v53 = v52 & 0xFFFFFFFFFFFFFFF8LL;
              v24 = v53;
              if ( (*(_BYTE *)(v53 + 44) & 4) == 0 )
                break;
              v52 = *(_QWORD *)v53;
            }
          }
          if ( v15 == v24 )
          {
            v49 = *(unsigned __int16 *)(v109 + 116);
            v54 = (_QWORD *)(v109 + 48);
            goto LABEL_102;
          }
        }
        v54 = (_QWORD *)v24;
      }
LABEL_102:
      if ( (unsigned int)(v49 - 1) > 1 || (*(_BYTE *)(*(_QWORD *)(v24 + 32) + 64LL) & 8) == 0 )
      {
        v55 = *(_DWORD *)(v24 + 44);
        if ( (v55 & 4) != 0 || (v55 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v24 + 16) + 24LL) & 0x80000LL) == 0 )
          {
LABEL_131:
            if ( (unsigned int)*(unsigned __int16 *)(v24 + 68) - 1 > 1
              || (*(_BYTE *)(*(_QWORD *)(v24 + 32) + 64LL) & 0x10) == 0 )
            {
              v66 = *(_DWORD *)(v24 + 44);
              if ( (v66 & 4) != 0 || (v66 & 8) == 0 )
                v67 = (*(_QWORD *)(*(_QWORD *)(v24 + 16) + 24LL) >> 20) & 1LL;
              else
                LOBYTE(v67) = sub_2E88A90(v24, 0x100000, 1);
              if ( !(_BYTE)v67 )
                goto LABEL_108;
            }
          }
        }
        else if ( !sub_2E88A90(v24, 0x80000, 1) )
        {
          goto LABEL_131;
        }
      }
      v131 = (unsigned __int8 *)v24;
      v132 = (unsigned __int16 *)v28;
      sub_2E87480(v24, *(_QWORD *)(v113 + 32), (__int64 *)&v131, 2, a5, a6);
LABEL_108:
      if ( (*(_DWORD *)(v24 + 40) & 0xFFFFFF) != 0 )
      {
        v56 = 0;
        v57 = 40LL * (*(_DWORD *)(v24 + 40) & 0xFFFFFF);
        do
        {
          v58 = (_BYTE *)(v56 + *(_QWORD *)(v24 + 32));
          if ( !*v58 )
          {
            v59 = v58[4];
            if ( (v59 & 1) != 0 && (*(_BYTE *)(*(_QWORD *)(v28 + 32) + v56 + 4) & 1) == 0 )
              v58[4] = v59 & 0xFE;
          }
          v56 += 40;
        }
        while ( v57 != v56 );
      }
      v60 = (_QWORD *)(*(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL);
      v61 = v60;
      if ( !v60 )
        goto LABEL_197;
      v28 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
      v62 = *v60;
      if ( (v62 & 4) == 0 && (*((_BYTE *)v61 + 44) & 4) != 0 )
      {
        for ( m = v62; ; m = *(_QWORD *)v28 )
        {
          v28 = m & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v28 + 44) & 4) == 0 )
            break;
        }
      }
      v64 = *v54 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v64 )
        goto LABEL_197;
      v24 = *v54 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v64 & 4) != 0 || (*(_BYTE *)(v64 + 44) & 4) == 0 )
      {
LABEL_38:
        v34 = v27-- == 0;
        if ( v34 )
          goto LABEL_39;
      }
      else
      {
        for ( n = *(_QWORD *)v64; ; n = *(_QWORD *)v24 )
        {
          v24 = n & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v24 + 44) & 4) == 0 )
            break;
        }
        v34 = v27-- == 0;
        if ( v34 )
        {
LABEL_39:
          v7 = *(_QWORD *)(a1 + 104);
          v8 = *(_QWORD *)(a1 + 112);
          goto LABEL_40;
        }
      }
    }
  }
LABEL_42:
  v35 = *(_QWORD *)(v109 + 56);
  v119 = (__int64)((__int64)v104 - v117) >> 3;
  v114 = v109 + 48;
  if ( v109 + 48 == v35 )
    goto LABEL_50;
  v107 = v10;
  do
  {
    while ( 1 )
    {
      v36 = *(unsigned __int16 *)(v35 + 68);
      v14 = (unsigned int)(v36 - 14);
      if ( (unsigned __int16)(v36 - 14) <= 4u || (_WORD)v36 == 3 )
        goto LABEL_47;
      v37 = *(_QWORD *)(v35 + 56);
      v127.m128i_i64[0] = v37;
      if ( v37 )
        sub_B96E90((__int64)&v127, v37, 1);
      v38 = 0;
      v39 = 0;
      if ( v119 )
      {
        v110 = v35;
        while ( v120 == v38 )
        {
LABEL_79:
          v39 = ++v38;
          if ( v38 >= v119 )
          {
            v35 = v110;
            goto LABEL_81;
          }
        }
        v40 = v117 + 8 * v39;
        v41 = *(_QWORD *)v40;
        v42 = *(_WORD *)(*(_QWORD *)v40 + 68LL);
        if ( (unsigned __int16)(v42 - 14) > 4u )
          goto LABEL_67;
        while ( 1 )
        {
          do
          {
            if ( (*(_BYTE *)v41 & 4) == 0 && (*(_BYTE *)(v41 + 44) & 8) != 0 )
            {
              do
                v41 = *(_QWORD *)(v41 + 8);
              while ( (*(_BYTE *)(v41 + 44) & 8) != 0 );
            }
            v41 = *(_QWORD *)(v41 + 8);
            *(_QWORD *)v40 = v41;
            v42 = *(_WORD *)(v41 + 68);
          }
          while ( (unsigned __int16)(v42 - 14) <= 4u );
LABEL_67:
          if ( v42 != 3 )
          {
            v43 = sub_B10CD0(v41 + 56);
            v44 = sub_B10CD0((__int64)&v127);
            v45 = sub_B026B0(v44, v43);
            sub_B10CB0(&v131, (__int64)v45);
            if ( v127.m128i_i64[0] )
              sub_B91220((__int64)&v127, v127.m128i_i64[0]);
            v127.m128i_i64[0] = (__int64)v131;
            if ( v131 )
              sub_B976B0((__int64)&v131, v131, (__int64)&v127);
            v46 = *(_QWORD *)v40;
            if ( !*(_QWORD *)v40 )
              goto LABEL_197;
            if ( (*(_BYTE *)v46 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v46 + 44) & 8) != 0 )
                v46 = *(_QWORD *)(v46 + 8);
            }
            *(_QWORD *)v40 = *(_QWORD *)(v46 + 8);
            goto LABEL_79;
          }
        }
      }
LABEL_81:
      v47 = (unsigned __int8 **)(v35 + 56);
      v131 = (unsigned __int8 *)v127.m128i_i64[0];
      if ( !v127.m128i_i64[0] )
      {
        if ( v47 == &v131 )
          goto LABEL_47;
        v68 = *(_QWORD *)(v35 + 56);
        if ( !v68 )
          goto LABEL_47;
LABEL_141:
        sub_B91220(v35 + 56, v68);
        goto LABEL_142;
      }
      sub_B96E90((__int64)&v131, v127.m128i_i64[0], 1);
      if ( v47 == &v131 )
      {
        if ( v131 )
          sub_B91220((__int64)&v131, (__int64)v131);
LABEL_85:
        v48 = v127.m128i_i64[0];
        goto LABEL_86;
      }
      v68 = *(_QWORD *)(v35 + 56);
      if ( v68 )
        goto LABEL_141;
LABEL_142:
      v69 = v131;
      *(_QWORD *)(v35 + 56) = v131;
      if ( !v69 )
        goto LABEL_85;
      sub_B976B0((__int64)&v131, v69, v35 + 56);
      v48 = v127.m128i_i64[0];
LABEL_86:
      if ( v48 )
        sub_B91220((__int64)&v127, v48);
LABEL_47:
      if ( (*(_BYTE *)v35 & 4) == 0 )
        break;
      v35 = *(_QWORD *)(v35 + 8);
      if ( v114 == v35 )
        goto LABEL_49;
    }
    while ( (*(_BYTE *)(v35 + 44) & 8) != 0 )
      v35 = *(_QWORD *)(v35 + 8);
    v35 = *(_QWORD *)(v35 + 8);
  }
  while ( v114 != v35 );
LABEL_49:
  v10 = v107;
LABEL_50:
  if ( *(_BYTE *)(v10 + 131) )
  {
    v70 = *(_QWORD *)(v10 + 152);
    v133 = 0;
    v132 = (unsigned __int16 *)v135;
    v131 = (unsigned __int8 *)v70;
    v134 = 8;
    v136 = 0;
    v137 = 0;
    v71 = *(_DWORD *)(v70 + 16);
    if ( v71 )
    {
      v72 = (unsigned __int8 *)_libc_calloc(v71, 1u);
      if ( !v72 )
        goto LABEL_196;
      v136 = v72;
      v137 = v71;
    }
    v73 = v10 + 168;
    sub_3508F40(&v131, v109, v14);
    v74 = *(_QWORD *)(v10 + 152);
    *(_QWORD *)(v10 + 184) = 0;
    *(_QWORD *)(v10 + 168) = v74;
    v75 = *(_DWORD *)(v74 + 16);
    if ( v75 < *(_DWORD *)(v10 + 224) >> 2 || v75 > *(_DWORD *)(v10 + 224) )
    {
      v76 = (__int64)_libc_calloc(v75, 1u);
      if ( !v76 && (v75 || (v76 = malloc(1u)) == 0) )
LABEL_196:
        sub_C64F00("Allocation failed", 1u);
      v77 = *(_QWORD *)(v10 + 216);
      *(_QWORD *)(v10 + 216) = v76;
      if ( v77 )
        _libc_free(v77);
      *(_DWORD *)(v10 + 224) = v75;
    }
    v78 = (_QWORD *)v10;
    v79 = *(__int64 **)(v109 + 64);
    v108 = v79;
    v101 = &v79[*(unsigned int *)(v109 + 72)];
    if ( v79 == v101 )
    {
LABEL_186:
      sub_2E330D0(v109);
      sub_35095B0(v109, &v131);
      if ( v136 )
        _libc_free((unsigned __int64)v136);
      if ( v132 != (unsigned __int16 *)v135 )
        _libc_free((unsigned __int64)v132);
      goto LABEL_51;
    }
    while ( 1 )
    {
      v80 = *v108;
      v78[23] = 0;
      v105 = v80;
      sub_3508720(v73, v80);
      v81 = sub_2E313E0(v80);
      v82 = v132;
      v83 = (__int64 *)(v80 + 40);
      v84 = v78;
      v103 = (__int64 *)v81;
      v102 = v83;
      v121 = &v132[v133];
      if ( v121 != v132 )
        break;
LABEL_185:
      if ( v101 == ++v108 )
        goto LABEL_186;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v92 = *v82;
        if ( (unsigned __int8)sub_35080D0(v73, v84[18], v92) )
          break;
LABEL_172:
        if ( v121 == ++v82 )
          goto LABEL_184;
      }
      v93 = (__int16 *)(*(_QWORD *)(v84[19] + 56LL) + 2LL * *(unsigned int *)(*(_QWORD *)(v84[19] + 8LL) + 24 * v92 + 8));
      v94 = v93 + 1;
      v95 = *v93;
      v96 = v92 + v95;
      if ( !(_WORD)v95 )
        goto LABEL_160;
      for ( ii = v92 + v95; ; ii = v96 )
      {
        v98 = v136[ii];
        if ( v98 < (unsigned int)v133 )
        {
          while ( 1 )
          {
            v99 = &v132[v98];
            if ( *v99 == ii )
              break;
            v98 += 256;
            if ( (unsigned int)v133 <= v98 )
              goto LABEL_190;
          }
          if ( &v132[v133] != v99
            && (*(_QWORD *)(*(_QWORD *)(v84[18] + 384LL) + 8 * ((unsigned __int64)ii >> 6)) & (1LL << ii)) == 0 )
          {
            break;
          }
        }
LABEL_190:
        v100 = *v94++;
        if ( !(_WORD)v100 )
          goto LABEL_160;
        v96 += v100;
      }
      if ( !v94 )
      {
LABEL_160:
        v85 = v84[17];
        v122 = 0;
        v123 = 0;
        v124 = 0;
        v86 = *(_QWORD *)(v85 + 8);
        v125 = 0;
        v126 = 0;
        v87 = *(_QWORD **)(v105 + 32);
        v127.m128i_i64[0] = 0;
        v88 = sub_2E7B380(v87, v86 - 400, (unsigned __int8 **)&v127, 0);
        v89 = (__int64)v88;
        if ( v127.m128i_i64[0] )
        {
          v115 = v88;
          sub_B91220((__int64)&v127, v127.m128i_i64[0]);
          v89 = (__int64)v115;
        }
        v111 = v89;
        sub_2E31040(v102, v89);
        v90 = v111;
        v91 = *v103;
        *(_QWORD *)(v111 + 8) = v103;
        *(_QWORD *)v111 = v91 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v111 & 7LL;
        *(_QWORD *)((v91 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v111;
        *v103 = v111 | *v103 & 7;
        if ( v125 )
        {
          sub_2E882B0(v111, (__int64)v87, v125);
          v90 = v111;
        }
        if ( v126 )
        {
          v116 = v90;
          sub_2E88680(v90, (__int64)v87, v126);
          v90 = v116;
        }
        v127.m128i_i64[0] = 0x10000000;
        v128 = 0;
        v127.m128i_i32[2] = v92;
        v129 = 0;
        v130 = 0;
        sub_2E8EAD0(v90, (__int64)v87, &v127);
        if ( v124 )
          sub_B91220((__int64)&v124, v124);
        if ( v123 )
          sub_B91220((__int64)&v123, v123);
        if ( v122 )
          sub_B91220((__int64)&v122, v122);
        goto LABEL_172;
      }
      if ( v121 == ++v82 )
      {
LABEL_184:
        v78 = v84;
        goto LABEL_185;
      }
    }
  }
LABEL_51:
  if ( v117 )
    j_j___libc_free_0(v117);
}
