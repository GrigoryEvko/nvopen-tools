// Function: sub_199B200
// Address: 0x199b200
//
char __fastcall sub_199B200(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r13
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int16 v7; // cx
  __int64 v8; // r14
  __int64 v9; // rbx
  unsigned int v10; // r12d
  __int64 v11; // r12
  _QWORD *v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // r15
  __int64 v18; // r10
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  char v22; // al
  _QWORD *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r14
  __int64 v29; // r14
  __int64 v30; // rax
  __m128i *v31; // rax
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // r14
  __int64 v35; // r15
  _QWORD *v36; // rbx
  _QWORD *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rbx
  _QWORD *v40; // rdx
  __int64 v41; // rdx
  _QWORD *v42; // rsi
  _QWORD *v43; // rax
  _QWORD *v44; // r8
  __int64 v45; // rax
  _QWORD *v46; // rax
  _QWORD *v47; // rdi
  unsigned int v48; // r8d
  _QWORD *v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  bool v54; // zf
  _QWORD *v55; // rax
  _QWORD *v56; // rbx
  unsigned __int64 *v57; // rdi
  unsigned __int64 v58; // rdx
  _QWORD *v59; // r15
  __int64 v60; // r14
  unsigned __int64 v61; // rdi
  unsigned __int64 v62; // rdi
  _QWORD *v63; // rdx
  __int64 v64; // r8
  __int64 *v65; // rax
  __int64 v66; // rdx
  __int64 *v67; // r15
  __int64 v68; // rsi
  __int64 *v69; // rbx
  __int64 *v70; // r10
  __int64 *v71; // r9
  __int64 v72; // r12
  __int64 *v73; // rax
  __int64 *v74; // rdi
  unsigned int v75; // r11d
  __int64 *v76; // rax
  __int64 *v77; // rcx
  __int64 v78; // rsi
  unsigned __int64 v79; // rax
  unsigned __int64 v80; // rdx
  __int64 v81; // rsi
  unsigned __int64 v82; // rax
  __int64 v83; // rdx
  unsigned __int64 v84; // r15
  __int64 v85; // r12
  __int64 v86; // r13
  _QWORD *v87; // rbx
  _QWORD *v88; // rbx
  __int64 *v89; // r15
  __int64 v90; // r13
  _QWORD *v91; // r12
  unsigned __int64 v92; // rdi
  unsigned __int64 v93; // rdi
  unsigned int v95; // [rsp+8h] [rbp-E8h]
  __int64 v96; // [rsp+8h] [rbp-E8h]
  __int64 *v98; // [rsp+18h] [rbp-D8h]
  __int64 v99; // [rsp+18h] [rbp-D8h]
  __int64 v101; // [rsp+28h] [rbp-C8h]
  __int64 v102; // [rsp+28h] [rbp-C8h]
  char v103; // [rsp+28h] [rbp-C8h]
  _QWORD *v104; // [rsp+28h] [rbp-C8h]
  __int64 v105; // [rsp+28h] [rbp-C8h]
  __int64 v106; // [rsp+28h] [rbp-C8h]
  __int64 v107; // [rsp+28h] [rbp-C8h]
  __int64 v108; // [rsp+28h] [rbp-C8h]
  __int64 v109; // [rsp+30h] [rbp-C0h]
  __int64 v110; // [rsp+30h] [rbp-C0h]
  __int64 v111; // [rsp+30h] [rbp-C0h]
  __int64 *v113; // [rsp+40h] [rbp-B0h]
  __int64 v114; // [rsp+40h] [rbp-B0h]
  int v115; // [rsp+40h] [rbp-B0h]
  __int64 v116; // [rsp+48h] [rbp-A8h]
  __int64 v117; // [rsp+48h] [rbp-A8h]
  unsigned int v118; // [rsp+48h] [rbp-A8h]
  __m128i v119; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v120; // [rsp+60h] [rbp-90h] BYREF
  __int64 v121; // [rsp+68h] [rbp-88h]
  __int64 v122; // [rsp+70h] [rbp-80h]
  _QWORD v123[15]; // [rsp+78h] [rbp-78h] BYREF

  v4 = a1;
  v113 = (__int64 *)a3;
  if ( *(_BYTE *)(a3 + 16) == 60 )
    v113 = *(__int64 **)(a3 - 24);
  v109 = sub_146F1B0(a1[1], (__int64)v113);
  v7 = *(_WORD *)(v109 + 24);
  v8 = v109;
LABEL_4:
  while ( 2 )
  {
    switch ( v7 )
    {
      case 0u:
        v8 = 0;
        goto LABEL_6;
      case 1u:
      case 2u:
      case 3u:
        v8 = *(_QWORD *)(v8 + 32);
        v7 = *(_WORD *)(v8 + 24);
        if ( v7 <= 7u )
          continue;
        goto LABEL_6;
      case 4u:
        v5 = *(_QWORD *)(v8 + 32);
        v25 = v5 + 8LL * *(_QWORD *)(v8 + 40);
        if ( v5 == v25 )
          goto LABEL_6;
        break;
      case 7u:
        v27 = *(__int64 **)(v8 + 32);
        v8 = *v27;
        v7 = *(_WORD *)(*v27 + 24);
        continue;
      default:
        goto LABEL_6;
    }
    break;
  }
  while ( 1 )
  {
    v26 = *(unsigned __int16 *)(*(_QWORD *)(v25 - 8) + 24LL);
    if ( v26 == 4 )
    {
      v8 = *(_QWORD *)(v25 - 8);
      goto LABEL_4;
    }
    if ( v26 != 5 )
      break;
    v25 -= 8;
    if ( v5 == v25 )
      goto LABEL_6;
  }
  v8 = *(_QWORD *)(v25 - 8);
LABEL_6:
  v95 = *((_DWORD *)a1 + 8078);
  if ( v95 )
  {
    v116 = *((unsigned int *)a1 + 8078);
    v9 = 0;
    do
    {
      v11 = 48 * v9;
      v6 = 48 * v9 + a1[4038];
      if ( *(_QWORD *)(v6 + 40) == v8 )
      {
        v12 = (_QWORD *)(*(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8) - 24);
        v13 = v12[1];
        if ( *(_BYTE *)(v13 + 16) == 60 )
          v13 = *(_QWORD *)(v13 - 24);
        v14 = *(_QWORD *)v13;
        v15 = *v113;
        if ( (*(_QWORD *)v13 == *v113
           || *(_BYTE *)(v14 + 8) == 15
           && *(_BYTE *)(v15 + 8) == 15
           && *(_DWORD *)(v14 + 8) >> 8 == *(_DWORD *)(v15 + 8) >> 8)
          && (*(_BYTE *)(a2 + 16) != 77 || *(_BYTE *)(*v12 + 16LL) != 77) )
        {
          v101 = 48 * v9 + a1[4038];
          v16 = sub_146F1B0(a1[1], v13);
          v17 = sub_14806B0(a1[1], v109, v16, 0, 0);
          if ( sub_146CEE0(a1[1], v17, a1[5]) )
          {
            v18 = a1[1];
            if ( !*(_WORD *)(v17 + 24) )
              goto LABEL_22;
            v19 = *(_QWORD *)(*(_QWORD *)v101 + 8LL);
            if ( *(_BYTE *)(v19 + 16) == 60 )
              v19 = *(_QWORD *)(v19 - 24);
            v102 = a1[1];
            v20 = sub_146F1B0(v102, v19);
            v21 = sub_14806B0(v102, v109, v20, 0, 0);
            v18 = v102;
            if ( *(_WORD *)(v21 + 24) )
            {
LABEL_22:
              v119.m128i_i64[0] = 0;
              v119.m128i_i64[1] = (__int64)v123;
              v120 = v123;
              v121 = 8;
              LODWORD(v122) = 0;
              v22 = sub_199B0A0(v17, (__int64)&v119, v18);
              if ( v120 != (_QWORD *)v119.m128i_i64[1] )
              {
                v103 = v22;
                _libc_free((unsigned __int64)v120);
                v22 = v103;
              }
              if ( !v22 )
                goto LABEL_42;
            }
          }
        }
      }
      v10 = ++v9;
    }
    while ( v116 != v9 );
    LOBYTE(v23) = v95 > 7;
    if ( v10 != v95 )
    {
      v9 = v10;
      v17 = 0;
      v11 = 48LL * v10;
LABEL_42:
      v28 = a1[4038];
      v120 = (_QWORD *)v17;
      v119.m128i_i64[0] = a2;
      v29 = v11 + v28;
      v119.m128i_i64[1] = a3;
      v30 = *(unsigned int *)(v29 + 8);
      if ( (unsigned int)v30 >= *(_DWORD *)(v29 + 12) )
      {
        sub_16CD150(v29, (const void *)(v29 + 16), 0, 24, v5, v6);
        v30 = *(unsigned int *)(v29 + 8);
      }
      v31 = (__m128i *)(*(_QWORD *)v29 + 24 * v30);
      *v31 = _mm_loadu_si128(&v119);
      v31[1].m128i_i64[0] = (__int64)v120;
      ++*(_DWORD *)(v29 + 8);
      v32 = *(_QWORD *)a4;
      goto LABEL_45;
    }
  }
  else
  {
    v10 = 0;
    LOBYTE(v23) = 0;
  }
  v24 = a2;
  if ( *(_BYTE *)(a2 + 16) == 77 )
    return (char)v23;
  if ( (_BYTE)v23 )
    return (char)v23;
  LOBYTE(v23) = v109;
  if ( *(_WORD *)(v109 + 24) != 7 )
    return (char)v23;
  v123[0] = v8;
  v52 = *((unsigned int *)a1 + 8078);
  v118 = v10 + 1;
  v119.m128i_i64[1] = 0x100000001LL;
  v119.m128i_i64[0] = (__int64)&v120;
  v121 = a3;
  v120 = (_QWORD *)a2;
  v122 = v109;
  if ( (unsigned int)v52 >= *((_DWORD *)a1 + 8079) )
  {
    sub_1995CB0((__int64)(a1 + 4038), 0);
    v52 = *((unsigned int *)a1 + 8078);
  }
  v53 = 48LL * (unsigned int)v52;
  v54 = a1[4038] + v53 == 0;
  v55 = (_QWORD *)(a1[4038] + v53);
  v56 = v55;
  if ( !v54 )
  {
    *v55 = v55 + 2;
    v55[1] = 0x100000000LL;
    if ( v119.m128i_i32[2] )
      sub_19938C0((__int64)v55, (char **)&v119, v52, v24, v5, v6);
    v56[5] = v123[0];
    LODWORD(v52) = *((_DWORD *)a1 + 8078);
  }
  v57 = (unsigned __int64 *)v119.m128i_i64[0];
  *((_DWORD *)v4 + 8078) = v52 + 1;
  if ( v57 != (unsigned __int64 *)&v120 )
    _libc_free((unsigned __int64)v57);
  v9 = v10;
  v58 = *(unsigned int *)(a4 + 8);
  v11 = 48LL * v10;
  if ( v118 < v58 )
  {
    v32 = *(_QWORD *)a4;
    v59 = (_QWORD *)(*(_QWORD *)a4 + 144 * v58);
    v60 = *(_QWORD *)a4 + 144LL * v118;
    if ( v59 == (_QWORD *)v60 )
    {
LABEL_103:
      v17 = v109;
      *(_DWORD *)(a4 + 8) = v118;
      goto LABEL_45;
    }
    do
    {
      v59 -= 18;
      v61 = v59[11];
      if ( v61 != v59[10] )
        _libc_free(v61);
      v62 = v59[2];
      if ( v62 != v59[1] )
        _libc_free(v62);
    }
    while ( (_QWORD *)v60 != v59 );
LABEL_102:
    v32 = *(_QWORD *)a4;
    goto LABEL_103;
  }
  if ( v118 > v58 )
  {
    v79 = *(unsigned int *)(a4 + 12);
    if ( v118 > v79 )
    {
      v82 = ((((v79 + 2) | ((v79 + 2) >> 1)) >> 2) | (v79 + 2) | ((v79 + 2) >> 1)) + 1;
      if ( v118 > v82 )
      {
        v115 = v118;
        v82 = v118;
      }
      else
      {
        v115 = v82;
      }
      v105 = *(unsigned int *)(a4 + 8);
      v32 = malloc(144 * v82);
      v83 = v105;
      if ( !v32 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v83 = *(unsigned int *)(a4 + 8);
        v32 = 0;
      }
      v84 = *(_QWORD *)a4 + 144 * v83;
      if ( *(_QWORD *)a4 != v84 )
      {
        v106 = v11;
        v85 = *(_QWORD *)a4;
        v98 = v4;
        v86 = v9;
        v87 = (_QWORD *)v32;
        do
        {
          if ( v87 )
          {
            v96 = v32;
            sub_16CCEE0(v87, (__int64)(v87 + 5), 4, v85);
            sub_16CCEE0(v87 + 9, (__int64)(v87 + 14), 4, v85 + 72);
            v32 = v96;
          }
          v85 += 144;
          v87 += 18;
        }
        while ( v84 != v85 );
        v9 = v86;
        v11 = v106;
        v4 = v98;
        v84 = *(_QWORD *)a4 + 144LL * *(unsigned int *)(a4 + 8);
        if ( v84 != *(_QWORD *)a4 )
        {
          v107 = v9;
          v88 = (_QWORD *)(*(_QWORD *)a4 + 144LL * *(unsigned int *)(a4 + 8));
          v89 = v98;
          v90 = v32;
          v99 = v11;
          v91 = *(_QWORD **)a4;
          do
          {
            v88 -= 18;
            v92 = v88[11];
            if ( v92 != v88[10] )
              _libc_free(v92);
            v93 = v88[2];
            if ( v93 != v88[1] )
              _libc_free(v93);
          }
          while ( v88 != v91 );
          v32 = v90;
          v9 = v107;
          v4 = v89;
          v11 = v99;
          v84 = *(_QWORD *)a4;
        }
      }
      if ( v84 != a4 + 16 )
      {
        v108 = v32;
        _libc_free(v84);
        v32 = v108;
      }
      *(_QWORD *)a4 = v32;
      v58 = *(unsigned int *)(a4 + 8);
      *(_DWORD *)(a4 + 12) = v115;
    }
    else
    {
      v32 = *(_QWORD *)a4;
    }
    v80 = v32 + 144 * v58;
    v81 = v32 + 144LL * v118;
    if ( v80 == v81 )
      goto LABEL_103;
    do
    {
      if ( v80 )
      {
        memset((void *)v80, 0, 0x90u);
        *(_DWORD *)(v80 + 24) = 4;
        *(_QWORD *)(v80 + 8) = v80 + 40;
        *(_QWORD *)(v80 + 16) = v80 + 40;
        *(_QWORD *)(v80 + 80) = v80 + 112;
        *(_QWORD *)(v80 + 88) = v80 + 112;
        *(_DWORD *)(v80 + 96) = 4;
      }
      v80 += 144LL;
    }
    while ( v81 != v80 );
    goto LABEL_102;
  }
  v17 = v109;
  v32 = *(_QWORD *)a4;
LABEL_45:
  v33 = v4[4038] + v11;
  v117 = 144 * v9;
  v34 = v32 + 144 * v9;
  if ( !sub_14560B0(v17) )
  {
    v114 = v34 + 72;
    v64 = *(_QWORD *)a4 + v117;
    v65 = *(__int64 **)(v34 + 88);
    if ( v65 == *(__int64 **)(v34 + 80) )
      v66 = *(unsigned int *)(v34 + 100);
    else
      v66 = *(unsigned int *)(v34 + 96);
    v67 = &v65[v66];
    if ( v65 == v67 )
      goto LABEL_120;
    while ( 1 )
    {
      v68 = *v65;
      v69 = v65;
      if ( (unsigned __int64)*v65 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v67 == ++v65 )
        goto LABEL_120;
    }
    if ( v65 == v67 )
    {
LABEL_120:
      sub_18CE100(v114);
    }
    else
    {
      v70 = *(__int64 **)(v64 + 16);
      v71 = *(__int64 **)(v64 + 8);
      v111 = v33;
      v72 = *(_QWORD *)a4 + v117;
      if ( v71 == v70 )
        goto LABEL_130;
LABEL_123:
      sub_16CCBA0(v72, v68);
      v70 = *(__int64 **)(v72 + 16);
      v71 = *(__int64 **)(v72 + 8);
LABEL_124:
      while ( 1 )
      {
        v73 = v69 + 1;
        if ( v69 + 1 == v67 )
          break;
        while ( 1 )
        {
          v68 = *v73;
          v69 = v73;
          if ( (unsigned __int64)*v73 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v67 == ++v73 )
            goto LABEL_127;
        }
        if ( v73 == v67 )
          break;
        if ( v71 != v70 )
          goto LABEL_123;
LABEL_130:
        v74 = &v71[*(unsigned int *)(v72 + 28)];
        v75 = *(_DWORD *)(v72 + 28);
        if ( v74 == v71 )
        {
LABEL_140:
          if ( v75 >= *(_DWORD *)(v72 + 24) )
            goto LABEL_123;
          *(_DWORD *)(v72 + 28) = v75 + 1;
          *v74 = v68;
          v71 = *(__int64 **)(v72 + 8);
          ++*(_QWORD *)v72;
          v70 = *(__int64 **)(v72 + 16);
        }
        else
        {
          v76 = v71;
          v77 = 0;
          while ( *v76 != v68 )
          {
            if ( *v76 == -2 )
              v77 = v76;
            if ( v74 == ++v76 )
            {
              if ( !v77 )
                goto LABEL_140;
              *v77 = v68;
              v70 = *(__int64 **)(v72 + 16);
              --*(_DWORD *)(v72 + 32);
              v71 = *(__int64 **)(v72 + 8);
              ++*(_QWORD *)v72;
              goto LABEL_124;
            }
          }
        }
      }
LABEL_127:
      v33 = v111;
      sub_18CE100(v114);
    }
  }
  v35 = *(_QWORD *)(a3 + 8);
  if ( v35 )
  {
    while ( 1 )
    {
LABEL_47:
      v36 = sub_1648700(v35);
      if ( *((_BYTE *)v36 + 16) <= 0x17u )
        goto LABEL_52;
      v37 = *(_QWORD **)v33;
      v38 = *(_QWORD *)v33 + 24LL * *(unsigned int *)(v33 + 8);
      if ( v38 != *(_QWORD *)v33 )
      {
        while ( (_QWORD *)*v37 != v36 )
        {
          v37 += 3;
          if ( (_QWORD *)v38 == v37 )
            goto LABEL_59;
        }
        goto LABEL_52;
      }
LABEL_59:
      if ( sub_1456C80(v4[1], *v36) && *(_WORD *)(sub_146F1B0(v4[1], (__int64)v36) + 24) != 10 )
        break;
LABEL_66:
      v46 = *(_QWORD **)(v34 + 80);
      if ( *(_QWORD **)(v34 + 88) == v46 )
      {
        v47 = &v46[*(unsigned int *)(v34 + 100)];
        v48 = *(_DWORD *)(v34 + 100);
        if ( v46 == v47 )
        {
LABEL_113:
          if ( v48 >= *(_DWORD *)(v34 + 96) )
            goto LABEL_67;
          *(_DWORD *)(v34 + 100) = v48 + 1;
          *v47 = v36;
          ++*(_QWORD *)(v34 + 72);
        }
        else
        {
          v49 = 0;
          while ( v36 != (_QWORD *)*v46 )
          {
            if ( *v46 == -2 )
              v49 = v46;
            if ( v47 == ++v46 )
            {
              if ( !v49 )
                goto LABEL_113;
              *v49 = v36;
              --*(_DWORD *)(v34 + 104);
              ++*(_QWORD *)(v34 + 72);
              v35 = *(_QWORD *)(v35 + 8);
              if ( v35 )
                goto LABEL_47;
              goto LABEL_53;
            }
          }
        }
LABEL_52:
        v35 = *(_QWORD *)(v35 + 8);
        if ( !v35 )
          goto LABEL_53;
      }
      else
      {
LABEL_67:
        sub_16CCBA0(v34 + 72, (__int64)v36);
        v35 = *(_QWORD *)(v35 + 8);
        if ( !v35 )
          goto LABEL_53;
      }
    }
    v41 = *v4;
    v42 = *(_QWORD **)(*v4 + 56);
    v43 = *(_QWORD **)(*v4 + 48);
    if ( v42 == v43 )
    {
      v63 = &v43[*(unsigned int *)(v41 + 68)];
      if ( v43 == v63 )
      {
        v44 = *(_QWORD **)(*v4 + 48);
      }
      else
      {
        do
        {
          if ( v36 == (_QWORD *)*v43 )
            break;
          ++v43;
        }
        while ( v63 != v43 );
        v44 = v63;
      }
    }
    else
    {
      v110 = *v4;
      v104 = &v42[*(unsigned int *)(v41 + 64)];
      v43 = sub_16CC9F0(v41 + 40, (__int64)v36);
      v44 = v104;
      if ( v36 == (_QWORD *)*v43 )
      {
        v78 = *(_QWORD *)(v110 + 56);
        if ( v78 == *(_QWORD *)(v110 + 48) )
          v63 = (_QWORD *)(v78 + 8LL * *(unsigned int *)(v110 + 68));
        else
          v63 = (_QWORD *)(v78 + 8LL * *(unsigned int *)(v110 + 64));
      }
      else
      {
        v45 = *(_QWORD *)(v110 + 56);
        if ( v45 != *(_QWORD *)(v110 + 48) )
        {
          v43 = (_QWORD *)(v45 + 8LL * *(unsigned int *)(v110 + 64));
          goto LABEL_65;
        }
        v43 = (_QWORD *)(v45 + 8LL * *(unsigned int *)(v110 + 68));
        v63 = v43;
      }
    }
    while ( v63 != v43 && *v43 >= 0xFFFFFFFFFFFFFFFELL )
      ++v43;
LABEL_65:
    if ( v43 != v44 )
      goto LABEL_52;
    goto LABEL_66;
  }
LABEL_53:
  v39 = *(_QWORD *)a4 + v117;
  v23 = *(_QWORD **)(v39 + 8);
  if ( *(_QWORD **)(v39 + 16) == v23 )
  {
    v40 = &v23[*(unsigned int *)(v39 + 28)];
    if ( v23 == v40 )
    {
LABEL_138:
      v23 = v40;
    }
    else
    {
      while ( a2 != *v23 )
      {
        if ( v40 == ++v23 )
          goto LABEL_138;
      }
    }
  }
  else
  {
    v23 = sub_16CC9F0(*(_QWORD *)a4 + v117, a2);
    if ( a2 == *v23 )
    {
      v50 = *(_QWORD *)(v39 + 16);
      if ( v50 == *(_QWORD *)(v39 + 8) )
        v51 = *(unsigned int *)(v39 + 28);
      else
        v51 = *(unsigned int *)(v39 + 24);
      v40 = (_QWORD *)(v50 + 8 * v51);
    }
    else
    {
      v23 = *(_QWORD **)(v39 + 16);
      if ( v23 != *(_QWORD **)(v39 + 8) )
        return (char)v23;
      v23 += *(unsigned int *)(v39 + 28);
      v40 = v23;
    }
  }
  if ( v40 != v23 )
  {
    *v23 = -2;
    ++*(_DWORD *)(v39 + 32);
  }
  return (char)v23;
}
