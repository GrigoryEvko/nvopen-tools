// Function: sub_1885EF0
// Address: 0x1885ef0
//
void __fastcall sub_1885EF0(char *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // rdi
  __int64 v6; // rbx
  __m128i *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // r13
  __int64 *v13; // r13
  __int64 v14; // rax
  char *v15; // r13
  __int64 *v16; // rbx
  __m128i *v17; // rbx
  __m128i *v18; // r12
  __int64 v19; // rax
  __m128i *v20; // r12
  unsigned int v21; // r13d
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r9
  unsigned int v26; // eax
  char *v27; // rax
  int v28; // edx
  __int64 v29; // rax
  __m128i *v30; // r12
  unsigned int v31; // r15d
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r13
  char *v36; // rdx
  int v37; // ecx
  __m128i *v38; // rbx
  __m128i *v39; // r12
  __m128i *v40; // rbx
  __m128i *v41; // r12
  bool v42; // zf
  __int64 v43; // rax
  _QWORD *v44; // rbx
  _BYTE *v45; // r13
  _BYTE *v46; // rsi
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // r12
  void *v50; // r15
  size_t v51; // rbx
  __int64 v52; // r13
  size_t v53; // r14
  size_t v54; // rdx
  int v55; // eax
  __int64 v56; // r14
  __m128i *v57; // rsi
  __int64 i; // r13
  __m128i *v59; // [rsp+28h] [rbp-118h]
  __int64 v60; // [rsp+48h] [rbp-F8h]
  __int64 v61; // [rsp+50h] [rbp-F0h]
  __int64 v62; // [rsp+68h] [rbp-D8h]
  __int64 v63; // [rsp+68h] [rbp-D8h]
  __int64 v64; // [rsp+68h] [rbp-D8h]
  _BYTE *v65; // [rsp+68h] [rbp-D8h]
  __int64 v66; // [rsp+70h] [rbp-D0h]
  __m128i *v67; // [rsp+70h] [rbp-D0h]
  __m128i *v68; // [rsp+70h] [rbp-D0h]
  __m128i *v69; // [rsp+70h] [rbp-D0h]
  __int64 v71; // [rsp+78h] [rbp-C8h]
  char v72; // [rsp+8Eh] [rbp-B2h] BYREF
  char v73; // [rsp+8Fh] [rbp-B1h] BYREF
  __int64 v74; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v75; // [rsp+98h] [rbp-A8h] BYREF
  __m128i *v76; // [rsp+A0h] [rbp-A0h] BYREF
  __m128i *v77; // [rsp+A8h] [rbp-98h]
  __int64 v78; // [rsp+B0h] [rbp-90h]
  __m128i *p_s2; // [rsp+C0h] [rbp-80h] BYREF
  __m128i *v80; // [rsp+C8h] [rbp-78h]
  _QWORD v81[2]; // [rsp+D0h] [rbp-70h] BYREF
  void *s2; // [rsp+E0h] [rbp-60h] BYREF
  __int64 *v83; // [rsp+E8h] [rbp-58h] BYREF
  char *v84; // [rsp+F0h] [rbp-50h] BYREF
  void **v85; // [rsp+F8h] [rbp-48h]
  void **v86; // [rsp+100h] [rbp-40h]
  __int64 j; // [rsp+108h] [rbp-38h]

  v2 = (__int64)a1;
  if ( (*(unsigned __int8 (__fastcall **)(char *, const char *, _QWORD, _QWORD, __m128i **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "GlobalValueMap",
         0,
         0,
         &v76) )
  {
    v42 = (*(unsigned __int8 (__fastcall **)(char *))(*(_QWORD *)a1 + 16LL))(a1) == 0;
    v43 = *(_QWORD *)a1;
    if ( v42 )
    {
      (*(void (__fastcall **)(char *))(v43 + 104))(a1);
      (*(void (__fastcall **)(void **, char *))(*(_QWORD *)a1 + 136LL))(&s2, a1);
      v44 = s2;
      v45 = v83;
      if ( s2 != v83 )
      {
        do
        {
          v46 = (_BYTE *)*v44;
          v47 = v44[1];
          v44 += 2;
          sub_1884660((__int64)a1, v46, v47, (_QWORD *)a2);
        }
        while ( v45 != (_BYTE *)v44 );
        v45 = s2;
      }
      if ( v45 )
        j_j___libc_free_0(v45, v84 - v45);
      (*(void (__fastcall **)(char *))(*(_QWORD *)a1 + 112LL))(a1);
    }
    else
    {
      (*(void (__fastcall **)(char *))(v43 + 104))(a1);
      sub_1884ED0(a1, (char *)a2);
      (*(void (__fastcall **)(char *))(*(_QWORD *)a1 + 112LL))(a1);
    }
    (*(void (__fastcall **)(char *, __m128i *))(*(_QWORD *)a1 + 128LL))(a1, p_s2);
  }
  if ( (*(unsigned __int8 (__fastcall **)(char *, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TypeIdMap",
         0,
         0,
         &v72,
         &v74) )
  {
    v42 = (*(unsigned __int8 (__fastcall **)(char *))(*(_QWORD *)a1 + 16LL))(a1) == 0;
    v48 = *(_QWORD *)a1;
    if ( v42 )
    {
      (*(void (__fastcall **)(char *))(v48 + 104))(a1);
      (*(void (__fastcall **)(__m128i **, char *))(*(_QWORD *)a1 + 136LL))(&v76, a1);
      v59 = v77;
      if ( v76 != v77 )
      {
        v69 = v76;
        while ( 1 )
        {
          v60 = v69->m128i_i64[1];
          v65 = (_BYTE *)v69->m128i_i64[0];
          s2 = &v84;
          if ( v65 )
          {
            sub_18736F0((__int64 *)&s2, v65, (__int64)&v65[v60]);
          }
          else
          {
            v83 = 0;
            LOBYTE(v84) = 0;
          }
          v49 = *(_QWORD *)(a2 + 96);
          if ( v49 )
            break;
          v52 = a2 + 88;
LABEL_139:
          p_s2 = (__m128i *)&s2;
          v52 = sub_1880E00((_QWORD *)(a2 + 80), (_QWORD *)v52, &p_s2);
LABEL_117:
          p_s2 = (__m128i *)v81;
          if ( v65 )
          {
            sub_18736F0((__int64 *)&p_s2, v65, (__int64)&v65[v60]);
            v57 = p_s2;
          }
          else
          {
            v80 = 0;
            LOBYTE(v81[0]) = 0;
            v57 = (__m128i *)v81;
          }
          if ( (*(unsigned __int8 (__fastcall **)(char *, __m128i *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
                 a1,
                 v57,
                 1,
                 0,
                 &v73,
                 &v75) )
          {
            sub_1882690((__int64)a1, v52 + 64);
            (*(void (__fastcall **)(char *, __int64))(*(_QWORD *)a1 + 128LL))(a1, v75);
          }
          if ( p_s2 != (__m128i *)v81 )
            j_j___libc_free_0(p_s2, v81[0] + 1LL);
          if ( s2 != &v84 )
            j_j___libc_free_0(s2, v84 + 1);
          if ( v59 == ++v69 )
          {
            v2 = (__int64)a1;
            v59 = v76;
            goto LABEL_127;
          }
        }
        v50 = s2;
        v51 = (size_t)v83;
        v52 = a2 + 88;
        while ( 1 )
        {
          v53 = *(_QWORD *)(v49 + 40);
          v54 = v51;
          if ( v53 <= v51 )
            v54 = *(_QWORD *)(v49 + 40);
          if ( v54 )
          {
            v55 = memcmp(*(const void **)(v49 + 32), v50, v54);
            if ( v55 )
              goto LABEL_113;
          }
          v56 = v53 - v51;
          if ( v56 >= 0x80000000LL )
          {
LABEL_114:
            v52 = v49;
            v49 = *(_QWORD *)(v49 + 16);
            if ( !v49 )
            {
LABEL_115:
              if ( a2 + 88 != v52 && sub_1872D20(v50, v51, *(const void **)(v52 + 32), *(_QWORD *)(v52 + 40)) >= 0 )
                goto LABEL_117;
              goto LABEL_139;
            }
          }
          else
          {
            if ( v56 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
              goto LABEL_105;
            v55 = v56;
LABEL_113:
            if ( v55 >= 0 )
              goto LABEL_114;
LABEL_105:
            v49 = *(_QWORD *)(v49 + 24);
            if ( !v49 )
              goto LABEL_115;
          }
        }
      }
LABEL_127:
      if ( v59 )
        j_j___libc_free_0(v59, v78 - (_QWORD)v59);
    }
    else
    {
      (*(void (__fastcall **)(char *))(v48 + 104))(a1);
      for ( i = *(_QWORD *)(a2 + 104); a2 + 88 != i; i = sub_220EEE0(i) )
      {
        if ( (*(unsigned __int8 (__fastcall **)(char *, _QWORD, __int64, _QWORD, __m128i **, void **))(*(_QWORD *)a1 + 120LL))(
               a1,
               *(_QWORD *)(i + 32),
               1,
               0,
               &p_s2,
               &s2) )
        {
          sub_1882690((__int64)a1, i + 64);
          (*(void (__fastcall **)(char *, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
        }
      }
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 112LL))(v2);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v2 + 128LL))(v2, v74);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i **, void **))(*(_QWORD *)v2 + 120LL))(
         v2,
         "WithGlobalValueDeadStripping",
         0,
         0,
         &p_s2,
         &s2) )
  {
    sub_1879DE0(v2, (_BYTE *)(a2 + 176));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, s2);
  }
  v3 = a2 + 192;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v2 + 16LL))(v2) )
  {
    v4 = *(_QWORD *)(a2 + 208);
    p_s2 = 0;
    v80 = 0;
    v81[0] = 0;
    if ( v4 == v3 )
    {
      v7 = 0;
    }
    else
    {
      v5 = v4;
      v6 = 0;
      do
      {
        ++v6;
        v5 = sub_220EF30(v5);
      }
      while ( v5 != v3 );
      if ( v6 > 0x3FFFFFFFFFFFFFFLL )
        goto LABEL_144;
      v66 = 2 * v6;
      p_s2 = (__m128i *)sub_22077B0(32 * v6);
      v7 = p_s2;
      v81[0] = &p_s2[v66];
      do
      {
        if ( v7 )
        {
          v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
          sub_1872C70(v7->m128i_i64, *(_BYTE **)(v4 + 32), *(_QWORD *)(v4 + 32) + *(_QWORD *)(v4 + 40));
        }
        v7 += 2;
        v4 = sub_220EF30(v4);
      }
      while ( v4 != v3 );
    }
    v8 = *(_QWORD *)v2;
    v80 = v7;
    if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(v8 + 56))(v2) || v80 != p_s2)
      && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i **, void **))(*(_QWORD *)v2 + 120LL))(
           v2,
           "CfiFunctionDefs",
           0,
           0,
           &v76,
           &s2) )
    {
      sub_1885D90(v2, &p_s2);
      (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, s2);
    }
    s2 = 0;
    v83 = 0;
    v9 = *(_QWORD *)(a2 + 256);
    v10 = a2 + 240;
    v84 = 0;
    if ( a2 + 240 == v9 )
    {
      v13 = 0;
LABEL_26:
      v14 = *(_QWORD *)v2;
      v83 = v13;
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(v14 + 56))(v2) || (v15 = (char *)s2, v83 != s2) )
      {
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, __m128i **))(*(_QWORD *)v2 + 120LL))(
               v2,
               "CfiFunctionDecls",
               0,
               0,
               &v75,
               &v76) )
        {
          sub_1885D90(v2, (__m128i **)&s2);
          (*(void (__fastcall **)(__int64, __m128i *))(*(_QWORD *)v2 + 128LL))(v2, v76);
        }
        v16 = v83;
        v15 = (char *)s2;
        if ( v83 != s2 )
        {
          do
          {
            if ( *(char **)v15 != v15 + 16 )
              j_j___libc_free_0(*(_QWORD *)v15, *((_QWORD *)v15 + 2) + 1LL);
            v15 += 32;
          }
          while ( v16 != (__int64 *)v15 );
          v15 = (char *)s2;
        }
      }
      if ( v15 )
        j_j___libc_free_0(v15, v84 - v15);
      v17 = v80;
      v18 = p_s2;
      if ( v80 != p_s2 )
      {
        do
        {
          if ( (__m128i *)v18->m128i_i64[0] != &v18[1] )
            j_j___libc_free_0(v18->m128i_i64[0], v18[1].m128i_i64[0] + 1);
          v18 += 2;
        }
        while ( v17 != v18 );
        v18 = p_s2;
      }
      if ( v18 )
        j_j___libc_free_0(v18, v81[0] - (_QWORD)v18);
      return;
    }
    v11 = v9;
    v12 = 0;
    do
    {
      ++v12;
      v11 = sub_220EF30(v11);
    }
    while ( v10 != v11 );
    if ( v12 <= 0x3FFFFFFFFFFFFFFLL )
    {
      v71 = 32 * v12;
      s2 = (void *)sub_22077B0(32 * v12);
      v13 = (__int64 *)s2;
      v84 = (char *)s2 + v71;
      do
      {
        if ( v13 )
        {
          *v13 = (__int64)(v13 + 2);
          sub_1872C70(v13, *(_BYTE **)(v9 + 32), *(_QWORD *)(v9 + 32) + *(_QWORD *)(v9 + 40));
        }
        v13 += 4;
        v9 = sub_220EF30(v9);
      }
      while ( v10 != v9 );
      goto LABEL_26;
    }
LABEL_144:
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  }
  v19 = *(_QWORD *)v2;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(v19 + 56))(v2) && v77 == v76 )
  {
    LODWORD(v83) = 0;
    v84 = 0;
    v85 = (void **)&v83;
    v86 = (void **)&v83;
    j = 0;
  }
  else
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i **, void **))(*(_QWORD *)v2 + 120LL))(
           v2,
           "CfiFunctionDefs",
           0,
           0,
           &p_s2,
           &s2) )
    {
      sub_1885D90(v2, &v76);
      (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, s2);
    }
    LODWORD(v83) = 0;
    v84 = 0;
    v67 = v77;
    v85 = (void **)&v83;
    v86 = (void **)&v83;
    j = 0;
    if ( v77 != v76 )
    {
      v61 = v2;
      v20 = v76;
      do
      {
        v23 = sub_1253B60(&s2, &v83, (__int64)v20);
        v25 = v24;
        if ( v24 )
        {
          if ( v23 || (__int64 **)v24 == &v83 )
          {
            v21 = 1;
          }
          else
          {
            v63 = v24;
            v26 = sub_1872D20(
                    (const void *)v20->m128i_i64[0],
                    v20->m128i_u64[1],
                    *(const void **)(v24 + 32),
                    *(_QWORD *)(v24 + 40));
            v25 = v63;
            v21 = v26 >> 31;
          }
          v62 = v25;
          v22 = sub_22077B0(64);
          *(_QWORD *)(v22 + 32) = v22 + 48;
          sub_1872C70((__int64 *)(v22 + 32), v20->m128i_i64[0], v20->m128i_i64[0] + v20->m128i_i64[1]);
          sub_220F040(v21, v22, v62, &v83);
          ++j;
        }
        v20 += 2;
      }
      while ( v67 != v20 );
      v3 = a2 + 192;
      v2 = v61;
    }
  }
  sub_1875D60(*(_QWORD **)(a2 + 200));
  v27 = v84;
  *(_QWORD *)(a2 + 200) = 0;
  *(_QWORD *)(a2 + 208) = v3;
  *(_QWORD *)(a2 + 216) = v3;
  *(_QWORD *)(a2 + 224) = 0;
  if ( v27 )
  {
    v28 = (int)v83;
    *(_QWORD *)(a2 + 200) = v27;
    *(_DWORD *)(a2 + 192) = v28;
    *(_QWORD *)(a2 + 208) = v85;
    *(_QWORD *)(a2 + 216) = v86;
    *((_QWORD *)v27 + 1) = v3;
    v84 = 0;
    *(_QWORD *)(a2 + 224) = j;
    v85 = (void **)&v83;
    v86 = (void **)&v83;
    j = 0;
  }
  sub_1875D60(0);
  v29 = *(_QWORD *)v2;
  p_s2 = 0;
  v80 = 0;
  v81[0] = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(v29 + 56))(v2) && v80 == p_s2 )
  {
    LODWORD(v83) = 0;
    v84 = 0;
    v85 = (void **)&v83;
    v86 = (void **)&v83;
    j = 0;
  }
  else
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, void **))(*(_QWORD *)v2 + 120LL))(
           v2,
           "CfiFunctionDecls",
           0,
           0,
           &v75,
           &s2) )
    {
      sub_1885D90(v2, &p_s2);
      (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, s2);
    }
    v30 = p_s2;
    v85 = (void **)&v83;
    LODWORD(v83) = 0;
    v68 = v80;
    v84 = 0;
    v86 = (void **)&v83;
    for ( j = 0; v68 != v30; v30 += 2 )
    {
      v33 = sub_1253B60(&s2, &v83, (__int64)v30);
      v35 = v34;
      if ( v34 )
      {
        if ( v33 || (__int64 **)v34 == &v83 )
          v31 = 1;
        else
          v31 = (unsigned int)sub_1872D20(
                                (const void *)v30->m128i_i64[0],
                                v30->m128i_u64[1],
                                *(const void **)(v34 + 32),
                                *(_QWORD *)(v34 + 40)) >> 31;
        v32 = sub_22077B0(64);
        *(_QWORD *)(v32 + 32) = v32 + 48;
        v64 = v32;
        sub_1872C70((__int64 *)(v32 + 32), v30->m128i_i64[0], v30->m128i_i64[0] + v30->m128i_i64[1]);
        sub_220F040(v31, v64, v35, &v83);
        ++j;
      }
    }
  }
  sub_1875D60(*(_QWORD **)(a2 + 248));
  v36 = v84;
  *(_QWORD *)(a2 + 248) = 0;
  *(_QWORD *)(a2 + 256) = a2 + 240;
  *(_QWORD *)(a2 + 264) = a2 + 240;
  *(_QWORD *)(a2 + 272) = 0;
  if ( v36 )
  {
    v37 = (int)v83;
    *(_QWORD *)(a2 + 248) = v36;
    *(_DWORD *)(a2 + 240) = v37;
    *(_QWORD *)(a2 + 256) = v85;
    *(_QWORD *)(a2 + 264) = v86;
    *((_QWORD *)v36 + 1) = a2 + 240;
    v84 = 0;
    *(_QWORD *)(a2 + 272) = j;
    v85 = (void **)&v83;
    v86 = (void **)&v83;
    j = 0;
  }
  sub_1875D60(0);
  v38 = v80;
  v39 = p_s2;
  if ( v80 != p_s2 )
  {
    do
    {
      if ( (__m128i *)v39->m128i_i64[0] != &v39[1] )
        j_j___libc_free_0(v39->m128i_i64[0], v39[1].m128i_i64[0] + 1);
      v39 += 2;
    }
    while ( v38 != v39 );
    v39 = p_s2;
  }
  if ( v39 )
    j_j___libc_free_0(v39, v81[0] - (_QWORD)v39);
  v40 = v77;
  v41 = v76;
  if ( v77 != v76 )
  {
    do
    {
      if ( (__m128i *)v41->m128i_i64[0] != &v41[1] )
        j_j___libc_free_0(v41->m128i_i64[0], v41[1].m128i_i64[0] + 1);
      v41 += 2;
    }
    while ( v40 != v41 );
    v41 = v76;
  }
  if ( v41 )
    j_j___libc_free_0(v41, v78 - (_QWORD)v41);
}
