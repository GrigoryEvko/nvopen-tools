// Function: sub_E713E0
// Address: 0xe713e0
//
unsigned __int64 __fastcall sub_E713E0(
        __int64 a1,
        size_t *a2,
        int a3,
        unsigned int a4,
        int a5,
        __int64 a6,
        unsigned __int8 a7,
        int a8,
        __int64 a9)
{
  __int64 v11; // r8
  __int64 v12; // r9
  char *v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  int v18; // eax
  unsigned __int64 v19; // rsi
  unsigned int v20; // eax
  void *v21; // r11
  __int64 v22; // r14
  bool v23; // r15
  size_t *v24; // rsi
  unsigned __int64 v25; // r13
  int v27; // eax
  unsigned int v28; // r9d
  __int64 v29; // rax
  size_t v30; // rdx
  unsigned int v31; // r9d
  _QWORD *v32; // r15
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  void *v36; // r8
  unsigned int v37; // r9d
  _QWORD *v38; // rcx
  void *v39; // rdi
  __int64 *v40; // rax
  __int64 v41; // rdx
  unsigned __int8 v42; // al
  size_t v43; // r14
  const char *v44; // r15
  int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned int v49; // r8d
  _QWORD *v50; // rcx
  _QWORD *v51; // r9
  size_t **v52; // rax
  size_t v53; // rdx
  size_t *v54; // rcx
  size_t v55; // r15
  const void *v56; // r8
  unsigned __int64 v57; // rcx
  size_t *v58; // rcx
  size_t v59; // r15
  const void *v60; // r8
  unsigned __int64 v61; // rcx
  size_t v62; // rdx
  const void *v63; // rsi
  void *v64; // rax
  size_t v65; // rax
  unsigned int v66; // [rsp+8h] [rbp-128h]
  unsigned int v67; // [rsp+8h] [rbp-128h]
  _QWORD *v68; // [rsp+8h] [rbp-128h]
  unsigned int v72; // [rsp+24h] [rbp-10Ch]
  size_t na; // [rsp+28h] [rbp-108h]
  size_t n; // [rsp+28h] [rbp-108h]
  unsigned int nb; // [rsp+28h] [rbp-108h]
  size_t nc; // [rsp+28h] [rbp-108h]
  void *src; // [rsp+30h] [rbp-100h]
  const char *srca; // [rsp+30h] [rbp-100h]
  void *srcb; // [rsp+30h] [rbp-100h]
  _QWORD *srcc; // [rsp+30h] [rbp-100h]
  void *srcd; // [rsp+30h] [rbp-100h]
  size_t v82; // [rsp+38h] [rbp-F8h]
  size_t v83; // [rsp+38h] [rbp-F8h]
  unsigned int v84; // [rsp+38h] [rbp-F8h]
  size_t v85; // [rsp+38h] [rbp-F8h]
  size_t v86; // [rsp+38h] [rbp-F8h]
  const char *v87; // [rsp+40h] [rbp-F0h]
  size_t v88; // [rsp+48h] [rbp-E8h]
  const char *v89; // [rsp+60h] [rbp-D0h] BYREF
  char *v90; // [rsp+68h] [rbp-C8h]
  unsigned __int64 v91; // [rsp+70h] [rbp-C0h]
  _BYTE v92[184]; // [rsp+78h] [rbp-B8h] BYREF

  if ( a6 || (v23 = a8 != -1 || a9 != 0) )
  {
    v90 = 0;
    v89 = v92;
    v91 = 128;
    sub_CA0EC0((__int64)a2, (__int64)&v89);
    v13 = v90;
    v72 = (unsigned int)v90;
    if ( (unsigned __int64)(v90 + 1) > v91 )
    {
      sub_C8D290((__int64)&v89, v92, (__int64)(v90 + 1), 1u, v11, v12);
      v13 = v90;
    }
    v13[(_QWORD)v89] = 0;
    v14 = (unsigned __int64)(v90 + 1);
    v15 = v91;
    ++v90;
    if ( !a6 )
    {
LABEL_8:
      if ( v15 < v14 + 1 )
      {
        sub_C8D290((__int64)&v89, v92, v14 + 1, 1u, v14 + 1, v12);
        v14 = (unsigned __int64)v90;
      }
      v89[v14] = 0;
      v16 = (unsigned __int64)(v90 + 1);
      v17 = v91;
      ++v90;
      if ( !a9 )
      {
LABEL_14:
        if ( v16 + 4 > v17 )
        {
          sub_C8D290((__int64)&v89, v92, v16 + 4, 1u, v16 + 4, v12);
          v16 = (unsigned __int64)v90;
        }
        *(_DWORD *)&v89[v16] = a8;
        v82 = (size_t)v90;
        v90 += 4;
        v87 = v89;
        v88 = (size_t)v90;
        na = (size_t)v89;
        src = v90;
        v18 = sub_C92610();
        v19 = (unsigned __int64)v87;
        v20 = sub_C92740(a1 + 2048, v87, v88, v18);
        v21 = (void *)na;
        v22 = *(_QWORD *)(a1 + 2048) + 8LL * v20;
        if ( *(_QWORD *)v22 )
        {
          if ( *(_QWORD *)v22 != -8 )
          {
            v23 = 0;
            goto LABEL_19;
          }
          --*(_DWORD *)(a1 + 2064);
        }
        v67 = v20;
        n = (size_t)src;
        srcb = v21;
        v35 = sub_C7D670(v82 + 21, 8);
        v36 = (void *)n;
        v37 = v67;
        v38 = (_QWORD *)v35;
        v39 = (void *)(v35 + 16);
        if ( n )
        {
          v62 = n;
          v63 = srcb;
          nb = v67;
          srcd = v36;
          v68 = (_QWORD *)v35;
          v64 = memcpy(v39, v63, v62);
          v36 = srcd;
          v37 = nb;
          v38 = v68;
          v39 = v64;
        }
        v19 = v37;
        *((_BYTE *)v39 + v82 + 4) = 0;
        *v38 = v36;
        v38[1] = 0;
        *(_QWORD *)v22 = v38;
        ++*(_DWORD *)(a1 + 2060);
        v22 = *(_QWORD *)(a1 + 2048) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 2048), v37);
        if ( *(_QWORD *)v22 == -8 || !*(_QWORD *)v22 )
        {
          v40 = (__int64 *)(v22 + 8);
          do
          {
            do
            {
              v41 = *v40;
              v22 = (__int64)v40++;
            }
            while ( !v41 );
          }
          while ( v41 == -8 );
          v23 = 1;
LABEL_19:
          if ( v89 != v92 )
            _libc_free(v89, v19);
          v24 = *(size_t **)v22;
          if ( !v23 )
            return v24[1];
          goto LABEL_62;
        }
LABEL_36:
        v23 = 1;
        goto LABEL_19;
      }
      if ( (*(_BYTE *)(a9 + 8) & 1) != 0 )
      {
        v58 = *(size_t **)(a9 - 8);
        v59 = *v58;
        v60 = v58 + 3;
        v61 = v16 + *v58;
        if ( v61 <= v91 )
        {
LABEL_70:
          if ( v59 )
          {
            memcpy((void *)&v89[v16], v60, v59);
            v16 = (unsigned __int64)&v90[v59];
          }
          v90 = (char *)v16;
          v17 = v91;
          goto LABEL_14;
        }
        v16 = v61;
      }
      else
      {
        if ( v16 <= v91 )
        {
          v90 = (char *)v16;
          goto LABEL_14;
        }
        v60 = 0;
        v59 = 0;
      }
      v85 = (size_t)v60;
      sub_C8D290((__int64)&v89, v92, v16, 1u, (__int64)v60, v12);
      v16 = (unsigned __int64)v90;
      v60 = (const void *)v85;
      goto LABEL_70;
    }
    if ( (*(_BYTE *)(a6 + 8) & 1) != 0 )
    {
      v54 = *(size_t **)(a6 - 8);
      v55 = *v54;
      v56 = v54 + 3;
      v57 = v14 + *v54;
      if ( v57 <= v91 )
      {
LABEL_66:
        if ( v55 )
        {
          memcpy((void *)&v89[v14], v56, v55);
          v14 = (unsigned __int64)&v90[v55];
        }
        v90 = (char *)v14;
        v15 = v91;
        goto LABEL_8;
      }
      v14 = v57;
    }
    else
    {
      if ( v14 <= v91 )
      {
        v90 = (char *)v14;
        goto LABEL_8;
      }
      v56 = 0;
      v55 = 0;
    }
    v86 = (size_t)v56;
    sub_C8D290((__int64)&v89, v92, v14, 1u, (__int64)v56, v12);
    v14 = (unsigned __int64)v90;
    v56 = (const void *)v86;
    goto LABEL_66;
  }
  if ( *((_BYTE *)a2 + 33) != 1 )
  {
LABEL_26:
    v90 = 0;
    v89 = v92;
    v91 = 128;
    sub_CA0EC0((__int64)a2, (__int64)&v89);
    v72 = (unsigned int)v90;
    srca = v89;
    v83 = (size_t)v90;
    v27 = sub_C92610();
    v19 = (unsigned __int64)srca;
    v28 = sub_C92740(a1 + 2048, srca, v83, v27);
    v22 = *(_QWORD *)(a1 + 2048) + 8LL * v28;
    if ( *(_QWORD *)v22 )
    {
      if ( *(_QWORD *)v22 != -8 )
        goto LABEL_19;
      --*(_DWORD *)(a1 + 2064);
    }
    v66 = v28;
    v29 = sub_C7D670(v83 + 17, 8);
    v30 = v83;
    v31 = v66;
    v32 = (_QWORD *)v29;
    if ( v83 )
    {
      memcpy((void *)(v29 + 16), srca, v83);
      v30 = v83;
      v31 = v66;
    }
    *((_BYTE *)v32 + v30 + 16) = 0;
    v19 = v31;
    *v32 = v30;
    v32[1] = 0;
    *(_QWORD *)v22 = v32;
    ++*(_DWORD *)(a1 + 2060);
    v22 = *(_QWORD *)(a1 + 2048) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 2048), v31);
    if ( !*(_QWORD *)v22 || *(_QWORD *)v22 == -8 )
    {
      v33 = (__int64 *)(v22 + 8);
      do
      {
        do
        {
          v34 = *v33;
          v22 = (__int64)v33++;
        }
        while ( v34 == -8 );
      }
      while ( !v34 );
    }
    goto LABEL_36;
  }
  v42 = *((_BYTE *)a2 + 32);
  if ( v42 == 1 )
  {
    v44 = 0;
    goto LABEL_84;
  }
  if ( (unsigned __int8)(v42 - 3) > 3u )
    goto LABEL_26;
  if ( v42 == 4 )
  {
    v43 = *(_QWORD *)(*a2 + 8);
    v44 = *(const char **)*a2;
    v72 = v43;
    goto LABEL_52;
  }
  if ( v42 <= 4u )
  {
    if ( v42 != 3 )
LABEL_85:
      BUG();
    v44 = (const char *)*a2;
    if ( *a2 )
    {
      v65 = strlen((const char *)*a2);
      v72 = v65;
      v43 = v65;
      goto LABEL_52;
    }
LABEL_84:
    v72 = 0;
    v43 = 0;
    goto LABEL_52;
  }
  if ( (unsigned __int8)(v42 - 5) > 1u )
    goto LABEL_85;
  v43 = a2[1];
  v44 = (const char *)*a2;
  v72 = v43;
LABEL_52:
  v89 = v44;
  v90 = (char *)v43;
  v45 = sub_C92610();
  v46 = (unsigned int)sub_C92740(a1 + 2048, v44, v43, v45);
  v47 = *(_QWORD *)(a1 + 2048);
  v24 = *(size_t **)(v47 + 8 * v46);
  if ( v24 )
  {
    if ( v24 != (size_t *)-8LL )
      return v24[1];
    --*(_DWORD *)(a1 + 2064);
  }
  srcc = (_QWORD *)(v47 + 8 * v46);
  v84 = v46;
  v48 = sub_C7D670(v43 + 17, 8);
  v49 = v84;
  v50 = srcc;
  v51 = (_QWORD *)v48;
  if ( v43 )
  {
    nc = v48;
    memcpy((void *)(v48 + 16), v44, v43);
    v49 = v84;
    v50 = srcc;
    v51 = (_QWORD *)nc;
  }
  *((_BYTE *)v51 + v43 + 16) = 0;
  *v51 = v43;
  v51[1] = 0;
  *v50 = v51;
  ++*(_DWORD *)(a1 + 2060);
  v22 = *(_QWORD *)(a1 + 2048) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 2048), v49);
  v24 = *(size_t **)v22;
  if ( !*(_QWORD *)v22 || v24 == (size_t *)-8LL )
  {
    v52 = (size_t **)(v22 + 8);
    do
    {
      do
      {
        v24 = *v52;
        v22 = (__int64)v52++;
      }
      while ( v24 == (size_t *)-8LL );
    }
    while ( !v24 );
  }
LABEL_62:
  v53 = v72;
  if ( *v24 <= v72 )
    v53 = *v24;
  v25 = sub_E6CBD0((_QWORD *)a1, v24 + 2, v53, a3, a4, a5, a6, a7, a8, a9);
  *(_QWORD *)(*(_QWORD *)v22 + 8LL) = v25;
  sub_E71020(
    a1,
    *(_QWORD **)(v25 + 128),
    *(_QWORD *)(v25 + 136),
    *(_DWORD *)(v25 + 152),
    *(_DWORD *)(v25 + 156),
    *(_DWORD *)(v25 + 160));
  return v25;
}
