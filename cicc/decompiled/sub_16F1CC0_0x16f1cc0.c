// Function: sub_16F1CC0
// Address: 0x16f1cc0
//
__int64 __fastcall sub_16F1CC0(char *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  char *v4; // rbx
  size_t v5; // r12
  const void *v6; // r9
  char *v7; // rax
  __int64 v8; // r14
  size_t v9; // r13
  const void *v10; // r8
  int v11; // eax
  _QWORD *v12; // r10
  size_t v13; // rcx
  const void *v14; // rsi
  int v15; // eax
  char *v16; // rcx
  char *v17; // rax
  char *v18; // r13
  size_t v19; // r15
  const void *v20; // r12
  char *v21; // rbx
  size_t v22; // rax
  char *v23; // r15
  char *v24; // r14
  size_t v25; // r13
  int v26; // eax
  void *v27; // r9
  char *v28; // rax
  const void *v29; // rdi
  const void *v30; // rbx
  int v31; // eax
  size_t v32; // r12
  const void *v33; // rsi
  size_t v34; // rcx
  const void *v35; // rsi
  int v36; // eax
  __int64 v37; // r12
  __int64 i; // rbx
  __int64 *v39; // r14
  __int64 v40; // rcx
  __int64 v41; // r12
  int v42; // eax
  int v43; // eax
  size_t v44; // [rsp+0h] [rbp-70h]
  size_t v45; // [rsp+0h] [rbp-70h]
  const void *v46; // [rsp+0h] [rbp-70h]
  const void *v47; // [rsp+0h] [rbp-70h]
  char *v48; // [rsp+8h] [rbp-68h]
  __int64 v49; // [rsp+10h] [rbp-60h]
  char *v50; // [rsp+18h] [rbp-58h]
  const void *v52; // [rsp+28h] [rbp-48h]
  char *v53; // [rsp+28h] [rbp-48h]
  const void *v54; // [rsp+28h] [rbp-48h]
  _QWORD *v55; // [rsp+28h] [rbp-48h]
  _QWORD *v56; // [rsp+28h] [rbp-48h]
  void *s1a; // [rsp+30h] [rbp-40h]
  _QWORD *s1b; // [rsp+30h] [rbp-40h]
  char *s1; // [rsp+30h] [rbp-40h]
  void *s1c; // [rsp+30h] [rbp-40h]
  _QWORD *s1d; // [rsp+30h] [rbp-40h]
  void *s1e; // [rsp+30h] [rbp-40h]
  void *s1f; // [rsp+30h] [rbp-40h]
  void *s1g; // [rsp+30h] [rbp-40h]
  void *s1h; // [rsp+30h] [rbp-40h]
  _QWORD *s1i; // [rsp+30h] [rbp-40h]
  _QWORD *s1j; // [rsp+30h] [rbp-40h]
  void *s2a; // [rsp+38h] [rbp-38h]
  void *s2b; // [rsp+38h] [rbp-38h]
  void *s2c; // [rsp+38h] [rbp-38h]
  char *s2; // [rsp+38h] [rbp-38h]
  void *s2d; // [rsp+38h] [rbp-38h]
  _QWORD *s2e; // [rsp+38h] [rbp-38h]
  _QWORD *s2f; // [rsp+38h] [rbp-38h]
  void *s2g; // [rsp+38h] [rbp-38h]
  void *s2h; // [rsp+38h] [rbp-38h]

  result = a2 - a1;
  v49 = a3;
  v50 = a2;
  if ( a2 - a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v53 = a2;
    goto LABEL_50;
  }
  v48 = a1 + 8;
  while ( 2 )
  {
    --v49;
    v4 = (char *)*((_QWORD *)a1 + 1);
    v5 = *((_QWORD *)v4 + 2);
    v6 = (const void *)*((_QWORD *)v4 + 1);
    v7 = &a1[8 * ((__int64)(((v50 - a1) >> 3) + ((unsigned __int64)(v50 - a1) >> 63)) >> 1)];
    v8 = *(_QWORD *)v7;
    v9 = *(_QWORD *)(*(_QWORD *)v7 + 16LL);
    v10 = *(const void **)(*(_QWORD *)v7 + 8LL);
    if ( v5 > v9 )
    {
      if ( !v9 )
        goto LABEL_43;
      s1c = *(void **)(*(_QWORD *)v7 + 8LL);
      s2d = (void *)*((_QWORD *)v4 + 1);
      v11 = memcmp(s2d, s1c, *(_QWORD *)(*(_QWORD *)v7 + 16LL));
      v6 = s2d;
      v10 = s1c;
      if ( !v11 )
        goto LABEL_8;
    }
    else
    {
      if ( !v5 )
      {
        if ( !v9 )
        {
          v12 = (_QWORD *)*((_QWORD *)v50 - 1);
          v35 = (const void *)v12[1];
          v34 = v12[2];
          goto LABEL_46;
        }
LABEL_8:
        if ( v5 < v9 )
          goto LABEL_9;
LABEL_43:
        v12 = (_QWORD *)*((_QWORD *)v50 - 1);
        v34 = v12[2];
        v35 = (const void *)v12[1];
        if ( v5 > v34 )
        {
          if ( !v34 )
          {
            if ( v9 )
              goto LABEL_14;
LABEL_62:
            if ( v9 == v34 )
              goto LABEL_14;
            goto LABEL_63;
          }
          v46 = v10;
          v55 = (_QWORD *)*((_QWORD *)v50 - 1);
          s1e = (void *)v12[2];
          v36 = memcmp(v6, v35, (size_t)s1e);
          v34 = (size_t)s1e;
          v12 = v55;
          v10 = v46;
          if ( !v36 )
          {
LABEL_47:
            if ( v5 < v34 )
              goto LABEL_48;
LABEL_59:
            if ( v9 > v34 )
            {
              if ( !v34 )
                goto LABEL_14;
              s1j = v12;
              s2h = (void *)v34;
              v42 = memcmp(v10, v35, v34);
              v34 = (size_t)s2h;
              v12 = s1j;
              if ( !v42 )
              {
LABEL_63:
                if ( v9 >= v34 )
                  goto LABEL_14;
                goto LABEL_73;
              }
            }
            else
            {
              if ( !v9 )
                goto LABEL_62;
              s1f = (void *)v34;
              s2e = v12;
              v42 = memcmp(v10, v35, v9);
              v12 = s2e;
              v34 = (size_t)s1f;
              if ( !v42 )
                goto LABEL_62;
            }
            if ( v42 >= 0 )
              goto LABEL_14;
LABEL_73:
            v16 = v50;
            v18 = *(char **)a1;
            *(_QWORD *)a1 = v12;
            *((_QWORD *)v50 - 1) = v18;
            v4 = *(char **)a1;
            s1 = (char *)*((_QWORD *)a1 + 1);
            goto LABEL_15;
          }
        }
        else if ( !v5
               || (v45 = v12[2],
                   v54 = v10,
                   s1d = (_QWORD *)*((_QWORD *)v50 - 1),
                   v36 = memcmp(v6, v35, v5),
                   v12 = s1d,
                   v10 = v54,
                   v34 = v45,
                   !v36) )
        {
LABEL_46:
          if ( v5 == v34 )
            goto LABEL_59;
          goto LABEL_47;
        }
        if ( v36 < 0 )
          goto LABEL_48;
        goto LABEL_59;
      }
      s1a = *(void **)(*(_QWORD *)v7 + 8LL);
      s2a = (void *)*((_QWORD *)v4 + 1);
      v11 = memcmp(s2a, s1a, *((_QWORD *)v4 + 2));
      v6 = s2a;
      v10 = s1a;
      if ( !v11 )
      {
        if ( v5 == v9 )
          goto LABEL_43;
        goto LABEL_8;
      }
    }
    if ( v11 >= 0 )
      goto LABEL_43;
LABEL_9:
    v12 = (_QWORD *)*((_QWORD *)v50 - 1);
    v13 = v12[2];
    v14 = (const void *)v12[1];
    if ( v9 > v13 )
    {
      if ( !v13 )
      {
        if ( v5 )
          goto LABEL_48;
LABEL_71:
        if ( v5 == v13 )
          goto LABEL_48;
        goto LABEL_72;
      }
      v47 = v6;
      v56 = (_QWORD *)*((_QWORD *)v50 - 1);
      s1g = (void *)v12[2];
      v15 = memcmp(v10, v14, (size_t)s1g);
      v13 = (size_t)s1g;
      v12 = v56;
      v6 = v47;
      if ( !v15 )
        goto LABEL_13;
LABEL_67:
      if ( v15 < 0 )
        goto LABEL_14;
LABEL_68:
      if ( v5 > v13 )
      {
        if ( !v13 )
          goto LABEL_48;
        s1i = v12;
        s2g = (void *)v13;
        v43 = memcmp(v6, v14, v13);
        v13 = (size_t)s2g;
        v12 = s1i;
        if ( !v43 )
        {
LABEL_72:
          if ( v5 < v13 )
            goto LABEL_73;
LABEL_48:
          v16 = *(char **)a1;
          *(_QWORD *)a1 = v4;
          *((_QWORD *)a1 + 1) = v16;
          s1 = v16;
          v18 = (char *)*((_QWORD *)v50 - 1);
          goto LABEL_15;
        }
      }
      else
      {
        if ( !v5 )
          goto LABEL_71;
        s1h = (void *)v13;
        s2f = v12;
        v43 = memcmp(v6, v14, v5);
        v12 = s2f;
        v13 = (size_t)s1h;
        if ( !v43 )
          goto LABEL_71;
      }
      if ( v43 < 0 )
        goto LABEL_73;
      goto LABEL_48;
    }
    if ( v9 )
    {
      v44 = v12[2];
      v52 = v6;
      s1b = (_QWORD *)*((_QWORD *)v50 - 1);
      v15 = memcmp(v10, v14, v9);
      v12 = s1b;
      v6 = v52;
      v13 = v44;
      if ( v15 )
        goto LABEL_67;
    }
    if ( v9 == v13 )
      goto LABEL_68;
LABEL_13:
    if ( v9 >= v13 )
      goto LABEL_68;
LABEL_14:
    v16 = a1;
    v17 = *(char **)a1;
    *(_QWORD *)a1 = v8;
    *(_QWORD *)&a1[8 * ((__int64)(((v50 - a1) >> 3) + ((unsigned __int64)(v50 - a1) >> 63)) >> 1)] = v17;
    v4 = *(char **)a1;
    s1 = (char *)*((_QWORD *)a1 + 1);
    v18 = (char *)*((_QWORD *)v50 - 1);
LABEL_15:
    v19 = *((_QWORD *)v4 + 2);
    v20 = (const void *)*((_QWORD *)v4 + 1);
    v21 = v48;
    v22 = v19;
    v23 = v50;
    v24 = v18;
    v25 = v22;
    while ( 1 )
    {
      v53 = v21;
      v27 = (void *)*((_QWORD *)s1 + 2);
      v29 = (const void *)*((_QWORD *)s1 + 1);
      if ( (unsigned __int64)v27 > v25 )
        break;
      if ( v27 )
      {
        s2b = (void *)*((_QWORD *)s1 + 2);
        v26 = memcmp(v29, v20, (size_t)s2b);
        v27 = s2b;
        if ( v26 )
          goto LABEL_24;
      }
      if ( v27 == (void *)v25 )
        goto LABEL_25;
LABEL_19:
      if ( (unsigned __int64)v27 >= v25 )
        goto LABEL_25;
LABEL_20:
      v28 = (char *)*((_QWORD *)v21 + 1);
      v21 += 8;
      s1 = v28;
    }
    if ( !v25 )
      goto LABEL_25;
    s2c = (void *)*((_QWORD *)s1 + 2);
    v26 = memcmp(v29, v20, v25);
    v27 = s2c;
    if ( !v26 )
      goto LABEL_19;
LABEL_24:
    if ( v26 < 0 )
      goto LABEL_20;
LABEL_25:
    s2 = v21;
    v23 -= 8;
    v30 = v20;
    while ( 1 )
    {
      v32 = *((_QWORD *)v24 + 2);
      v33 = (const void *)*((_QWORD *)v24 + 1);
      if ( v32 < v25 )
        break;
      if ( v25 )
      {
        v31 = memcmp(v30, v33, v25);
        if ( v31 )
          goto LABEL_34;
      }
      if ( v32 == v25 )
        goto LABEL_35;
LABEL_29:
      if ( v32 <= v25 )
        goto LABEL_35;
LABEL_30:
      v24 = (char *)*((_QWORD *)v23 - 1);
      v23 -= 8;
    }
    if ( !v32 )
      goto LABEL_35;
    v31 = memcmp(v30, v33, *((_QWORD *)v24 + 2));
    if ( !v31 )
      goto LABEL_29;
LABEL_34:
    if ( v31 < 0 )
      goto LABEL_30;
LABEL_35:
    v21 = s2;
    if ( s2 < v23 )
    {
      *(_QWORD *)s2 = v24;
      v24 = (char *)*((_QWORD *)v23 - 1);
      *(_QWORD *)v23 = s1;
      v20 = *(const void **)(*(_QWORD *)a1 + 8LL);
      v25 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
      goto LABEL_20;
    }
    sub_16F1CC0(s2, v50, v49, v16, v10);
    result = s2 - a1;
    if ( s2 - a1 > 128 )
    {
      if ( v49 )
      {
        v50 = s2;
        continue;
      }
LABEL_50:
      v37 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_16F1780((__int64)a1, i, v37, *(_QWORD *)&a1[8 * i]);
        if ( !i )
          break;
      }
      v39 = (__int64 *)(v53 - 8);
      do
      {
        v40 = *v39;
        v41 = (char *)v39-- - a1;
        v39[1] = *(_QWORD *)a1;
        result = sub_16F1780((__int64)a1, 0, v41 >> 3, v40);
      }
      while ( v41 > 8 );
    }
    return result;
  }
}
