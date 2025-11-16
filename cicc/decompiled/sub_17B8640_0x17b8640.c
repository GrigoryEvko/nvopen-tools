// Function: sub_17B8640
// Address: 0x17b8640
//
int __fastcall sub_17B8640(__int64 a1, size_t **a2, __int64 a3)
{
  __int64 v3; // rax
  size_t **v4; // rax
  size_t *v5; // r14
  size_t *v6; // rcx
  size_t **v7; // r15
  size_t v8; // rbx
  const void *v9; // r12
  size_t v10; // r13
  const void *v11; // r10
  int v12; // eax
  size_t *v13; // r9
  size_t v14; // r8
  const void *v15; // rsi
  int v16; // eax
  size_t *v17; // rax
  size_t *v18; // r13
  size_t **v19; // rbx
  size_t **v20; // r15
  size_t v21; // rax
  size_t *v22; // r14
  size_t v23; // r13
  int v24; // eax
  size_t v25; // r9
  size_t *v26; // rax
  const void *v27; // rdi
  const void *v28; // rbx
  int v29; // eax
  size_t v30; // r12
  const void *v31; // rsi
  size_t v32; // r8
  const void *v33; // rsi
  int v34; // eax
  size_t *v35; // rcx
  __int64 v36; // rbx
  __int64 v37; // r12
  size_t **v38; // rbx
  size_t *v39; // rcx
  int v40; // eax
  int v41; // eax
  size_t v43; // [rsp+8h] [rbp-78h]
  const void *v44; // [rsp+8h] [rbp-78h]
  size_t v45; // [rsp+10h] [rbp-70h]
  const void *v46; // [rsp+10h] [rbp-70h]
  size_t *v47; // [rsp+10h] [rbp-70h]
  size_t *v48; // [rsp+10h] [rbp-70h]
  __int64 v49; // [rsp+18h] [rbp-68h]
  __int64 v50; // [rsp+20h] [rbp-60h]
  size_t **v51; // [rsp+28h] [rbp-58h]
  size_t *v53; // [rsp+38h] [rbp-48h]
  size_t **v54; // [rsp+38h] [rbp-48h]
  size_t *v55; // [rsp+38h] [rbp-48h]
  size_t *v56; // [rsp+38h] [rbp-48h]
  size_t v57; // [rsp+38h] [rbp-48h]
  size_t *v58; // [rsp+38h] [rbp-48h]
  size_t *v59; // [rsp+38h] [rbp-48h]
  size_t *v60; // [rsp+40h] [rbp-40h]
  size_t *v61; // [rsp+40h] [rbp-40h]
  size_t *v62; // [rsp+40h] [rbp-40h]
  size_t *v63; // [rsp+40h] [rbp-40h]
  size_t *v64; // [rsp+40h] [rbp-40h]
  size_t v65; // [rsp+40h] [rbp-40h]
  size_t *v66; // [rsp+40h] [rbp-40h]
  size_t v67; // [rsp+40h] [rbp-40h]
  size_t v68; // [rsp+40h] [rbp-40h]
  size_t *v69; // [rsp+40h] [rbp-40h]
  size_t *v70; // [rsp+40h] [rbp-40h]
  void *s1a; // [rsp+48h] [rbp-38h]
  void *s1b; // [rsp+48h] [rbp-38h]
  void *s1c; // [rsp+48h] [rbp-38h]
  void *s1d; // [rsp+48h] [rbp-38h]
  size_t **s1; // [rsp+48h] [rbp-38h]
  void *s1e; // [rsp+48h] [rbp-38h]
  void *s1f; // [rsp+48h] [rbp-38h]
  void *s1g; // [rsp+48h] [rbp-38h]
  size_t *s1h; // [rsp+48h] [rbp-38h]
  void *s1i; // [rsp+48h] [rbp-38h]
  size_t *s1j; // [rsp+48h] [rbp-38h]
  void *s1k; // [rsp+48h] [rbp-38h]
  void *s1l; // [rsp+48h] [rbp-38h]

  v3 = (__int64)a2 - a1;
  v51 = a2;
  v50 = a3;
  if ( (__int64)a2 - a1 <= 128 )
    return v3;
  if ( !a3 )
  {
    v54 = a2;
    goto LABEL_50;
  }
  v49 = a1 + 8;
  while ( 2 )
  {
    --v50;
    v4 = (size_t **)(a1 + 8 * (v3 >> 4));
    v5 = *(size_t **)(a1 + 8);
    v6 = *v4;
    v7 = v4;
    v8 = *v5;
    v9 = v5 + 22;
    v10 = **v4;
    v11 = *v4 + 22;
    if ( v10 < *v5 )
    {
      if ( !v10 )
        goto LABEL_43;
      v63 = *v4;
      s1e = *v4 + 22;
      v12 = memcmp(v5 + 22, s1e, **v4);
      v11 = s1e;
      v6 = v63;
      if ( !v12 )
        goto LABEL_8;
    }
    else
    {
      if ( !v8 )
      {
        if ( !v10 )
        {
          v13 = *(v51 - 1);
          v32 = *v13;
          v33 = v13 + 22;
          goto LABEL_46;
        }
LABEL_8:
        if ( v10 > v8 )
          goto LABEL_9;
LABEL_43:
        v13 = *(v51 - 1);
        v32 = *v13;
        v33 = v13 + 22;
        if ( v8 > *v13 )
        {
          if ( !v32 )
          {
            if ( v10 )
              goto LABEL_14;
LABEL_61:
            if ( v10 == v32 )
              goto LABEL_14;
            goto LABEL_62;
          }
          v44 = v11;
          v47 = *(v51 - 1);
          v56 = v6;
          v65 = *v13;
          s1g = v13 + 22;
          v34 = memcmp(v5 + 22, v33, *v13);
          v33 = s1g;
          v32 = v65;
          v6 = v56;
          v13 = v47;
          v11 = v44;
          if ( !v34 )
          {
LABEL_47:
            if ( v8 < v32 )
              goto LABEL_48;
LABEL_58:
            if ( v10 > v32 )
            {
              if ( !v32 )
                goto LABEL_14;
              v59 = v13;
              v70 = v6;
              s1l = (void *)v32;
              v40 = memcmp(v11, v33, v32);
              v32 = (size_t)s1l;
              v6 = v70;
              v13 = v59;
              if ( !v40 )
              {
LABEL_62:
                if ( v10 >= v32 )
                  goto LABEL_14;
                goto LABEL_72;
              }
            }
            else
            {
              if ( !v10 )
                goto LABEL_61;
              v57 = v32;
              v66 = v13;
              s1h = v6;
              v40 = memcmp(v11, v33, v10);
              v6 = s1h;
              v13 = v66;
              v32 = v57;
              if ( !v40 )
                goto LABEL_61;
            }
            if ( v40 >= 0 )
              goto LABEL_14;
LABEL_72:
            v18 = *(size_t **)a1;
            *(_QWORD *)a1 = v13;
            *(v51 - 1) = v18;
            v5 = *(size_t **)a1;
            v9 = (const void *)(*(_QWORD *)a1 + 176LL);
            v62 = *(size_t **)(a1 + 8);
            goto LABEL_15;
          }
        }
        else
        {
          if ( !v8 )
            goto LABEL_46;
          v43 = *v13;
          v46 = v11;
          v55 = *(v51 - 1);
          v64 = v6;
          s1f = v13 + 22;
          v34 = memcmp(v5 + 22, v33, v8);
          v33 = s1f;
          v6 = v64;
          v13 = v55;
          v11 = v46;
          v32 = v43;
          if ( !v34 )
          {
LABEL_46:
            if ( v8 == v32 )
              goto LABEL_58;
            goto LABEL_47;
          }
        }
        if ( v34 < 0 )
          goto LABEL_48;
        goto LABEL_58;
      }
      v60 = *v4;
      s1a = *v4 + 22;
      v12 = memcmp(v5 + 22, s1a, *v5);
      v11 = s1a;
      v6 = v60;
      if ( !v12 )
      {
        if ( v10 == v8 )
          goto LABEL_43;
        goto LABEL_8;
      }
    }
    if ( v12 >= 0 )
      goto LABEL_43;
LABEL_9:
    v13 = *(v51 - 1);
    v14 = *v13;
    v15 = v13 + 22;
    if ( v10 > *v13 )
    {
      if ( !v14 )
      {
        if ( v8 )
          goto LABEL_48;
LABEL_70:
        if ( v8 == v14 )
          goto LABEL_48;
        goto LABEL_71;
      }
      v48 = *(v51 - 1);
      v58 = v6;
      v67 = *v13;
      s1i = v13 + 22;
      v16 = memcmp(v11, v15, *v13);
      v15 = s1i;
      v14 = v67;
      v6 = v58;
      v13 = v48;
      if ( !v16 )
        goto LABEL_13;
LABEL_66:
      if ( v16 < 0 )
        goto LABEL_14;
LABEL_67:
      if ( v8 > v14 )
      {
        if ( !v14 )
          goto LABEL_48;
        v69 = v13;
        s1k = (void *)v14;
        v41 = memcmp(v5 + 22, v15, v14);
        v14 = (size_t)s1k;
        v13 = v69;
        if ( !v41 )
        {
LABEL_71:
          if ( v8 < v14 )
            goto LABEL_72;
LABEL_48:
          v35 = *(size_t **)a1;
          *(_QWORD *)a1 = v5;
          *(_QWORD *)(a1 + 8) = v35;
          v62 = v35;
          v18 = *(v51 - 1);
          goto LABEL_15;
        }
      }
      else
      {
        if ( !v8 )
          goto LABEL_70;
        v68 = v14;
        s1j = v13;
        v41 = memcmp(v5 + 22, v15, v8);
        v13 = s1j;
        v14 = v68;
        if ( !v41 )
          goto LABEL_70;
      }
      if ( v41 < 0 )
        goto LABEL_72;
      goto LABEL_48;
    }
    if ( v10 )
    {
      v45 = *v13;
      v53 = *(v51 - 1);
      v61 = v6;
      s1b = v13 + 22;
      v16 = memcmp(v11, v15, v10);
      v15 = s1b;
      v6 = v61;
      v13 = v53;
      v14 = v45;
      if ( v16 )
        goto LABEL_66;
    }
    if ( v10 == v14 )
      goto LABEL_67;
LABEL_13:
    if ( v10 >= v14 )
      goto LABEL_67;
LABEL_14:
    v17 = *(size_t **)a1;
    *(_QWORD *)a1 = v6;
    *v7 = v17;
    v5 = *(size_t **)a1;
    v62 = *(size_t **)(a1 + 8);
    v9 = (const void *)(*(_QWORD *)a1 + 176LL);
    v18 = *(v51 - 1);
LABEL_15:
    v19 = (size_t **)v49;
    v20 = v51;
    v21 = *v5;
    v22 = v18;
    v23 = v21;
    while ( 1 )
    {
      v54 = v19;
      v25 = *v62;
      v27 = v62 + 22;
      if ( *v62 > v23 )
        break;
      if ( v25 )
      {
        s1c = (void *)*v62;
        v24 = memcmp(v27, v9, *v62);
        v25 = (size_t)s1c;
        if ( v24 )
          goto LABEL_24;
      }
      if ( v25 == v23 )
        goto LABEL_25;
LABEL_19:
      if ( v25 >= v23 )
        goto LABEL_25;
LABEL_20:
      v26 = v19[1];
      ++v19;
      v62 = v26;
    }
    if ( !v23 )
      goto LABEL_25;
    s1d = (void *)*v62;
    v24 = memcmp(v27, v9, v23);
    v25 = (size_t)s1d;
    if ( !v24 )
      goto LABEL_19;
LABEL_24:
    if ( v24 < 0 )
      goto LABEL_20;
LABEL_25:
    s1 = v19;
    --v20;
    v28 = v9;
    while ( 1 )
    {
      v30 = *v22;
      v31 = v22 + 22;
      if ( *v22 < v23 )
        break;
      if ( v23 )
      {
        v29 = memcmp(v28, v31, v23);
        if ( v29 )
          goto LABEL_34;
      }
      if ( v30 == v23 )
        goto LABEL_35;
LABEL_29:
      if ( v30 <= v23 )
        goto LABEL_35;
LABEL_30:
      v22 = *--v20;
    }
    if ( !v30 )
      goto LABEL_35;
    v29 = memcmp(v28, v31, *v22);
    if ( !v29 )
      goto LABEL_29;
LABEL_34:
    if ( v29 < 0 )
      goto LABEL_30;
LABEL_35:
    v19 = s1;
    if ( s1 < v20 )
    {
      *s1 = v22;
      v22 = *(v20 - 1);
      *v20 = v62;
      v23 = **(_QWORD **)a1;
      v9 = (const void *)(*(_QWORD *)a1 + 176LL);
      goto LABEL_20;
    }
    sub_17B8640(s1, v51, v50);
    v3 = (__int64)s1 - a1;
    if ( (__int64)s1 - a1 > 128 )
    {
      if ( v50 )
      {
        v51 = s1;
        continue;
      }
LABEL_50:
      v36 = v3 >> 3;
      v37 = ((v3 >> 3) - 2) >> 1;
      sub_17B7740(a1, v37, v3 >> 3, *(size_t **)(a1 + 8 * v37));
      do
      {
        --v37;
        sub_17B7740(a1, v37, v36, *(size_t **)(a1 + 8 * v37));
      }
      while ( v37 );
      v38 = v54;
      do
      {
        v39 = *--v38;
        *v38 = *(size_t **)a1;
        LODWORD(v3) = sub_17B7740(a1, 0, ((__int64)v38 - a1) >> 3, v39);
      }
      while ( (__int64)v38 - a1 > 8 );
    }
    return v3;
  }
}
