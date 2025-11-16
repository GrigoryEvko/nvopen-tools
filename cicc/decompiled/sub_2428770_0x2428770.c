// Function: sub_2428770
// Address: 0x2428770
//
int __fastcall sub_2428770(__int64 a1, size_t **a2, __int64 a3)
{
  __int64 v3; // rax
  size_t **v4; // rax
  size_t *v5; // r14
  size_t *v6; // r15
  size_t v7; // rbx
  const void *v8; // r12
  size_t v9; // r13
  const void *v10; // r10
  size_t v11; // rdx
  int v12; // eax
  size_t *v13; // r9
  void *v14; // r8
  const void *v15; // rsi
  size_t v16; // rdx
  int v17; // eax
  size_t v18; // rdx
  int v19; // eax
  size_t *v20; // rdi
  size_t *v21; // r13
  size_t v22; // r14
  size_t **v23; // rbx
  size_t **v24; // r15
  size_t *v25; // rax
  size_t v26; // rdx
  size_t v27; // rcx
  int v28; // eax
  const void *v29; // rbx
  size_t v30; // r12
  size_t v31; // rdx
  int v32; // eax
  const void *v33; // rsi
  size_t v34; // rdx
  int v35; // eax
  size_t v36; // rdx
  int v37; // eax
  size_t *v38; // rax
  size_t **v39; // rax
  __int64 v40; // rbx
  __int64 v41; // r12
  size_t **v42; // rbx
  size_t *v43; // rcx
  size_t *v44; // rax
  size_t v46; // [rsp+8h] [rbp-78h]
  size_t v47; // [rsp+10h] [rbp-70h]
  const void *v48; // [rsp+10h] [rbp-70h]
  size_t **v49; // [rsp+18h] [rbp-68h]
  __int64 v50; // [rsp+20h] [rbp-60h]
  size_t **v51; // [rsp+28h] [rbp-58h]
  size_t *v53; // [rsp+38h] [rbp-48h]
  size_t **v54; // [rsp+38h] [rbp-48h]
  size_t *v55; // [rsp+38h] [rbp-48h]
  void *v56; // [rsp+38h] [rbp-48h]
  void *s2a; // [rsp+40h] [rbp-40h]
  void *s2b; // [rsp+40h] [rbp-40h]
  size_t *s2; // [rsp+40h] [rbp-40h]
  void *s2c; // [rsp+40h] [rbp-40h]
  size_t *s2d; // [rsp+40h] [rbp-40h]
  size_t **v62; // [rsp+48h] [rbp-38h]
  size_t *v63; // [rsp+48h] [rbp-38h]
  size_t v64; // [rsp+48h] [rbp-38h]
  size_t **v65; // [rsp+48h] [rbp-38h]

  v3 = (__int64)a2 - a1;
  v51 = a2;
  v50 = a3;
  if ( (__int64)a2 - a1 <= 128 )
    return v3;
  if ( !a3 )
  {
    v54 = a2;
    goto LABEL_66;
  }
  v49 = (size_t **)(a1 + 8);
  while ( 2 )
  {
    --v50;
    v4 = (size_t **)(a1 + 8 * (v3 >> 4));
    v5 = *(size_t **)(a1 + 8);
    v6 = *v4;
    v62 = v4;
    v7 = *v5;
    v8 = v5 + 24;
    v9 = **v4;
    v10 = *v4 + 24;
    v11 = *v5;
    if ( v9 <= *v5 )
      v11 = *v6;
    if ( v11 )
    {
      v12 = memcmp(v5 + 24, v6 + 24, v11);
      v10 = v6 + 24;
      if ( v12 )
      {
        if ( v12 < 0 )
          goto LABEL_10;
LABEL_42:
        v13 = *(v51 - 1);
        v14 = (void *)*v13;
        v33 = v13 + 24;
        v34 = *v13;
        if ( v7 <= *v13 )
          v34 = v7;
        if ( !v34 )
          goto LABEL_47;
        v46 = *v13;
        v48 = v10;
        v55 = *(v51 - 1);
        s2c = v13 + 24;
        v35 = memcmp(v5 + 24, v33, v34);
        v33 = s2c;
        v13 = v55;
        v10 = v48;
        v14 = (void *)v46;
        if ( v35 )
        {
          if ( v35 < 0 )
            goto LABEL_22;
        }
        else
        {
LABEL_47:
          if ( (void *)v7 != v14 && v7 < (unsigned __int64)v14 )
            goto LABEL_22;
        }
        v36 = (size_t)v14;
        if ( v9 <= (unsigned __int64)v14 )
          v36 = v9;
        if ( v36 && (v56 = v14, s2d = v13, v37 = memcmp(v10, v33, v36), v13 = s2d, v14 = v56, v37) )
        {
          if ( v37 < 0 )
            goto LABEL_54;
        }
        else if ( (void *)v9 != v14 && v9 < (unsigned __int64)v14 )
        {
          goto LABEL_54;
        }
        v44 = *(size_t **)a1;
        *(_QWORD *)a1 = v6;
        *v62 = v44;
        v39 = (size_t **)a1;
LABEL_64:
        v5 = *v39;
        s2 = v39[1];
        v8 = *v39 + 24;
        v21 = *(v51 - 1);
        goto LABEL_23;
      }
    }
    if ( v9 == v7 || v9 <= v7 )
      goto LABEL_42;
LABEL_10:
    v13 = *(v51 - 1);
    v14 = (void *)*v13;
    v15 = v13 + 24;
    v16 = *v13;
    if ( v9 <= *v13 )
      v16 = v9;
    if ( !v16
      || (v47 = *v13,
          v53 = *(v51 - 1),
          s2a = v13 + 24,
          v17 = memcmp(v10, v15, v16),
          v15 = s2a,
          v13 = v53,
          v14 = (void *)v47,
          !v17) )
    {
      if ( (void *)v9 == v14 || v9 >= (unsigned __int64)v14 )
        goto LABEL_16;
      goto LABEL_63;
    }
    if ( v17 < 0 )
    {
LABEL_63:
      v38 = *(size_t **)a1;
      *(_QWORD *)a1 = v6;
      *v62 = v38;
      v39 = (size_t **)a1;
      goto LABEL_64;
    }
LABEL_16:
    v18 = (size_t)v14;
    if ( v7 <= (unsigned __int64)v14 )
      v18 = v7;
    if ( !v18 || (s2b = v14, v63 = v13, v19 = memcmp(v5 + 24, v15, v18), v13 = v63, v14 = s2b, !v19) )
    {
      if ( (void *)v7 == v14 || v7 >= (unsigned __int64)v14 )
        goto LABEL_22;
LABEL_54:
      v21 = *(size_t **)a1;
      *(_QWORD *)a1 = v13;
      *(v51 - 1) = v21;
      v5 = *(size_t **)a1;
      v8 = (const void *)(*(_QWORD *)a1 + 192LL);
      s2 = *(size_t **)(a1 + 8);
      goto LABEL_23;
    }
    if ( v19 < 0 )
      goto LABEL_54;
LABEL_22:
    v20 = *(size_t **)a1;
    *(_QWORD *)a1 = v5;
    *(_QWORD *)(a1 + 8) = v20;
    s2 = v20;
    v21 = *(v51 - 1);
LABEL_23:
    v22 = *v5;
    v23 = v49;
    v24 = v51;
    while ( 1 )
    {
      v26 = v22;
      v54 = v23;
      v27 = *s2;
      if ( *s2 <= v22 )
        v26 = *s2;
      if ( !v26 )
        break;
      v64 = *s2;
      v28 = memcmp(s2 + 24, v8, v26);
      v27 = v64;
      if ( !v28 )
        break;
      if ( v28 >= 0 )
        goto LABEL_33;
LABEL_26:
      v25 = v23[1];
      ++v23;
      s2 = v25;
    }
    if ( v27 != v22 && v27 < v22 )
      goto LABEL_26;
LABEL_33:
    v65 = v23;
    --v24;
    v29 = v8;
    while ( 1 )
    {
      v30 = *v21;
      v31 = v22;
      if ( *v21 <= v22 )
        v31 = *v21;
      if ( v31 )
      {
        v32 = memcmp(v29, v21 + 24, v31);
        if ( v32 )
          break;
      }
      if ( v30 == v22 || v30 <= v22 )
      {
        v23 = v65;
        if ( v65 >= v24 )
          goto LABEL_57;
LABEL_25:
        *v23 = v21;
        v21 = *(v24 - 1);
        *v24 = s2;
        v22 = **(_QWORD **)a1;
        v8 = (const void *)(*(_QWORD *)a1 + 192LL);
        goto LABEL_26;
      }
LABEL_40:
      v21 = *--v24;
    }
    if ( v32 < 0 )
      goto LABEL_40;
    v23 = v65;
    if ( v65 < v24 )
      goto LABEL_25;
LABEL_57:
    sub_2428770(v23, v51, v50, v27, v14, v13);
    v3 = (__int64)v23 - a1;
    if ( (__int64)v23 - a1 > 128 )
    {
      if ( v50 )
      {
        v51 = v23;
        continue;
      }
LABEL_66:
      v40 = v3 >> 3;
      v41 = ((v3 >> 3) - 2) >> 1;
      sub_2425EA0(a1, v41, v3 >> 3, *(size_t **)(a1 + 8 * v41));
      do
      {
        --v41;
        sub_2425EA0(a1, v41, v40, *(size_t **)(a1 + 8 * v41));
      }
      while ( v41 );
      v42 = v54;
      do
      {
        v43 = *--v42;
        *v42 = *(size_t **)a1;
        LODWORD(v3) = sub_2425EA0(a1, 0, ((__int64)v42 - a1) >> 3, v43);
      }
      while ( (__int64)v42 - a1 > 8 );
    }
    return v3;
  }
}
