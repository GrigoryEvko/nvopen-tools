// Function: sub_C695B0
// Address: 0xc695b0
//
__int64 __fastcall sub_C695B0(char *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  size_t v5; // rbx
  const void *v6; // r8
  char *v7; // r15
  __int64 v8; // r13
  size_t v9; // r12
  const void *v10; // r10
  size_t v11; // rdx
  int v12; // eax
  _QWORD *v13; // r9
  size_t v14; // rcx
  const void *v15; // rsi
  size_t v16; // rdx
  int v17; // eax
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // r13
  const void *v21; // r12
  size_t v22; // r15
  char *v23; // rbx
  char *v24; // r14
  __int64 v25; // rax
  void *v26; // rcx
  size_t v27; // rdx
  int v28; // eax
  const void *v29; // rbx
  size_t v30; // r12
  size_t v31; // rdx
  int v32; // eax
  size_t v33; // rcx
  const void *v34; // rsi
  size_t v35; // rdx
  int v36; // eax
  size_t v37; // rdx
  int v38; // eax
  __int64 v39; // rax
  size_t v40; // rdx
  int v41; // eax
  __int64 v42; // rcx
  __int64 v43; // r12
  __int64 i; // rbx
  __int64 *v45; // r14
  __int64 v46; // rcx
  __int64 v47; // r12
  size_t v48; // [rsp+0h] [rbp-70h]
  size_t v49; // [rsp+0h] [rbp-70h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  char *v51; // [rsp+18h] [rbp-58h]
  const void *v52; // [rsp+28h] [rbp-48h]
  char *v53; // [rsp+28h] [rbp-48h]
  const void *v54; // [rsp+28h] [rbp-48h]
  const void *v55; // [rsp+30h] [rbp-40h]
  _QWORD *v56; // [rsp+30h] [rbp-40h]
  __int64 v57; // [rsp+30h] [rbp-40h]
  _QWORD *v58; // [rsp+30h] [rbp-40h]
  size_t v59; // [rsp+30h] [rbp-40h]
  size_t v60; // [rsp+30h] [rbp-40h]
  void *s2a; // [rsp+38h] [rbp-38h]
  void *s2b; // [rsp+38h] [rbp-38h]
  char *s2; // [rsp+38h] [rbp-38h]
  _QWORD *s2c; // [rsp+38h] [rbp-38h]
  _QWORD *s2d; // [rsp+38h] [rbp-38h]

  result = a2 - a1;
  v50 = a3;
  v51 = a2;
  if ( a2 - a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v53 = a2;
    goto LABEL_67;
  }
  while ( 2 )
  {
    --v50;
    v4 = *((_QWORD *)a1 + 1);
    v5 = *(_QWORD *)(v4 + 16);
    v6 = *(const void **)(v4 + 8);
    v7 = &a1[8 * ((__int64)(((v51 - a1) >> 3) + ((unsigned __int64)(v51 - a1) >> 63)) >> 1)];
    v8 = *(_QWORD *)v7;
    v9 = *(_QWORD *)(*(_QWORD *)v7 + 16LL);
    v10 = *(const void **)(*(_QWORD *)v7 + 8LL);
    v11 = v9;
    if ( v5 <= v9 )
      v11 = *(_QWORD *)(v4 + 16);
    if ( v11 )
    {
      v55 = *(const void **)(*(_QWORD *)v7 + 8LL);
      s2a = *(void **)(v4 + 8);
      v12 = memcmp(s2a, v55, v11);
      v6 = s2a;
      v10 = v55;
      if ( v12 )
      {
        if ( v12 < 0 )
          goto LABEL_9;
LABEL_43:
        v13 = (_QWORD *)*((_QWORD *)v51 - 1);
        v33 = v13[2];
        v34 = (const void *)v13[1];
        v35 = v33;
        if ( v5 <= v33 )
          v35 = v5;
        if ( v35
          && (v49 = v13[2],
              v54 = v10,
              v58 = (_QWORD *)*((_QWORD *)v51 - 1),
              v36 = memcmp(v6, v34, v35),
              v13 = v58,
              v10 = v54,
              v33 = v49,
              v36) )
        {
          if ( v36 < 0 )
            goto LABEL_65;
        }
        else if ( v5 != v33 && v5 < v33 )
        {
          goto LABEL_65;
        }
        v37 = v33;
        if ( v9 <= v33 )
          v37 = v9;
        if ( v37 && (v59 = v33, s2c = v13, v38 = memcmp(v10, v34, v37), v13 = s2c, v33 = v59, v38) )
        {
          if ( v38 < 0 )
            goto LABEL_55;
        }
        else if ( v9 != v33 && v9 < v33 )
        {
          goto LABEL_55;
        }
        v39 = *(_QWORD *)a1;
        *(_QWORD *)a1 = v8;
        *(_QWORD *)v7 = v39;
        v19 = (__int64 *)a1;
        goto LABEL_16;
      }
    }
    if ( v5 == v9 || v5 >= v9 )
      goto LABEL_43;
LABEL_9:
    v13 = (_QWORD *)*((_QWORD *)v51 - 1);
    v14 = v13[2];
    v15 = (const void *)v13[1];
    v16 = v14;
    if ( v9 <= v14 )
      v16 = v9;
    if ( !v16
      || (v48 = v13[2],
          v52 = v6,
          v56 = (_QWORD *)*((_QWORD *)v51 - 1),
          v17 = memcmp(v10, v15, v16),
          v13 = v56,
          v6 = v52,
          v14 = v48,
          !v17) )
    {
      if ( v9 != v14 && v9 < v14 )
        goto LABEL_15;
      goto LABEL_59;
    }
    if ( v17 >= 0 )
    {
LABEL_59:
      v40 = v14;
      if ( v5 <= v14 )
        v40 = v5;
      if ( !v40 || (v60 = v14, s2d = v13, v41 = memcmp(v6, v15, v40), v13 = s2d, v14 = v60, !v41) )
      {
        if ( v5 == v14 || v5 >= v14 )
          goto LABEL_65;
LABEL_55:
        v20 = *(_QWORD *)a1;
        *(_QWORD *)a1 = v13;
        *((_QWORD *)v51 - 1) = v20;
        v4 = *(_QWORD *)a1;
        v57 = *((_QWORD *)a1 + 1);
        goto LABEL_17;
      }
      if ( v41 < 0 )
        goto LABEL_55;
LABEL_65:
      v42 = *(_QWORD *)a1;
      *(_QWORD *)a1 = v4;
      *((_QWORD *)a1 + 1) = v42;
      v57 = v42;
      v20 = *((_QWORD *)v51 - 1);
      goto LABEL_17;
    }
LABEL_15:
    v18 = *(_QWORD *)a1;
    *(_QWORD *)a1 = v8;
    *(_QWORD *)v7 = v18;
    v19 = (__int64 *)a1;
LABEL_16:
    v4 = *v19;
    v57 = v19[1];
    v20 = *((_QWORD *)v51 - 1);
LABEL_17:
    v21 = *(const void **)(v4 + 8);
    v22 = *(_QWORD *)(v4 + 16);
    v23 = a1 + 8;
    v24 = v51;
    while ( 1 )
    {
      v53 = v23;
      v26 = *(void **)(v57 + 16);
      v27 = (size_t)v26;
      if ( v22 <= (unsigned __int64)v26 )
        v27 = v22;
      if ( !v27 )
        break;
      s2b = *(void **)(v57 + 16);
      v28 = memcmp(*(const void **)(v57 + 8), v21, v27);
      v26 = s2b;
      if ( !v28 )
        break;
      if ( v28 >= 0 )
        goto LABEL_27;
LABEL_20:
      v25 = *((_QWORD *)v23 + 1);
      v23 += 8;
      v57 = v25;
    }
    if ( (void *)v22 != v26 && v22 > (unsigned __int64)v26 )
      goto LABEL_20;
LABEL_27:
    s2 = v23;
    v24 -= 8;
    v29 = v21;
    while ( 1 )
    {
      v30 = *(_QWORD *)(v20 + 16);
      v31 = v30;
      if ( v22 <= v30 )
        v31 = v22;
      if ( v31 )
      {
        v32 = memcmp(v29, *(const void **)(v20 + 8), v31);
        if ( v32 )
          break;
      }
      if ( v22 == v30 || v22 >= v30 )
      {
        v23 = s2;
        if ( s2 >= v24 )
          goto LABEL_37;
LABEL_19:
        *(_QWORD *)v23 = v20;
        v20 = *((_QWORD *)v24 - 1);
        *(_QWORD *)v24 = v57;
        v21 = *(const void **)(*(_QWORD *)a1 + 8LL);
        v22 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
        goto LABEL_20;
      }
LABEL_34:
      v24 -= 8;
      v20 = *(_QWORD *)v24;
    }
    if ( v32 < 0 )
      goto LABEL_34;
    v23 = s2;
    if ( s2 < v24 )
      goto LABEL_19;
LABEL_37:
    sub_C695B0(v23, v51, v50, v26, v6, v13);
    result = v23 - a1;
    if ( v23 - a1 > 128 )
    {
      if ( v50 )
      {
        v51 = v23;
        continue;
      }
LABEL_67:
      v43 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_C69040((__int64)a1, i, v43, *(_QWORD *)&a1[8 * i]);
        if ( !i )
          break;
      }
      v45 = (__int64 *)(v53 - 8);
      do
      {
        v46 = *v45;
        v47 = (char *)v45-- - a1;
        v45[1] = *(_QWORD *)a1;
        result = sub_C69040((__int64)a1, 0, v47 >> 3, v46);
      }
      while ( v47 > 8 );
    }
    return result;
  }
}
