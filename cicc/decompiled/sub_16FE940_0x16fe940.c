// Function: sub_16FE940
// Address: 0x16fe940
//
__int64 __fastcall sub_16FE940(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  size_t v5; // r13
  size_t v6; // r14
  void *v8; // r8
  const void *v9; // r9
  int v10; // eax
  __int64 result; // rax
  __int64 v12; // rax
  void *v13; // rcx
  const void *v14; // rdi
  __int64 v15; // r14
  const void *v16; // rsi
  int v17; // eax
  size_t v18; // rdx
  __int64 v19; // rbx
  unsigned __int64 v20; // r14
  const void *v21; // rsi
  unsigned __int64 v22; // r13
  const void *v23; // rdi
  int v24; // eax
  const void *v25; // rsi
  int v26; // eax
  __int64 v27; // rax
  void *v28; // rcx
  const void *v29; // rsi
  __int64 v30; // r14
  const void *v31; // rdi
  int v32; // eax
  const void *v33; // rdi
  void *s1; // [rsp+0h] [rbp-40h]
  void *s1a; // [rsp+0h] [rbp-40h]
  void *s2b; // [rsp+8h] [rbp-38h]
  void *s2; // [rsp+8h] [rbp-38h]
  void *s2c; // [rsp+8h] [rbp-38h]
  void *s2d; // [rsp+8h] [rbp-38h]
  void *s2e; // [rsp+8h] [rbp-38h]
  void *s2f; // [rsp+8h] [rbp-38h]
  void *s2a; // [rsp+8h] [rbp-38h]
  void *s2g; // [rsp+8h] [rbp-38h]
  void *s2h; // [rsp+8h] [rbp-38h]
  void *s2i; // [rsp+8h] [rbp-38h]

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_16FE7C0((__int64)a1, a3);
    v19 = a1[4];
    v20 = *(_QWORD *)(a3 + 8);
    v21 = *(const void **)a3;
    v22 = *(_QWORD *)(v19 + 40);
    v23 = *(const void **)(v19 + 32);
    if ( v22 > v20 )
    {
      if ( !v20 )
        return sub_16FE7C0((__int64)a1, a3);
      v24 = memcmp(v23, v21, *(_QWORD *)(a3 + 8));
      if ( !v24 )
      {
LABEL_25:
        if ( v22 >= v20 )
          return sub_16FE7C0((__int64)a1, a3);
        return 0;
      }
    }
    else if ( !v22 || (v24 = memcmp(v23, v21, *(_QWORD *)(v19 + 40))) == 0 )
    {
      if ( v22 == v20 )
        return sub_16FE7C0((__int64)a1, a3);
      goto LABEL_25;
    }
    if ( v24 >= 0 )
      return sub_16FE7C0((__int64)a1, a3);
    return 0;
  }
  v5 = *(_QWORD *)(a3 + 8);
  v6 = a2[5];
  v8 = *(void **)a3;
  v9 = (const void *)a2[4];
  if ( v6 < v5 )
  {
    if ( !v6 )
      goto LABEL_35;
    s1a = (void *)a2[4];
    s2d = *(void **)a3;
    v10 = memcmp(*(const void **)a3, s1a, v6);
    v8 = s2d;
    v9 = s1a;
    if ( !v10 )
    {
      if ( v6 <= v5 )
      {
        v18 = v6;
        goto LABEL_34;
      }
LABEL_7:
      result = (__int64)a2;
      if ( (_QWORD *)a1[3] == a2 )
        return result;
      s2 = v8;
      v12 = sub_220EF80(a2);
      v13 = *(void **)(v12 + 40);
      v14 = *(const void **)(v12 + 32);
      v15 = v12;
      if ( (unsigned __int64)v13 > v5 )
      {
        if ( !v5 )
          return sub_16FE7C0((__int64)a1, a3);
        v25 = s2;
        s2e = *(void **)(v12 + 40);
        v17 = memcmp(v14, v25, v5);
        v13 = s2e;
        if ( !v17 )
        {
LABEL_12:
          if ( (unsigned __int64)v13 < v5 )
            goto LABEL_13;
          return sub_16FE7C0((__int64)a1, a3);
        }
      }
      else if ( !v13 || (v16 = s2, s2c = *(void **)(v12 + 40), v17 = memcmp(v14, v16, (size_t)s2c), v13 = s2c, !v17) )
      {
        if ( v13 == (void *)v5 )
          return sub_16FE7C0((__int64)a1, a3);
        goto LABEL_12;
      }
      if ( v17 < 0 )
      {
LABEL_13:
        result = 0;
        if ( *(_QWORD *)(v15 + 24) )
          return (__int64)a2;
        return result;
      }
      return sub_16FE7C0((__int64)a1, a3);
    }
LABEL_31:
    if ( v10 >= 0 )
    {
      if ( v6 <= v5 )
        goto LABEL_55;
      v18 = v5;
      if ( !v5 )
        return (__int64)a2;
LABEL_34:
      s2f = v8;
      v26 = memcmp(v9, v8, v18);
      v8 = s2f;
      if ( !v26 )
        goto LABEL_35;
      goto LABEL_49;
    }
    goto LABEL_7;
  }
  if ( v5 )
  {
    s1 = (void *)a2[4];
    s2b = *(void **)a3;
    v10 = memcmp(*(const void **)a3, s1, *(_QWORD *)(a3 + 8));
    v8 = s2b;
    v9 = s1;
    if ( v10 )
      goto LABEL_31;
  }
  if ( v6 != v5 )
  {
    if ( v6 > v5 )
      goto LABEL_7;
LABEL_55:
    if ( !v6 )
    {
LABEL_47:
      if ( v6 == v5 )
        return (__int64)a2;
LABEL_35:
      if ( v6 < v5 )
        goto LABEL_36;
      return (__int64)a2;
    }
    goto LABEL_46;
  }
  if ( !v6 )
    return (__int64)a2;
LABEL_46:
  s2h = v8;
  v26 = memcmp(v9, v8, v6);
  v8 = s2h;
  if ( !v26 )
    goto LABEL_47;
LABEL_49:
  if ( v26 >= 0 )
    return (__int64)a2;
LABEL_36:
  if ( (_QWORD *)a1[4] == a2 )
    return 0;
  s2a = v8;
  v27 = sub_220EEE0(a2);
  v28 = *(void **)(v27 + 40);
  v29 = *(const void **)(v27 + 32);
  v30 = v27;
  if ( (unsigned __int64)v28 >= v5 )
  {
    if ( !v5 || (v31 = s2a, s2g = *(void **)(v27 + 40), v32 = memcmp(v31, v29, v5), v28 = s2g, !v32) )
    {
      if ( v28 == (void *)v5 )
        return sub_16FE7C0((__int64)a1, a3);
      goto LABEL_41;
    }
LABEL_59:
    if ( v32 >= 0 )
      return sub_16FE7C0((__int64)a1, a3);
    goto LABEL_42;
  }
  if ( !v28 )
    return sub_16FE7C0((__int64)a1, a3);
  v33 = s2a;
  s2i = *(void **)(v27 + 40);
  v32 = memcmp(v33, v29, (size_t)s2i);
  v28 = s2i;
  if ( v32 )
    goto LABEL_59;
LABEL_41:
  if ( (unsigned __int64)v28 <= v5 )
    return sub_16FE7C0((__int64)a1, a3);
LABEL_42:
  result = 0;
  if ( a2[3] )
    return v30;
  return result;
}
