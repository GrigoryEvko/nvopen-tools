// Function: sub_C1C960
// Address: 0xc1c960
//
_QWORD *__fastcall sub_C1C960(_QWORD *a1, _QWORD *a2, const void **a3)
{
  size_t v5; // rcx
  size_t v6; // r14
  const void *v8; // r8
  const void *v9; // r15
  size_t v10; // rdx
  int v11; // eax
  size_t v12; // rdx
  int v13; // eax
  _QWORD *result; // rax
  __int64 v15; // rbx
  const void *v16; // r15
  const void *v17; // rsi
  const void *v18; // r14
  const void *v19; // rdi
  size_t v20; // rdx
  int v21; // eax
  __int64 v22; // rax
  size_t v23; // r8
  const void *v24; // rdi
  _QWORD *v25; // rcx
  size_t v26; // rdx
  int v27; // eax
  __int64 v28; // rax
  size_t v29; // r8
  const void *v30; // rsi
  __int64 v31; // rcx
  size_t v32; // rdx
  int v33; // eax
  size_t n; // [rsp+8h] [rbp-48h]
  void *s1; // [rsp+10h] [rbp-40h]
  _QWORD *s1a; // [rsp+10h] [rbp-40h]
  void *s1b; // [rsp+10h] [rbp-40h]
  size_t v38; // [rsp+18h] [rbp-38h]
  size_t v39; // [rsp+18h] [rbp-38h]
  size_t v40; // [rsp+18h] [rbp-38h]

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_C1C800((__int64)a1, (__int64)a3);
    v15 = a1[4];
    v16 = a3[1];
    v17 = *a3;
    v18 = *(const void **)(v15 + 40);
    v19 = *(const void **)(v15 + 32);
    if ( v16 < v18 )
    {
      if ( v19 == v17 )
      {
LABEL_22:
        if ( v16 <= v18 )
          return sub_C1C800((__int64)a1, (__int64)a3);
        return 0;
      }
      v20 = (size_t)a3[1];
    }
    else
    {
      if ( v19 == v17 )
      {
LABEL_21:
        if ( v16 == v18 )
          return sub_C1C800((__int64)a1, (__int64)a3);
        goto LABEL_22;
      }
      v20 = *(_QWORD *)(v15 + 40);
    }
    if ( !v19 )
      return 0;
    if ( !v17 )
      return sub_C1C800((__int64)a1, (__int64)a3);
    v21 = memcmp(v19, v17, v20);
    if ( v21 )
    {
      if ( v21 >= 0 )
        return sub_C1C800((__int64)a1, (__int64)a3);
      return 0;
    }
    goto LABEL_21;
  }
  v5 = a2[5];
  v8 = (const void *)a2[4];
  v9 = *a3;
  v10 = (size_t)a3[1];
  v6 = v10;
  if ( v5 <= v10 )
    v10 = a2[5];
  if ( v8 == v9 )
  {
    if ( v5 == v6 )
      return a2;
    if ( v5 <= v6 )
      goto LABEL_11;
    goto LABEL_25;
  }
  v38 = a2[5];
  if ( !v9 )
    goto LABEL_25;
  if ( !v8 )
    goto LABEL_38;
  n = v10;
  s1 = (void *)a2[4];
  v11 = memcmp(v9, v8, v10);
  v12 = n;
  if ( v11 )
  {
    if ( v11 < 0 )
      goto LABEL_25;
LABEL_10:
    v13 = memcmp(s1, v9, v12);
    v5 = v38;
    if ( v13 )
    {
      if ( v13 >= 0 )
        return a2;
      goto LABEL_38;
    }
LABEL_11:
    if ( v5 >= v6 )
      return a2;
LABEL_38:
    if ( (_QWORD *)a1[4] != a2 )
    {
      v28 = sub_220EEE0(a2);
      v29 = *(_QWORD *)(v28 + 40);
      v30 = *(const void **)(v28 + 32);
      v31 = v28;
      if ( v6 > v29 )
      {
        if ( v9 == v30 )
        {
LABEL_46:
          if ( v6 < v29 )
            goto LABEL_47;
          return sub_C1C800((__int64)a1, (__int64)a3);
        }
        v32 = *(_QWORD *)(v28 + 40);
      }
      else
      {
        if ( v9 == v30 )
        {
LABEL_45:
          if ( v6 == v29 )
            return sub_C1C800((__int64)a1, (__int64)a3);
          goto LABEL_46;
        }
        v32 = v6;
      }
      v40 = *(_QWORD *)(v28 + 40);
      if ( !v9 )
        goto LABEL_47;
      s1b = (void *)v28;
      if ( !v30 )
        return sub_C1C800((__int64)a1, (__int64)a3);
      v33 = memcmp(v9, v30, v32);
      v31 = (__int64)s1b;
      v29 = v40;
      if ( v33 )
      {
        if ( v33 < 0 )
        {
LABEL_47:
          result = 0;
          if ( a2[3] )
            return (_QWORD *)v31;
          return result;
        }
        return sub_C1C800((__int64)a1, (__int64)a3);
      }
      goto LABEL_45;
    }
    return 0;
  }
  if ( v38 == v6 )
  {
    v12 = v38;
    goto LABEL_10;
  }
  if ( v38 <= v6 )
    goto LABEL_10;
LABEL_25:
  result = a2;
  if ( (_QWORD *)a1[3] == a2 )
    return result;
  v22 = sub_220EF80(a2);
  v23 = *(_QWORD *)(v22 + 40);
  v24 = *(const void **)(v22 + 32);
  v25 = (_QWORD *)v22;
  if ( v6 >= v23 )
  {
    if ( v9 == v24 )
    {
LABEL_32:
      if ( v6 == v23 )
        return sub_C1C800((__int64)a1, (__int64)a3);
      goto LABEL_33;
    }
    v26 = *(_QWORD *)(v22 + 40);
LABEL_29:
    v39 = *(_QWORD *)(v22 + 40);
    if ( !v24 )
      goto LABEL_34;
    s1a = (_QWORD *)v22;
    if ( !v9 )
      return sub_C1C800((__int64)a1, (__int64)a3);
    v27 = memcmp(v24, v9, v26);
    v25 = s1a;
    v23 = v39;
    if ( v27 )
    {
      if ( v27 >= 0 )
        return sub_C1C800((__int64)a1, (__int64)a3);
      goto LABEL_34;
    }
    goto LABEL_32;
  }
  if ( v9 != v24 )
  {
    v26 = v6;
    goto LABEL_29;
  }
LABEL_33:
  if ( v6 <= v23 )
    return sub_C1C800((__int64)a1, (__int64)a3);
LABEL_34:
  result = 0;
  if ( v25[3] )
    return a2;
  return result;
}
