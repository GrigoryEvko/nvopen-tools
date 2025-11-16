// Function: sub_CAF3B0
// Address: 0xcaf3b0
//
__int64 __fastcall sub_CAF3B0(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  size_t v4; // r14
  size_t v5; // r15
  void *v6; // r8
  const void *v7; // r9
  size_t v8; // rdx
  int v9; // eax
  int v10; // eax
  __int64 v11; // rax
  size_t v12; // rdx
  void *v13; // rcx
  __int64 v14; // r15
  int v15; // eax
  __int64 result; // rax
  __int64 v17; // rax
  size_t v18; // rdx
  void *v19; // rcx
  __int64 v20; // r15
  int v21; // eax
  __int64 v22; // rbx
  size_t v23; // r14
  size_t v24; // r15
  size_t v25; // rdx
  int v26; // eax
  size_t n; // [rsp+8h] [rbp-48h]
  void *s1; // [rsp+10h] [rbp-40h]
  void *s1a; // [rsp+10h] [rbp-40h]
  void *s1b; // [rsp+10h] [rbp-40h]
  void *s2; // [rsp+18h] [rbp-38h]
  void *s2c; // [rsp+18h] [rbp-38h]
  void *s2a; // [rsp+18h] [rbp-38h]
  void *s2b; // [rsp+18h] [rbp-38h]

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_CAF250((__int64)a1, a3);
    v22 = a1[4];
    v23 = *(_QWORD *)(a3 + 8);
    v24 = *(_QWORD *)(v22 + 40);
    v25 = v23;
    if ( v24 <= v23 )
      v25 = *(_QWORD *)(v22 + 40);
    if ( v25 && (v26 = memcmp(*(const void **)(v22 + 32), *(const void **)a3, v25)) != 0 )
    {
      if ( v26 >= 0 )
        return sub_CAF250((__int64)a1, a3);
    }
    else if ( v24 == v23 || v24 >= v23 )
    {
      return sub_CAF250((__int64)a1, a3);
    }
    return 0;
  }
  v4 = *(_QWORD *)(a3 + 8);
  v5 = a2[5];
  v6 = *(void **)a3;
  v7 = (const void *)a2[4];
  v8 = v5;
  if ( v4 <= v5 )
    v8 = v4;
  if ( !v8 )
  {
    if ( v4 == v5 )
      return (__int64)a2;
    goto LABEL_7;
  }
  n = v8;
  s1 = (void *)a2[4];
  s2 = v6;
  v9 = memcmp(v6, s1, v8);
  v6 = s2;
  v7 = s1;
  v8 = n;
  if ( !v9 )
  {
    if ( v4 == v5 )
    {
      v10 = memcmp(s1, s2, n);
      v6 = s2;
      if ( !v10 )
        return (__int64)a2;
      goto LABEL_44;
    }
LABEL_7:
    if ( v4 >= v5 )
    {
      if ( !v8 || (s2c = v6, v10 = memcmp(v7, v6, v8), v6 = s2c, !v10) )
      {
LABEL_10:
        if ( v4 > v5 )
          goto LABEL_11;
        return (__int64)a2;
      }
      goto LABEL_44;
    }
    goto LABEL_23;
  }
  if ( v9 >= 0 )
  {
    v10 = memcmp(s1, s2, n);
    v6 = s2;
    if ( !v10 )
    {
      if ( v4 == v5 )
        return (__int64)a2;
      goto LABEL_10;
    }
LABEL_44:
    if ( v10 < 0 )
    {
LABEL_11:
      if ( (_QWORD *)a1[4] != a2 )
      {
        s2a = v6;
        v11 = sub_220EEE0(a2);
        v12 = v4;
        v13 = *(void **)(v11 + 40);
        v14 = v11;
        if ( (unsigned __int64)v13 <= v4 )
          v12 = *(_QWORD *)(v11 + 40);
        if ( v12 && (s1a = *(void **)(v11 + 40), v15 = memcmp(s2a, *(const void **)(v11 + 32), v12), v13 = s1a, v15) )
        {
          if ( v15 < 0 )
          {
LABEL_18:
            result = 0;
            if ( a2[3] )
              return v14;
            return result;
          }
        }
        else if ( v13 != (void *)v4 && (unsigned __int64)v13 > v4 )
        {
          goto LABEL_18;
        }
        return sub_CAF250((__int64)a1, a3);
      }
      return 0;
    }
    return (__int64)a2;
  }
LABEL_23:
  result = (__int64)a2;
  if ( (_QWORD *)a1[3] == a2 )
    return result;
  s2b = v6;
  v17 = sub_220EF80(a2);
  v18 = v4;
  v19 = *(void **)(v17 + 40);
  v20 = v17;
  if ( (unsigned __int64)v19 <= v4 )
    v18 = *(_QWORD *)(v17 + 40);
  if ( v18 && (s1b = *(void **)(v17 + 40), v21 = memcmp(*(const void **)(v17 + 32), s2b, v18), v19 = s1b, v21) )
  {
    if ( v21 >= 0 )
      return sub_CAF250((__int64)a1, a3);
  }
  else if ( v19 == (void *)v4 || (unsigned __int64)v19 >= v4 )
  {
    return sub_CAF250((__int64)a1, a3);
  }
  result = 0;
  if ( *(_QWORD *)(v20 + 24) )
    return (__int64)a2;
  return result;
}
