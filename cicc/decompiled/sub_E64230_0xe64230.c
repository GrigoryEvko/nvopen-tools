// Function: sub_E64230
// Address: 0xe64230
//
char __fastcall sub_E64230(__int64 a1, __int64 a2)
{
  char result; // al
  char v3; // r13
  size_t v4; // r14
  size_t v5; // r15
  const void *v6; // r8
  const void *v7; // r9
  size_t v8; // rdx
  int v9; // eax
  bool v10; // sf
  signed __int64 v11; // rcx
  int v12; // edx
  __int64 v13; // r15
  size_t v14; // r13
  size_t v15; // r14
  const void *v16; // r8
  const void *v17; // r9
  size_t v18; // r15
  int v19; // eax
  bool v20; // sf
  int v21; // edx
  __int64 v22; // r14
  signed __int64 v23; // rdx
  size_t n; // [rsp+8h] [rbp-48h]
  void *s1; // [rsp+10h] [rbp-40h]
  void *s1a; // [rsp+10h] [rbp-40h]
  void *s2; // [rsp+18h] [rbp-38h]
  void *s2a; // [rsp+18h] [rbp-38h]

  result = *(_BYTE *)(a1 + 36);
  v3 = *(_BYTE *)(a2 + 36);
  if ( !result )
  {
    if ( v3 )
      return result;
    v14 = *(_QWORD *)(a1 + 8);
    v15 = *(_QWORD *)(a2 + 8);
    v16 = *(const void **)a1;
    v17 = *(const void **)a2;
    v18 = v15;
    if ( v14 <= v15 )
      v18 = *(_QWORD *)(a1 + 8);
    if ( v18 )
    {
      s1a = *(void **)a2;
      s2a = *(void **)a1;
      v19 = memcmp(*(const void **)a1, *(const void **)a2, v18);
      v16 = s2a;
      v17 = s1a;
      v20 = v19 < 0;
      if ( v19 )
      {
        result = 1;
        if ( v20 )
          return result;
        goto LABEL_24;
      }
      v23 = v14 - v15;
      if ( (__int64)(v14 - v15) > 0x7FFFFFFF )
        goto LABEL_24;
    }
    else
    {
      v23 = v14 - v15;
      if ( (__int64)(v14 - v15) > 0x7FFFFFFF )
        goto LABEL_25;
    }
    result = 1;
    if ( v23 < (__int64)0xFFFFFFFF80000000LL || (int)v23 < 0 )
      return result;
    if ( v18 )
    {
LABEL_24:
      v21 = memcmp(v17, v16, v18);
      if ( v21 )
        goto LABEL_28;
    }
LABEL_25:
    v22 = v15 - v14;
    if ( v22 > 0x7FFFFFFF )
      return *(_DWORD *)(a1 + 32) < *(_DWORD *)(a2 + 32);
    if ( v22 < (__int64)0xFFFFFFFF80000000LL )
      return 0;
    v21 = v22;
LABEL_28:
    result = 0;
    if ( v21 < 0 )
      return result;
    return *(_DWORD *)(a1 + 32) < *(_DWORD *)(a2 + 32);
  }
  if ( !v3 )
    return result;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(const void **)a1;
  v7 = *(const void **)a2;
  v8 = v5;
  if ( v4 <= v5 )
    v8 = *(_QWORD *)(a1 + 8);
  if ( v8 )
  {
    n = v8;
    s1 = *(void **)a2;
    s2 = *(void **)a1;
    v9 = memcmp(*(const void **)a1, *(const void **)a2, v8);
    v6 = s2;
    v7 = s1;
    v10 = v9 < 0;
    v8 = n;
    if ( v9 )
    {
      result = v3;
      if ( v10 )
        return result;
    }
    else
    {
      v11 = v4 - v5;
      if ( (__int64)(v4 - v5) <= 0x7FFFFFFF )
        goto LABEL_8;
    }
LABEL_11:
    v12 = memcmp(v7, v6, v8);
    if ( v12 )
      goto LABEL_15;
LABEL_12:
    v13 = v5 - v4;
    if ( v13 > 0x7FFFFFFF )
      return *(_BYTE *)(a1 + 32) < *(_BYTE *)(a2 + 32);
    if ( v13 >= (__int64)0xFFFFFFFF80000000LL )
    {
      v12 = v13;
LABEL_15:
      result = 0;
      if ( v12 < 0 )
        return result;
      return *(_BYTE *)(a1 + 32) < *(_BYTE *)(a2 + 32);
    }
    return 0;
  }
  v11 = v4 - v5;
  if ( (__int64)(v4 - v5) > 0x7FFFFFFF )
    goto LABEL_12;
LABEL_8:
  result = v3;
  if ( v11 >= (__int64)0xFFFFFFFF80000000LL && (int)v11 >= 0 )
  {
    if ( !v8 )
      goto LABEL_12;
    goto LABEL_11;
  }
  return result;
}
