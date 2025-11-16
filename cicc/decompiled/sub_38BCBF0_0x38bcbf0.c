// Function: sub_38BCBF0
// Address: 0x38bcbf0
//
char __fastcall sub_38BCBF0(__int64 a1, __int64 a2)
{
  size_t v4; // r12
  size_t v5; // r13
  const void *v6; // rdi
  const void *v7; // rsi
  size_t v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // r12
  int v11; // eax
  size_t v12; // r12
  size_t v13; // r13
  const void *v14; // r15
  const void *v15; // rsi
  void *s2; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(const void **)a1;
  v7 = *(const void **)a2;
  if ( v4 == v5 )
  {
    if ( v4 )
    {
      v9 = memcmp(v6, v7, v4);
      if ( v9 )
        return v9 >> 31;
    }
    v12 = *(_QWORD *)(a1 + 40);
    v13 = *(_QWORD *)(a2 + 40);
    v14 = *(const void **)(a1 + 32);
    v15 = *(const void **)(a2 + 32);
    if ( v12 == v13 )
    {
      if ( !v12 || (s2 = *(void **)(a2 + 32), !memcmp(*(const void **)(a1 + 32), v15, *(_QWORD *)(a1 + 40))) )
      {
        LOBYTE(v11) = *(_DWORD *)(a1 + 48) < *(_DWORD *)(a2 + 48);
        return v11;
      }
      v9 = memcmp(v14, s2, v12);
      if ( !v9 )
      {
        LOBYTE(v11) = 0;
        return v11;
      }
      return v9 >> 31;
    }
    if ( v12 > v13 )
    {
      LOBYTE(v11) = 0;
      if ( !v13 )
        return v11;
      v9 = memcmp(*(const void **)(a1 + 32), v15, *(_QWORD *)(a2 + 40));
      if ( v9 )
        return v9 >> 31;
    }
    else if ( v12 )
    {
      v9 = memcmp(*(const void **)(a1 + 32), v15, *(_QWORD *)(a1 + 40));
      if ( v9 )
        return v9 >> 31;
    }
    else
    {
      LOBYTE(v11) = 0;
      if ( !v13 )
        return v11;
    }
    LOBYTE(v11) = v12 < v13;
    return v11;
  }
  v8 = v5;
  if ( v4 <= v5 )
    v8 = v4;
  if ( v8 )
  {
    v9 = memcmp(v6, v7, v8);
    if ( v9 )
      return v9 >> 31;
  }
  v10 = v4 - v5;
  LOBYTE(v11) = 0;
  if ( v10 <= 0x7FFFFFFF )
  {
    if ( v10 < (__int64)0xFFFFFFFF80000000LL )
      LOBYTE(v11) = 1;
    else
      return (unsigned int)v10 >> 31;
  }
  return v11;
}
