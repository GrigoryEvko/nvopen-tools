// Function: sub_E640F0
// Address: 0xe640f0
//
char __fastcall sub_E640F0(__int64 a1, __int64 a2)
{
  size_t v4; // r12
  size_t v5; // r13
  const void *v6; // rdi
  const void *v7; // rsi
  size_t v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // r12
  size_t v11; // r13
  size_t v12; // r12
  const void *v13; // r15
  const void *v14; // rsi
  size_t v15; // rdx
  void *s2; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(const void **)a1;
  v7 = *(const void **)a2;
  if ( v4 == v5 )
  {
    if ( !v4 || (v9 = memcmp(v6, v7, v4)) == 0 )
    {
      v11 = *(_QWORD *)(a1 + 40);
      v12 = *(_QWORD *)(a2 + 40);
      v13 = *(const void **)(a1 + 32);
      v14 = *(const void **)(a2 + 32);
      if ( v12 == v11 )
      {
        if ( !v12 || (s2 = *(void **)(a2 + 32), !memcmp(*(const void **)(a1 + 32), v14, *(_QWORD *)(a2 + 40))) )
        {
          LOBYTE(v9) = *(_DWORD *)(a1 + 48) < *(_DWORD *)(a2 + 48);
          return v9;
        }
        v9 = memcmp(v13, s2, v12);
        if ( !v9 )
          return v9;
      }
      else
      {
        v15 = *(_QWORD *)(a1 + 40);
        if ( v12 <= v11 )
          v15 = *(_QWORD *)(a2 + 40);
        if ( !v15 || (v9 = memcmp(*(const void **)(a1 + 32), v14, v15)) == 0 )
        {
          LOBYTE(v9) = v12 > v11;
          return v9;
        }
      }
    }
    goto LABEL_12;
  }
  v8 = v5;
  if ( v4 <= v5 )
    v8 = v4;
  if ( v8 )
  {
    v9 = memcmp(v6, v7, v8);
    if ( v9 )
    {
LABEL_12:
      v9 >>= 31;
      return v9;
    }
  }
  v10 = v4 - v5;
  LOBYTE(v9) = 0;
  if ( v10 <= 0x7FFFFFFF )
  {
    if ( v10 < (__int64)0xFFFFFFFF80000000LL )
      LOBYTE(v9) = 1;
    else
      return (unsigned int)v10 >> 31;
  }
  return v9;
}
