// Function: sub_B23EA0
// Address: 0xb23ea0
//
__int64 __fastcall sub_B23EA0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 *v4; // r13
  char v5; // r14
  unsigned int v6; // eax
  __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 *v11; // rcx
  bool v12; // zf
  __int64 *v13; // r14
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 result; // rax
  __int64 *v19; // rax
  __int64 *v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 *k; // rbx
  __int64 v25; // [rsp+8h] [rbp-88h]
  __int64 *v26; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v27[14]; // [rsp+20h] [rbp-70h] BYREF

  v2 = a2;
  v4 = *(__int64 **)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    v10 = (__int64 *)(a1 + 16);
    v11 = (__int64 *)(a1 + 80);
    if ( !v5 )
    {
      v7 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
  }
  else
  {
    v6 = sub_AF1560(a2 - 1);
    v2 = v6;
    if ( v6 > 0x40 )
    {
      v10 = (__int64 *)(a1 + 16);
      v11 = (__int64 *)(a1 + 80);
      if ( !v5 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v8 = 8LL * v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v8 = 512;
        v2 = 64;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v25 = 8 * v7;
        v13 = &v4[v7];
        if ( v12 )
        {
          v14 = *(_QWORD **)(a1 + 16);
          v15 = *(unsigned int *)(a1 + 24);
        }
        else
        {
          v14 = (_QWORD *)(a1 + 16);
          v15 = 8;
        }
        for ( i = &v14[v15]; i != v14; ++v14 )
        {
          if ( v14 )
            *v14 = -4096;
        }
        for ( j = v4; v13 != j; ++j )
        {
          if ( *j != -4096 && *j != -8192 )
          {
            sub_B1D950(a1, j, v27);
            *(_QWORD *)v27[0] = *j;
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          }
        }
        return sub_C7D6A0(v4, v25, 8);
      }
      v10 = (__int64 *)(a1 + 16);
      v11 = (__int64 *)(a1 + 80);
      v2 = 64;
    }
  }
  v19 = v10;
  v20 = v27;
  do
  {
    v21 = *v19;
    if ( *v19 != -4096 && v21 != -8192 )
    {
      if ( v20 )
        *v20 = v21;
      ++v20;
    }
    ++v19;
  }
  while ( v19 != v11 );
  if ( v2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v22 = sub_C7D670(8LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v22;
  }
  v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v23 = 8;
  if ( v12 )
  {
    v10 = *(__int64 **)(a1 + 16);
    v23 = *(unsigned int *)(a1 + 24);
  }
  for ( result = (__int64)&v10[v23]; (__int64 *)result != v10; ++v10 )
  {
    if ( v10 )
      *v10 = -4096;
  }
  for ( k = v27; v20 != k; ++k )
  {
    result = *k;
    if ( *k != -8192 && result != -4096 )
    {
      sub_B1D950(a1, k, &v26);
      *v26 = *k;
      result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
      *(_DWORD *)(a1 + 8) = result;
    }
  }
  return result;
}
