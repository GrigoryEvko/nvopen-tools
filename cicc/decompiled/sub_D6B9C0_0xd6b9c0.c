// Function: sub_D6B9C0
// Address: 0xd6b9c0
//
__int64 __fastcall sub_D6B9C0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r14
  char v5; // bl
  unsigned int v6; // eax
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // r13
  __int64 *v11; // rcx
  __int64 *v12; // r13
  bool v13; // zf
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 result; // rax
  __int64 *v19; // rbx
  __int64 *v20; // rax
  __int64 *v21; // r14
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-88h]
  __int64 *v26; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v27[14]; // [rsp+20h] [rbp-70h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
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
        v8 = 16LL * v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v8 = 1024;
        v2 = 64;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v25 = 16 * v7;
        v12 = (__int64 *)(v4 + 16 * v7);
        v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        if ( v13 )
        {
          v14 = *(_QWORD **)(a1 + 16);
          v15 = 2LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v14 = (_QWORD *)(a1 + 16);
          v15 = 8;
        }
        for ( i = &v14[v15]; i != v14; v14 += 2 )
        {
          if ( v14 )
            *v14 = -4096;
        }
        for ( j = (__int64 *)v4; v12 != j; j += 2 )
        {
          if ( *j != -4096 && *j != -8192 )
          {
            sub_D69D00(a1, j, v27);
            *(_QWORD *)v27[0] = *j;
            *(_QWORD *)(v27[0] + 8LL) = j[1];
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
  v19 = v27;
  v20 = v10;
  v21 = v27;
  do
  {
    v22 = *v20;
    if ( *v20 != -4096 && v22 != -8192 )
    {
      if ( v21 )
        *v21 = v22;
      v21 += 2;
      *(v21 - 1) = v20[1];
    }
    v20 += 2;
  }
  while ( v20 != v11 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v23 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v23;
  }
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v24 = 8;
  if ( v13 )
  {
    v10 = *(__int64 **)(a1 + 16);
    v24 = 2LL * *(unsigned int *)(a1 + 24);
  }
  for ( result = (__int64)&v10[v24]; (__int64 *)result != v10; v10 += 2 )
  {
    if ( v10 )
      *v10 = -4096;
  }
  if ( v21 != v27 )
  {
    do
    {
      result = *v19;
      if ( *v19 != -8192 && result != -4096 )
      {
        sub_D69D00(a1, v19, &v26);
        *v26 = *v19;
        v26[1] = v19[1];
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
      }
      v19 += 2;
    }
    while ( v21 != v19 );
  }
  return result;
}
