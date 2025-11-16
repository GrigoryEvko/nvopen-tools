// Function: sub_15D6B70
// Address: 0x15d6b70
//
__int64 __fastcall sub_15D6B70(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // r14
  unsigned int v5; // eax
  __int64 v6; // r13
  int v7; // esi
  __int64 v8; // rbx
  _QWORD *v9; // r13
  __int64 *v10; // r15
  __int64 v11; // rax
  __int64 *v12; // r14
  __int64 v13; // rdx
  bool v14; // zf
  _QWORD *v15; // rdi
  __int64 *v16; // rax
  __int64 v17; // r15
  __int64 v18; // r15
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v23; // rax
  __int64 *v24; // [rsp+18h] [rbp-B8h] BYREF
  _QWORD v25[22]; // [rsp+20h] [rbp-B0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v6 = *(_QWORD *)(a1 + 16);
    v17 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
    v5 = sub_1454B60(a2 - 1);
    v6 = *(_QWORD *)(a1 + 16);
    v7 = v5;
    if ( v5 > 0x40 )
    {
      v8 = 2LL * v5;
      if ( v4 )
      {
LABEL_5:
        v9 = (_QWORD *)(a1 + 16);
        v10 = v25;
        v11 = a1 + 16;
        v12 = v25;
        do
        {
          v13 = *(_QWORD *)v11;
          if ( *(_QWORD *)v11 != -8 && v13 != -16 )
          {
            if ( v12 )
              *v12 = v13;
            v12 += 2;
            *((_DWORD *)v12 - 2) = *(_DWORD *)(v11 + 8);
          }
          v11 += 16;
        }
        while ( v11 != a1 + 144 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        result = sub_22077B0(v8 * 8);
        v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        *(_QWORD *)(a1 + 16) = result;
        *(_DWORD *)(a1 + 24) = v7;
        if ( v14 )
        {
          v9 = (_QWORD *)result;
        }
        else
        {
          result = a1 + 16;
          v8 = 16;
        }
        v15 = &v9[v8];
        while ( 1 )
        {
          if ( result )
            *v9 = -8;
          v9 += 2;
          if ( v15 == v9 )
            break;
          result = (__int64)v9;
        }
        if ( v12 != v25 )
        {
          do
          {
            result = *v10;
            if ( *v10 != -8 && result != -16 )
            {
              sub_15D0950(a1, v10, &v24);
              v16 = v24;
              *v24 = *v10;
              *((_DWORD *)v16 + 2) = *((_DWORD *)v10 + 2);
              result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
              *(_DWORD *)(a1 + 8) = result;
            }
            v10 += 2;
          }
          while ( v12 != v10 );
        }
        return result;
      }
      v17 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 128;
        v7 = 64;
        goto LABEL_5;
      }
      v17 = *(unsigned int *)(a1 + 24);
      v8 = 128;
      v7 = 64;
    }
    *(_QWORD *)(a1 + 16) = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v7;
  }
  v18 = v6 + 16 * v17;
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v14 )
  {
    v19 = *(_QWORD **)(a1 + 16);
    v20 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v19 = (_QWORD *)(a1 + 16);
    v20 = 16;
  }
  for ( i = &v19[v20]; i != v19; v19 += 2 )
  {
    if ( v19 )
      *v19 = -8;
  }
  for ( j = v6; v18 != j; j += 16 )
  {
    if ( *(_QWORD *)j != -8 && *(_QWORD *)j != -16 )
    {
      sub_15D0950(a1, (__int64 *)j, v25);
      v23 = v25[0];
      *(_QWORD *)v25[0] = *(_QWORD *)j;
      *(_DWORD *)(v23 + 8) = *(_DWORD *)(j + 8);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v6);
}
