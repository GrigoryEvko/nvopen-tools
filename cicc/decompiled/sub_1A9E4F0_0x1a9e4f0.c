// Function: sub_1A9E4F0
// Address: 0x1a9e4f0
//
__int64 __fastcall sub_1A9E4F0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // bl
  unsigned int v5; // eax
  __int64 *v6; // r13
  int v7; // esi
  __int64 v8; // r15
  _QWORD *v9; // r14
  __int64 *v10; // r13
  __int64 *v11; // rax
  __int64 *v12; // rbx
  __int64 v13; // rdx
  bool v14; // zf
  _QWORD *v15; // rdi
  __int64 v16; // rbx
  __int64 *v17; // r15
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 *v22; // [rsp+18h] [rbp-138h] BYREF
  _QWORD v23[38]; // [rsp+20h] [rbp-130h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x1F )
  {
    if ( v4 )
      return result;
    v6 = *(__int64 **)(a1 + 16);
    v16 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
    v5 = sub_1454B60(a2 - 1);
    v6 = *(__int64 **)(a1 + 16);
    v7 = v5;
    if ( v5 > 0x40 )
    {
      v8 = v5;
      if ( v4 )
      {
LABEL_5:
        v9 = (_QWORD *)(a1 + 16);
        v10 = v23;
        v11 = (__int64 *)(a1 + 16);
        v12 = v23;
        do
        {
          v13 = *v11;
          if ( *v11 != -8 && v13 != -16 )
          {
            if ( v12 )
              *v12 = v13;
            ++v12;
          }
          ++v11;
        }
        while ( v11 != (__int64 *)(a1 + 272) );
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
          v8 = 32;
        }
        v15 = &v9[v8];
        while ( 1 )
        {
          if ( result )
            *v9 = -8;
          if ( v15 == ++v9 )
            break;
          result = (__int64)v9;
        }
        if ( v12 != v23 )
        {
          do
          {
            result = *v10;
            if ( *v10 != -8 && result != -16 )
            {
              sub_1A97A00(a1, v10, &v22);
              *v22 = *v10;
              result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
              *(_DWORD *)(a1 + 8) = result;
            }
            ++v10;
          }
          while ( v12 != v10 );
        }
        return result;
      }
      v16 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 64;
        v7 = 64;
        goto LABEL_5;
      }
      v16 = *(unsigned int *)(a1 + 24);
      v8 = 64;
      v7 = 64;
    }
    *(_QWORD *)(a1 + 16) = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v7;
  }
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v17 = &v6[v16];
  if ( v14 )
  {
    v18 = *(_QWORD **)(a1 + 16);
    v19 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v18 = (_QWORD *)(a1 + 16);
    v19 = 32;
  }
  for ( i = &v18[v19]; i != v18; ++v18 )
  {
    if ( v18 )
      *v18 = -8;
  }
  for ( j = v6; v17 != j; ++j )
  {
    if ( *j != -8 && *j != -16 )
    {
      sub_1A97A00(a1, j, v23);
      *(_QWORD *)v23[0] = *j;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v6);
}
