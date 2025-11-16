// Function: sub_1AED960
// Address: 0x1aed960
//
__int64 *__fastcall sub_1AED960(__int64 a1, unsigned int a2)
{
  __int64 *result; // rax
  char v3; // dl
  __int64 **v4; // r13
  unsigned __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // r15
  _QWORD *v8; // rbx
  __int64 **v9; // rax
  __int64 **v10; // r14
  __int64 *v11; // rdx
  __int64 v12; // rax
  bool v13; // zf
  _QWORD *v14; // rdx
  __int64 **i; // rbx
  __int64 v16; // rbx
  __int64 **v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *j; // rdx
  __int64 **k; // rbx
  int v22; // [rsp+Ch] [rbp-64h]
  int v23; // [rsp+Ch] [rbp-64h]
  __int64 **v24; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v25[10]; // [rsp+20h] [rbp-50h] BYREF

  result = (__int64 *)*(unsigned __int8 *)(a1 + 8);
  v3 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v3 )
      return result;
    v4 = *(__int64 ***)(a1 + 16);
    v16 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = (unsigned __int8)result | 1;
  }
  else
  {
    v4 = *(__int64 ***)(a1 + 16);
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v6 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v7 = (unsigned int)v5;
      if ( v3 )
      {
LABEL_5:
        v8 = (_QWORD *)(a1 + 16);
        v9 = (__int64 **)(a1 + 16);
        v10 = v25;
        do
        {
          v11 = *v9;
          if ( *v9 != (__int64 *)-8LL && v11 != (__int64 *)-16LL )
          {
            if ( v10 )
              *v10 = v11;
            ++v10;
          }
          ++v9;
        }
        while ( v9 != (__int64 **)(a1 + 48) );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v22 = v6;
        v12 = sub_22077B0(v7 * 8);
        v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        *(_QWORD *)(a1 + 16) = v12;
        *(_DWORD *)(a1 + 24) = v22;
        if ( v13 )
        {
          v8 = (_QWORD *)v12;
          v14 = (_QWORD *)v12;
        }
        else
        {
          v14 = (_QWORD *)(a1 + 16);
          v7 = 4;
        }
        result = &v8[v7];
        while ( 1 )
        {
          if ( v14 )
            *v8 = -8;
          if ( result == ++v8 )
            break;
          v14 = v8;
        }
        for ( i = v25; v10 != i; ++i )
        {
          result = *i;
          if ( *i != (__int64 *)-16LL && result != (__int64 *)-8LL )
          {
            sub_1AED800(a1, i, (__int64 **)&v24);
            *v24 = *i;
            result = (__int64 *)((2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u);
            *(_DWORD *)(a1 + 8) = (_DWORD)result;
          }
        }
        return result;
      }
      v16 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v3 )
      {
        v7 = 64;
        v6 = 64;
        goto LABEL_5;
      }
      v16 = *(unsigned int *)(a1 + 24);
      v7 = 64;
      v6 = 64;
    }
    v23 = v6;
    *(_QWORD *)(a1 + 16) = sub_22077B0(v7 * 8);
    *(_DWORD *)(a1 + 24) = v23;
  }
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v17 = &v4[v16];
  if ( v13 )
  {
    v18 = *(_QWORD **)(a1 + 16);
    v19 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v18 = (_QWORD *)(a1 + 16);
    v19 = 4;
  }
  for ( j = &v18[v19]; j != v18; ++v18 )
  {
    if ( v18 )
      *v18 = -8;
  }
  for ( k = v4; v17 != k; ++k )
  {
    if ( *k != (__int64 *)-16LL && *k != (__int64 *)-8LL )
    {
      sub_1AED800(a1, k, v25);
      *v25[0] = (__int64)*k;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return (__int64 *)j___libc_free_0(v4);
}
