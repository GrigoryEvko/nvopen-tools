// Function: sub_1B989E0
// Address: 0x1b989e0
//
__int64 *__fastcall sub_1B989E0(__int64 a1, unsigned int a2)
{
  __int64 *result; // rax
  char v4; // bl
  unsigned int v5; // eax
  __int64 v6; // r14
  int v7; // ecx
  __int64 v8; // r15
  __int64 *v9; // rbx
  __int64 **v10; // rax
  __int64 **v11; // r14
  __int64 *v12; // rdx
  bool v13; // zf
  __int64 *v14; // rdi
  __int64 **i; // rbx
  __int64 v16; // r13
  __int64 v17; // r13
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *j; // rdx
  __int64 k; // rbx
  __int64 *v22; // rax
  __int64 *v23; // rax
  int v24; // [rsp+Ch] [rbp-84h]
  int v25; // [rsp+Ch] [rbp-84h]
  __int64 *v26; // [rsp+18h] [rbp-78h] BYREF
  __int64 *v27[14]; // [rsp+20h] [rbp-70h] BYREF

  result = (__int64 *)*(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v6 = *(_QWORD *)(a1 + 16);
    v16 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = (unsigned __int8)result | 1;
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
        v9 = (__int64 *)(a1 + 16);
        v10 = (__int64 **)(a1 + 16);
        v11 = v27;
        do
        {
          v12 = *v10;
          if ( *v10 != (__int64 *)-8LL && v12 != (__int64 *)-16LL )
          {
            if ( v11 )
              *v11 = v12;
            v11 += 2;
            *(v11 - 1) = v10[1];
          }
          v10 += 2;
        }
        while ( v10 != (__int64 **)(a1 + 80) );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v24 = v7;
        result = (__int64 *)sub_22077B0(v8 * 8);
        v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        *(_QWORD *)(a1 + 16) = result;
        *(_DWORD *)(a1 + 24) = v24;
        if ( v13 )
        {
          v9 = result;
        }
        else
        {
          result = (__int64 *)(a1 + 16);
          v8 = 8;
        }
        v14 = &v9[v8];
        while ( 1 )
        {
          if ( result )
            *v9 = -8;
          v9 += 2;
          if ( v14 == v9 )
            break;
          result = v9;
        }
        for ( i = v27; v11 != i; i += 2 )
        {
          result = *i;
          if ( *i != (__int64 *)-8LL && result != (__int64 *)-16LL )
          {
            sub_1B98860(a1, i, &v26);
            v23 = v26;
            *v26 = (__int64)*i;
            v23[1] = (__int64)i[1];
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
      if ( v4 )
      {
        v8 = 128;
        v7 = 64;
        goto LABEL_5;
      }
      v16 = *(unsigned int *)(a1 + 24);
      v8 = 128;
      v7 = 64;
    }
    v25 = v7;
    *(_QWORD *)(a1 + 16) = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v25;
  }
  v17 = v6 + 16 * v16;
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v13 )
  {
    v18 = *(_QWORD **)(a1 + 16);
    v19 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v18 = (_QWORD *)(a1 + 16);
    v19 = 8;
  }
  for ( j = &v18[v19]; j != v18; v18 += 2 )
  {
    if ( v18 )
      *v18 = -8;
  }
  for ( k = v6; v17 != k; k += 16 )
  {
    if ( *(_QWORD *)k != -8 && *(_QWORD *)k != -16 )
    {
      sub_1B98860(a1, (__int64 **)k, v27);
      v22 = v27[0];
      *v27[0] = *(_QWORD *)k;
      v22[1] = *(_QWORD *)(k + 8);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return (__int64 *)j___libc_free_0(v6);
}
