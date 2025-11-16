// Function: sub_2ABF860
// Address: 0x2abf860
//
__int64 __fastcall sub_2ABF860(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r14
  char v5; // r13
  unsigned int v6; // eax
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 *v11; // rcx
  __int64 v12; // r15
  bool v13; // zf
  __int64 **v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  __int64 **v18; // rbx
  __int64 **j; // rcx
  __int64 *v20; // rdx
  __int64 result; // rax
  __int64 *v22; // rax
  __int64 *v23; // r14
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *k; // rbx
  __int64 **v28; // rsi
  __int64 **v29; // [rsp+8h] [rbp-88h]
  __int64 *v30; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v31[14]; // [rsp+20h] [rbp-70h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    v10 = (__int64 *)(a1 + 16);
    v11 = (__int64 *)(a1 + 80);
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
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
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 16LL * v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 1024;
        v2 = 64;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v12 = 16LL * v7;
        v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v14 = (__int64 **)(v4 + v12);
        if ( v13 )
        {
          v15 = *(_QWORD **)(a1 + 16);
          v16 = 2LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v15 = (_QWORD *)(a1 + 16);
          v16 = 8;
        }
        for ( i = &v15[v16]; i != v15; v15 += 2 )
        {
          if ( v15 )
            *v15 = -4096;
        }
        v18 = (__int64 **)v4;
        for ( j = (__int64 **)v31;
              v14 != v18;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( *v18 == (__int64 *)-4096LL || *v18 == (__int64 *)-8192LL )
          {
            v18 += 2;
            if ( v14 == v18 )
              return sub_C7D6A0(v4, v12, 8);
          }
          v29 = j;
          sub_2ABF6D0(a1, v18, j);
          v20 = *v18;
          v18 += 2;
          j = v29;
          *(_QWORD *)v31[0] = v20;
          *(_QWORD *)(v31[0] + 8LL) = *(v18 - 1);
        }
        return sub_C7D6A0(v4, v12, 8);
      }
      v10 = (__int64 *)(a1 + 16);
      v11 = (__int64 *)(a1 + 80);
      v2 = 64;
    }
  }
  v22 = v10;
  v23 = v31;
  do
  {
    v24 = *v22;
    if ( *v22 != -4096 && v24 != -8192 )
    {
      if ( v23 )
        *v23 = v24;
      v23 += 2;
      *(v23 - 1) = v22[1];
    }
    v22 += 2;
  }
  while ( v22 != v11 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v25 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v25;
  }
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v26 = 8;
  if ( v13 )
  {
    v10 = *(__int64 **)(a1 + 16);
    v26 = 2LL * *(unsigned int *)(a1 + 24);
  }
  for ( result = (__int64)&v10[v26]; (__int64 *)result != v10; v10 += 2 )
  {
    if ( v10 )
      *v10 = -4096;
  }
  for ( k = v31; v23 != k; *(_DWORD *)(a1 + 8) = result )
  {
    while ( 1 )
    {
      result = *k;
      if ( *k != -8192 && result != -4096 )
        break;
      k += 2;
      if ( v23 == k )
        return result;
    }
    v28 = (__int64 **)k;
    k += 2;
    sub_2ABF6D0(a1, v28, &v30);
    *v30 = *(k - 2);
    v30[1] = *(k - 1);
    result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
  }
  return result;
}
