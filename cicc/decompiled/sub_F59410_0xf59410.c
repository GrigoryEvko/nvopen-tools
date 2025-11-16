// Function: sub_F59410
// Address: 0xf59410
//
__int64 __fastcall sub_F59410(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  __int64 v4; // r13
  char v5; // dl
  unsigned __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 *v11; // rcx
  __int64 v12; // r15
  bool v13; // zf
  __int64 **v14; // r14
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  __int64 **v18; // rbx
  __int64 **j; // rcx
  __int64 *v20; // rdx
  __int64 result; // rax
  __int64 *v22; // rax
  __int64 *v23; // r15
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *k; // rbx
  __int64 **v28; // rsi
  __int64 **v29; // [rsp+8h] [rbp-68h]
  __int64 *v30; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v31[10]; // [rsp+20h] [rbp-50h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    v10 = (__int64 *)(a1 + 16);
    v11 = (__int64 *)(a1 + 48);
    if ( !v5 )
    {
      v7 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
  }
  else
  {
    v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v2 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v10 = (__int64 *)(a1 + 16);
      v11 = (__int64 *)(a1 + 48);
      if ( !v5 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v8 = 8LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v2 = 64;
        v8 = 512;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v12 = 8 * v7;
        v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v14 = (__int64 **)(v4 + v12);
        if ( v13 )
        {
          v15 = *(_QWORD **)(a1 + 16);
          v16 = *(unsigned int *)(a1 + 24);
        }
        else
        {
          v15 = (_QWORD *)(a1 + 16);
          v16 = 4;
        }
        for ( i = &v15[v16]; i != v15; ++v15 )
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
            if ( v14 == ++v18 )
              return sub_C7D6A0(v4, v12, 8);
          }
          v29 = j;
          sub_F592A0(a1, v18, j);
          v20 = *v18++;
          j = v29;
          *(_QWORD *)v31[0] = v20;
        }
        return sub_C7D6A0(v4, v12, 8);
      }
      v10 = (__int64 *)(a1 + 16);
      v11 = (__int64 *)(a1 + 48);
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
      ++v23;
    }
    ++v22;
  }
  while ( v22 != v11 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v25 = sub_C7D670(8LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v25;
  }
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v26 = 4;
  if ( v13 )
  {
    v10 = *(__int64 **)(a1 + 16);
    v26 = *(unsigned int *)(a1 + 24);
  }
  for ( result = (__int64)&v10[v26]; (__int64 *)result != v10; ++v10 )
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
      if ( v23 == ++k )
        return result;
    }
    v28 = (__int64 **)k++;
    sub_F592A0(a1, v28, &v30);
    *v30 = *(k - 1);
    result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
  }
  return result;
}
