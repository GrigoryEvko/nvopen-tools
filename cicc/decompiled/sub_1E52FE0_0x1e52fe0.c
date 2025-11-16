// Function: sub_1E52FE0
// Address: 0x1e52fe0
//
__int64 __fastcall sub_1E52FE0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // r13
  unsigned int v5; // eax
  __int64 *v6; // r12
  int v7; // esi
  __int64 v8; // r15
  _QWORD *v9; // r13
  __int64 *v10; // r12
  __int64 *v11; // rax
  __int64 *v12; // r14
  __int64 v13; // rdx
  bool v14; // zf
  _QWORD *v15; // rdi
  __int64 v16; // r13
  __int64 *v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v22; // r10
  int v23; // r9d
  int v24; // r14d
  _QWORD *v25; // r13
  unsigned int v26; // ecx
  _QWORD *v27; // rdi
  __int64 v28; // r11
  __int64 v29; // rdx
  int v30; // ecx
  __int64 *v31; // [rsp+18h] [rbp-78h] BYREF
  _BYTE v32[112]; // [rsp+20h] [rbp-70h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
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
        v10 = (__int64 *)v32;
        v11 = (__int64 *)(a1 + 16);
        v12 = (__int64 *)v32;
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
        while ( v11 != (__int64 *)(a1 + 80) );
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
          v8 = 8;
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
        if ( v12 != (__int64 *)v32 )
        {
          do
          {
            result = *v10;
            if ( *v10 != -8 && result != -16 )
            {
              sub_1E492D0(a1, v10, &v31);
              *v31 = *v10;
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
    v19 = 8;
  }
  for ( i = &v18[v19]; i != v18; ++v18 )
  {
    if ( v18 )
      *v18 = -8;
  }
  for ( j = v6; v17 != j; ++j )
  {
    v29 = *j;
    if ( *j != -16 && v29 != -8 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v22 = a1 + 16;
        v23 = 7;
      }
      else
      {
        v30 = *(_DWORD *)(a1 + 24);
        v22 = *(_QWORD *)(a1 + 16);
        if ( !v30 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v23 = v30 - 1;
      }
      v24 = 1;
      v25 = 0;
      v26 = v23 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v27 = (_QWORD *)(v22 + 8LL * v26);
      v28 = *v27;
      if ( *v27 != v29 )
      {
        while ( v28 != -8 )
        {
          if ( v28 == -16 && !v25 )
            v25 = v27;
          v26 = v23 & (v24 + v26);
          v27 = (_QWORD *)(v22 + 8LL * v26);
          v28 = *v27;
          if ( v29 == *v27 )
            goto LABEL_37;
          ++v24;
        }
        if ( v25 )
          v27 = v25;
      }
LABEL_37:
      *v27 = v29;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v6);
}
