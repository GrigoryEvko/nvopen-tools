// Function: sub_1A9AF30
// Address: 0x1a9af30
//
__int64 __fastcall sub_1A9AF30(__int64 a1, unsigned int a2)
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
  __int64 *v16; // rax
  __int64 v17; // r13
  __int64 *v18; // rsi
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v23; // r9
  int v24; // r10d
  int v25; // r14d
  _QWORD *v26; // r13
  unsigned int v27; // edi
  _QWORD *v28; // rcx
  __int64 v29; // r11
  __int64 v30; // rdx
  int v31; // ecx
  __int64 *v32; // [rsp+18h] [rbp-B8h] BYREF
  _BYTE v33[176]; // [rsp+20h] [rbp-B0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v6 = *(__int64 **)(a1 + 16);
    v17 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
    v5 = sub_1454B60(a2 - 1);
    v6 = *(__int64 **)(a1 + 16);
    v7 = v5;
    if ( v5 > 0x40 )
    {
      v8 = 2LL * v5;
      if ( v4 )
      {
LABEL_5:
        v9 = (_QWORD *)(a1 + 16);
        v10 = (__int64 *)v33;
        v11 = (__int64 *)(a1 + 16);
        v12 = (__int64 *)v33;
        do
        {
          v13 = *v11;
          if ( *v11 != -8 && v13 != -16 )
          {
            if ( v12 )
              *v12 = v13;
            v12 += 2;
            *(v12 - 1) = v11[1];
          }
          v11 += 2;
        }
        while ( v11 != (__int64 *)(a1 + 144) );
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
        if ( v12 != (__int64 *)v33 )
        {
          do
          {
            result = *v10;
            if ( *v10 != -8 && result != -16 )
            {
              sub_1A97280(a1, v10, &v32);
              v16 = v32;
              *v32 = *v10;
              v16[1] = v10[1];
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
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v18 = &v6[2 * v17];
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
  for ( j = v6; v18 != j; j += 2 )
  {
    v30 = *j;
    if ( *j != -16 && v30 != -8 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v23 = a1 + 16;
        v24 = 7;
      }
      else
      {
        v31 = *(_DWORD *)(a1 + 24);
        v23 = *(_QWORD *)(a1 + 16);
        if ( !v31 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v24 = v31 - 1;
      }
      v25 = 1;
      v26 = 0;
      v27 = v24 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v28 = (_QWORD *)(v23 + 16LL * v27);
      v29 = *v28;
      if ( *v28 != v30 )
      {
        while ( v29 != -8 )
        {
          if ( v29 == -16 && !v26 )
            v26 = v28;
          v27 = v24 & (v25 + v27);
          v28 = (_QWORD *)(v23 + 16LL * v27);
          v29 = *v28;
          if ( v30 == *v28 )
            goto LABEL_37;
          ++v25;
        }
        if ( v26 )
          v28 = v26;
      }
LABEL_37:
      *v28 = v30;
      v28[1] = j[1];
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v6);
}
