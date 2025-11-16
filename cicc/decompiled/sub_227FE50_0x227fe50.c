// Function: sub_227FE50
// Address: 0x227fe50
//
__int64 __fastcall sub_227FE50(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r12
  char v5; // r14
  unsigned int v6; // eax
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // r13
  __int64 *v11; // rcx
  __int64 v12; // r9
  bool v13; // zf
  __int64 *v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v19; // r10
  int v20; // edi
  int v21; // r15d
  _QWORD *v22; // r14
  unsigned int v23; // esi
  _QWORD *v24; // rcx
  __int64 v25; // r11
  __int64 v26; // rdx
  int v27; // edi
  __int64 *v28; // r12
  __int64 *v29; // rax
  __int64 *v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 result; // rax
  __int64 *v35; // [rsp+18h] [rbp-78h] BYREF
  _BYTE v36[112]; // [rsp+20h] [rbp-70h] BYREF

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
    goto LABEL_26;
  }
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
LABEL_26:
    v28 = (__int64 *)v36;
    v29 = v10;
    v30 = (__int64 *)v36;
    do
    {
      v31 = *v29;
      if ( *v29 != -4096 && v31 != -8192 )
      {
        if ( v30 )
          *v30 = v31;
        v30 += 2;
        *(v30 - 1) = v29[1];
      }
      v29 += 2;
    }
    while ( v29 != v11 );
    if ( v2 > 4 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v32 = sub_C7D670(16LL * v2, 8);
      *(_DWORD *)(a1 + 24) = v2;
      *(_QWORD *)(a1 + 16) = v32;
    }
    v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    v33 = 8;
    if ( v13 )
    {
      v10 = *(__int64 **)(a1 + 16);
      v33 = 2LL * *(unsigned int *)(a1 + 24);
    }
    for ( result = (__int64)&v10[v33]; (__int64 *)result != v10; v10 += 2 )
    {
      if ( v10 )
        *v10 = -4096;
    }
    if ( v30 != (__int64 *)v36 )
    {
      do
      {
        result = *v28;
        if ( *v28 != -4096 && result != -8192 )
        {
          sub_227C450(a1, v28, &v35);
          *v35 = *v28;
          v35[1] = v28[1];
          result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
          *(_DWORD *)(a1 + 8) = result;
        }
        v28 += 2;
      }
      while ( v30 != v28 );
    }
    return result;
  }
  if ( v5 )
  {
    v10 = (__int64 *)(a1 + 16);
    v11 = (__int64 *)(a1 + 80);
    v2 = 64;
    goto LABEL_26;
  }
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
  v14 = (__int64 *)(v4 + v12);
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
  for ( j = (__int64 *)v4; v14 != j; j += 2 )
  {
    v26 = *j;
    if ( *j != -4096 && v26 != -8192 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v19 = a1 + 16;
        v20 = 3;
      }
      else
      {
        v27 = *(_DWORD *)(a1 + 24);
        v19 = *(_QWORD *)(a1 + 16);
        if ( !v27 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v20 = v27 - 1;
      }
      v21 = 1;
      v22 = 0;
      v23 = v20 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v24 = (_QWORD *)(v19 + 16LL * v23);
      v25 = *v24;
      if ( *v24 != v26 )
      {
        while ( v25 != -4096 )
        {
          if ( v25 == -8192 && !v22 )
            v22 = v24;
          v23 = v20 & (v21 + v23);
          v24 = (_QWORD *)(v19 + 16LL * v23);
          v25 = *v24;
          if ( v26 == *v24 )
            goto LABEL_18;
          ++v21;
        }
        if ( v22 )
          v24 = v22;
      }
LABEL_18:
      *v24 = v26;
      v24[1] = j[1];
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return sub_C7D6A0(v4, v12, 8);
}
