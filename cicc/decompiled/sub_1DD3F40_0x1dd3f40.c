// Function: sub_1DD3F40
// Address: 0x1dd3f40
//
__int64 __fastcall sub_1DD3F40(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  unsigned __int64 v5; // rax
  int v6; // r14d
  __int64 v7; // rdi
  _DWORD *v8; // rsi
  _DWORD *v9; // rax
  _DWORD *v10; // r12
  _DWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  _DWORD *v14; // rdx
  _DWORD *v15; // rdi
  int v16; // edx
  __int64 v17; // r9
  int v18; // edi
  int v19; // r11d
  _DWORD *v20; // r10
  unsigned int v21; // ecx
  _DWORD *v22; // rsi
  int v23; // r8d
  int *v24; // r12
  __int64 v25; // r13
  bool v26; // zf
  int *v27; // rsi
  _DWORD *v28; // rax
  __int64 v29; // rdx
  _DWORD *i; // rdx
  int *j; // rax
  int v32; // edx
  __int64 v33; // r11
  int v34; // r8d
  int v35; // r14d
  _DWORD *v36; // r13
  unsigned int v37; // ecx
  _DWORD *v38; // rdi
  int v39; // r10d
  int v40; // ecx
  __int64 v41; // rax
  int v42; // ecx
  _BYTE v43[80]; // [rsp+10h] [rbp-50h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v24 = *(int **)(a1 + 16);
    v25 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
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
      if ( v4 )
      {
LABEL_5:
        v8 = (_DWORD *)(a1 + 48);
        v9 = (_DWORD *)(a1 + 16);
        v10 = v43;
        do
        {
          while ( (unsigned int)(*v9 + 0x7FFFFFFF) > 0xFFFFFFFD )
          {
            if ( ++v9 == v8 )
              goto LABEL_11;
          }
          if ( v10 )
            *v10 = *v9;
          ++v9;
          ++v10;
        }
        while ( v9 != v8 );
LABEL_11:
        *(_BYTE *)(a1 + 8) &= ~1u;
        v11 = (_DWORD *)sub_22077B0(v7 * 4);
        v12 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 16) = v11;
        v13 = v12 & 1;
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 8) = v13;
        if ( (_BYTE)v13 )
        {
          v11 = (_DWORD *)(a1 + 16);
          v7 = 8;
        }
        v14 = v11;
        v15 = &v11[v7];
        while ( 1 )
        {
          if ( v14 )
            *v11 = 0x7FFFFFFF;
          if ( v15 == ++v11 )
            break;
          v14 = v11;
        }
        for ( result = (__int64)v43;
              v10 != (_DWORD *)result;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v16 = *(_DWORD *)result;
            if ( (unsigned int)(*(_DWORD *)result + 0x7FFFFFFF) <= 0xFFFFFFFD )
              break;
            result += 4;
            if ( v10 == (_DWORD *)result )
              return result;
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v17 = a1 + 16;
            v18 = 7;
          }
          else
          {
            v42 = *(_DWORD *)(a1 + 24);
            v17 = *(_QWORD *)(a1 + 16);
            if ( !v42 )
              goto LABEL_71;
            v18 = v42 - 1;
          }
          v19 = 1;
          v20 = 0;
          v21 = v18 & (37 * v16);
          v22 = (_DWORD *)(v17 + 4LL * v21);
          v23 = *v22;
          if ( v16 != *v22 )
          {
            while ( v23 != 0x7FFFFFFF )
            {
              if ( v23 == 0x80000000 && !v20 )
                v20 = v22;
              v21 = v18 & (v19 + v21);
              v22 = (_DWORD *)(v17 + 4LL * v21);
              v23 = *v22;
              if ( v16 == *v22 )
                goto LABEL_25;
              ++v19;
            }
            if ( v20 )
              v22 = v20;
          }
LABEL_25:
          *v22 = v16;
          result += 4;
        }
        return result;
      }
      v24 = *(int **)(a1 + 16);
      v25 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v7 = 64;
        v6 = 64;
        goto LABEL_5;
      }
      v24 = *(int **)(a1 + 16);
      v25 = *(unsigned int *)(a1 + 24);
      v6 = 64;
      v7 = 64;
    }
    v41 = sub_22077B0(v7 * 4);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v41;
  }
  v26 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v27 = &v24[v25];
  if ( v26 )
  {
    v28 = *(_DWORD **)(a1 + 16);
    v29 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v28 = (_DWORD *)(a1 + 16);
    v29 = 8;
  }
  for ( i = &v28[v29]; i != v28; ++v28 )
  {
    if ( v28 )
      *v28 = 0x7FFFFFFF;
  }
  for ( j = v24; v27 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v32 = *j;
      if ( (unsigned int)(*j + 0x7FFFFFFF) <= 0xFFFFFFFD )
        break;
      if ( v27 == ++j )
        return j___libc_free_0(v24);
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v33 = a1 + 16;
      v34 = 7;
    }
    else
    {
      v40 = *(_DWORD *)(a1 + 24);
      v33 = *(_QWORD *)(a1 + 16);
      if ( !v40 )
      {
LABEL_71:
        MEMORY[0] = 0;
        BUG();
      }
      v34 = v40 - 1;
    }
    v35 = 1;
    v36 = 0;
    v37 = v34 & (37 * v32);
    v38 = (_DWORD *)(v33 + 4LL * v37);
    v39 = *v38;
    if ( *v38 != v32 )
    {
      while ( v39 != 0x7FFFFFFF )
      {
        if ( !v36 && v39 == 0x80000000 )
          v36 = v38;
        v37 = v34 & (v35 + v37);
        v38 = (_DWORD *)(v33 + 4LL * v37);
        v39 = *v38;
        if ( v32 == *v38 )
          goto LABEL_42;
        ++v35;
      }
      if ( v36 )
        v38 = v36;
    }
LABEL_42:
    *v38 = v32;
    ++j;
  }
  return j___libc_free_0(v24);
}
