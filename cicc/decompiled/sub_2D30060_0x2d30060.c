// Function: sub_2D30060
// Address: 0x2d30060
//
_DWORD *__fastcall sub_2D30060(__int64 a1, int a2)
{
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  int *v9; // rsi
  _DWORD *i; // rdx
  int *v11; // rax
  int v12; // edx
  int v13; // ecx
  int v14; // ecx
  __int64 v15; // r10
  int v16; // r14d
  _DWORD *v17; // r11
  unsigned int v18; // edi
  _DWORD *v19; // r8
  int v20; // r9d
  __int64 v21; // rdx
  _DWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_C7D670(4LL * v5, 4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = 4 * v3;
    *(_QWORD *)(a1 + 16) = 0;
    v9 = (int *)(v4 + v8);
    for ( i = &result[v7]; i != result; ++result )
    {
      if ( result )
        *result = -1;
    }
    if ( v9 != (int *)v4 )
    {
      v11 = (int *)v4;
      do
      {
        while ( 1 )
        {
          v12 = *v11;
          if ( (unsigned int)*v11 <= 0xFFFFFFFD )
            break;
          if ( v9 == ++v11 )
            return (_DWORD *)sub_C7D6A0(v4, v8, 4);
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        v18 = v14 & (37 * v12);
        v19 = (_DWORD *)(v15 + 4LL * v18);
        v20 = *v19;
        if ( *v19 != v12 )
        {
          while ( v20 != -1 )
          {
            if ( !v17 && v20 == -2 )
              v17 = v19;
            v18 = v14 & (v16 + v18);
            v19 = (_DWORD *)(v15 + 4LL * v18);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_14;
            ++v16;
          }
          if ( v17 )
            v19 = v17;
        }
LABEL_14:
        ++v11;
        *v19 = v12;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v9 != v11 );
    }
    return (_DWORD *)sub_C7D6A0(v4, v8, 4);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[v21]; j != result; ++result )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
