// Function: sub_342D120
// Address: 0x342d120
//
_DWORD *__fastcall sub_342D120(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  int *v10; // rdi
  _DWORD *i; // rdx
  int *v12; // rax
  int v13; // edx
  int v14; // ecx
  int v15; // esi
  __int64 v16; // r11
  int v17; // r14d
  _DWORD *v18; // rbx
  unsigned int v19; // r9d
  _DWORD *v20; // rcx
  int v21; // r10d
  int v22; // edx
  __int64 v23; // rdx
  _DWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_DWORD *)sub_C7D670(8LL * v6, 4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 8 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = (int *)(v5 + 8 * v4);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -1;
    }
    if ( v10 != (int *)v5 )
    {
      v12 = (int *)v5;
      do
      {
        while ( 1 )
        {
          v13 = *v12;
          if ( (unsigned int)*v12 <= 0xFFFFFFFD )
            break;
          v12 += 2;
          if ( v10 == v12 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 4);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = *v12;
          BUG();
        }
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 1;
        v18 = 0;
        v19 = (v14 - 1) & (37 * v13);
        v20 = (_DWORD *)(v16 + 8LL * v19);
        v21 = *v20;
        if ( v13 != *v20 )
        {
          while ( v21 != -1 )
          {
            if ( !v18 && v21 == -2 )
              v18 = v20;
            v19 = v15 & (v17 + v19);
            v20 = (_DWORD *)(v16 + 8LL * v19);
            v21 = *v20;
            if ( v13 == *v20 )
              goto LABEL_14;
            ++v17;
          }
          if ( v18 )
            v20 = v18;
        }
LABEL_14:
        v22 = *v12;
        v12 += 2;
        *v20 = v22;
        v20[1] = *(v12 - 1);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v23]; j != result; result += 2 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
