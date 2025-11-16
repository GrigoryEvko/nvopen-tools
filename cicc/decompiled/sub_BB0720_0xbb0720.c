// Function: sub_BB0720
// Address: 0xbb0720
//
_QWORD *__fastcall sub_BB0720(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 *v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 *v10; // rdi
  _QWORD *i; // rdx
  __int64 *v12; // rax
  int v13; // ecx
  int v14; // esi
  unsigned __int64 v15; // rdx
  __int64 v16; // r11
  _QWORD *v17; // rbx
  int v18; // r14d
  unsigned int v19; // r10d
  _QWORD *v20; // r9
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdx
  _QWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(8LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 8 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v5[v4];
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -8;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        while ( (*v12 & 0xFFFFFFFFFFFFFFF0LL) == 0xFFFFFFFFFFFFFFF0LL )
        {
          if ( v10 == ++v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *v12;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 0;
        v18 = 1;
        v19 = v15 & (v13 - 1);
        v20 = (_QWORD *)(v16 + 8LL * v19);
        v21 = *v20 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v15 != v21 )
        {
          while ( v21 != -8 )
          {
            if ( v21 == -16 && !v17 )
              v17 = v20;
            v19 = v14 & (v18 + v19);
            v20 = (_QWORD *)(v16 + 8LL * v19);
            v21 = *v20 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v15 == v21 )
              goto LABEL_14;
            ++v18;
          }
          if ( v17 )
            v20 = v17;
        }
LABEL_14:
        v22 = *v12++;
        *v20 = v22;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[v23]; j != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
