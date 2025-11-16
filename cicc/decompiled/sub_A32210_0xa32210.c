// Function: sub_A32210
// Address: 0xa32210
//
_QWORD *__fastcall sub_A32210(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 *v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  _QWORD *v10; // rdx
  __int64 *i; // rdi
  __int64 *v12; // rax
  __int64 v13; // rdx
  int v14; // esi
  int v15; // esi
  __int64 v16; // r11
  int v17; // r15d
  _QWORD *v18; // r14
  unsigned int v19; // ecx
  _QWORD *v20; // r9
  __int64 v21; // r10
  __int64 v22; // rdx
  _QWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 8LL * v4;
    v10 = &result[v8];
    for ( i = &v5[(unsigned __int64)v9 / 8]; v10 != result; ++result )
    {
      if ( result )
        *result = -1;
    }
    if ( i != v5 )
    {
      v12 = v5;
      do
      {
        while ( 1 )
        {
          v13 = *v12;
          if ( (unsigned __int64)*v12 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          if ( i == ++v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 1;
        v18 = 0;
        v19 = v15 & (((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * v13));
        v20 = (_QWORD *)(v16 + 8LL * v19);
        v21 = *v20;
        if ( v13 != *v20 )
        {
          while ( v21 != -1 )
          {
            if ( !v18 && v21 == -2 )
              v18 = v20;
            v19 = v15 & (v17 + v19);
            v20 = (_QWORD *)(v16 + 8LL * v19);
            v21 = *v20;
            if ( v13 == *v20 )
              goto LABEL_14;
            ++v17;
          }
          if ( v18 )
            v20 = v18;
        }
LABEL_14:
        ++v12;
        *v20 = v13;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( i != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[v22]; j != result; ++result )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
