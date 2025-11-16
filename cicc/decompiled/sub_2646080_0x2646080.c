// Function: sub_2646080
// Address: 0x2646080
//
_QWORD *__fastcall sub_2646080(__int64 a1, int a2)
{
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 *v9; // rdi
  _QWORD *i; // rdx
  __int64 *v11; // rax
  __int64 v12; // rdx
  int v13; // ecx
  int v14; // esi
  __int64 v15; // r10
  int v16; // r14d
  _QWORD *v17; // r11
  unsigned int v18; // ecx
  _QWORD *v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(8LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = 8 * v3;
    *(_QWORD *)(a1 + 16) = 0;
    v9 = (__int64 *)(v4 + v8);
    for ( i = &result[v7]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != (__int64 *)v4 )
    {
      v11 = (__int64 *)v4;
      do
      {
        v12 = *v11;
        if ( *v11 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *v11;
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = (_QWORD *)(v15 + 8LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( !v17 && v20 == -8192 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (_QWORD *)(v15 + 8LL * v18);
              v20 = *v19;
              if ( v12 == *v19 )
                goto LABEL_14;
              ++v16;
            }
            if ( v17 )
              v19 = v17;
          }
LABEL_14:
          *v19 = v12;
          ++*(_DWORD *)(a1 + 16);
        }
        ++v11;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[v21]; j != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
