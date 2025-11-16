// Function: sub_2A8BBE0
// Address: 0x2a8bbe0
//
_QWORD *__fastcall sub_2A8BBE0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  _QWORD *v10; // rdx
  __int64 *i; // rcx
  __int64 *v12; // rax
  __int64 v13; // rdx
  int v14; // edi
  __int64 v15; // r10
  int v16; // edi
  __int64 v17; // r13
  unsigned int v18; // r8d
  _QWORD *v19; // r9
  __int64 v20; // r11
  int v21; // r15d
  _QWORD *v22; // r14
  __int64 v23; // rdx
  _QWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
  result = (_QWORD *)sub_C7D670(8LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 8LL * v4;
    v10 = &result[v8];
    for ( i = (__int64 *)(v5 + v9); v10 != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    if ( i != (__int64 *)v5 )
    {
      v12 = (__int64 *)v5;
      do
      {
        v13 = *v12;
        if ( *v12 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *v12;
            BUG();
          }
          v15 = *(_QWORD *)(v13 + 16);
          v16 = v14 - 1;
          v17 = *(_QWORD *)(a1 + 8);
          v18 = v16 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v19 = (_QWORD *)(v17 + 8LL * v18);
          v20 = *v19;
          if ( *v19 != -4096 )
          {
            v21 = 1;
            v22 = 0;
            while ( 1 )
            {
              if ( v20 == -8192 )
              {
                if ( v22 )
                  v19 = v22;
              }
              else
              {
                if ( v15 == *(_QWORD *)(v20 + 16) )
                  goto LABEL_21;
                v19 = v22;
              }
              v18 = v16 & (v21 + v18);
              v20 = *(_QWORD *)(v17 + 8LL * v18);
              if ( v20 == -4096 )
                break;
              ++v21;
              v22 = v19;
              v19 = (_QWORD *)(v17 + 8LL * v18);
            }
            if ( !v19 )
              v19 = (_QWORD *)(v17 + 8LL * v18);
          }
LABEL_21:
          *v19 = v13;
          ++*(_DWORD *)(a1 + 16);
        }
        ++v12;
      }
      while ( i != v12 );
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
        *result = -4096;
    }
  }
  return result;
}
