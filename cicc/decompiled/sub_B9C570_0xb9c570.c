// Function: sub_B9C570
// Address: 0xb9c570
//
_QWORD *__fastcall sub_B9C570(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 *v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 *v10; // rsi
  _QWORD *i; // rdx
  __int64 *v12; // rax
  __int64 v13; // rdx
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // r11
  unsigned int v17; // r10d
  _QWORD *v18; // rdi
  __int64 v19; // r8
  int v20; // r14d
  _QWORD *v21; // rbx
  __int64 v22; // rdx
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
        *result = -4096;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
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
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = v15 & *(_DWORD *)(v13 + 4);
          v18 = (_QWORD *)(v16 + 8LL * v17);
          v19 = *v18;
          if ( v13 != *v18 )
          {
            v20 = 1;
            v21 = 0;
            while ( v19 != -4096 )
            {
              if ( v19 != -8192 || v21 )
                v18 = v21;
              v17 = v15 & (v20 + v17);
              v19 = *(_QWORD *)(v16 + 8LL * v17);
              if ( v13 == v19 )
              {
                v18 = (_QWORD *)(v16 + 8LL * v17);
                goto LABEL_22;
              }
              ++v20;
              v21 = v18;
              v18 = (_QWORD *)(v16 + 8LL * v17);
            }
            if ( v21 )
              v18 = v21;
          }
LABEL_22:
          *v18 = v13;
          ++*(_DWORD *)(a1 + 16);
        }
        ++v12;
      }
      while ( v10 != v12 );
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
        *result = -4096;
    }
  }
  return result;
}
