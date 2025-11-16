// Function: sub_2FC1820
// Address: 0x2fc1820
//
_QWORD *__fastcall sub_2FC1820(__int64 a1, int a2)
{
  unsigned __int64 v2; // rcx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r8
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r14
  _QWORD *i; // rcx
  __int64 v13; // rbx
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // rcx
  __int64 v17; // r9
  int v18; // r11d
  __int64 v19; // r10
  unsigned int v20; // esi
  __int64 v21; // r15
  __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  _QWORD *j; // rdx
  __int64 v26; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(80LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v9 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v10 = 80 * v4;
    v26 = 80 * v4;
    v11 = v5 + 80 * v4;
    for ( i = &result[10 * v9]; i != result; result += 10 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v11 != v5 )
    {
      v13 = v5;
      do
      {
        v14 = *(_QWORD *)v13;
        if ( *(_QWORD *)v13 != -8192 && v14 != -4096 )
        {
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
          {
            MEMORY[0] = *(_QWORD *)v13;
            BUG();
          }
          v16 = (unsigned int)(v15 - 1);
          v17 = *(_QWORD *)(a1 + 8);
          v18 = 1;
          v19 = 0;
          v20 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v21 = v17 + 80LL * v20;
          v22 = *(_QWORD *)v21;
          if ( v14 != *(_QWORD *)v21 )
          {
            while ( v22 != -4096 )
            {
              if ( v22 == -8192 && !v19 )
                v19 = v21;
              v10 = (unsigned int)(v18 + 1);
              v20 = v16 & (v18 + v20);
              v21 = v17 + 80LL * v20;
              v22 = *(_QWORD *)v21;
              if ( v14 == *(_QWORD *)v21 )
                goto LABEL_14;
              ++v18;
            }
            if ( v19 )
              v21 = v19;
          }
LABEL_14:
          *(_QWORD *)v21 = v14;
          *(_QWORD *)(v21 + 8) = v21 + 24;
          *(_QWORD *)(v21 + 16) = 0x600000000LL;
          if ( *(_DWORD *)(v13 + 16) )
            sub_2FBECA0(v21 + 8, (char **)(v13 + 8), v10, v16, v8, v17);
          *(_DWORD *)(v21 + 72) = *(_DWORD *)(v13 + 72);
          ++*(_DWORD *)(a1 + 16);
          v23 = *(_QWORD *)(v13 + 8);
          if ( v23 != v13 + 24 )
            _libc_free(v23);
        }
        v13 += 80;
      }
      while ( v11 != v13 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v26, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[10 * v24]; j != result; result += 10 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
