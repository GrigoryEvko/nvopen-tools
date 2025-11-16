// Function: sub_2A33350
// Address: 0x2a33350
//
_QWORD *__fastcall sub_2A33350(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r15
  _QWORD *i; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  int v14; // edx
  int v15; // esi
  __int64 v16; // rdi
  int v17; // r10d
  __int64 *v18; // r9
  unsigned int v19; // edx
  __int64 *v20; // rcx
  __int64 v21; // r8
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  _QWORD *j; // rdx

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
  result = (_QWORD *)sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 24 * v4;
    v10 = v5 + 24 * v4;
    for ( i = &result[3 * v8]; i != result; result += 3 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        v13 = *(_QWORD *)v12;
        if ( *(_QWORD *)v12 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *(_QWORD *)v12;
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (__int64 *)(v16 + 24LL * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( v21 == -8192 && !v18 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = (__int64 *)(v16 + 24LL * v19);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_14;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_14:
          *v20 = v13;
          *((_DWORD *)v20 + 4) = *(_DWORD *)(v12 + 16);
          v20[1] = *(_QWORD *)(v12 + 8);
          *(_DWORD *)(v12 + 16) = 0;
          ++*(_DWORD *)(a1 + 16);
          if ( *(_DWORD *)(v12 + 16) > 0x40u )
          {
            v22 = *(_QWORD *)(v12 + 8);
            if ( v22 )
              j_j___libc_free_0_0(v22);
          }
        }
        v12 += 24;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[3 * v23]; j != result; result += 3 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
