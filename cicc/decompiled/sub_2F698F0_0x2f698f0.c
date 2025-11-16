// Function: sub_2F698F0
// Address: 0x2f698f0
//
_DWORD *__fastcall sub_2F698F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r15
  _DWORD *i; // rdx
  __int64 v12; // rbx
  int v13; // eax
  int v14; // edx
  int v15; // ecx
  __int64 v16; // r8
  int v17; // r10d
  int *v18; // r9
  unsigned int v19; // esi
  int *v20; // rdx
  int v21; // edi
  unsigned __int64 v22; // rdi
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
  result = (_DWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v5 + v9;
    for ( i = &result[8 * v8]; i != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        while ( 1 )
        {
          v13 = *(_DWORD *)v12;
          if ( *(_DWORD *)v12 <= 0xFFFFFFFD )
          {
            v14 = *(_DWORD *)(a1 + 24);
            if ( !v14 )
            {
              MEMORY[0] = *(_DWORD *)v12;
              BUG();
            }
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = 1;
            v18 = 0;
            v19 = (v14 - 1) & (37 * v13);
            v20 = (int *)(v16 + 32LL * v19);
            v21 = *v20;
            if ( v13 != *v20 )
            {
              while ( v21 != -1 )
              {
                if ( !v18 && v21 == -2 )
                  v18 = v20;
                v19 = v15 & (v17 + v19);
                v20 = (int *)(v16 + 32LL * v19);
                v21 = *v20;
                if ( v13 == *v20 )
                  goto LABEL_14;
                ++v17;
              }
              if ( v18 )
                v20 = v18;
            }
LABEL_14:
            *v20 = *(_DWORD *)v12;
            *((_QWORD *)v20 + 1) = *(_QWORD *)(v12 + 8);
            *((_QWORD *)v20 + 2) = *(_QWORD *)(v12 + 16);
            *((_QWORD *)v20 + 3) = *(_QWORD *)(v12 + 24);
            *(_QWORD *)(v12 + 24) = 0;
            *(_QWORD *)(v12 + 8) = 0;
            *(_QWORD *)(v12 + 16) = 0;
            ++*(_DWORD *)(a1 + 16);
            v22 = *(_QWORD *)(v12 + 8);
            if ( v22 )
              break;
          }
          v12 += 32;
          if ( v10 == v12 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        j_j___libc_free_0(v22);
        v12 += 32;
      }
      while ( v10 != v12 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * v23]; j != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
