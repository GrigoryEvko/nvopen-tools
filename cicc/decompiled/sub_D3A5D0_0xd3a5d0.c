// Function: sub_D3A5D0
// Address: 0xd3a5d0
//
_QWORD *__fastcall sub_D3A5D0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  unsigned __int64 *v10; // r15
  _QWORD *i; // rdx
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // rax
  int v14; // edx
  int v15; // esi
  __int64 v16; // rdi
  int v17; // r10d
  unsigned __int64 *v18; // r9
  unsigned int v19; // ecx
  unsigned __int64 *v20; // rdx
  unsigned __int64 v21; // r8
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
  result = (_QWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = (unsigned __int64 *)(v5 + v9);
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
        *result = -4;
    }
    if ( v10 != (unsigned __int64 *)v5 )
    {
      v12 = (unsigned __int64 *)v5;
      do
      {
        v13 = *v12;
        if ( *v12 != -16 && v13 != -4 )
        {
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
          v19 = (v14 - 1) & (v13 ^ (v13 >> 9));
          v20 = (unsigned __int64 *)(v16 + 32LL * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4 )
            {
              if ( v21 == -16 && !v18 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = (unsigned __int64 *)(v16 + 32LL * v19);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_14;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_14:
          *v20 = *v12;
          v20[1] = v12[1];
          v20[2] = v12[2];
          v20[3] = v12[3];
          v12[3] = 0;
          v12[1] = 0;
          v12[2] = 0;
          ++*(_DWORD *)(a1 + 16);
          v22 = v12[1];
          if ( v22 )
            j_j___libc_free_0(v22, v12[3] - v22);
        }
        v12 += 4;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v23]; j != result; result += 4 )
    {
      if ( result )
        *result = -4;
    }
  }
  return result;
}
