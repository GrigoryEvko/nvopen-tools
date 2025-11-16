// Function: sub_AC8740
// Address: 0xac8740
//
_DWORD *__fastcall sub_AC8740(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  int *v5; // r12
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  int *v10; // rcx
  _DWORD *i; // rdx
  int *v12; // rbx
  int v13; // eax
  int v14; // edx
  int v15; // esi
  __int64 v16; // r9
  int v17; // r11d
  int *v18; // r10
  unsigned int v19; // edi
  int *v20; // rdx
  int v21; // r8d
  __int64 v22; // r15
  __int64 v23; // rdi
  __int64 v24; // rdx
  _DWORD *j; // rdx
  int *v26; // [rsp+8h] [rbp-38h]
  int *v27; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(int **)(a1 + 8);
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
  result = (_DWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 16 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v5[(unsigned __int64)v9 / 4];
    for ( i = &result[4 * v8]; i != result; result += 4 )
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
          v13 = *v12;
          if ( (unsigned int)*v12 <= 0xFFFFFFFD )
          {
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
            v19 = (v14 - 1) & (37 * v13);
            v20 = (int *)(v16 + 16LL * v19);
            v21 = *v20;
            if ( v13 != *v20 )
            {
              while ( v21 != -1 )
              {
                if ( !v18 && v21 == -2 )
                  v18 = v20;
                v19 = v15 & (v17 + v19);
                v20 = (int *)(v16 + 16LL * v19);
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
            *((_QWORD *)v20 + 1) = *((_QWORD *)v12 + 1);
            *((_QWORD *)v12 + 1) = 0;
            ++*(_DWORD *)(a1 + 16);
            v22 = *((_QWORD *)v12 + 1);
            if ( v22 )
              break;
          }
          v12 += 4;
          if ( v10 == v12 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        if ( *(_DWORD *)(v22 + 32) > 0x40u )
        {
          v23 = *(_QWORD *)(v22 + 24);
          if ( v23 )
          {
            v26 = v10;
            j_j___libc_free_0_0(v23);
            v10 = v26;
          }
        }
        v27 = v10;
        v12 += 4;
        sub_BD7260(v22);
        sub_BD2DD0(v22);
        v10 = v27;
      }
      while ( v27 != v12 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v24]; j != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
