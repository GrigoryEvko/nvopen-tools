// Function: sub_2E56D40
// Address: 0x2e56d40
//
_QWORD *__fastcall sub_2E56D40(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r14
  _QWORD *i; // rdx
  __int64 *v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // esi
  __int64 v15; // r8
  int v16; // r11d
  _QWORD *v17; // r10
  unsigned int v18; // edi
  _QWORD *v19; // rdx
  __int64 v20; // r9
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdi
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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v26 = 16 * v4;
    v9 = (__int64 *)(v5 + 16 * v4);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != (__int64 *)v5 )
    {
      v11 = (__int64 *)v5;
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
          v19 = (_QWORD *)(v15 + 16LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v17 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (_QWORD *)(v15 + 16LL * v18);
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
          v19[1] = v11[1];
          v11[1] = 0;
          ++*(_DWORD *)(a1 + 16);
          v21 = v11[1];
          if ( v21 )
          {
            v22 = *(_QWORD *)(v21 + 96);
            if ( v22 != v21 + 112 )
              _libc_free(v22);
            v23 = *(_QWORD *)(v21 + 24);
            if ( v23 != v21 + 40 )
              _libc_free(v23);
            j_j___libc_free_0(v21);
          }
        }
        v11 += 2;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v26, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v24]; j != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
