// Function: sub_2C943F0
// Address: 0x2c943f0
//
_QWORD *__fastcall sub_2C943F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r14
  __int64 *v5; // rbx
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r13
  _QWORD *i; // rdx
  __int64 *j; // r14
  __int64 v12; // rdx
  int v13; // ecx
  int v14; // edi
  __int64 v15; // r8
  int v16; // r11d
  _QWORD *v17; // r10
  unsigned int v18; // esi
  _QWORD *v19; // rcx
  __int64 v20; // r9
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r15
  _QWORD *k; // rdx
  __int64 v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
  v25 = (__int64)v5;
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
    *(_QWORD *)(a1 + 16) = 0;
    v24 = 32 * v4;
    v9 = &v5[4 * v4];
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v9 != j; j += 4 )
    {
      v12 = *j;
      if ( *j != -8192 && v12 != -4096 )
      {
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v19 = (_QWORD *)(v15 + 32LL * v18);
        v20 = *v19;
        if ( *v19 != v12 )
        {
          while ( v20 != -4096 )
          {
            if ( v20 == -8192 && !v17 )
              v17 = v19;
            v18 = v14 & (v16 + v18);
            v19 = (_QWORD *)(v15 + 32LL * v18);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_13;
            ++v16;
          }
          if ( v17 )
            v19 = v17;
        }
LABEL_13:
        *v19 = v12;
        v19[1] = j[1];
        v19[2] = j[2];
        v19[3] = j[3];
        j[2] = 0;
        j[1] = 0;
        j[3] = 0;
        ++*(_DWORD *)(a1 + 16);
        v21 = (unsigned __int64 *)j[2];
        v22 = (unsigned __int64 *)j[1];
        if ( v21 != v22 )
        {
          do
          {
            if ( *v22 )
              j_j___libc_free_0(*v22);
            v22 += 3;
          }
          while ( v21 != v22 );
          v22 = (unsigned __int64 *)j[1];
        }
        if ( v22 )
          j_j___libc_free_0((unsigned __int64)v22);
      }
    }
    return (_QWORD *)sub_C7D6A0(v25, v24, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[4 * *(unsigned int *)(a1 + 24)]; k != result; result += 4 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
