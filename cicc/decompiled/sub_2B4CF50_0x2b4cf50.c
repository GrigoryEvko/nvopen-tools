// Function: sub_2B4CF50
// Address: 0x2b4cf50
//
_QWORD *__fastcall sub_2B4CF50(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r12
  _QWORD *i; // rdx
  __int64 j; // r15
  __int64 v11; // rdx
  int v12; // ecx
  __int64 v13; // rcx
  __int64 v14; // r9
  int v15; // r11d
  __int64 *v16; // r10
  unsigned int v17; // esi
  __int64 *v18; // rdi
  __int64 v19; // r8
  unsigned __int64 *v20; // r14
  __int64 v21; // r8
  unsigned __int64 *v22; // r13
  __int64 v23; // rcx
  _QWORD *k; // rdx
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v26 = v5;
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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 56 * v4;
    v8 = 56 * v4 + v5;
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v8 != j; j += 56 )
    {
      v11 = *(_QWORD *)j;
      if ( *(_QWORD *)j != -8192 && v11 != -4096 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v13 = (unsigned int)(v12 - 1);
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 1;
        v16 = 0;
        v17 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v18 = (__int64 *)(v14 + 56LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -4096 )
          {
            if ( v19 == -8192 && !v16 )
              v16 = v18;
            v17 = v13 & (v15 + v17);
            v18 = (__int64 *)(v14 + 56LL * v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_13;
            ++v15;
          }
          if ( v16 )
            v18 = v16;
        }
LABEL_13:
        *v18 = v11;
        v18[1] = (__int64)(v18 + 3);
        v18[2] = 0x100000000LL;
        if ( *(_DWORD *)(j + 16) )
          sub_2B425B0((__int64)(v18 + 1), j + 8, (__int64)(v18 + 3), v13, v19, v14);
        ++*(_DWORD *)(a1 + 16);
        v20 = *(unsigned __int64 **)(j + 8);
        v21 = 4LL * *(unsigned int *)(j + 16);
        v22 = &v20[v21];
        if ( v20 != &v20[v21] )
        {
          do
          {
            v22 -= 4;
            if ( (unsigned __int64 *)*v22 != v22 + 2 )
              _libc_free(*v22);
          }
          while ( v20 != v22 );
          v22 = *(unsigned __int64 **)(j + 8);
        }
        if ( v22 != (unsigned __int64 *)(j + 24) )
          _libc_free((unsigned __int64)v22);
      }
    }
    return (_QWORD *)sub_C7D6A0(v26, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[7 * v23]; k != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
