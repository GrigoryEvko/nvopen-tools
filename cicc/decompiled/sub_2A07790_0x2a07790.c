// Function: sub_2A07790
// Address: 0x2a07790
//
_QWORD *__fastcall sub_2A07790(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rsi
  __int64 v9; // r12
  _QWORD *i; // rdx
  __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  int v15; // esi
  int v16; // esi
  __int64 v17; // r9
  int v18; // r11d
  __int64 *v19; // r10
  unsigned int v20; // edi
  __int64 *v21; // rdx
  __int64 v22; // r8
  __int64 v23; // rcx
  __int64 v24; // rcx
  _QWORD *j; // rdx
  __int64 *v26; // [rsp+0h] [rbp-40h]

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
  result = (_QWORD *)sub_C7D670(136LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = v5 + 136 * v4;
    for ( i = &result[17 * v8]; i != result; result += 17 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != v5 )
    {
      v11 = v5;
      do
      {
        v14 = *(_QWORD *)v11;
        if ( *(_QWORD *)v11 != -8192 && v14 != -4096 )
        {
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
          {
            MEMORY[0] = *(_QWORD *)v11;
            BUG();
          }
          v16 = v15 - 1;
          v17 = *(_QWORD *)(a1 + 8);
          v18 = 1;
          v19 = 0;
          v20 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v21 = (__int64 *)(v17 + 136LL * v20);
          v22 = *v21;
          if ( v14 != *v21 )
          {
            while ( v22 != -4096 )
            {
              if ( v22 == -8192 && !v19 )
                v19 = v21;
              v20 = v16 & (v18 + v20);
              v21 = (__int64 *)(v17 + 136LL * v20);
              v22 = *v21;
              if ( v14 == *v21 )
                goto LABEL_19;
              ++v18;
            }
            if ( v19 )
              v21 = v19;
          }
LABEL_19:
          *v21 = v14;
          v21[1] = (__int64)(v21 + 3);
          v21[2] = 0xC00000000LL;
          v23 = *(unsigned int *)(v11 + 16);
          if ( (_DWORD)v23 )
          {
            v26 = v21;
            sub_2A045D0((__int64)(v21 + 1), (char **)(v11 + 8), (__int64)v21, v23, v22, v17);
            v21 = v26;
          }
          v21[10] = 0xC00000000LL;
          v21[9] = (__int64)(v21 + 11);
          if ( *(_DWORD *)(v11 + 80) )
            sub_2A044F0((__int64)(v21 + 9), v11 + 72, (__int64)v21, v23, v22, v17);
          ++*(_DWORD *)(a1 + 16);
          v12 = *(_QWORD *)(v11 + 72);
          if ( v12 != v11 + 88 )
            _libc_free(v12);
          v13 = *(_QWORD *)(v11 + 8);
          if ( v13 != v11 + 24 )
            _libc_free(v13);
        }
        v11 += 136;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, 136 * v4, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[17 * v24]; j != result; result += 17 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
