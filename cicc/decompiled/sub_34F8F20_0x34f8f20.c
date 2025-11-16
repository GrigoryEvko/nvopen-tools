// Function: sub_34F8F20
// Address: 0x34f8f20
//
_QWORD *__fastcall sub_34F8F20(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r14
  _QWORD *i; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // rdi
  int v15; // r10d
  unsigned int v16; // edx
  _QWORD *v17; // r8
  _QWORD *v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rdx
  _QWORD *j; // rdx
  __int64 v22; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(176LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v22 = 176 * v4;
    v8 = v5 + 176 * v4;
    for ( i = &result[22 * *(unsigned int *)(a1 + 24)]; i != result; result += 22 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v8 != v5 )
    {
      v10 = v5;
      do
      {
        v11 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 != -8192 && v11 != -4096 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(_QWORD *)v10;
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v17 = 0;
          v18 = (_QWORD *)(v14 + 176LL * v16);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -4096 )
            {
              if ( v19 == -8192 && !v17 )
                v17 = v18;
              v16 = v13 & (v15 + v16);
              v18 = (_QWORD *)(v14 + 176LL * v16);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_14;
              ++v15;
            }
            if ( v17 )
              v18 = v17;
          }
LABEL_14:
          *v18 = v11;
          sub_C8CF70((__int64)(v18 + 1), v18 + 5, 16, v10 + 40, v10 + 8);
          v18[21] = *(_QWORD *)(v10 + 168);
          ++*(_DWORD *)(a1 + 16);
          if ( !*(_BYTE *)(v10 + 36) )
            _libc_free(*(_QWORD *)(v10 + 16));
        }
        v10 += 176;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v22, 8);
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[22 * v20]; j != result; result += 22 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
