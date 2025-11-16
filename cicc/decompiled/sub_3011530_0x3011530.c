// Function: sub_3011530
// Address: 0x3011530
//
_QWORD *__fastcall sub_3011530(__int64 a1, int a2)
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
  __int64 v14; // rdx
  int v15; // edx
  int v16; // edx
  __int64 v17; // r8
  int v18; // r10d
  _QWORD *v19; // r9
  unsigned int v20; // ecx
  _QWORD *v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rdi
  _QWORD *v24; // rsi
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
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
  result = (_QWORD *)sub_C7D670(72LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 72 * v4;
    v10 = v5 + 72 * v4;
    for ( i = &result[9 * v8]; i != result; result += 9 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        while ( 1 )
        {
          v13 = *(_QWORD *)v12;
          v14 = *(_QWORD *)v12;
          BYTE1(v14) = BYTE1(*(_QWORD *)v12) & 0xEF;
          if ( v14 != -8192 )
          {
            v15 = *(_DWORD *)(a1 + 24);
            if ( !v15 )
            {
              MEMORY[0] = *(_QWORD *)v12;
              BUG();
            }
            v16 = v15 - 1;
            v17 = *(_QWORD *)(a1 + 8);
            v18 = 1;
            v19 = 0;
            v20 = v16 & (37 * v13);
            v21 = (_QWORD *)(v17 + 72LL * v20);
            v22 = *v21;
            if ( v13 != *v21 )
            {
              while ( v22 != -4096 )
              {
                if ( !v19 && v22 == -8192 )
                  v19 = v21;
                v20 = v16 & (v18 + v20);
                v21 = (_QWORD *)(v17 + 72LL * v20);
                v22 = *v21;
                if ( v13 == *v21 )
                  goto LABEL_14;
                ++v18;
              }
              if ( v19 )
                v21 = v19;
            }
LABEL_14:
            v23 = (__int64)(v21 + 1);
            v24 = v21 + 5;
            *(v24 - 5) = *(_QWORD *)v12;
            sub_C8CF70(v23, v24, 4, v12 + 40, v12 + 8);
            ++*(_DWORD *)(a1 + 16);
            if ( !*(_BYTE *)(v12 + 36) )
              break;
          }
          v12 += 72;
          if ( v10 == v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v25 = *(_QWORD *)(v12 + 16);
        v12 += 72;
        _libc_free(v25);
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[9 * v26]; j != result; result += 9 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
