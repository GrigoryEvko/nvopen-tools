// Function: sub_2E9B270
// Address: 0x2e9b270
//
_QWORD *__fastcall sub_2E9B270(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r13
  _QWORD *i; // rdx
  __int64 j; // r12
  __int64 v11; // rdx
  int v12; // eax
  int v13; // edi
  __int64 v14; // r8
  int v15; // r11d
  __int64 *v16; // r10
  unsigned int v17; // esi
  __int64 *v18; // rax
  __int64 v19; // r9
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  _QWORD *k; // rdx
  __int64 v28; // [rsp+0h] [rbp-40h]
  __int64 v29; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v28 = 40 * v4;
    v8 = v5 + 40 * v4;
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v8 != j; j += 40 )
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
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 1;
        v16 = 0;
        v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v18 = (__int64 *)(v14 + 40LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -4096 )
          {
            if ( v19 == -8192 && !v16 )
              v16 = v18;
            v17 = v13 & (v15 + v17);
            v18 = (__int64 *)(v14 + 40LL * v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_13;
            ++v15;
          }
          if ( v16 )
            v18 = v16;
        }
LABEL_13:
        v18[3] = 0;
        v18[2] = 0;
        *((_DWORD *)v18 + 8) = 0;
        *v18 = v11;
        v18[1] = 1;
        v20 = *(_QWORD *)(j + 16);
        ++*(_QWORD *)(j + 8);
        v21 = v18[2];
        v18[2] = v20;
        LODWORD(v20) = *(_DWORD *)(j + 24);
        *(_QWORD *)(j + 16) = v21;
        LODWORD(v21) = *((_DWORD *)v18 + 6);
        *((_DWORD *)v18 + 6) = v20;
        LODWORD(v20) = *(_DWORD *)(j + 28);
        *(_DWORD *)(j + 24) = v21;
        LODWORD(v21) = *((_DWORD *)v18 + 7);
        *((_DWORD *)v18 + 7) = v20;
        LODWORD(v20) = *(_DWORD *)(j + 32);
        *(_DWORD *)(j + 28) = v21;
        LODWORD(v21) = *((_DWORD *)v18 + 8);
        *((_DWORD *)v18 + 8) = v20;
        *(_DWORD *)(j + 32) = v21;
        ++*(_DWORD *)(a1 + 16);
        v22 = *(unsigned int *)(j + 32);
        if ( (_DWORD)v22 )
        {
          v23 = *(_QWORD *)(j + 16);
          v24 = v23 + 32 * v22;
          do
          {
            while ( 1 )
            {
              if ( *(_DWORD *)v23 <= 0xFFFFFFFD )
              {
                v25 = *(_QWORD *)(v23 + 8);
                if ( v25 )
                  break;
              }
              v23 += 32;
              if ( v24 == v23 )
                goto LABEL_19;
            }
            v29 = v24;
            v23 += 32;
            j_j___libc_free_0(v25);
            v24 = v29;
          }
          while ( v29 != v23 );
LABEL_19:
          v22 = *(unsigned int *)(j + 32);
        }
        sub_C7D6A0(*(_QWORD *)(j + 16), 32 * v22, 8);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[5 * v26]; k != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
