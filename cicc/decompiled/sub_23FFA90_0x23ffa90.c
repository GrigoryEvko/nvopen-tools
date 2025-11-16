// Function: sub_23FFA90
// Address: 0x23ffa90
//
_QWORD *__fastcall sub_23FFA90(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // r13
  _QWORD *i; // rdx
  __int64 v11; // r14
  __int64 v12; // rdx
  int v13; // ecx
  int v14; // edi
  __int64 v15; // r8
  int v16; // r11d
  __int64 *v17; // r10
  unsigned int v18; // esi
  __int64 *v19; // rcx
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 *v22; // rdi
  __int64 v23; // rsi
  int v24; // r8d
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // rdi
  _QWORD *j; // rdx
  __int64 v28; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v28 = 56 * v4;
    v9 = v5 + 56 * v4;
    for ( i = &result[7 * v8]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != v5 )
    {
      v11 = v5;
      do
      {
        v12 = *(_QWORD *)v11;
        if ( *(_QWORD *)v11 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *(_QWORD *)v11;
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = (__int64 *)(v15 + 56LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v17 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (__int64 *)(v15 + 56LL * v18);
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
          v21 = *(_QWORD *)(v11 + 24);
          v22 = v19 + 2;
          v23 = v11 + 16;
          if ( v21 )
          {
            v24 = *(_DWORD *)(v11 + 16);
            v19[3] = v21;
            *((_DWORD *)v19 + 4) = v24;
            v19[4] = *(_QWORD *)(v11 + 32);
            v19[5] = *(_QWORD *)(v11 + 40);
            *(_QWORD *)(v21 + 8) = v22;
            v19[6] = *(_QWORD *)(v11 + 48);
            *(_QWORD *)(v11 + 24) = 0;
            *(_QWORD *)(v11 + 32) = v23;
            *(_QWORD *)(v11 + 40) = v23;
            *(_QWORD *)(v11 + 48) = 0;
          }
          else
          {
            *((_DWORD *)v19 + 4) = 0;
            v19[3] = 0;
            v19[4] = (__int64)v22;
            v19[5] = (__int64)v22;
            v19[6] = 0;
          }
          ++*(_DWORD *)(a1 + 16);
          v25 = *(_QWORD *)(v11 + 24);
          while ( v25 )
          {
            sub_23FBEA0(*(_QWORD *)(v25 + 24));
            v26 = v25;
            v25 = *(_QWORD *)(v25 + 16);
            j_j___libc_free_0(v26);
          }
        }
        v11 += 56;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * *(unsigned int *)(a1 + 24)]; j != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
