// Function: sub_34A5290
// Address: 0x34a5290
//
_QWORD *__fastcall sub_34A5290(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // r12
  _QWORD *i; // rdx
  __int64 v11; // r15
  __int64 v12; // rcx
  int v13; // esi
  int v14; // esi
  __int64 v15; // r9
  int v16; // r11d
  __int64 v17; // r10
  unsigned int v18; // edi
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // r8
  __int64 v22; // rcx
  __int64 v23; // rsi
  int v24; // r8d
  unsigned __int64 v25; // r13
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 v28; // rcx
  _QWORD *j; // rdx
  __int64 v30; // [rsp+8h] [rbp-38h]

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
    v30 = 136 * v4;
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
          v18 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = 17LL * v18;
          v20 = v15 + 136LL * v18;
          v21 = *(_QWORD *)v20;
          if ( *(_QWORD *)v20 != v12 )
          {
            while ( v21 != -4096 )
            {
              if ( v21 == -8192 && !v17 )
                v17 = v20;
              v18 = v14 & (v16 + v18);
              v19 = 17LL * v18;
              v20 = v15 + 136LL * v18;
              v21 = *(_QWORD *)v20;
              if ( v12 == *(_QWORD *)v20 )
                goto LABEL_14;
              ++v16;
            }
            if ( v17 )
              v20 = v17;
          }
LABEL_14:
          *(_QWORD *)v20 = v12;
          *(_QWORD *)(v20 + 8) = v20 + 24;
          *(_QWORD *)(v20 + 16) = 0x400000000LL;
          if ( *(_DWORD *)(v11 + 16) )
            sub_349D880(v20 + 8, (char **)(v11 + 8), v19, v20 + 24, v21, v15);
          v22 = *(_QWORD *)(v11 + 104);
          v23 = v20 + 96;
          if ( v22 )
          {
            v24 = *(_DWORD *)(v11 + 96);
            *(_QWORD *)(v20 + 104) = v22;
            *(_DWORD *)(v20 + 96) = v24;
            *(_QWORD *)(v20 + 112) = *(_QWORD *)(v11 + 112);
            *(_QWORD *)(v20 + 120) = *(_QWORD *)(v11 + 120);
            *(_QWORD *)(v22 + 8) = v23;
            *(_QWORD *)(v20 + 128) = *(_QWORD *)(v11 + 128);
            *(_QWORD *)(v11 + 104) = 0;
            *(_QWORD *)(v11 + 112) = v11 + 96;
            *(_QWORD *)(v11 + 120) = v11 + 96;
            *(_QWORD *)(v11 + 128) = 0;
          }
          else
          {
            *(_DWORD *)(v20 + 96) = 0;
            *(_QWORD *)(v20 + 104) = 0;
            *(_QWORD *)(v20 + 112) = v23;
            *(_QWORD *)(v20 + 120) = v23;
            *(_QWORD *)(v20 + 128) = 0;
          }
          ++*(_DWORD *)(a1 + 16);
          v25 = *(_QWORD *)(v11 + 104);
          while ( v25 )
          {
            sub_349E6D0(*(_QWORD *)(v25 + 24));
            v26 = v25;
            v25 = *(_QWORD *)(v25 + 16);
            j_j___libc_free_0(v26);
          }
          v27 = *(_QWORD *)(v11 + 8);
          if ( v27 != v11 + 24 )
            _libc_free(v27);
        }
        v11 += 136;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v30, 8);
  }
  else
  {
    v28 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[17 * v28]; j != result; result += 17 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
