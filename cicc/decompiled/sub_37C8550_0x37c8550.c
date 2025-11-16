// Function: sub_37C8550
// Address: 0x37c8550
//
_QWORD *__fastcall sub_37C8550(__int64 a1, int a2)
{
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // r12
  _QWORD *i; // rdx
  __int64 v9; // r15
  __int64 v10; // rcx
  int v11; // esi
  int v12; // esi
  __int64 v13; // r9
  int v14; // r11d
  __int64 v15; // r10
  unsigned int v16; // edi
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rsi
  int v22; // r8d
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  _QWORD *j; // rdx
  __int64 v28; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(88LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v28 = 88 * v3;
    v7 = v4 + 88 * v3;
    for ( i = &result[11 * *(unsigned int *)(a1 + 24)]; i != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        v10 = *(_QWORD *)v9;
        if ( *(_QWORD *)v9 != -8192 && v10 != -4096 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *(_QWORD *)v9;
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = 0;
          v16 = v12 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v17 = 11LL * v16;
          v18 = v13 + 88LL * v16;
          v19 = *(_QWORD *)v18;
          if ( v10 != *(_QWORD *)v18 )
          {
            while ( v19 != -4096 )
            {
              if ( !v15 && v19 == -8192 )
                v15 = v18;
              v16 = v12 & (v14 + v16);
              v17 = 11LL * v16;
              v18 = v13 + 88LL * v16;
              v19 = *(_QWORD *)v18;
              if ( v10 == *(_QWORD *)v18 )
                goto LABEL_14;
              ++v14;
            }
            if ( v15 )
              v18 = v15;
          }
LABEL_14:
          *(_QWORD *)v18 = v10;
          *(_QWORD *)(v18 + 8) = v18 + 24;
          *(_QWORD *)(v18 + 16) = 0x400000000LL;
          if ( *(_DWORD *)(v9 + 16) )
            sub_37B6F50(v18 + 8, (char **)(v9 + 8), v17, v18 + 24, v19, v13);
          v20 = *(_QWORD *)(v9 + 56);
          v21 = v18 + 48;
          if ( v20 )
          {
            v22 = *(_DWORD *)(v9 + 48);
            *(_QWORD *)(v18 + 56) = v20;
            *(_DWORD *)(v18 + 48) = v22;
            *(_QWORD *)(v18 + 64) = *(_QWORD *)(v9 + 64);
            *(_QWORD *)(v18 + 72) = *(_QWORD *)(v9 + 72);
            *(_QWORD *)(v20 + 8) = v21;
            *(_QWORD *)(v18 + 80) = *(_QWORD *)(v9 + 80);
            *(_QWORD *)(v9 + 56) = 0;
            *(_QWORD *)(v9 + 64) = v9 + 48;
            *(_QWORD *)(v9 + 72) = v9 + 48;
            *(_QWORD *)(v9 + 80) = 0;
          }
          else
          {
            *(_DWORD *)(v18 + 48) = 0;
            *(_QWORD *)(v18 + 56) = 0;
            *(_QWORD *)(v18 + 64) = v21;
            *(_QWORD *)(v18 + 72) = v21;
            *(_QWORD *)(v18 + 80) = 0;
          }
          ++*(_DWORD *)(a1 + 16);
          v23 = *(_QWORD *)(v9 + 56);
          while ( v23 )
          {
            sub_37B80B0(*(_QWORD *)(v23 + 24));
            v24 = v23;
            v23 = *(_QWORD *)(v23 + 16);
            j_j___libc_free_0(v24);
          }
          v25 = *(_QWORD *)(v9 + 8);
          if ( v25 != v9 + 24 )
            _libc_free(v25);
        }
        v9 += 88;
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v28, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[11 * v26]; j != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
