// Function: sub_2E1E780
// Address: 0x2e1e780
//
_QWORD *__fastcall sub_2E1E780(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r12
  _QWORD *i; // rdx
  __int64 v10; // rbx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  int v14; // esi
  int v15; // esi
  __int64 v16; // r9
  int v17; // r11d
  __int64 v18; // r10
  unsigned int v19; // edi
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // rdx
  _QWORD *j; // rdx
  __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+0h] [rbp-40h]
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
  result = (_QWORD *)sub_C7D670(152LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v28 = 152 * v4;
    v8 = v5 + 152 * v4;
    for ( i = &result[19 * *(unsigned int *)(a1 + 24)]; i != result; result += 19 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v8 != v5 )
    {
      v10 = v5;
      do
      {
        v13 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *(_QWORD *)v10;
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = v16 + 152LL * v19;
          v21 = *(_QWORD *)v20;
          if ( v13 != *(_QWORD *)v20 )
          {
            while ( v21 != -4096 )
            {
              if ( v21 == -8192 && !v18 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = v16 + 152LL * v19;
              v21 = *(_QWORD *)v20;
              if ( v13 == *(_QWORD *)v20 )
                goto LABEL_19;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_19:
          *(_QWORD *)v20 = v13;
          *(_QWORD *)(v20 + 8) = v20 + 24;
          *(_QWORD *)(v20 + 16) = 0x600000000LL;
          v22 = *(unsigned int *)(v10 + 16);
          if ( (_DWORD)v22 )
          {
            v26 = v20;
            sub_2E1D470(v20 + 8, (char **)(v10 + 8), v20, v22, v21, v16);
            v20 = v26;
          }
          v23 = *(_DWORD *)(v10 + 72);
          *(_QWORD *)(v20 + 88) = 0x600000000LL;
          *(_DWORD *)(v20 + 72) = v23;
          *(_QWORD *)(v20 + 80) = v20 + 96;
          if ( *(_DWORD *)(v10 + 88) )
          {
            v27 = v20;
            sub_2E1D470(v20 + 80, (char **)(v10 + 80), v20, v22, v21, v16);
            v20 = v27;
          }
          *(_DWORD *)(v20 + 144) = *(_DWORD *)(v10 + 144);
          ++*(_DWORD *)(a1 + 16);
          v11 = *(_QWORD *)(v10 + 80);
          if ( v11 != v10 + 96 )
            _libc_free(v11);
          v12 = *(_QWORD *)(v10 + 8);
          if ( v12 != v10 + 24 )
            _libc_free(v12);
        }
        v10 += 152;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[19 * v24]; j != result; result += 19 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
