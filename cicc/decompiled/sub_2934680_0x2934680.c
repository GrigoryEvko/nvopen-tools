// Function: sub_2934680
// Address: 0x2934680
//
_QWORD *__fastcall sub_2934680(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r13
  __int64 v9; // rbx
  _QWORD *i; // rdx
  __int64 j; // r14
  __int64 v12; // rax
  int v13; // edx
  int v14; // esi
  __int64 v15; // rdi
  int v16; // r10d
  _QWORD *v17; // r9
  unsigned int v18; // ecx
  _QWORD *v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rdi
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  __int64 v24; // rdx
  _QWORD *k; // rdx

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
  result = (_QWORD *)sub_C7D670(88LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 88 * v4;
    v9 = v5 + 88 * v4;
    for ( i = &result[11 * *(unsigned int *)(a1 + 24)]; i != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v9 != j; j += 88 )
    {
      v12 = *(_QWORD *)j;
      if ( *(_QWORD *)j != -8192 && v12 != -4096 )
      {
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v19 = (_QWORD *)(v15 + 88LL * v18);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -4096 )
          {
            if ( !v17 && v20 == -8192 )
              v17 = v19;
            v18 = v14 & (v16 + v18);
            v19 = (_QWORD *)(v15 + 88LL * v18);
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
        v21 = (__int64)(v19 + 1);
        v22 = v19 + 3;
        v23 = v19 + 11;
        *(v23 - 10) = 0;
        *(v23 - 9) = 1;
        do
        {
          if ( v22 )
            *v22 = -4096;
          v22 += 2;
        }
        while ( v23 != v22 );
        sub_2934500(v21, j + 8);
        ++*(_DWORD *)(a1 + 16);
        if ( (*(_BYTE *)(j + 16) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(j + 24), 16LL * *(unsigned int *)(j + 32), 8);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v8, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[11 * v24]; k != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
