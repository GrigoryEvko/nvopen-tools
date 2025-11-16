// Function: sub_22B3AE0
// Address: 0x22b3ae0
//
_DWORD *__fastcall sub_22B3AE0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r15
  _DWORD *i; // rdx
  __int64 v12; // rbx
  int v13; // edx
  int v14; // eax
  int v15; // ecx
  __int64 v16; // r8
  int *v17; // r9
  int v18; // r10d
  unsigned int v19; // esi
  int *v20; // rax
  int v21; // edi
  __int64 v22; // rcx
  __int64 v23; // rdx
  _DWORD *j; // rdx

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
  result = (_DWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 40 * v4;
    v10 = v5 + 40 * v4;
    for ( i = &result[10 * v8]; i != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        while ( 1 )
        {
          v13 = *(_DWORD *)v12;
          if ( *(_DWORD *)v12 <= 0xFFFFFFFD )
            break;
          v12 += 40;
          if ( v10 == v12 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 0;
        v18 = 1;
        v19 = (v14 - 1) & (37 * v13);
        v20 = (int *)(v16 + 40LL * v19);
        v21 = *v20;
        if ( v13 != *v20 )
        {
          while ( v21 != -1 )
          {
            if ( !v17 && v21 == -2 )
              v17 = v20;
            v19 = v15 & (v18 + v19);
            v20 = (int *)(v16 + 40LL * v19);
            v21 = *v20;
            if ( v13 == *v20 )
              goto LABEL_14;
            ++v18;
          }
          if ( v17 )
            v20 = v17;
        }
LABEL_14:
        *((_QWORD *)v20 + 3) = 0;
        *((_QWORD *)v20 + 2) = 0;
        v20[8] = 0;
        *v20 = v13;
        *((_QWORD *)v20 + 1) = 1;
        v22 = *(_QWORD *)(v12 + 16);
        ++*(_QWORD *)(v12 + 8);
        v23 = *((_QWORD *)v20 + 2);
        v12 += 40;
        *((_QWORD *)v20 + 2) = v22;
        LODWORD(v22) = *(_DWORD *)(v12 - 16);
        *(_QWORD *)(v12 - 24) = v23;
        LODWORD(v23) = v20[6];
        v20[6] = v22;
        LODWORD(v22) = *(_DWORD *)(v12 - 12);
        *(_DWORD *)(v12 - 16) = v23;
        LODWORD(v23) = v20[7];
        v20[7] = v22;
        LODWORD(v22) = *(_DWORD *)(v12 - 8);
        *(_DWORD *)(v12 - 12) = v23;
        LODWORD(v23) = v20[8];
        v20[8] = v22;
        *(_DWORD *)(v12 - 8) = v23;
        ++*(_DWORD *)(a1 + 16);
        sub_C7D6A0(*(_QWORD *)(v12 - 24), 8LL * *(unsigned int *)(v12 - 8), 8);
      }
      while ( v10 != v12 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[10 * *(unsigned int *)(a1 + 24)]; j != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
