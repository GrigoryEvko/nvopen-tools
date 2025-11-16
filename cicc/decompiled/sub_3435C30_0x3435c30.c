// Function: sub_3435C30
// Address: 0x3435c30
//
_QWORD *__fastcall sub_3435C30(__int64 a1, int a2)
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
  __int64 v13; // rdx
  int v14; // eax
  int v15; // esi
  __int64 v16; // rdi
  int v17; // r10d
  __int64 *v18; // r9
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rdx
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
  result = (_QWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 40 * v4;
    v10 = v5 + 40 * v4;
    for ( i = &result[5 * v8]; i != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        v13 = *(_QWORD *)v12;
        if ( *(_QWORD *)v12 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *(_QWORD *)v12;
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (__int64 *)(v16 + 40LL * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( !v18 && v21 == -8192 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = (__int64 *)(v16 + 40LL * v19);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_14;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_14:
          v20[3] = 0;
          v20[2] = 0;
          *((_DWORD *)v20 + 8) = 0;
          *v20 = v13;
          v20[1] = 1;
          v22 = *(_QWORD *)(v12 + 16);
          ++*(_QWORD *)(v12 + 8);
          v23 = v20[2];
          v20[2] = v22;
          LODWORD(v22) = *(_DWORD *)(v12 + 24);
          *(_QWORD *)(v12 + 16) = v23;
          LODWORD(v23) = *((_DWORD *)v20 + 6);
          *((_DWORD *)v20 + 6) = v22;
          LODWORD(v22) = *(_DWORD *)(v12 + 28);
          *(_DWORD *)(v12 + 24) = v23;
          LODWORD(v23) = *((_DWORD *)v20 + 7);
          *((_DWORD *)v20 + 7) = v22;
          LODWORD(v22) = *(_DWORD *)(v12 + 32);
          *(_DWORD *)(v12 + 28) = v23;
          LODWORD(v23) = *((_DWORD *)v20 + 8);
          *((_DWORD *)v20 + 8) = v22;
          *(_DWORD *)(v12 + 32) = v23;
          ++*(_DWORD *)(a1 + 16);
          sub_C7D6A0(*(_QWORD *)(v12 + 16), 16LL * *(unsigned int *)(v12 + 32), 8);
        }
        v12 += 40;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * v24]; j != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
