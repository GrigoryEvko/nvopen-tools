// Function: sub_26D02C0
// Address: 0x26d02c0
//
_QWORD *__fastcall sub_26D02C0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 *v10; // r8
  _QWORD *i; // rdx
  __int64 *v12; // rax
  __int64 v13; // rdx
  int v14; // edi
  int v15; // edi
  __int64 v16; // r11
  int v17; // r15d
  _QWORD *v18; // r14
  unsigned int v19; // esi
  _QWORD *v20; // rcx
  __int64 v21; // r10
  __int64 v22; // rdx
  __int64 v23; // rdx
  _QWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 16LL * v4;
    v10 = (__int64 *)(v5 + v9);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -1;
    }
    if ( v10 != (__int64 *)v5 )
    {
      v12 = (__int64 *)v5;
      do
      {
        while ( 1 )
        {
          v13 = *v12;
          if ( (unsigned __int64)*v12 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v12 += 2;
          if ( v10 == v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 1;
        v18 = 0;
        v19 = v15 & (((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * v13));
        v20 = (_QWORD *)(v16 + 16LL * v19);
        v21 = *v20;
        if ( v13 != *v20 )
        {
          while ( v21 != -1 )
          {
            if ( !v18 && v21 == -2 )
              v18 = v20;
            v19 = v15 & (v17 + v19);
            v20 = (_QWORD *)(v16 + 16LL * v19);
            v21 = *v20;
            if ( v13 == *v20 )
              goto LABEL_14;
            ++v17;
          }
          if ( v18 )
            v20 = v18;
        }
LABEL_14:
        *v20 = v13;
        v22 = v12[1];
        v12 += 2;
        v20[1] = v22;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v23]; j != result; result += 2 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
