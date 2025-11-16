// Function: sub_373BA10
// Address: 0x373ba10
//
_QWORD *__fastcall sub_373BA10(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 *v10; // r15
  _QWORD *i; // rdx
  __int64 *v12; // rbx
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // rdi
  int v17; // r10d
  _QWORD *v18; // r9
  __int64 v19; // rsi
  _QWORD *v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rdi
  __int64 v23; // rdx
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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 16 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = (__int64 *)(v5 + v9);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != (__int64 *)v5 )
    {
      v12 = (__int64 *)v5;
      do
      {
        v13 = *v12;
        if ( *v12 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *v12;
            BUG();
          }
          v15 = (unsigned int)(v14 - 1);
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = (unsigned int)v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (_QWORD *)(v16 + 16 * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( v21 == -8192 && !v18 )
                v18 = v20;
              v19 = (unsigned int)v15 & (v17 + (_DWORD)v19);
              v20 = (_QWORD *)(v16 + 16LL * (unsigned int)v19);
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
          v20[1] = v12[1];
          v12[1] = 0;
          ++*(_DWORD *)(a1 + 16);
          v22 = v12[1];
          if ( v22 )
            (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64, _QWORD *))(*(_QWORD *)v22 + 8LL))(
              v22,
              v19,
              v20,
              v15,
              v21,
              v18);
        }
        v12 += 2;
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
        *result = -4096;
    }
  }
  return result;
}
