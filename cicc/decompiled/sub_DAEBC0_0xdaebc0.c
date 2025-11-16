// Function: sub_DAEBC0
// Address: 0xdaebc0
//
_QWORD *__fastcall sub_DAEBC0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r14
  __int64 v9; // r15
  _QWORD *i; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // ecx
  __int64 v15; // rdi
  int v16; // r10d
  _QWORD *v17; // r9
  unsigned int v18; // edx
  _QWORD *v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rdi
  void *v22; // rsi
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
  result = (_QWORD *)sub_C7D670(104LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 104 * v4;
    v9 = v5 + 104 * v4;
    for ( i = &result[13 * *(unsigned int *)(a1 + 24)]; i != result; result += 13 )
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
          v19 = (_QWORD *)(v15 + 104LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v17 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (_QWORD *)(v15 + 104LL * v18);
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
          v21 = (__int64)(v19 + 1);
          v22 = v19 + 5;
          sub_C8CF70(v21, v22, 8, v11 + 40, v11 + 8);
          ++*(_DWORD *)(a1 + 16);
          if ( !*(_BYTE *)(v11 + 36) )
            _libc_free(*(_QWORD *)(v11 + 16), v22);
        }
        v11 += 104;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v8, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[13 * v23]; j != result; result += 13 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
