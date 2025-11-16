// Function: sub_2AC6C60
// Address: 0x2ac6c60
//
_QWORD *__fastcall sub_2AC6C60(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // r14
  _QWORD *i; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // r9
  int v17; // r11d
  _QWORD *v18; // r10
  __int64 v19; // rcx
  _QWORD *v20; // rdi
  __int64 v21; // rsi
  unsigned __int64 v22; // rdi
  __int64 v23; // rcx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(56LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 56 * v3;
    v10 = v4 + 56 * v3;
    for ( i = &result[7 * v8]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != v4 )
    {
      v12 = v4;
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
          v15 = (unsigned int)(v14 - 1);
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = (unsigned int)v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (_QWORD *)(v16 + 56 * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( !v18 && v21 == -8192 )
                v18 = v20;
              v7 = (unsigned int)(v17 + 1);
              v19 = (unsigned int)v15 & (v17 + (_DWORD)v19);
              v20 = (_QWORD *)(v16 + 56LL * (unsigned int)v19);
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
          v20[1] = v20 + 3;
          v20[2] = 0x400000000LL;
          if ( *(_DWORD *)(v12 + 16) )
            sub_2AA8480((__int64)(v20 + 1), (char **)(v12 + 8), v15, v19, v7, v16);
          ++*(_DWORD *)(a1 + 16);
          v22 = *(_QWORD *)(v12 + 8);
          if ( v22 != v12 + 24 )
            _libc_free(v22);
        }
        v12 += 56;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * v23]; j != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
