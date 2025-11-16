// Function: sub_2ACC850
// Address: 0x2acc850
//
_QWORD *__fastcall sub_2ACC850(__int64 a1, int a2)
{
  unsigned int v3; // r12d
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdi
  _QWORD *i; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // ecx
  int v14; // r8d
  __int64 v15; // r10
  int v16; // r14d
  __int64 *v17; // r12
  unsigned int v18; // r9d
  __int64 *v19; // rcx
  __int64 v20; // r11
  __int64 v21; // rdx
  _QWORD *j; // rdx

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(16LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 16LL * v3;
    v9 = v4 + v8;
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != v4 )
    {
      v11 = v4;
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
          v19 = (__int64 *)(v15 + 16LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( !v17 && v20 == -8192 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (__int64 *)(v15 + 16LL * v18);
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
          *((_DWORD *)v19 + 2) = *(_DWORD *)(v11 + 8);
          ++*(_DWORD *)(a1 + 16);
        }
        v11 += 16;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v21]; j != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
