// Function: sub_2854A40
// Address: 0x2854a40
//
_QWORD *__fastcall sub_2854A40(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 *v10; // rdi
  _QWORD *i; // rdx
  unsigned __int64 *v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // ecx
  int v15; // r8d
  __int64 v16; // r10
  unsigned __int64 *v17; // rbx
  int v18; // r14d
  unsigned int v19; // r9d
  unsigned __int64 *v20; // rcx
  unsigned __int64 v21; // r11
  __int64 v22; // rdx
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
    v10 = (unsigned __int64 *)(v5 + v9);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -2;
    }
    if ( v10 != (unsigned __int64 *)v5 )
    {
      v12 = (unsigned __int64 *)v5;
      do
      {
        v13 = *v12;
        if ( *v12 != -16 && v13 != -2 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *v12;
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 0;
          v18 = 1;
          v19 = (v14 - 1) & (v13 ^ (v13 >> 9));
          v20 = (unsigned __int64 *)(v16 + 16LL * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -2 )
            {
              if ( v21 == -16 && !v17 )
                v17 = v20;
              v19 = v15 & (v18 + v19);
              v20 = (unsigned __int64 *)(v16 + 16LL * v19);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_14;
              ++v18;
            }
            if ( v17 )
              v20 = v17;
          }
LABEL_14:
          *v20 = *v12;
          v20[1] = v12[1];
          ++*(_DWORD *)(a1 + 16);
        }
        v12 += 2;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v22]; j != result; result += 2 )
    {
      if ( result )
        *result = -2;
    }
  }
  return result;
}
