// Function: sub_2D36220
// Address: 0x2d36220
//
_QWORD *__fastcall sub_2D36220(__int64 a1, int a2)
{
  unsigned int v3; // r12d
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  _QWORD *v9; // rdi
  _QWORD *i; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // ecx
  int v15; // r8d
  __int64 v16; // r11
  int v17; // r14d
  __int64 *v18; // r12
  unsigned int v19; // r9d
  __int64 *v20; // rcx
  __int64 v21; // r10
  __int64 v22; // rdx
  __int64 v23; // rdx
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
    v9 = (_QWORD *)(v4 + v8);
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != (_QWORD *)v4 )
    {
      v11 = (_QWORD *)v4;
      do
      {
        while ( 1 )
        {
          v12 = *v11;
          v13 = *v11;
          BYTE1(v13) = BYTE1(*v11) & 0xEF;
          if ( v13 != -8192 )
            break;
          v11 += 2;
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v4, v8, 8);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = *v11;
          BUG();
        }
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 1;
        v18 = 0;
        v19 = (v14 - 1) & (37 * v12);
        v20 = (__int64 *)(v16 + 16LL * v19);
        v21 = *v20;
        if ( v12 != *v20 )
        {
          while ( v21 != -4096 )
          {
            if ( !v18 && v21 == -8192 )
              v18 = v20;
            v19 = v15 & (v17 + v19);
            v20 = (__int64 *)(v16 + 16LL * v19);
            v21 = *v20;
            if ( v12 == *v20 )
              goto LABEL_14;
            ++v17;
          }
          if ( v18 )
            v20 = v18;
        }
LABEL_14:
        v22 = *v11;
        v11 += 2;
        *v20 = v22;
        *((_DWORD *)v20 + 2) = *((_DWORD *)v11 - 2);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v8, 8);
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
