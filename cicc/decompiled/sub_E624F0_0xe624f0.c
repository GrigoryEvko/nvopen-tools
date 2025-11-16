// Function: sub_E624F0
// Address: 0xe624f0
//
_DWORD *__fastcall sub_E624F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r12
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rdi
  _DWORD *i; // rdx
  __int64 v12; // rax
  int v13; // edx
  int v14; // ecx
  int v15; // r8d
  __int64 v16; // r11
  int *v17; // rbx
  int v18; // r14d
  unsigned int v19; // r9d
  int *v20; // rcx
  int v21; // r10d
  __int64 v22; // rdx
  __int64 v23; // rdx
  _DWORD *j; // rdx

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
  result = (_DWORD *)sub_C7D670(16LL * v6, 4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 16LL * v4;
    v10 = v5 + v9;
    for ( i = &result[4 * v8]; i != result; result += 4 )
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
          v12 += 16;
          if ( v10 == v12 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 4);
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
        v20 = (int *)(v16 + 16LL * v19);
        v21 = *v20;
        if ( v13 != *v20 )
        {
          while ( v21 != -1 )
          {
            if ( !v17 && v21 == -2 )
              v17 = v20;
            v19 = v15 & (v18 + v19);
            v20 = (int *)(v16 + 16LL * v19);
            v21 = *v20;
            if ( v13 == *v20 )
              goto LABEL_14;
            ++v18;
          }
          if ( v17 )
            v20 = v17;
        }
LABEL_14:
        *v20 = v13;
        v22 = *(_QWORD *)(v12 + 4);
        v12 += 16;
        *(_QWORD *)(v20 + 1) = v22;
        v20[3] = *(_DWORD *)(v12 - 4);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v23]; j != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
