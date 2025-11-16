// Function: sub_37C7C70
// Address: 0x37c7c70
//
_DWORD *__fastcall sub_37C7C70(__int64 a1, int a2)
{
  unsigned int v3; // r12d
  __int64 v4; // r13
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  _DWORD *v9; // rdi
  _DWORD *i; // rdx
  _DWORD *v11; // rax
  int v12; // edx
  int v13; // ecx
  int v14; // r8d
  __int64 v15; // r11
  int *v16; // r12
  unsigned int v17; // r10d
  int v18; // r14d
  int *v19; // rcx
  int v20; // r9d
  int v21; // edx
  __int64 v22; // rdx
  _DWORD *j; // rdx

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_C7D670(16LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 16LL * v3;
    v9 = (_DWORD *)(v4 + v8);
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
    if ( v9 != (_DWORD *)v4 )
    {
      v11 = (_DWORD *)v4;
      do
      {
        while ( 1 )
        {
          v12 = *v11;
          if ( *v11 <= 0xFFFFFFFD )
            break;
          v11 += 4;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v4, v8, 8);
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *v11;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 0;
        v17 = (v13 - 1) & v12;
        v18 = 1;
        v19 = (int *)(v15 + 16LL * v17);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -1 )
          {
            if ( !v16 && v20 == -2 )
              v16 = v19;
            v17 = v14 & (v18 + v17);
            v19 = (int *)(v15 + 16LL * v17);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_14;
            ++v18;
          }
          if ( v16 )
            v19 = v16;
        }
LABEL_14:
        v21 = *v11;
        v11 += 4;
        *v19 = v21;
        *((_QWORD *)v19 + 1) = *((_QWORD *)v11 - 1);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v9 != v11 );
    }
    return (_DWORD *)sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v22]; j != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
