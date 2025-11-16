// Function: sub_2D2E3D0
// Address: 0x2d2e3d0
//
_DWORD *__fastcall sub_2D2E3D0(__int64 a1, int a2)
{
  unsigned int v3; // r12d
  __int64 v4; // r13
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdi
  _DWORD *i; // rdx
  __int64 v11; // rax
  int v12; // edx
  int v13; // ecx
  int v14; // r8d
  __int64 v15; // r11
  int v16; // r14d
  int *v17; // r12
  unsigned int v18; // r9d
  int *v19; // rcx
  int v20; // r10d
  __int64 v21; // rdx
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
    v9 = v4 + v8;
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
    if ( v9 != v4 )
    {
      v11 = v4;
      do
      {
        while ( 1 )
        {
          v12 = *(_DWORD *)v11;
          if ( *(_DWORD *)v11 <= 0xFFFFFFFD )
            break;
          v11 += 16;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v4, v8, 8);
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        v18 = (v13 - 1) & (37 * v12);
        v19 = (int *)(v15 + 16LL * v18);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -1 )
          {
            if ( !v17 && v20 == -2 )
              v17 = v19;
            v18 = v14 & (v16 + v18);
            v19 = (int *)(v15 + 16LL * v18);
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
        v21 = *(_QWORD *)(v11 + 8);
        v11 += 16;
        *((_QWORD *)v19 + 1) = v21;
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
