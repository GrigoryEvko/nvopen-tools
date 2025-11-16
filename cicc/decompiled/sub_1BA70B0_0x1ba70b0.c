// Function: sub_1BA70B0
// Address: 0x1ba70b0
//
_DWORD *__fastcall sub_1BA70B0(__int64 a1, int a2)
{
  __int64 v3; // r12
  int *v4; // r13
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // rdx
  int *v8; // rcx
  _DWORD *i; // rdx
  int *v10; // rax
  unsigned int v11; // edx
  int v12; // esi
  int v13; // edi
  __int64 v14; // r9
  int v15; // r14d
  int *v16; // r11
  unsigned int v17; // r8d
  int *v18; // rsi
  int v19; // r10d
  __int64 v20; // rdx
  __int64 v21; // rdx
  _DWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(int **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_22077B0(16LL * v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[4 * v3];
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( 1 )
        {
          v11 = *v10;
          if ( (unsigned int)*v10 <= 0xFFFFFFFD )
            break;
          v10 += 4;
          if ( v8 == v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 1;
        v16 = 0;
        v17 = (v12 - 1) & (37 * v11);
        v18 = (int *)(v14 + 16LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -1 )
          {
            if ( !v16 && v19 == -2 )
              v16 = v18;
            v17 = v13 & (v15 + v17);
            v18 = (int *)(v14 + 16LL * v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_14;
            ++v15;
          }
          if ( v16 )
            v18 = v16;
        }
LABEL_14:
        *v18 = v11;
        v20 = *((_QWORD *)v10 + 1);
        v10 += 4;
        *((_QWORD *)v18 + 1) = v20;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v8 != v10 );
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v21]; j != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
