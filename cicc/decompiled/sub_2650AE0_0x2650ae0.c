// Function: sub_2650AE0
// Address: 0x2650ae0
//
_DWORD *__fastcall sub_2650AE0(__int64 a1, int a2)
{
  __int64 v3; // r14
  __int64 v4; // r13
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // r15
  _DWORD *i; // rdx
  __int64 v11; // rbx
  int v12; // eax
  int v13; // edx
  int v14; // esi
  __int64 v15; // r8
  int *v16; // r9
  int v17; // r10d
  unsigned int v18; // ecx
  int *v19; // rdx
  int v20; // edi
  unsigned __int64 v21; // rdi
  _DWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_C7D670(32LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = 32 * v3;
    *(_QWORD *)(a1 + 16) = 0;
    v9 = v4 + v8;
    for ( i = &result[8 * v7]; i != result; result += 8 )
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
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v14 = v13 - 1;
            v15 = *(_QWORD *)(a1 + 8);
            v16 = 0;
            v17 = 1;
            v18 = (v13 - 1) & (37 * v12);
            v19 = (int *)(v15 + 32LL * v18);
            v20 = *v19;
            if ( v12 != *v19 )
            {
              while ( v20 != -1 )
              {
                if ( !v16 && v20 == -2 )
                  v16 = v19;
                v18 = v14 & (v17 + v18);
                v19 = (int *)(v15 + 32LL * v18);
                v20 = *v19;
                if ( v12 == *v19 )
                  goto LABEL_14;
                ++v17;
              }
              if ( v16 )
                v19 = v16;
            }
LABEL_14:
            *v19 = v12;
            *((_QWORD *)v19 + 1) = *(_QWORD *)(v11 + 8);
            *((_QWORD *)v19 + 2) = *(_QWORD *)(v11 + 16);
            *((_QWORD *)v19 + 3) = *(_QWORD *)(v11 + 24);
            *(_QWORD *)(v11 + 24) = 0;
            *(_QWORD *)(v11 + 8) = 0;
            *(_QWORD *)(v11 + 16) = 0;
            ++*(_DWORD *)(a1 + 16);
            v21 = *(_QWORD *)(v11 + 8);
            if ( v21 )
              break;
          }
          v11 += 32;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v4, v8, 8);
        }
        j_j___libc_free_0(v21);
        v11 += 32;
      }
      while ( v9 != v11 );
    }
    return (_DWORD *)sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * *(unsigned int *)(a1 + 24)]; j != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
