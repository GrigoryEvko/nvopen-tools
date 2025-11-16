// Function: sub_2645780
// Address: 0x2645780
//
_QWORD *__fastcall sub_2645780(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 *v9; // r15
  _QWORD *i; // rdx
  __int64 *v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // esi
  __int64 v15; // rdi
  int v16; // r10d
  _QWORD *v17; // r9
  unsigned int v18; // ecx
  _QWORD *v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rax
  unsigned __int64 v22; // rdi
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(40LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 40 * v3;
    v9 = (__int64 *)(v4 + 40 * v3);
    for ( i = &result[5 * v7]; i != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != (__int64 *)v4 )
    {
      v11 = (__int64 *)v4;
      do
      {
        v12 = *v11;
        if ( *v11 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *v11;
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = (_QWORD *)(v15 + 40LL * v18);
          v20 = *v19;
          if ( *v19 != v12 )
          {
            while ( v20 != -4096 )
            {
              if ( !v17 && v20 == -8192 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (_QWORD *)(v15 + 40LL * v18);
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
          v19[1] = v11[1];
          v19[2] = v11[2];
          v19[3] = v11[3];
          v21 = v11[4];
          v11[3] = 0;
          v11[1] = 0;
          v11[2] = 0;
          v19[4] = v21;
          ++*(_DWORD *)(a1 + 16);
          v22 = v11[1];
          if ( v22 )
            j_j___libc_free_0(v22);
        }
        v11 += 5;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * *(unsigned int *)(a1 + 24)]; j != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
