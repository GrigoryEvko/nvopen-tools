// Function: sub_1C8F490
// Address: 0x1c8f490
//
_QWORD *__fastcall sub_1C8F490(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r15
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *v10; // r12
  __int64 v11; // rax
  int v12; // edx
  int v13; // esi
  __int64 v14; // rdi
  int v15; // r10d
  __int64 *v16; // r9
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 *v21; // rsi
  __int64 *v22; // rcx
  int v23; // edi
  __int64 v24; // rbx
  __int64 v25; // rdi
  __int64 v26; // rcx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(56LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[7 * v3];
    for ( i = &result[7 * v7]; i != result; result += 7 )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        v11 = *v10;
        if ( *v10 != -16 && v11 != -8 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *v10;
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (__int64 *)(v14 + 56LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( v19 == -16 && !v16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 56LL * v17);
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
          v20 = v10[3];
          v21 = v18 + 2;
          v22 = v10 + 2;
          if ( v20 )
          {
            v23 = *((_DWORD *)v10 + 4);
            v18[3] = v20;
            *((_DWORD *)v18 + 4) = v23;
            v18[4] = v10[4];
            v18[5] = v10[5];
            *(_QWORD *)(v20 + 8) = v21;
            v18[6] = v10[6];
            v10[3] = 0;
            v10[4] = (__int64)v22;
            v10[5] = (__int64)v22;
            v10[6] = 0;
          }
          else
          {
            *((_DWORD *)v18 + 4) = 0;
            v18[3] = 0;
            v18[4] = (__int64)v21;
            v18[5] = (__int64)v21;
            v18[6] = 0;
          }
          ++*(_DWORD *)(a1 + 16);
          v24 = v10[3];
          while ( v24 )
          {
            sub_1C8D820(*(_QWORD *)(v24 + 24));
            v25 = v24;
            v24 = *(_QWORD *)(v24 + 16);
            j_j___libc_free_0(v25, 40);
          }
        }
        v10 += 7;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * v26]; j != result; result += 7 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
