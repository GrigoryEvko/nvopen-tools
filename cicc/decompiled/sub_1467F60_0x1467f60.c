// Function: sub_1467F60
// Address: 0x1467f60
//
_QWORD *__fastcall sub_1467F60(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r13
  _QWORD *i; // rdx
  __int64 *v10; // rbx
  __int64 v11; // rsi
  int v12; // eax
  int v13; // ecx
  __int64 v14; // rdi
  int v15; // r10d
  __int64 *v16; // r9
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rdx
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
  result = (_QWORD *)sub_22077B0((unsigned __int64)(unsigned int)v5 << 6);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[8 * v3];
    for ( i = &result[8 * v7]; i != result; result += 8 )
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
          v18 = (__int64 *)(v14 + ((unsigned __int64)v17 << 6));
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( v19 == -16 && !v16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + ((unsigned __int64)v17 << 6));
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_14;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_14:
          v18[3] = 0;
          v18[2] = 0;
          *((_DWORD *)v18 + 8) = 0;
          *v18 = v11;
          v18[1] = 1;
          v20 = v10[2];
          ++v10[1];
          v21 = v18[2];
          v18[2] = v20;
          LODWORD(v20) = *((_DWORD *)v10 + 6);
          v10[2] = v21;
          LODWORD(v21) = *((_DWORD *)v18 + 6);
          *((_DWORD *)v18 + 6) = v20;
          LODWORD(v20) = *((_DWORD *)v10 + 7);
          *((_DWORD *)v10 + 6) = v21;
          LODWORD(v21) = *((_DWORD *)v18 + 7);
          *((_DWORD *)v18 + 7) = v20;
          LODWORD(v20) = *((_DWORD *)v10 + 8);
          *((_DWORD *)v10 + 7) = v21;
          LODWORD(v21) = *((_DWORD *)v18 + 8);
          *((_DWORD *)v18 + 8) = v20;
          *((_DWORD *)v10 + 8) = v21;
          v18[5] = v10[5];
          v18[6] = v10[6];
          v18[7] = v10[7];
          v10[7] = 0;
          v10[6] = 0;
          v10[5] = 0;
          ++*(_DWORD *)(a1 + 16);
          v22 = v10[5];
          if ( v22 )
            j_j___libc_free_0(v22, v10[7] - v22);
          j___libc_free_0(v10[2]);
        }
        v10 += 8;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * v23]; j != result; result += 8 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
