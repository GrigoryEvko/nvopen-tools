// Function: sub_17B8370
// Address: 0x17b8370
//
_QWORD *__fastcall sub_17B8370(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 *v7; // r13
  _QWORD *i; // rdx
  __int64 *j; // r15
  __int64 v10; // rdx
  int v11; // esi
  int v12; // esi
  __int64 v13; // r9
  int v14; // r11d
  __int64 *v15; // r10
  unsigned int v16; // edi
  __int64 *v17; // rcx
  __int64 v18; // r8
  int v19; // edx
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // rbx
  __int64 v24; // r14
  unsigned __int64 v25; // r8
  unsigned __int64 v26; // rdi
  _QWORD *k; // rdx
  __int64 *v28; // [rsp+0h] [rbp-40h]
  unsigned __int64 v29; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v28 = v4;
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
  result = (_QWORD *)sub_22077B0(104LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[13 * v3];
    for ( i = &result[13 * *(unsigned int *)(a1 + 24)]; i != result; result += 13 )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v4; v7 != j; j += 13 )
    {
      v10 = *j;
      if ( *j != -16 && v10 != -8 )
      {
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = 1;
        v15 = 0;
        v16 = v12 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v17 = (__int64 *)(v13 + 104LL * v16);
        v18 = *v17;
        if ( *v17 != v10 )
        {
          while ( v18 != -8 )
          {
            if ( v18 == -16 && !v15 )
              v15 = v17;
            v16 = v12 & (v14 + v16);
            v17 = (__int64 *)(v13 + 104LL * v16);
            v18 = *v17;
            if ( v10 == *v17 )
              goto LABEL_13;
            ++v14;
          }
          if ( v15 )
            v17 = v15;
        }
LABEL_13:
        *v17 = v10;
        v17[1] = j[1];
        v19 = *((_DWORD *)j + 4);
        v17[5] = 0xB000000000LL;
        *((_DWORD *)v17 + 4) = v19;
        v17[7] = (__int64)(v17 + 9);
        v17[3] = 0;
        v17[4] = 0;
        v17[8] = 0x400000000LL;
        ++*(_DWORD *)(a1 + 16);
        v20 = j[7];
        if ( (__int64 *)v20 != j + 9 )
          _libc_free(v20);
        v21 = j[3];
        if ( *((_DWORD *)j + 9) )
        {
          v22 = *((unsigned int *)j + 8);
          if ( (_DWORD)v22 )
          {
            v23 = 8 * v22;
            v24 = 0;
            do
            {
              v25 = *(_QWORD *)(v21 + v24);
              if ( v25 && v25 != -8 )
              {
                v26 = *(_QWORD *)(v25 + 32);
                if ( v26 != v25 + 48 )
                {
                  v29 = v25;
                  _libc_free(v26);
                  v25 = v29;
                }
                _libc_free(v25);
                v21 = j[3];
              }
              v24 += 8;
            }
            while ( v23 != v24 );
          }
        }
        _libc_free(v21);
      }
    }
    return (_QWORD *)j___libc_free_0(v28);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[13 * *(unsigned int *)(a1 + 24)]; k != result; result += 13 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
