// Function: sub_1469520
// Address: 0x1469520
//
_QWORD *__fastcall sub_1469520(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 *v3; // r15
  unsigned __int64 v4; // rax
  _QWORD *result; // rax
  __int64 v6; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v9; // rax
  int v10; // edx
  int v11; // esi
  __int64 v12; // rcx
  int v13; // r9d
  __int64 *v14; // r8
  unsigned int v15; // edx
  __int64 *v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rsi
  unsigned __int64 v19; // r12
  __int64 v20; // r15
  __int64 v21; // rdx
  _QWORD *v22; // r13
  _QWORD *v23; // r14
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  _QWORD *k; // rdx
  __int64 *v28; // [rsp+0h] [rbp-50h]
  __int64 *v30; // [rsp+10h] [rbp-40h]

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(__int64 **)(a1 + 8);
  v28 = v3;
  v4 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
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
  if ( (unsigned int)v4 < 0x40 )
    LODWORD(v4) = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_22077B0((unsigned __int64)(unsigned int)v4 << 6);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v30 = &v3[8 * v2];
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( i = &result[8 * v6]; i != result; result += 8 )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v3; v30 != j; j += 8 )
    {
      v9 = *j;
      if ( *j != -16 && v9 != -8 )
      {
        v10 = *(_DWORD *)(a1 + 24);
        if ( !v10 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v11 = v10 - 1;
        v12 = *(_QWORD *)(a1 + 8);
        v13 = 1;
        v14 = 0;
        v15 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v16 = (__int64 *)(v12 + ((unsigned __int64)v15 << 6));
        v17 = *v16;
        if ( v9 != *v16 )
        {
          while ( v17 != -8 )
          {
            if ( v17 == -16 && !v14 )
              v14 = v16;
            v15 = v11 & (v13 + v15);
            v16 = (__int64 *)(v12 + ((unsigned __int64)v15 << 6));
            v17 = *v16;
            if ( v9 == *v16 )
              goto LABEL_13;
            ++v13;
          }
          if ( v14 )
            v16 = v14;
        }
LABEL_13:
        *v16 = v9;
        v16[1] = (__int64)(v16 + 3);
        v16[2] = 0x100000000LL;
        if ( *((_DWORD *)j + 4) )
          sub_145E880((__int64)(v16 + 1), (__int64)(j + 1));
        v16[6] = j[6];
        *((_BYTE *)v16 + 56) = *((_BYTE *)j + 56);
        ++*(_DWORD *)(a1 + 16);
        v18 = j[1];
        v19 = v18 + 24LL * *((unsigned int *)j + 4);
        if ( v18 != v19 )
        {
          do
          {
            v20 = *(_QWORD *)(v19 - 8);
            v19 -= 24LL;
            if ( v20 )
            {
              v21 = *(unsigned int *)(v20 + 208);
              *(_QWORD *)v20 = &unk_49EC708;
              if ( (_DWORD)v21 )
              {
                v22 = *(_QWORD **)(v20 + 192);
                v23 = &v22[7 * v21];
                do
                {
                  if ( *v22 != -8 && *v22 != -16 )
                  {
                    v24 = v22[1];
                    if ( (_QWORD *)v24 != v22 + 3 )
                      _libc_free(v24);
                  }
                  v22 += 7;
                }
                while ( v23 != v22 );
              }
              j___libc_free_0(*(_QWORD *)(v20 + 192));
              v25 = *(_QWORD *)(v20 + 40);
              if ( v25 != v20 + 56 )
                _libc_free(v25);
              j_j___libc_free_0(v20, 216);
            }
          }
          while ( v18 != v19 );
          v19 = j[1];
        }
        if ( (__int64 *)v19 != j + 3 )
          _libc_free(v19);
      }
    }
    return (_QWORD *)j___libc_free_0(v28);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[8 * v26]; k != result; result += 8 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
