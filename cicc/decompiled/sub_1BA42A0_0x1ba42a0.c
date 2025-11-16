// Function: sub_1BA42A0
// Address: 0x1ba42a0
//
_QWORD *__fastcall sub_1BA42A0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r14
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 *v7; // r13
  _QWORD *i; // rdx
  __int64 *v9; // rbx
  unsigned __int64 v10; // rcx
  int v11; // eax
  int v12; // edx
  __int64 v13; // rdi
  int v14; // r9d
  unsigned int v15; // eax
  unsigned __int64 *v16; // r8
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(176LL * v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[22 * v3];
    for ( i = &result[22 * *(unsigned int *)(a1 + 24)]; i != result; result += 22 )
    {
      if ( result )
        *result = -8;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        v10 = *v9;
        if ( *v9 != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *v9;
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v16 = 0;
          v17 = (unsigned __int64 *)(v13 + 176LL * v15);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != -8 )
            {
              if ( !v16 && v18 == -16 )
                v16 = v17;
              v15 = v12 & (v14 + v15);
              v17 = (unsigned __int64 *)(v13 + 176LL * v15);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_14;
              ++v14;
            }
            if ( v16 )
              v17 = v16;
          }
LABEL_14:
          *v17 = v10;
          v17[1] = 6;
          v17[2] = 0;
          v19 = v9[3];
          v17[3] = v19;
          if ( v19 != -8 && v19 != 0 && v19 != -16 )
            sub_1649AC0(v17 + 1, v9[1] & 0xFFFFFFFFFFFFFFF8LL);
          v17[4] = v9[4];
          *((_DWORD *)v17 + 10) = *((_DWORD *)v9 + 10);
          *((_DWORD *)v17 + 11) = *((_DWORD *)v9 + 11);
          v17[6] = v9[6];
          v17[7] = v9[7];
          *((_BYTE *)v17 + 64) = *((_BYTE *)v9 + 64);
          sub_16CCEE0(v17 + 9, (__int64)(v17 + 14), 8, (__int64)(v9 + 9));
          ++*(_DWORD *)(a1 + 16);
          v20 = v9[11];
          if ( v20 != v9[10] )
            _libc_free(v20);
          v21 = v9[3];
          if ( v21 != 0 && v21 != -8 && v21 != -16 )
            sub_1649B30(v9 + 1);
        }
        v9 += 22;
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[22 * v22]; j != result; result += 22 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
