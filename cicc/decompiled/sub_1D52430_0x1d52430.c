// Function: sub_1D52430
// Address: 0x1d52430
//
_QWORD *__fastcall sub_1D52430(__int64 a1, int a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  _QWORD *v7; // r15
  _QWORD *i; // rdx
  _QWORD *v9; // rbx
  __int64 v10; // rax
  int v11; // edx
  int v12; // ecx
  __int64 v13; // rdi
  int v14; // r10d
  __int64 *v15; // r9
  unsigned int v16; // edx
  __int64 *v17; // r8
  __int64 v18; // rsi
  void *v19; // rdi
  unsigned int v20; // r9d
  unsigned __int64 v21; // rdi
  _DWORD *v22; // r10
  const void *v23; // rsi
  size_t v24; // rdx
  _QWORD *j; // rdx
  __int64 *v26; // [rsp+8h] [rbp-48h]
  __int64 *v27; // [rsp+8h] [rbp-48h]
  _DWORD *v28; // [rsp+10h] [rbp-40h]
  _DWORD *v29; // [rsp+10h] [rbp-40h]
  unsigned int v30; // [rsp+1Ch] [rbp-34h]
  unsigned int v31; // [rsp+1Ch] [rbp-34h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD **)(a1 + 8);
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
  result = (_QWORD *)sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[5 * v3];
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
    v9 = v4 + 3;
    if ( v7 != v4 )
    {
      while ( 1 )
      {
        v10 = *(v9 - 3);
        if ( v10 != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *(v9 - 3);
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = 0;
          v16 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v17 = (__int64 *)(v13 + 40LL * v16);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != -8 )
            {
              if ( !v15 && v18 == -16 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = (__int64 *)(v13 + 40LL * v16);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_15;
              ++v14;
            }
            if ( v15 )
              v17 = v15;
          }
LABEL_15:
          v19 = v17 + 3;
          *v17 = v10;
          v17[1] = (__int64)(v17 + 3);
          v17[2] = 0x400000000LL;
          v20 = *((_DWORD *)v9 - 2);
          if ( v20 && v17 + 1 != v9 - 2 )
          {
            v22 = (_DWORD *)*(v9 - 2);
            if ( v9 == (_QWORD *)v22 )
            {
              v23 = v9;
              v24 = 4LL * v20;
              if ( v20 <= 4 )
                goto LABEL_24;
              v27 = v17;
              v29 = (_DWORD *)*(v9 - 2);
              v31 = *((_DWORD *)v9 - 2);
              sub_16CD150((__int64)(v17 + 1), v17 + 3, v20, 4, (int)v17, v20);
              v17 = v27;
              v23 = (const void *)*(v9 - 2);
              v20 = v31;
              v24 = 4LL * *((unsigned int *)v9 - 2);
              v19 = (void *)v27[1];
              v22 = v29;
              if ( v24 )
              {
LABEL_24:
                v26 = v17;
                v28 = v22;
                v30 = v20;
                memcpy(v19, v23, v24);
                v17 = v26;
                v22 = v28;
                v20 = v30;
              }
              *((_DWORD *)v17 + 4) = v20;
              *(v22 - 2) = 0;
            }
            else
            {
              v17[1] = (__int64)v22;
              *((_DWORD *)v17 + 4) = *((_DWORD *)v9 - 2);
              *((_DWORD *)v17 + 5) = *((_DWORD *)v9 - 1);
              *(v9 - 2) = v9;
              *((_DWORD *)v9 - 1) = 0;
              *((_DWORD *)v9 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v21 = *(v9 - 2);
          if ( (_QWORD *)v21 != v9 )
            _libc_free(v21);
        }
        if ( v7 == v9 + 2 )
          break;
        v9 += 5;
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * *(unsigned int *)(a1 + 24)]; j != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
