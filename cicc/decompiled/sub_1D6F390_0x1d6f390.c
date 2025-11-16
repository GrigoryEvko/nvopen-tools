// Function: sub_1D6F390
// Address: 0x1d6f390
//
_QWORD *__fastcall sub_1D6F390(__int64 a1, int a2)
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
  int v12; // edx
  __int64 v13; // rdi
  __int64 *v14; // r10
  unsigned int v15; // r9d
  int v16; // ecx
  __int64 *v17; // r8
  __int64 v18; // rsi
  void *v19; // rdi
  unsigned int v20; // r9d
  unsigned __int64 v21; // rdi
  _DWORD *v22; // r10
  const void *v23; // rsi
  size_t v24; // rdx
  int v25; // r11d
  __int64 v26; // rcx
  _QWORD *j; // rdx
  __int64 *v28; // [rsp+8h] [rbp-48h]
  __int64 *v29; // [rsp+8h] [rbp-48h]
  _DWORD *v30; // [rsp+10h] [rbp-40h]
  _DWORD *v31; // [rsp+10h] [rbp-40h]
  unsigned int v32; // [rsp+1Ch] [rbp-34h]
  unsigned int v33; // [rsp+1Ch] [rbp-34h]

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
  result = (_QWORD *)sub_22077B0(152LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[19 * v3];
    for ( i = &result[19 * *(unsigned int *)(a1 + 24)]; i != result; result += 19 )
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
          v14 = 0;
          v15 = v12 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v16 = 1;
          v17 = (__int64 *)(v13 + 152LL * v15);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != -8 )
            {
              if ( !v14 && v18 == -16 )
                v14 = v17;
              v25 = v16 + 1;
              v26 = v12 & (v15 + v16);
              v15 = v26;
              v17 = (__int64 *)(v13 + 152 * v26);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_15;
              v16 = v25;
            }
            if ( v14 )
              v17 = v14;
          }
LABEL_15:
          v19 = v17 + 3;
          *v17 = v10;
          v17[1] = (__int64)(v17 + 3);
          v17[2] = 0x1000000000LL;
          v20 = *((_DWORD *)v9 - 2);
          if ( v20 && v17 + 1 != v9 - 2 )
          {
            v22 = (_DWORD *)*(v9 - 2);
            if ( v9 == (_QWORD *)v22 )
            {
              v23 = v9;
              v24 = 8LL * v20;
              if ( v20 <= 0x10 )
                goto LABEL_24;
              v29 = v17;
              v31 = (_DWORD *)*(v9 - 2);
              v33 = *((_DWORD *)v9 - 2);
              sub_16CD150((__int64)(v17 + 1), v17 + 3, v20, 8, (int)v17, v20);
              v17 = v29;
              v23 = (const void *)*(v9 - 2);
              v20 = v33;
              v24 = 8LL * *((unsigned int *)v9 - 2);
              v19 = (void *)v29[1];
              v22 = v31;
              if ( v24 )
              {
LABEL_24:
                v28 = v17;
                v30 = v22;
                v32 = v20;
                memcpy(v19, v23, v24);
                v17 = v28;
                v22 = v30;
                v20 = v32;
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
        if ( v7 == v9 + 16 )
          break;
        v9 += 19;
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[19 * *(unsigned int *)(a1 + 24)]; j != result; result += 19 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
