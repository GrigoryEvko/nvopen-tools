// Function: sub_1D6B340
// Address: 0x1d6b340
//
_QWORD *__fastcall sub_1D6B340(__int64 a1, int a2)
{
  unsigned int v3; // ebx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r14
  _QWORD *i; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // rsi
  int v15; // r10d
  __int64 *v16; // r9
  unsigned int v17; // edx
  __int64 *v18; // r15
  __int64 v19; // rdi
  void *v20; // rdi
  unsigned int v21; // r9d
  unsigned __int64 v22; // rdi
  _DWORD *v23; // r10
  const void *v24; // rsi
  size_t v25; // rdx
  _QWORD *j; // rdx
  _DWORD *v27; // [rsp+0h] [rbp-40h]
  _DWORD *v28; // [rsp+0h] [rbp-40h]
  unsigned int v29; // [rsp+Ch] [rbp-34h]
  unsigned int v30; // [rsp+Ch] [rbp-34h]

  v3 = *(_DWORD *)(a1 + 24);
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
  result = (_QWORD *)sub_22077B0(536LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[67 * v3];
    for ( i = &result[67 * v7]; i != result; result += 67 )
    {
      if ( result )
        *result = -8;
    }
    v10 = v4 + 3;
    if ( v8 != v4 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 3);
        if ( v11 != -16 && v11 != -8 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(v10 - 3);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (__int64 *)(v14 + 536LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( !v16 && v19 == -16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 536LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_15;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_15:
          *v18 = v11;
          v20 = v18 + 3;
          v18[1] = (__int64)(v18 + 3);
          v18[2] = 0x2000000000LL;
          v21 = *((_DWORD *)v10 - 2);
          if ( v21 && v18 + 1 != v10 - 2 )
          {
            v23 = (_DWORD *)*(v10 - 2);
            if ( v10 == (_QWORD *)v23 )
            {
              v24 = v10;
              v25 = 16LL * v21;
              if ( v21 <= 0x20 )
                goto LABEL_24;
              v28 = (_DWORD *)*(v10 - 2);
              v30 = *((_DWORD *)v10 - 2);
              sub_16CD150((__int64)(v18 + 1), v18 + 3, v21, 16, v21, v21);
              v20 = (void *)v18[1];
              v24 = (const void *)*(v10 - 2);
              v21 = v30;
              v25 = 16LL * *((unsigned int *)v10 - 2);
              v23 = v28;
              if ( v25 )
              {
LABEL_24:
                v27 = v23;
                v29 = v21;
                memcpy(v20, v24, v25);
                v23 = v27;
                v21 = v29;
              }
              *((_DWORD *)v18 + 4) = v21;
              *(v23 - 2) = 0;
            }
            else
            {
              v18[1] = (__int64)v23;
              *((_DWORD *)v18 + 4) = *((_DWORD *)v10 - 2);
              *((_DWORD *)v18 + 5) = *((_DWORD *)v10 - 1);
              *(v10 - 2) = v10;
              *((_DWORD *)v10 - 1) = 0;
              *((_DWORD *)v10 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v22 = *(v10 - 2);
          if ( (_QWORD *)v22 != v10 )
            _libc_free(v22);
        }
        if ( v8 == v10 + 64 )
          break;
        v10 += 67;
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[67 * *(unsigned int *)(a1 + 24)]; j != result; result += 67 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
