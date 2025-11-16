// Function: sub_1B509C0
// Address: 0x1b509c0
//
_QWORD *__fastcall sub_1B509C0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 v7; // rcx
  _QWORD *v8; // r15
  _QWORD *i; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // rdi
  int v15; // r10d
  __int64 *v16; // r9
  unsigned int v17; // edx
  __int64 *v18; // r8
  __int64 v19; // rsi
  void *v20; // rdi
  unsigned int v21; // r9d
  unsigned __int64 v22; // rdi
  _DWORD *v23; // r10
  const void *v24; // rsi
  size_t v25; // rdx
  _QWORD *j; // rdx
  __int64 *v27; // [rsp+8h] [rbp-48h]
  __int64 *v28; // [rsp+8h] [rbp-48h]
  _DWORD *v29; // [rsp+10h] [rbp-40h]
  _DWORD *v30; // [rsp+10h] [rbp-40h]
  unsigned int v31; // [rsp+1Ch] [rbp-34h]
  unsigned int v32; // [rsp+1Ch] [rbp-34h]

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
          v18 = (__int64 *)(v14 + 56LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( !v16 && v19 == -16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 56LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_15;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_15:
          v20 = v18 + 3;
          *v18 = v11;
          v18[1] = (__int64)(v18 + 3);
          v18[2] = 0x400000000LL;
          v21 = *((_DWORD *)v10 - 2);
          if ( v21 && v18 + 1 != v10 - 2 )
          {
            v23 = (_DWORD *)*(v10 - 2);
            if ( v10 == (_QWORD *)v23 )
            {
              v24 = v10;
              v25 = 8LL * v21;
              if ( v21 <= 4 )
                goto LABEL_24;
              v28 = v18;
              v30 = (_DWORD *)*(v10 - 2);
              v32 = *((_DWORD *)v10 - 2);
              sub_16CD150((__int64)(v18 + 1), v18 + 3, v21, 8, (int)v18, v21);
              v18 = v28;
              v24 = (const void *)*(v10 - 2);
              v21 = v32;
              v25 = 8LL * *((unsigned int *)v10 - 2);
              v20 = (void *)v28[1];
              v23 = v30;
              if ( v25 )
              {
LABEL_24:
                v27 = v18;
                v29 = v23;
                v31 = v21;
                memcpy(v20, v24, v25);
                v18 = v27;
                v23 = v29;
                v21 = v31;
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
        if ( v8 == v10 + 4 )
          break;
        v10 += 7;
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * *(unsigned int *)(a1 + 24)]; j != result; result += 7 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
