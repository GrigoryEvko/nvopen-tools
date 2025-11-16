// Function: sub_D33BC0
// Address: 0xd33bc0
//
unsigned __int64 __fastcall sub_D33BC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned int v7; // edx
  unsigned __int64 v8; // rdi
  _QWORD *v9; // rax
  __int64 v10; // rdx
  _QWORD *i; // rdx
  unsigned __int64 result; // rax
  __int64 v13; // r12
  char *v14; // rax
  __int64 v15; // r14
  char *v16; // r13
  __int64 v17; // rsi

  v3 = *(_QWORD *)(a2 + 40) - *(_QWORD *)(a2 + 32);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = 0;
  v4 = (unsigned int)(v3 >> 3) | ((unsigned __int64)(unsigned int)(v3 >> 3) >> 1);
  v5 = (((v4 >> 2) | v4) >> 4) | (v4 >> 2) | v4;
  v6 = (((v5 >> 8) | v5) >> 16) | (v5 >> 8) | v5;
  if ( (_DWORD)v6 == -1 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    *(_DWORD *)(a1 + 32) = 0;
  }
  else
  {
    v7 = 4 * ((int)v6 + 1) / 3u;
    v8 = ((((((((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
            | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
            | (v7 + 1)
            | ((unsigned __int64)(v7 + 1) >> 1)) >> 8)
          | (((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
          | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
          | (v7 + 1)
          | ((unsigned __int64)(v7 + 1) >> 1)) >> 16)
        | (((((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
          | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
          | (v7 + 1)
          | ((unsigned __int64)(v7 + 1) >> 1)) >> 8)
        | (((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
        | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
        | (v7 + 1)
        | ((unsigned __int64)(v7 + 1) >> 1))
       + 1;
    *(_DWORD *)(a1 + 32) = v8;
    v9 = (_QWORD *)sub_C7D670(16 * v8, 8);
    v10 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v9;
    for ( i = &v9[2 * v10]; i != v9; v9 += 2 )
    {
      if ( v9 )
        *v9 = -4096;
    }
  }
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  result = (unsigned int)((__int64)(*(_QWORD *)(a2 + 40) - *(_QWORD *)(a2 + 32)) >> 3);
  if ( (_DWORD)result )
  {
    v13 = 8LL * (unsigned int)result;
    v14 = (char *)sub_22077B0(v13);
    v15 = *(_QWORD *)(a1 + 40);
    v16 = v14;
    if ( *(_QWORD *)(a1 + 48) - v15 > 0 )
    {
      memmove(v14, *(const void **)(a1 + 40), *(_QWORD *)(a1 + 48) - v15);
      v17 = *(_QWORD *)(a1 + 56) - v15;
    }
    else
    {
      if ( !v15 )
      {
LABEL_9:
        result = (unsigned __int64)&v16[v13];
        *(_QWORD *)(a1 + 40) = v16;
        *(_QWORD *)(a1 + 48) = v16;
        *(_QWORD *)(a1 + 56) = &v16[v13];
        return result;
      }
      v17 = *(_QWORD *)(a1 + 56) - v15;
    }
    j_j___libc_free_0(v15, v17);
    goto LABEL_9;
  }
  return result;
}
