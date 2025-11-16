// Function: sub_1425BA0
// Address: 0x1425ba0
//
_QWORD *__fastcall sub_1425BA0(__int64 a1, int a2)
{
  _QWORD *v3; // r14
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  _QWORD *v9; // r13
  _QWORD *i; // rdx
  __int64 v11; // rax
  int v12; // edx
  int v13; // edi
  __int64 v14; // r8
  int v15; // r12d
  _QWORD *v16; // r10
  __int64 v17; // rsi
  _QWORD *v18; // rdx
  __int64 v19; // r9
  unsigned __int64 *v20; // r12
  unsigned __int64 *v21; // r15
  unsigned __int64 *v22; // rdi
  unsigned __int64 v23; // rdx
  __int64 v24; // rdx
  _QWORD *j; // rdx
  _QWORD *v26; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD **)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  v26 = v3;
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
  result = (_QWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = &v3[2 * v4];
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
    for ( ; v9 != v3; v3 += 2 )
    {
      v11 = *v3;
      if ( *v3 != -16 && v11 != -8 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *v3;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 1;
        v16 = 0;
        v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v18 = (_QWORD *)(v14 + 16 * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -8 )
          {
            if ( v19 == -16 && !v16 )
              v16 = v18;
            v17 = v13 & (unsigned int)(v15 + v17);
            v18 = (_QWORD *)(v14 + 16LL * (unsigned int)v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_13;
            ++v15;
          }
          if ( v16 )
            v18 = v16;
        }
LABEL_13:
        *v18 = v11;
        v18[1] = v3[1];
        v3[1] = 0;
        ++*(_DWORD *)(a1 + 16);
        v20 = (unsigned __int64 *)v3[1];
        if ( v20 )
        {
          v21 = (unsigned __int64 *)v20[1];
          while ( v20 != v21 )
          {
            v22 = v21;
            v21 = (unsigned __int64 *)v21[1];
            v23 = *v22 & 0xFFFFFFFFFFFFFFF8LL;
            *v21 = v23 | *v21 & 7;
            *(_QWORD *)(v23 + 8) = v21;
            *v22 &= 7u;
            v22 -= 4;
            v22[5] = 0;
            sub_164BEC0(v22, v17, v23, v7, v14);
          }
          j_j___libc_free_0(v20, 16);
        }
      }
    }
    return (_QWORD *)j___libc_free_0(v26);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v24]; j != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
