// Function: sub_15BA380
// Address: 0x15ba380
//
_QWORD *__fastcall sub_15BA380(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r15d
  __int64 v13; // rsi
  int v14; // r15d
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rsi
  unsigned int v18; // eax
  _QWORD *v19; // rdx
  __int64 v20; // rdi
  int v21; // r10d
  _QWORD *v22; // r9
  __int64 v23; // rdx
  _QWORD *k; // rdx
  __int64 v25; // [rsp+8h] [rbp-78h]
  int v26; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v27; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+30h] [rbp-50h]
  __int64 v30; // [rsp+38h] [rbp-48h]
  int v31; // [rsp+40h] [rbp-40h]
  int v32; // [rsp+44h] [rbp-3Ch] BYREF
  __int64 v33[7]; // [rsp+48h] [rbp-38h] BYREF

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
  result = (_QWORD *)sub_22077B0(8LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[v3];
    for ( i = &result[v7]; i != result; ++result )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v4; v8 != j; ++j )
    {
      v11 = *j;
      if ( *j != -16 && v11 != -8 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v28 = 0;
        v13 = *(unsigned int *)(v11 + 8);
        v27 = 0;
        v14 = v12 - 1;
        v25 = *(_QWORD *)(a1 + 8);
        v29 = v11 + 8 * (1 - v13);
        v30 = (-8 * (1 - v13)) >> 3;
        v31 = *(_DWORD *)(v11 + 4);
        v32 = *(unsigned __int16 *)(v11 + 2);
        v15 = *(_QWORD *)(v11 - 8LL * *(unsigned int *)(v11 + 8));
        v26 = v31;
        v33[0] = v15;
        v16 = sub_15B64F0(&v26, &v32, v33);
        v17 = *j;
        v18 = v14 & v16;
        v19 = (_QWORD *)(v25 + 8LL * v18);
        v20 = *v19;
        if ( *j != *v19 )
        {
          v21 = 1;
          v22 = 0;
          while ( v20 != -8 )
          {
            if ( v20 != -16 || v22 )
              v19 = v22;
            v18 = v14 & (v21 + v18);
            v20 = *(_QWORD *)(v25 + 8LL * v18);
            if ( v17 == v20 )
            {
              v19 = (_QWORD *)(v25 + 8LL * v18);
              goto LABEL_21;
            }
            ++v21;
            v22 = v19;
            v19 = (_QWORD *)(v25 + 8LL * v18);
          }
          if ( v22 )
            v19 = v22;
        }
LABEL_21:
        *v19 = v17;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v23]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
