// Function: sub_15BB7A0
// Address: 0x15bb7a0
//
_QWORD *__fastcall sub_15BB7A0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r15
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r14d
  __int64 v13; // r11
  int v14; // r14d
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rcx
  unsigned int v18; // eax
  _QWORD *v19; // rdx
  __int64 v20; // rsi
  int v21; // r8d
  _QWORD *v22; // rdi
  _QWORD *k; // rdx
  __int64 v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+18h] [rbp-68h] BYREF
  char v27[8]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v28; // [rsp+28h] [rbp-58h] BYREF
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+38h] [rbp-48h] BYREF
  __int64 v31[8]; // [rsp+40h] [rbp-40h] BYREF

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
    v8 = &v4[v3];
    *(_QWORD *)(a1 + 16) = 0;
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
        v13 = *(_QWORD *)(a1 + 8);
        v14 = v12 - 1;
        v25 = *(_QWORD *)(v11 + 24);
        v24 = v13;
        v26 = *(_QWORD *)(v11 + 32);
        v27[0] = *(_BYTE *)(v11 + 40);
        v15 = *(unsigned int *)(v11 + 8);
        v28 = *(_QWORD *)(v11 - 8 * v15);
        v29 = *(_QWORD *)(v11 + 8 * (1 - v15));
        v30 = *(_QWORD *)(v11 + 8 * (2 - v15));
        v31[0] = *(_QWORD *)(v11 + 8 * (3 - v15));
        v16 = sub_15B3480(&v25, &v26, v27, &v30, v31, &v28, &v29);
        v17 = *j;
        v18 = v14 & v16;
        v19 = (_QWORD *)(v24 + 8LL * v18);
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
            v20 = *(_QWORD *)(v24 + 8LL * v18);
            if ( v17 == v20 )
            {
              v19 = (_QWORD *)(v24 + 8LL * v18);
              goto LABEL_21;
            }
            ++v21;
            v22 = v19;
            v19 = (_QWORD *)(v24 + 8LL * v18);
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
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[*(unsigned int *)(a1 + 24)]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
