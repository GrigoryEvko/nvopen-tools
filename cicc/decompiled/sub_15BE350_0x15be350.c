// Function: sub_15BE350
// Address: 0x15be350
//
_QWORD *__fastcall sub_15BE350(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r14
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r15
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // r11
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r13d
  int v17; // eax
  __int64 v18; // rcx
  unsigned int v19; // eax
  __int64 *v20; // rdx
  __int64 v21; // rsi
  int v22; // r8d
  __int64 *v23; // rdi
  _QWORD *k; // rdx
  __int64 v25; // [rsp+8h] [rbp-88h]
  __int64 v26; // [rsp+18h] [rbp-78h] BYREF
  __int64 v27; // [rsp+20h] [rbp-70h] BYREF
  int v28; // [rsp+28h] [rbp-68h] BYREF
  __int64 v29; // [rsp+30h] [rbp-60h] BYREF
  __int64 v30[3]; // [rsp+38h] [rbp-58h] BYREF
  int v31; // [rsp+50h] [rbp-40h]
  int v32; // [rsp+54h] [rbp-3Ch]
  __int64 v33[7]; // [rsp+58h] [rbp-38h] BYREF

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
        v14 = *(unsigned int *)(v11 + 8);
        v26 = *(_QWORD *)(v11 + 8 * (2 - v14));
        v15 = v11;
        if ( *(_BYTE *)v11 != 15 )
          v15 = *(_QWORD *)(v11 - 8LL * *(unsigned int *)(v11 + 8));
        v27 = v15;
        v25 = v13;
        v16 = v12 - 1;
        v28 = *(_DWORD *)(v11 + 24);
        v29 = *(_QWORD *)(v11 + 8 * (1 - v14));
        v30[0] = *(_QWORD *)(v11 + 8 * (3 - v14));
        v30[1] = *(_QWORD *)(v11 + 32);
        v30[2] = *(_QWORD *)(v11 + 40);
        v31 = *(_DWORD *)(v11 + 48);
        v32 = *(_DWORD *)(v11 + 28);
        v33[0] = *(_QWORD *)(v11 + 8 * (4 - v14));
        v17 = sub_15B5D10(&v26, &v27, &v28, v30, &v29, v33);
        v18 = *j;
        v19 = v16 & v17;
        v20 = (__int64 *)(v25 + 8LL * v19);
        v21 = *v20;
        if ( *v20 != *j )
        {
          v22 = 1;
          v23 = 0;
          while ( v21 != -8 )
          {
            if ( v21 != -16 || v23 )
              v20 = v23;
            v19 = v16 & (v22 + v19);
            v21 = *(_QWORD *)(v25 + 8LL * v19);
            if ( v21 == v18 )
            {
              v20 = (__int64 *)(v25 + 8LL * v19);
              goto LABEL_23;
            }
            ++v22;
            v23 = v20;
            v20 = (__int64 *)(v25 + 8LL * v19);
          }
          if ( v23 )
            v20 = v23;
        }
LABEL_23:
        *v20 = v18;
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
