// Function: sub_15C5700
// Address: 0x15c5700
//
_QWORD *__fastcall sub_15C5700(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned int **v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  unsigned int **v8; // r15
  _QWORD *i; // rdx
  unsigned int **j; // rbx
  unsigned int *v11; // rax
  int v12; // r14d
  __int64 v13; // rdx
  int v14; // r14d
  int v15; // eax
  unsigned int *v16; // rcx
  unsigned int v17; // eax
  unsigned int **v18; // rdx
  unsigned int *v19; // rsi
  int v20; // r8d
  unsigned int **v21; // rdi
  _QWORD *k; // rdx
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h] BYREF
  __int64 v25; // [rsp+18h] [rbp-68h] BYREF
  unsigned int v26; // [rsp+20h] [rbp-60h] BYREF
  __int64 v27; // [rsp+28h] [rbp-58h] BYREF
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-48h] BYREF
  __int64 v30[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(unsigned int ***)(a1 + 8);
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
      if ( *j != (unsigned int *)-16LL && v11 != (unsigned int *)-8LL )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v13 = v11[2];
        v14 = v12 - 1;
        v23 = *(_QWORD *)(a1 + 8);
        v24 = *(_QWORD *)&v11[-2 * v13];
        v25 = *(_QWORD *)&v11[2 * (1 - v13)];
        v26 = v11[6];
        v27 = *(_QWORD *)&v11[2 * (2 - v13)];
        v28 = *(_QWORD *)&v11[2 * (3 - v13)];
        v29 = v11[7];
        v30[0] = *(_QWORD *)&v11[2 * (4 - v13)];
        v15 = sub_15B52D0(&v24, &v25, (int *)&v26, &v27, &v28, (int *)&v29, v30);
        v16 = *j;
        v17 = v14 & v15;
        v18 = (unsigned int **)(v23 + 8LL * v17);
        v19 = *v18;
        if ( *v18 != *j )
        {
          v20 = 1;
          v21 = 0;
          while ( v19 != (unsigned int *)-8LL )
          {
            if ( v19 != (unsigned int *)-16LL || v21 )
              v18 = v21;
            v17 = v14 & (v20 + v17);
            v19 = *(unsigned int **)(v23 + 8LL * v17);
            if ( v19 == v16 )
            {
              v18 = (unsigned int **)(v23 + 8LL * v17);
              goto LABEL_21;
            }
            ++v20;
            v21 = v18;
            v18 = (unsigned int **)(v23 + 8LL * v17);
          }
          if ( v21 )
            v18 = v21;
        }
LABEL_21:
        *v18 = v16;
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
