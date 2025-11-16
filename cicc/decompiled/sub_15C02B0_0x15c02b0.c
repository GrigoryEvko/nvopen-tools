// Function: sub_15C02B0
// Address: 0x15c02b0
//
_QWORD *__fastcall sub_15C02B0(__int64 a1, int a2)
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
  __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rcx
  int v16; // edx
  int v17; // eax
  int v18; // r15d
  int v19; // eax
  __int64 v20; // rcx
  unsigned int v21; // eax
  __int64 *v22; // rdx
  __int64 v23; // rsi
  int v24; // r10d
  __int64 *v25; // rdi
  __int64 v26; // rdx
  _QWORD *k; // rdx
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h] BYREF
  __int64 v30; // [rsp+18h] [rbp-48h] BYREF
  int v31; // [rsp+20h] [rbp-40h] BYREF
  int v32[15]; // [rsp+24h] [rbp-3Ch] BYREF

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
        v13 = *(unsigned int *)(v11 + 8);
        v14 = *(_QWORD *)(a1 + 8);
        v15 = *j;
        v29 = *(_QWORD *)(v11 + 8 * (1 - v13));
        if ( *(_BYTE *)v11 != 15 )
          v15 = *(_QWORD *)(v11 - 8 * v13);
        v16 = *(_DWORD *)(v11 + 24);
        v17 = *(unsigned __int16 *)(v11 + 28);
        v30 = v15;
        v28 = v14;
        v18 = v12 - 1;
        v31 = v16;
        v32[0] = v17;
        v19 = sub_15B2700(&v29, &v30, &v31, v32);
        v20 = *j;
        v21 = v18 & v19;
        v22 = (__int64 *)(v28 + 8LL * v21);
        v23 = *v22;
        if ( *v22 != *j )
        {
          v24 = 1;
          v25 = 0;
          while ( v23 != -8 )
          {
            if ( v23 != -16 || v25 )
              v22 = v25;
            v21 = v18 & (v24 + v21);
            v23 = *(_QWORD *)(v28 + 8LL * v21);
            if ( v23 == v20 )
            {
              v22 = (__int64 *)(v28 + 8LL * v21);
              goto LABEL_23;
            }
            ++v24;
            v25 = v22;
            v22 = (__int64 *)(v28 + 8LL * v21);
          }
          if ( v25 )
            v22 = v25;
        }
LABEL_23:
        *v22 = v20;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v26]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
