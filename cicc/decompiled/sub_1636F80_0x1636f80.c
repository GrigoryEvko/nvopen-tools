// Function: sub_1636F80
// Address: 0x1636f80
//
_QWORD *__fastcall sub_1636F80(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *v10; // r12
  __int64 v11; // rax
  int v12; // edx
  int v13; // esi
  __int64 v14; // rdi
  int v15; // r10d
  _QWORD *v16; // r9
  unsigned int v17; // ecx
  _QWORD *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rsi
  _QWORD *v21; // rax
  _QWORD *v22; // rcx
  __int64 *v23; // rbx
  __int64 *v24; // r15
  __int64 v25; // rdi
  __int64 v26; // rdx
  _QWORD *j; // rdx
  __int64 *v28; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v28 = v4;
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
  result = (_QWORD *)sub_22077B0(32LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[4 * v3];
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
        *result = -8;
    }
    v10 = v28 + 1;
    if ( v8 != v28 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 1);
        if ( v11 != -16 && v11 != -8 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(v10 - 1);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (_QWORD *)(v14 + 32LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( v19 == -16 && !v16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (_QWORD *)(v14 + 32LL * v17);
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
          v20 = *v10;
          v21 = v18 + 1;
          v18[1] = *v10;
          v22 = (_QWORD *)v10[1];
          v18[2] = v22;
          v18[3] = v10[2];
          if ( v10 == (__int64 *)v20 )
          {
            v18[2] = v21;
            v18[1] = v21;
          }
          else
          {
            *v22 = v21;
            v18 = (_QWORD *)v18[1];
            v18[1] = v21;
            v10[1] = (__int64)v10;
            *v10 = (__int64)v10;
            v10[2] = 0;
          }
          ++*(_DWORD *)(a1 + 16);
          v23 = (__int64 *)*v10;
          while ( v10 != v23 )
          {
            v24 = v23;
            v23 = (__int64 *)*v23;
            v25 = v24[3];
            if ( v25 )
              (*(void (__fastcall **)(__int64, __int64, _QWORD *, _QWORD *, __int64, _QWORD *))(*(_QWORD *)v25 + 8LL))(
                v25,
                v20,
                v18,
                v22,
                v19,
                v16);
            v20 = 32;
            j_j___libc_free_0(v24, 32);
          }
        }
        if ( v8 == v10 + 3 )
          break;
        v10 += 4;
      }
    }
    return (_QWORD *)j___libc_free_0(v28);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v26]; j != result; result += 4 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
