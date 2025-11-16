// Function: sub_1041470
// Address: 0x1041470
//
_QWORD *__fastcall sub_1041470(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 *v4; // r15
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r13
  _QWORD *i; // rdx
  __int64 *j; // r14
  __int64 v12; // rax
  int v13; // edx
  int v14; // ecx
  __int64 v15; // rdi
  int v16; // r11d
  _QWORD *v17; // r10
  __int64 v18; // rsi
  _QWORD *v19; // rdx
  __int64 v20; // r9
  unsigned __int64 *v21; // r12
  unsigned __int64 *v22; // r15
  unsigned __int64 *v23; // rdi
  unsigned __int64 v24; // rcx
  __int64 v25; // rdx
  _QWORD *k; // rdx
  __int64 v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v28 = (__int64)v4;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v27 = 16 * v5;
    v9 = &v4[2 * v5];
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v4; v9 != j; j += 2 )
    {
      v12 = *j;
      if ( *j != -8192 && v12 != -4096 )
      {
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v19 = (_QWORD *)(v15 + 16 * v18);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -4096 )
          {
            if ( v20 == -8192 && !v17 )
              v17 = v19;
            v18 = v14 & (unsigned int)(v16 + v18);
            v19 = (_QWORD *)(v15 + 16LL * (unsigned int)v18);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_13;
            ++v16;
          }
          if ( v17 )
            v19 = v17;
        }
LABEL_13:
        *v19 = v12;
        v19[1] = j[1];
        j[1] = 0;
        ++*(_DWORD *)(a1 + 16);
        v21 = (unsigned __int64 *)j[1];
        if ( v21 )
        {
          v22 = (unsigned __int64 *)v21[1];
          while ( v21 != v22 )
          {
            v23 = v22;
            v22 = (unsigned __int64 *)v22[1];
            v24 = *v23 & 0xFFFFFFFFFFFFFFF8LL;
            *v22 = v24 | *v22 & 7;
            *(_QWORD *)(v24 + 8) = v22;
            *v23 &= 7u;
            v23 -= 4;
            v23[5] = 0;
            sub_BD72D0((__int64)v23, v18);
          }
          j_j___libc_free_0(v21, 16);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v28, v27, 8);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[2 * v25]; k != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
