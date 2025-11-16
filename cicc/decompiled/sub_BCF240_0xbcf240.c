// Function: sub_BCF240
// Address: 0xbcf240
//
_QWORD *__fastcall sub_BCF240(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r15
  __int64 *v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 *v10; // r13
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v13; // rax
  __int64 *v14; // rsi
  __int64 v15; // rdx
  int v16; // eax
  _QWORD *v17; // r10
  __int64 v18; // rsi
  int v19; // r11d
  unsigned int v20; // edx
  _QWORD *v21; // rcx
  __int64 v22; // rdi
  _QWORD *k; // rdx
  __int64 v24; // [rsp+10h] [rbp-70h]
  int v25; // [rsp+1Ch] [rbp-64h]
  unsigned __int64 v26; // [rsp+28h] [rbp-58h] BYREF
  __int64 v27[3]; // [rsp+30h] [rbp-50h] BYREF
  bool v28[56]; // [rsp+48h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(8LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v5[v9];
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v10 != j; ++j )
    {
      v13 = *j;
      if ( *j != -8192 && v13 != -4096 )
      {
        v25 = *(_DWORD *)(a1 + 24);
        if ( !v25 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v14 = *(__int64 **)(v13 + 16);
        v24 = *(_QWORD *)(a1 + 8);
        v27[0] = *v14;
        v15 = *(unsigned int *)(v13 + 12);
        v27[1] = (__int64)(v14 + 1);
        v15 *= 8;
        v27[2] = (v15 - 8) >> 3;
        v28[0] = *(_DWORD *)(v13 + 8) >> 8 != 0;
        v26 = sub_BCC330(v14 + 1, (__int64)v14 + v15);
        v16 = sub_BCC1E0(v27, (__int64 *)&v26, v28);
        v17 = 0;
        v18 = *j;
        v19 = 1;
        v20 = (v25 - 1) & v16;
        v21 = (_QWORD *)(v24 + 8LL * v20);
        v22 = *v21;
        if ( *j != *v21 )
        {
          while ( v22 != -4096 )
          {
            if ( !v17 && v22 == -8192 )
              v17 = v21;
            v20 = (v25 - 1) & (v19 + v20);
            v21 = (_QWORD *)(v24 + 8LL * v20);
            v22 = *v21;
            if ( v18 == *v21 )
              goto LABEL_13;
            ++v19;
          }
          if ( v17 )
            v21 = v17;
        }
LABEL_13:
        *v21 = v18;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v9 * 8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[*(unsigned int *)(a1 + 24)]; k != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
