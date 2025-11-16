// Function: sub_BCF650
// Address: 0xbcf650
//
_QWORD *__fastcall sub_BCF650(__int64 a1, int a2)
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
  _QWORD *v14; // rdi
  __int64 v15; // r9
  int v16; // eax
  int v17; // eax
  _QWORD *v18; // r10
  __int64 v19; // rsi
  int v20; // r11d
  unsigned int v21; // edx
  _QWORD *v22; // rcx
  __int64 v23; // rdi
  _QWORD *k; // rdx
  __int64 v25; // [rsp+10h] [rbp-70h]
  int v26; // [rsp+1Ch] [rbp-64h]
  __int64 v27[2]; // [rsp+28h] [rbp-58h] BYREF
  __int64 v28; // [rsp+38h] [rbp-48h]
  bool v29[64]; // [rsp+40h] [rbp-40h] BYREF

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
        v26 = *(_DWORD *)(a1 + 24);
        if ( !v26 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v14 = *(_QWORD **)(v13 + 16);
        v15 = *(_QWORD *)(a1 + 8);
        v28 = *(unsigned int *)(v13 + 12);
        v16 = *(_DWORD *)(v13 + 8);
        v25 = v15;
        v27[1] = (__int64)v14;
        v29[0] = (v16 & 0x200) != 0;
        v27[0] = sub_BCC330(v14, (__int64)&v14[v28]);
        v17 = sub_BCC160(v27, v29);
        v18 = 0;
        v19 = *j;
        v20 = 1;
        v21 = (v26 - 1) & v17;
        v22 = (_QWORD *)(v25 + 8LL * v21);
        v23 = *v22;
        if ( *v22 != *j )
        {
          while ( v23 != -4096 )
          {
            if ( !v18 && v23 == -8192 )
              v18 = v22;
            v21 = (v26 - 1) & (v20 + v21);
            v22 = (_QWORD *)(v25 + 8LL * v21);
            v23 = *v22;
            if ( *v22 == v19 )
              goto LABEL_13;
            ++v20;
          }
          if ( v18 )
            v22 = v18;
        }
LABEL_13:
        *v22 = v19;
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
