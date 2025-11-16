// Function: sub_B06170
// Address: 0xb06170
//
_QWORD *__fastcall sub_B06170(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 *v5; // r15
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r14
  _QWORD *i; // rdx
  __int64 *v11; // rbx
  char *j; // r11
  __int64 v13; // rsi
  int v14; // r13d
  int v15; // r13d
  int v16; // eax
  __int64 v17; // rcx
  unsigned int v18; // eax
  __int64 *v19; // rdx
  __int64 v20; // rsi
  int v21; // r8d
  __int64 *v22; // rdi
  _QWORD *k; // rdx
  __int64 v24; // [rsp+8h] [rbp-F8h]
  __int64 v25; // [rsp+10h] [rbp-F0h]
  char *v26; // [rsp+18h] [rbp-E8h]
  char v27; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v29; // [rsp+30h] [rbp-D0h] BYREF
  int v30; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v31; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v32[4]; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v33[3]; // [rsp+68h] [rbp-98h] BYREF
  __int64 v34[7]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v35[9]; // [rsp+B8h] [rbp-48h] BYREF

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
    *(_QWORD *)(a1 + 16) = 0;
    v24 = 8 * v4;
    v9 = &v5[v4];
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    v11 = v5;
    for ( j = &v27; v9 != v11; ++v11 )
    {
      v13 = *v11;
      if ( *v11 != -8192 && v13 != -4096 )
      {
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = *v11;
          BUG();
        }
        v26 = j;
        v15 = v14 - 1;
        v25 = *(_QWORD *)(a1 + 8);
        sub_AF54F0((__int64)j, v13);
        v16 = sub_AFADE0(&v28, &v29, &v30, v32, &v31, v33, v34, v35);
        v17 = *v11;
        v18 = v15 & v16;
        j = v26;
        v19 = (__int64 *)(v25 + 8LL * v18);
        v20 = *v19;
        if ( *v19 != *v11 )
        {
          v21 = 1;
          v22 = 0;
          while ( v20 != -4096 )
          {
            if ( v20 != -8192 || v22 )
              v19 = v22;
            v18 = v15 & (v21 + v18);
            v20 = *(_QWORD *)(v25 + 8LL * v18);
            if ( v20 == v17 )
            {
              v19 = (__int64 *)(v25 + 8LL * v18);
              goto LABEL_21;
            }
            ++v21;
            v22 = v19;
            v19 = (__int64 *)(v25 + 8LL * v18);
          }
          if ( v22 )
            v19 = v22;
        }
LABEL_21:
        *v19 = v17;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v24, 8);
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
