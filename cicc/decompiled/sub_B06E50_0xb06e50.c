// Function: sub_B06E50
// Address: 0xb06e50
//
_QWORD *__fastcall sub_B06E50(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 *v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r15
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v12; // rax
  int v13; // ecx
  __int64 v14; // r8
  unsigned __int8 v15; // dl
  __int64 v16; // rax
  int v17; // edx
  __int64 v18; // rsi
  unsigned int v19; // edx
  _QWORD *v20; // rcx
  __int64 v21; // r9
  int v22; // r11d
  _QWORD *v23; // r10
  _QWORD *k; // rdx
  int v25; // [rsp+14h] [rbp-4Ch]
  __int64 v26; // [rsp+18h] [rbp-48h]
  int v27; // [rsp+20h] [rbp-40h] BYREF
  __int8 v28[4]; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v29[7]; // [rsp+28h] [rbp-38h] BYREF

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
    v9 = &v5[v4];
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v9 != j; ++j )
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
        v14 = *(_QWORD *)(a1 + 8);
        v27 = *(_DWORD *)(v12 + 20);
        v28[0] = *(_BYTE *)(v12 + 44);
        v15 = *(_BYTE *)(v12 - 16);
        if ( (v15 & 2) != 0 )
          v16 = *(_QWORD *)(v12 - 32);
        else
          v16 = v12 - 16 - 8LL * ((v15 >> 2) & 0xF);
        v25 = v13;
        v26 = v14;
        v29[0] = *(_QWORD *)(v16 + 24);
        v17 = sub_AF8410(&v27, v28, v29);
        v18 = *j;
        v19 = (v25 - 1) & v17;
        v20 = (_QWORD *)(v26 + 8LL * v19);
        v21 = *v20;
        if ( *j != *v20 )
        {
          v22 = 1;
          v23 = 0;
          while ( v21 != -4096 )
          {
            if ( v21 != -8192 || v23 )
              v20 = v23;
            v19 = (v25 - 1) & (v22 + v19);
            v21 = *(_QWORD *)(v26 + 8LL * v19);
            if ( v21 == v18 )
            {
              v20 = (_QWORD *)(v26 + 8LL * v19);
              goto LABEL_23;
            }
            ++v22;
            v23 = v20;
            v20 = (_QWORD *)(v26 + 8LL * v19);
          }
          if ( v23 )
            v20 = v23;
        }
LABEL_23:
        *v20 = v18;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, 8 * v4, 8);
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
