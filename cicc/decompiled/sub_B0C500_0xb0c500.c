// Function: sub_B0C500
// Address: 0xb0c500
//
_QWORD *__fastcall sub_B0C500(__int64 a1, int a2)
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
  unsigned __int8 v14; // dl
  __int64 v15; // r8
  __int64 v16; // rsi
  __int64 *v17; // r9
  unsigned __int8 v18; // dl
  __int64 v19; // r9
  unsigned __int8 v20; // dl
  __int64 v21; // rsi
  int v22; // edx
  __int64 v23; // rsi
  unsigned int v24; // edx
  _QWORD *v25; // rcx
  __int64 v26; // r9
  int v27; // r11d
  _QWORD *v28; // r10
  _QWORD *k; // rdx
  int v30; // [rsp+14h] [rbp-5Ch]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h] BYREF
  __int64 v33[2]; // [rsp+28h] [rbp-48h] BYREF
  int v34[14]; // [rsp+38h] [rbp-38h] BYREF

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
        v14 = *(_BYTE *)(v12 - 16);
        v15 = *(_QWORD *)(a1 + 8);
        v16 = v12 - 16;
        if ( (v14 & 2) != 0 )
          v17 = *(__int64 **)(v12 - 32);
        else
          v17 = (__int64 *)(v16 - 8LL * ((v14 >> 2) & 0xF));
        v32 = *v17;
        v18 = *(_BYTE *)(v12 - 16);
        if ( (v18 & 2) != 0 )
          v19 = *(_QWORD *)(v12 - 32);
        else
          v19 = v16 - 8LL * ((v18 >> 2) & 0xF);
        v33[0] = *(_QWORD *)(v19 + 8);
        v20 = *(_BYTE *)(v12 - 16);
        if ( (v20 & 2) != 0 )
          v21 = *(_QWORD *)(v12 - 32);
        else
          v21 = v16 - 8LL * ((v20 >> 2) & 0xF);
        v30 = v13;
        v31 = v15;
        v33[1] = *(_QWORD *)(v21 + 16);
        v34[0] = *(_DWORD *)(v12 + 4);
        v22 = sub_AF8830(&v32, v33, v34);
        v23 = *j;
        v24 = (v30 - 1) & v22;
        v25 = (_QWORD *)(v31 + 8LL * v24);
        v26 = *v25;
        if ( *v25 != *j )
        {
          v27 = 1;
          v28 = 0;
          while ( v26 != -4096 )
          {
            if ( v26 != -8192 || v28 )
              v25 = v28;
            v24 = (v30 - 1) & (v27 + v24);
            v26 = *(_QWORD *)(v31 + 8LL * v24);
            if ( v26 == v23 )
            {
              v25 = (_QWORD *)(v31 + 8LL * v24);
              goto LABEL_27;
            }
            ++v27;
            v28 = v25;
            v25 = (_QWORD *)(v31 + 8LL * v24);
          }
          if ( v28 )
            v25 = v28;
        }
LABEL_27:
        *v25 = v23;
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
