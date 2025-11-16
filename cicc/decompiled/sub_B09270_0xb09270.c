// Function: sub_B09270
// Address: 0xb09270
//
_QWORD *__fastcall sub_B09270(__int64 a1, int a2)
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
  __int64 v17; // rdi
  unsigned __int8 v18; // dl
  __int64 v19; // rsi
  int v20; // edx
  __int64 v21; // rsi
  unsigned int v22; // edx
  _QWORD *v23; // rcx
  __int64 v24; // rdi
  int v25; // r11d
  _QWORD *v26; // r10
  _QWORD *k; // rdx
  int v28; // [rsp+14h] [rbp-5Ch]
  __int64 v29; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+20h] [rbp-50h] BYREF
  __int64 v31; // [rsp+28h] [rbp-48h] BYREF
  char v32; // [rsp+30h] [rbp-40h]

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
          v17 = *(_QWORD *)(v12 - 32);
        else
          v17 = v16 - 8LL * ((v14 >> 2) & 0xF);
        v30 = *(_QWORD *)(v17 + 8);
        v18 = *(_BYTE *)(v12 - 16);
        if ( (v18 & 2) != 0 )
          v19 = *(_QWORD *)(v12 - 32);
        else
          v19 = v16 - 8LL * ((v18 >> 2) & 0xF);
        v28 = v13;
        v29 = v15;
        v31 = *(_QWORD *)(v19 + 16);
        v32 = *(_BYTE *)(v12 + 1) >> 7;
        v20 = sub_AFB5F0(&v30, &v31);
        v21 = *j;
        v22 = (v28 - 1) & v20;
        v23 = (_QWORD *)(v29 + 8LL * v22);
        v24 = *v23;
        if ( *v23 != *j )
        {
          v25 = 1;
          v26 = 0;
          while ( v24 != -4096 )
          {
            if ( v24 != -8192 || v26 )
              v23 = v26;
            v22 = (v28 - 1) & (v25 + v22);
            v24 = *(_QWORD *)(v29 + 8LL * v22);
            if ( v24 == v21 )
            {
              v23 = (_QWORD *)(v29 + 8LL * v22);
              goto LABEL_25;
            }
            ++v25;
            v26 = v23;
            v23 = (_QWORD *)(v29 + 8LL * v22);
          }
          if ( v26 )
            v23 = v26;
        }
LABEL_25:
        *v23 = v21;
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
