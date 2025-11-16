// Function: sub_B084F0
// Address: 0xb084f0
//
_QWORD *__fastcall sub_B084F0(__int64 a1, int a2)
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
  int v13; // r9d
  unsigned __int8 v14; // dl
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned __int8 v19; // dl
  __int64 *v20; // rcx
  int v21; // edx
  __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 *v24; // rcx
  __int64 v25; // r9
  int v26; // r11d
  __int64 *v27; // r10
  _QWORD *k; // rdx
  int v29; // [rsp+14h] [rbp-5Ch]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h] BYREF
  __int64 v32; // [rsp+28h] [rbp-48h] BYREF
  int v33; // [rsp+30h] [rbp-40h] BYREF
  int v34[15]; // [rsp+34h] [rbp-3Ch] BYREF

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
        v31 = *(_QWORD *)(v17 + 8);
        v18 = v12;
        if ( *(_BYTE *)v12 != 16 )
        {
          v19 = *(_BYTE *)(v12 - 16);
          if ( (v19 & 2) != 0 )
            v20 = *(__int64 **)(v12 - 32);
          else
            v20 = (__int64 *)(v16 - 8LL * ((v19 >> 2) & 0xF));
          v18 = *v20;
        }
        v32 = v18;
        v29 = v13;
        v33 = *(_DWORD *)(v12 + 4);
        v30 = v15;
        v34[0] = *(unsigned __int16 *)(v12 + 16);
        v21 = sub_AF7510(&v31, &v32, &v33, v34);
        v22 = *j;
        v23 = (v29 - 1) & v21;
        v24 = (__int64 *)(v30 + 8LL * v23);
        v25 = *v24;
        if ( *v24 != *j )
        {
          v26 = 1;
          v27 = 0;
          while ( v25 != -4096 )
          {
            if ( v25 != -8192 || v27 )
              v24 = v27;
            v23 = (v29 - 1) & (v26 + v23);
            v25 = *(_QWORD *)(v30 + 8LL * v23);
            if ( v25 == v22 )
            {
              v24 = (__int64 *)(v30 + 8LL * v23);
              goto LABEL_27;
            }
            ++v26;
            v27 = v24;
            v24 = (__int64 *)(v30 + 8LL * v23);
          }
          if ( v27 )
            v24 = v27;
        }
LABEL_27:
        *v24 = v22;
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
