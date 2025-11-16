// Function: sub_B09F80
// Address: 0xb09f80
//
_QWORD *__fastcall sub_B09F80(__int64 a1, int a2)
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
  int v13; // r10d
  __int64 v14; // r8
  __int64 v15; // rdx
  unsigned __int8 v16; // cl
  __int64 v17; // rsi
  unsigned __int8 v18; // cl
  __int64 v19; // rsi
  unsigned __int8 v20; // cl
  __int64 v21; // rsi
  unsigned __int8 v22; // cl
  __int64 v23; // rsi
  unsigned __int8 v24; // cl
  __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // rsi
  unsigned int v28; // edx
  _QWORD *v29; // rcx
  __int64 v30; // rdi
  int v31; // r11d
  _QWORD *v32; // r10
  _QWORD *k; // rdx
  int v34; // [rsp+14h] [rbp-7Ch]
  __int64 v35; // [rsp+18h] [rbp-78h]
  __int64 v36; // [rsp+28h] [rbp-68h] BYREF
  __int64 v37; // [rsp+30h] [rbp-60h] BYREF
  __int64 v38; // [rsp+38h] [rbp-58h] BYREF
  __int64 v39[2]; // [rsp+40h] [rbp-50h] BYREF
  int v40; // [rsp+50h] [rbp-40h]
  char v41; // [rsp+54h] [rbp-3Ch]

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
        v15 = v12 - 16;
        v16 = *(_BYTE *)(v12 - 16);
        if ( (v16 & 2) != 0 )
          v17 = *(_QWORD *)(v12 - 32);
        else
          v17 = v15 - 8LL * ((v16 >> 2) & 0xF);
        v36 = *(_QWORD *)(v17 + 8);
        v18 = *(_BYTE *)(v12 - 16);
        if ( (v18 & 2) != 0 )
          v19 = *(_QWORD *)(v12 - 32);
        else
          v19 = v15 - 8LL * ((v18 >> 2) & 0xF);
        v37 = *(_QWORD *)(v19 + 16);
        v20 = *(_BYTE *)(v12 - 16);
        if ( (v20 & 2) != 0 )
          v21 = *(_QWORD *)(v12 - 32);
        else
          v21 = v15 - 8LL * ((v20 >> 2) & 0xF);
        v38 = *(_QWORD *)(v21 + 24);
        v22 = *(_BYTE *)(v12 - 16);
        if ( (v22 & 2) != 0 )
          v23 = *(_QWORD *)(v12 - 32);
        else
          v23 = v15 - 8LL * ((v22 >> 2) & 0xF);
        v39[0] = *(_QWORD *)(v23 + 32);
        v24 = *(_BYTE *)(v12 - 16);
        if ( (v24 & 2) != 0 )
          v25 = *(_QWORD *)(v12 - 32);
        else
          v25 = v15 - 8LL * ((v24 >> 2) & 0xF);
        v34 = v13;
        v39[1] = *(_QWORD *)(v25 + 40);
        v35 = v14;
        v40 = *(_DWORD *)(v12 + 4);
        v41 = *(_BYTE *)(v12 + 1) >> 7;
        v26 = sub_AFBE30(&v36, &v37, &v38, v39);
        v27 = *j;
        v28 = (v34 - 1) & v26;
        v29 = (_QWORD *)(v35 + 8LL * v28);
        v30 = *v29;
        if ( *j != *v29 )
        {
          v31 = 1;
          v32 = 0;
          while ( v30 != -4096 )
          {
            if ( v30 != -8192 || v32 )
              v29 = v32;
            v28 = (v34 - 1) & (v31 + v28);
            v30 = *(_QWORD *)(v35 + 8LL * v28);
            if ( v30 == v27 )
            {
              v29 = (_QWORD *)(v35 + 8LL * v28);
              goto LABEL_31;
            }
            ++v31;
            v32 = v29;
            v29 = (_QWORD *)(v35 + 8LL * v28);
          }
          if ( v32 )
            v29 = v32;
        }
LABEL_31:
        *v29 = v27;
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
