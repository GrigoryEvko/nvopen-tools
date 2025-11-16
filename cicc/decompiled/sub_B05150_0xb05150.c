// Function: sub_B05150
// Address: 0xb05150
//
_QWORD *__fastcall sub_B05150(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 *v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 *v8; // rbx
  _QWORD *i; // rdx
  __int64 *j; // r15
  __int64 v11; // r12
  int v12; // r13d
  unsigned __int16 v13; // ax
  __int64 v14; // rcx
  unsigned __int8 v15; // al
  __int64 v16; // rsi
  unsigned __int8 v17; // al
  __int64 v18; // rsi
  unsigned __int8 v19; // al
  __int64 v20; // rsi
  unsigned __int8 v21; // al
  __int64 v22; // rcx
  unsigned int v23; // edx
  __int64 *v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rdi
  int v27; // r11d
  __int64 *v28; // r10
  __int64 v29; // rdx
  _QWORD *k; // rdx
  __int64 v31; // [rsp+0h] [rbp-90h]
  __int64 *v32; // [rsp+10h] [rbp-80h]
  __int64 v33; // [rsp+18h] [rbp-78h]
  int v34; // [rsp+20h] [rbp-70h] BYREF
  __int64 v35; // [rsp+28h] [rbp-68h] BYREF
  __int64 v36[4]; // [rsp+30h] [rbp-60h] BYREF
  int v37; // [rsp+50h] [rbp-40h]
  int v38[15]; // [rsp+54h] [rbp-3Ch] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v32 = v4;
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
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v31 = 8 * v5;
    v8 = &v4[v5];
    for ( i = &result[*(unsigned int *)(a1 + 24)]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v32; v8 != j; ++j )
    {
      v11 = *j;
      if ( *j != -8192 && v11 != -4096 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v33 = *(_QWORD *)(a1 + 8);
        v13 = sub_AF18C0(*j);
        v14 = v11 - 16;
        v34 = v13;
        v15 = *(_BYTE *)(v11 - 16);
        if ( (v15 & 2) != 0 )
          v16 = *(_QWORD *)(v11 - 32);
        else
          v16 = v14 - 8LL * ((v15 >> 2) & 0xF);
        v35 = *(_QWORD *)(v16 + 16);
        v17 = *(_BYTE *)(v11 - 16);
        if ( (v17 & 2) != 0 )
          v18 = *(_QWORD *)(v11 - 32);
        else
          v18 = v14 - 8LL * ((v17 >> 2) & 0xF);
        v36[0] = *(_QWORD *)(v18 + 24);
        v19 = *(_BYTE *)(v11 - 16);
        if ( (v19 & 2) != 0 )
          v20 = *(_QWORD *)(v11 - 32);
        else
          v20 = v14 - 8LL * ((v19 >> 2) & 0xF);
        v36[1] = *(_QWORD *)(v20 + 32);
        v21 = *(_BYTE *)(v11 - 16);
        if ( (v21 & 2) != 0 )
          v22 = *(_QWORD *)(v11 - 32);
        else
          v22 = v14 - 8LL * ((v21 >> 2) & 0xF);
        v36[2] = *(_QWORD *)(v22 + 40);
        v36[3] = *(_QWORD *)(v11 + 24);
        v37 = sub_AF18D0(v11);
        v38[0] = *(_DWORD *)(v11 + 44);
        v23 = (v12 - 1) & sub_AFB0E0(&v34, &v35, v36, v38);
        v24 = (__int64 *)(v33 + 8LL * v23);
        v25 = *j;
        v26 = *v24;
        if ( *j != *v24 )
        {
          v27 = 1;
          v28 = 0;
          while ( v26 != -4096 )
          {
            if ( v26 != -8192 || v28 )
              v24 = v28;
            v23 = (v12 - 1) & (v27 + v23);
            v26 = *(_QWORD *)(v33 + 8LL * v23);
            if ( v25 == v26 )
            {
              v24 = (__int64 *)(v33 + 8LL * v23);
              goto LABEL_29;
            }
            ++v27;
            v28 = v24;
            v24 = (__int64 *)(v33 + 8LL * v23);
          }
          if ( v28 )
            v24 = v28;
        }
LABEL_29:
        *v24 = v25;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v32, v31, 8);
  }
  else
  {
    v29 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v29]; k != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
