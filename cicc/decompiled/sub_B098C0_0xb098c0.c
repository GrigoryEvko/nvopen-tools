// Function: sub_B098C0
// Address: 0xb098c0
//
_QWORD *__fastcall sub_B098C0(__int64 a1, int a2)
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
  unsigned __int8 v14; // dl
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 *v17; // rsi
  unsigned __int8 v18; // dl
  __int64 v19; // rsi
  unsigned __int8 v20; // dl
  __int64 v21; // rsi
  unsigned __int8 v22; // dl
  __int64 v23; // rcx
  int v24; // edx
  __int64 v25; // rsi
  unsigned int v26; // edx
  _QWORD *v27; // rcx
  __int64 v28; // r8
  int v29; // r11d
  _QWORD *v30; // r10
  _QWORD *k; // rdx
  int v32; // [rsp+14h] [rbp-6Ch]
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+20h] [rbp-60h] BYREF
  __int64 v35; // [rsp+28h] [rbp-58h] BYREF
  __int64 v36; // [rsp+30h] [rbp-50h] BYREF
  __int64 v37; // [rsp+38h] [rbp-48h] BYREF
  int v38[16]; // [rsp+40h] [rbp-40h] BYREF

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
        v34 = *v17;
        v18 = *(_BYTE *)(v12 - 16);
        if ( (v18 & 2) != 0 )
          v19 = *(_QWORD *)(v12 - 32);
        else
          v19 = v16 - 8LL * ((v18 >> 2) & 0xF);
        v35 = *(_QWORD *)(v19 + 8);
        v20 = *(_BYTE *)(v12 - 16);
        if ( (v20 & 2) != 0 )
          v21 = *(_QWORD *)(v12 - 32);
        else
          v21 = v16 - 8LL * ((v20 >> 2) & 0xF);
        v36 = *(_QWORD *)(v21 + 16);
        v22 = *(_BYTE *)(v12 - 16);
        if ( (v22 & 2) != 0 )
          v23 = *(_QWORD *)(v12 - 32);
        else
          v23 = v16 - 8LL * ((v22 >> 2) & 0xF);
        v32 = v13;
        v37 = *(_QWORD *)(v23 + 24);
        v33 = v15;
        v38[0] = *(_DWORD *)(v12 + 4);
        v24 = sub_AF9890(&v34, &v35, &v36, &v37, v38);
        v25 = *j;
        v26 = (v32 - 1) & v24;
        v27 = (_QWORD *)(v33 + 8LL * v26);
        v28 = *v27;
        if ( *j != *v27 )
        {
          v29 = 1;
          v30 = 0;
          while ( v28 != -4096 )
          {
            if ( v28 != -8192 || v30 )
              v27 = v30;
            v26 = (v32 - 1) & (v29 + v26);
            v28 = *(_QWORD *)(v33 + 8LL * v26);
            if ( v25 == v28 )
            {
              v27 = (_QWORD *)(v33 + 8LL * v26);
              goto LABEL_29;
            }
            ++v29;
            v30 = v27;
            v27 = (_QWORD *)(v33 + 8LL * v26);
          }
          if ( v30 )
            v27 = v30;
        }
LABEL_29:
        *v27 = v25;
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
