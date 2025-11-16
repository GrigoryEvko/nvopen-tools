// Function: sub_B01330
// Address: 0xb01330
//
_QWORD *__fastcall sub_B01330(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 *v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r15
  _QWORD *i; // rdx
  __int64 *v11; // rbx
  __int64 v12; // rax
  int v13; // r10d
  __int64 v14; // r9
  __int64 v15; // rcx
  unsigned __int8 v16; // dl
  unsigned __int8 v17; // dl
  __int64 v18; // rdx
  int v19; // edx
  __int64 v20; // rsi
  unsigned int v21; // edx
  _QWORD *v22; // rcx
  __int64 v23; // r8
  int v24; // r11d
  _QWORD *v25; // r10
  __int64 v26; // rcx
  _QWORD *j; // rdx
  int v28; // [rsp+14h] [rbp-5Ch]
  __int64 v29; // [rsp+18h] [rbp-58h]
  int v30; // [rsp+20h] [rbp-50h] BYREF
  int v31; // [rsp+24h] [rbp-4Ch] BYREF
  __int64 v32; // [rsp+28h] [rbp-48h] BYREF
  __int64 v33; // [rsp+30h] [rbp-40h] BYREF
  __int8 v34[56]; // [rsp+38h] [rbp-38h] BYREF

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
    v11 = v5;
    if ( v9 != v5 )
    {
      while ( 1 )
      {
        v12 = *v11;
        if ( *v11 != -8192 && v12 != -4096 )
          break;
LABEL_25:
        if ( v9 == ++v11 )
          return (_QWORD *)sub_C7D6A0(v5, 8 * v4, 8);
      }
      v13 = *(_DWORD *)(a1 + 24);
      if ( !v13 )
      {
        MEMORY[0] = *v11;
        BUG();
      }
      v14 = *(_QWORD *)(a1 + 8);
      v15 = v12 - 16;
      v30 = *(_DWORD *)(v12 + 4);
      v31 = *(unsigned __int16 *)(v12 + 2);
      v16 = *(_BYTE *)(v12 - 16);
      if ( (v16 & 2) != 0 )
      {
        v32 = **(_QWORD **)(v12 - 32);
        v17 = *(_BYTE *)(v12 - 16);
        if ( (v17 & 2) != 0 )
        {
LABEL_14:
          v18 = 0;
          if ( *(_DWORD *)(v12 - 24) != 2 )
          {
LABEL_15:
            v33 = v18;
            v28 = v13;
            v29 = v14;
            v34[0] = *(_BYTE *)(v12 + 1) >> 7;
            v19 = sub_AF71E0(&v30, &v31, &v32, &v33, v34);
            v20 = *v11;
            v21 = (v28 - 1) & v19;
            v22 = (_QWORD *)(v29 + 8LL * v21);
            v23 = *v22;
            if ( *v22 != *v11 )
            {
              v24 = 1;
              v25 = 0;
              while ( v23 != -4096 )
              {
                if ( v23 != -8192 || v25 )
                  v22 = v25;
                v21 = (v28 - 1) & (v24 + v21);
                v23 = *(_QWORD *)(v29 + 8LL * v21);
                if ( v20 == v23 )
                {
                  v22 = (_QWORD *)(v29 + 8LL * v21);
                  goto LABEL_24;
                }
                ++v24;
                v25 = v22;
                v22 = (_QWORD *)(v29 + 8LL * v21);
              }
              if ( v25 )
                v22 = v25;
            }
LABEL_24:
            *v22 = v20;
            ++*(_DWORD *)(a1 + 16);
            goto LABEL_25;
          }
          v26 = *(_QWORD *)(v12 - 32);
LABEL_30:
          v18 = *(_QWORD *)(v26 + 8);
          goto LABEL_15;
        }
      }
      else
      {
        v32 = *(_QWORD *)(v15 - 8LL * ((v16 >> 2) & 0xF));
        v17 = *(_BYTE *)(v12 - 16);
        if ( (v17 & 2) != 0 )
          goto LABEL_14;
      }
      if ( ((*(_WORD *)(v12 - 16) >> 6) & 0xF) != 2 )
      {
        v18 = 0;
        goto LABEL_15;
      }
      v26 = v15 - 8LL * ((v17 >> 2) & 0xF);
      goto LABEL_30;
    }
    return (_QWORD *)sub_C7D6A0(v5, 8 * v4, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[*(unsigned int *)(a1 + 24)]; j != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
