// Function: sub_B04210
// Address: 0xb04210
//
_QWORD *__fastcall sub_B04210(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r12
  __int64 *v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 *v9; // r12
  _QWORD *i; // rcx
  __int64 *j; // rbx
  __int64 v12; // rax
  int v13; // r15d
  unsigned __int8 v14; // cl
  __int64 *v15; // rsi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // ecx
  _QWORD *v20; // rsi
  __int64 v21; // r9
  int v22; // r11d
  _QWORD *v23; // r10
  __int64 v24; // rdx
  _QWORD *k; // rdx
  __int64 v26; // [rsp+8h] [rbp-78h]
  int v27; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  __int64 v29; // [rsp+28h] [rbp-58h]
  __int64 v30; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-48h]
  __int64 v32; // [rsp+40h] [rbp-40h] BYREF
  bool v33; // [rsp+48h] [rbp-38h]

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
    v28 = 8 * v4;
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
        v29 = *(_QWORD *)(a1 + 8);
        v31 = *(_DWORD *)(v12 + 24);
        if ( v31 > 0x40 )
        {
          v26 = v12;
          sub_C43780(&v30, v12 + 16);
          v12 = v26;
        }
        else
        {
          v30 = *(_QWORD *)(v12 + 16);
        }
        v14 = *(_BYTE *)(v12 - 16);
        if ( (v14 & 2) != 0 )
          v15 = *(__int64 **)(v12 - 32);
        else
          v15 = (__int64 *)(v12 - 16 - 8LL * ((v14 >> 2) & 0xF));
        v32 = *v15;
        v33 = *(_DWORD *)(v12 + 4) != 0;
        v16 = sub_AFB7E0((__int64)&v30, &v32);
        v17 = v16;
        if ( v31 > 0x40 && v30 )
        {
          v27 = v16;
          j_j___libc_free_0_0(v30);
          v17 = v27;
        }
        v18 = *j;
        v19 = (v13 - 1) & v17;
        v20 = (_QWORD *)(v29 + 8LL * v19);
        v21 = *v20;
        if ( *j != *v20 )
        {
          v22 = 1;
          v23 = 0;
          while ( v21 != -4096 )
          {
            if ( v21 != -8192 || v23 )
              v20 = v23;
            v19 = (v13 - 1) & (v22 + v19);
            v21 = *(_QWORD *)(v29 + 8LL * v19);
            if ( v18 == v21 )
            {
              v20 = (_QWORD *)(v29 + 8LL * v19);
              goto LABEL_28;
            }
            ++v22;
            v23 = v20;
            v20 = (_QWORD *)(v29 + 8LL * v19);
          }
          if ( v23 )
            v20 = v23;
        }
LABEL_28:
        *v20 = v18;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v24]; k != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
