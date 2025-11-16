// Function: sub_B0F820
// Address: 0xb0f820
//
_QWORD *__fastcall sub_B0F820(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 *v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r12
  _QWORD *i; // rdx
  __int64 *j; // rbx
  int v12; // r15d
  unsigned __int8 v13; // cl
  __int64 v14; // rax
  __int64 *v15; // rsi
  unsigned __int8 v16; // cl
  __int64 v17; // rsi
  unsigned __int8 v18; // cl
  __int64 v19; // rsi
  unsigned __int8 v20; // cl
  __int64 v21; // rsi
  unsigned __int8 v22; // cl
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rcx
  unsigned int v26; // eax
  __int64 *v27; // rdx
  __int64 v28; // rsi
  int v29; // r8d
  __int64 *v30; // rdi
  __int64 v31; // rdx
  _QWORD *k; // rdx
  __int64 v33; // [rsp+0h] [rbp-90h]
  __int64 v34; // [rsp+10h] [rbp-80h]
  __int64 v35; // [rsp+18h] [rbp-78h]
  int v36; // [rsp+20h] [rbp-70h] BYREF
  __int64 v37; // [rsp+28h] [rbp-68h] BYREF
  __int64 v38; // [rsp+30h] [rbp-60h] BYREF
  __int64 v39; // [rsp+38h] [rbp-58h] BYREF
  int v40; // [rsp+40h] [rbp-50h] BYREF
  __int64 v41; // [rsp+48h] [rbp-48h] BYREF
  __int64 v42[8]; // [rsp+50h] [rbp-40h] BYREF

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
    v33 = 8 * v4;
    v9 = &v5[v4];
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v9 != j; ++j )
    {
      if ( *j != -8192 && *j != -4096 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v34 = *j;
        v35 = *(_QWORD *)(a1 + 8);
        v36 = (unsigned __int16)sub_AF18C0(*j);
        v13 = *(_BYTE *)(v34 - 16);
        v14 = v34 - 16;
        if ( (v13 & 2) != 0 )
          v15 = *(__int64 **)(v34 - 32);
        else
          v15 = (__int64 *)(v14 - 8LL * ((v13 >> 2) & 0xF));
        v37 = *v15;
        v16 = *(_BYTE *)(v34 - 16);
        if ( (v16 & 2) != 0 )
          v17 = *(_QWORD *)(v34 - 32);
        else
          v17 = v14 - 8LL * ((v16 >> 2) & 0xF);
        v38 = *(_QWORD *)(v17 + 8);
        v18 = *(_BYTE *)(v34 - 16);
        if ( (v18 & 2) != 0 )
          v19 = *(_QWORD *)(v34 - 32);
        else
          v19 = v14 - 8LL * ((v18 >> 2) & 0xF);
        v39 = *(_QWORD *)(v19 + 24);
        v40 = *(_DWORD *)(v34 + 4);
        v20 = *(_BYTE *)(v34 - 16);
        if ( (v20 & 2) != 0 )
          v21 = *(_QWORD *)(v34 - 32);
        else
          v21 = v14 - 8LL * ((v20 >> 2) & 0xF);
        v41 = *(_QWORD *)(v21 + 16);
        v22 = *(_BYTE *)(v34 - 16);
        if ( (v22 & 2) != 0 )
          v23 = *(_QWORD *)(v34 - 32);
        else
          v23 = v14 - 8LL * ((v22 >> 2) & 0xF);
        v42[0] = *(_QWORD *)(v23 + 32);
        v24 = sub_AFB320(&v36, &v37, &v38, &v39, &v40, &v41, v42);
        v25 = *j;
        v26 = (v12 - 1) & v24;
        v27 = (__int64 *)(v35 + 8LL * v26);
        v28 = *v27;
        if ( *j != *v27 )
        {
          v29 = 1;
          v30 = 0;
          while ( v28 != -4096 )
          {
            if ( v28 != -8192 || v30 )
              v27 = v30;
            v26 = (v12 - 1) & (v29 + v26);
            v28 = *(_QWORD *)(v35 + 8LL * v26);
            if ( v25 == v28 )
            {
              v27 = (__int64 *)(v35 + 8LL * v26);
              goto LABEL_31;
            }
            ++v29;
            v30 = v27;
            v27 = (__int64 *)(v35 + 8LL * v26);
          }
          if ( v30 )
            v27 = v30;
        }
LABEL_31:
        *v27 = v25;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v33, 8);
  }
  else
  {
    v31 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v31]; k != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
