// Function: sub_B0F0F0
// Address: 0xb0f0f0
//
_QWORD *__fastcall sub_B0F0F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __int64 *v5; // r15
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r14
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v12; // rax
  int v13; // r13d
  unsigned __int8 v14; // cl
  __int64 v15; // r11
  __int64 v16; // rdx
  __int64 *v17; // rsi
  unsigned __int8 v18; // cl
  __int64 v19; // rsi
  unsigned __int8 v20; // cl
  __int64 v21; // rsi
  unsigned __int8 v22; // cl
  __int64 v23; // rsi
  unsigned __int8 v24; // cl
  __int64 v25; // rdx
  int v26; // r13d
  int v27; // eax
  __int64 v28; // rcx
  unsigned int v29; // eax
  _QWORD *v30; // rdx
  __int64 v31; // rsi
  int v32; // r9d
  _QWORD *v33; // r8
  _QWORD *k; // rdx
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v36; // [rsp+10h] [rbp-80h]
  __int64 v37; // [rsp+20h] [rbp-70h] BYREF
  __int64 v38; // [rsp+28h] [rbp-68h] BYREF
  int v39; // [rsp+30h] [rbp-60h] BYREF
  __int64 v40; // [rsp+38h] [rbp-58h] BYREF
  __int64 v41; // [rsp+40h] [rbp-50h] BYREF
  int v42; // [rsp+48h] [rbp-48h] BYREF
  __int64 v43[8]; // [rsp+50h] [rbp-40h] BYREF

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
    v35 = 8 * v4;
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
        v37 = *v17;
        v18 = *(_BYTE *)(v12 - 16);
        if ( (v18 & 2) != 0 )
          v19 = *(_QWORD *)(v12 - 32);
        else
          v19 = v16 - 8LL * ((v18 >> 2) & 0xF);
        v38 = *(_QWORD *)(v19 + 8);
        v39 = *(_DWORD *)(v12 + 16);
        v20 = *(_BYTE *)(v12 - 16);
        if ( (v20 & 2) != 0 )
          v21 = *(_QWORD *)(v12 - 32);
        else
          v21 = v16 - 8LL * ((v20 >> 2) & 0xF);
        v40 = *(_QWORD *)(v21 + 16);
        v22 = *(_BYTE *)(v12 - 16);
        if ( (v22 & 2) != 0 )
          v23 = *(_QWORD *)(v12 - 32);
        else
          v23 = v16 - 8LL * ((v22 >> 2) & 0xF);
        v41 = *(_QWORD *)(v23 + 24);
        v42 = *(_DWORD *)(v12 + 20);
        v24 = *(_BYTE *)(v12 - 16);
        if ( (v24 & 2) != 0 )
          v25 = *(_QWORD *)(v12 - 32);
        else
          v25 = v16 - 8LL * ((v24 >> 2) & 0xF);
        v36 = v15;
        v26 = v13 - 1;
        v43[0] = *(_QWORD *)(v25 + 32);
        v27 = sub_AF9E80(&v37, &v38, &v39, &v40, &v41, &v42, v43);
        v28 = *j;
        v29 = v26 & v27;
        v30 = (_QWORD *)(v36 + 8LL * v29);
        v31 = *v30;
        if ( *j != *v30 )
        {
          v32 = 1;
          v33 = 0;
          while ( v31 != -4096 )
          {
            if ( v31 != -8192 || v33 )
              v30 = v33;
            v29 = v26 & (v32 + v29);
            v31 = *(_QWORD *)(v36 + 8LL * v29);
            if ( v28 == v31 )
            {
              v30 = (_QWORD *)(v36 + 8LL * v29);
              goto LABEL_31;
            }
            ++v32;
            v33 = v30;
            v30 = (_QWORD *)(v36 + 8LL * v29);
          }
          if ( v33 )
            v30 = v33;
        }
LABEL_31:
        *v30 = v28;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v35, 8);
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
