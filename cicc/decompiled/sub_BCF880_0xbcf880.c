// Function: sub_BCF880
// Address: 0xbcf880
//
_QWORD *__fastcall sub_BCF880(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 *v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  _QWORD *v10; // rdx
  __int64 *i; // r13
  __int64 *j; // rbx
  __int64 v13; // rax
  __int64 v14; // r11
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rdx
  _QWORD *v18; // rdi
  __int64 v19; // rcx
  int v20; // eax
  __int64 v21; // rdi
  _QWORD *v22; // r11
  int v23; // r8d
  unsigned int v24; // edx
  _QWORD *v25; // rsi
  __int64 v26; // r10
  _QWORD *k; // rdx
  __int64 v28; // [rsp+10h] [rbp-90h]
  _QWORD *v29; // [rsp+18h] [rbp-88h]
  __int64 v30; // [rsp+20h] [rbp-80h]
  int v31; // [rsp+2Ch] [rbp-74h]
  unsigned __int64 v32; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v33; // [rsp+38h] [rbp-68h] BYREF
  __int64 v34[5]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v35; // [rsp+68h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
    v9 = v4;
    v10 = &result[v8];
    for ( i = &v5[v9]; v10 != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; i != j; ++j )
    {
      v13 = *j;
      if ( *j != -8192 && v13 != -4096 )
      {
        v31 = *(_DWORD *)(a1 + 24);
        if ( !v31 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v14 = *(_QWORD *)(v13 + 16);
        v15 = *(_QWORD *)(a1 + 8);
        v16 = *(_QWORD *)(v13 + 24);
        v34[1] = *(_QWORD *)(v13 + 32);
        v17 = *(unsigned int *)(v13 + 12);
        v28 = v15;
        v18 = *(_QWORD **)(v13 + 40);
        v19 = 8 * v17;
        v34[2] = v14;
        v34[3] = v17;
        LODWORD(v17) = *(_DWORD *)(v13 + 8);
        v34[0] = v16;
        v30 = v19;
        v35 = (unsigned int)v17 >> 8;
        v29 = (_QWORD *)v14;
        v34[4] = (__int64)v18;
        v33 = sub_939680(v18, (__int64)v18 + 4 * v35);
        v32 = sub_BCC330(v29, (__int64)v29 + v30);
        v20 = sub_BCC270(v34, &v32, &v33);
        v21 = *j;
        v22 = 0;
        v23 = 1;
        v24 = (v31 - 1) & v20;
        v25 = (_QWORD *)(v28 + 8LL * v24);
        v26 = *v25;
        if ( *v25 != *j )
        {
          while ( v26 != -4096 )
          {
            if ( !v22 && v26 == -8192 )
              v22 = v25;
            v24 = (v31 - 1) & (v23 + v24);
            v25 = (_QWORD *)(v28 + 8LL * v24);
            v26 = *v25;
            if ( *v25 == v21 )
              goto LABEL_13;
            ++v23;
          }
          if ( v22 )
            v25 = v22;
        }
LABEL_13:
        *v25 = v21;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v9 * 8, 8);
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
