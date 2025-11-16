// Function: sub_104FA70
// Address: 0x104fa70
//
_QWORD *__fastcall sub_104FA70(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r13
  _QWORD *i; // rdx
  __int64 j; // r15
  int v11; // ecx
  __int64 v12; // rdx
  int v13; // ecx
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rcx
  int v19; // esi
  int v20; // esi
  int v21; // r11d
  __int64 v22; // r10
  unsigned int v23; // edi
  __int64 v24; // rdx
  __int64 v25; // r14
  int v26; // ecx
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  _QWORD *k; // rdx
  __int64 v33; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v33 = v5;
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
  result = (_QWORD *)sub_C7D670(296LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 296 * v4 + v5;
    for ( i = &result[37 * *(unsigned int *)(a1 + 24)]; i != result; result += 37 )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v8 != j; j += 296 )
    {
      v18 = *(_QWORD *)j;
      if ( *(_QWORD *)j != -8192 && v18 != -4096 )
      {
        v19 = *(_DWORD *)(a1 + 24);
        if ( !v19 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v20 = v19 - 1;
        v30 = *(_QWORD *)(a1 + 8);
        v21 = 1;
        v22 = 0;
        v23 = v20 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v24 = 37LL * v23;
        v25 = v30 + 296LL * v23;
        v29 = *(_QWORD *)v25;
        if ( v18 != *(_QWORD *)v25 )
        {
          while ( v29 != -4096 )
          {
            if ( v29 == -8192 && !v22 )
              v22 = v25;
            v23 = v20 & (v21 + v23);
            v24 = 37LL * v23;
            v25 = v30 + 296LL * v23;
            v29 = *(_QWORD *)v25;
            if ( v18 == *(_QWORD *)v25 )
              goto LABEL_27;
            ++v21;
          }
          if ( v22 )
            v25 = v22;
        }
LABEL_27:
        *(_QWORD *)v25 = v18;
        *(_QWORD *)(v25 + 8) = v25 + 24;
        *(_QWORD *)(v25 + 16) = 0x600000000LL;
        v28 = *(unsigned int *)(j + 16);
        if ( (_DWORD)v28 )
        {
          v28 = j + 8;
          sub_104D0F0(v25 + 8, (char **)(j + 8), v24, v25 + 24, v29, v30);
        }
        v26 = *(_DWORD *)(j + 72);
        *(_QWORD *)(v25 + 88) = 0x600000000LL;
        *(_DWORD *)(v25 + 72) = v26;
        *(_QWORD *)(v25 + 80) = v25 + 96;
        v27 = *(unsigned int *)(j + 88);
        if ( (_DWORD)v27 )
        {
          v28 = j + 80;
          sub_104D0F0(v25 + 80, (char **)(j + 80), v24, v27, v29, v30);
        }
        v11 = *(_DWORD *)(j + 144);
        *(_QWORD *)(v25 + 160) = 0x600000000LL;
        *(_DWORD *)(v25 + 144) = v11;
        *(_QWORD *)(v25 + 152) = v25 + 168;
        v12 = *(unsigned int *)(j + 160);
        if ( (_DWORD)v12 )
        {
          v28 = j + 152;
          sub_104D0F0(v25 + 152, (char **)(j + 152), v12, v25 + 168, v29, v30);
        }
        v13 = *(_DWORD *)(j + 216);
        *(_QWORD *)(v25 + 232) = 0x600000000LL;
        *(_DWORD *)(v25 + 216) = v13;
        *(_QWORD *)(v25 + 224) = v25 + 240;
        if ( *(_DWORD *)(j + 232) )
        {
          v28 = j + 224;
          sub_104D0F0(v25 + 224, (char **)(j + 224), v12, v25 + 240, v29, v30);
        }
        *(_DWORD *)(v25 + 288) = *(_DWORD *)(j + 288);
        ++*(_DWORD *)(a1 + 16);
        v14 = *(_QWORD *)(j + 224);
        if ( v14 != j + 240 )
          _libc_free(v14, v28);
        v15 = *(_QWORD *)(j + 152);
        if ( v15 != j + 168 )
          _libc_free(v15, v28);
        v16 = *(_QWORD *)(j + 80);
        if ( v16 != j + 96 )
          _libc_free(v16, v28);
        v17 = *(_QWORD *)(j + 8);
        if ( v17 != j + 24 )
          _libc_free(v17, v28);
      }
    }
    return (_QWORD *)sub_C7D6A0(v33, 296 * v4, 8);
  }
  else
  {
    v31 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[37 * v31]; k != result; result += 37 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
