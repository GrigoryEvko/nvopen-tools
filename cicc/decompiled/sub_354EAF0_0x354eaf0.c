// Function: sub_354EAF0
// Address: 0x354eaf0
//
_QWORD *__fastcall sub_354EAF0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  _QWORD *i; // rdx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rdx
  int v15; // esi
  int v16; // esi
  __int64 v17; // r9
  int v18; // r11d
  _QWORD *v19; // r10
  __int64 v20; // r8
  _QWORD *v21; // r14
  __int64 v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rdx
  _QWORD *j; // rdx
  __int64 v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v28 = v4;
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
  result = (_QWORD *)sub_C7D670(632LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v27 = 632 * v5;
    v9 = 632 * v5 + v4;
    for ( i = &result[79 * v8]; i != result; result += 79 )
    {
      if ( result )
        *result = -4096;
    }
    for ( ; v9 != v4; v4 += 632 )
    {
      v14 = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 != -8192 && v14 != -4096 )
      {
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
        {
          MEMORY[0] = *(_QWORD *)v4;
          BUG();
        }
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 1;
        v19 = 0;
        v20 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v21 = (_QWORD *)(v17 + 632 * v20);
        v22 = *v21;
        if ( v14 != *v21 )
        {
          while ( v22 != -4096 )
          {
            if ( v22 == -8192 && !v19 )
              v19 = v21;
            v20 = v16 & (unsigned int)(v18 + v20);
            v21 = (_QWORD *)(v17 + 632LL * (unsigned int)v20);
            v22 = *v21;
            if ( v14 == *v21 )
              goto LABEL_23;
            ++v18;
          }
          if ( v19 )
            v21 = v19;
        }
LABEL_23:
        *v21 = v14;
        v21[1] = v21 + 3;
        v21[2] = 0x800000000LL;
        v23 = *(unsigned int *)(v4 + 16);
        if ( (_DWORD)v23 )
          sub_353D820((__int64)(v21 + 1), v4 + 8, (__int64)(v21 + 3), v23, v20, v17);
        v21[28] = 0x800000000LL;
        v21[27] = v21 + 29;
        v24 = *(unsigned int *)(v4 + 224);
        if ( (_DWORD)v24 )
          sub_353D820((__int64)(v21 + 27), v4 + 216, v24, v23, v20, v17);
        v21[54] = 0x800000000LL;
        v21[53] = v21 + 55;
        if ( *(_DWORD *)(v4 + 432) )
          sub_353D820((__int64)(v21 + 53), v4 + 424, (__int64)(v21 + 55), v23, v20, v17);
        ++*(_DWORD *)(a1 + 16);
        v11 = *(_QWORD *)(v4 + 424);
        if ( v11 != v4 + 440 )
          _libc_free(v11);
        v12 = *(_QWORD *)(v4 + 216);
        if ( v12 != v4 + 232 )
          _libc_free(v12);
        v13 = *(_QWORD *)(v4 + 8);
        if ( v13 != v4 + 24 )
          _libc_free(v13);
      }
    }
    return (_QWORD *)sub_C7D6A0(v28, v27, 8);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[79 * v25]; j != result; result += 79 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
