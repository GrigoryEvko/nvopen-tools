// Function: sub_2D25ED0
// Address: 0x2d25ed0
//
_QWORD *__fastcall sub_2D25ED0(__int64 a1, int a2)
{
  __int64 v3; // r12
  __int64 v4; // r14
  unsigned int v5; // edi
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 v8; // r12
  _QWORD *i; // rdx
  __int64 v10; // r15
  __int64 v11; // rdx
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // rcx
  int v17; // eax
  __int64 v18; // rdi
  int v19; // r10d
  unsigned int v20; // edx
  __int64 v21; // r14
  int v22; // edx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rcx
  _QWORD *j; // rdx
  __int64 v28; // [rsp+0h] [rbp-40h]
  __int64 v29; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(272LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v29 = 272 * v3;
    v8 = v4 + 272 * v3;
    for ( i = &result[34 * v7]; i != result; result += 34 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v8 != v4 )
    {
      v28 = v4;
      v10 = v4;
      do
      {
        v16 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 != -8192 && v16 != -4096 )
        {
          v17 = *(_DWORD *)(a1 + 24);
          if ( !v17 )
          {
            MEMORY[0] = *(_QWORD *)v10;
            BUG();
          }
          v18 = *(_QWORD *)(a1 + 8);
          v19 = 1;
          v25 = 0;
          v20 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v21 = v18 + 272LL * v20;
          v24 = *(_QWORD *)v21;
          if ( v16 != *(_QWORD *)v21 )
          {
            while ( v24 != -4096 )
            {
              if ( !v25 && v24 == -8192 )
                v25 = v21;
              v20 = (v17 - 1) & (v19 + v20);
              v21 = v18 + 272LL * v20;
              v24 = *(_QWORD *)v21;
              if ( v16 == *(_QWORD *)v21 )
                goto LABEL_27;
              ++v19;
            }
            if ( v25 )
              v21 = v25;
          }
LABEL_27:
          *(_QWORD *)v21 = v16;
          *(_QWORD *)(v21 + 8) = v21 + 24;
          *(_QWORD *)(v21 + 16) = 0x600000000LL;
          if ( *(_DWORD *)(v10 + 16) )
            sub_2D23900(v21 + 8, (char **)(v10 + 8), v21 + 24, v16, v24, v25);
          v22 = *(_DWORD *)(v10 + 72);
          *(_QWORD *)(v21 + 88) = 0x200000000LL;
          *(_DWORD *)(v21 + 72) = v22;
          *(_QWORD *)(v21 + 80) = v21 + 96;
          v23 = *(unsigned int *)(v10 + 88);
          if ( (_DWORD)v23 )
            sub_2D235D0(v21 + 80, (char **)(v10 + 80), v21 + 96, v23, v24, v25);
          *(_QWORD *)(v21 + 152) = 0x200000000LL;
          *(_QWORD *)(v21 + 144) = v21 + 160;
          v11 = *(unsigned int *)(v10 + 152);
          if ( (_DWORD)v11 )
            sub_2D235D0(v21 + 144, (char **)(v10 + 144), v11, v23, v24, v25);
          *(_QWORD *)(v21 + 208) = v21 + 224;
          *(_QWORD *)(v21 + 216) = 0xC00000000LL;
          if ( *(_DWORD *)(v10 + 216) )
            sub_2D23470(v21 + 208, (char **)(v10 + 208), v21 + 224, v23, v24, v25);
          ++*(_DWORD *)(a1 + 16);
          v12 = *(_QWORD *)(v10 + 208);
          if ( v12 != v10 + 224 )
            _libc_free(v12);
          v13 = *(_QWORD *)(v10 + 144);
          if ( v13 != v10 + 160 )
            _libc_free(v13);
          v14 = *(_QWORD *)(v10 + 80);
          if ( v14 != v10 + 96 )
            _libc_free(v14);
          v15 = *(_QWORD *)(v10 + 8);
          if ( v15 != v10 + 24 )
            _libc_free(v15);
        }
        v10 += 272;
      }
      while ( v8 != v10 );
      v4 = v28;
    }
    return (_QWORD *)sub_C7D6A0(v4, v29, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[34 * v26]; j != result; result += 34 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
