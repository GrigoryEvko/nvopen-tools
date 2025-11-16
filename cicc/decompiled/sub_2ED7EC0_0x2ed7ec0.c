// Function: sub_2ED7EC0
// Address: 0x2ed7ec0
//
_QWORD *__fastcall sub_2ED7EC0(__int64 a1, _QWORD *a2)
{
  unsigned int v4; // esi
  __int64 v5; // rcx
  int v6; // r13d
  __int64 v7; // rdi
  _QWORD *v8; // r8
  unsigned int i; // eax
  _QWORD *v10; // r9
  __int64 v11; // r11
  unsigned int v12; // eax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  int v17; // edi
  __int64 v18; // rcx
  int v19; // edi
  _QWORD *v20; // r10
  __int64 v21; // rsi
  int v22; // r11d
  unsigned int j; // eax
  __int64 v24; // r9
  unsigned int v25; // eax
  int v26; // edi
  __int64 v27; // rcx
  int v28; // edi
  __int64 v29; // rsi
  int v30; // r11d
  unsigned int k; // eax
  __int64 v32; // r9
  unsigned int v33; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_23;
  }
  v5 = a2[1];
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
              | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; i = (v4 - 1) & v12 )
  {
    v10 = (_QWORD *)(v7 + 24LL * i);
    v11 = *v10;
    if ( *v10 == *a2 && v10[1] == v5 )
      return v10 + 2;
    if ( v11 == -4096 )
      break;
    if ( v11 == -8192 && v10[1] == -8192 && !v8 )
      v8 = (_QWORD *)(v7 + 24LL * i);
LABEL_9:
    v12 = v6 + i;
    ++v6;
  }
  if ( v10[1] != -4096 )
    goto LABEL_9;
  v14 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v4 )
  {
LABEL_23:
    sub_2ED7BF0(a1, 2 * v4);
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = a2[1];
      v19 = v17 - 1;
      v20 = 0;
      v22 = 1;
      for ( j = v19
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)
                  | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)))); ; j = v19 & v25 )
      {
        v21 = *(_QWORD *)(a1 + 8);
        v8 = (_QWORD *)(v21 + 24LL * j);
        v24 = *v8;
        if ( *v8 == *a2 && v8[1] == v18 )
          break;
        if ( v24 == -4096 )
        {
          if ( v8[1] == -4096 )
          {
LABEL_46:
            if ( v20 )
              v8 = v20;
            v15 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_17;
          }
        }
        else if ( v24 == -8192 && v8[1] == -8192 && !v20 )
        {
          v20 = (_QWORD *)(v21 + 24LL * j);
        }
        v25 = v22 + j;
        ++v22;
      }
      goto LABEL_42;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v15 <= v4 >> 3 )
  {
    sub_2ED7BF0(a1, v4);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = a2[1];
      v28 = v26 - 1;
      v20 = 0;
      v30 = 1;
      for ( k = v28
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4)
                  | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4)))); ; k = v28 & v33 )
      {
        v29 = *(_QWORD *)(a1 + 8);
        v8 = (_QWORD *)(v29 + 24LL * k);
        v32 = *v8;
        if ( *v8 == *a2 && v8[1] == v27 )
          break;
        if ( v32 == -4096 )
        {
          if ( v8[1] == -4096 )
            goto LABEL_46;
        }
        else if ( v32 == -8192 && v8[1] == -8192 && !v20 )
        {
          v20 = (_QWORD *)(v29 + 24LL * k);
        }
        v33 = v30 + k;
        ++v30;
      }
LABEL_42:
      v15 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_17;
    }
    goto LABEL_51;
  }
LABEL_17:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v8 != -4096 || v8[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v8 = *a2;
  v16 = a2[1];
  *((_BYTE *)v8 + 16) = 0;
  v8[1] = v16;
  return v8 + 2;
}
