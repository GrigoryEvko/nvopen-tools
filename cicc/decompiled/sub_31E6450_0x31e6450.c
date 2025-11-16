// Function: sub_31E6450
// Address: 0x31e6450
//
__int64 __fastcall sub_31E6450(__int64 a1, int *a2)
{
  int v4; // r13d
  int v5; // r14d
  unsigned int v6; // esi
  __int64 v7; // rcx
  _DWORD *v8; // r15
  int v9; // r10d
  __int64 v10; // r9
  __int64 i; // r8
  _DWORD *v12; // rdi
  int v13; // r8d
  __int64 v14; // rax
  int v16; // edx
  int v17; // edx
  int v18; // edx
  __int64 v19; // rcx
  _DWORD *v20; // rdi
  unsigned int j; // eax
  int v22; // eax
  int v23; // ecx
  int v24; // ecx
  __int64 v25; // rax
  __int64 v26; // rax
  int v27; // edx
  int v28; // edx
  __int64 v29; // rsi
  unsigned int k; // eax
  int v31; // eax
  int v32; // ecx
  int v33; // esi
  __int64 v34; // [rsp+14h] [rbp-3Ch]

  v4 = *a2;
  v5 = a2[1];
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_15;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  v9 = 1;
  v10 = v6 - 1;
  for ( i = ((unsigned int)((0xBF58476D1CE4E5B9LL
                           * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))) >> 31)
           ^ (756364221 * v5))
          & (v6 - 1); ; i = (unsigned int)v10 & v13 )
  {
    v12 = (_DWORD *)(v7 + 12LL * (unsigned int)i);
    if ( v4 == *v12 && v5 == v12[1] )
    {
      v14 = (unsigned int)v12[2];
      return *(_QWORD *)(a1 + 32) + 12 * v14 + 8;
    }
    if ( !*v12 )
      break;
LABEL_5:
    v13 = v9 + i;
    ++v9;
  }
  v16 = v12[1];
  if ( v16 != -1 )
  {
    if ( v16 == -2 && !v8 )
      v8 = (_DWORD *)(v7 + 12LL * (unsigned int)i);
    goto LABEL_5;
  }
  v23 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
    v8 = v12;
  ++*(_QWORD *)a1;
  v24 = v23 + 1;
  if ( 4 * v24 >= 3 * v6 )
  {
LABEL_15:
    sub_31E5EB0(a1, 2 * v6);
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      i = 1;
      v20 = 0;
      for ( j = v18
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))) >> 31)
               ^ (756364221 * v5)); ; j = v18 & v22 )
      {
        v19 = *(_QWORD *)(a1 + 8);
        v8 = (_DWORD *)(v19 + 12LL * j);
        if ( v4 == *v8 && v5 == v8[1] )
          break;
        if ( !*v8 )
        {
          v33 = v8[1];
          if ( v33 == -1 )
          {
LABEL_50:
            if ( v20 )
              v8 = v20;
            v24 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_24;
          }
          if ( v33 == -2 && !v20 )
            v20 = (_DWORD *)(v19 + 12LL * j);
        }
        v22 = i + j;
        i = (unsigned int)(i + 1);
      }
      goto LABEL_37;
    }
LABEL_53:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v24 <= v6 >> 3 )
  {
    sub_31E5EB0(a1, v6);
    v27 = *(_DWORD *)(a1 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v20 = 0;
      i = 1;
      for ( k = v28
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))) >> 31)
               ^ (756364221 * v5)); ; k = v28 & v31 )
      {
        v29 = *(_QWORD *)(a1 + 8);
        v8 = (_DWORD *)(v29 + 12LL * k);
        if ( v4 == *v8 && v5 == v8[1] )
          break;
        if ( !*v8 )
        {
          v32 = v8[1];
          if ( v32 == -1 )
            goto LABEL_50;
          if ( v32 == -2 && !v20 )
            v20 = (_DWORD *)(v29 + 12LL * k);
        }
        v31 = i + k;
        i = (unsigned int)(i + 1);
      }
LABEL_37:
      v24 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_24;
    }
    goto LABEL_53;
  }
LABEL_24:
  *(_DWORD *)(a1 + 16) = v24;
  if ( *v8 || v8[1] != -1 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  v8[1] = v5;
  v8[2] = 0;
  v34 = *(_QWORD *)a2;
  v25 = *(unsigned int *)(a1 + 40);
  if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v25 + 1, 0xCu, i, v10);
    v25 = *(unsigned int *)(a1 + 40);
  }
  v26 = *(_QWORD *)(a1 + 32) + 12 * v25;
  *(_QWORD *)v26 = v34;
  *(_DWORD *)(v26 + 8) = 0;
  v14 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v14 + 1;
  v8[2] = v14;
  return *(_QWORD *)(a1 + 32) + 12 * v14 + 8;
}
