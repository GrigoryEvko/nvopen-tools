// Function: sub_2C67750
// Address: 0x2c67750
//
char *__fastcall sub_2C67750(char *a1, char *a2, char *a3, __int64 a4, __int64 a5, _DWORD *a6, __int64 a7)
{
  __int64 v7; // rbx
  __int64 v8; // r10
  __int64 v9; // rax
  _DWORD *v10; // r8
  char *v11; // rcx
  int v12; // r14d
  __int64 v13; // r10
  __int64 v14; // r8
  __int64 v15; // r11
  char *v16; // rsi
  char *v17; // rdx
  char *v18; // rax
  __int64 v19; // rdx
  int v20; // ecx
  __int64 v21; // rax
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // r10
  __int64 v26; // r8
  _DWORD *v27; // rcx
  char *v28; // rax
  int v29; // r11d
  __int64 v30; // r10
  int v31; // eax
  __int64 v32; // rsi
  __int64 v33; // rcx
  _DWORD *v34; // r9
  char *v35; // rax

  if ( a4 > a5 && a5 <= a7 )
  {
    if ( !a5 )
      return a1;
    v7 = a2 - a1;
    v8 = (a3 - a2) >> 3;
    v9 = (a2 - a1) >> 3;
    if ( a3 - a2 <= 0 )
    {
      if ( v7 <= 0 )
        return a1;
      v14 = 0;
      v13 = 0;
    }
    else
    {
      v10 = a6;
      v11 = a2;
      do
      {
        v12 = *(_DWORD *)v11;
        v10 += 2;
        v11 += 8;
        *(v10 - 2) = v12;
        *(v10 - 1) = *((_DWORD *)v11 - 1);
        --v8;
      }
      while ( v8 );
      v13 = 8;
      if ( a3 - a2 > 0 )
        v13 = a3 - a2;
      v14 = v13 >> 3;
      if ( v7 <= 0 )
      {
LABEL_12:
        if ( v13 > 0 )
        {
          v18 = a1;
          v19 = v14;
          do
          {
            v20 = *a6;
            v18 += 8;
            a6 += 2;
            *((_DWORD *)v18 - 2) = v20;
            *((_DWORD *)v18 - 1) = *(a6 - 1);
            --v19;
          }
          while ( v19 );
          v21 = 8 * v14;
          if ( v14 <= 0 )
            v21 = 8;
          return &a1[v21];
        }
        return a1;
      }
    }
    v15 = -8 * ((a2 - a1) >> 3);
    v16 = &a2[v15];
    v17 = &a3[v15];
    do
    {
      *(_DWORD *)&v17[8 * v9 - 8] = *(_DWORD *)&v16[8 * v9 - 8];
      *(_DWORD *)&v17[8 * v9 - 4] = *(_DWORD *)&v16[8 * v9 - 4];
      --v9;
    }
    while ( v9 );
    goto LABEL_12;
  }
  if ( a4 > a7 )
    return sub_2C4CF50(a1, a2, a3);
  if ( !a4 )
    return a3;
  v23 = a2 - a1;
  v24 = a3 - a2;
  v25 = (a2 - a1) >> 3;
  v26 = (a3 - a2) >> 3;
  if ( a2 - a1 <= 0 )
  {
    if ( v24 <= 0 )
      return a3;
    v30 = 0;
    v23 = 0;
    goto LABEL_26;
  }
  v27 = a6;
  v28 = a1;
  do
  {
    v29 = *(_DWORD *)v28;
    v27 += 2;
    v28 += 8;
    *(v27 - 2) = v29;
    *(v27 - 1) = *((_DWORD *)v28 - 1);
    --v25;
  }
  while ( v25 );
  if ( v23 <= 0 )
    v23 = 8;
  a6 = (_DWORD *)((char *)a6 + v23);
  v30 = v23 >> 3;
  if ( v24 > 0 )
  {
    do
    {
LABEL_26:
      v31 = *(_DWORD *)a2;
      a1 += 8;
      a2 += 8;
      *((_DWORD *)a1 - 2) = v31;
      *((_DWORD *)a1 - 1) = *((_DWORD *)a2 - 1);
      --v26;
    }
    while ( v26 );
  }
  if ( v23 <= 0 )
    return a3;
  v32 = v30;
  v33 = -8 * v30;
  v34 = &a6[-2 * v30];
  v35 = &a3[-8 * v30];
  do
  {
    *(_DWORD *)&v35[8 * v32 - 8] = v34[2 * v32 - 2];
    *(_DWORD *)&v35[8 * v32 - 4] = v34[2 * v32 - 1];
    --v32;
  }
  while ( v32 );
  if ( v30 <= 0 )
    v33 = -8;
  return &a3[v33];
}
