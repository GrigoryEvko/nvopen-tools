// Function: sub_2F33020
// Address: 0x2f33020
//
int *__fastcall sub_2F33020(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  int v6; // r13d
  int v7; // edi
  int *v8; // rdx
  unsigned int i; // eax
  int *v10; // r9
  int v11; // r11d
  unsigned int v12; // eax
  int v14; // eax
  int v15; // ecx
  int v16; // eax
  int v17; // edx
  int v18; // esi
  int v19; // r9d
  int *v20; // r10
  __int64 v21; // rdi
  int v22; // r11d
  unsigned int j; // eax
  int v24; // r8d
  unsigned int v25; // eax
  int v26; // edx
  int v27; // esi
  int v28; // r9d
  __int64 v29; // rdi
  int v30; // r11d
  unsigned int k; // eax
  int v32; // r8d
  unsigned int v33; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_23;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = a2[1];
  v8 = 0;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v7) | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
           ^ (756364221 * v7)); ; i = (v4 - 1) & v12 )
  {
    v10 = (int *)(v5 + 12LL * i);
    v11 = *v10;
    if ( *a2 == *v10 && v7 == v10[1] )
      return v10 + 2;
    if ( v11 == -1 )
      break;
    if ( v11 == -2 && v10[1] == -2 && !v8 )
      v8 = (int *)(v5 + 12LL * i);
LABEL_9:
    v12 = v6 + i;
    ++v6;
  }
  if ( v10[1] != -1 )
    goto LABEL_9;
  v14 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v4 )
  {
LABEL_23:
    sub_2F32D80(a1, 2 * v4);
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = a2[1];
      v19 = v17 - 1;
      v20 = 0;
      v22 = 1;
      for ( j = (v17 - 1)
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v18) | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
               ^ (756364221 * v18)); ; j = v19 & v25 )
      {
        v21 = *(_QWORD *)(a1 + 8);
        v8 = (int *)(v21 + 12LL * j);
        v24 = *v8;
        if ( *a2 == *v8 && v18 == v8[1] )
          break;
        if ( v24 == -1 )
        {
          if ( v8[1] == -1 )
          {
LABEL_46:
            if ( v20 )
              v8 = v20;
            v15 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_17;
          }
        }
        else if ( v24 == -2 && v8[1] == -2 && !v20 )
        {
          v20 = (int *)(v21 + 12LL * j);
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
    sub_2F32D80(a1, v4);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = a2[1];
      v28 = v26 - 1;
      v20 = 0;
      v30 = 1;
      for ( k = (v26 - 1)
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v27) | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
               ^ (756364221 * v27)); ; k = v28 & v33 )
      {
        v29 = *(_QWORD *)(a1 + 8);
        v8 = (int *)(v29 + 12LL * k);
        v32 = *v8;
        if ( *a2 == *v8 && v27 == v8[1] )
          break;
        if ( v32 == -1 )
        {
          if ( v8[1] == -1 )
            goto LABEL_46;
        }
        else if ( v32 == -2 && v8[1] == -2 && !v20 )
        {
          v20 = (int *)(v29 + 12LL * k);
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
  if ( *v8 != -1 || v8[1] != -1 )
    --*(_DWORD *)(a1 + 20);
  *v8 = *a2;
  v16 = a2[1];
  v8[2] = 0;
  v8[1] = v16;
  return v8 + 2;
}
