// Function: sub_2953940
// Address: 0x2953940
//
__int64 *__fastcall sub_2953940(__int64 a1, _QWORD *a2, __int64 *a3)
{
  __int64 *result; // rax
  unsigned int v5; // esi
  int v7; // edx
  int v8; // edx
  int v9; // edx
  __int64 v10; // r8
  int v11; // edx
  __int64 *v12; // r10
  __int64 v13; // rdi
  int v14; // r11d
  unsigned int i; // ecx
  __int64 v16; // r9
  unsigned int v17; // ecx
  int v18; // edx
  __int64 v19; // r8
  int v20; // edx
  __int64 v21; // rdi
  int v22; // r11d
  unsigned int j; // ecx
  __int64 v24; // r9
  unsigned int v25; // ecx

  result = a3;
  v5 = *(_DWORD *)(a1 + 24);
  v7 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v8 = v7 + 1;
  if ( 4 * v8 >= 3 * v5 )
  {
    sub_2953570(a1, 2 * v5);
    v9 = *(_DWORD *)(a1 + 24);
    if ( v9 )
    {
      v10 = a2[1];
      v11 = v9 - 1;
      v12 = 0;
      v14 = 1;
      for ( i = v11
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
                  | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = v11 & v17 )
      {
        v13 = *(_QWORD *)(a1 + 8);
        result = (__int64 *)(v13 + 48LL * i);
        v16 = *result;
        if ( *result == *a2 && result[1] == v10 )
          break;
        if ( v16 == -4096 )
        {
          if ( result[1] == -4096 )
          {
LABEL_30:
            if ( v12 )
              result = v12;
            v8 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_3;
          }
        }
        else if ( v16 == -8192 && result[1] == -8192 && !v12 )
        {
          v12 = (__int64 *)(v13 + 48LL * i);
        }
        v17 = v14 + i;
        ++v14;
      }
      goto LABEL_26;
    }
LABEL_35:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v8 <= v5 >> 3 )
  {
    sub_2953570(a1, v5);
    v18 = *(_DWORD *)(a1 + 24);
    if ( v18 )
    {
      v19 = a2[1];
      v20 = v18 - 1;
      v12 = 0;
      v22 = 1;
      for ( j = v20
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)
                  | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)))); ; j = v20 & v25 )
      {
        v21 = *(_QWORD *)(a1 + 8);
        result = (__int64 *)(v21 + 48LL * j);
        v24 = *result;
        if ( *result == *a2 && result[1] == v19 )
          break;
        if ( v24 == -4096 )
        {
          if ( result[1] == -4096 )
            goto LABEL_30;
        }
        else if ( v24 == -8192 && result[1] == -8192 && !v12 )
        {
          v12 = (__int64 *)(v21 + 48LL * j);
        }
        v25 = v22 + j;
        ++v22;
      }
LABEL_26:
      v8 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_3;
    }
    goto LABEL_35;
  }
LABEL_3:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *result != -4096 || result[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  return result;
}
