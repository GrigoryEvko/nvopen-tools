// Function: sub_E70AE0
// Address: 0xe70ae0
//
__int64 __fastcall sub_E70AE0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v5; // rdi
  __int64 v7; // rsi
  int v8; // r11d
  unsigned int *v9; // r8
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  __int64 i; // r9
  __int64 v13; // rdx
  int v14; // r14d
  unsigned int v15; // r9d
  __int64 result; // rax
  __int64 *v17; // rbx
  int v18; // ecx
  int v19; // edx
  unsigned int *v20; // rdi
  __int64 v21; // rcx
  unsigned int j; // eax
  int v23; // eax
  int v24; // edx
  unsigned int k; // eax
  int v26; // ecx
  int v27; // eax
  int v28; // [rsp+8h] [rbp-28h]

  v5 = a1 + 1376;
  v7 = *(unsigned int *)(a1 + 1400);
  if ( !(_DWORD)v7 )
  {
    ++*(_QWORD *)(a1 + 1376);
    goto LABEL_25;
  }
  v8 = 1;
  v9 = 0;
  v10 = *(_QWORD *)(a1 + 1384);
  v11 = ((0xBF58476D1CE4E5B9LL * ((37 * a3) | ((unsigned __int64)(37 * a2) << 32))) >> 31)
      ^ (0xBF58476D1CE4E5B9LL * ((37 * a3) | ((unsigned __int64)(37 * a2) << 32)));
  for ( i = ((unsigned int)((0xBF58476D1CE4E5B9LL * ((37 * a3) | ((unsigned __int64)(37 * a2) << 32))) >> 31)
           ^ (756364221 * a3))
          & ((_DWORD)v7 - 1); ; i = ((_DWORD)v7 - 1) & v15 )
  {
    v13 = v10 + 16LL * (unsigned int)i;
    v14 = *(_DWORD *)v13;
    if ( a2 == *(_DWORD *)v13 && a3 == *(_DWORD *)(v13 + 4) )
    {
      result = *(_QWORD *)(v13 + 8);
      v17 = (__int64 *)(v13 + 8);
      if ( !result )
        goto LABEL_21;
      return result;
    }
    if ( v14 == -1 )
      break;
    if ( v14 == -2 && *(_DWORD *)(v13 + 4) == -2 && !v9 )
      v9 = (unsigned int *)(v10 + 16LL * (unsigned int)i);
LABEL_9:
    v15 = v8 + i;
    ++v8;
  }
  if ( *(_DWORD *)(v13 + 4) != -1 )
    goto LABEL_9;
  v18 = *(_DWORD *)(a1 + 1392);
  if ( !v9 )
    v9 = (unsigned int *)v13;
  ++*(_QWORD *)(a1 + 1376);
  v10 = (unsigned int)(v18 + 1);
  if ( 4 * (int)v10 >= (unsigned int)(3 * v7) )
  {
LABEL_25:
    sub_E70840(v5, 2 * v7);
    v19 = *(_DWORD *)(a1 + 1400);
    if ( v19 )
    {
      v13 = (unsigned int)(v19 - 1);
      v20 = 0;
      i = 1;
      for ( j = v13
              & (((0xBF58476D1CE4E5B9LL * ((37 * a3) | ((unsigned __int64)(37 * a2) << 32))) >> 31)
               ^ (756364221 * a3)); ; j = v13 & v23 )
      {
        v21 = *(_QWORD *)(a1 + 1384);
        v9 = (unsigned int *)(v21 + 16LL * j);
        v7 = *v9;
        if ( __PAIR64__(a3, a2) == *(_QWORD *)v9 )
          break;
        if ( (_DWORD)v7 == -1 )
        {
          if ( v9[1] == -1 )
          {
LABEL_47:
            if ( v20 )
              v9 = v20;
            v10 = (unsigned int)(*(_DWORD *)(a1 + 1392) + 1);
            goto LABEL_18;
          }
        }
        else if ( (_DWORD)v7 == -2 && v9[1] == -2 && !v20 )
        {
          v20 = (unsigned int *)(v21 + 16LL * j);
        }
        v23 = i + j;
        i = (unsigned int)(i + 1);
      }
      goto LABEL_43;
    }
LABEL_52:
    ++*(_DWORD *)(a1 + 1392);
    BUG();
  }
  v13 = (unsigned int)(v7 - *(_DWORD *)(a1 + 1396) - v10);
  i = (unsigned int)v7 >> 3;
  if ( (unsigned int)v13 <= (unsigned int)i )
  {
    v28 = v11;
    sub_E70840(v5, v7);
    v24 = *(_DWORD *)(a1 + 1400);
    if ( v24 )
    {
      v13 = (unsigned int)(v24 - 1);
      i = 1;
      v20 = 0;
      v7 = *(_QWORD *)(a1 + 1384);
      for ( k = v13 & v28; ; k = v13 & v27 )
      {
        v9 = (unsigned int *)(v7 + 16LL * k);
        v26 = *v9;
        if ( a2 == *v9 && a3 == v9[1] )
          break;
        if ( v26 == -1 )
        {
          if ( v9[1] == -1 )
            goto LABEL_47;
        }
        else if ( v26 == -2 && v9[1] == -2 && !v20 )
        {
          v20 = (unsigned int *)(v7 + 16LL * k);
        }
        v27 = i + k;
        i = (unsigned int)(i + 1);
      }
LABEL_43:
      v10 = (unsigned int)(*(_DWORD *)(a1 + 1392) + 1);
      goto LABEL_18;
    }
    goto LABEL_52;
  }
LABEL_18:
  *(_DWORD *)(a1 + 1392) = v10;
  if ( *v9 != -1 || v9[1] != -1 )
    --*(_DWORD *)(a1 + 1396);
  v9[1] = a3;
  *((_QWORD *)v9 + 1) = 0;
  *v9 = a2;
  v17 = (__int64 *)(v9 + 2);
LABEL_21:
  result = sub_E6C270(a1, v7, v13, v10, (__int64)v9, i);
  *v17 = result;
  return result;
}
