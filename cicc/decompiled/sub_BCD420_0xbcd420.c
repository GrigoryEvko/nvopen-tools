// Function: sub_BCD420
// Address: 0xbcd420
//
_QWORD *__fastcall sub_BCD420(__int64 *a1, __int64 a2)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // rcx
  int v8; // r14d
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // rax
  __int64 **v11; // rdx
  unsigned int i; // r10d
  _QWORD *v13; // r8
  __int64 *v14; // r9
  unsigned int v15; // r10d
  _QWORD *result; // rax
  _QWORD *v17; // r14
  int v18; // ecx
  int v19; // r8d
  int v20; // edx
  __int64 v21; // rcx
  __int64 **v22; // r9
  int v23; // r8d
  int v24; // edi
  unsigned int j; // eax
  __int64 *v26; // rsi
  unsigned int v27; // eax
  int v28; // edx
  int v29; // ecx
  int v30; // r8d
  __int64 v31; // rdi
  unsigned int k; // eax
  __int64 *v33; // rsi
  unsigned int v34; // eax
  _QWORD *v35; // [rsp+8h] [rbp-28h]
  int v36; // [rsp+8h] [rbp-28h]

  v4 = *(_QWORD *)*a1;
  v5 = *(_DWORD *)(v4 + 3056);
  v6 = v4 + 3032;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 3032);
    goto LABEL_27;
  }
  v7 = *(_QWORD *)(v4 + 3040);
  v8 = 1;
  v9 = (0xBF58476D1CE4E5B9LL * a2) >> 31;
  v10 = ((0xBF58476D1CE4E5B9LL
        * ((unsigned int)v9 ^ (484763065 * (_DWORD)a2)
         | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))) >> 31)
      ^ (0xBF58476D1CE4E5B9LL
       * ((unsigned int)v9 ^ (484763065 * (_DWORD)a2)
        | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32)));
  v11 = 0;
  for ( i = (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)v9 ^ (484763065 * (_DWORD)a2)
              | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))) >> 31)
           ^ (484763065 * (v9 ^ (484763065 * a2))))
          & (v5 - 1); ; i = (v5 - 1) & v15 )
  {
    v13 = (_QWORD *)(v7 + 24LL * i);
    v14 = (__int64 *)*v13;
    if ( a1 == (__int64 *)*v13 && a2 == v13[1] )
    {
      result = (_QWORD *)v13[2];
      v17 = v13 + 2;
      if ( !result )
        goto LABEL_21;
      return result;
    }
    if ( v14 == (__int64 *)-4096LL )
      break;
    if ( v14 == (__int64 *)-8192LL && v13[1] == -2 && !v11 )
      v11 = (__int64 **)(v7 + 24LL * i);
LABEL_9:
    v15 = v8 + i;
    ++v8;
  }
  if ( v13[1] != -1 )
    goto LABEL_9;
  v18 = *(_DWORD *)(v4 + 3048);
  if ( !v11 )
    v11 = (__int64 **)v13;
  ++*(_QWORD *)(v4 + 3032);
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v5 )
  {
LABEL_27:
    sub_BCD150(v6, 2 * v5);
    v20 = *(_DWORD *)(v4 + 3056);
    if ( v20 )
    {
      v21 = *(_QWORD *)(v4 + 3040);
      v22 = 0;
      v23 = 1;
      v24 = v20 - 1;
      for ( j = (v20 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * ((unsigned int)((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * (_DWORD)a2)
                  | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * a2)))); ; j = v24 & v27 )
      {
        v11 = (__int64 **)(v21 + 24LL * j);
        v26 = *v11;
        if ( a1 == *v11 && (__int64 *)a2 == v11[1] )
          break;
        if ( v26 == (__int64 *)-4096LL )
        {
          if ( v11[1] == (__int64 *)-1LL )
          {
LABEL_50:
            if ( v22 )
              v11 = v22;
            v19 = *(_DWORD *)(v4 + 3048) + 1;
            goto LABEL_18;
          }
        }
        else if ( v26 == (__int64 *)-8192LL && v11[1] == (__int64 *)-2LL && !v22 )
        {
          v22 = (__int64 **)(v21 + 24LL * j);
        }
        v27 = v23 + j;
        ++v23;
      }
      goto LABEL_46;
    }
LABEL_55:
    ++*(_DWORD *)(v4 + 3048);
    BUG();
  }
  if ( v5 - *(_DWORD *)(v4 + 3052) - v19 <= v5 >> 3 )
  {
    v36 = v10;
    sub_BCD150(v6, v5);
    v28 = *(_DWORD *)(v4 + 3056);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = 1;
      v22 = 0;
      for ( k = (v28 - 1) & v36; ; k = v29 & v34 )
      {
        v31 = *(_QWORD *)(v4 + 3040);
        v11 = (__int64 **)(v31 + 24LL * k);
        v33 = *v11;
        if ( a1 == *v11 && (__int64 *)a2 == v11[1] )
          break;
        if ( v33 == (__int64 *)-4096LL )
        {
          if ( v11[1] == (__int64 *)-1LL )
            goto LABEL_50;
        }
        else if ( v33 == (__int64 *)-8192LL && v11[1] == (__int64 *)-2LL && !v22 )
        {
          v22 = (__int64 **)(v31 + 24LL * k);
        }
        v34 = v30 + k;
        ++v30;
      }
LABEL_46:
      v19 = *(_DWORD *)(v4 + 3048) + 1;
      goto LABEL_18;
    }
    goto LABEL_55;
  }
LABEL_18:
  *(_DWORD *)(v4 + 3048) = v19;
  if ( *v11 != (__int64 *)-4096LL || v11[1] != (__int64 *)-1LL )
    --*(_DWORD *)(v4 + 3052);
  *v11 = a1;
  v17 = v11 + 2;
  v11[1] = (__int64 *)a2;
  v11[2] = 0;
LABEL_21:
  result = (_QWORD *)sub_A777F0(0x28u, (__int64 *)(v4 + 2640));
  if ( result )
  {
    v35 = result;
    sub_BCBC30(result, a1, a2);
    result = v35;
  }
  *v17 = result;
  return result;
}
