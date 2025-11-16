// Function: sub_39A95F0
// Address: 0x39a95f0
//
__int64 __fastcall sub_39A95F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  signed __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // r15
  __int64 i; // r10
  __int64 v11; // rsi
  __int64 *v12; // r8
  __int64 v13; // rax
  __int64 v14; // r9
  char *v15; // r12
  char *v16; // rdx
  char *v17; // rcx
  char *v18; // rax
  __int64 v19; // r9
  _DWORD *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r11
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v26; // [rsp+10h] [rbp-30h]

  v5 = a3 & 1;
  v7 = a3 - 1;
  v26 = v5;
  result = v7 + ((unsigned __int64)v7 >> 63);
  v9 = v7 / 2;
  if ( a2 >= v7 / 2 )
  {
    v12 = (__int64 *)(a1 + 8 * a2);
    if ( v5 )
      goto LABEL_32;
    v11 = a2;
    goto LABEL_28;
  }
  for ( i = a2; ; i = v11 )
  {
    v11 = 2 * (i + 1);
    v12 = (__int64 *)(a1 + 16 * (i + 1));
    v13 = *(v12 - 1);
    v14 = *v12;
    v15 = *(char **)(v13 + 104);
    v16 = *(char **)(v13 + 96);
    v17 = *(char **)(*v12 + 104);
    v18 = *(char **)(*v12 + 96);
    if ( v17 - v18 > v15 - v16 )
      v17 = &v18[v15 - v16];
    if ( v18 != v17 )
      break;
LABEL_13:
    if ( v15 != v16 )
      goto LABEL_10;
    *(_QWORD *)(a1 + 8 * i) = v14;
    if ( v11 >= v9 )
      goto LABEL_15;
LABEL_12:
    ;
  }
  while ( *(_DWORD *)v18 >= *(_DWORD *)v16 )
  {
    if ( *(_DWORD *)v18 > *(_DWORD *)v16 )
      goto LABEL_11;
    v18 += 4;
    v16 += 4;
    if ( v17 == v18 )
      goto LABEL_13;
  }
LABEL_10:
  --v11;
  v12 = (__int64 *)(a1 + 8 * v11);
  v14 = *v12;
LABEL_11:
  *(_QWORD *)(a1 + 8 * i) = v14;
  if ( v11 < v9 )
    goto LABEL_12;
LABEL_15:
  if ( !v26 )
  {
LABEL_28:
    if ( (a3 - 2) / 2 == v11 )
    {
      v23 = 2 * v11 + 2;
      v24 = *(_QWORD *)(a1 + 8 * v23 - 8);
      v11 = v23 - 1;
      *v12 = v24;
      v12 = (__int64 *)(a1 + 8 * v11);
    }
  }
  result = v11 - 1;
  v19 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    while ( 1 )
    {
      v12 = (__int64 *)(a1 + 8 * v19);
      v20 = *(_DWORD **)(a4 + 96);
      v21 = *(_QWORD *)(*v12 + 104);
      result = *(_QWORD *)(*v12 + 96);
      v22 = *(_QWORD *)(a4 + 104) - (_QWORD)v20;
      if ( v21 - result > v22 )
        v21 = result + v22;
      if ( result == v21 )
      {
LABEL_30:
        if ( v20 == *(_DWORD **)(a4 + 104) )
        {
LABEL_31:
          v12 = (__int64 *)(a1 + 8 * v11);
          break;
        }
      }
      else
      {
        while ( *(_DWORD *)result >= *v20 )
        {
          if ( *(_DWORD *)result > *v20 )
            goto LABEL_31;
          result += 4;
          ++v20;
          if ( v21 == result )
            goto LABEL_30;
        }
      }
      *(_QWORD *)(a1 + 8 * v11) = *v12;
      v11 = v19;
      result = (v19 - 1) / 2;
      if ( a2 >= v19 )
        break;
      v19 = (v19 - 1) / 2;
    }
  }
LABEL_32:
  *v12 = a4;
  return result;
}
