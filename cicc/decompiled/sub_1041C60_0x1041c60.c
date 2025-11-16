// Function: sub_1041C60
// Address: 0x1041c60
//
__int64 *__fastcall sub_1041C60(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 *v7; // rax
  __int64 *v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 *v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 *v15; // rsi
  __int64 *v16; // rdx
  __int64 *result; // rax
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rdx
  _QWORD *i; // rax
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rsi

  v7 = (__int64 *)sub_10416E0(a1, a3);
  v8 = v7;
  if ( a4 )
  {
    v25 = *v7;
    *(_QWORD *)(a2 + 40) = v7;
    v25 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a2 + 32) = v25 | *(_QWORD *)(a2 + 32) & 7LL;
    *(_QWORD *)(v25 + 8) = a2 + 32;
    *v7 = *v7 & 7 | (a2 + 32);
    if ( *(_BYTE *)a2 != 26 )
    {
      i = (_QWORD *)sub_1041AC0(a1, a3);
      goto LABEL_20;
    }
LABEL_4:
    if ( *(_BYTE *)(a1 + 164) )
      goto LABEL_5;
LABEL_21:
    result = sub_C8CA60(a1 + 136, a3);
    if ( result )
    {
      *result = -2;
      ++*(_DWORD *)(a1 + 160);
      ++*(_QWORD *)(a1 + 136);
    }
    return result;
  }
  v9 = (__int64 *)v7[1];
  if ( *(_BYTE *)a2 == 28 )
  {
    v10 = *v9;
    v11 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a2 + 40) = v9;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a2 + 32) = v10 | v11 & 7;
    *(_QWORD *)(v10 + 8) = a2 + 32;
    *v9 = *v9 & 7 | (a2 + 32);
    v12 = *(__int64 **)(sub_1041AC0(a1, a3) + 8);
    v13 = *v12;
    v14 = *(_QWORD *)(a2 + 48) & 7LL;
    *(_QWORD *)(a2 + 56) = v12;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a2 + 48) = v13 | v14;
    *(_QWORD *)(v13 + 8) = a2 + 48;
    *v12 = *v12 & 7 | (a2 + 48);
    goto LABEL_4;
  }
  while ( v8 != v9 )
  {
    if ( !v9 )
      BUG();
    if ( *((_BYTE *)v9 - 32) != 28 )
      break;
    v9 = (__int64 *)v9[1];
  }
  v19 = *v9;
  v20 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a2 + 40) = v9;
  v19 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a2 + 32) = v19 | v20 & 7;
  *(_QWORD *)(v19 + 8) = a2 + 32;
  *v9 = *v9 & 7 | (a2 + 32);
  if ( *(_BYTE *)a2 == 26 )
    goto LABEL_4;
  v21 = sub_1041AC0(a1, a3);
  for ( i = *(_QWORD **)(v21 + 8); (_QWORD *)v21 != i; i = (_QWORD *)i[1] )
  {
    if ( !i )
      BUG();
    if ( *((_BYTE *)i - 48) != 28 )
      break;
  }
LABEL_20:
  v23 = *i;
  v24 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a2 + 56) = i;
  v23 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a2 + 48) = v23 | v24 & 7;
  *(_QWORD *)(v23 + 8) = a2 + 48;
  *i = *i & 7LL | (a2 + 48);
  if ( !*(_BYTE *)(a1 + 164) )
    goto LABEL_21;
LABEL_5:
  v15 = *(__int64 **)(a1 + 144);
  v16 = &v15[*(unsigned int *)(a1 + 156)];
  result = v15;
  if ( v15 != v16 )
  {
    while ( a3 != *result )
    {
      if ( v16 == ++result )
        return result;
    }
    v18 = (unsigned int)(*(_DWORD *)(a1 + 156) - 1);
    *(_DWORD *)(a1 + 156) = v18;
    *result = v15[v18];
    ++*(_QWORD *)(a1 + 136);
  }
  return result;
}
