// Function: sub_14264F0
// Address: 0x14264f0
//
_QWORD *__fastcall sub_14264F0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 *v7; // rax
  __int64 *v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *i; // rax
  __int64 v14; // rsi
  __int64 v15; // rdx
  _QWORD *result; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rsi

  v7 = (__int64 *)sub_1425DF0(a1, a3);
  v8 = v7;
  if ( a4 )
  {
    v25 = *v7;
    *(_QWORD *)(a2 + 40) = v7;
    v25 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a2 + 32) = v25 | *(_QWORD *)(a2 + 32) & 7LL;
    *(_QWORD *)(v25 + 8) = a2 + 32;
    *v7 = *v7 & 7 | (a2 + 32);
    if ( *(_BYTE *)(a2 + 16) != 21 )
    {
      i = (_QWORD *)sub_1426290(a1, a3);
      goto LABEL_12;
    }
  }
  else
  {
    v9 = (__int64 *)v7[1];
    if ( *(_BYTE *)(a2 + 16) == 23 )
    {
      v18 = *v9;
      v19 = *(_QWORD *)(a2 + 32);
      *(_QWORD *)(a2 + 40) = v9;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a2 + 32) = v18 | v19 & 7;
      *(_QWORD *)(v18 + 8) = a2 + 32;
      *v9 = *v9 & 7 | (a2 + 32);
      v20 = *(__int64 **)(sub_1426290(a1, a3) + 8);
      v21 = *v20;
      v22 = *(_QWORD *)(a2 + 48) & 7LL;
      *(_QWORD *)(a2 + 56) = v20;
      v21 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a2 + 48) = v21 | v22;
      *(_QWORD *)(v21 + 8) = a2 + 48;
      *v20 = *v20 & 7 | (a2 + 48);
    }
    else
    {
      for ( ; v8 != v9; v9 = (__int64 *)v9[1] )
      {
        if ( !v9 )
          BUG();
        if ( *((_BYTE *)v9 - 16) != 23 )
          break;
      }
      v10 = *v9;
      v11 = *(_QWORD *)(a2 + 32);
      *(_QWORD *)(a2 + 40) = v9;
      v10 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a2 + 32) = v10 | v11 & 7;
      *(_QWORD *)(v10 + 8) = a2 + 32;
      *v9 = *v9 & 7 | (a2 + 32);
      if ( *(_BYTE *)(a2 + 16) != 21 )
      {
        v12 = sub_1426290(a1, a3);
        for ( i = *(_QWORD **)(v12 + 8); (_QWORD *)v12 != i; i = (_QWORD *)i[1] )
        {
          if ( !i )
            BUG();
          if ( *((_BYTE *)i - 32) != 23 )
            break;
        }
LABEL_12:
        v14 = *i;
        v15 = *(_QWORD *)(a2 + 48);
        *(_QWORD *)(a2 + 56) = i;
        v14 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a2 + 48) = v14 | v15 & 7;
        *(_QWORD *)(v14 + 8) = a2 + 48;
        *i = *i & 7LL | (a2 + 48);
        result = *(_QWORD **)(a1 + 136);
        if ( *(_QWORD **)(a1 + 144) == result )
          goto LABEL_13;
        goto LABEL_21;
      }
    }
  }
  result = *(_QWORD **)(a1 + 136);
  if ( *(_QWORD **)(a1 + 144) == result )
  {
LABEL_13:
    v17 = &result[*(unsigned int *)(a1 + 156)];
    if ( result == v17 )
    {
LABEL_29:
      result = v17;
    }
    else
    {
      while ( a3 != *result )
      {
        if ( v17 == ++result )
          goto LABEL_29;
      }
    }
    goto LABEL_17;
  }
LABEL_21:
  result = (_QWORD *)sub_16CC9F0(a1 + 128, a3);
  if ( a3 == *result )
  {
    v23 = *(_QWORD *)(a1 + 144);
    if ( v23 == *(_QWORD *)(a1 + 136) )
      v24 = *(unsigned int *)(a1 + 156);
    else
      v24 = *(unsigned int *)(a1 + 152);
    v17 = (_QWORD *)(v23 + 8 * v24);
  }
  else
  {
    result = *(_QWORD **)(a1 + 144);
    if ( result != *(_QWORD **)(a1 + 136) )
      return result;
    result += *(unsigned int *)(a1 + 156);
    v17 = result;
  }
LABEL_17:
  if ( v17 != result )
  {
    *result = -2;
    ++*(_DWORD *)(a1 + 160);
  }
  return result;
}
