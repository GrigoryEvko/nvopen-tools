// Function: sub_33CFA90
// Address: 0x33cfa90
//
__int64 __fastcall sub_33CFA90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  _QWORD *v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rsi
  signed __int64 v8; // rdx
  _QWORD *v9; // rdx

  v5 = *(_QWORD **)(a2 + 40);
  v6 = 5LL * *(unsigned int *)(a2 + 64);
  v7 = &v5[v6];
  v8 = 0xCCCCCCCCCCCCCCCDLL * ((v6 * 8) >> 3);
  if ( v8 >> 2 )
  {
    v9 = &v5[20 * (v8 >> 2)];
    while ( a1 != *v5 )
    {
      if ( a1 == v5[5] )
      {
        LOBYTE(a5) = v7 != v5 + 5;
        return a5;
      }
      if ( a1 == v5[10] )
      {
        LOBYTE(a5) = v7 != v5 + 10;
        return a5;
      }
      if ( a1 == v5[15] )
      {
        LOBYTE(a5) = v7 != v5 + 15;
        return a5;
      }
      v5 += 20;
      if ( v9 == v5 )
      {
        v8 = 0xCCCCCCCCCCCCCCCDLL * (v7 - v5);
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
LABEL_11:
  if ( v8 == 2 )
  {
LABEL_18:
    if ( a1 != *v5 )
    {
      v5 += 5;
LABEL_14:
      a5 = 0;
      if ( a1 != *v5 )
        return a5;
      goto LABEL_8;
    }
    goto LABEL_8;
  }
  if ( v8 != 3 )
  {
    a5 = 0;
    if ( v8 != 1 )
      return a5;
    goto LABEL_14;
  }
  if ( a1 != *v5 )
  {
    v5 += 5;
    goto LABEL_18;
  }
LABEL_8:
  LOBYTE(a5) = v7 != v5;
  return a5;
}
