// Function: sub_1DA0B60
// Address: 0x1da0b60
//
_QWORD *__fastcall sub_1DA0B60(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v4; // rbx
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rax
  unsigned __int64 v7; // rdx
  bool v8; // cl
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rax
  bool v11; // cf
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rdi

  v4 = *(_QWORD **)(a1 + 16);
  if ( v4 )
  {
    v5 = *a2;
    while ( 1 )
    {
      v7 = v4[4];
      v8 = v5 < v7;
      if ( v5 == v7 )
      {
        v9 = a2[1];
        v10 = v4[5];
        v8 = v9 < v10;
        if ( v9 == v10 )
          v8 = a2[15] < v4[19];
      }
      v6 = (_QWORD *)v4[3];
      if ( v8 )
        v6 = (_QWORD *)v4[2];
      if ( !v6 )
        break;
      v4 = v6;
    }
    if ( !v8 )
    {
      v11 = v7 < v5;
      if ( v7 != v5 )
        goto LABEL_12;
LABEL_17:
      v15 = a2[1];
      v11 = v4[5] < v15;
      if ( v4[5] == v15 )
      {
        if ( v4[19] >= a2[15] )
          return v4;
        return 0;
      }
LABEL_12:
      if ( !v11 )
        return v4;
      return 0;
    }
  }
  else
  {
    v4 = (_QWORD *)(a1 + 8);
  }
  if ( *(_QWORD **)(a1 + 24) != v4 )
  {
    v13 = sub_220EF80(v4);
    v14 = *(_QWORD *)(v13 + 32);
    v4 = (_QWORD *)v13;
    v11 = v14 < *a2;
    if ( v14 != *a2 )
      goto LABEL_12;
    goto LABEL_17;
  }
  return 0;
}
