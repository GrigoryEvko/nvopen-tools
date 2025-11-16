// Function: sub_154EE80
// Address: 0x154ee80
//
void __fastcall sub_154EE80(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r14
  _QWORD *v7; // rbx
  _QWORD *v8; // rbx
  _QWORD *v9; // r15
  _QWORD *v10; // r14
  unsigned int v11; // ebx
  int v12; // r13d
  __int64 v13; // rsi
  _QWORD *v14; // rbx
  _QWORD *i; // r13
  __int64 v16; // rsi
  __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(_QWORD **)a1;
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
  v6 = *(_QWORD *)a1 + 8LL;
  if ( v5 != v6 )
  {
    do
    {
      if ( !v5 )
        BUG();
      if ( (*(_BYTE *)(v5 - 33) & 0x20) == 0 )
        sub_154E220((__int64)a1, v5 - 56);
      sub_154E850((__int64)a1, v5 - 56);
      a2 = *(_QWORD *)(v5 + 16);
      if ( a2 )
        sub_154EC60((__int64)a1, a2);
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v6 != v5 );
    v4 = *(_QWORD **)a1;
  }
  v7 = (_QWORD *)v4[6];
  if ( v4 + 5 != v7 )
  {
    do
    {
      if ( !v7 )
        BUG();
      if ( (*((_BYTE *)v7 - 25) & 0x20) == 0 )
      {
        a2 = (__int64)(v7 - 6);
        sub_154E220((__int64)a1, (__int64)(v7 - 6));
      }
      v7 = (_QWORD *)v7[1];
    }
    while ( v4 + 5 != v7 );
    v4 = *(_QWORD **)a1;
  }
  v8 = (_QWORD *)v4[8];
  if ( v4 + 7 != v8 )
  {
    do
    {
      if ( !v8 )
        BUG();
      if ( (*((_BYTE *)v8 - 25) & 0x20) == 0 )
      {
        a2 = (__int64)(v8 - 6);
        sub_154E220((__int64)a1, (__int64)(v8 - 6));
      }
      v8 = (_QWORD *)v8[1];
    }
    while ( v4 + 7 != v8 );
    v4 = *(_QWORD **)a1;
  }
  v9 = (_QWORD *)v4[10];
  v10 = v4 + 9;
  if ( v4 + 9 != v9 )
  {
    do
    {
      v11 = 0;
      v12 = sub_161F520(v9, a2, a3, a4);
      if ( v12 )
      {
        do
        {
          v13 = v11++;
          a2 = sub_161F530(v9, v13);
          sub_154E670((__int64)a1, a2);
        }
        while ( v12 != v11 );
      }
      v9 = (_QWORD *)v9[1];
    }
    while ( v10 != v9 );
    v4 = *(_QWORD **)a1;
  }
  v14 = (_QWORD *)v4[4];
  for ( i = v4 + 3; i != v14; v14 = (_QWORD *)v14[1] )
  {
    if ( !v14 )
      BUG();
    if ( (*((_BYTE *)v14 - 33) & 0x20) == 0 )
      sub_154E220((__int64)a1, (__int64)(v14 - 7));
    if ( a1[17] )
      sub_154EA00((__int64)a1, (__int64)(v14 - 7));
    v17[0] = v14[7];
    v16 = sub_1560250(v17);
    if ( v16 )
      sub_154EC60((__int64)a1, v16);
  }
}
