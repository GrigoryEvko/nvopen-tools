// Function: sub_7E4FA0
// Address: 0x7e4fa0
//
_QWORD *__fastcall sub_7E4FA0(_QWORD *a1)
{
  __int64 i; // rbx
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rdi

  for ( i = a1[21]; i; i = *(_QWORD *)(i + 112) )
  {
    while ( (*(_BYTE *)(i + 124) & 1) != 0 )
    {
      i = *(_QWORD *)(i + 112);
      if ( !i )
        goto LABEL_6;
    }
    sub_7E4FA0(*(_QWORD *)(i + 128));
  }
LABEL_6:
  if ( a1[13] )
  {
    v3 = sub_85EB10(a1);
    v4 = a1[13];
    v5 = qword_4D03FF0;
    v6 = *(_QWORD *)(qword_4D03FF0 + 56);
    if ( v6 )
      *(_QWORD *)(v6 + 112) = v4;
    else
      *(_QWORD *)(unk_4F07288 + 104LL) = v4;
    *(_QWORD *)(v5 + 56) = *(_QWORD *)(v3 + 32);
    a1[13] = 0;
    *(_QWORD *)(v3 + 32) = 0;
  }
  v7 = a1[12];
  while ( v7 )
  {
    v8 = v7;
    v7 = *(_QWORD *)(v7 + 120);
    sub_733310(v8, 1);
  }
  a1[12] = 0;
  v9 = a1[14];
  while ( v9 )
  {
    while ( 1 )
    {
      v10 = v9;
      v9 = *(_QWORD *)(v9 + 112);
      sub_735E40(v10, 0);
      if ( (*(_BYTE *)(v10 + 170) & 0x10) != 0 && !*(_BYTE *)(v10 + 136) )
        break;
      if ( !v9 )
        goto LABEL_18;
    }
    sub_7E4C10(v10);
  }
LABEL_18:
  a1[14] = 0;
  sub_7DFE30((__int64)a1);
  v11 = a1[19];
  while ( v11 )
  {
    v12 = v11;
    v11 = *(_QWORD *)(v11 + 112);
    sub_733510(v12);
  }
  a1[19] = 0;
  return sub_7DF680((__int64)a1);
}
