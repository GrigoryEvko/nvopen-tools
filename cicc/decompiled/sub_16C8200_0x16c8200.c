// Function: sub_16C8200
// Address: 0x16c8200
//
_QWORD *__fastcall sub_16C8200(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  _QWORD *v5; // r14
  _QWORD *v7; // rbx
  _BYTE *v8; // rsi
  __int64 v9; // rax
  _BYTE *v11; // rsi
  _QWORD v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = 2 * a3;
  v5 = a2;
  v7 = &a2[v4];
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a2 == &a2[v4] )
  {
    v12[0] = 0;
    v11 = 0;
LABEL_13:
    sub_C87970((__int64)a1, v11, v12);
    return a1;
  }
  do
  {
    while ( 1 )
    {
      v9 = sub_16D3940(a4, *v5, v5[1]);
      v8 = (_BYTE *)a1[1];
      v12[0] = v9;
      if ( v8 != (_BYTE *)a1[2] )
        break;
      v5 += 2;
      sub_C87970((__int64)a1, v8, v12);
      if ( v7 == v5 )
        goto LABEL_8;
    }
    if ( v8 )
    {
      *(_QWORD *)v8 = v9;
      v8 = (_BYTE *)a1[1];
    }
    v5 += 2;
    a1[1] = v8 + 8;
  }
  while ( v7 != v5 );
LABEL_8:
  v12[0] = 0;
  v11 = (_BYTE *)a1[1];
  if ( (_BYTE *)a1[2] == v11 )
    goto LABEL_13;
  if ( v11 )
  {
    *(_QWORD *)v11 = 0;
    v11 = (_BYTE *)a1[1];
  }
  a1[1] = v11 + 8;
  return a1;
}
