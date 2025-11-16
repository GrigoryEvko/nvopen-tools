// Function: sub_C87B00
// Address: 0xc87b00
//
_QWORD *__fastcall sub_C87B00(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  _QWORD *v5; // r14
  _QWORD *v7; // rbx
  _BYTE *v8; // rsi
  __int64 v9; // rax
  _QWORD *v10; // rax
  _BYTE *v12; // rsi
  _QWORD v13[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = 2 * a3;
  v5 = a2;
  v7 = &a2[v4];
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a2 == &a2[v4] )
  {
    v13[0] = 0;
    v12 = 0;
LABEL_13:
    sub_C87970((__int64)a1, v12, v13);
    return a1;
  }
  do
  {
    while ( 1 )
    {
      v9 = sub_C948A0(a4, *v5, v5[1]);
      v8 = (_BYTE *)a1[1];
      v13[0] = v9;
      if ( v8 != (_BYTE *)a1[2] )
        break;
      v5 += 2;
      sub_C87970((__int64)a1, v8, v13);
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
  v10 = (_QWORD *)a1[1];
  v12 = (_BYTE *)a1[2];
  v13[0] = 0;
  if ( v10 == (_QWORD *)v12 )
    goto LABEL_13;
  if ( v10 )
  {
    *v10 = 0;
    v10 = (_QWORD *)a1[1];
  }
  a1[1] = v10 + 1;
  return a1;
}
