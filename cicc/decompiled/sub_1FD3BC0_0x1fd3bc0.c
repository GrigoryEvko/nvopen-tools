// Function: sub_1FD3BC0
// Address: 0x1fd3bc0
//
_QWORD *__fastcall sub_1FD3BC0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r13
  __int64 v6; // rsi
  unsigned __int8 *v7; // rsi
  __int64 v8; // rsi
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int8 *v11; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)(v3 + 792);
  v10 = v4;
  if ( v4 )
    sub_1623A60((__int64)&v10, v4, 2);
  sub_1FD3A30(a2);
  v6 = *(_QWORD *)(a2 + 80);
  v11 = 0;
  if ( v6 )
  {
    sub_161E7C0(a2 + 80, v6);
    v7 = v11;
    *(_QWORD *)(a2 + 80) = v11;
    if ( v7 )
      sub_1623210((__int64)&v11, v7, a2 + 80);
  }
  v8 = v10;
  *a1 = v5;
  a1[1] = v8;
  if ( v8 )
  {
    sub_1623A60((__int64)(a1 + 1), v8, 2);
    if ( v10 )
      sub_161E7C0((__int64)&v10, v10);
  }
  return a1;
}
