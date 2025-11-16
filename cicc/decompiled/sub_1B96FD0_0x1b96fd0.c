// Function: sub_1B96FD0
// Address: 0x1b96fd0
//
_QWORD *__fastcall sub_1B96FD0(__int64 *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v6; // rdi
  unsigned __int64 *v7; // r14
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  _QWORD v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = a1[1];
  if ( v6 )
  {
    v7 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v6 + 40, (__int64)a2);
    v8 = a2[3];
    v9 = *v7;
    a2[4] = v7;
    v9 &= 0xFFFFFFFFFFFFFFF8LL;
    a2[3] = v9 | v8 & 7;
    *(_QWORD *)(v9 + 8) = a2 + 3;
    *v7 = *v7 & 7 | (unsigned __int64)(a2 + 3);
  }
  sub_164B780((__int64)a2, a3);
  v10 = *a1;
  if ( *a1 )
  {
    v14[0] = *a1;
    sub_1623A60((__int64)v14, v10, 2);
    v11 = a2[6];
    if ( v11 )
      sub_161E7C0((__int64)(a2 + 6), v11);
    v12 = (unsigned __int8 *)v14[0];
    a2[6] = v14[0];
    if ( v12 )
      sub_1623210((__int64)v14, v12, (__int64)(a2 + 6));
  }
  return a2;
}
