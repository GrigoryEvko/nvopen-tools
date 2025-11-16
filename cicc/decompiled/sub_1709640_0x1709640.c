// Function: sub_1709640
// Address: 0x1709640
//
_QWORD *__fastcall sub_1709640(__int64 a1, _QWORD *a2, __int64 *a3)
{
  __int64 v6; // rdi
  unsigned __int64 *v7; // r14
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  bool v11; // zf
  __int64 v12; // rsi
  __int64 v13; // rsi
  unsigned __int8 *v14; // rsi
  _QWORD v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  if ( v6 )
  {
    v7 = *(unsigned __int64 **)(a1 + 16);
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
  v11 = *(_QWORD *)(a1 + 80) == 0;
  v16[0] = a2;
  if ( v11 )
    sub_4263D6(a2, a3, v10);
  (*(void (__fastcall **)(__int64, _QWORD *))(a1 + 88))(a1 + 64, v16);
  v12 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v16[0] = *(_QWORD *)a1;
    sub_1623A60((__int64)v16, v12, 2);
    v13 = a2[6];
    if ( v13 )
      sub_161E7C0((__int64)(a2 + 6), v13);
    v14 = (unsigned __int8 *)v16[0];
    a2[6] = v16[0];
    if ( v14 )
      sub_1623210((__int64)v16, v14, (__int64)(a2 + 6));
  }
  return a2;
}
