// Function: sub_2304630
// Address: 0x2304630
//
_QWORD *__fastcall sub_2304630(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  _BYTE v8[1584]; // [rsp+0h] [rbp-C80h] BYREF
  _BYTE v9[1616]; // [rsp+630h] [rbp-650h] BYREF

  sub_D9A8C0((__int64)v8, a2 + 8, a3, a4);
  v4 = v8;
  sub_D99330((__int64)v9, (__int64)v8);
  v5 = (_QWORD *)sub_22077B0(0x630u);
  v6 = v5;
  if ( v5 )
  {
    v4 = v9;
    *v5 = &unk_4A0AE30;
    sub_D99330((__int64)(v5 + 1), (__int64)v9);
  }
  sub_DA11D0((__int64)v9, (__int64)v4);
  *a1 = v6;
  sub_DA11D0((__int64)v8, (__int64)v4);
  return a1;
}
