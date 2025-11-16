// Function: sub_3363950
// Address: 0x3363950
//
_QWORD *__fastcall sub_3363950(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  _QWORD *v3; // rax
  _QWORD *v4; // r12
  _QWORD *v5; // rsi
  __int64 v6; // r13
  __int64 (*v7)(); // rax
  __int64 v8; // rax

  v1 = sub_22077B0(0xC8u);
  v2 = v1;
  if ( v1 )
    sub_37B4EC0(v1, a1);
  v3 = (_QWORD *)sub_22077B0(0x2A0u);
  v4 = v3;
  if ( v3 )
  {
    v5 = *(_QWORD **)(a1 + 40);
    sub_335DCC0(v3, v5);
    v4[80] = 0;
    v4[81] = 0;
    v4[82] = 0;
    v6 = v5[2];
    *v4 = off_4A36640;
    v4[79] = v2;
    v7 = *(__int64 (**)())(*(_QWORD *)v6 + 128LL);
    if ( v7 == sub_2DAC790 )
      BUG();
    v8 = ((__int64 (__fastcall *)(__int64))v7)(v6);
    v4[83] = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *))(*(_QWORD *)v8 + 1032LL))(v8, v6, v4);
  }
  return v4;
}
