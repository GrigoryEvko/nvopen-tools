// Function: sub_1D122D0
// Address: 0x1d122d0
//
_QWORD *__fastcall sub_1D122D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  _QWORD *v3; // rax
  _QWORD *v4; // r12
  __int64 v5; // r14
  __int64 v6; // rsi
  __int64 v7; // r13
  __int64 (*v8)(); // rax
  __int64 v9; // rax

  v1 = sub_22077B0(200);
  v2 = v1;
  if ( v1 )
    sub_2042C10(v1, a1);
  v3 = (_QWORD *)sub_22077B0(712);
  v4 = v3;
  if ( v3 )
  {
    v5 = *(_QWORD *)(a1 + 288);
    v6 = *(_QWORD *)(a1 + 256);
    sub_1D0DA30(v3, v6);
    v4[83] = v2;
    v4[84] = 0;
    v4[85] = 0;
    v4[86] = 0;
    v7 = *(_QWORD *)(v6 + 16);
    *v4 = off_49F9898;
    v4[88] = v5;
    v8 = *(__int64 (**)())(*(_QWORD *)v7 + 40LL);
    if ( v8 == sub_1D00B00 )
      BUG();
    v9 = ((__int64 (__fastcall *)(__int64))v8)(v7);
    v4[87] = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *))(*(_QWORD *)v9 + 752LL))(v9, v7, v4);
  }
  return v4;
}
