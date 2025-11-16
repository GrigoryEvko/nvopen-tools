// Function: sub_216E0A0
// Address: 0x216e0a0
//
__int64 __fastcall sub_216E0A0(__int64 a1)
{
  _QWORD *v1; // rsi
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rsi
  __int64 result; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  _QWORD *v11; // rax

  sub_1F49190(a1, (__int64)&unk_4FC9148, 0, 0);
  sub_1F49190(a1, (__int64)&unk_4FC5C8C, 0, 0);
  sub_1F49190(a1, (__int64)&unk_4FCAE51, 0, 0);
  sub_1F49190(a1, (__int64)&unk_4FCAACC, 0, 0);
  sub_1F49190(a1, (__int64)&unk_4FC434C, 0, 0);
  sub_1F49190(a1, (__int64)&unk_4FC7F6C, 0, 0);
  sub_1F49190(a1, (__int64)&unk_4FC8CB4, 0, 0);
  sub_1F49190(a1, (__int64)&unk_4FC3604, 0, 0);
  sub_1F49190(a1, (__int64)&unk_4FC84CC, 0, 0);
  sub_1F49190(a1, (__int64)&unk_4FCA74C, 0, 0);
  v1 = (_QWORD *)sub_1CB9000();
  sub_1F46490(a1, v1, 1, 1, 0);
  if ( (unsigned int)sub_1F45DD0(a1) )
  {
    v10 = (_QWORD *)sub_21BD490();
    sub_1F46490(a1, v10, 1, 1, 0);
  }
  v2 = (_QWORD *)sub_21BCE60();
  sub_1F46490(a1, v2, 1, 1, 0);
  v3 = (_QWORD *)sub_215D9D0();
  sub_1F46490(a1, v3, 1, 1, 1u);
  if ( (unsigned int)sub_1F45DD0(a1) )
  {
    v6 = (_QWORD *)sub_1A6A110();
    sub_1F46490(a1, v6, 1, 1, 0);
    v7 = (_QWORD *)sub_17060B0(1, 0);
    sub_1F46490(a1, v7, 1, 1, 0);
    sub_2165990(a1, 0);
    v8 = (_QWORD *)sub_19DD530();
    sub_1F46490(a1, v8, 1, 1, 0);
    v9 = (_QWORD *)sub_18FD350(0);
    sub_1F46490(a1, v9, 1, 1, 0);
  }
  v4 = (_QWORD *)sub_1B6D760(1);
  sub_1F46490(a1, v4, 1, 1, 0);
  if ( (unsigned int)sub_1F45DD0(a1) && byte_4FD24E0 )
  {
    v11 = (_QWORD *)sub_395C520(*(unsigned int *)(*(_QWORD *)(a1 + 208) + 1212LL));
    sub_1F46490(a1, v11, 1, 1, 0);
    v4 = (_QWORD *)sub_18F1D50();
    sub_1F46490(a1, v4, 1, 1, 0);
  }
  sub_1F46B80(a1, v4);
  result = sub_1F45DD0(a1);
  if ( (_DWORD)result )
    return sub_2165990(a1, 1);
  return result;
}
