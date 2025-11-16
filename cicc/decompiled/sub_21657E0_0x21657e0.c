// Function: sub_21657E0
// Address: 0x21657e0
//
__int64 __fastcall sub_21657E0(__int64 a1)
{
  _QWORD *v1; // rsi
  __int64 result; // rax
  _QWORD *v3; // rax

  v1 = (_QWORD *)sub_21DBC70();
  sub_1F46490(a1, v1, 0, 1, 1u);
  result = sub_1F45DD0(a1);
  if ( (_DWORD)result )
  {
    v3 = (_QWORD *)sub_21DB3B0();
    return sub_1F46490(a1, v3, 1, 1, 0);
  }
  return result;
}
