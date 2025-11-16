// Function: sub_305C5A0
// Address: 0x305c5a0
//
__int64 __fastcall sub_305C5A0(__int64 a1)
{
  _QWORD *v1; // rsi
  __int64 result; // rax
  _QWORD *v3; // rax

  v1 = (_QWORD *)sub_36F5A30();
  sub_2FF0E80(a1, v1, 1u);
  result = sub_2FF0570(a1);
  if ( (_DWORD)result )
  {
    v3 = (_QWORD *)sub_36F50A0();
    return sub_2FF0E80(a1, v3, 0);
  }
  return result;
}
