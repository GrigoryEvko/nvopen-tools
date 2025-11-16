// Function: sub_823EE0
// Address: 0x823ee0
//
_QWORD *__fastcall sub_823EE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax

  result = (_QWORD *)unk_4D03FE8;
  if ( !unk_4D03FE8 )
  {
    dword_4F073B8[0] = dword_4F073A8 + 1;
    return sub_8231F0(dword_4F073A8 + 1, 0, a3, a4, a5, a6);
  }
  return result;
}
