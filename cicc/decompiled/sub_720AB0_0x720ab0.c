// Function: sub_720AB0
// Address: 0x720ab0
//
_DWORD *__fastcall sub_720AB0(__int64 a1, int a2)
{
  _DWORD *result; // rax

  result = dword_4F07680;
  if ( dword_4F07680[0] )
  {
    if ( unk_4F0759C )
      sub_7209D0(a1, &qword_4F076A8, &qword_4F076A0);
    else
      sub_720AA0(a1);
    result = (_DWORD *)qword_4F076A8;
    *(_DWORD *)(qword_4F076A8 + 8LL) = a2;
  }
  return result;
}
