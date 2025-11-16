// Function: sub_720B20
// Address: 0x720b20
//
_QWORD *__fastcall sub_720B20(__int64 a1, int a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx

  result = dword_4F07680;
  if ( dword_4F07680[0] )
  {
    if ( unk_4F0759C )
    {
      result = (_QWORD *)qword_4F076A8;
      qword_4F076A8 = *(_QWORD *)(qword_4F076A8 + 16LL);
      v3 = qword_4F076A8;
      v4 = qword_4F07940;
      qword_4F07940 = (__int64)result;
      result[2] = v4;
      *(_DWORD *)(v3 + 8) = a2;
    }
    else
    {
      sub_720AA0(a1);
      *(_DWORD *)(qword_4F076A8 + 8LL) = a2;
      return &qword_4F076A8;
    }
  }
  return result;
}
