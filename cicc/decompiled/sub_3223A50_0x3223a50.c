// Function: sub_3223A50
// Address: 0x3223a50
//
_QWORD *__fastcall sub_3223A50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *result; // rax
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  result = (_QWORD *)sub_37362B0(a2, a3);
  if ( !result )
  {
    v7[0] = a4;
    result = sub_32239E0((_QWORD *)(a1 + 192), v7);
    if ( result )
      return (_QWORD *)sub_373BC10(a2, a3, result + 2);
  }
  return result;
}
