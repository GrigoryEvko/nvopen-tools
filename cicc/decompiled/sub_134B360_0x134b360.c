// Function: sub_134B360
// Address: 0x134b360
//
__int64 (__fastcall *__fastcall sub_134B360(_QWORD *a1))(_QWORD, _QWORD, _QWORD)
{
  __int64 (__fastcall *result)(_QWORD, _QWORD, _QWORD); // rax

  sub_134B260(a1);
  result = (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))a1[30];
  if ( result )
    return (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))result(a1[31], a1[103], a1[105]);
  return result;
}
