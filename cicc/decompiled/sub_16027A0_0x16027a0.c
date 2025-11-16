// Function: sub_16027A0
// Address: 0x16027a0
//
__int64 (__fastcall *__fastcall sub_16027A0(__int64 a1))(__int64, _QWORD)
{
  __int64 (__fastcall *result)(__int64, _QWORD); // rax

  result = *(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 120LL);
  if ( result )
    return (__int64 (__fastcall *)(__int64, _QWORD))result(a1, *(_QWORD *)(*(_QWORD *)a1 + 128LL));
  return result;
}
