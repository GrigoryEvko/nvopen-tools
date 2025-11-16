// Function: sub_B6EAA0
// Address: 0xb6eaa0
//
__int64 (__fastcall *__fastcall sub_B6EAA0(__int64 a1))(__int64, _QWORD)
{
  __int64 (__fastcall *result)(__int64, _QWORD); // rax

  result = *(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 160LL);
  if ( result )
    return (__int64 (__fastcall *)(__int64, _QWORD))result(a1, *(_QWORD *)(*(_QWORD *)a1 + 168LL));
  return result;
}
