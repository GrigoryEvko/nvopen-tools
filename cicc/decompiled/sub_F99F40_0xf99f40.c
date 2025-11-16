// Function: sub_F99F40
// Address: 0xf99f40
//
void (__fastcall *__fastcall sub_F99F40(__int64 a1, __int64 a2))(__int64, __int64, __int64)
{
  void (__fastcall *result)(__int64, __int64, __int64); // rax

  *(_QWORD *)(a1 + 16) = 0;
  result = *(void (__fastcall **)(__int64, __int64, __int64))(a2 + 16);
  if ( result )
  {
    result(a1, a2, 2);
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
    result = *(void (__fastcall **)(__int64, __int64, __int64))(a2 + 16);
    *(_QWORD *)(a1 + 16) = result;
  }
  return result;
}
