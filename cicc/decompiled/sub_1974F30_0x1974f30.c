// Function: sub_1974F30
// Address: 0x1974f30
//
void (__fastcall *__fastcall sub_1974F30(__int64 a1, __int64 a2))(__int64, __int64, __int64)
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
