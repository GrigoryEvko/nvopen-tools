// Function: sub_D898E0
// Address: 0xd898e0
//
void (__fastcall *__fastcall sub_D898E0(_QWORD *a1, __int64 a2, __int64 a3))(_QWORD *, __int64, __int64)
{
  void (__fastcall *result)(_QWORD *, __int64, __int64); // rax

  *a1 = a2;
  a1[3] = 0;
  result = *(void (__fastcall **)(_QWORD *, __int64, __int64))(a3 + 16);
  if ( result )
  {
    result(a1 + 1, a3, 2);
    a1[4] = *(_QWORD *)(a3 + 24);
    result = *(void (__fastcall **)(_QWORD *, __int64, __int64))(a3 + 16);
    a1[3] = result;
  }
  a1[5] = 0;
  return result;
}
