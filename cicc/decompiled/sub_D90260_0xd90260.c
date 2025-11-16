// Function: sub_D90260
// Address: 0xd90260
//
void (__fastcall *__fastcall sub_D90260(__int64 *a1, __int64 a2, __int64 a3, __int64 a4))(__int64 *, __int64, __int64)
{
  void (__fastcall *result)(__int64 *, __int64, __int64); // rax
  bool v7; // zf

  *a1 = a2;
  a1[3] = 0;
  result = *(void (__fastcall **)(__int64 *, __int64, __int64))(a3 + 16);
  if ( result )
  {
    a2 = a3;
    result(a1 + 1, a3, 2);
    a1[4] = *(_QWORD *)(a3 + 24);
    result = *(void (__fastcall **)(__int64 *, __int64, __int64))(a3 + 16);
    a1[3] = (__int64)result;
  }
  v7 = (_BYTE)qword_4F87FC8 == 0;
  a1[5] = a4;
  a1[6] = 0;
  if ( !v7 )
    return (void (__fastcall *)(__int64 *, __int64, __int64))sub_D8E7E0(a1, a2);
  return result;
}
