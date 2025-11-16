// Function: sub_38DD230
// Address: 0x38dd230
//
void (__fastcall *__fastcall sub_38DD230(__int64 a1))(__int64 a1, __int64 a2)
{
  void (__fastcall *result)(__int64, __int64); // rax
  void (__fastcall *v2)(__int64, __int64); // rsi

  result = (void (__fastcall *)(__int64, __int64))sub_38DD140(a1);
  if ( result )
  {
    v2 = result;
    result = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 8LL);
    if ( result == sub_38DBC00 )
      *((_QWORD *)v2 + 1) = 1;
    else
      return (void (__fastcall *)(__int64, __int64))((__int64 (__fastcall *)(__int64, _QWORD))result)(a1, v2);
  }
  return result;
}
