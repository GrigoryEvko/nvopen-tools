// Function: sub_A17130
// Address: 0xa17130
//
__int64 (__fastcall *__fastcall sub_A17130(__int64 a1))(__int64, __int64, __int64)
{
  __int64 (__fastcall *result)(__int64, __int64, __int64); // rax

  result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 16);
  if ( result )
    return (__int64 (__fastcall *)(__int64, __int64, __int64))result(a1, a1, 3);
  return result;
}
