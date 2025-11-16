// Function: sub_39F88E0
// Address: 0x39f88e0
//
__int64 (__fastcall *__fastcall sub_39F88E0(__int64 a1))(__int64, __int64)
{
  __int64 (__fastcall *result)(__int64, __int64); // rax

  result = *(__int64 (__fastcall **)(__int64, __int64))(a1 + 8);
  if ( result )
    return (__int64 (__fastcall *)(__int64, __int64))result(1, a1);
  return result;
}
