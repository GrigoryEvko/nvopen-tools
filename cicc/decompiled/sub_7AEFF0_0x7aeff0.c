// Function: sub_7AEFF0
// Address: 0x7aeff0
//
__int64 *__fastcall sub_7AEFF0(unsigned __int64 a1)
{
  __int64 *result; // rax
  _QWORD *v2; // rcx
  __int64 *v3; // r8

  result = (__int64 *)qword_4F06440;
  if ( !qword_4F06440 )
    return 0;
  v2 = 0;
  while ( result[7] > a1 || a1 >= result[8] + 1 )
  {
    v3 = (__int64 *)*result;
    v2 = result;
    if ( !*result )
      return v3;
    result = (__int64 *)*result;
  }
  v3 = result;
  if ( !v2 || (result[6] & 0x10) != 0 )
    return v3;
  *v2 = *result;
  *result = qword_4F06440;
  qword_4F06440 = result;
  return result;
}
