// Function: sub_16DB5E0
// Address: 0x16db5e0
//
__int64 **sub_16DB5E0()
{
  __int64 **result; // rax
  __int64 *v1; // rdi

  result = (__int64 **)sub_16D40F0((__int64)&qword_4FA1650);
  v1 = (__int64 *)qword_4FA1660;
  if ( result )
    v1 = *result;
  if ( v1 )
    return (__int64 **)sub_16DA8A0(v1);
  return result;
}
