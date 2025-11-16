// Function: sub_1101580
// Address: 0x1101580
//
unsigned __int8 *__fastcall sub_1101580(__m128i *a1, unsigned __int8 *a2)
{
  unsigned __int8 *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 *v6; // r9

  result = sub_11013B0(a1, (__int64)a2);
  if ( !result )
  {
    result = sub_1100510((__int64)a2, a1);
    if ( !result )
      return sub_11005E0(a1, a2, v3, v4, v5, v6);
  }
  return result;
}
