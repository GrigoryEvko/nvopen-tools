// Function: sub_10194C0
// Address: 0x10194c0
//
unsigned __int8 *__fastcall sub_10194C0(
        __int64 a1,
        unsigned __int8 *a2,
        unsigned __int8 **a3,
        unsigned __int64 a4,
        __m128i *a5)
{
  int v8; // edx
  unsigned __int8 *result; // rax

  if ( sub_B49200(a1) )
    return 0;
  v8 = *a2;
  if ( (unsigned int)(v8 - 12) <= 1 || (_BYTE)v8 == 20 )
    return (unsigned __int8 *)sub_ACADE0(*(__int64 ***)(a1 + 8));
  result = (unsigned __int8 *)sub_FFF020(a1, a2, a3, a4, (__int64)a5);
  if ( !result )
  {
    if ( !*a2 && (a2[33] & 0x20) != 0 )
      return sub_1018220(a1, (__int64)a2, (__int64 *)a3, a4, a5);
    return 0;
  }
  return result;
}
