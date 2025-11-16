// Function: sub_10FF0A0
// Address: 0x10ff0a0
//
char __fastcall sub_10FF0A0(unsigned __int8 *a1, __int64 a2, const __m128i *a3, __int64 a4)
{
  char result; // al
  __int64 v7; // r8
  __int64 v8; // r9

  result = sub_10FD310(a1, a2);
  if ( !result )
    return sub_10FE2C0(a1, a2, a3, a4, v7, v8);
  return result;
}
