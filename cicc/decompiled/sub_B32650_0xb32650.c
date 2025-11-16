// Function: sub_B32650
// Address: 0xb32650
//
const char *__fastcall sub_B32650(_BYTE *a1, __int64 a2)
{
  const char *result; // rax

  if ( *a1 != 1 )
  {
    result = 0;
    if ( (a1[35] & 4) == 0 )
      return result;
    return (const char *)sub_B31D10((__int64)a1, a2, 0);
  }
  a1 = (_BYTE *)sub_B325F0((__int64)a1);
  result = byte_3F871B3;
  if ( a1 )
  {
    result = 0;
    if ( (a1[35] & 4) != 0 )
      return (const char *)sub_B31D10((__int64)a1, a2, 0);
  }
  return result;
}
