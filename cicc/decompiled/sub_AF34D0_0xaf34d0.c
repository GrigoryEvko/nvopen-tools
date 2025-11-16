// Function: sub_AF34D0
// Address: 0xaf34d0
//
unsigned __int8 *__fastcall sub_AF34D0(unsigned __int8 *a1)
{
  unsigned __int8 *result; // rax
  unsigned __int8 v2; // dl
  __int64 v3; // rax

  for ( result = a1; (unsigned int)*result - 19 <= 1; result = *(unsigned __int8 **)(v3 + 8) )
  {
    v2 = *(result - 16);
    if ( (v2 & 2) != 0 )
      v3 = *((_QWORD *)result - 4);
    else
      v3 = (__int64)&result[-8 * ((v2 >> 2) & 0xF) - 16];
  }
  return result;
}
