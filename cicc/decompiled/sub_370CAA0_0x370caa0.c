// Function: sub_370CAA0
// Address: 0x370caa0
//
const char *__fastcall sub_370CAA0(_QWORD *a1, __int16 a2, _WORD *a3, __int64 a4)
{
  const char *result; // rax
  _WORD *v5; // rcx

  if ( !a1[7] )
    return byte_3F871B3;
  if ( a1[5] )
    return byte_3F871B3;
  result = (const char *)a1[6];
  if ( result )
    return byte_3F871B3;
  v5 = &a3[20 * a4];
  if ( v5 != a3 )
  {
    while ( a3[16] != a2 )
    {
      a3 += 20;
      if ( v5 == a3 )
        return result;
    }
    return *(const char **)a3;
  }
  return result;
}
