// Function: sub_370C9C0
// Address: 0x370c9c0
//
const char *__fastcall sub_370C9C0(_QWORD *a1, char a2, _BYTE *a3, __int64 a4)
{
  const char *result; // rax
  _BYTE *v5; // rcx

  if ( !a1[7] )
    return byte_3F871B3;
  if ( a1[5] )
    return byte_3F871B3;
  result = (const char *)a1[6];
  if ( result )
    return byte_3F871B3;
  v5 = &a3[40 * a4];
  if ( v5 != a3 )
  {
    while ( a3[32] != a2 )
    {
      a3 += 40;
      if ( v5 == a3 )
        return result;
    }
    return *(const char **)a3;
  }
  return result;
}
