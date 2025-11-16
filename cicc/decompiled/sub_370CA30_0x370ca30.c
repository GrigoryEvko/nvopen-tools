// Function: sub_370CA30
// Address: 0x370ca30
//
const char *__fastcall sub_370CA30(_QWORD *a1, int a2, unsigned __int8 *a3, __int64 a4)
{
  const char *result; // rax
  unsigned __int8 *v5; // rdi

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
