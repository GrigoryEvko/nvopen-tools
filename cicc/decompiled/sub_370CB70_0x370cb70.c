// Function: sub_370CB70
// Address: 0x370cb70
//
const char *__fastcall sub_370CB70(_QWORD *a1, int a2)
{
  const char *result; // rax
  __int64 *v3; // rdx

  if ( !a1[7] )
    return byte_3F871B3;
  if ( a1[5] )
    return byte_3F871B3;
  result = (const char *)a1[6];
  if ( result )
    return byte_3F871B3;
  v3 = &qword_504EEE0;
  while ( *((unsigned __int16 *)v3 + 16) != a2 )
  {
    v3 += 5;
    if ( v3 == (__int64 *)&dword_5050998 )
      return result;
  }
  return (const char *)*v3;
}
