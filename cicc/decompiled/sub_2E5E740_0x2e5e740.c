// Function: sub_2E5E740
// Address: 0x2e5e740
//
__int64 *__fastcall sub_2E5E740(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *result; // rax
  unsigned int v4; // edx
  unsigned int i; // ecx

  result = a3;
  if ( !a2 || !a3 )
    return 0;
  v4 = *((_DWORD *)a2 + 42);
  for ( i = *((_DWORD *)result + 42); v4 > i; v4 = *((_DWORD *)a2 + 42) )
    a2 = (__int64 *)*a2;
  if ( v4 < i )
  {
    do
      result = (__int64 *)*result;
    while ( *((_DWORD *)result + 42) > v4 );
  }
  for ( ; a2 != result; result = (__int64 *)*result )
    a2 = (__int64 *)*a2;
  return result;
}
