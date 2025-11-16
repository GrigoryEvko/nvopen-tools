// Function: sub_89ED80
// Address: 0x89ed80
//
__int64 *__fastcall sub_89ED80(__int64 ***a1, __int64 ***a2)
{
  __int64 **v2; // rax
  int v3; // edx
  __int64 *result; // rax

  v2 = *a2;
  do
  {
    v2 = (__int64 **)*v2;
    *a2 = v2;
  }
  while ( v2 && ((_BYTE)v2[3] & 8) != 0 );
  v3 = *((_DWORD *)*a1 + 15);
  result = **a1;
  *a1 = (__int64 **)result;
  if ( v3 )
  {
    while ( result && *((_DWORD *)result + 15) == v3 )
    {
      result = (__int64 *)*result;
      *a1 = (__int64 **)result;
    }
  }
  return result;
}
