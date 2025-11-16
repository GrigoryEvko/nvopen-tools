// Function: sub_7E04B0
// Address: 0x7e04b0
//
__int64 __fastcall sub_7E04B0(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 ***v2; // rax
  __int64 **v3; // rax

  v1 = sub_8D2310(a1);
  if ( !v1 )
    return v1;
  v2 = *(__int64 ****)(a1 + 168);
  v1 = 0;
  if ( !v2 )
    return v1;
  v3 = *v2;
  if ( !v3 )
    return v1;
  do
  {
    if ( ((_BYTE)v3[4] & 1) != 0 )
      return 1;
    v3 = (__int64 **)*v3;
  }
  while ( v3 );
  return 0;
}
