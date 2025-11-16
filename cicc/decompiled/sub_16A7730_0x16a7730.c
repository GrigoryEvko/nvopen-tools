// Function: sub_16A7730
// Address: 0x16a7730
//
__int64 __fastcall sub_16A7730(unsigned __int64 *a1, unsigned __int64 a2, int a3)
{
  __int64 v3; // rcx
  unsigned __int64 v4; // rax

  if ( !a3 )
    return 1;
  v3 = (__int64)&a1[(unsigned int)(a3 - 1) + 1];
  while ( 1 )
  {
    v4 = *a1;
    *a1 -= a2;
    if ( v4 >= a2 )
      break;
    ++a1;
    a2 = 1;
    if ( a1 == (unsigned __int64 *)v3 )
      return 1;
  }
  return 0;
}
