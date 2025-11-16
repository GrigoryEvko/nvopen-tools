// Function: sub_22409D0
// Address: 0x22409d0
//
__int64 __fastcall sub_22409D0(__int64 a1, unsigned __int64 *a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdx

  v3 = *a2;
  if ( *a2 > 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::_M_create");
  if ( v3 > a3 )
  {
    v4 = 2 * a3;
    if ( v3 < v4 )
    {
      if ( v4 > 0x3FFFFFFFFFFFFFFFLL )
      {
        *a2 = 0x3FFFFFFFFFFFFFFFLL;
        return sub_22077B0(0x4000000000000000uLL);
      }
      *a2 = v4;
      v3 = v4;
    }
  }
  return sub_22077B0(v3 + 1);
}
