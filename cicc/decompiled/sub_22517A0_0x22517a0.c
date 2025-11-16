// Function: sub_22517A0
// Address: 0x22517a0
//
__int64 __fastcall sub_22517A0(__int64 a1, unsigned __int64 *a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx

  v3 = *a2;
  if ( *a2 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::_M_create");
  if ( v3 > a3 )
  {
    v4 = 2 * a3;
    if ( v3 < v4 )
    {
      if ( v4 > 0xFFFFFFFFFFFFFFFLL )
      {
        *a2 = 0xFFFFFFFFFFFFFFFLL;
        return sub_22077B0(0x4000000000000000uLL);
      }
      *a2 = v4;
      v3 = v4;
    }
  }
  return sub_22077B0(4 * v3 + 4);
}
