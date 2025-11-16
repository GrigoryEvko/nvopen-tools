// Function: sub_2254240
// Address: 0x2254240
//
__int64 __fastcall sub_2254240(__int64 *a1)
{
  __int64 result; // rax

  result = __newlocale();
  *a1 = result;
  if ( !result )
    sub_42638E((__int64)"locale::facet::_S_create_c_locale name not valid");
  return result;
}
