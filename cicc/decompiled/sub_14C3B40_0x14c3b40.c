// Function: sub_14C3B40
// Address: 0x14c3b40
//
__int64 __fastcall sub_14C3B40(__int64 a1, __int64 *a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d

  v2 = sub_14AB140(a1 | 4, a2);
  if ( !v2 )
    return 0;
  v3 = v2;
  if ( !(unsigned __int8)sub_14C3AC0(v2) && v3 - 116 > 1 && v3 != 4 && v3 != 191 )
    return 0;
  return v3;
}
