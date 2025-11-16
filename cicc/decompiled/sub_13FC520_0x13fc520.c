// Function: sub_13FC520
// Address: 0x13fc520
//
__int64 __fastcall sub_13FC520(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rdi

  v1 = sub_13FC470(a1);
  if ( !v1 )
    return 0;
  v2 = v1;
  if ( !(unsigned __int8)sub_157F630(v1) )
    return 0;
  v3 = sub_157EBA0(v2);
  if ( !v3 || (unsigned int)sub_15F4D60(v3) != 1 )
    return 0;
  return v2;
}
