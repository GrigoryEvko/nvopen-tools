// Function: sub_70D540
// Address: 0x70d540
//
_BOOL8 __fastcall sub_70D540(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax
  __int64 v3; // r12
  __int64 v4; // rcx
  __int64 v5; // rsi
  __int64 v6; // r8

  if ( !(unsigned int)sub_8D2E30(a1) || !(unsigned int)sub_8D2E30(a2) )
    return 0;
  v3 = sub_8D46C0(a1);
  v5 = sub_8D46C0(a2);
  result = 1;
  if ( v3 != v5 )
    return (unsigned int)sub_8D97D0(v3, v5, 32, v4, v6) != 0;
  return result;
}
