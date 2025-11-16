// Function: sub_87D6D0
// Address: 0x87d6d0
//
_BOOL8 __fastcall sub_87D6D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  _BOOL8 result; // rax
  __int64 **v4; // rdx

  v2 = sub_8D5CE0(a2, a1);
  result = 0;
  if ( v2 )
  {
    if ( (*(_BYTE *)(v2 + 96) & 2) != 0 )
      v4 = sub_72B780(v2);
    else
      v4 = *(__int64 ***)(v2 + 112);
    return (unsigned __int8)sub_87D630(1u, (__int64)v4[1], (__int64)v4) != 3;
  }
  return result;
}
