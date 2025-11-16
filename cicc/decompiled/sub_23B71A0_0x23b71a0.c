// Function: sub_23B71A0
// Address: 0x23b71a0
//
__int64 __fastcall sub_23B71A0(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // r12

  result = a1[3];
  if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v2 = (result >> 1) & 1;
    if ( (result & 4) != 0 )
      result = (*(__int64 (**)(void))((a1[3] & 0xFFFFFFFFFFFFFFF8LL) + 16))();
    if ( !(_BYTE)v2 )
      return sub_C7D6A0(*a1, a1[1], a1[2]);
  }
  return result;
}
