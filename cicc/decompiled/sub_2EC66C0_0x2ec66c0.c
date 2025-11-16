// Function: sub_2EC66C0
// Address: 0x2ec66c0
//
__int64 __fastcall sub_2EC66C0(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax

  if ( a3 )
    result = sub_2EC61D0(a1, a2);
  else
    result = sub_2EC62A0(a1, a2);
  *(_BYTE *)(a2 + 249) |= 4u;
  return result;
}
