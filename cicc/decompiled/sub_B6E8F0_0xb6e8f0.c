// Function: sub_B6E8F0
// Address: 0xb6e8f0
//
__int64 __fastcall sub_B6E8F0(__int64 *a1, char a2)
{
  __int64 result; // rax

  result = *a1;
  *(_BYTE *)(*a1 + 113) = a2;
  return result;
}
