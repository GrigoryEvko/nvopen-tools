// Function: sub_B6F950
// Address: 0xb6f950
//
__int64 __fastcall sub_B6F950(__int64 *a1, char a2)
{
  __int64 result; // rax

  result = *a1;
  *(_BYTE *)(*a1 + 3496) = a2;
  return result;
}
