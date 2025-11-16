// Function: sub_1516150
// Address: 0x1516150
//
__int64 __fastcall sub_1516150(__int64 *a1, char a2)
{
  __int64 result; // rax

  result = *a1;
  *(_BYTE *)(*a1 + 1008) = a2;
  return result;
}
