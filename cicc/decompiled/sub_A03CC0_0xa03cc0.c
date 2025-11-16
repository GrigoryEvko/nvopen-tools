// Function: sub_A03CC0
// Address: 0xa03cc0
//
__int64 __fastcall sub_A03CC0(__int64 *a1, char a2)
{
  __int64 result; // rax

  result = *a1;
  *(_BYTE *)(*a1 + 1096) = a2;
  return result;
}
