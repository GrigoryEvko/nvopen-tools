// Function: sub_15160A0
// Address: 0x15160a0
//
__int64 __fastcall sub_15160A0(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax

  result = *a2;
  *a1 = *a2;
  *a2 = 0;
  return result;
}
