// Function: sub_16033C0
// Address: 0x16033c0
//
__int64 __fastcall sub_16033C0(__int64 *a1, char a2)
{
  __int64 result; // rax

  result = *a1;
  *(_BYTE *)(*a1 + 2960) = a2;
  return result;
}
