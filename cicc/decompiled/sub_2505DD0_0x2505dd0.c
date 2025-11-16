// Function: sub_2505DD0
// Address: 0x2505dd0
//
__int64 __fastcall sub_2505DD0(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 12);
  *(_BYTE *)(a1 + 12) = 1;
  return result;
}
