// Function: sub_E8D6F0
// Address: 0xe8d6f0
//
__int64 __fastcall sub_E8D6F0(__int64 a1, unsigned __int8 a2, unsigned __int8 a3)
{
  __int64 result; // rax

  result = nullsub_349(a1, a2, a3);
  *(_BYTE *)(a1 + 304) = a2;
  *(_BYTE *)(a1 + 305) = a3;
  return result;
}
