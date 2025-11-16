// Function: sub_38D62A0
// Address: 0x38d62a0
//
__int64 __fastcall sub_38D62A0(__int64 a1, unsigned __int8 a2, unsigned __int8 a3)
{
  __int64 result; // rax

  result = nullsub_1944(a1, a2, a3);
  *(_BYTE *)(a1 + 280) = a2;
  *(_BYTE *)(a1 + 281) = a3;
  return result;
}
