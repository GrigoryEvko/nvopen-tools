// Function: sub_38BE3C0
// Address: 0x38be3c0
//
__int64 __fastcall sub_38BE3C0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 40);
  if ( result )
    *(_BYTE *)(result + 16) = 0;
  return result;
}
