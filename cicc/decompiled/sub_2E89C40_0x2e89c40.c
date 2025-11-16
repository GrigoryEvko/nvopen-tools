// Function: sub_2E89C40
// Address: 0x2e89c40
//
__int64 __fastcall sub_2E89C40(__int64 a1)
{
  __int64 result; // rax

  for ( result = 0; (*(_BYTE *)(a1 + 44) & 8) != 0; result = (unsigned int)(result + 1) )
    a1 = *(_QWORD *)(a1 + 8);
  return result;
}
