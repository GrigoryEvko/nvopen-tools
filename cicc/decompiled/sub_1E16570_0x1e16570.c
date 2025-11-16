// Function: sub_1E16570
// Address: 0x1e16570
//
__int64 __fastcall sub_1E16570(__int64 a1)
{
  __int64 result; // rax

  for ( result = 0; (*(_BYTE *)(a1 + 46) & 8) != 0; result = (unsigned int)(result + 1) )
    a1 = *(_QWORD *)(a1 + 8);
  return result;
}
