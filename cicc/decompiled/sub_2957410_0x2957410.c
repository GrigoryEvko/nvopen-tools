// Function: sub_2957410
// Address: 0x2957410
//
__int64 __fastcall sub_2957410(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  if ( result == *(_QWORD *)(a1 + 32) )
    return 0;
  return result;
}
