// Function: sub_29574B0
// Address: 0x29574b0
//
__int64 __fastcall sub_29574B0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  if ( result == *(_QWORD *)(a1 + 24) )
    return 0;
  return result;
}
