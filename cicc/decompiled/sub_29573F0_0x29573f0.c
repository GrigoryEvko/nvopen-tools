// Function: sub_29573F0
// Address: 0x29573f0
//
__int64 __fastcall sub_29573F0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  if ( result == *(_QWORD *)(a1 + 40) )
    return 0;
  return result;
}
