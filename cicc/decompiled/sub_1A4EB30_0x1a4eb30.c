// Function: sub_1A4EB30
// Address: 0x1a4eb30
//
__int64 __fastcall sub_1A4EB30(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 32);
  if ( result == *(_QWORD *)(a1 + 40) )
    return 0;
  return result;
}
