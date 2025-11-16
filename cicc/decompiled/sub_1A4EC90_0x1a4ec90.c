// Function: sub_1A4EC90
// Address: 0x1a4ec90
//
__int64 __fastcall sub_1A4EC90(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  if ( result == *(_QWORD *)(a1 + 24) )
    return 0;
  return result;
}
