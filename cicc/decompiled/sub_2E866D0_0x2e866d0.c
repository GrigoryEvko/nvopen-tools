// Function: sub_2E866D0
// Address: 0x2e866d0
//
__int64 __fastcall sub_2E866D0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 24);
  if ( result )
    return *(_QWORD *)(*(_QWORD *)(result + 32) + 32LL);
  return result;
}
