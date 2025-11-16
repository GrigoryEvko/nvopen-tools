// Function: sub_AA4890
// Address: 0xaa4890
//
__int64 __fastcall sub_AA4890(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 72);
  if ( result )
    return *(_QWORD *)(result + 112);
  return result;
}
