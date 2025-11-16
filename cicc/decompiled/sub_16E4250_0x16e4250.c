// Function: sub_16E4250
// Address: 0x16e4250
//
__int64 __fastcall sub_16E4250(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 264);
  if ( result )
    return *(_QWORD *)(result + 8);
  return result;
}
