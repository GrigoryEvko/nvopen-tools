// Function: sub_2F393E0
// Address: 0x2f393e0
//
__int64 __fastcall sub_2F393E0(__int64 a1)
{
  __int64 result; // rax

  sub_2F90C80();
  result = *(_QWORD *)(a1 + 3584);
  if ( result != *(_QWORD *)(a1 + 3592) )
    *(_QWORD *)(a1 + 3592) = result;
  return result;
}
