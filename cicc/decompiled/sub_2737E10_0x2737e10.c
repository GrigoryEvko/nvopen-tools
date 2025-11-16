// Function: sub_2737E10
// Address: 0x2737e10
//
__int64 __fastcall sub_2737E10(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( v1 == *(_QWORD *)(a1 + 24) )
    return 0;
  *(_QWORD *)(a1 + 8) = v1 + 24;
  return 1;
}
