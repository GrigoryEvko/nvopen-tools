// Function: sub_2957360
// Address: 0x2957360
//
__int64 __fastcall sub_2957360(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( v1 == *(_QWORD *)(a1 + 40) )
    return 0;
  *(_QWORD *)(a1 + 16) = v1 + 8;
  return 1;
}
