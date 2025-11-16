// Function: sub_1548A30
// Address: 0x1548a30
//
__int64 __fastcall sub_1548A30(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 24) == v1 )
    return 0;
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(v1 + 8);
  return 1;
}
