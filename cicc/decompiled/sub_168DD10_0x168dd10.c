// Function: sub_168DD10
// Address: 0x168dd10
//
__int64 __fastcall sub_168DD10(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 24) == v1 )
    return 0;
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(v1 + 8);
  return 1;
}
