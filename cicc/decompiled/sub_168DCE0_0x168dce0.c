// Function: sub_168DCE0
// Address: 0x168dce0
//
__int64 __fastcall sub_168DCE0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 32);
  if ( *(_QWORD *)(a1 + 40) == v1 )
    return 0;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(v1 + 8);
  return 1;
}
