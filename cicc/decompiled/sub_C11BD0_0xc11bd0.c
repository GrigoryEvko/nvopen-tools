// Function: sub_C11BD0
// Address: 0xc11bd0
//
__int64 __fastcall sub_C11BD0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 40) == v1 )
    return 0;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(v1 + 8);
  return 1;
}
