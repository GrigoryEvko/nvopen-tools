// Function: sub_C11B70
// Address: 0xc11b70
//
__int64 __fastcall sub_C11B70(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 24);
  if ( *(_QWORD *)(a1 + 56) == v1 )
    return 0;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(v1 + 8);
  return 1;
}
