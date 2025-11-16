// Function: sub_25AC560
// Address: 0x25ac560
//
__int64 __fastcall sub_25AC560(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 24) == v1 )
    return 0;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(v1 + 8);
  return 1;
}
